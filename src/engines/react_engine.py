import json
import os
import time

import numpy as np
import torch

from src.utils.dataloader import DataLoader, load_index, load_year
from src.utils.metrics import masked_mae, masked_mape, masked_rmse, null_value
from src.utils.react_data import num_train_nodes, test_nodes


def to_tensor(array, device):
    return torch.tensor(array, dtype=torch.float32, device=device)


def predict(model, loader, scaler, nodes, device):
    model.eval()
    preds, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            pred, _ = model(to_tensor(x, device), nodes)
            preds.append(scaler.inverse_transform(pred).squeeze(-1).cpu())
            labels.append(scaler.inverse_transform(to_tensor(y, device)).squeeze(-1).cpu())
    return torch.cat(preds, dim=0), torch.cat(labels, dim=0)


def horizon_metrics(preds, labels):
    mv = null_value(labels)
    rows = []
    for h in range(preds.shape[1]):
        p, y = preds[:, h, :], labels[:, h, :]
        rows.append([masked_mae(p, y, mv).item(), masked_rmse(p, y, mv).item(), masked_mape(p, y, mv).item()])
    return np.array(rows)


def evaluate(preds, labels, num_unseen):
    results = {"All": horizon_metrics(preds, labels)}
    if num_unseen:
        results["UnF"] = horizon_metrics(preds[..., -num_unseen:], labels[..., -num_unseen:])
    return results


def train(model, cfg, loaders, scaler, nodes, ckpt_path, device, logger):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma)
    best, wait = np.inf, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        model.train()
        loaders["train"].shuffle()
        losses = []
        for x, y in loaders["train"]:
            optimizer.zero_grad()
            x, y = to_tensor(x, device), to_tensor(y, device)
            prior_mask = torch.rand(len(nodes)) >= cfg.prior_mask if cfg.prior_mask > 0 else None
            pred, alpha = model(x, nodes, prior_mask)
            pred, y = scaler.inverse_transform(pred), scaler.inverse_transform(y)
            loss = masked_mae(pred, y, null_value(y))
            if cfg.lambda_ctx > 0:
                loss = loss + cfg.lambda_ctx * alpha.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            optimizer.step()
            losses.append(loss.item())
        train_time = time.time() - start
        preds, labels = predict(model, loaders["val"], scaler, nodes, device)
        val_loss = masked_mae(preds, labels, null_value(labels)).item()
        lr = scheduler.get_last_lr()[0]
        scheduler.step()
        memory = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else 0.0
        logger.info("Epoch %03d | train loss %.4f | val MAE %.4f | lr %.2e | %.1fs | %.0f MiB",
                    epoch, float(np.mean(losses)), val_loss, lr, train_time, memory)
        if val_loss < best:
            best, wait = val_loss, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= cfg.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break
    logger.info("Best val MAE %.4f, checkpoint saved to %s", best, ckpt_path)


def scenarios(cfg):
    groups = {
        "main": [("joint", year, cfg.test_increase_ratio, True) for year in cfg.test_years],
        "sood": [("sood", cfg.train_year, ratio, True) for ratio in (0.1, 0.2, 0.3)],
        "tood": [("tood", year, 0.0, False) for year in cfg.test_years],
    }
    return groups[cfg.protocol]


def fmt(values):
    return "MAE {:.4f} RMSE {:.4f} MAPE {:.2f}%".format(values[0], values[1], values[2] * 100)


def test(model, cfg, order, root, device, logger):
    num_train = num_train_nodes(len(order), cfg.max_increase_ratio)
    settings = scenarios(cfg)
    records = []
    for year in sorted({s[1] for s in settings}):
        data, scaler = load_year(root, year, cfg.input_dim)
        idx = load_index(root, year, "test")
        for kind, _, ratio, spatial in [s for s in settings if s[1] == year]:
            nodes, num_unseen = test_nodes(order, num_train, ratio, cfg.test_decrease_ratio, spatial)
            loader = DataLoader(data[:, nodes], idx, cfg.seq_len, cfg.horizon, cfg.batch_size)
            results = evaluate(*predict(model, loader, scaler, nodes, device), num_unseen)
            records.append(dict(setting=kind, year=year, ratio=ratio if spatial else None,
                                **{k: v.tolist() for k, v in results.items()}))
            full = results["All"]
            message = "[{} {}{}] All {} | H3 {} | H6 {} | H12 {}".format(
                kind, year, " r_s={}".format(ratio) if spatial else "",
                fmt(full.mean(0)), fmt(full[2]), fmt(full[5]), fmt(full[11]))
            if "UnF" in results:
                message += " | UnF {}".format(fmt(results["UnF"].mean(0)))
            logger.info(message)
        del data

    for ratio in sorted({r["ratio"] for r in records if r["setting"] == "joint"}):
        group = [r for r in records if r["setting"] == "joint" and r["ratio"] == ratio]
        logger.info("[joint r_s=%s, mean over %s] All %s | UnF %s", ratio, "/".join(str(r["year"]) for r in group),
                    fmt(np.mean([np.mean(r["All"], axis=0) for r in group], axis=0)),
                    fmt(np.mean([np.mean(r["UnF"], axis=0) for r in group], axis=0)))
    with open(os.path.join(cfg.save_dir, "results_{}.json".format(cfg.protocol)), "w") as f:
        json.dump(dict(dataset=cfg.dataset, checkpoint=cfg.ckpt, results=records), f, indent=1)
