import os

import numpy as np
import torch

from src.configs.config import get_config
from src.engines.react_engine import test, train
from src.models.react import React
from src.utils.dataloader import DataLoader, load_index, load_year
from src.utils.logging import get_logger
from src.utils.react_data import dataset_files, num_train_nodes, static_prior


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    cfg = get_config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    logger = get_logger(os.path.join(cfg.save_dir, "{}.log".format(cfg.mode)))
    logger.info(vars(cfg))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    root = os.path.join(cfg.data_root, cfg.dataset)
    adj_path, meta_path = dataset_files(root, cfg.dataset)
    adj = np.load(adj_path)
    order = np.arange(adj.shape[0])
    np.random.shuffle(order)
    train_nodes = order[: num_train_nodes(len(order), cfg.max_increase_ratio)]
    model = React(cfg, static_prior(adj, meta_path, train_nodes), torch.tensor(adj, dtype=torch.float32)).to(device)
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    if cfg.mode == "train":
        data, scaler = load_year(root, cfg.train_year, cfg.input_dim)
        data = data[:, train_nodes]
        loaders = {split: DataLoader(data, load_index(root, cfg.train_year, split), cfg.seq_len, cfg.horizon,
                                     cfg.batch_size) for split in ("train", "val")}
        train(model, cfg, loaders, scaler, train_nodes, cfg.ckpt, device, logger)
        del data, loaders

    state = torch.load(cfg.ckpt, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    test(model, cfg, order, root, device, logger)


if __name__ == "__main__":
    main()
