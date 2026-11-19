import torch


def _mask(labels, null_val):
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean(mask)
    return torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)


def _reduce(loss):
    return torch.mean(torch.where(torch.isnan(loss), torch.zeros_like(loss), loss))


def masked_mae(preds, labels, null_val):
    return _reduce(torch.abs(preds - labels) * _mask(labels, null_val))


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(_reduce((preds - labels) ** 2 * _mask(labels, null_val)))


def masked_mape(preds, labels, null_val):
    denom = torch.clamp(torch.abs(labels), min=1e-3)
    return _reduce(torch.abs(preds - labels) / denom * _mask(labels, null_val))


def null_value(labels):
    if labels.min() < 1:
        return labels.min()
    return torch.tensor(0.0, device=labels.device, dtype=labels.dtype)
