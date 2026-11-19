import os

import numpy as np
import pandas as pd
import torch


def dataset_files(root, name):
    adj = os.path.join(root, "{}_rn_adj.npy".format(name))
    meta = os.path.join(root, "{}_meta.csv".format(name))
    adj = adj if os.path.exists(adj) else os.path.join(root, "adj.npy")
    return adj, meta if os.path.exists(meta) else None


def num_train_nodes(num_nodes, max_increase_ratio):
    return int(num_nodes / (1.0 + float(max_increase_ratio)))


def test_nodes(order, num_train, increase_ratio, decrease_ratio, spatial_shift):
    if not spatial_shift:
        return order[:num_train], 0
    num_increase = int(num_train * float(increase_ratio))
    num_decrease = int(num_train * float(decrease_ratio))
    nodes = np.concatenate([order[: num_train - num_decrease], order[num_train: num_train + num_increase]])
    return nodes, num_increase


def metadata_prior(meta_path, num_nodes, train_nodes):
    meta = pd.read_csv(meta_path).iloc[:num_nodes].copy()
    fit_meta = meta.iloc[train_nodes].copy()
    blocks = []

    numeric = meta.select_dtypes(include=[np.number]).copy()
    fit_numeric = fit_meta.select_dtypes(include=[np.number]).copy()
    drop = [c for c in numeric.columns if c.lower().startswith("id") or numeric[c].nunique(dropna=False) <= 1]
    numeric = numeric.drop(columns=drop, errors="ignore")
    fit_numeric = fit_numeric.drop(columns=drop, errors="ignore")
    if not numeric.empty:
        fit_numeric = fit_numeric.fillna(fit_numeric.median())
        numeric = numeric.fillna(fit_numeric.median())
        values = numeric.to_numpy(dtype=np.float32)
        fit_values = fit_numeric.to_numpy(dtype=np.float32)
        mean = fit_values.mean(axis=0, keepdims=True)
        std = fit_values.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        blocks.append((values - mean) / std)

    columns = [
        c for c in meta.columns
        if c not in numeric.columns and not c.lower().startswith("id") and not pd.api.types.is_numeric_dtype(meta[c])
    ]
    if columns:
        fit_categorical = fit_meta[columns].fillna("UNK").copy()
        categorical = meta[columns].fillna("UNK").copy()
        for c in columns:
            seen = set(fit_categorical[c].astype(str).tolist())
            categorical[c] = categorical[c].astype(str).where(categorical[c].astype(str).isin(seen), "UNK")
        fit_columns = pd.get_dummies(fit_categorical, dummy_na=False, dtype=np.float32).columns
        categorical = pd.get_dummies(categorical, dummy_na=False, dtype=np.float32)
        blocks.append(categorical.reindex(columns=fit_columns, fill_value=0.0).to_numpy(dtype=np.float32))
    return torch.tensor(np.concatenate(blocks, axis=1).astype(np.float32), dtype=torch.float32)


def static_prior(adj, meta_path, train_nodes):
    blocks = []
    if meta_path is not None:
        blocks.append(metadata_prior(meta_path, adj.shape[0], train_nodes))
    degree = np.log1p(np.asarray(adj, dtype=np.float32).sum(axis=1, keepdims=True))
    degree_max = float(degree[train_nodes].max())
    if degree_max > 0:
        degree = degree / degree_max
    blocks.append(torch.tensor(degree, dtype=torch.float32))
    return torch.cat(blocks, dim=-1)
