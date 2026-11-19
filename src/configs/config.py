import argparse
import os

import torch


def get_config():
    parser = argparse.ArgumentParser(description="REACT")
    parser.add_argument("--dataset", type=str, default="sd")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--train_year", type=int, default=2017)
    parser.add_argument("--test_years", type=int, nargs="+", default=[2018, 2019, 2020, 2021])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--protocol", type=str, default="main", choices=["main", "sood", "tood"])
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--input_dim", type=int, default=3)
    parser.add_argument("--steps_per_day", type=int, default=96)
    parser.add_argument("--days_per_week", type=int, default=7)
    parser.add_argument("--max_increase_ratio", type=float, default=0.3)
    parser.add_argument("--test_increase_ratio", type=float, default=0.3)
    parser.add_argument("--test_decrease_ratio", type=float, default=0.1)

    parser.add_argument("--embed_dim", type=int, default=48)
    parser.add_argument("--tod_dim", type=int, default=48)
    parser.add_argument("--dow_dim", type=int, default=48)
    parser.add_argument("--block", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--res_scale", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--context_dim", type=int, default=32)
    parser.add_argument("--context_hidden", type=int, default=64)
    parser.add_argument("--alpha_min", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.2)
    parser.add_argument("--prior_mask", type=float, default=0.1)
    parser.add_argument("--lambda_ctx", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 40, 45])
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5.0)

    args = parser.parse_args()
    args.save_dir = args.save_dir or os.path.join("experiments", args.dataset)
    args.ckpt = args.ckpt or os.path.join(args.save_dir, "model.pt")
    return args
