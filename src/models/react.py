import torch
import torch.nn.functional as F
from torch import nn


def temporal_index(values, size):
    if values.max() <= 1.0 + 1e-6:
        values = values * size
    return values.long().clamp_(0, size - 1)


def window_statistics(series, tod):
    mean = series.mean(dim=-1)
    std = torch.sqrt(series.var(dim=-1, unbiased=False) + 1e-5)
    last = series[..., -1]
    trend = last - series[..., 0]
    volatility = (series[..., 1:] - series[..., :-1]).abs().mean(dim=-1)
    return torch.stack([mean, std, last, trend, volatility, tod], dim=-1)


class MLPBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        return x + self.fc2(self.drop(self.act(self.fc1(x))))


class ChannelLinear(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x.squeeze(-1).transpose(1, 2).contiguous())
        return x.transpose(1, 2).unsqueeze(-1).contiguous()


class LinearBlock(nn.Module):
    def __init__(self, dim, dropout, res_scale):
        super().__init__()
        self.mixer = ChannelLinear(dim)
        self.drop = nn.Dropout(p=dropout)
        self.res_scale = nn.Parameter(torch.tensor(float(res_scale)))

    def forward(self, x):
        return x + self.res_scale * self.drop(self.mixer(x))


class React(nn.Module):
    def __init__(self, cfg, static_prior, adj):
        super().__init__()
        self.input_dim = cfg.input_dim
        self.steps_per_day = cfg.steps_per_day
        self.days_per_week = cfg.days_per_week
        self.alpha_min = cfg.alpha_min
        self.alpha_max = cfg.alpha_max
        hidden_dim = cfg.embed_dim + cfg.tod_dim + cfg.dow_dim

        self.time_series_emb_layer = nn.Conv2d(
            cfg.input_dim * cfg.seq_len, cfg.embed_dim, kernel_size=(1, 1), bias=True
        )
        self.time_in_day_emb = nn.Parameter(torch.empty(cfg.steps_per_day, cfg.tod_dim))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = None
        if cfg.dow_dim > 0:
            self.day_in_week_emb = nn.Parameter(torch.empty(cfg.days_per_week, cfg.dow_dim))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        if cfg.block == "mlp":
            blocks = [MLPBlock(hidden_dim, cfg.dropout) for _ in range(cfg.num_layer)]
        else:
            blocks = [LinearBlock(hidden_dim, cfg.dropout, cfg.res_scale) for _ in range(cfg.num_layer)]
        self.encoder = nn.Sequential(*blocks)
        self.base_head = nn.Conv2d(hidden_dim, cfg.horizon, kernel_size=(1, 1), bias=True)

        self.register_buffer("static_prior", static_prior.float())
        adj = adj.float()
        self.register_buffer("adj_normed", adj / adj.sum(dim=-1, keepdim=True).clamp(min=1e-8))

        self.context_mlp = nn.Sequential(
            nn.Linear(7 + static_prior.shape[-1], cfg.context_hidden),
            nn.GELU(),
            nn.Linear(cfg.context_hidden, cfg.context_dim),
        )
        self.context_norm = nn.LayerNorm(cfg.context_dim)
        self.router = nn.Linear(cfg.context_dim, 4)
        self.context_to_hidden = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.context_hidden),
            nn.GELU(),
            nn.Linear(cfg.context_hidden, 2 * hidden_dim),
        )
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim + cfg.context_dim, cfg.context_hidden),
            nn.GELU(),
            nn.Linear(cfg.context_hidden, hidden_dim),
            nn.Sigmoid(),
        )

    def encode(self, x, tod, dow):
        batch_size, _, num_nodes, _ = x.shape
        series = x.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        blocks = [self.time_series_emb_layer(series.transpose(1, 2).unsqueeze(-1))]
        tod_idx = temporal_index(tod, self.steps_per_day)
        blocks.append(self.time_in_day_emb[tod_idx].transpose(1, 2).unsqueeze(-1))
        if self.day_in_week_emb is not None:
            dow_idx = temporal_index(dow, self.days_per_week)
            blocks.append(self.day_in_week_emb[dow_idx].transpose(1, 2).unsqueeze(-1))
        return self.encoder(torch.cat(blocks, dim=1))

    def context_state(self, series, tod, index, prior_mask):
        stats = window_statistics(series, tod)
        batch_size, num_nodes, _ = stats.shape
        keep = stats.new_ones(batch_size, num_nodes, 1)
        prior = self.static_prior.index_select(0, index).to(stats.device)
        prior = prior.unsqueeze(0).expand(batch_size, -1, -1)
        if prior_mask is not None:
            keep = prior_mask.to(stats.device).float().view(1, num_nodes, 1).expand(batch_size, -1, -1)
            prior = prior * keep
        return torch.cat([stats, prior, keep], dim=-1)

    def uncertainty(self, context):
        probs = F.softmax(self.router(context), dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)
        return entropy / torch.log(probs.new_tensor(float(probs.shape[-1]))).clamp_min(1e-8)

    def calibrate(self, context, index):
        entropy = self.uncertainty(context) if self.alpha_max > self.alpha_min else None
        neighbor = torch.einsum("mn,bnd->bmd", self.adj_normed[index][:, index], context)
        if entropy is None:
            alpha = context.new_full((context.shape[0], context.shape[1], 1), self.alpha_min)
        else:
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * entropy
        alpha = alpha.clamp_(0.0, 1.0)
        return self.context_norm((1.0 - alpha) * context + alpha * neighbor), alpha

    def modulate(self, hidden, context):
        scale, bias = torch.chunk(self.context_to_hidden(context), chunks=2, dim=-1)
        normed = self.hidden_norm(hidden)
        delta = normed * torch.tanh(scale) + bias
        gate = self.context_gate(torch.cat([normed, context], dim=-1))
        return self.output_norm(hidden + gate * delta)

    def forward(self, x, nodes, prior_mask=None):
        index = torch.as_tensor(nodes, device=x.device, dtype=torch.long)
        tod = x[:, -1, :, 1]
        dow = x[:, -1, :, 2]
        series = x[:, :, :, 0].transpose(2, 1).contiguous()
        hidden = self.encode(x[..., : self.input_dim], tod, dow).squeeze(-1).transpose(1, 2).contiguous()
        context = self.context_norm(self.context_mlp(self.context_state(series, tod, index, prior_mask)))
        context, alpha = self.calibrate(context, index)
        hidden = self.modulate(hidden, context)
        return self.base_head(hidden.transpose(1, 2).unsqueeze(-1).contiguous()), alpha
