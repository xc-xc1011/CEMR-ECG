import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from metrics_eval import classification_metrics, metrics_row


ROOT = Path.cwd().resolve()
RESULTS = ROOT / "results"
CACHE = RESULTS / "external_base_features_cache.npz"
DETAIL_PATH = RESULTS / "modern_deep_baselines_detail.csv"
SUMMARY_PATH = RESULTS / "modern_deep_baselines_summary.csv"
REPORT_PATH = RESULTS / "modern_deep_baselines_report.md"
CONFUSION_PATH = RESULTS / "modern_deep_baselines_confusion.json"

SEEDS = [303, 1303, 2303, 3303, 4303]
CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    use_benchmark = os.environ.get("CEMR_CUDNN_BENCHMARK", "0") == "1"
    torch.backends.cudnn.benchmark = use_benchmark
    torch.backends.cudnn.deterministic = not use_benchmark


def load_cache(input_mode):
    data = np.load(CACHE, allow_pickle=True)
    if input_mode == "single":
        x_tr = data["mit_tr_b0"][:, None, :].astype(np.float32)
        x_te = data["mit_te_b0"][:, None, :].astype(np.float32)
    elif input_mode == "dual":
        x_tr = np.stack([data["mit_tr_b0"], data["mit_tr_b1"]], axis=1).astype(np.float32)
        x_te = np.stack([data["mit_te_b0"], data["mit_te_b1"]], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")
    y_tr = data["mit_tr_y"].astype(np.int64)
    y_te = data["mit_te_y"].astype(np.int64)
    return x_tr, y_tr, x_te, y_te


def stratified_split(y, seed, val_fraction=0.15):
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []
    y = np.asarray(y)
    for cls in range(config.N_CLASSES):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction))) if len(idx) else 0
        n_val = min(n_val, max(1, len(idx) - 1)) if len(idx) > 1 else len(idx)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


def normalize_from_train(x_train, x_val, x_test):
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return (
        ((x_train - mean) / std).astype(np.float32),
        ((x_val - mean) / std).astype(np.float32),
        ((x_test - mean) / std).astype(np.float32),
    )


def smote_tomek_waveforms(x, y, seed):
    """Apply SMOTE-Tomek to flattened waveform windows when available.

    CAT-Net reports SMOTE-Tomek for imbalance handling. For the shared
    heartbeat-window interface we apply it to flattened waveform vectors and
    reshape the resampled vectors back to [N, C, T].
    """
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTE
    except Exception as exc:
        print(f"[SMOTE-Tomek] unavailable, falling back to sampler: {exc}", flush=True)
        return x, y, False
    shape = x.shape[1:]
    flat = x.reshape(len(x), -1)
    counts = np.bincount(y, minlength=config.N_CLASSES)
    nonzero = counts[counts > 0]
    if len(nonzero) < 2 or nonzero.min() < 2:
        print("[SMOTE-Tomek] skipped because at least one class has fewer than two samples", flush=True)
        return x, y, False
    k_neighbors = int(max(1, min(5, nonzero.min() - 1)))
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    sampler = SMOTETomek(random_state=seed, smote=smote)
    try:
        x_res, y_res = sampler.fit_resample(flat, y)
    except Exception as exc:
        print(f"[SMOTE-Tomek] failed, falling back to sampler: {exc}", flush=True)
        return x, y, False
    return x_res.reshape((-1, *shape)).astype(np.float32), y_res.astype(np.int64), True


class BeatDataset(Dataset):
    def __init__(self, x, y, augment=None):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.x[idx].clone()
        y = self.y[idx]
        if self.augment == "jitter_mask":
            x = x + 0.01 * torch.randn_like(x)
            if random.random() < 0.25:
                width = random.randint(8, 24)
                start = random.randint(0, max(0, x.shape[-1] - width))
                x[:, start:start + width] = 0.0
        return x, y


def make_loader(x, y, batch_size, train=False, sampler_mode=None, augment=None):
    dataset = BeatDataset(x, y, augment=augment if train else None)
    pin_memory = os.environ.get("CEMR_PIN_MEMORY", "0") == "1"
    if train and sampler_mode == "inverse_frequency":
        counts = np.bincount(y, minlength=config.N_CLASSES).astype(np.float64)
        weights = 1.0 / np.maximum(counts, 1.0)
        sample_weights = weights[y]
        sampler = WeightedRandomSampler(
            torch.from_numpy(sample_weights).float(),
            num_samples=len(y),
            replacement=True,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=pin_memory)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0, pin_memory=pin_memory)


def class_weights(y, mode, device):
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=config.N_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    if mode in {"balanced_sqrt", "domain"}:
        w = np.sqrt(counts.max() / counts)
        w = np.clip(w, 0.5, 10.0)
    elif mode == "balanced_log":
        w = np.log1p(counts.max() / counts)
        w = np.clip(w, 0.5, 8.0)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    return torch.tensor(w, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class ContextAwareLoss(nn.Module):
    def __init__(self, weight=None, gamma=1.5, margin=0.35, margin_weight=0.15):
        super().__init__()
        self.focal = FocalLoss(weight=weight, gamma=gamma)
        self.margin = margin
        self.margin_weight = margin_weight

    def forward(self, logits, target):
        base = self.focal(logits, target)
        mask = (target == 1) | (target == 3)
        if not mask.any():
            return base
        target_logit = logits[mask].gather(1, target[mask, None]).squeeze(1)
        n_logit = logits[mask, 0]
        margin_loss = F.relu(self.margin - (target_logit - n_logit)).mean()
        return base + self.margin_weight * margin_loss


class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - mean) / std


class SEBlock1d(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(4, channels // ratio)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.net(x).unsqueeze(-1)
        return x * w


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride=1, groups=1, dropout=0.0):
        super().__init__()
        pad = kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=pad, groups=groups, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


def transformer_encoder(d_model, heads, layers, d_ff=None, dropout=0.1):
    if d_ff is None:
        d_ff = d_model * 2
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=heads,
        dim_feedforward=d_ff,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(enc_layer, num_layers=layers)


class CATNetLite(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(1, 32, 7, dropout=0.05),
            nn.MaxPool1d(2),
            ConvBNAct(32, 64, 5, dropout=0.05),
            nn.MaxPool1d(2),
            ConvBNAct(64, 128, 5, dropout=0.1),
            nn.MaxPool1d(2),
            ConvBNAct(128, 128, 3, dropout=0.1),
        )
        self.attn = SEBlock1d(128, ratio=8)
        self.pos = SinusoidalPosition(128)
        self.encoder = transformer_encoder(128, heads=4, layers=2, d_ff=256, dropout=0.3)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        z = self.attn(self.conv(x))
        z = z.transpose(1, 2)
        z = self.encoder(self.pos(z))
        return self.head(z.mean(dim=1))


class ECGTransFormLite(nn.Module):
    def __init__(self, in_ch=1, n_classes=5):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNAct(in_ch, 64, k, dropout=0.05) for k in (3, 5, 7)])
        self.project = ConvBNAct(64 * 3, 128, 1, dropout=0.05)
        self.crm = SEBlock1d(128, ratio=8)
        self.pos = SinusoidalPosition(128)
        self.forward_encoder = transformer_encoder(128, heads=4, layers=2, d_ff=256, dropout=0.2)
        self.backward_encoder = transformer_encoder(128, heads=4, layers=2, d_ff=256, dropout=0.2)
        self.fuse = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.2), nn.Linear(128, n_classes))

    def forward(self, x):
        z = torch.cat([branch(x) for branch in self.branches], dim=1)
        z = self.crm(self.project(z)).transpose(1, 2)
        zf = self.forward_encoder(self.pos(z)).mean(dim=1)
        zb = self.backward_encoder(self.pos(torch.flip(z, dims=[1]))).mean(dim=1)
        z = self.fuse(torch.cat([zf, zb], dim=1))
        return self.head(z)


class RelativeEncoderLayer(nn.Module):
    def __init__(self, d_model=128, heads=4, d_ff=256, dropout=0.15, max_len=256):
        super().__init__()
        self.heads = heads
        self.max_len = max_len
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.rel_bias = nn.Parameter(torch.zeros(heads, 2 * max_len - 1))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        b, l, _ = x.shape
        pos = torch.arange(l, device=x.device)
        rel = pos[None, :] - pos[:, None] + self.max_len - 1
        rel = rel.clamp(0, 2 * self.max_len - 2)
        bias = self.rel_bias[:, rel].repeat(b, 1, 1)
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, attn_mask=bias, need_weights=False)
        x = x + attn
        x = x + self.ff(self.norm2(x))
        return x


class MSFTLite(nn.Module):
    def __init__(self, in_ch=1, n_classes=5):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNAct(in_ch, 32, k, dropout=0.05) for k in (3, 5, 9, 15)])
        self.project = ConvBNAct(128, 128, 1, dropout=0.05)
        self.pos = SinusoidalPosition(128)
        self.layers = nn.ModuleList([
            RelativeEncoderLayer(128, heads=4, d_ff=256, dropout=0.15, max_len=256)
            for _ in range(2)
        ])
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.15), nn.Linear(128, n_classes))

    def forward(self, x):
        z = torch.cat([branch(x) for branch in self.branches], dim=1)
        z = self.project(z).transpose(1, 2)
        z = self.pos(z)
        for layer in self.layers:
            z = layer(z)
        return self.head(z.mean(dim=1))


class PatchBranch(nn.Module):
    def __init__(self, in_ch, patch, d_model, heads, layers, dropout):
        super().__init__()
        self.patch = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=patch, bias=False)
        self.pos = SinusoidalPosition(d_model)
        self.encoder = transformer_encoder(d_model, heads=heads, layers=layers, d_ff=d_model * 2, dropout=dropout)

    def forward(self, x):
        z = self.patch(x).transpose(1, 2)
        return self.encoder(self.pos(z)).mean(dim=1)


class MCTnetLite(nn.Module):
    def __init__(self, in_ch=1, n_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 32, 3),
            ConvBNAct(32, 64, 7),
            ConvBNAct(64, 128, 11),
        )
        self.branch4 = PatchBranch(128, patch=4, d_model=128, heads=4, layers=1, dropout=0.15)
        self.branch8 = PatchBranch(128, patch=8, d_model=128, heads=4, layers=1, dropout=0.15)
        self.mu = nn.Linear(256, 128)
        self.logvar = nn.Linear(256, 128)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.15), nn.Linear(128, n_classes))

    def forward(self, x, return_aux=False):
        z = self.stem(x)
        h = torch.cat([self.branch4(z), self.branch8(z)], dim=1)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-6.0, 4.0)
        if self.training:
            eps = torch.randn_like(mu)
            z = mu + eps * torch.exp(0.5 * logvar)
        else:
            z = mu
        logits = self.head(z)
        if not return_aux:
            return logits
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return logits, {"ib_kl": kl}


class MedformerLite(nn.Module):
    def __init__(self, in_ch=2, n_classes=5):
        super().__init__()
        self.patchers = nn.ModuleList([
            nn.Conv1d(in_ch, 128, kernel_size=p, stride=p, bias=False)
            for p in (2, 4, 8)
        ])
        self.pos = SinusoidalPosition(128, max_len=512)
        self.encoder = transformer_encoder(128, heads=4, layers=6, d_ff=256, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.1), nn.Linear(128, n_classes))

    def forward(self, x):
        tokens = [patcher(x).transpose(1, 2) for patcher in self.patchers]
        z = torch.cat(tokens, dim=1)
        z = self.encoder(self.pos(z))
        return self.head(z.mean(dim=1))


class PatchTSTClassifier(nn.Module):
    def __init__(self, in_ch=2, n_classes=5):
        super().__init__()
        self.revin = RevIN()
        self.patch_len = 16
        self.stride = 8
        self.proj = nn.Linear(self.patch_len, 128)
        self.pos = SinusoidalPosition(128, max_len=64)
        self.encoder = transformer_encoder(128, heads=16, layers=3, d_ff=256, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.1), nn.Linear(128, n_classes))

    def forward(self, x):
        x = self.revin(x)
        b, c, _ = x.shape
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().view(b * c, patches.shape[2], self.patch_len)
        z = self.proj(patches)
        z = self.encoder(self.pos(z)).mean(dim=1)
        z = z.view(b, c, -1).mean(dim=1)
        return self.head(z)


class ModernTCNBlock(nn.Module):
    def __init__(self, channels, kernel, dropout=0.1):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels, bias=False)
        self.pw1 = nn.Conv1d(channels, channels * 2, 1)
        self.pw2 = nn.Conv1d(channels * 2, channels, 1)
        self.bn = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dw(x)
        h = self.bn(h)
        h = F.gelu(self.pw1(h))
        h = self.drop(self.pw2(h))
        return x + h


class ModernTCNClassifier(nn.Module):
    def __init__(self, in_ch=2, n_classes=5):
        super().__init__()
        self.stem = ConvBNAct(in_ch, 64, 7, stride=1, dropout=0.05)
        self.stage1 = nn.Sequential(ModernTCNBlock(64, 13), ModernTCNBlock(64, 13))
        self.down1 = ConvBNAct(64, 128, 3, stride=2)
        self.stage2 = nn.Sequential(ModernTCNBlock(128, 25), ModernTCNBlock(128, 25))
        self.down2 = ConvBNAct(128, 256, 3, stride=2)
        self.stage3 = nn.Sequential(ModernTCNBlock(256, 51), ModernTCNBlock(256, 51))
        self.head = nn.Sequential(nn.LayerNorm(256), nn.Dropout(0.1), nn.Linear(256, n_classes))

    def forward(self, x):
        z = self.stage1(self.stem(x))
        z = self.stage2(self.down1(z))
        z = self.stage3(self.down2(z))
        z = z.mean(dim=-1)
        return self.head(z)


class MovingAvg(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        pad = (self.kernel - 1) // 2
        front = x[:, :, :1].repeat(1, 1, pad)
        end = x[:, :, -1:].repeat(1, 1, pad)
        xp = torch.cat([front, x, end], dim=-1)
        return F.avg_pool1d(xp, kernel_size=self.kernel, stride=1)


class TimeMixerClassifier(nn.Module):
    def __init__(self, in_ch=2, n_classes=5):
        super().__init__()
        self.decomp = MovingAvg(25)
        self.scales = nn.ModuleList([
            nn.Sequential(ConvBNAct(in_ch * 2, 64, 3), ConvBNAct(64, 64, 3)),
            nn.Sequential(ConvBNAct(in_ch * 2, 64, 5), ConvBNAct(64, 64, 5)),
            nn.Sequential(ConvBNAct(in_ch * 2, 64, 7), ConvBNAct(64, 64, 7)),
        ])
        self.mix = nn.Sequential(nn.Linear(64 * 3, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, n_classes))

    def forward(self, x):
        outs = []
        current = x
        for i, block in enumerate(self.scales):
            trend = self.decomp(current)
            seasonal = current - trend
            z = block(torch.cat([seasonal, trend], dim=1)).mean(dim=-1)
            outs.append(z)
            if i < 2:
                current = F.avg_pool1d(current, kernel_size=2, stride=2)
        z = self.mix(torch.cat(outs, dim=1))
        return self.head(z)


class Inception2d(nn.Module):
    def __init__(self, channels, num_kernels=6):
        super().__init__()
        kernels = [1, 3, 5, 7, 9, 11][:num_kernels]
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=(1, k), padding=(0, k // 2), bias=False)
            for k in kernels
        ])
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        z = torch.stack([conv(x) for conv in self.convs], dim=0).mean(dim=0)
        return F.gelu(self.bn(z))


class TimesBlock(nn.Module):
    def __init__(self, d_model=64, top_k=3, num_kernels=6):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(Inception2d(d_model, num_kernels), Inception2d(d_model, num_kernels))

    def forward(self, x):
        b, c, l = x.shape
        with torch.amp.autocast("cuda", enabled=False):
            xf = torch.fft.rfft(x.float(), dim=-1)
        amp = xf.abs().mean(dim=(0, 1))
        amp[0] = 0
        k = min(self.top_k, amp.numel() - 1)
        freqs = torch.topk(amp, k=k).indices
        outs = []
        for f in freqs:
            period = max(2, int(round(l / int(f.item()))))
            length = int(math.ceil(l / period) * period)
            if length > l:
                pad = torch.zeros(b, c, length - l, device=x.device, dtype=x.dtype)
                xp = torch.cat([x, pad], dim=-1)
            else:
                xp = x
            xp = xp.reshape(b, c, length // period, period)
            zp = self.conv(xp).reshape(b, c, length)[..., :l]
            outs.append(zp)
        return x + torch.stack(outs, dim=0).mean(dim=0)


class TimesNetClassifier(nn.Module):
    def __init__(self, in_ch=2, n_classes=5):
        super().__init__()
        self.embed = ConvBNAct(in_ch, 64, 3)
        self.blocks = nn.Sequential(TimesBlock(64, top_k=3, num_kernels=6), TimesBlock(64, top_k=3, num_kernels=6))
        self.ff = nn.Sequential(ConvBNAct(64, 128, 1), ConvBNAct(128, 64, 1))
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(0.1), nn.Linear(64, n_classes))

    def forward(self, x):
        z = self.embed(x)
        z = self.blocks(z)
        z = self.ff(z).mean(dim=-1)
        return self.head(z)


@dataclass
class ModelSpec:
    name: str
    input_mode: str
    make_model: callable
    optimizer: str
    lr: float
    weight_decay: float
    batch_size: int
    epochs: int
    patience: int
    loss: str
    weight_mode: str
    sampler_mode: str = None
    augment: str = None
    resample_mode: str = None
    ib_beta: float = 0.0
    source: str = ""
    params: str = ""


def model_specs():
    return [
        ModelSpec(
            name="CAT-Net",
            input_mode="single",
            make_model=lambda: CATNetLite(config.N_CLASSES),
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="ce",
            weight_mode="none",
            sampler_mode="inverse_frequency",
            resample_mode="smote_tomek",
            source="2024 BSPC CAT-Net, CNN + attention + Transformer with imbalance handling.",
            params="4 CNN layers, SE/channel attention, 2 Transformer layers, d_model=128, heads=4, dropout=0.3, Adam lr=1e-3.",
        ),
        ModelSpec(
            name="ECGTransForm",
            input_mode="single",
            make_model=lambda: ECGTransFormLite(1, config.N_CLASSES),
            optimizer="adam",
            lr=1e-3,
            weight_decay=0.0,
            batch_size=128,
            epochs=100,
            patience=12,
            loss="context",
            weight_mode="domain",
            source="2024 BSPC ECGTransForm, multi-scale CNN, channel recalibration, bidirectional Transformer, context-aware loss.",
            params="Kernels 3/5/7, d_model=128, heads=4, Transformer layers=2, CRM ratio=8, Adam lr=1e-3.",
        ),
        ModelSpec(
            name="MSFT",
            input_mode="single",
            make_model=lambda: MSFTLite(1, config.N_CLASSES),
            optimizer="adamw",
            lr=5e-4,
            weight_decay=1e-4,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="focal",
            weight_mode="domain",
            source="2024 BSPC MSFT, multi-scale feature transformer with sparse attention idea and biased relative position encoding.",
            params="Kernels 3/5/9/15, d_model=128, heads=4, layers=2, relative bias, AdamW lr=5e-4.",
        ),
        ModelSpec(
            name="MCTnet",
            input_mode="single",
            make_model=lambda: MCTnetLite(1, config.N_CLASSES),
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="ce",
            weight_mode="domain",
            ib_beta=1e-3,
            source="2024 Information Sciences MCTnet, multi-scale CNN, dual-branch Transformer, information bottleneck.",
            params="Patch branches 4/8, d_model=128, heads=4, IB beta=1e-3, AdamW lr=1e-3.",
        ),
        ModelSpec(
            name="Medformer",
            input_mode="dual",
            make_model=lambda: MedformerLite(2, config.N_CLASSES),
            optimizer="adamw",
            lr=1e-4,
            weight_decay=1e-4,
            batch_size=128,
            epochs=100,
            patience=10,
            loss="ce",
            weight_mode="balanced_sqrt",
            augment="jitter_mask",
            source="NeurIPS 2024 Medformer, multi-granularity patching for medical time series.",
            params="Patch list 2/4/8, e_layers=6, d_model=128, d_ff=256, AdamW lr=1e-4.",
        ),
        ModelSpec(
            name="PatchTST",
            input_mode="dual",
            make_model=lambda: PatchTSTClassifier(2, config.N_CLASSES),
            optimizer="adamw",
            lr=1e-4,
            weight_decay=1e-4,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="ce",
            weight_mode="balanced_sqrt",
            source="ICLR 2023 PatchTST, channel-independent patch Transformer with RevIN.",
            params="patch_len=16, stride=8, e_layers=3, d_model=128, d_ff=256, heads=16, RevIN, AdamW lr=1e-4.",
        ),
        ModelSpec(
            name="ModernTCN",
            input_mode="dual",
            make_model=lambda: ModernTCNClassifier(2, config.N_CLASSES),
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="ce",
            weight_mode="balanced_log",
            source="ICLR 2024 ModernTCN, modern large-kernel depthwise temporal convolution network.",
            params="Large kernels 13/25/51, stage dims 64/128/256, dropout=0.1, AdamW lr=1e-3.",
        ),
        ModelSpec(
            name="TimeMixer",
            input_mode="dual",
            make_model=lambda: TimeMixerClassifier(2, config.N_CLASSES),
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="ce",
            weight_mode="balanced_log",
            source="ICLR 2024 TimeMixer, multi-scale decomposable mixing for time series.",
            params="2 downsampling levels, moving average=25, d_model=64, d_ff=128, e_layers=2, AdamW lr=1e-3.",
        ),
        ModelSpec(
            name="TimesNet",
            input_mode="dual",
            make_model=lambda: TimesNetClassifier(2, config.N_CLASSES),
            optimizer="adamw",
            lr=1e-3,
            weight_decay=1e-4,
            batch_size=128,
            epochs=80,
            patience=10,
            loss="ce",
            weight_mode="balanced_log",
            source="ICLR 2023 TimesNet, period-aware 2D temporal variation modeling.",
            params="FFT top_k=3, num_kernels=6, d_model=64, d_ff=128, e_layers=2, AdamW lr=1e-3.",
        ),
    ]


def make_loss(spec, y_train, device):
    weights = class_weights(y_train, spec.weight_mode, device)
    if spec.loss == "ce":
        return nn.CrossEntropyLoss(weight=weights)
    if spec.loss == "focal":
        return FocalLoss(weight=weights, gamma=2.0)
    if spec.loss == "context":
        return ContextAwareLoss(weight=weights, gamma=1.5, margin=0.35, margin_weight=0.15)
    raise ValueError(f"Unknown loss: {spec.loss}")


def make_optimizer(spec, model):
    if spec.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    if spec.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=spec.lr, weight_decay=spec.weight_decay)
    raise ValueError(f"Unknown optimizer: {spec.optimizer}")


def forward_with_aux(model, x, spec):
    if spec.ib_beta:
        return model(x, return_aux=True)
    return model(x), {}


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(targets), np.concatenate(preds)


def train_once(spec, seed, device, smoke=False, batch_size_override=None):
    set_seed(seed)
    x_all, y_all, x_test, y_test = load_cache(spec.input_mode)
    train_idx, val_idx = stratified_split(y_all, seed)
    x_train, y_train = x_all[train_idx], y_all[train_idx]
    x_val, y_val = x_all[val_idx], y_all[val_idx]
    x_train, x_val, x_test = normalize_from_train(x_train, x_val, x_test)
    used_resampling = ""
    if spec.resample_mode == "smote_tomek":
        x_res, y_res, ok = smote_tomek_waveforms(x_train, y_train, seed)
        if ok:
            x_train, y_train = x_res, y_res
            used_resampling = "smote_tomek"
        else:
            used_resampling = "sampler_fallback"

    batch_size = batch_size_override or spec.batch_size
    epochs = 1 if smoke else spec.epochs
    patience = 1 if smoke else spec.patience
    train_loader = make_loader(
        x_train,
        y_train,
        batch_size,
        train=True,
        sampler_mode=spec.sampler_mode,
        augment=spec.augment,
    )
    val_loader = make_loader(x_val, y_val, batch_size, train=False)
    test_loader = make_loader(x_test, y_test, batch_size, train=False)

    model = spec.make_model().to(device)
    loss_fn = make_loss(spec, y_train, device)
    optimizer = make_optimizer(spec, model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_state = None
    best_val = -1.0
    best_epoch = 0
    bad_epochs = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, aux = forward_with_aux(model, xb, spec)
                loss = loss_fn(logits, yb)
                if spec.ib_beta:
                    loss = loss + spec.ib_beta * aux["ib_kl"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(yb)

        y_val_true, y_val_pred = predict(model, val_loader, device)
        val_metrics = classification_metrics(y_val_true, y_val_pred)
        val_score = val_metrics["macro_f1_4"]
        if val_score > best_val:
            best_val = val_score
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(
            f"[{spec.name} seed={seed}] epoch={epoch:03d} "
            f"loss={total_loss / max(1, len(y_train)):.4f} val_MF1(4)={val_score:.4f} best={best_val:.4f}",
            flush=True,
        )
        if bad_epochs >= patience:
            break

    train_time = time.time() - start
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_start = time.time()
    y_true, y_pred = predict(model, test_loader, device)
    predict_time = time.time() - pred_start
    metrics = metrics_row(y_true, y_pred)
    cm = classification_metrics(y_true, y_pred)["confusion_matrix"].tolist()
    row = {
        "model": spec.name,
        "seed": seed,
        "input_mode": spec.input_mode,
        "source": spec.source,
        "params": spec.params,
        "optimizer": spec.optimizer,
        "lr": spec.lr,
        "weight_decay": spec.weight_decay,
        "batch_size": batch_size,
        "epochs_config": spec.epochs,
        "epochs_run": best_epoch if not smoke else 1,
        "best_val_macro_f1_4": best_val,
        "loss": spec.loss,
        "weight_mode": spec.weight_mode,
        "sampler_mode": spec.sampler_mode or "",
        "augment": spec.augment or "",
        "resample_mode": used_resampling or (spec.resample_mode or ""),
        "ib_beta": spec.ib_beta,
        "train_time": train_time,
        "predict_time": predict_time,
        **metrics,
    }
    return row, cm


def completed_keys():
    if not DETAIL_PATH.exists():
        return set()
    df = pd.read_csv(DETAIL_PATH)
    return set(zip(df["model"].astype(str), df["seed"].astype(int)))


def append_detail(row):
    df = pd.DataFrame([row])
    header = not DETAIL_PATH.exists()
    df.to_csv(DETAIL_PATH, mode="a", header=header, index=False, encoding="utf-8")


def load_confusion():
    if CONFUSION_PATH.exists():
        return json.loads(CONFUSION_PATH.read_text(encoding="utf-8"))
    return {}


def save_confusion(data):
    CONFUSION_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize():
    if not DETAIL_PATH.exists():
        return pd.DataFrame()
    detail = pd.read_csv(DETAIL_PATH)
    metrics = [
        "accuracy", "macro_f1_5", "macro_f1_4", "macro_f1_3_nsv",
        "Se_N", "Se_S", "Se_V", "Se_F", "Se_Q",
        "Pr_N", "Pr_S", "Pr_V", "Pr_F", "Pr_Q",
        "F1_N", "F1_S", "F1_V", "F1_F", "F1_Q",
        "train_time", "predict_time", "best_val_macro_f1_4", "epochs_run",
    ]
    rows = []
    for name, g in detail.groupby("model"):
        row = {
            "model": name,
            "n_seeds": int(g["seed"].nunique()),
            "seeds": " ".join(str(int(s)) for s in sorted(g["seed"].unique())),
            "input_mode": str(g["input_mode"].iloc[0]),
            "source": str(g["source"].iloc[0]),
            "params": str(g["params"].iloc[0]),
        }
        for metric in metrics:
            if metric in g.columns:
                row[f"{metric}_mean"] = float(g[metric].mean())
                row[f"{metric}_std"] = float(g[metric].std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("macro_f1_4_mean", ascending=False)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    return summary


def markdown_table(df, cols):
    if df is None or df.empty:
        return "_No data._"
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_markdown(index=False)


def build_report():
    summary = summarize()
    final_path = RESULTS / "final_method_summary.csv"
    lines = [
        "# Modern Deep MIT-BIH Baselines",
        "",
        "Protocol: MIT-BIH AAMI inter-patient DS1 -> DS2, 5 seeds, DS2 is never used for early stopping.",
        "These are compact in-repository reproductions of recent architectures with model-specific hyperparameters.",
        "",
        "## Modern Deep Baseline Ranking",
        "",
        markdown_table(
            summary,
            [
                "model", "n_seeds", "accuracy_mean", "macro_f1_5_mean", "macro_f1_4_mean",
                "Se_N_mean", "Se_S_mean", "Se_V_mean", "Se_F_mean",
                "F1_S_mean", "F1_F_mean", "train_time_mean",
            ],
        ),
        "",
    ]
    if final_path.exists() and not summary.empty:
        final_df = pd.read_csv(final_path)
        best_final = final_df.sort_values("macro_f1_4", ascending=False).head(1).copy()
        combined_rows = []
        for _, r in best_final.iterrows():
            combined_rows.append({
                "method": f"FinalMethod:{r.get('model', '')}",
                "accuracy": r.get("accuracy"),
                "macro_f1_5": r.get("macro_f1_5"),
                "macro_f1_4": r.get("macro_f1_4"),
                "Se_S": r.get("Se_S"),
                "Se_F": r.get("Se_F"),
                "F1_S": r.get("F1_S"),
                "F1_F": r.get("F1_F"),
                "train_time": r.get("train_time"),
            })
        for _, r in summary.iterrows():
            combined_rows.append({
                "method": r["model"],
                "accuracy": r.get("accuracy_mean"),
                "macro_f1_5": r.get("macro_f1_5_mean"),
                "macro_f1_4": r.get("macro_f1_4_mean"),
                "Se_S": r.get("Se_S_mean"),
                "Se_F": r.get("Se_F_mean"),
                "F1_S": r.get("F1_S_mean"),
                "F1_F": r.get("F1_F_mean"),
                "train_time": r.get("train_time_mean"),
            })
        combined = pd.DataFrame(combined_rows).sort_values("macro_f1_4", ascending=False)
        lines.extend([
            "## Final Method vs Modern Deep Baselines",
            "",
            combined.to_markdown(index=False),
            "",
        ])
    lines.extend([
        "## Model-Specific Settings",
        "",
        markdown_table(summary, ["model", "input_mode", "source", "params"]),
        "",
    ])
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {SUMMARY_PATH}")
    print(f"Saved {REPORT_PATH}")


def run_spec_seed(spec, seed, device, smoke=False):
    batch_size = spec.batch_size
    while True:
        try:
            return train_once(spec, seed, device, smoke=smoke, batch_size_override=batch_size)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg and device.type == "cuda" and batch_size > 16:
                torch.cuda.empty_cache()
                batch_size = max(16, batch_size // 2)
                print(f"[{spec.name} seed={seed}] CUDA OOM; retry with batch_size={batch_size}", flush=True)
                continue
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="Modern 2023-2026 deep baselines on MIT-BIH DS1 -> DS2")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=SEEDS, help="Seeds to run")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch for quick verification")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--rebuild-report", action="store_true", help="Only rebuild summary/report from existing details")
    args = parser.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    if args.rebuild_report:
        build_report()
        return
    if not CACHE.exists():
        raise FileNotFoundError(f"Missing cache: {CACHE}")

    specs = model_specs()
    if args.models:
        wanted = {m.lower() for m in args.models}
        specs = [s for s in specs if s.name.lower() in wanted]
        missing = wanted - {s.name.lower() for s in specs}
        if missing:
            raise ValueError(f"Unknown model names: {sorted(missing)}")
    if args.smoke and not args.models:
        specs = specs[:1]
        args.seeds = args.seeds[:1]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    done = completed_keys()
    conf = load_confusion()
    for spec in specs:
        for seed in args.seeds:
            key = (spec.name, int(seed))
            if key in done and not args.smoke:
                print(f"Skip completed {spec.name} seed={seed}")
                continue
            print(f"Running {spec.name} seed={seed}")
            row, cm = run_spec_seed(spec, int(seed), device, smoke=args.smoke)
            if not args.smoke:
                append_detail(row)
                conf[f"{spec.name}|{seed}"] = cm
                save_confusion(conf)
                done.add(key)
            print(
                f"[{spec.name} seed={seed}] Acc={row['accuracy'] * 100:.2f}% "
                f"M-F1(5)={row['macro_f1_5']:.4f} M-F1(4)={row['macro_f1_4']:.4f} "
                f"Se_S={row.get('Se_S', 0) * 100:.1f}% Se_F={row.get('Se_F', 0) * 100:.1f}% "
                f"train={row['train_time']:.1f}s",
                flush=True,
            )
    if not args.smoke:
        build_report()


if __name__ == "__main__":
    main()
