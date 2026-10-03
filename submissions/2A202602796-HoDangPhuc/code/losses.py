"""losses.py - loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """q' = (1-eps)*onehot + eps/K, tự cài đặt; eps=0 trùng CE."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.eps = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        return ((1 - self.eps) * nll + self.eps * -logp.mean(dim=-1)).mean()


class FocalLoss(nn.Module):
    """FL = -alpha_t (1-p_t)^gamma log p_t; gamma=0, alpha=None -> đúng CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp.exp()) ** self.gamma) * logp
        if self.alpha is not None:
            loss = loss * torch.as_tensor(self.alpha, device=loss.device, dtype=loss.dtype)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """beta=0: 1/n_c chuẩn hoá về trung bình 1; beta>0: (1-beta)/(1-beta^n_c), tổng = K."""
    n = torch.as_tensor(np.asarray(counts), dtype=torch.double)
    w = 1.0 / n if not beta else (1 - beta) / (1 - beta ** n)
    return (w * len(n) / w.sum()).float()


def build_criterion(kind: str = "ce", **kw):
    """kind: ce | ls | focal | ce_weighted. kw: smoothing, gamma, alpha, weight."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"loss không hợp lệ: {kind}")


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x = lam * x + (1 - lam) * x[perm]
    elif mode == "cutmix":
        H, W = x.shape[-2:]
        r = np.sqrt(1 - lam)
        h, w = int(H * r), int(W * r)
        cy, cx = np.random.randint(H), np.random.randint(W)
        y1, y2 = int(np.clip(cy - h // 2, 0, H)), int(np.clip(cy + h // 2, 0, H))
        x1, x2 = int(np.clip(cx - w // 2, 0, W)), int(np.clip(cx + w // 2, 0, W))
        x = x.clone()
        x[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)  # diện tích thực sau khi cắt ra ngoài biên
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
