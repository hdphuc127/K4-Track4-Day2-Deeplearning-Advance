"""inference.py - các phương pháp suy luận (Bước 3). Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp sang test."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    """view: hàm batch -> batch, hoặc -> list các batch (multi-view; khi đó logit được TRUNG BÌNH
    trên view). Với TTA muốn gộp theo xác suất, gọi từng view riêng rồi aggregate_views."""
    model.eval()
    names, ys, outs = [], [], []
    for x, y, f in loader:
        x = x.to(device)
        v = view(x) if view else x
        o = torch.stack([model(b).float() for b in v]).mean(0) if isinstance(v, list) else model(v).float()
        outs.append(o.cpu())
        ys.append(y)
        names += list(f)
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy()


def view_identity(x):
    return x


def view_hflip(x):
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) cỡ `crop`; flip=True thêm bản lật của từng crop (10 view)."""
    H, W = x.shape[-2:]
    cy, cx = (H - crop) // 2, (W - crop) // 2
    pos = [(0, 0), (0, W - crop), (H - crop, 0), (H - crop, W - crop), (cy, cx)]
    v = [x[..., i:i + crop, j:j + crop] for i, j in pos]
    return v + [torch.flip(b, dims=[-1]) for b in v] if flip else v


def views_multiscale(x, sizes):
    """Resize về từng kích thước. Chỉ dùng với CNN có global pooling; ViT/Swin cần img_size cố định."""
    return [F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]


def _softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """space='prob': trung bình softmax; 'logit': trung bình logit rồi softmax."""
    if space == "prob":
        return np.mean([_softmax(l) for l in logits_per_view], axis=0)
    if space == "logit":
        return _softmax(np.mean(logits_per_view, axis=0))
    raise ValueError(f"space không hợp lệ: {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất; các mảng phải cùng tập ảnh và thứ tự file."""
    shapes = {np.shape(p) for p in list_of_probs}
    if len(shapes) != 1:
        raise ValueError(f"khác dạng: {shapes}")
    return np.mean(list_of_probs, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL (LBFGS trên log T)."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.detach().exp())


def apply_temperature(logits, T: float):
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def fuse_conv_bn(model):
    """Gộp BN2d vào Conv2d liền trước (chính xác lúc suy luận). Trả về bản sao đã gộp.
    Chỉ gộp cặp (conv, bn) liền kề trong cùng một Sequential hoặc cặp thuộc tính tên conv*/bn*
    (BasicBlock/Bottleneck của timm). Với ViT/Swin/ConvNeXt (LayerNorm) trả về nguyên bản không đổi."""
    m = copy.deepcopy(model).eval()

    def fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        out = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                        conv.padding, conv.dilation, conv.groups, bias=True)
        b0 = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        out.weight.data = conv.weight.data * s.view(-1, 1, 1, 1)
        out.bias.data = bn.bias + s * (b0 - bn.running_mean)
        return out

    with torch.no_grad():
        for parent in m.modules():
            names = list(parent._modules)
            for i, n in enumerate(names):
                c = parent._modules[n]
                if not isinstance(c, nn.Conv2d):
                    continue
                if isinstance(parent, nn.Sequential) and i + 1 < len(names):
                    bn_name = names[i + 1]
                else:  # timm: conv1/bn1, conv2/bn2, ...
                    bn_name = n.replace("conv", "bn") if n.startswith("conv") else None
                bn = parent._modules.get(bn_name) if bn_name else None
                if isinstance(bn, nn.BatchNorm2d) and bn.num_features == c.out_channels:
                    parent._modules[n] = fuse(c, bn)
                    parent._modules[bn_name] = nn.Identity()
    return m
