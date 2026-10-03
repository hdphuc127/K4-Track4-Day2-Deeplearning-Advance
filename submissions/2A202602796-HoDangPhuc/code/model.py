"""model.py - backbone qua timm, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import timm
import torch

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """init: scratch | frozen | finetune. Tag trọng số thực tế nằm ở model.pretrained_cfg."""
    name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_ids(model) -> set:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi thứ trừ head. Train loop gọi set_train_mode() để giữ BN backbone ở eval."""
    head = _head_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head


def set_train_mode(model, frozen: bool) -> None:
    """model.train(); nếu backbone đóng băng thì mọi module ngoài head về eval, để BN không cập nhật
    running stats trong khi trọng số đứng yên (nếu không, train/eval lệch nhau)."""
    model.train()
    if frozen:
        head_mods = set(model.get_classifier().modules())
        for m in model.modules():
            if m not in head_mods and not list(m.children()):
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm (slide trang 52): backbone (decay), norm+bias backbone (không decay), head (lr cao)."""
    head = _head_ids(model)
    g = {"bb": [], "bb_nodecay": [], "head": []}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        g["head" if id(p) in head else ("bb" if p.ndim > 1 else "bb_nodecay")].append(p)
    out = [
        {"params": g["bb"], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": g["bb_nodecay"], "lr": lr_backbone, "weight_decay": 0.0},
        {"params": g["head"], "lr": lr_head, "weight_decay": weight_decay},
    ]
    return [d for d in out if d["params"]]


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """torch.utils.flop_counter đếm FLOPs (2 x MAC); chia 2 để ra GMAC. Chỉ đếm matmul/conv."""
    from torch.utils.flop_counter import FlopCounterMode
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, 3, img_size, img_size, device=dev))
    model.train(was)
    return fc.get_total_flops() / 2 / 1e9
