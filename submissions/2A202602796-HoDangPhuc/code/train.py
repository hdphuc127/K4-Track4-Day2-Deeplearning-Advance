"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

    python train.py --set exp_id=B01 backbone=resnet50 seed=0
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset as D
import losses as L
import model as M
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST (quy tắc S4) ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định random/numpy/torch. cudnn.benchmark tắt, nhưng một số kernel CUDA vẫn không
    deterministic hoàn toàn, nên chạy lại chỉ tái lập gần đúng (ghi vào báo cáo)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    return torch.optim.AdamW(M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật theo bước. Nhân hệ số lên LR gốc của từng nhóm."""
    total = cfg.epochs * steps_per_epoch
    warm = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class EMA:
    """W_ema <- d*W_ema + (1-d)*W cho tham số; buffer (BN running stats) được sao chép thẳng."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model) -> None:
        for e, p in zip(self.module.parameters(), model.parameters()):
            e.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for e, b in zip(self.module.buffers(), model.buffers()):
            e.copy_(b)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    M.set_train_mode(model, cfg.init == "frozen")
    use_amp = cfg.amp and device.type == "cuda"
    tot, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if cfg.mix:
            x, tgt = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            out = model(x)
        loss = L.mixed_loss(criterion, out, tgt) if cfg.mix else criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema:
            ema.update(model)
        tot += loss.item() * len(y)
        n += len(y)
    return {"train_loss": tot / n, "lr": optimizer.param_groups[-1]["lr"]}


@torch.inference_mode()
def evaluate(model, loader, criterion, device, amp: bool = True):
    """Trả về (filenames, y_true[N], logits[N,9], loss), đúng thứ tự loader."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    names, ys, outs = [], [], []
    for x, y, f in loader:
        with torch.autocast(device.type, enabled=use_amp):
            outs.append(model(x.to(device)).float().cpu())
        ys.append(y)
        names += list(f)
    logits, y_true = torch.cat(outs), torch.cat(ys)
    return names, y_true.numpy(), logits.numpy(), criterion(logits, y_true).item()


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(h.epoch, h.train_loss, label="train")
    ax[0].plot(h.epoch, h.val_loss, label="val")
    ax[0].set(title="Loss", xlabel="epoch", ylabel="loss")
    ax[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1")
    ax[1].plot(h.epoch, h.val_top1, label="val top-1")
    ax[1].set(title="Metric (val)", xlabel="epoch", ylabel="score")
    ax[2].plot(h.epoch, h.lr)
    ax[2].set(title="LR (nhóm head, cuối epoch)", xlabel="epoch", ylabel="lr")
    for a in ax[:2]:
        a.legend()
        a.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _metrics(y, logits):
    p = torch.softmax(torch.from_numpy(logits).double(), 1).numpy()
    return compute_metrics(y, p.argmax(1), p), p


def run(cfg: Config) -> dict:
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, te = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(tr, va, te, cfg.images_dir, D.TOTAL_IMAGES if cfg.images_dir != "synthetic" else None)

    mk = lambda df, train: D.make_loader(  # noqa: E731
        df, cfg.images_dir, D.build_transforms(train, cfg.img_size, cfg.aug), cfg.batch_size, train,
        cfg.sampler if train else None, cfg.num_workers)
    train_loader, val_loader = mk(tr, True), mk(va, False)

    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init).to(device)
    import timm, torchvision
    info = {**dataclasses.asdict(cfg), "weights_tag": str(getattr(model, "pretrained_cfg", {}).get("tag")),
            "torch": torch.__version__, "timm": timm.__version__, "torchvision": torchvision.__version__,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    (rd / "config.json").write_text(json.dumps(info, indent=2))
    kw = {"smoothing": cfg.label_smoothing, "gamma": cfg.focal_gamma}
    if cfg.loss == "ce_weighted":  # trọng số chỉ từ TRAIN
        cnt = np.bincount(tr["Label"], minlength=D.NUM_CLASSES)
        kw["weight"] = L.class_weights(cnt, cfg.class_weight_beta or 0.0).to(device)
    criterion = L.build_criterion(cfg.loss, **kw)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    eval_model = ema.module if ema else model  # đánh giá bằng trọng số EMA nếu có

    history, best, best_state, t_train = [], (-1.0, -1), None, 0.0
    for ep in range(cfg.epochs):
        t0 = time.perf_counter()
        row = train_one_epoch(model, train_loader, criterion, opt, sched, scaler, cfg, device, ema)
        t_train += time.perf_counter() - t0
        _, y, lg, vloss = evaluate(eval_model, val_loader, torch.nn.CrossEntropyLoss(), device)
        m, _ = _metrics(y, lg)
        history.append({"epoch": ep + 1, **row, "val_loss": vloss,
                        "val_macro_f1": m["macro_f1"], "val_top1": m["top1"]})
        if m["macro_f1"] > best[0]:  # hòa thì giữ epoch sớm hơn (so sánh nghiêm ngặt)
            best, best_state = (m["macro_f1"], ep + 1), copy.deepcopy(eval_model.state_dict())
        print(history[-1], flush=True)

    eval_model.load_state_dict(best_state)
    torch.save(best_state, rd / "best.pt")
    names, y, lg, _ = evaluate(eval_model, val_loader, torch.nn.CrossEntropyLoss(), device)
    np.save(rd / "val_logits.npy", lg)
    m, p = _metrics(y, lg)
    save_predictions(pred_path(cfg, "val"), names, y, p)

    if cfg.save_test_predictions:  # đúng MỘT lần, chỉ ở chung kết
        test_loader = mk(te, False)
        names, y, lg, _ = evaluate(eval_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        np.save(rd / "test_logits.npy", lg)
        save_predictions(pred_path(cfg, "test"), names, y, _metrics(y, lg)[1])

    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png",
                f"{cfg.exp_id} {cfg.backbone} seed{cfg.seed}")
    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "best_epoch": best[1],
               "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "train_s_per_epoch": t_train / cfg.epochs,
               "params_m": M.count_params(model), "gmacs": M.count_gmacs(model, cfg.img_size)}
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict ép kiểu theo field của Config."""
    types = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for s in pairs:
        if "=" not in s:
            raise ValueError(f"cần dạng KEY=VALUE, nhận '{s}'")
        k, v = s.split("=", 1)
        if k not in types:
            raise ValueError(f"'{k}' không có trong Config")
        t = types[k]
        if v.lower() == "none":
            if "None" not in t:
                raise ValueError(f"'{k}' không nhận None")
            out[k] = None
        elif t.startswith("bool"):
            out[k] = v.lower() in ("1", "true", "yes")
        elif t.startswith("int"):
            out[k] = int(v)
        elif t.startswith("float"):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()
