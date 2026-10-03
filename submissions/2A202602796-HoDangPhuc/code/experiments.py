"""experiments.py - lớp mỏng quanh train.run cho notebook: chạy có resume, kiểm tra pipeline,
chung kết (TTA + temperature scaling, test đúng một lần), gom kết quả ra results.xlsx."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import dataset as D
import inference as I
import model as M
import train as Tr
from eval import compute_metrics, save_predictions


def run_resume(cfg: Tr.Config) -> dict:
    """Bỏ qua nếu đã có summary.json (phiên Colab/Kaggle bị ngắt thì chạy lại ô là tiếp tục)."""
    f = Tr.run_dir(cfg) / "summary.json"
    if f.exists() and (not cfg.save_test_predictions or Tr.pred_path(cfg, "test").exists()):
        print("skip", cfg.exp_id, cfg.seed)
        return json.loads(f.read_text())
    return Tr.run(cfg)


# ---------------------------------------------------------------- kiểm tra pipeline (GUIDE 1.3)
def sanity(cfg: Tr.Config, n_overfit: int = 16, steps: int = 60) -> dict:
    """loss ban đầu ≈ ln9 và overfit một batch nhỏ (dùng TRAIN, không dùng test)."""
    Tr.set_seed(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, _, _ = D.load_split(cfg.labels_dir, cfg.fold)
    loader = D.make_loader(tr, cfg.images_dir, D.build_transforms(False, cfg.img_size), n_overfit, False,
                           num_workers=0)
    x, y, _ = next(iter(loader))
    x, y = x.to(dev), y.to(dev)
    model = M.build_model(cfg.backbone, True, D.NUM_CLASSES, 0.0, cfg.init).to(dev)
    ce = torch.nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        loss0 = ce(model(x), y).item()
    opt = torch.optim.AdamW(model.parameters(), 1e-3)
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        loss = ce(model(x), y)
        loss.backward()
        opt.step()
    out = {"loss_init": loss0, "ln9": float(np.log(9)), "loss_overfit": loss.item()}
    print(out)
    return out


# ---------------------------------------------------------------- suy luận từ checkpoint
def _loader(cfg, df, img_size=None):
    return D.make_loader(df, cfg.images_dir, D.build_transforms(False, img_size or cfg.img_size),
                         cfg.batch_size, False, None, cfg.num_workers)


def load_best(cfg: Tr.Config, device):
    m = M.build_model(cfg.backbone, False, D.NUM_CLASSES, 0.0, "finetune").to(device)
    m.load_state_dict(torch.load(Tr.run_dir(cfg) / "best.pt", map_location=device))
    return m.eval()


def view_logits(model, loader, device, views: dict):
    """views: {tên: hàm view}. Trả (filenames, y, {tên: logits[N,9]}); mỗi view một lượt forward."""
    out, names, y = {}, None, None
    for k, v in views.items():
        names, y, out[k] = I.predict_logits(model, loader, device, v)
    return names, y, out


def finalize(cfg: Tr.Config, views: dict | None = None, space: str = "prob") -> dict:
    """Chung kết cho MỘT seed. Chọn views/space bằng VAL trước khi gọi. T khớp trên VAL.
    Ghi <exp_id>_seed<k>_val.csv, _test.csv (đã hiệu chuẩn) và <exp_id>uncal_seed<k>_test.csv.
    Test được forward đúng MỘT lần ở đây (huấn luyện phải để save_test_predictions=False)."""
    views = views or {"id": I.view_identity}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best(cfg, dev)
    _, va, te = D.load_split(cfg.labels_dir, cfg.fold)
    nv, yv, lv = view_logits(model, _loader(cfg, va), dev, views)
    nt, yt, lt = view_logits(model, _loader(cfg, te), dev, views)
    pv_logits = np.log(np.clip(I.aggregate_views(list(lv.values()), space), 1e-12, None))  # logit giả để khớp T
    T = I.fit_temperature(pv_logits, yv)
    pv = I.apply_temperature(pv_logits, T)
    pt_logits = np.log(np.clip(I.aggregate_views(list(lt.values()), space), 1e-12, None))
    pt_uncal, pt = I.aggregate_views(list(lt.values()), space), I.apply_temperature(pt_logits, T)
    save_predictions(Tr.pred_path(cfg, "val"), nv, yv, pv)
    save_predictions(Tr.pred_path(cfg, "test"), nt, yt, pt)
    unc = Tr.Config(**{**vars(cfg), "exp_id": cfg.exp_id + "uncal"})
    save_predictions(Tr.pred_path(unc, "test"), nt, yt, pt_uncal)
    r = {"exp_id": cfg.exp_id, "seed": cfg.seed, "T": T,
         "val_macro_f1": compute_metrics(yv, pv.argmax(1), pv)["macro_f1"]}
    print(r)  # cố ý KHÔNG in số test: xem bằng eval.py ở Bước 4
    return r


# ---------------------------------------------------------------- gom kết quả
def collect_runs(out_dir: str, prefix: str) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(out_dir).glob(f"{prefix}*/seed*/summary.json")):
        s = json.loads(f.read_text())
        c = json.loads((f.parent / "config.json").read_text())
        rows.append({**s, **{k: c[k] for k in ("backbone", "weights_tag", "img_size", "epochs", "init", "aug",
                                               "mix", "loss", "sampler", "ema_decay", "lr_backbone",
                                               "lr_head", "batch_size", "torch", "timm", "gpu")}})
    return pd.DataFrame(rows)


def write_xlsx(sheets: dict, path: str) -> None:
    """sheets: {tên sheet: DataFrame}. Đóng băng hàng tiêu đề, làm tròn 4 chữ số."""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.round(4).to_excel(w, sheet_name=name, index=False)
            w.sheets[name].freeze_panes = "A2"
