"""make_notebooks.py - sinh các notebook Kaggle (smoke / A / B) từ cùng một bộ ô chung.

    python make_notebooks.py          # ghi ../kaggle/{smoke,a,b}/<slug>.ipynb + kernel-metadata.json

A = Bước 0-2 (EDA, backbone, ablation), B = Bước 3-5 (suy luận, chung kết, xlsx). B đọc output của A
qua kernel_sources. BEST/COMBO của B được điền tay sau khi xem kết quả val của A.
"""
import json
import sys
from pathlib import Path

USER = "ashuraotsuki"
REPO_URL = "https://github.com/hdphuc127/K4-Track4-Day2-Deeplearning-Advance.git"
SUB = "submissions/2A202602796-HoDangPhuc/code"
OUT = Path(__file__).resolve().parent.parent / "kaggle"


def nb(cells):
    C = []
    for kind, s in cells:
        src = s.strip("\n").splitlines(True)
        C.append({"cell_type": kind, "metadata": {}, "source": src,
                  **({"execution_count": None, "outputs": []} if kind == "code" else {})})
    return {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = ("code", f'''import os, sys, platform
REPO_URL = "{REPO_URL}"
SUB = "{SUB}"
ROOT = "/kaggle/working"
os.chdir(ROOT)
if not os.path.exists("lab"):
    !git clone -q {{REPO_URL}} lab
else:
    !git -C lab pull -q
CODE = f"{{ROOT}}/lab/{{SUB}}"
sys.path.insert(0, CODE)
!pip -q install timm openpyxl
import torch, timm
print("python", platform.python_version(), "| torch", torch.__version__, "| timm", timm.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ GPU")
!nvidia-smi --query-gpu=name,memory.total --format=csv''')

DATA = ("code", '''import glob, hashlib
os.makedirs("data/labels", exist_ok=True)
# Ảnh: dataset Kaggle ashuraotsuki/deepweeds-fold0 (images.zip nguyên bản Zenodo, MD5 b7b30f96...). Có thể đã được giải nén sẵn.
zips = glob.glob("/kaggle/input/**/images.zip", recursive=True)
if zips:
    h = hashlib.md5()
    with open(zips[0], "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    assert h.hexdigest() == "b7b30f96d466fba86016aa5a26606e0f", h.hexdigest()
    os.makedirs("data/images", exist_ok=True)
    !unzip -q -n {zips[0]} -d data/images     # zip phẳng: 17.509 file .jpg ở gốc
    IMAGES_DIR = f"{ROOT}/data/images"
else:
    jp = glob.glob("/kaggle/input/**/*.jpg", recursive=True)
    assert jp, "không thấy ảnh trong /kaggle/input: gắn dataset ashuraotsuki/deepweeds-fold0"
    IMAGES_DIR = os.path.dirname(jp[0])
print("IMAGES_DIR", IMAGES_DIR, len(os.listdir(IMAGES_DIR)))
B_ = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"   # nhãn/fold nguyên bản của tác giả
for n in ["labels", "train_subset0", "val_subset0", "test_subset0"]:
    !wget -q -O data/labels/{n}.csv {B_}/{n}.csv
!wc -l data/labels/*.csv''')

PATHS = ("code", '''LABELS_DIR = f"{ROOT}/data/labels"
OUT = f"{ROOT}/out"
os.makedirs(OUT, exist_ok=True)
import experiments as E, train as Tr, dataset as D
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PATHS = dict(images_dir=IMAGES_DIR, labels_dir=LABELS_DIR, out_dir=f"{OUT}/runs",
             pred_dir=f"{OUT}/predictions", curves_dir=f"{OUT}/curves", num_workers=4)
EPOCHS = 12
def cfg(**kw): return Tr.Config(**{**PATHS, "epochs": EPOCHS, **kw})''')

EDA = ("code", '''tr, va, te = D.load_split(LABELS_DIR)
info = D.check_split(tr, va, te, IMAGES_DIR)
cnt = pd.DataFrame({k: v["Label"].value_counts().sort_index() for k, v in zip(["train", "val", "test"], [tr, va, te])})
cnt.index = D.CLASS_NAMES; cnt["total"] = cnt.sum(1); print(cnt)
os.makedirs(f"{OUT}/eda", exist_ok=True); cnt.to_csv(f"{OUT}/eda/class_counts.csv")
ax = cnt[["train", "val", "test"]].plot.bar(figsize=(9, 3.5), title="Số ảnh theo lớp (fold 0)"); plt.tight_layout()
plt.savefig(f"{OUT}/eda/class_dist.png", dpi=130); plt.close()
print("tỉ lệ lớp lớn/nhỏ:", cnt.total.max() / cnt.total.min())
from PIL import Image
fig, ax = plt.subplots(9, 3, figsize=(7, 20))
for c in range(9):
    for j, f in enumerate(tr[tr.Label == c].Filename.head(3)):
        ax[c, j].imshow(Image.open(f"{IMAGES_DIR}/{f}")); ax[c, j].axis("off")
    ax[c, 0].set_title(D.CLASS_NAMES[c], fontsize=8, loc="left")
plt.tight_layout(); plt.savefig(f"{OUT}/eda/samples.png", dpi=90); plt.close()''')

SANITY = ("code", '''import json, torch
S = E.sanity(cfg(backbone="resnet50"))
json.dump(S, open(f"{OUT}/eda/sanity.json", "w"))
assert abs(S["loss_init"] - S["ln9"]) < 0.8 and S["loss_overfit"] < 0.2, S
ds = D.DeepWeedsDataset(tr.sample(8, random_state=0), IMAGES_DIR, D.build_transforms(True, 224, "color"))
m, s = torch.tensor(D.IMAGENET_MEAN)[:, None, None], torch.tensor(D.IMAGENET_STD)[:, None, None]
fig, ax = plt.subplots(1, 8, figsize=(16, 2.5))
for a, (x, y, _) in zip(ax, ds): a.imshow((x * s + m).clamp(0, 1).permute(1, 2, 0)); a.set_title(D.CLASS_NAMES[y], fontsize=7); a.axis("off")
plt.savefig(f"{OUT}/eda/aug_check.png", dpi=110); plt.close()''')

STEP1 = ("code", '''BACKBONES = ["resnet50", "resnext50", "convnext_tiny", "deit_small", "swin_tiny", "efficientnet_b0", "mobilenetv3"]
for i, bb in enumerate(BACKBONES, 1):
    try:
        E.run_resume(cfg(exp_id=f"B{i:02d}_{bb}", backbone=bb, seed=0))
    except Exception as e:   # vd. hết VRAM: ghi lại, chạy tiếp
        print("FAILED", bb, repr(e)); torch.cuda.empty_cache()
        open(f"{OUT}/failed.txt", "a").write(f"{bb}: {e!r}\\n")
B = E.collect_runs(PATHS["out_dir"], "B"); B.to_csv(f"{OUT}/backbones.csv", index=False); print(B[["exp_id", "val_macro_f1", "val_top1", "train_s_per_epoch", "params_m", "gmacs"]])''')

STEP2 = ("code", '''BEST = "resnet50"   # backbone chuẩn của ablation (chốt từ trước để chạy tự động; lý do ở báo cáo)
T = {
 "T00": ("nền", {}),
 "T01": ("A khởi tạo", dict(init="frozen")),
 "T02": ("A khởi tạo", dict(init="scratch")),
 "T03": ("B augmentation", dict(aug="color")),
 "T04": ("B augmentation", dict(aug="trivial")),
 "T05": ("B augmentation", dict(mix="cutmix")),
 "T06": ("B augmentation", dict(mix="mixup")),
 "T07": ("C loss", dict(loss="ls", label_smoothing=0.1)),
 "T08": ("C loss", dict(loss="focal", focal_gamma=2.0)),
 "T09": ("C loss", dict(loss="ce_weighted", class_weight_beta=0.0)),
 "T10": ("D sampler", dict(sampler="balanced")),
 "T11": ("F EMA", dict(ema_decay=0.995)),
 "T12": ("E LR", dict(lr_head=1e-4)),
}
for k, (ax_, o) in T.items():
    try:
        E.run_resume(cfg(exp_id=k, backbone=BEST, seed=0, **o))
    except Exception as e:
        print("FAILED", k, repr(e)); torch.cuda.empty_cache(); open(f"{OUT}/failed.txt", "a").write(f"{k}: {e!r}\\n")
Tdf = E.collect_runs(PATHS["out_dir"], "T")
Tdf["delta_vs_T00"] = Tdf.val_macro_f1 - Tdf.loc[Tdf.exp_id == "T00", "val_macro_f1"].iloc[0]
Tdf["axis"] = Tdf.exp_id.map(lambda k: T[k][0]); Tdf.to_csv(f"{OUT}/training.csv", index=False)
print(Tdf[["exp_id", "axis", "val_macro_f1", "val_top1", "delta_vs_T00", "train_s_per_epoch"]])''')

STEP_C = ("code", '''BEST = "convnext_tiny"   # chọn sau Bước 1: macro-F1 val 0.971 so với ResNet-50 0.812 (xem báo cáo)
T = {  # exp_id: (trục, override so với C00)
 "C00": ("nền", {}),
 "C01": ("A khởi tạo", dict(init="frozen")),
 "C02": ("A khởi tạo", dict(init="scratch")),
 "C03": ("B augmentation", dict(aug="color")),
 "C04": ("B augmentation", dict(aug="trivial")),
 "C05": ("B augmentation", dict(mix="cutmix")),
 "C06": ("B augmentation", dict(mix="mixup")),
 "C07": ("C loss", dict(loss="ls", label_smoothing=0.1)),
 "C08": ("C loss", dict(loss="focal", focal_gamma=2.0)),
 "C09": ("C loss", dict(loss="ce_weighted", class_weight_beta=0.0)),
 "C10": ("D sampler", dict(sampler="balanced")),
 "C11": ("F EMA", dict(ema_decay=0.995)),
 "C12": ("E LR", dict(lr_backbone=2e-4, lr_head=2e-3)),
}
for k, (ax_, o) in T.items():
    try:
        E.run_resume(cfg(exp_id=k, backbone=BEST, seed=0, **o))
    except Exception as e:
        print("FAILED", k, repr(e)); torch.cuda.empty_cache(); open(f"{OUT}/failed.txt", "a").write(f"{k}: {e!r}\\n")
# chạy lại T09 của pilot ResNet-50 (lỗi dtype ở phase A đã sửa)
E.run_resume(cfg(exp_id="T09", backbone="resnet50", seed=0, loss="ce_weighted", class_weight_beta=0.0))
Cdf = E.collect_runs(PATHS["out_dir"], "C")
Cdf["delta_vs_C00"] = Cdf.val_macro_f1 - Cdf.loc[Cdf.exp_id == "C00", "val_macro_f1"].iloc[0]
Cdf["axis"] = Cdf.exp_id.map(lambda k: T[k][0]); Cdf.to_csv(f"{OUT}/training_c.csv", index=False)
print(Cdf[["exp_id", "axis", "val_macro_f1", "val_top1", "delta_vs_C00", "train_s_per_epoch"]])''')

CLEAN = ("code", '''# bỏ checkpoint để output nhẹ (giữ logits, history, predictions, curves)
!find {OUT}/runs -name "best.pt" -delete
!du -sh {OUT}; ls {OUT}''')

COPY_A = ("code", '''import glob, shutil
cands = sorted(set(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/runs", recursive=True)))
print("output các phase trước:", cands)
assert cands, "không thấy output phase A/C trong /kaggle/input"
for src_out in cands:
    for sub in ("runs", "predictions", "curves", "eda"):
        if os.path.isdir(f"{src_out}/{sub}"):
            shutil.copytree(f"{src_out}/{sub}", f"{OUT}/{sub}", dirs_exist_ok=True)
    for f in ("backbones.csv", "training.csv", "training_c.csv", "failed.txt"):
        if os.path.exists(f"{src_out}/{f}"): shutil.copy(f"{src_out}/{f}", f"{OUT}/{f}")
B = pd.read_csv(f"{OUT}/backbones.csv"); print(B.shape)''')


def smoke():
    return nb([("markdown", "# Smoke test Kaggle: GPU, tải dữ liệu, 1 epoch, finalize"), SETUP, DATA, PATHS,
               ("code", 'EPOCHS = 1'), EDA, SANITY,
               ("code", '''import time, inference as I, benchmark as BM
c = cfg(exp_id="S00", backbone="resnet50", seed=0)
t = time.time(); s = E.run_resume(c); print("1 epoch resnet50:", s, "tổng", time.time() - t)
print(E.finalize(c, {"id": I.view_identity, "flip": I.view_hflip}))
m = E.load_best(c, torch.device("cuda")); print(BM.latency_report(m, 1, 224, "fp32"))
c2 = cfg(exp_id="S01", backbone="mobilenetv3", seed=0); print(E.run_resume(c2))'''),
               CLEAN])


def phase_a():
    return nb([("markdown", "# DeepWeeds — Phase A: Bước 0-2 (EDA, backbone, ablation)"), SETUP, DATA, PATHS,
               EDA, SANITY, STEP1, STEP2, CLEAN])


def phase_c():
    return nb([("markdown", "# DeepWeeds — Phase C: ablation chính trên ConvNeXt-T (C00-C12) + T09 pilot"), SETUP, DATA, PATHS,
               STEP_C, CLEAN])


def phase_b(best="convnext_tiny"):
    return nb([("markdown", "# DeepWeeds — Phase B: Bước 3-5 (suy luận, chung kết, xlsx)"), SETUP, DATA, PATHS, COPY_A,
               ("code", f'''import numpy as np, inference as I, benchmark as BM
from eval import compute_metrics
tr, va, te = D.load_split(LABELS_DIR)
BEST = "{best}"
# Tổ hợp ứng viên = các yếu tố có Δ val > 0 ở ablation C (TrivialAugment C04, CutMix C05, CE có trọng số C09); EMA (Δ≈0) bỏ.
COMBO_CAND = dict(aug="trivial", mix="cutmix", loss="ce_weighted", class_weight_beta=0.0)
E.run_resume(cfg(exp_id="C13", backbone=BEST, seed=0, **COMBO_CAND))
Pil = E.collect_runs(PATHS["out_dir"], "T")   # pilot ResNet-50 (T00-T12)
Tdf = E.collect_runs(PATHS["out_dir"], "C")   # ablation chính ConvNeXt-T (C00-C13)
f = Tdf.set_index("exp_id").val_macro_f1
USE_COMBO = bool(f["C13"] > f["C00"])         # quyết định CHỈ bằng val (seed 0)
COMBO = COMBO_CAND if USE_COMBO else {{}}
REF_ID = "C13" if USE_COMBO else "C14"        # C14 = công thức nền huấn luyện lại để có checkpoint
if not USE_COMBO:
    E.run_resume(cfg(exp_id="C14", backbone=BEST, seed=0))
    Tdf = E.collect_runs(PATHS["out_dir"], "C")
print("USE_COMBO", USE_COMBO, "| C00", f["C00"], "| C13", f["C13"])
SPACE = "prob"
Tdf["delta_vs_C00"] = Tdf.val_macro_f1 - Tdf.loc[Tdf.exp_id == "C00", "val_macro_f1"].iloc[0]'''),
               ("code", '''REF = cfg(exp_id=REF_ID, backbone=BEST, seed=0, **COMBO)
dev = torch.device("cuda"); model = E.load_best(REF, dev)
va_loader = E._loader(REF, va)
V = {"I00 1-view": {"id": I.view_identity},
     "I01 flip": {"id": I.view_identity, "flip": I.view_hflip},
     "I02 5crop192": {"c": lambda x: I.views_multicrop(x, 192)},
     "I02b 5crop+flip192": {"c": lambda x: I.views_multicrop(x, 192, True)}}
rows = []
for name, views in V.items():
    for space in ("prob", "logit"):
        _, y, lg = E.view_logits(model, va_loader, dev, views)
        p = I.aggregate_views(list(lg.values()), space)
        m_ = compute_metrics(y, p.argmax(1), p)
        rows.append({"exp_id": name.split()[0], "method": name, "space": space, "K": len(views),
                     "macro_f1_val": m_["macro_f1"], "top1_val": m_["top1"], "ece_val": m_["ece"]})
for s_ in (224, 256, 288, 320):
    _, y, lg = E.view_logits(model, E._loader(REF, va, s_), dev, {"id": I.view_identity})
    p = I._softmax(lg["id"]); m_ = compute_metrics(y, p.argmax(1), p)
    rows.append({"exp_id": "I04", "method": f"res{s_}", "space": "prob", "K": 1, "macro_f1_val": m_["macro_f1"],
                 "top1_val": m_["top1"], "ece_val": m_["ece"]})
_, y, lg = E.view_logits(model, va_loader, dev, {"id": I.view_identity})
z = lg["id"]; Tt = I.fit_temperature(z, y)
for nm, p in (("trước", I._softmax(z)), ("sau", I.apply_temperature(z, Tt))):
    m_ = compute_metrics(y, p.argmax(1), p)
    rows.append({"exp_id": "I07", "method": f"temperature {nm} (T={Tt:.3f})", "space": "prob", "K": 1,
                 "macro_f1_val": m_["macro_f1"], "top1_val": m_["top1"], "ece_val": m_["ece"]})
Inf = pd.DataFrame(rows); Inf.to_csv(f"{OUT}/inference.csv", index=False); print(Inf)
def _f(i, sp="prob"): return float(Inf[(Inf.exp_id == i) & (Inf.space == sp)].macro_f1_val.iloc[0])
FLIP_OK = _f("I01") > _f("I00")                               # chọn trên VAL
FINAL_VIEWS = {"id": I.view_identity, "flip": I.view_hflip} if FLIP_OK else {"id": I.view_identity}
r4 = Inf[Inf.exp_id == "I04"].assign(res=lambda d: d.method.str.replace("res", "").astype(int))
RES = int(r4[r4.macro_f1_val >= r4.macro_f1_val.max() - 0.001].res.min())   # độ phân giải nhỏ nhất trong 0.001 của tốt nhất (VAL)
print("RES", RES)
print("FLIP_OK", FLIP_OK, list(FINAL_VIEWS))'''),
               ("code", '''# Độ trễ đúng cách: warmup 10, synchronize, 100 lần; batch 1 và 32; fp32/amp/fp16; gộp BN
lat = []
for b in (1, 32):
    for dt in ("fp32", "amp", "fp16"):
        lat.append({**BM.latency_report(model, b, 224, dt), "fuse_bn": False})
fused = I.fuse_conv_bn(model)
for b in (1, 32):
    lat.append({**BM.latency_report(fused, b, 224, "fp32"), "fuse_bn": True})
lat.append({**BM.tta_latency(model, 2, batch_size=1, img_size=224, dtype="fp32"), "gpu": torch.cuda.get_device_name()})
Lat = pd.DataFrame(lat); Lat.to_csv(f"{OUT}/latency.csv", index=False); print(Lat)
LAT95 = float(Lat[(Lat.batch == 1) & (Lat["dtype"] == "fp32") & (Lat.fuse_bn == False)].p95.iloc[0])   # p95 batch-1 fp32, ms
print("LAT95", LAT95)'''),
               ("code", '''# Chung kết: huấn luyện KHÔNG ghi test; finalize() chạy test đúng một lần/seed; mốc T00 ghi test khi huấn luyện
for seed in (0, 1, 2):
    c = cfg(exp_id="F01", backbone=BEST, seed=seed, **COMBO)
    E.run_resume(c); E.finalize(c, FINAL_VIEWS, SPACE, RES)
    E.run_resume(cfg(exp_id="C00", backbone=BEST, seed=seed, save_test_predictions=True))'''),
               ("code", '''P = f"{OUT}/predictions"; EV = f"{CODE}/eval.py"; LB = LABELS_DIR
!python {EV} score --pred "{P}/F01_seed*_test.csv" --test-csv {LB}/test_subset0.csv --labels {LB}/labels.csv --tag F01 --out {OUT}/eval_out
!python {EV} score --pred "{P}/C00_seed*_test.csv" --test-csv {LB}/test_subset0.csv --labels {LB}/labels.csv --tag C00 --out {OUT}/eval_out
!python {EV} grade --final "{P}/F01_seed*_test.csv" --baseline "{P}/C00_seed*_test.csv" --uncal "{P}/F01uncal_seed*_test.csv" --final-val "{P}/F01_seed*_val.csv" --test-csv {LB}/test_subset0.csv --val-csv {LB}/val_subset0.csv --labels {LB}/labels.csv --latency-p95-ms {LAT95} --out {OUT}/eval_out | tee {OUT}/grade.txt'''),
               ("code", '''Final = E.collect_runs(PATHS["out_dir"], "F01")
Summ = pd.concat([B, Tdf, Pil]).sort_values("val_macro_f1", ascending=False).head(10)
E.write_xlsx({"Backbones": B, "Training": Tdf, "Training_pilot_resnet50": Pil, "Inference": Inf, "Final": Final, "Latency": Lat, "Summary": Summ},
             f"{OUT}/results.xlsx")
!find {OUT}/runs -name "best.pt" -delete
!du -sh {OUT}; ls {OUT}; ls {OUT}/eval_out''')])


def phase_lat():
    return nb([("markdown", "# DeepWeeds — Độ trễ bổ sung: 7 backbone (batch 1) và các phương pháp suy luận, T4"), SETUP,
               ("code", '''import benchmark as BM, model as M, pandas as pd
OUT = f"{ROOT}/out"; os.makedirs(OUT, exist_ok=True)
rows = []
for name in ["resnet50", "resnext50", "convnext_tiny", "deit_small", "swin_tiny", "efficientnet_b0", "mobilenetv3"]:
    m = M.build_model(name, pretrained=False)   # độ trễ không phụ thuộc giá trị trọng số
    for dt in ["fp32", "fp16"]:
        r = BM.latency_report(m, 1, 224, dt)
        rows.append({"nhóm": "backbone", "config": name, **r}); print(name, dt, round(r["p50"], 2), round(r["p95"], 2))
pd.DataFrame(rows).to_csv(f"{OUT}/latency_backbones.csv", index=False)'''),
               ("code", '''m = M.build_model("convnext_tiny", pretrained=False)
rows = []
def add(cfgname, k, size, bs=1):
    r = BM.latency_report(m, bs * k, size, "fp32")      # K view = batch K (cách triển khai thực tế)
    rows.append({"nhóm": "inference", "config": cfgname, "K": k, "img_size": size, **r}); print(cfgname, round(r["p50"], 2), round(r["p95"], 2))
add("I00 1-view 224", 1, 224)
add("I01 flip (K=2) 224", 2, 224)
add("I02b 5crop+flip 192 (K=10)", 10, 192)
add("I04 res256", 1, 256)
add("I04 res288 (F01)", 1, 288)
add("I04 res320", 1, 320)
for sz in (224, 288):
    for dt in ("fp16",):
        r = BM.latency_report(m, 1, sz, dt); rows.append({"nhóm": "inference", "config": f"1-view {sz} {dt}", "K": 1, "img_size": sz, **r})
pd.DataFrame(rows).to_csv(f"{OUT}/latency_inference.csv", index=False)'''),
               ("code", "!ls /kaggle/working/out")])


def meta(slug, title, notebook, gpu=True, sources=()):
    return {"id": f"{USER}/{slug}", "title": title, "code_file": notebook, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USER}/deepweeds-fold0"], "competition_sources": [], "model_sources": [],
            "kernel_sources": list(sources)}


def write(name, slug, title, notebook_obj, sources=()):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    nbf = f"{slug}.ipynb"
    (d / nbf).write_text(json.dumps(notebook_obj, ensure_ascii=False, indent=1), encoding="utf8")
    (d / "kernel-metadata.json").write_text(json.dumps(meta(slug, title, nbf, sources=sources), indent=2))


if __name__ == "__main__":
    best, combo, views = (sys.argv[1:4] + [None] * 3)[:3] if len(sys.argv) > 3 else (None, None, None)
    write("smoke", "deepweeds-smoke", "deepweeds-smoke", smoke())
    write("a", "deepweeds-a", "deepweeds-a", phase_a())
    write("c", "deepweeds-c", "deepweeds-c", phase_c())
    kw = {k: v for k, v in dict(best=best, combo=combo, views=views).items() if v}
    write("b", "deepweeds-b", "deepweeds-b", phase_b(**kw), sources=[f"{USER}/deepweeds-a", f"{USER}/deepweeds-c"])
    write("lat", "deepweeds-lat", "deepweeds-lat", phase_lat())
    print("ok", OUT)
