"""Smoke test CPU với dữ liệu giả: python test_smoke.py"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

import benchmark as B
import inference as I
import losses as L
import model as M
import train as Tr
from dataset import NUM_CLASSES


def make_data(root, n=180):
    img, lab = root / "images", root / "labels"
    img.mkdir(); lab.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)).save(img / f"{i}.jpg")
        rows.append((f"{i}.jpg", i % NUM_CLASSES, "x"))
    df = pd.DataFrame(rows, columns=["Filename", "Label", "Species"])
    for s, sl in (("train", slice(0, 108)), ("val", slice(108, 144)), ("test", slice(144, 180))):
        df.iloc[sl].to_csv(lab / f"{s}_subset0.csv", index=False)
    return img, lab


def test_losses():
    z, y = torch.randn(16, 9), torch.randint(0, 9, (16,))
    ce = torch.nn.functional.cross_entropy(z, y)
    assert abs(L.FocalLoss(0.0)(z, y) - ce) < 1e-6
    assert abs(L.LabelSmoothingCE(0.0)(z, y) - ce) < 1e-6
    w = L.class_weights([9106, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])
    assert abs(w.sum() - 9) < 1e-4 and w[0] < w[1]
    x, (ya, yb, lam) = L.mix_batch(torch.randn(8, 3, 32, 32), torch.arange(8), mode="cutmix")
    assert 0 <= lam <= 1


def test_inference():
    z = np.random.randn(500, 9) * 3
    y = np.random.randint(0, 9, 500)
    T = I.fit_temperature(z, y)
    assert T > 0
    assert np.allclose(I.apply_temperature(z, T).sum(1), 1)
    assert np.allclose(I.aggregate_views([z, z], "prob"), I.aggregate_views([z, z], "logit"))


def test_fuse_bn():
    m = M.build_model("resnet18", pretrained=False, init="scratch").eval()
    for mod in m.modules():  # BN ngẫu nhiên để kiểm tra có ý nghĩa
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(); mod.running_var.uniform_(0.5, 2)
    f = I.fuse_conv_bn(m)
    x = torch.randn(2, 3, 64, 64)
    err = (m(x) - f(x)).abs().max().item()
    assert err < 1e-4, err
    assert not any(isinstance(k, torch.nn.BatchNorm2d) for k in f.modules())


def test_frozen_and_groups():
    m = M.build_model("resnet18", pretrained=False, init="scratch")
    M.freeze_backbone(m)
    assert len(M.param_groups(m, 1e-4, 1e-3, .05)) == 1
    M.set_train_mode(m, True)
    assert not m.bn1.training and m.fc.training


def test_benchmark():
    r = B.latency_report(M.build_model("resnet18", False, init="scratch"), 1, 64, device="cpu", warmup=2, iters=5)
    assert r["p50"] <= r["p99"]


def test_run_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img, lab = make_data(d)
        cfg = Tr.Config(exp_id="S00", backbone="resnet18", init="scratch", epochs=2, batch_size=16,
                        img_size=64, images_dir=str(img), labels_dir=str(lab), out_dir=str(d / "runs"),
                        pred_dir=str(d / "pred"), curves_dir=str(d / "curves"), num_workers=0,
                        mix="cutmix", ema_decay=0.99, save_test_predictions=True)
        # check_split kỳ vọng 17.509 ảnh nên dữ liệu giả phải bỏ kiểm tra tổng:
        import dataset as D
        orig = D.check_split
        D.check_split = lambda *a, **k: orig(*a[:4], None)
        try:
            s = Tr.run(cfg)
        finally:
            D.check_split = orig
        assert (d / "pred" / "S00_seed0_test.csv").exists() and (d / "curves" / "S00_seed0.png").exists()
        from eval import read_pred
        read_pred(str(d / "pred" / "S00_seed0_val.csv"))
        print(s)


def test_experiments():
    import dataset as D, experiments as E
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        img, lab = make_data(d)
        orig = D.check_split
        D.check_split = lambda *a, **k: orig(*a[:4], None)
        try:
            c = Tr.Config(exp_id="F00", backbone="resnet18", init="scratch", epochs=1, batch_size=16, img_size=64,
                          images_dir=str(img), labels_dir=str(lab), out_dir=str(d / "r"), pred_dir=str(d / "p"),
                          curves_dir=str(d / "c"), num_workers=0)
            E.run_resume(c); E.run_resume(c)  # lần 2 phải skip
            r = E.finalize(c, {"id": I.view_identity, "flip": I.view_hflip})
            assert r["T"] > 0
            from eval import read_pred
            for f in ("F00_seed0_val", "F00_seed0_test", "F00uncal_seed0_test"):
                read_pred(str(d / "p" / f"{f}.csv"))
            assert len(E.collect_runs(str(d / "r"), "F")) == 1
            E.write_xlsx({"a": E.collect_runs(str(d / "r"), "F")}, str(d / "x.xlsx"))
        finally:
            D.check_split = orig


def test_weighted_ce_half():
    w = L.class_weights([9106] + [1000] * 8)
    z = torch.randn(4, 9).half()
    L.build_criterion("ce_weighted", weight=w)(z.float(), torch.tensor([0, 1, 2, 3]))  # không được lỗi dtype


def test_overrides():
    assert Tr.parse_overrides(["seed=3", "ema_decay=none", "amp=false", "lr_head=1e-2"]) == \
        {"seed": 3, "ema_decay": None, "amp": False, "lr_head": 1e-2}


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
