"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0):
    """S1: đọc nguyên bản train/val/test_subset{fold}.csv."""
    d = Path(labels_dir)
    return tuple(pd.read_csv(d / f"{s}_subset{fold}.csv") for s in ("train", "val", "test"))


def check_split(train_df, val_df, test_df, images_dir, expect_total: int | None = TOTAL_IMAGES) -> dict:
    """Kiểm tra bắt buộc (README 2.1). Dừng bằng assert nếu vi phạm; trả về dict số liệu."""
    sets = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: len(v) for k, v in sets.items()}
    total = sum(n.values())
    per_class = {k: v["Label"].value_counts().sort_index().to_dict() for k, v in sets.items()}
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    overlap = {f"{a}&{b}": len(names[a] & names[b]) for a, b in
               (("train", "val"), ("train", "test"), ("val", "test"))}
    assert all(c == 0 for c in overlap.values()), f"giao khác rỗng: {overlap}"
    if expect_total is not None:
        assert total == expect_total, f"hợp ba tập = {total}, kỳ vọng {expect_total}"
        for k, share in (("train", .6), ("val", .2), ("test", .2)):
            assert abs(n[k] / total - share) < 0.01, f"tỉ lệ {k} lệch 60/20/20: {n}"
    missing = [f for v in sets.values() for f in v["Filename"] if not (Path(images_dir) / f).exists()]
    assert not missing, f"{len(missing)} file thiếu, ví dụ {missing[:3]}"
    out = {"n": n, "total": total, "per_class": per_class, "overlap": overlap, "missing": 0}
    print(out)
    return out


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """aug: basic | color | trivial | randaug. Chỉ lật ngang (lật dọc chưa được kiểm chứng là hợp lệ)."""
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:  # ảnh gốc 256x256 -> CenterCrop; img_size > 256 thì Resize (I04)
        pre = [T.Resize(img_size)] if img_size > 256 else [T.CenterCrop(img_size)]
        return T.Compose(pre + norm)
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment())
    elif aug != "basic":
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose(ops + norm)


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.files = df["Filename"].tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.dir, self.transform = Path(images_dir), transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = Image.open(self.dir / self.files[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i], self.files[i]


def _worker_init(worker_id: int):
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(df, images_dir, transform, batch_size: int, train: bool,
                sampler: str | None = None, num_workers: int = 2):
    ds = DeepWeedsDataset(df, images_dir, transform)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              worker_init_fn=_worker_init, persistent_workers=num_workers > 0)
    if not train:
        return DataLoader(ds, shuffle=False, **kw)
    if sampler == "balanced":
        cnt = np.bincount(ds.labels, minlength=NUM_CLASSES)
        w = (1.0 / cnt)[ds.labels]
        s = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(ds), replacement=True)
        return DataLoader(ds, sampler=s, drop_last=True, **kw)
    return DataLoader(ds, shuffle=True, drop_last=True, **kw)
