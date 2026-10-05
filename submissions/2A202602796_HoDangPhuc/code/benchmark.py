"""benchmark.py - đo độ trễ suy luận đúng cách: warmup, đồng bộ GPU, >= 50 lần đo, p50/p95/p99."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    p50, p95, p99 = np.percentile(ts, [50, 95, 99])
    return {"p50": p50, "p95": p95, "p99": p99, "mean": float(np.mean(ts)), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Độ trễ forward (không gồm tiền xử lý) với đầu vào ngẫu nhiên. dtype: fp32 | amp | fp16."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("không có CUDA")
    model = copy.deepcopy(model).to(device).eval()  # bản sao: .half() không được đổi model của người gọi
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        model, x = model.half(), x.half()
    sync = torch.cuda.synchronize if device == "cuda" else None

    def fn():
        with torch.inference_mode(), torch.autocast(device, enabled=dtype == "amp"):
            model(x)

    r = bench(fn, warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu", "dtype": dtype,
            "batch": batch_size, "img_size": img_size, "p50": r["p50"], "p95": r["p95"],
            "p99": r["p99"], "images_per_s": batch_size / (r["p50"] / 1000), "torch": torch.__version__}


def tta_latency(model, k_views: int, **kw) -> dict:
    """Đo thật K view bằng batch K*B (cách thực tế) và so với K * p50 của 1 view."""
    one = latency_report(model, **kw)
    kb = latency_report(model, **{**kw, "batch_size": kw["batch_size"] * k_views})
    return {"k": k_views, "p50_1view": one["p50"], "p50_batched": kb["p50"],
            "p50_k_times_1view": k_views * one["p50"], "p95_batched": kb["p95"], "p99_batched": kb["p99"]}
