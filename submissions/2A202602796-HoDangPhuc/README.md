# Bài nộp Lab Day 2 (DeepWeeds)

**Trạng thái:** hoàn tất. Kết quả test cuối (F01, 3 seed): macro-F1 **0,9774 ± 0,0006**, top-1 **0,9812 ± 0,0005**; mốc C00: 0,9692 ± 0,0024 (xem [report.md](report.md)).

## Nội dung thư mục
| Mục | File |
|---|---|
| Báo cáo | [report.md](report.md) |
| Bảng so sánh | [results.xlsx](results.xlsx) (Backbones, Training, Training_pilot_resnet50, Inference, Final, PerClass, Latency, Summary) |
| Biểu đồ training | [curves/](curves) (mỗi exp_id một ảnh), [eda/](eda), [figures/](figures) (ma trận nhầm lẫn, ảnh sai) |
| Dự đoán | [predictions/](predictions) (F01, F01uncal, C00: test và val, seed 0–2) |
| Số tính từ `eval.py` | [eval_out/](eval_out), [logs/grade.txt](logs/grade.txt) |
| Log từng lần chạy | [logs/runs/](logs/runs) (config.json, summary.json, history.csv) |
| Code | [code/](code), notebook sinh bởi `code/make_notebooks.py` → [kaggle/](kaggle) |

Tính lại chỉ số: `python code/eval.py score --pred "predictions/F01_seed*_test.csv" --test-csv <test_subset0.csv> --labels <labels.csv> --tag F01`.

## Notebook chạy lại được (Kaggle, GPU T4, Internet bật)
| Phase | Nội dung | Notebook |
|---|---|---|
| A | EDA, kiểm tra pipeline, 7 backbone (B01–B07), pilot ResNet-50 (T00–T12) | https://www.kaggle.com/code/ashuraotsuki/deepweeds-a |
| C | ablation chính ConvNeXt-T (C00–C12) + T09 chạy lại | https://www.kaggle.com/code/ashuraotsuki/deepweeds-c |
| B | tổ hợp C13, suy luận, độ trễ, chung kết F01 + mốc C00 (3 seed), `results.xlsx` | https://www.kaggle.com/code/ashuraotsuki/deepweeds-b |

Notebook được sinh bởi [code/make_notebooks.py](code/make_notebooks.py) (thư mục [kaggle/](kaggle/)); code lấy từ GitHub `hdphuc127/K4-Track4-Day2-Deeplearning-Advance`. Ảnh đọc từ Kaggle Dataset `ashuraotsuki/deepweeds-fold0` (`images.zip` nguyên bản Zenodo, MD5 `b7b30f96d466fba86016aa5a26606e0f`); nhãn và fold 0 tải từ GitHub của tác giả DeepWeeds.

**Thứ tự chạy:** A và C độc lập (chạy trước), B đọc output của A và C qua `kernel_sources`. Mỗi lần chạy có resume (`experiments.run_resume`).

## Môi trường (phiên bản thực tế ghi trong `runs/*/seed*/config.json`)
Python 3.13, torch 2.11.0+cu128, torchvision 0.26.0, timm 1.0.29, Tesla T4. Thư viện: pandas, scikit-learn, matplotlib, openpyxl.

## Seed
Seed 0 cho các bước sàng (B, T, C, I); seed 0, 1, 2 cho chung kết F01 và mốc C00.

## Quy ước exp_id
`B01–B07` backbone · `T00–T12` pilot ResNet-50 · `C00–C13` ablation ConvNeXt-T (mốc = C00) · `I00–I07` suy luận · `F01` chung kết. Biểu đồ: `curves/<exp_id>_seed<k>.png`.

## Kiểm tra cục bộ
`cd code && python test_smoke.py` (CPU, dữ liệu giả). `code/eval.py` là bản gốc, không sửa.
