# Bài nộp Lab Day 2 (DeepWeeds)

**Trạng thái:** code hoàn chỉnh, đã qua `test_smoke.py` (CPU, dữ liệu giả). CHƯA có thí nghiệm thật, nên chưa có `results.xlsx`, `report.md`, `curves/`, `predictions/`.

## Chạy trên Kaggle / Colab
1. Đẩy repo lên GitHub, sửa `REPO_URL` ở ô đầu của [code/lab_day2.ipynb](code/lab_day2.ipynb). Link notebook chạy lại: _TODO sau khi upload_.
2. Kaggle: bật **Internet** và **GPU**; Colab: bật GPU. Chạy các ô từ trên xuống (mỗi ô có resume, phiên ngắt thì chạy lại ô).
3. Kết quả ghi vào `/kaggle/working/out` (Kaggle) hoặc Drive (Colab): `runs/`, `predictions/`, `curves/`, `results.xlsx`.
4. Chép `predictions/`, `curves/`, `results.xlsx` về thư mục này, viết `report.md`, rồi commit (không commit `runs/*.pt`).

Thứ tự: Bước 0 (EDA, kiểm tra pipeline) → 1 (7 backbone) → 2 (T00–T13) → 3 (suy luận, độ trễ) → 4 (F01 ≥3 seed + mốc T00) → 5 (xlsx).

## Môi trường
Python 3.10+, torch ≥ 2.3 (bản CUDA), torchvision, timm, pandas, scikit-learn, matplotlib, openpyxl. Phiên bản thực tế được ghi vào `runs/<exp_id>/seed<k>/config.json` (kèm tag trọng số timm và GPU).
Seed: 0, 1, 2 cho chung kết; 0 cho các bước sàng.

## Kiểm tra cục bộ
`cd code && python test_smoke.py` (Windows/CPU nếu lỗi OpenMP: `set KMP_DUPLICATE_LIB_OK=TRUE`).
`code/eval.py` là bản gốc, không sửa.
