# EDR-CNN10: Kế Hoạch Thực Hiện

> Tài liệu này ghi lại **các bước cụ thể** để triển khai EDR-CNN10 từ đầu đến cuối.
> Đánh dấu `[x]` khi hoàn thành từng bước.

---

## GIAI ĐOẠN 1 — Tạo Module `edrcnn10`

### [x] Bước 1.1 — Tạo `network.py`
- Tạo file `ldctbench/methods/edrcnn10/network.py`
- Copy class `FixedSobelLayer` và `EdgeDilatedResidualBlock` từ `edrrednet/network.py` (không sửa)
- Viết mới class `Model` cho EDR-CNN10 với cấu trúc 4 lớp + injection

**Sanity check sau khi viết:**
```python
from argparse import Namespace
import torch
from ldctbench.methods.edrcnn10.network import Model
args = Namespace(num_edge_blocks=2, use_sobel_input=True)
model = Model(args)
x = torch.randn(1, 1, 128, 128)
out = model(x)
print(out.shape)  # phải ra torch.Size([1, 1, 128, 128])
```

### [x] Bước 1.2 — Tạo `loss.py`
- Copy nguyên `ldctbench/methods/edrrednet/loss.py` → `ldctbench/methods/edrcnn10/loss.py`
- Không sửa gì

### [x] Bước 1.3 — Tạo `Trainer.py`
- Copy `ldctbench/methods/edrrednet/Trainer.py` → `ldctbench/methods/edrcnn10/Trainer.py`
- Sửa duy nhất phần docstring (thay "EDR-REDNet" → "EDR-CNN10")
- Import Model từ `.network` — giữ nguyên

### [x] Bước 1.4 — Tạo `argparser.py`
- Copy nguyên `ldctbench/methods/edrrednet/argparser.py` → `ldctbench/methods/edrcnn10/argparser.py`
- Không sửa gì

### [x] Bước 1.5 — Tạo `__init__.py`
- Tạo `ldctbench/methods/edrcnn10/__init__.py`
- Nội dung: `from .Trainer import Trainer`

---

## GIAI ĐOẠN 2 — Tạo Config

### [x] Bước 2.1 — Tạo `configs/edrcnn10.yaml`
- Dựa trên `configs/edrrednet.yaml`
- Đổi `trainer: edrrednet` → `trainer: edrcnn10`
- Giữ: `loss_alpha: 0.1`, `num_edge_blocks: 2`, `optimizer: adam`
- Điều chỉnh từ CNN10 gốc: `lr: 0.00015837`, `patchsize: 92`
- Giai đoạn debug: `mbs: 8`, `max_iterations: 2000`, `data_subset: 0.1`

---

## GIAI ĐOẠN 3 — Kiểm Tra Cục Bộ (Local Debug)

### [x] Bước 3.1 — Forward pass test
```bash
python -c "
from argparse import Namespace
import torch
from ldctbench.methods.edrcnn10.network import Model
args = Namespace(num_edge_blocks=2, use_sobel_input=True)
model = Model(args)
x = torch.randn(1, 1, 128, 128)
out = model(x)
print('Shape OK:', x.shape == out.shape)
print('Params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
"
```
- **Kỳ vọng:** `Shape OK: True`, params ~171K

### [x] Bước 3.2 — Debug training 2000 iter
```bash
python -m ldctbench.train --config configs/edrcnn10.yaml
```
- **Kỳ vọng:** Loss giảm dần, không có NaN, không OOM

### [x] Bước 3.3 — Ghi kết quả debug vào nhật ký `CacThayDoi.md`

---

## GIAI ĐOẠN 4 — Ablation Study (Kaggle T4)

### Chuẩn bị
- [ ] Upload notebook lên Kaggle (tương tự quy trình EDR-REDNet)
- [ ] Kiểm tra data path và checkpoint path

### [ ] Bước 4.1 — Train Variant B (+ EdgeBlock, không SobelInput)
```bash
python -m ldctbench.train --method edrcnn10 --seed 1339 \
    --num_edge_blocks 2 --use_sobel_input False \
    --max_iterations 31000 --run_name edrcnn10_B_s1339

python -m ldctbench.train --method edrcnn10 --seed 2024 \
    --num_edge_blocks 2 --use_sobel_input False \
    --max_iterations 31000 --run_name edrcnn10_B_s2024

python -m ldctbench.train --method edrcnn10 --seed 42 \
    --num_edge_blocks 2 --use_sobel_input False \
    --max_iterations 31000 --run_name edrcnn10_B_s42
```

### [ ] Bước 4.2 — Train Variant C (+ SobelInput, không SobelLoss)
```bash
python -m ldctbench.train --method edrcnn10 --seed 1339 \
    --num_edge_blocks 2 --use_sobel_input True \
    --max_iterations 31000 --run_name edrcnn10_C_s1339

python -m ldctbench.train --method edrcnn10 --seed 2024 \
    --num_edge_blocks 2 --use_sobel_input True \
    --max_iterations 31000 --run_name edrcnn10_C_s2024

python -m ldctbench.train --method edrcnn10 --seed 42 \
    --num_edge_blocks 2 --use_sobel_input True \
    --max_iterations 31000 --run_name edrcnn10_C_s42
```

### [ ] Bước 4.3 — Train Variant D (Full EDR-CNN10)
```bash
python -m ldctbench.train --method edrcnn10 --seed 1339 \
    --num_edge_blocks 2 --use_sobel_input True --loss_alpha 0.1 \
    --max_iterations 31000 --run_name edrcnn10_D_s1339

python -m ldctbench.train --method edrcnn10 --seed 2024 \
    --num_edge_blocks 2 --use_sobel_input True --loss_alpha 0.1 \
    --max_iterations 31000 --run_name edrcnn10_D_s2024

python -m ldctbench.train --method edrcnn10 --seed 42 \
    --num_edge_blocks 2 --use_sobel_input True --loss_alpha 0.1 \
    --max_iterations 31000 --run_name edrcnn10_D_s42
```

### [ ] Bước 4.4 — Chọn Best Model mỗi Variant
- Theo SSIM cao nhất trên val set (giống cách làm với EDR-REDNet)
- Ghi vào `CacThayDoi.md` mục 6 (Nhật ký)

---

## GIAI ĐOẠN 5 — Đánh Giá Kết Quả

### [ ] Bước 5.1 — Chạy Wilcoxon test trên 9 bệnh nhân test
- Sửa `paper_scripts/evaluate_statistical_test.py` để load EDR-CNN10 model
- Hoặc tạo `paper_scripts/evaluate_edrcnn10.py` riêng
```bash
python paper_scripts/evaluate_edrcnn10.py
```
- **Output:** `results/evaluation/edrcnn10_per_patient_scores.csv`
- **Output:** `results/evaluation/edrcnn10_summary_mean_std.csv`
- **Output:** `results/evaluation/edrcnn10_wilcoxon_pvalues.csv`

### [ ] Bước 5.2 — So sánh với Baseline CNN10 và EDR-REDNet
- Điền vào bảng so sánh trong `CacThayDoi.md`
- Kết luận: 2 module có cải thiện Edge SSIM trên CNN10 không?

### [ ] Bước 5.3 — Đo Efficiency
```bash
python paper_scripts/measure_model_stats.py --model edrcnn10
```
- **Output:** Params, MACs, Inference Time của EDR-CNN10 vs CNN10

---

## GIAI ĐOẠN 6 — Push lên GitHub

### [ ] Bước 6.1 — Commit tất cả thay đổi
```bash
git add ldctbench/methods/edrcnn10/
git add configs/edrcnn10.yaml
git add docs/markdown/EDR-CNN10/
git add results/training/edrcnn10/
git add results/evaluation/edrcnn10_*.csv
git commit -m "feat: add EDR-CNN10 module with FixedSobelLayer + EdgeDilatedResidualBlock"
```

### [ ] Bước 6.2 — Push lên cả 2 repo
```bash
# origin (lung-diagnosis)
git push origin main

# ERD-RedCNN (subtree)
git subtree split --prefix ldct-benchmark HEAD
git push erd-redcnn <hash>:main --force
```

---

## Ghi Chú Quan Trọng

> **Lưu ý khác biệt so với EDR-REDNet:**
> CNN10 không có bottleneck sẵn → phải thêm Conv(64→64) trung gian.
> Nếu forward pass ra shape sai (H, W thay đổi), kiểm tra lại `padding` của lớp mới.

> **Dự kiến thời gian train:** ~30–60 phút/run trên Kaggle T4 (CNN10 nhẹ hơn RED-CNN ~74×).

> **Nếu loss NaN:** Giảm `loss_alpha` từ 0.1 xuống 0.05, hoặc giảm `lr`.

---

*Cập nhật lần cuối: 30/05/2026*
