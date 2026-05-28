# EDR-REDNet: Edge-Dilated Residual RED-CNN

> **Phát triển bởi:** Vương — dựa trên framework [`eeulig/ldct-benchmark`](https://github.com/eeulig/ldct-benchmark)  
> **Bài báo tham khảo:** Gholizadeh-Ansari et al., *"Deep Learning for Low-Dose CT Denoising Using Perceptual Loss and Edge Detection Layer"*, J. Digital Imaging, 2020.

---

## Giới thiệu

Repo này là phần mở rộng của `ldct-benchmark` (Eulig et al., Medical Physics 2024), bổ sung thêm mô hình **EDR-REDNet** — một biến thể của RED-CNN tích hợp:

- **FixedSobelLayer** — Trích xuất edge map 4 hướng (không trainable)
- **EdgeDilatedResidualBlock** — Dilated conv (rate=2, 3) + residual tại bottleneck
- **SobelEdgeLoss** — Ép model bảo tồn gradient biên so với NDCT

**Kết quả chính:**
- Edge SSIM cải thiện đáng kể (Wilcoxon p = 0.0020 < 0.05)
- PSNR/SSIM không giảm so với RED-CNN baseline (p ≥ 0.05)
- Overhead tham số: chỉ +18% (+0.33M params)

---

## Cấu trúc bổ sung so với repo gốc

```
ldct-benchmark/
├── ldctbench/methods/edrrednet/    ← [MỚI] Module EDR-REDNet
│   ├── network.py                  — Kiến trúc + FixedSobelLayer + EdgeDilatedResidualBlock
│   ├── loss.py                     — CombinedLoss (Charbonnier + SobelEdgeLoss)
│   ├── Trainer.py                  — Training loop với combined loss
│   └── argparser.py                — Hyperparameters riêng
│
├── configs/edrrednet.yaml          ← [MỚI] Config training
│
├── paper_scripts/                  ← [MỚI] Scripts phân tích & xuất kết quả
│   ├── evaluate_statistical_test.py
│   ├── measure_model_stats.py
│   ├── generate_paper_figures.py
│   ├── generate_paper_tables.py
│   ├── generate_table2.py
│   ├── split_paper_figures.py
│   ├── denoise_single_image.py
│   └── denoise_folder_gif.py
│
├── app.py                          ← [MỚI] Web app Streamlit (ROI analysis, comparison)
│
├── results/                        ← [MỚI] Kết quả thực nghiệm
│   ├── figures/                    — Figure 1–5 journal-style PNG
│   ├── tables/                     — Table 1–5 booktabs PNG
│   └── evaluation/                 — CSV: per-patient, Wilcoxon, efficiency
│
└── docs/markdown/                  ← [MỚI] Tài liệu nghiên cứu
    ├── EDRREDNet_KeHoachThucHien.md
    ├── EDRREDNet_KeHoachNangCap_TapChi.md
    └── EDRREDNet_SoSanh_GocVaMoi.md
```

---

## Cài đặt

```bash
# Clone repo
git clone <your-repo-url>
cd ldct-benchmark

# Cài dependencies gốc
pip install -e .

# Cài thêm cho web app
pip install -r requirements-app.txt
```

---

## Train EDR-REDNet

```bash
# Ablation Variant D — Full EDR-REDNet (3 seeds)
python -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed 1339
python -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed 2024
python -m ldctbench.scripts.train --config configs/edrrednet.yaml --seed 42
```

**Sanity check nhanh:**
```bash
python test_my_model.py
```

---

## Chạy Web App

```bash
streamlit run app.py
```

Tính năng:
- So sánh LDCT / Variant A / B / C / D / NDCT
- Phân tích ROI (CNR, HU deviation)
- Export Figure 3–7 và Table 1–5

---

## Xuất Figures & Tables cho Bài Báo

```bash
# Chạy từ thư mục gốc
python paper_scripts/generate_paper_figures.py   # → results/figures/fig1_architecture.png, fig2_...
python paper_scripts/generate_paper_tables.py    # → results/tables/table1_...png ... table5_...png
python paper_scripts/evaluate_statistical_test.py # → results/evaluation/*.csv
```

---

## Kết Quả Ablation

| Variant | FixedSobelLayer | EdgeBlock | SobelLoss | Mean SSIM | Mean PSNR | Edge SSIM |
|:--------|:---:|:---:|:---:|:---:|:---:|:---:|
| A — RED-CNN (Baseline) | ✗ | ✗ | ✗ | 0.887 | ~40.2 | — |
| B — + EdgeBlock | ✗ | ✓ | ✗ | 0.8857 ± 0.016 | 40.24 ± 1.36 | — |
| C — + FixedSobelLayer | ✓ | ✓ | ✗ | 0.8855 ± 0.016 | 40.16 ± 1.38 | — |
| **D — EDR-REDNet (Ours)** | ✓ | ✓ | ✓ | **0.8890 ± 0.014** | **40.41 ± 1.31** | **p=0.0020** ✅ |

---

## Citation

Nếu sử dụng framework gốc, vui lòng trích dẫn:

```bibtex
@article{ldctbench-medphys,
  title   = {Benchmarking deep learning-based low-dose CT image denoising algorithms},
  author  = {Eulig, Elias and Ommer, Björn and Kachelrieß, Marc},
  journal = {Medical Physics},
  volume  = {51}, number = {12}, pages = {8776-8788},
  doi     = {10.1002/mp.17379}, year = {2024}
}
```
