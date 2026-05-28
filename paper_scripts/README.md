# paper_scripts/

Scripts phân tích và xuất kết quả bài báo EDR-REDNet.

| Script | Mô tả |
|--------|-------|
| `evaluate_statistical_test.py` | Tính PSNR/SSIM/Edge-SSIM + Wilcoxon test |
| `measure_model_stats.py` | Đo MACs, Params, inference time |
| `generate_paper_figures.py` | Tạo Figure 1-2 (kiến trúc, ablation design) |
| `generate_paper_tables.py` | Tạo Table 1-5 dạng PNG journal-style |
| `generate_table2.py` | Tổng hợp validation metrics từ 3 seeds |
| `split_paper_figures.py` | Tách Figure 3/4/5 từ combined PNG |
| `generate_stress_test.py` | Tạo data stress test |
| `find_matching_slices.py` | Tìm matching slices LDCT/NDCT |
| `denoise_single_image.py` | Denoise một ảnh đơn lẻ |
| `denoise_folder_gif.py` | Denoise cả folder + tạo GIF |

## legacy/
Scripts thử nghiệm từ giai đoạn đầu (không dùng nữa, giữ để tham khảo).

## Cách chạy
Chạy từ thư mục gốc `ldct-benchmark/`:
```bash
python paper_scripts/generate_paper_figures.py
python paper_scripts/generate_paper_tables.py
python paper_scripts/evaluate_statistical_test.py
```
