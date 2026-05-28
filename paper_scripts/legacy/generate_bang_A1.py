"""
Script tạo Bảng A.1 — Kết quả SSIM / PSNR / VIF của 8 model trên toàn bộ test set
Phân tách theo anatomy: Chest (C), Abdomen/Low (L), Neuro (N)
Output: 
  - results/bang_A1/test_metrics.yaml
  - results/bang_A1/Bang_A1_Ket_Qua_Benchmark.xlsx
  - results/bang_A1/Bang_A1_Ket_Qua_Benchmark.docx
"""

import os
import sys
import warnings
import time
from itertools import product
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
import io

# Fix encoding UTF-8 cho Windows Console
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from tqdm import tqdm

# ─── Tự động thêm thư mục gốc dự án vào sys.path ───────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ldctbench.data import TestData
from ldctbench.evaluate import compute_metric, save_raw, setup_trained_model
from ldctbench.hub import Methods, load_model
from ldctbench.utils import save_yaml

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
DATAFOLDER = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results", "bang_A1")
METHODS = ["cnn10", "redcnn", "wganvgg", "resnet", "qae", "dugan", "transct"]
# NOTE: "bilateral" bị loại bỏ do thiếu thư viện C++ tương thích trên Windows
METRICS = ["SSIM", "PSNR", "VIF"]
ANATOMY_MAP = {"C": "Chest", "L": "Abdomen", "N": "Neuro"}

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("  BANG A.1 -- DANH GIA CHAT LUONG ANH: SSIM / PSNR / VIF")
print("  Benchmark LDCT Denoising -- Mayo Clinic Dataset")
print(f"  Thoi diem chay: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("=" * 70)

# ─── KIỂM TRA GPU ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEV = torch.device("cuda:0")
    print(f"\n✅ GPU phát hiện: {torch.cuda.get_device_name(0)}")
else:
    DEV = torch.device("cpu")
    print("\n⚠️  Không có GPU. Đang chạy trên CPU (sẽ chậm hơn).")

# ─── TẢI DỮ LIỆU TEST ─────────────────────────────────────────────────────────
print(f"\n📂 Đang tải tập dữ liệu test từ: {DATAFOLDER}")
data = TestData(DATAFOLDER, "meanstd")
print(f"   → Đã tải {len(data.samples)} bệnh nhân test.")

# ─── TẢI CÁC MÔ HÌNH ──────────────────────────────────────────────────────────
print(f"\n🤖 Đang tải {len(METHODS)} mô hình...")
networks = {}
for method in METHODS:
    print(f"   → Tải {method}...", end=" ")
    networks[method] = load_model(method, eval=True).to(DEV)
    params_m = sum(p.numel() for p in networks[method].parameters()) / 1e6
    print(f"({params_m:.2f}M tham số) ✓")

# ─── THIẾT LẬP DICT LƯU KẾT QUẢ ──────────────────────────────────────────────
metrics_dict = {
    pat["info"]["id"]: {
        m: {method: [] for method in ["LD"] + METHODS} for m in METRICS
    }
    for pat in data.samples
}

# ─── CHẠY INFERENCE & TÍNH METRICS ────────────────────────────────────────────
print(f"\n🚀 Bắt đầu chạy inference trên RTX / GPU...\n")
start_total = time.time()

ldct_iqa = None  # Không dùng LDCTIQA trong bảng này

with torch.no_grad():
    for patient in (pbar := tqdm(data, desc="Inference", unit="BN")):
        patient_name = patient["info"]["id"]
        f_hd = patient["f_hd"]
        exam_type = f_hd[0][0].split("/")[1][0]  # "C", "L", or "N"

        # Lưu ảnh LD và HD gốc
        savedir = os.path.join(RESULTS_DIR, "images", patient_name)
        os.makedirs(savedir, exist_ok=True)
        save_raw(savedir, "LD", data._convert_hu(data.denormalize(patient["x"]).squeeze().numpy(), to_hu=True))
        save_raw(savedir, "HD", data._convert_hu(data.denormalize(patient["y"]).squeeze().numpy(), to_hu=True))

        for i_method, (method_name, method) in enumerate(networks.items()):
            pbar.set_description(f"  [{patient_name}] {method_name}")
            predictions = []

            for slice_idx in range(patient["info"]["n_slices"]):
                x = torch.unsqueeze(torch.unsqueeze(patient["x"][slice_idx], 0), 0).to(DEV)
                y = torch.unsqueeze(torch.unsqueeze(patient["y"][slice_idx], 0), 0).to(DEV)

                # Tính metric cho ảnh LD gốc (chỉ 1 lần)
                if i_method == 0:
                    res = compute_metric(y, x, metrics=METRICS, denormalize_fn=data.denormalize, exam_type=exam_type, ldct_iqa=ldct_iqa)
                    for m, r in res.items():
                        metrics_dict[patient_name][m]["LD"].extend(r)

                # Chạy mô hình và tính metric
                y_hat = method(x)
                res = compute_metric(y, y_hat, metrics=METRICS, denormalize_fn=data.denormalize, exam_type=exam_type, ldct_iqa=ldct_iqa)
                for m, r in res.items():
                    metrics_dict[patient_name][m][method_name].extend(r)
                predictions.append(data.denormalize(y_hat.squeeze().detach().cpu().numpy()))

            save_raw(savedir, method_name, data._convert_hu(np.stack(predictions, axis=0), to_hu=True))

total_time = time.time() - start_total
print(f"\n✅ Inference hoàn tất trong {total_time/60:.1f} phút.")

# ─── LƯU FILE YAML ────────────────────────────────────────────────────────────
yaml_path = os.path.join(RESULTS_DIR, "test_metrics.yaml")
save_yaml(metrics_dict, yaml_path)
print(f"\n💾 Đã lưu raw data: {yaml_path}")

# ─── TỔNG HỢP KẾT QUẢ ────────────────────────────────────────────────────────
records = []
for patient, metric_dict in metrics_dict.items():
    exam_type = patient[0]  # "C", "L", "N"
    for metric, method_dict in metric_dict.items():
        for method, values in method_dict.items():
            for v in values:
                records.append({
                    "exam_type_code": exam_type,
                    "anatomy": ANATOMY_MAP.get(exam_type, exam_type),
                    "patient": patient,
                    "metric": metric,
                    "method": method,
                    "value": v,
                })

df_all = pd.DataFrame(records)

# ─── TẠO BẢNG TỔNG HỢP TRUNG BÌNH THEO ANATOMY ──────────────────────────────
all_methods = ["LD"] + METHODS
rows = []
for method in all_methods:
    row = {"Mô hình": method}
    for metric in METRICS:
        for et_code, et_name in ANATOMY_MAP.items():
            subset = df_all[
                (df_all["metric"] == metric) &
                (df_all["method"] == method) &
                (df_all["exam_type_code"] == et_code)
            ]["value"]
            if len(subset) > 0:
                row[f"{metric} ({et_name})"] = round(float(subset.mean()), 4)
            else:
                row[f"{metric} ({et_name})"] = float("nan")
    rows.append(row)

df_table = pd.DataFrame(rows)

# In bảng ra terminal
print("\n" + "=" * 70)
print("  BẢNG A.1 — KẾT QUẢ SSIM / PSNR / VIF THEO ANATOMY")
print("=" * 70)
print(df_table.to_markdown(index=False))

# ─── XUẤT FILE EXCEL ──────────────────────────────────────────────────────────
xlsx_path = os.path.join(RESULTS_DIR, "Bang_A1_Ket_Qua_Benchmark.xlsx")
try:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Sheet 1: Bảng tổng hợp
        df_table.to_excel(writer, sheet_name="Bảng A.1 Tổng Hợp", index=False)
        ws = writer.sheets["Bảng A.1 Tổng Hợp"]

        # Format header
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        header_fill = PatternFill(start_color="2E4057", end_color="2E4057", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=11)
        border = Border(
            left=Side(style="thin"), right=Side(style="thin"),
            top=Side(style="thin"), bottom=Side(style="thin")
        )
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border

        # Tô màu dòng LD và auto-width
        ld_fill = PatternFill(start_color="FFE5B4", end_color="FFE5B4", fill_type="solid")
        best_fill = PatternFill(start_color="C8E6C9", end_color="C8E6C9", fill_type="solid")
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.border = border
                cell.alignment = Alignment(horizontal="center")
                if row[0].value == "LD":
                    cell.fill = ld_fill

        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(len(str(c.value or "")) for c in col) + 4

        # Sheet 2: Data chi tiết từng bệnh nhân
        df_all.to_excel(writer, sheet_name="Chi Tiết Từng Bệnh Nhân", index=False)

    print(f"\n📊 Đã xuất Excel: {xlsx_path}")
except ImportError:
    print("\n⚠️  Thiếu thư viện openpyxl. Đang cài thêm...")
    os.system(f'"{sys.executable}" -m pip install openpyxl -q')
    df_table.to_excel(xlsx_path, index=False)
    print(f"📊 Đã xuất Excel (cơ bản): {xlsx_path}")

# ─── XUẤT FILE WORD ──────────────────────────────────────────────────────────
docx_path = os.path.join(RESULTS_DIR, "Bang_A1_Ket_Qua_Benchmark.docx")
try:
    from docx import Document
    from docx.shared import Pt, RGBColor, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)

    # Tiêu đề
    title = doc.add_heading("BẢNG A.1 — KẾT QUẢ BENCHMARK LDCT DENOISING", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(f"Ngày chạy: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    doc.add_paragraph(f"Thiết bị: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    doc.add_paragraph(f"Số bệnh nhân test: {len(data.samples)}")
    doc.add_paragraph("")

    doc.add_heading("Bảng 1: SSIM / PSNR / VIF theo Anatomy", level=2)

    cols = list(df_table.columns)
    table = doc.add_table(rows=1 + len(df_table), cols=len(cols))
    table.style = "Table Grid"

    # Header
    hdr = table.rows[0]
    for i, col_name in enumerate(cols):
        cell = hdr.cells[i]
        cell.text = col_name
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:fill'), '2E4057')
        shd.set(qn('w:val'), 'clear')
        tcPr.append(shd)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Data rows
    for r_idx, row_data in df_table.iterrows():
        row = table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val) if pd.notna(val) else "—"
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")
    note = doc.add_paragraph("Ghi chú: ")
    note.runs[0].bold = True
    doc.add_paragraph("• LD: Kết quả ảnh Low Dose gốc (chưa khử nhiễu) — đây là điểm chuẩn thấp nhất.")
    doc.add_paragraph("• SSIM cao hơn = ảnh giống với Ground Truth hơn (thang 0–1).")
    doc.add_paragraph("• PSNR cao hơn = nhiễu ít hơn (đơn vị dB).")
    doc.add_paragraph("• VIF cao hơn = thông tin thị giác được bảo toàn tốt hơn (thang 0–1).")
    doc.add_paragraph("• Bilateral Filter không được bao gồm do thiếu thư viện C++ trên Windows.")

    doc.save(docx_path)
    print(f"📝 Đã xuất Word: {docx_path}")

except ImportError:
    print("\n⚠️  Thiếu thư viện python-docx. Đang cài thêm...")
    os.system(f'"{sys.executable}" -m pip install python-docx -q')
    print("Vui lòng chạy lại script để xuất file Word.")

print("\n" + "=" * 70)
print("  ✅ HOÀN THÀNH! Tất cả kết quả đã được lưu vào:")
print(f"     {RESULTS_DIR}")
print("=" * 70)
