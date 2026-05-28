import os
import numpy as np
import pydicom
import torch
import yaml

def add_ct_noise(image_hu, noise_level=100):
    """
    Bơm nhiễu vào ảnh HU. 
    noise_level = 100 HU là mức nhiễu cao (Ultra Low Dose).
    Mức bình thường trong Mayo chỉ khoảng 30-50 HU.
    """
    noise = np.random.normal(0, noise_level, image_hu.shape)
    noisy_image = image_hu + noise
    return noisy_image.astype(np.float32)

def create_stress_test(patient_id="L150", slice_idx=100):
    # 1. Tìm đường dẫn NDCT từ info.yml
    with open("ldctbench/data/info.yml", "r", encoding="utf-8") as f:
        info = yaml.safe_load(f)
    
    patient_info = None
    for s in ["test_set", "train_set", "val_set"]:
        for p in info.get(s, []):
            if p["id"] == patient_id:
                patient_info = p
                break
    
    if not patient_info:
        print(f"❌ Không tìm thấy bệnh nhân {patient_id}")
        return

    # Đường dẫn file gốc NDCT (Full Dose)
    target_dir = os.path.join("data", patient_info["target"])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".dcm")])
    
    if slice_idx >= len(target_files):
        slice_idx = len(target_files) // 2
    
    target_path = target_files[slice_idx]
    ds = pydicom.dcmread(target_path)
    
    # 2. Chuyển sang HU
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    pixel_hu = ds.pixel_array.astype(np.float32) * slope + intercept
    
    # 3. Tạo nhiễu siêu mạnh
    print(f"🧬 Đang tạo nhiễu mức độ cao cho lát cắt {slice_idx}...")
    noisy_hu = add_ct_noise(pixel_hu, noise_level=200) # 200 HU là cực kỳ nhiễu
    
    # 4. Lưu lại thành file DICOM mới
    output_dir = "stress_test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    def save_dcm(pixel_data, filename, original_ds):
        # Chuyển ngược từ HU về giá trị thô để lưu DICOM
        raw_pixels = (pixel_data - intercept) / slope
        original_ds.PixelData = raw_pixels.astype(np.int16).tobytes()
        original_ds.save_as(os.path.join(output_dir, filename))
        print(f"💾 Đã lưu: {os.path.join(output_dir, filename)}")

    save_dcm(pixel_hu, f"{patient_id}_target.dcm", ds.copy())
    save_dcm(noisy_hu, f"{patient_id}_ultra_low_dose.dcm", ds.copy())
    
    print("\n🚀 Xong! Bây giờ bạn hãy mở Web App, vào phần 'Tải lên file (.dcm)':")
    print(f"1. Tải file '{patient_id}_ultra_low_dose.dcm' vào ô LDCT.")
    print(f"2. Tải file '{patient_id}_target.dcm' vào ô NDCT.")
    print("3. Kéo Windowing và xem EDR-REDNet 'chiến đấu' với nhiễu nhé!")

if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "L150"
    create_stress_test(pid)
