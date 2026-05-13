import os
import pydicom
import yaml
from tqdm import tqdm

def get_slice_location(path):
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        return float(ds.SliceLocation)
    except:
        return None

def find_pairs(patient_id):
    # 1. Thử Load từ info.yml trước
    patient_info = None
    try:
        with open("ldctbench/data/info.yml", "r", encoding="utf-8") as f:
            info = yaml.safe_load(f)
        for s in ["test_set", "train_set", "val_set"]:
            for p in info.get(s, []):
                if p["id"] == patient_id:
                    patient_info = p
                    break
    except:
        pass
    
    if patient_info:
        in_dir = os.path.join("data", patient_info["input"])
        tg_dir = os.path.join("data", patient_info["target"])
    else:
        # 2. Nếu không có trong YAML, quét trực tiếp thư mục data
        print(f"⚠️ {patient_id} không có trong info.yml, đang quét thủ công...")
        base_path = os.path.join("data", "LDCT-and-Projection-data", patient_id)
        if not os.path.exists(base_path):
            print(f"❌ Không tìm thấy thư mục của bệnh nhân tại: {base_path}")
            return
        
        # Tìm tất cả thư mục chứa ảnh (Bỏ qua Projections)
        low_dose_dirs = []
        full_dose_dirs = []
        for root, dirs, files in os.walk(base_path):
            if "Projections" in root: continue # Bỏ qua dữ liệu thô
            
            if any(f.endswith(".dcm") for f in files):
                if "Low Dose" in root:
                    low_dose_dirs.append(root)
                if "Full Dose" in root:
                    full_dose_dirs.append(root)
        
        if not low_dose_dirs or not full_dose_dirs:
            print(f"❌ Không tìm thấy đủ thư mục Low Dose và Full Dose bên trong {base_path}")
            return
            
        # Ưu tiên chọn cặp thư mục nằm chung trong một thư mục ngày khám
        in_dir, tg_dir = None, None
        for ld in low_dose_dirs:
            parent = os.path.dirname(ld)
            for fd in full_dose_dirs:
                if os.path.dirname(fd) == parent:
                    in_dir, tg_dir = ld, fd
                    break
            if in_dir: break
            
        # Nếu không tìm thấy cặp chung parent, lấy đại cặp đầu tiên
        if not in_dir:
            in_dir, tg_dir = low_dose_dirs[0], full_dose_dirs[0]
            print("⚠️ Cảnh báo: Không tìm thấy cặp ảnh cùng ngày khám, đang lấy đại cặp đầu tiên tìm thấy.")

    # 3. Quét file và khớp cặp (giữ nguyên logic cũ)
    print(f"🔍 Đang quét bệnh nhân {patient_id}...")
    print(f"📂 Low Dose: {in_dir}")
    print(f"📂 Full Dose: {tg_dir}")

    def scan_dir(directory):
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")]
        results = []
        for f in tqdm(files, desc=f"Reading {os.path.basename(directory)}"):
            loc = get_slice_location(f)
            if loc is not None:
                results.append({"path": f, "loc": loc})
        # Sắp xếp theo vị trí lát cắt
        return sorted(results, key=lambda x: x["loc"])

    in_slices = scan_dir(in_dir)
    tg_slices = scan_dir(tg_dir)

    # 3. Khớp cặp dựa trên vị trí (SliceLocation)
    # Vì Mayo data thường khớp chính xác vị trí, chúng ta có thể dùng dictionary để map
    tg_map = {round(s["loc"], 2): s["path"] for s in tg_slices}
    
    print("\n✅ CÁC CẶP FILE KHỚP NHAU (GỢI Ý):")
    print("-" * 80)
    print(f"{'Vị trí (Loc)':<15} | {'File Low Dose':<30} | {'File Full Dose (Khớp)'}")
    print("-" * 80)
    
    count = 0
    for s in in_slices:
        loc_rounded = round(s["loc"], 2)
        if loc_rounded in tg_map:
            # Chỉ in ra 10 cặp đại diện (Đầu, giữa, cuối)
            if count % (len(in_slices)//10) == 0 or count == len(in_slices)-1:
                print(f"{loc_rounded:<15} | {os.path.basename(s['path']):<30} | {os.path.basename(tg_map[loc_rounded])}")
            count += 1
    
    print("-" * 80)
    print(f"Tìm thấy tổng cộng {count} cặp khớp nhau hoàn toàn về vị trí giải phẫu.")
    print("💡 Bạn hãy chọn 1 cặp file cùng hàng ở trên để Upload lên Web App nhé!")

if __name__ == "__main__":
    import sys
    pid = sys.argv[1] if len(sys.argv) > 1 else "L150"
    find_pairs(pid)
