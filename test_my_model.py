import os
import sys
import shutil
import torch

print("=== BẮT ĐẦU CHUẨN BỊ MÔI TRƯỜNG TEST ===")

# ==========================================
# 1. SỬA LỖI PYTORCH 2.6+ (Lỗi weights_only)
# ==========================================
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
import ldctbench.evaluate.utils
ldctbench.evaluate.utils.torch.load = safe_load

# Sửa lỗi đọc file YAML có tiếng Việt trên Windows (UnicodeDecodeError)
import ldctbench.utils
import yaml
def safe_load_yaml(path: str):
    with open(path, encoding='utf-8') as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
    return content
ldctbench.evaluate.utils.load_yaml = safe_load_yaml
ldctbench.utils.load_yaml = safe_load_yaml



# ==========================================
# 3. LẬP THƯ MỤC ẢO ĐỂ QUA MẶT THƯ VIỆN TEST
# ==========================================
CHECKPOINT_PATH = r"results\training\seed2024\seed2024_best_SSIM.pt" 
fake_run_dir = r"wandb\edr_redcnn_seed2024\files"

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ KHÔNG TÌM THẤY FILE TRỌNG SỐ TẠI: {CHECKPOINT_PATH}")
    print("Vui lòng tải file từ Kaggle về và để đúng vị trí này nhé!")
    exit(1)

os.makedirs(fake_run_dir, exist_ok=True)
shutil.copy(r"configs\edrrednet.yaml", os.path.join(fake_run_dir, "args.yaml"))
shutil.copy(CHECKPOINT_PATH, os.path.join(fake_run_dir, "best_SSIM.pt"))

print(f"✅ Đã tạo cấu trúc thư mục Test thành công!")

# ==========================================
# 4. CHẠY LỆNH TEST TRỰC TIẾP TRONG CÙNG TIẾN TRÌNH
# ==========================================
print("\n=== ĐANG CHẠY TEST (Quá trình này có thể mất 10-15 phút tùy cấu hình máy) ===")
# Giả lập tham số dòng lệnh
sys.argv = [
    "test_my_model.py",
    "--methods", "redcnn", "edr_redcnn_seed2024",
    "--metrics", "SSIM", "PSNR", "VIF",
    "--datafolder", "data",
    "--print_table"
]

try:
    from ldctbench.scripts.test import main
    main() 
except Exception as e:
    import traceback
    print(f"\n❌ Lỗi khi chạy Test: {e}")
    traceback.print_exc()
