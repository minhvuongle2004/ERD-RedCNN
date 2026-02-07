# 📦 Hướng dẫn cài đặt thư viện cho app.py

## 🎯 Tổng quan

File `app.py` là ứng dụng Streamlit để khử nhiễu ảnh CT. Để chạy được, bạn cần cài đặt các thư viện sau.

---

## 📋 Danh sách thư viện cần thiết

### 1. Thư viện chính (Core)

```bash
# PyTorch (Deep Learning framework)
torch>=2.0.0
torchvision>=0.15.0

# NumPy (Xử lý mảng số)
numpy>=1.21.0

# PyDICOM (Đọc file DICOM)
pydicom==3.0.1

# Pillow (Xử lý ảnh)
Pillow>=9.0.0
```

### 2. Streamlit (Web UI)

```bash
streamlit>=1.28.0
```

### 3. Thư viện hỗ trợ

```bash
# PyYAML (Đọc config files)
PyYAML>=6.0

# Pandas (Xử lý dữ liệu)
pandas>=1.3.0
```

### 4. Package ldctbench (Package của dự án)

```bash
# Cài từ source code
pip install -e .
```

**Lưu ý**: `-e` nghĩa là "editable mode" - cho phép chỉnh sửa code mà không cần cài lại.

---

## 🚀 Cách cài đặt

### Cách 1: Cài từng thư viện (Thủ công)

```bash
# 1. Cài PyTorch (chọn version phù hợp với GPU của bạn)
# CPU only:
pip install torch torchvision

# GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Cài các thư viện khác
pip install numpy pydicom==3.0.1 Pillow streamlit PyYAML pandas

# 3. Cài package ldctbench
pip install -e .
```

### Cách 2: Cài từ file requirements (Khuyến nghị)

```bash
# 1. Cài PyTorch trước (chọn version phù hợp)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Cài các thư viện còn lại từ file requirements-app.txt
pip install -r requirements-app.txt

# 3. Cài package ldctbench
pip install -e .
```

### Cách 3: Cài tất cả từ pyproject.toml

```bash
# Cài tất cả dependencies từ pyproject.toml
pip install -e .

# Sau đó cài thêm Streamlit (không có trong pyproject.toml)
pip install streamlit
```

---

## ✅ Kiểm tra cài đặt

### Test import các thư viện

```python
# Tạo file test_imports.py
import streamlit as st
import torch
import numpy as np
import pydicom
from PIL import Image
from ldctbench.hub import load_model
import tkinter as tk

print("✅ Tất cả thư viện đã được cài đặt thành công!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

Chạy:
```bash
python test_imports.py
```

### Test chạy app

```bash
streamlit run app.py
```

Nếu không có lỗi import, app sẽ chạy được!

---

## 🔧 Xử lý lỗi thường gặp

### Lỗi 1: ModuleNotFoundError: No module named 'streamlit'

**Giải pháp**:
```bash
streamlit : The term 'streamlit' is not recognized as the name of a cmdlet, function, script file, or operable program. Check the spelling of 
the name, or if a path was included, verify that the path is correct and try again.
At line:1 char:1
+ streamlit run app.py
+ ~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (streamlit:String) [], CommandNotFoundException
    + FullyQualifiedErrorId : CommandNotFoundException
```

### Lỗi 2: ModuleNotFoundError: No module named 'ldctbench'

**Giải pháp**:
```bash
# Đảm bảo đang ở thư mục gốc của dự án
cd /path/to/ldct-benchmark
pip install -e .
```

### Lỗi 3: Lỗi khi import tkinter

**Giải pháp**:
- **Windows**: tkinter có sẵn trong Python, không cần cài
- **Linux**: 
  ```bash
  sudo apt-get install python3-tk
  ```
- **Mac**: tkinter có sẵn trong Python

### Lỗi 4: PyTorch không nhận GPU

**Giải pháp**:
```bash
# Gỡ PyTorch cũ
pip uninstall torch torchvision

# Cài lại với CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Kiểm tra:
```python
import torch
print(torch.cuda.is_available())  # Phải là True
```

---

## 📝 Checklist cài đặt

Trước khi chạy `app.py`, đảm bảo:

```
□ 1. Đã cài Python 3.10+
□ 2. Đã cài PyTorch (với GPU nếu có)
□ 3. Đã cài Streamlit
□ 4. Đã cài các thư viện: numpy, pydicom, Pillow, PyYAML, pandas
□ 5. Đã cài package ldctbench (pip install -e .)
□ 6. Đã test import không có lỗi
□ 7. Đã có model weights (tự động download khi chạy lần đầu)
```

---

## 🎯 Tóm tắt nhanh

**Cài đặt nhanh nhất**:

```bash
# 1. Cài PyTorch (chọn version phù hợp)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Cài Streamlit
pip install streamlit

# 3. Cài package và dependencies
pip install -e .

# 4. Chạy app
streamlit run app.py
```

---

## 📚 Tham khảo

- **PyTorch**: https://pytorch.org/get-started/locally/
- **Streamlit**: https://docs.streamlit.io/
- **PyDICOM**: https://pydicom.github.io/

---

➡️ **Tiếp theo**: Xem `HUONG_DAN_TRAIN.md` để biết cách train model

