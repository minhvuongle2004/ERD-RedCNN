# Câu 20: Các tham số training quan trọng là gì?

## 🎯 Mục tiêu
Hiểu được các tham số quan trọng khi train RED-CNN và cách chọn giá trị phù hợp

---

## 📖 Trả lời

### Tham số training = Cài đặt để model học

**Giống như cài đặt máy điều hòa**:
```
Nhiệt độ: 25°C (vừa phải)
Quạt: Tự động
Chế độ: Làm mát

→ Cài đặt khác → Kết quả khác!
```

**Áp dụng vào training**:
```
Learning rate: 0.001 (tốc độ học)
Batch size: 16 (số ảnh/lần)
Epochs: 100 (số lần học)

→ Tham số khác → Model học khác!
```

---

## 🎛️ 5 tham số quan trọng nhất

### 1. Learning Rate (Tốc độ học)

**Là gì?**: Tốc độ model học từ mỗi ví dụ

**Ví dụ dễ hiểu**:
```
Học lái xe:
Learning rate cao: Học nhanh, nhưng có thể bỏ sót
Learning rate thấp: Học chậm, nhưng kỹ càng
Learning rate vừa: Cân bằng ⚖️
```

**Giá trị phổ biến**:
```
Quá cao:  0.1, 0.01     → Model không học được ❌
Cao:      0.001        → Có thể học, nhưng không ổn định ⚠️
Vừa:      0.0001       → Tốt cho hầu hết trường hợp ✅
Thấp:     0.00001      → Học chậm, nhưng ổn định ✅
Quá thấp: 0.000001     → Học quá chậm ❌
```

**Cho RED-CNN**:
```
Thường dùng: 0.0001 - 0.001
Bắt đầu:     0.001
Nếu không ổn định: Giảm xuống 0.0001
```

**Dấu hiệu Learning Rate không phù hợp**:
```
Quá cao:
- Loss tăng lên thay vì giảm
- Loss nhảy lung tung
- Model không học được

Quá thấp:
- Loss giảm rất chậm
- Mất nhiều thời gian train
- Có thể không đạt được kết quả tốt
```

---

### 2. Batch Size (Kích thước lô)

**Là gì?**: Số ảnh xử lý cùng lúc trong 1 lần

**Ví dụ dễ hiểu**:
```
Học bài:
Batch size = 1:  Học từng bài một (chậm, nhưng kỹ)
Batch size = 10: Học 10 bài cùng lúc (nhanh hơn)
Batch size = 100: Học 100 bài cùng lúc (rất nhanh, nhưng có thể không kỹ)
```

**Giá trị phổ biến**:
```
Nhỏ:     1, 2, 4       → Chậm, nhưng ổn định ✅
Vừa:     8, 16, 32     → Cân bằng ⚖️ ✅
Lớn:     64, 128       → Nhanh, nhưng cần GPU mạnh ⚠️
Rất lớn: 256+          → Cần GPU rất mạnh ❌
```

**Cho RED-CNN**:
```
GPU yếu (6GB):   Batch size = 4-8
GPU vừa (8GB):   Batch size = 8-16
GPU mạnh (16GB+): Batch size = 16-32
```

**Tại sao không dùng batch size quá lớn?**:
```
Batch size quá lớn:
→ Cần nhiều bộ nhớ GPU
→ Có thể không học tốt (gradient quá "mượt")
```

**Tại sao không dùng batch size quá nhỏ?**:
```
Batch size = 1:
→ Rất chậm
→ Gradient không ổn định
→ Model học không tốt
```

---

### 3. Epochs (Số kỷ nguyên)

**Là gì?**: Số lần model học qua TOÀN BỘ dataset

**Ví dụ dễ hiểu**:
```
Học sách giáo khoa:
Epoch 1: Đọc lần đầu (chưa hiểu nhiều)
Epoch 2: Đọc lần 2 (hiểu hơn)
Epoch 10: Đọc lần 10 (hiểu rất rõ)
```

**Giá trị phổ biến**:
```
Ít:     10-50 epochs    → Model chưa học đủ ❌
Vừa:    50-100 epochs   → Model học tốt ✅
Nhiều:  100-200 epochs  → Model học rất tốt ✅
Quá nhiều: 500+ epochs  → Có thể overfitting ⚠️
```

**Cho RED-CNN**:
```
Thường dùng: 50-100 epochs
Nếu dataset lớn: 100-200 epochs
Nếu dataset nhỏ: 30-50 epochs
```

**Early Stopping**:
```
Nếu validation loss không giảm sau 10 epochs:
→ Dừng sớm (early stopping)
→ Tránh overfitting
```

---

### 4. Loss Function (Hàm mất mát)

**Là gì?**: Cách tính sai số giữa dự đoán và đáp án

**Các loại phổ biến**:

**MSE Loss (Mean Squared Error)**:
```
MSE = (Dự đoán - Thật)²

Ưu điểm:
✅ Phạt nặng sai số lớn
✅ Phù hợp cho ảnh y tế

Nhược điểm:
❌ Có thể làm mờ ảnh
```

**MAE Loss (Mean Absolute Error)**:
```
MAE = |Dự đoán - Thật|

Ưu điểm:
✅ Giữ được chi tiết sắc nét

Nhược điểm:
❌ Không phạt nặng sai số lớn
```

**Cho RED-CNN**:
```
Thường dùng: MSE Loss
Hoặc: L1 Loss (MAE)
Hoặc: Kết hợp cả 2
```

---

### 5. Optimizer (Bộ tối ưu)

**Là gì?**: Thuật toán điều chỉnh model để giảm loss

**Các loại phổ biến**:

**SGD (Stochastic Gradient Descent)**:
```
Đơn giản, ổn định
Learning rate: 0.001
Momentum: 0.9
```

**Adam**:
```
Thông minh hơn, tự điều chỉnh
Learning rate: 0.0001
Beta1: 0.9
Beta2: 0.999

→ Phổ biến nhất! ✅
```

**Cho RED-CNN**:
```
Thường dùng: Adam
Learning rate: 0.0001
```

---

## 📊 Bảng tham số đề xuất

### Cho RED-CNN

```
┌─────────────────┬──────────────┬──────────────────┐
│ Tham số         │ Giá trị      │ Ghi chú          │
├─────────────────┼──────────────┼──────────────────┤
│ Learning rate   │ 0.0001       │ Bắt đầu từ đây   │
│ Batch size      │ 8-16         │ Tùy GPU          │
│ Epochs          │ 50-100       │ Tùy dataset      │
│ Loss function   │ MSE          │ Hoặc L1          │
│ Optimizer       │ Adam         │ Phổ biến nhất   │
│ Weight decay    │ 0.0001       │ Regularization   │
└─────────────────┴──────────────┴──────────────────┘
```

---

## 🔧 Các tham số khác

### 6. Weight Decay (Regularization)

**Là gì?**: Giảm overfitting bằng cách giới hạn weights

**Giá trị**:
```
Thường: 0.0001
Hoặc: 0.00001
```

---

### 7. Learning Rate Schedule

**Là gì?**: Thay đổi learning rate khi train

**Ví dụ**:
```
Epoch 1-50:   Learning rate = 0.001
Epoch 51-75:  Learning rate = 0.0001  (giảm 10 lần)
Epoch 76-100: Learning rate = 0.00001 (giảm 100 lần)
```

**Tại sao?**:
```
Ban đầu: Học nhanh (tìm vùng tốt)
Sau đó:  Học chậm (tinh chỉnh)
```

---

### 8. Validation Frequency

**Là gì?**: Bao lâu kiểm tra validation một lần

**Ví dụ**:
```
Sau mỗi epoch: Kiểm tra validation
Hoặc: Sau mỗi 5 epochs
```

---

## 💡 Cách chọn tham số

### Bước 1: Bắt đầu với giá trị mặc định

```
Learning rate: 0.0001
Batch size: 16
Epochs: 50
Optimizer: Adam
Loss: MSE
```

### Bước 2: Quan sát training

```
Nếu loss không giảm:
→ Giảm learning rate (0.0001 → 0.00001)

Nếu loss nhảy lung tung:
→ Giảm learning rate

Nếu loss giảm quá chậm:
→ Tăng learning rate (0.0001 → 0.001)

Nếu GPU out of memory:
→ Giảm batch size (16 → 8)
```

### Bước 3: Điều chỉnh

```
Thử các giá trị khác nhau
→ Chọn giá trị tốt nhất
```

---

## 🎯 Tóm tắt

**5 tham số quan trọng nhất**:

```
1. Learning Rate (0.0001)
   → Tốc độ học

2. Batch Size (8-16)
   → Số ảnh/lần

3. Epochs (50-100)
   → Số lần học

4. Loss Function (MSE)
   → Cách tính sai số

5. Optimizer (Adam)
   → Thuật toán tối ưu
```

**Cách chọn**:
- ✅ Bắt đầu với giá trị mặc định
- ✅ Quan sát training
- ✅ Điều chỉnh dần

**Kết quả**: Tham số phù hợp = Model học tốt! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Learning rate là gì? Giá trị nào phù hợp?
2. Batch size là gì? Tại sao không dùng quá lớn?
3. Epochs là gì? Bao nhiêu epochs là đủ?
4. Loss function nào phù hợp cho RED-CNN?
5. Làm sao biết tham số có phù hợp không?

**Hoạt động**: 
- Xem file config thật
- Giải thích từng tham số
- Thử thay đổi và xem kết quả

---

➡️ **Tiếp theo**: `CAU_HOI_21.md` - Cách chạy training như thế nào?

