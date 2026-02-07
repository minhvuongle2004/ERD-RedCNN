# Câu 23: Xử lý lỗi khi training như thế nào?

## 🎯 Mục tiêu
Biết cách xử lý các lỗi thường gặp khi train RED-CNN

---

## 📖 Trả lời

### Lỗi khi training = Vấn đề xảy ra trong quá trình train

**Giống như lái xe**:
```
Lái xe có thể gặp:
- Hết xăng
- Xe hỏng
- Lạc đường

→ Cần biết cách xử lý!
```

**Áp dụng vào training**:
```
Training có thể gặp:
- Out of memory
- Loss không giảm
- Model không học được

→ Cần biết cách xử lý!
```

---

## 🚨 Các lỗi thường gặp

### 1. Out of Memory (OOM) - Hết bộ nhớ GPU

**Lỗi**:
```
RuntimeError: CUDA out of memory
```

**Nguyên nhân**:
```
- Batch size quá lớn
- Ảnh quá lớn
- Model quá lớn
- GPU không đủ bộ nhớ
```

**Cách xử lý**:

**Giảm batch size**:
```python
# Trước: batch_size = 32
# Sau:   batch_size = 8 hoặc 4
train_loader = DataLoader(dataset, batch_size=8)
```

**Giảm kích thước ảnh**:
```python
# Trước: 512×512
# Sau:   256×256 hoặc 128×128
img = img.resize((256, 256))
```

**Gradient accumulation**:
```python
# Thay vì batch_size=32
# Dùng batch_size=8, accumulate 4 lần = tương đương 32
accumulation_steps = 4

for i, (noisy, clean) in enumerate(train_loader):
    output = model(noisy)
    loss = criterion(output, clean) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Xóa cache**:
```python
import torch
torch.cuda.empty_cache()  # Xóa bộ nhớ không dùng
```

---

### 2. Loss không giảm

**Dấu hiệu**:
```
Epoch 1:  Loss: 0.8
Epoch 10: Loss: 0.8  ← Không giảm
Epoch 20: Loss: 0.8  ← Vẫn không giảm
```

**Nguyên nhân có thể**:

**Learning rate quá cao**:
```
→ Loss nhảy lung tung, không giảm
→ Giải pháp: Giảm learning rate (0.001 → 0.0001)
```

**Learning rate quá thấp**:
```
→ Loss giảm quá chậm, gần như không giảm
→ Giải pháp: Tăng learning rate (0.00001 → 0.0001)
```

**Model không được khởi tạo đúng**:
```python
# Kiểm tra model có weights không
for name, param in model.named_parameters():
    print(name, param.data)
    # Nếu tất cả = 0 → Model chưa được khởi tạo!
```

**Dữ liệu có vấn đề**:
```python
# Kiểm tra dữ liệu
for noisy, clean in train_loader:
    print("Noisy range:", noisy.min(), noisy.max())
    print("Clean range:", clean.min(), clean.max())
    # Nếu range không hợp lý → Dữ liệu có vấn đề
    break
```

**Cách xử lý**:
1. Thử learning rate khác (0.0001, 0.001, 0.00001)
2. Kiểm tra model initialization
3. Kiểm tra dữ liệu
4. Kiểm tra loss function

---

### 3. Loss = NaN (Not a Number)

**Lỗi**:
```
Loss: nan
```

**Nguyên nhân**:
```
- Learning rate quá cao → Gradient explode
- Division by zero
- Log của số âm
```

**Cách xử lý**:

**Giảm learning rate**:
```python
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Thay vì 0.001
```

**Gradient clipping**:
```python
# Giới hạn gradient
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Kiểm tra dữ liệu**:
```python
# Đảm bảo không có NaN trong dữ liệu
assert not torch.isnan(noisy).any()
assert not torch.isnan(clean).any()
```

---

### 4. Loss nhảy lung tung

**Dấu hiệu**:
```
Epoch 1:  Loss: 0.8
Epoch 2:  Loss: 1.2  ← Nhảy lên
Epoch 3:  Loss: 0.5  ← Nhảy xuống
Epoch 4:  Loss: 1.0  ← Nhảy lên
```

**Nguyên nhân**:
- Learning rate quá cao

**Cách xử lý**:
```python
# Giảm learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Thay vì 0.001

# Hoặc dùng learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

---

### 5. Overfitting

**Dấu hiệu**:
```
Train Loss: 0.05  ← Rất thấp
Val Loss:   0.3   ← Rất cao
```

**Cách xử lý**:

**Early stopping**:
```python
best_val_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(num_epochs):
    # ... training ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping!")
            break
```

**Data augmentation**:
```python
# Tăng dữ liệu
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # ...
])
```

**Dropout**:
```python
# Thêm dropout vào model
class REDCNN(nn.Module):
    def __init__(self):
        # ...
        self.dropout = nn.Dropout(0.2)
```

**Weight decay**:
```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001  # Regularization
)
```

---

### 6. Training quá chậm

**Dấu hiệu**:
```
1 epoch mất 2 giờ (quá chậm!)
```

**Cách xử lý**:

**Tăng batch size** (nếu GPU cho phép):
```python
batch_size = 32  # Thay vì 8
```

**Dùng GPU**:
```python
if torch.cuda.is_available():
    model = model.cuda()
    noisy = noisy.cuda()
    clean = clean.cuda()
```

**Tăng num_workers**:
```python
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4  # Thay vì 1
)
```

**Mixed precision training**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for noisy, clean in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(noisy)
        loss = criterion(output, clean)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

### 7. Lỗi đọc dữ liệu

**Lỗi**:
```
FileNotFoundError: Cannot find image.png
```

**Cách xử lý**:

**Kiểm tra đường dẫn**:
```python
import os

# Kiểm tra file có tồn tại không
if not os.path.exists("dataset/train/noisy/img_001.png"):
    print("File not found!")
```

**Kiểm tra dataset**:
```python
# Đếm số file
train_files = os.listdir("dataset/train/noisy")
print(f"Number of files: {len(train_files)}")
```

**Xử lý lỗi trong DataLoader**:
```python
def collate_fn(batch):
    # Bỏ qua ảnh lỗi
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=collate_fn
)
```

---

### 8. Model không học được gì

**Dấu hiệu**:
```
Loss không giảm
Ảnh output giống hệt input
```

**Cách xử lý**:

**Kiểm tra model**:
```python
# Test forward pass
test_input = torch.randn(1, 1, 256, 256)
output = model(test_input)
print("Output shape:", output.shape)
print("Output range:", output.min(), output.max())
```

**Kiểm tra gradient**:
```python
# Xem gradient có được tính không
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.mean()}")
    else:
        print(f"{name}: No gradient!")  # ← Vấn đề!
```

**Kiểm tra loss function**:
```python
# Test loss function
output = torch.randn(1, 1, 256, 256)
target = torch.randn(1, 1, 256, 256)
loss = criterion(output, target)
print("Loss:", loss.item())  # Phải có giá trị hợp lý
```

---

## 🔧 Checklist xử lý lỗi

### Khi gặp lỗi

```
□ 1. Đọc kỹ thông báo lỗi
□ 2. Tìm nguyên nhân
□ 3. Thử giải pháp đơn giản trước
□ 4. Kiểm tra từng phần (data, model, training)
□ 5. Tìm trên Google/Stack Overflow
□ 6. Hỏi người có kinh nghiệm
```

---

## 💡 Ví dụ xử lý lỗi

### Scenario 1: Out of Memory

```
Lỗi: CUDA out of memory
Nguyên nhân: Batch size = 32, GPU chỉ có 6GB
Giải pháp: Giảm batch size xuống 8
Kết quả: Training chạy được! ✅
```

### Scenario 2: Loss không giảm

```
Lỗi: Loss = 0.8, không giảm
Nguyên nhân: Learning rate = 0.00001 (quá thấp)
Giải pháp: Tăng lên 0.0001
Kết quả: Loss bắt đầu giảm! ✅
```

### Scenario 3: Overfitting

```
Lỗi: Train loss = 0.05, Val loss = 0.3
Nguyên nhân: Model quá phức tạp, ít dữ liệu
Giải pháp: Early stopping + Data augmentation
Kết quả: Val loss giảm, không overfitting! ✅
```

---

## 🎯 Tóm tắt

**Các lỗi thường gặp**:

```
1. Out of Memory
   → Giảm batch size, kích thước ảnh

2. Loss không giảm
   → Điều chỉnh learning rate, kiểm tra model/data

3. Loss = NaN
   → Giảm learning rate, gradient clipping

4. Loss nhảy lung tung
   → Giảm learning rate

5. Overfitting
   → Early stopping, data augmentation, dropout

6. Training quá chậm
   → Dùng GPU, tăng batch size, mixed precision

7. Lỗi đọc dữ liệu
   → Kiểm tra đường dẫn, xử lý lỗi trong DataLoader

8. Model không học được
   → Kiểm tra model, gradient, loss function
```

**Cách xử lý**:
- ✅ Đọc kỹ lỗi
- ✅ Tìm nguyên nhân
- ✅ Thử giải pháp từ đơn giản đến phức tạp
- ✅ Kiểm tra từng phần

**Kết quả**: Xử lý được lỗi = Training thành công! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Lỗi out of memory xử lý như thế nào?
2. Loss không giảm có thể do nguyên nhân gì?
3. Overfitting là gì? Cách xử lý?
4. Làm sao biết model có đang học không?
5. Khi gặp lỗi, nên làm gì trước?

**Hoạt động**: 
- Xem log lỗi thật
- Thực hành xử lý từng loại lỗi
- Tìm hiểu thêm các lỗi khác

---

## 🎓 Kết thúc phần Training

**Chúc mừng!** Bạn đã học xong cách train RED-CNN! 🎉

**Bạn đã biết**:
- ✅ Chuẩn bị môi trường và dữ liệu
- ✅ Các tham số training quan trọng
- ✅ Cách chạy training
- ✅ Theo dõi và đánh giá training
- ✅ Xử lý lỗi thường gặp

**Tiếp theo bạn có thể**:
- 🚀 Bắt đầu train model của mình
- 📊 Thử nghiệm với các tham số khác nhau
- 🔬 Tối ưu hóa model
- 🎯 Deploy model vào thực tế

**Chúc bạn thành công!** 🚀

---

➡️ **Quay lại**: `DANH_SACH_CAU_HOI.md` - Xem lại toàn bộ lộ trình

