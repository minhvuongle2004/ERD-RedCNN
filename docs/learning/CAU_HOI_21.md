# Câu 21: Cách chạy training như thế nào?

## 🎯 Mục tiêu
Biết được quy trình chạy training RED-CNN từ đầu đến cuối

---

## 📖 Trả lời

### Chạy training = Bắt đầu dạy model học

**Giống như bắt đầu khóa học**:
```
1. Chuẩn bị sách vở (dataset)
2. Vào lớp (load model)
3. Học bài (train)
4. Làm bài kiểm tra (validation)
5. Kết thúc khóa học (save model)
```

**Áp dụng vào RED-CNN**:
```
1. Chuẩn bị dataset
2. Load model RED-CNN
3. Chạy training loop
4. Validation mỗi epoch
5. Lưu model tốt nhất
```

---

## 🚀 Quy trình chạy training

### Bước 1: Kiểm tra mọi thứ đã sẵn sàng

**Checklist**:
```
□ Dataset đã chuẩn bị xong
□ Code RED-CNN có sẵn
□ GPU hoạt động
□ Thư viện đã cài đặt
□ Tham số đã cấu hình
```

**Test nhanh**:
```python
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

---

### Bước 2: Load Dataset

**Tạo DataLoader**:
```python
from torch.utils.data import DataLoader, Dataset

# Tạo dataset
train_dataset = CTDataset("train/")
val_dataset = CTDataset("val/")

# Tạo dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,  # Xáo trộn ảnh
    num_workers=4  # Số thread đọc dữ liệu
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)
```

**Kiểm tra**:
```python
# Test đọc được dữ liệu
for batch in train_loader:
    noisy, clean = batch
    print("Batch shape:", noisy.shape, clean.shape)
    break  # Chỉ test 1 batch
```

---

### Bước 3: Khởi tạo Model

**Tạo model RED-CNN**:
```python
from model.red_cnn import REDCNN

# Khởi tạo model
model = REDCNN()

# Chuyển lên GPU (nếu có)
if torch.cuda.is_available():
    model = model.cuda()
    print("Model moved to GPU")
```

**Kiểm tra**:
```python
# Test forward pass
test_input = torch.randn(1, 1, 256, 256)
if torch.cuda.is_available():
    test_input = test_input.cuda()

output = model(test_input)
print("Output shape:", output.shape)
```

---

### Bước 4: Cấu hình Training

**Setup Loss, Optimizer**:
```python
import torch.nn as nn
import torch.optim as optim

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001,
    weight_decay=0.0001
)

# Learning rate scheduler (tùy chọn)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1  # Giảm LR 10 lần sau mỗi 30 epochs
)
```

---

### Bước 5: Training Loop

**Vòng lặp chính**:
```python
num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ===== TRAINING =====
    model.train()  # Chế độ training
    train_loss = 0.0
    
    for batch_idx, (noisy, clean) in enumerate(train_loader):
        # Chuyển lên GPU
        if torch.cuda.is_available():
            noisy = noisy.cuda()
            clean = clean.cuda()
        
        # Forward pass
        output = model(noisy)
        loss = criterion(output, clean)
        
        # Backward pass
        optimizer.zero_grad()  # Xóa gradient cũ
        loss.backward()         # Tính gradient
        optimizer.step()        # Cập nhật weights
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # ===== VALIDATION =====
    model.eval()  # Chế độ evaluation
    val_loss = 0.0
    
    with torch.no_grad():  # Không tính gradient khi validation
        for noisy, clean in val_loader:
            if torch.cuda.is_available():
                noisy = noisy.cuda()
                clean = clean.cuda()
            
            output = model(noisy)
            loss = criterion(output, clean)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # ===== LOGGING =====
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # ===== SAVE BEST MODEL =====
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")
    
    # Update learning rate
    scheduler.step()
```

---

### Bước 6: Lưu Model

**Lưu model tốt nhất**:
```python
# Đã lưu trong training loop
torch.save(model.state_dict(), "best_model.pth")

# Hoặc lưu toàn bộ (bao gồm optimizer, epoch...)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': val_loss,
}
torch.save(checkpoint, "checkpoint.pth")
```

---

## 📝 Script Training hoàn chỉnh (Minh họa)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.red_cnn import REDCNN
from data.dataset import CTDataset

# ===== CONFIG =====
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

# ===== DATASET =====
train_dataset = CTDataset(TRAIN_DIR)
val_dataset = CTDataset(VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== MODEL =====
model = REDCNN()
if torch.cuda.is_available():
    model = model.cuda()

# ===== LOSS & OPTIMIZER =====
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ===== TRAINING =====
best_val_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for noisy, clean in train_loader:
        if torch.cuda.is_available():
            noisy, clean = noisy.cuda(), clean.cuda()
        
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean in val_loader:
            if torch.cuda.is_available():
                noisy, clean = noisy.cuda(), clean.cuda()
            output = model(noisy)
            loss = criterion(output, clean)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Log
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
    
    # Save best
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model!")

print("Training completed!")
```

---

## 🖥️ Chạy Training

### Cách 1: Chạy trực tiếp

```bash
python train.py
```

### Cách 2: Chạy với tham số

```bash
python train.py --batch_size 16 --lr 0.0001 --epochs 100
```

### Cách 3: Chạy trên GPU

```bash
# Tự động detect GPU
python train.py

# Hoặc chỉ định GPU
CUDA_VISIBLE_DEVICES=0 python train.py
```

---

## 📊 Theo dõi Training

### Xem log trong terminal

```
Epoch 1/100: Train Loss: 0.8234, Val Loss: 0.7123
Epoch 2/100: Train Loss: 0.6543, Val Loss: 0.6234
Epoch 3/100: Train Loss: 0.5432, Val Loss: 0.5123
...
```

### Dùng TensorBoard (Tùy chọn)

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

# Trong training loop
writer.add_scalar('Loss/Train', train_loss, epoch)
writer.add_scalar('Loss/Val', val_loss, epoch)

# Xem: tensorboard --logdir runs
```

---

## ⏱️ Thời gian Training

**Ước tính**:
```
Dataset: 10,000 ảnh
Batch size: 16
GPU: RTX 3060

1 epoch: ~5-10 phút
100 epochs: ~8-16 giờ
```

**Tùy thuộc**:
- Kích thước dataset
- Batch size
- GPU mạnh/yếu
- Kích thước ảnh

---

## 🎯 Tóm tắt

**Quy trình chạy training**:

```
1. ✅ Kiểm tra sẵn sàng
2. ✅ Load dataset
3. ✅ Khởi tạo model
4. ✅ Cấu hình loss, optimizer
5. ✅ Training loop (epochs)
   - Training phase
   - Validation phase
   - Lưu model tốt nhất
6. ✅ Lưu model cuối cùng
```

**Các bước chính**:
- Load dữ liệu
- Forward pass (dự đoán)
- Tính loss
- Backward pass (tính gradient)
- Update weights
- Validation
- Lưu model

**Kết quả**: Model đã được train! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Quy trình chạy training có mấy bước?
2. Training loop làm gì?
3. Tại sao cần validation?
4. Khi nào lưu model?
5. Làm sao biết training đang chạy?

**Hoạt động**: 
- Xem code training thật
- Chạy thử training với dataset nhỏ
- Quan sát loss giảm dần

---

➡️ **Tiếp theo**: `CAU_HOI_22.md` - Làm sao biết training đang chạy tốt?

