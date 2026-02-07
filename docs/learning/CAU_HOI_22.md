# Câu 22: Làm sao biết training đang chạy tốt?

## 🎯 Mục tiêu
Biết cách theo dõi và đánh giá quá trình training

---

## 📖 Trả lời

### Làm sao biết training tốt?

**Giống như kiểm tra học sinh**:
```
Học sinh học bài:
→ Làm bài kiểm tra
→ Xem điểm số
→ Điểm cao = Học tốt ✅
```

**Áp dụng vào training**:
```
Model đang train:
→ Xem loss
→ Xem metrics
→ Loss giảm, metrics tăng = Train tốt ✅
```

---

## 📊 Các dấu hiệu training tốt

### 1. Loss giảm dần

**Dấu hiệu tốt**:
```
Epoch 1:  Train Loss: 0.8234, Val Loss: 0.7123
Epoch 10: Train Loss: 0.5432, Val Loss: 0.5123  ← Giảm ✅
Epoch 20: Train Loss: 0.3234, Val Loss: 0.3123  ← Giảm ✅
Epoch 50: Train Loss: 0.1234, Val Loss: 0.1123  ← Giảm ✅
Epoch 100: Train Loss: 0.0534, Val Loss: 0.0523 ← Giảm ✅
```

**Biểu đồ loss**:
```
Loss
Cao ^
    |  *                    ← Ban đầu
    |    *
    |      *                ← Đang giảm ✅
    |        *
    |          *
    |            *          ← Tiếp tục giảm ✅
    |              *
Thấp|                *______ ← Ổn định ✅
    └──────────────────────────> Epochs
```

**Dấu hiệu xấu**:
```
Epoch 1:  Loss: 0.8
Epoch 10: Loss: 0.8  ← Không giảm ❌
Epoch 20: Loss: 0.8  ← Vẫn không giảm ❌

Hoặc:
Epoch 1:  Loss: 0.8
Epoch 10: Loss: 1.2  ← Tăng lên ❌
Epoch 20: Loss: 1.5  ← Tiếp tục tăng ❌
```

---

### 2. Validation Loss gần Training Loss

**Dấu hiệu tốt**:
```
Epoch 50:
Train Loss: 0.1234
Val Loss:   0.1256  ← Gần nhau ✅

→ Model không bị overfitting
```

**Dấu hiệu xấu (Overfitting)**:
```
Epoch 50:
Train Loss: 0.0534  ← Rất thấp
Val Loss:   0.3123  ← Rất cao ❌

→ Model học thuộc lòng training data
→ Không biết xử lý dữ liệu mới
```

**Biểu đồ overfitting**:
```
Loss
    |  * (train)
    |    *
    |      *
    |        *
    |          *
    |            *
    |              *
    |                *______ (train)
    |
    |  * (val)
    |    *
    |      *
    |        *
    |          *
    |            *
    |              *
    |                *
    |                  *     ← Val loss không giảm ❌
    └──────────────────────────> Epochs
```

---

### 3. Metrics tăng dần

**PSNR (Peak Signal-to-Noise Ratio)**:
```
Epoch 1:  PSNR: 25.3 dB
Epoch 10: PSNR: 28.5 dB  ← Tăng ✅
Epoch 20: PSNR: 31.2 dB  ← Tăng ✅
Epoch 50: PSNR: 33.8 dB  ← Tăng ✅
```

**SSIM (Structural Similarity)**:
```
Epoch 1:  SSIM: 0.75
Epoch 10: SSIM: 0.82  ← Tăng ✅
Epoch 20: SSIM: 0.87  ← Tăng ✅
Epoch 50: SSIM: 0.91  ← Tăng ✅
```

**Dấu hiệu tốt**: Metrics tăng = Ảnh output đẹp hơn ✅

---

### 4. Ảnh kết quả đẹp hơn

**Quan sát ảnh output**:

**Epoch 1**:
```
Input:   🖼️ ▓▒░ Nhiễu nhiều
Output:  🖼️ ▓░ Vẫn nhiều nhiễu ❌
```

**Epoch 10**:
```
Input:   🖼️ ▓▒░ Nhiễu
Output:  🖼️ ░ Ít nhiễu hơn ⚠️
```

**Epoch 50**:
```
Input:   🖼️ ▓▒░ Nhiễu
Output:  🖼️ ✨ Gần sạch ✅
```

**Epoch 100**:
```
Input:   🖼️ ▓▒░ Nhiễu
Output:  🖼️ ✨✨ Rất sạch ✅✅
```

---

## 🔍 Cách theo dõi

### 1. Xem log trong terminal

```
Epoch 1/100
Train Loss: 0.8234, Val Loss: 0.7123, PSNR: 25.3, SSIM: 0.75
Time: 5m 23s

Epoch 2/100
Train Loss: 0.6543, Val Loss: 0.6234, PSNR: 27.1, SSIM: 0.78
Time: 5m 18s

...
```

**Quan sát**:
- ✅ Loss có giảm không?
- ✅ Val loss có gần train loss không?
- ✅ Metrics có tăng không?

---

### 2. Vẽ biểu đồ

**Dùng matplotlib**:
```python
import matplotlib.pyplot as plt

epochs = [1, 2, 3, ...]
train_loss = [0.8, 0.6, 0.4, ...]
val_loss = [0.7, 0.5, 0.3, ...]

plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curve.png')
```

**Kết quả**: Biểu đồ loss giảm dần ✅

---

### 3. Xem ảnh kết quả

**Lưu ảnh mỗi epoch**:
```python
# Trong validation
if epoch % 10 == 0:  # Mỗi 10 epochs
    # Lấy 1 ảnh mẫu
    sample_noisy = noisy[0]
    sample_clean = clean[0]
    sample_output = output[0]
    
    # Vẽ 3 ảnh cạnh nhau
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(sample_noisy, cmap='gray')
    axes[0].set_title('Noisy')
    axes[1].imshow(sample_output, cmap='gray')
    axes[1].set_title('Output')
    axes[2].imshow(sample_clean, cmap='gray')
    axes[2].set_title('Clean')
    
    plt.savefig(f'results/epoch_{epoch}.png')
```

**Quan sát**: Ảnh output có đẹp hơn không?

---

## ⚠️ Dấu hiệu training không tốt

### 1. Loss không giảm

```
Epoch 1:  Loss: 0.8
Epoch 10: Loss: 0.8
Epoch 20: Loss: 0.8

→ Model không học được!
```

**Nguyên nhân có thể**:
- Learning rate quá cao/thấp
- Model architecture sai
- Dữ liệu có vấn đề

**Cách xử lý**:
- Giảm learning rate
- Kiểm tra model
- Kiểm tra dữ liệu

---

### 2. Loss nhảy lung tung

```
Epoch 1:  Loss: 0.8
Epoch 2:  Loss: 1.2  ← Nhảy lên
Epoch 3:  Loss: 0.5  ← Nhảy xuống
Epoch 4:  Loss: 1.0  ← Nhảy lên
```

**Nguyên nhân**:
- Learning rate quá cao

**Cách xử lý**:
- Giảm learning rate (0.001 → 0.0001)

---

### 3. Overfitting

```
Train Loss: 0.05  ← Rất thấp
Val Loss:   0.3   ← Rất cao

→ Model học thuộc lòng
```

**Cách xử lý**:
- Thêm dữ liệu
- Early stopping
- Dropout
- Regularization

---

### 4. Underfitting

```
Train Loss: 0.5  ← Vẫn cao
Val Loss:   0.5  ← Cũng cao

→ Model chưa học đủ
```

**Cách xử lý**:
- Train thêm epochs
- Tăng model capacity
- Giảm regularization

---

## 📈 Metrics tốt cho RED-CNN

### Sau khi train xong

```
PSNR:  > 30 dB  → Tốt ✅
       > 33 dB  → Rất tốt ✅✅

SSIM:  > 0.85   → Tốt ✅
       > 0.90   → Rất tốt ✅✅
```

---

## 🎯 Tóm tắt

**Dấu hiệu training tốt**:

```
1. ✅ Loss giảm dần (train và val)
2. ✅ Val loss gần train loss (không overfitting)
3. ✅ Metrics tăng dần (PSNR, SSIM)
4. ✅ Ảnh output đẹp hơn qua các epochs
```

**Dấu hiệu training không tốt**:

```
1. ❌ Loss không giảm
2. ❌ Loss nhảy lung tung
3. ❌ Overfitting (val loss cao, train loss thấp)
4. ❌ Underfitting (cả 2 loss đều cao)
```

**Cách theo dõi**:
- ✅ Xem log trong terminal
- ✅ Vẽ biểu đồ loss
- ✅ Xem ảnh kết quả
- ✅ Tính metrics (PSNR, SSIM)

**Kết quả**: Training tốt khi loss giảm, metrics tăng, ảnh đẹp! 🎉

---

## ✍️ Kiểm tra hiểu bài

**Hỏi sinh viên**:
1. Làm sao biết training đang tốt?
2. Overfitting là gì? Dấu hiệu?
3. Tại sao cần xem cả train loss và val loss?
4. Metrics nào quan trọng?
5. Làm sao theo dõi training?

**Hoạt động**: 
- Xem log training thật
- Vẽ biểu đồ loss
- So sánh ảnh ở các epochs khác nhau

---

➡️ **Tiếp theo**: `CAU_HOI_23.md` - Xử lý lỗi khi training như thế nào?

