# Hướng dẫn chi tiết về RED-CNN cho người mới bắt đầu

## 📋 Mục lục
1. [Giới thiệu vấn đề](#1-giới-thiệu-vấn-đề)
2. [Kiến trúc RED-CNN](#2-kiến-trúc-red-cnn)
3. [Cách mô hình xử lý ảnh](#3-cách-mô-hình-xử-lý-ảnh)
4. [Quá trình training](#4-quá-trình-training)
5. [Thực hành](#5-thực-hành)

---

## 1. Giới thiệu vấn đề

### 1.1 CT Scan và vấn đề bức xạ
- **CT Scan** (Computed Tomography) là kỹ thuật chụp X-quang để tạo ảnh 3D bên trong cơ thể
- **Vấn đề**: Bức xạ từ CT có thể gây hại cho sức khỏe
- **Giải pháp**: Giảm liều bức xạ → Nhưng dẫn đến ảnh bị **nhiễu** (noisy)

### 1.2 Mục tiêu của RED-CNN
**Khử nhiễu ảnh CT liều thấp** (Low-Dose CT Denoising)

```
Ảnh CT liều thấp (nhiều nhiễu)  →  [RED-CNN]  →  Ảnh sạch (ít nhiễu)
        INPUT                                          OUTPUT
```

**Input**: Ảnh CT chất lượng thấp (low-dose CT)  
**Output**: Ảnh CT chất lượng cao (denoised CT)  
**Mục tiêu**: Giữ nguyên thông tin y khoa quan trọng, loại bỏ nhiễu

---

## 2. Kiến trúc RED-CNN

### 2.1 Tên gọi
**RED-CNN** = **R**esidual **E**ncoder-**D**ecoder **CNN**

### 2.2 Cấu trúc tổng quan

```
INPUT (1 channel, 128x128)
    ↓
┌─────────────────────────┐
│   ENCODER (Nén thông tin)   │
│   - Conv1 → ReLU            │
│   - Conv2 → ReLU            │ ──────┐ Skip connection 1
│   - Conv3 → ReLU            │       │
│   - Conv4 → ReLU            │ ──┐   │ Skip connection 2
│   - Conv5 → ReLU            │   │   │
└─────────────────────────┘   │   │
    ↓                         │   │
┌─────────────────────────┐   │   │
│  DECODER (Khôi phục ảnh)    │   │   │
│   - TransConv1              │   │   │
│   - TransConv2 ← ← ← ← ← ←─┘   │   │ (Add residual 2)
│   - TransConv3              │       │
│   - TransConv4 ← ← ← ← ← ← ← ← ←─┘   (Add residual 1)
│   - TransConv5              │
└─────────────────────────┘
    ↓
OUTPUT (1 channel, 128x128)
```

### 2.3 Các thành phần chính

#### A. Convolutional Layers (Conv)
```python
self.conv1 = nn.Conv2d(1, 96, kernel_size=5, stride=1, padding=0)
```
- **Input channels**: 1 (ảnh grayscale)
- **Output channels**: 96 features
- **Kernel size**: 5×5 (cửa sổ quan sát)
- **Stride**: 1 (dịch chuyển 1 pixel)
- **Padding**: 0 (không thêm viền)

**Hiệu ứng**: Mỗi Conv layer **giảm kích thước** ảnh đi 4 pixels (do kernel 5×5 với padding=0)

#### B. Transpose Convolutional Layers (TransConv)
```python
self.tconv1 = nn.ConvTranspose2d(96, 96, kernel_size=5, stride=1, padding=0)
```
- Ngược lại với Conv: **tăng kích thước** ảnh
- Mỗi TransConv layer tăng kích thước lên 4 pixels

#### C. ReLU Activation
```python
self.relu = nn.ReLU()
```
- **Công thức**: `ReLU(x) = max(0, x)`
- **Mục đích**: Tạo tính phi tuyến, giúp mô hình học được các pattern phức tạp

#### D. Skip Connections (Residual Connections)
```python
# Trong forward():
residual_1 = x  # Lưu input ban đầu
# ... qua nhiều layers ...
out += residual_1  # Cộng lại với input
```

**Ý nghĩa**: 
- Giữ lại thông tin chi tiết từ các lớp trước
- Giúp gradient flow tốt hơn (tránh vanishing gradient)
- Network chỉ cần học **phần khác biệt** (residual) thay vì toàn bộ mapping

---

## 3. Cách mô hình xử lý ảnh

### 3.1 Forward Pass - Luồng xử lý

#### Bước 1: Encoder (Nén thông tin)
```python
# INPUT: (batch_size, 1, 128, 128)
residual_1 = x  # Lưu ảnh gốc

out = self.relu(self.conv1(x))    # → (batch, 96, 124, 124)
out = self.relu(self.conv2(out))  # → (batch, 96, 120, 120)
residual_2 = out  # Lưu feature map này

out = self.relu(self.conv3(out))  # → (batch, 96, 116, 116)
out = self.relu(self.conv4(out))  # → (batch, 96, 112, 112)
residual_3 = out  # Lưu feature map này

out = self.relu(self.conv5(out))  # → (batch, 96, 108, 108) - Bottleneck
```

**Ý nghĩa**:
- Mỗi Conv layer trích xuất **features** (edges, textures, patterns)
- Càng sâu, features càng abstract (trừu tượng)
- Bottleneck (108×108): Representation nén nhất

#### Bước 2: Decoder (Khôi phục ảnh)
```python
# Từ bottleneck, reconstruct ảnh
out = self.tconv1(out)           # → (batch, 96, 112, 112)
out += residual_3                # Thêm thông tin chi tiết
out = self.tconv2(self.relu(out)) # → (batch, 96, 116, 116)
out = self.tconv3(self.relu(out)) # → (batch, 96, 120, 120)
out += residual_2                # Thêm thông tin chi tiết
out = self.tconv4(self.relu(out)) # → (batch, 96, 124, 124)
out = self.tconv5(self.relu(out)) # → (batch, 1, 128, 128)
out += residual_1                # Thêm ảnh gốc

# OUTPUT: (batch_size, 1, 128, 128)
```

**Ý nghĩa**:
- TransConv tăng resolution, reconstruct ảnh
- Skip connections thêm lại chi tiết đã mất trong encoding
- Output cuối cùng: **Ảnh đã khử nhiễu**

### 3.2 Ví dụ cụ thể

Giả sử input là ảnh CT 128×128 pixels, nhiều nhiễu:

```
Original Image (Low-dose CT)     Feature Map (Encoder)      Denoised Output
       🔲🔳🔲                           📊📈📊                      🔲🔲🔲
     🔳◼️🔲🔳     →  [ENCODER]  →     📈📊📈      →  [DECODER]  →  🔲◼️🔲
       🔲🔳🔲                           📊📈📊                      🔲🔲🔲
     (nhiễu nhiều)              (features trừu tượng)        (sạch hơn)
```

**Quá trình**:
1. Conv layers phát hiện noise patterns và anatomical structures
2. Bottleneck layer chứa representation quan trọng nhất
3. TransConv layers reconstruct ảnh, skip connections giữ lại chi tiết

---

## 4. Quá trình Training

### 4.1 Loss Function
```python
self.criterion = nn.MSELoss()
```

**Mean Squared Error (MSE)**:
```
MSE = (1/N) * Σ(predicted_pixel - target_pixel)²
```

- **predicted_pixel**: Output từ RED-CNN
- **target_pixel**: Ảnh CT liều cao (ground truth)
- **Mục tiêu**: Minimize MSE → Output càng gần ground truth càng tốt

### 4.2 Optimizer
```python
optimizer: Adam
learning_rate: 9.58e-05
beta1: 0.9
beta2: 0.999
```

**Adam Optimizer**: Thuật toán tối ưu adaptive, tự động điều chỉnh learning rate

### 4.3 Training Loop

```python
def train_step(self, batch):
    # 1. Lấy input và target
    inputs = batch["x"]   # Low-dose CT
    targets = batch["y"]  # High-dose CT (ground truth)
    
    # 2. Forward pass: Chạy qua model
    outputs = self.model(inputs)
    
    # 3. Tính loss
    loss = self.criterion(outputs, targets)  # MSE
    
    # 4. Backward pass: Tính gradient
    self.optimizer.zero_grad()  # Reset gradient
    loss.backward()             # Backpropagation
    
    # 5. Update weights
    self.optimizer.step()
```

### 4.4 Training Flow

```
┌──────────────────────────────────────────────┐
│  1. Load batch (73 images, 128×128)          │
│     - inputs: Low-dose CT                    │
│     - targets: High-dose CT                  │
└────────────────┬─────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│  2. Forward pass                             │
│     outputs = model(inputs)                  │
└────────────────┬─────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│  3. Calculate Loss                           │
│     loss = MSE(outputs, targets)             │
└────────────────┬─────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│  4. Backward pass (Gradient descent)         │
│     loss.backward()                          │
│     optimizer.step()                         │
└────────────────┬─────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────┐
│  5. Validation (sau 1000 iterations)         │
│     - Evaluate on validation set             │
│     - Calculate metrics: SSIM, PSNR, RMSE    │
│     - Save best model                        │
└──────────────────────────────────────────────┘
```

### 4.5 Metrics

**SSIM (Structural Similarity Index)**:
- Đo độ tương đồng về cấu trúc
- Giá trị: 0 → 1 (1 là giống nhất)

**PSNR (Peak Signal-to-Noise Ratio)**:
- Đo tỷ lệ tín hiệu/nhiễu
- Đơn vị: dB (càng cao càng tốt)

**RMSE (Root Mean Squared Error)**:
- Căn bậc 2 của MSE
- Đơn vị: Hounsfield Units (HU)

---

## 5. Thực hành

### 5.1 Cấu trúc thư mục
```
ldct-benchmark/
├── configs/
│   └── redcnn.yaml          # Config file
├── ldctbench/
│   ├── methods/
│   │   ├── base.py          # Base Trainer class
│   │   └── redcnn/
│   │       ├── network.py   # Kiến trúc RED-CNN
│   │       ├── Trainer.py   # Training logic
│   │       └── argparser.py # Arguments
│   ├── data/
│   │   └── LDCTMayo.py      # Dataset loader
│   └── utils/               # Utility functions
└── app.py                   # Main entry point
```

### 5.2 Cách chạy training

#### Bước 1: Chuẩn bị môi trường
```bash
# Install dependencies
pip install -e .

# Prepare data
# (Follow instructions in documentation)
```

#### Bước 2: Chạy training
```bash
python app.py --config configs/redcnn.yaml
```

#### Bước 3: Theo dõi training
- WandB dashboard sẽ log:
  - Training loss
  - Validation metrics (SSIM, PSNR, RMSE)
  - Sample images (low-dose, prediction, high-dose)

### 5.3 Hiểu config file

```yaml
# configs/redcnn.yaml
lr: 9.583417460320728e-05    # Learning rate
mbs: 73                       # Mini-batch size (số ảnh/batch)
patchsize: 128                # Kích thước patch (128×128)
max_iterations: 92994         # Tổng số iterations
iterations_before_val: 1000   # Validate sau mỗi 1000 iterations
optimizer: adam               # Optimizer type
adam_b1: 0.9                  # Adam beta1
adam_b2: 0.999                # Adam beta2
cuda: true                    # Sử dụng GPU
devices: 0                    # GPU ID
num_workers: 8                # Số workers load data
seed: 1339                    # Random seed
```

### 5.4 Đọc code - Workflow

#### network.py - Định nghĩa model
```python
class Model(nn.Module):
    def __init__(self, args, out_ch=96):
        # Khởi tạo các layers
        self.conv1 = ...
        self.tconv1 = ...
        
    def forward(self, x):
        # Định nghĩa luồng xử lý
        out = self.conv1(x)
        ...
        return out
```

#### Trainer.py - Training logic
```python
class Trainer(BaseTrainer):
    def __init__(self, args, device):
        # Setup model, optimizer, criterion
        self.model = Model(args)
        self.optimizer = setup_optimizer(...)
        self.criterion = nn.MSELoss()
```

#### base.py - Training loop
```python
def fit(self):
    while iteration < max_iterations:
        self.train()      # Train 1 epoch
        self.validate()   # Validate
        self.log()        # Log metrics, save checkpoint
```

### 5.5 Checkpoint và Inference

**Lưu model**:
```python
# Model tốt nhất được lưu tự động
checkpoint_path = "wandb/run-xxx/best_SSIM.pt"
```

**Load model để inference**:
```python
# Load checkpoint
checkpoint = torch.load("best_SSIM.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Inference
model.eval()
with torch.no_grad():
    denoised = model(low_dose_image)
```

---

## 6. Câu hỏi thường gặp

### Q1: Tại sao dùng skip connections?
**A**: Giúp gradient flow tốt hơn, giữ lại chi tiết từ input, model chỉ cần học phần residual.

### Q2: Tại sao dùng MSE Loss?
**A**: MSE đơn giản, hiệu quả cho image reconstruction, optimize pixel-wise differences.

### Q3: Patchsize 128 có ý nghĩa gì?
**A**: Chia ảnh lớn thành patches 128×128 để training, tiết kiệm memory.

### Q4: Tại sao không có ReLU cuối cùng?
**A**: Để cho phép output có giá trị âm (do data được normalize về zero-mean).

### Q5: 96 channels trong Conv layers có nghĩa gì?
**A**: Số lượng features khác nhau mà model học được (edges, textures, patterns...).

---

## 7. Tài liệu tham khảo

📄 **Paper gốc**:  
H. Chen et al., "Low-dose CT with a residual encoder-decoder convolutional neural network," IEEE TMI, 2017.

📂 **Code repository**:  
https://github.com/eeulig/ldct-benchmark

📖 **Documentation**:  
https://eeulig.github.io/ldct-benchmark/

---

## 8. Bài tập thực hành

### Bài 1: Hiểu kiến trúc
- Vẽ sơ đồ chi tiết của RED-CNN trên giấy
- Tính toán kích thước output của từng layer

### Bài 2: Đọc code
- Đọc file `network.py` và comment từng dòng
- Trace forward pass với 1 ví dụ cụ thể

### Bài 3: Experiment
- Thay đổi số channels từ 96 → 48, quan sát kết quả
- Thay đổi learning rate, so sánh convergence

### Bài 4: Visualization
- Visualize feature maps ở các layer khác nhau
- Plot loss curve, metric curves

---

**Chúc bạn học tốt! 🚀**

Nếu có thắc mắc, hãy đọc lại từng phần và thực hành code.
