# Outline hướng dẫn RED-CNN cho sinh viên

## 📅 Thời lượng đề xuất: 2-3 giờ

---

## Phần 1: Giới thiệu vấn đề (15 phút)

### Mục tiêu:
- Sinh viên hiểu được vấn đề CT imaging và tại sao cần LDCT denoising

### Nội dung:
1. **CT scan là gì?**
   - Show ví dụ ảnh CT (nếu có)
   - Giải thích cách hoạt động cơ bản

2. **Vấn đề bức xạ**
   - Tại sao cần giảm dose?
   - Trade-off: Low dose → More noise

3. **Giải pháp với Deep Learning**
   - Traditional methods vs Deep learning
   - Ưu điểm của CNN cho image denoising

### Hoạt động:
- ✅ Show ví dụ: Low-dose CT vs High-dose CT side-by-side
- ✅ Q&A: "Theo bạn, AI có thể giúp gì trong trường hợp này?"

---

## Phần 2: Kiến trúc RED-CNN (30 phút)

### Mục tiêu:
- Sinh viên hiểu cấu trúc tổng quan của RED-CNN
- Nắm được các thành phần chính

### Nội dung:

#### 2.1 Giới thiệu Encoder-Decoder architecture (10 phút)
```
Encoder: Nén thông tin, extract features
Decoder: Reconstruct ảnh từ features
```

**Hoạt động**: Vẽ sơ đồ đơn giản trên bảng/giấy

#### 2.2 Skip Connections (10 phút)
- **Tại sao cần skip connections?**
  - Gradient flow
  - Giữ thông tin chi tiết
  - Residual learning

**Demo**: Chạy `python scripts/visualize_redcnn.py` - phần skip connections

#### 2.3 Chi tiết từng layer (10 phút)
- Conv2D: Kernel, stride, padding
- TransConv2D: Upsampling
- ReLU activation
- Dimensions sau mỗi layer

**Tài liệu**: Mở `RED-CNN_QUICK_REFERENCE.md` - phần Layer dimensions

---

## Phần 3: Forward Pass - Xử lý ảnh (25 phút)

### Mục tiêu:
- Hiểu cách một ảnh đi qua network
- Biết shape transformation ở mỗi layer

### Nội dung:

#### 3.1 Live demo (15 phút)
**Chạy script visualization**:
```bash
python scripts/visualize_redcnn.py
```

Quan sát:
- Input shape: (1, 1, 128, 128)
- Sau mỗi Conv: Giảm 4 pixels
- Bottleneck: (1, 96, 108, 108)
- Sau mỗi TransConv: Tăng 4 pixels
- Output shape: (1, 1, 128, 128)

#### 3.2 Hands-on exercise (10 phút)
**Bài tập**: Cho sinh viên tính toán dimensions

```python
# Cho trước:
input_shape = (1, 128, 128)
kernel_size = 5
padding = 0
stride = 1

# Tính output_size sau Conv1?
# Formula: output_size = (input_size - kernel_size + 2*padding) / stride + 1
```

**Đáp án**: output_size = (128 - 5 + 0) / 1 + 1 = 124

---

## Phần 4: Training Process (30 phút)

### Mục tiêu:
- Hiểu training loop
- Nắm được loss function và optimizer

### Nội dung:

#### 4.1 Loss Function - MSE (10 phút)
```python
loss = MSE(predicted, target)
     = (1/N) * Σ(predicted_pixel - target_pixel)²
```

**Giải thích**:
- Tại sao dùng MSE cho image reconstruction?
- Pixel-wise comparison

#### 4.2 Training Loop (15 phút)

**Đọc code cùng nhau**: `ldctbench/methods/redcnn/Trainer.py`

Giải thích từng bước:
1. Forward pass: `outputs = model(inputs)`
2. Calculate loss: `loss = criterion(outputs, targets)`
3. Backward pass: `loss.backward()`
4. Update weights: `optimizer.step()`

**Vẽ diagram**:
```
[Data] → [Model] → [Loss] → [Backward] → [Optimizer] → [Update weights]
   ↑                                                           |
   └───────────────────────────────────────────────────────────┘
                        (Loop continues)
```

#### 4.3 Metrics (5 phút)
- SSIM: Structural similarity
- PSNR: Peak signal-to-noise ratio
- RMSE: Root mean squared error

**Hỏi**: "Metric nào quan trọng nhất cho medical images?" → SSIM

---

## Phần 5: Code Walkthrough (30 phút)

### Mục tiêu:
- Sinh viên đọc được code
- Hiểu cấu trúc project

### Nội dung:

#### 5.1 Project structure (5 phút)
Giải thích cấu trúc thư mục:
```
ldct-benchmark/
├── configs/redcnn.yaml      # Hyperparameters
├── ldctbench/methods/redcnn/
│   ├── network.py            # Model definition
│   ├── Trainer.py            # Training logic
│   └── argparser.py
└── ldctbench/methods/base.py # Base trainer
```

#### 5.2 Đọc network.py (15 phút)

**Mở file**: `ldctbench/methods/redcnn/network.py`

Đi qua từng phần:
1. `__init__`: Khởi tạo layers
2. `forward`: Định nghĩa luồng xử lý

**Exercise**: Cho sinh viên thêm print statements để debug
```python
def forward(self, x):
    print(f"Input shape: {x.shape}")  # Add this
    residual_1 = x
    out = self.relu(self.conv1(x))
    print(f"After Conv1: {out.shape}")  # Add this
    # ...
```

#### 5.3 Đọc Trainer.py (10 phút)

**Highlights**:
- Line 30: `self.criterion = nn.MSELoss()`
- Line 31: `self.model = Model(args).to(self.dev)`
- Line 34: `self.optimizer = setup_optimizer(...)`

---

## Phần 6: Hands-on Practice (30 phút)

### Mục tiêu:
- Sinh viên chạy được training
- Hiểu cách modify hyperparameters

### Nội dung:

#### 6.1 Setup environment (10 phút)
```bash
# Check installation
pip list | grep torch

# Kiểm tra GPU (nếu có)
python -c "import torch; print(torch.cuda.is_available())"
```

#### 6.2 Run training với dryrun (10 phút)
```bash
# Training with small iterations for testing
python app.py --config configs/redcnn.yaml
```

**Quan sát**:
- Terminal output
- WandB dashboard (nếu có)
- Loss curves

#### 6.3 Modify hyperparameters (10 phút)

**Exercise**: Thay đổi config và quan sát

```yaml
# configs/redcnn.yaml
mbs: 32  # Thay đổi từ 73 → 32
lr: 1e-4 # Thay đổi learning rate
```

Chạy lại và so sánh kết quả

---

## Phần 7: Advanced Topics (20 phút - Optional)

### Nội dung:

#### 7.1 Model parameters (5 phút)
Chạy phần count parameters trong visualization script
- Total params: ~X million
- Model size: ~X MB

#### 7.2 Inference (10 phút)

**Demo**: Load checkpoint và inference

```python
# Load model
checkpoint = torch.load("best_SSIM.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# Inference
model.eval()
with torch.no_grad():
    denoised = model(low_dose_image)
```

#### 7.3 Visualization (5 phút)
- Visualize feature maps
- Compare input vs output

---

## Phần 8: Q&A và Bài tập về nhà (10 phút)

### Câu hỏi thường gặp:

**Q1**: Tại sao không dùng ReLU ở layer cuối?  
**A**: Để cho phép output có giá trị âm (zero-mean normalized data)

**Q2**: Skip connections khác gì với concatenation?  
**A**: Skip connections dùng **addition** (+), không phải concatenation

**Q3**: Tại sao dùng patch 128×128 thay vì full image?  
**A**: Tiết kiệm memory, tăng số training samples

### Bài tập về nhà:

#### Bài 1: Implement modifications
```
- Thay đổi out_ch từ 96 → 64
- So sánh performance và training time
```

#### Bài 2: Visualization
```
- Visualize feature maps ở Conv1, Conv3, Conv5
- So sánh features ở các layers khác nhau
```

#### Bài 3: Experiment
```
- Thử remove một skip connection
- Train và so sánh kết quả
- Giải thích tại sao performance thay đổi
```

#### Bài 4: Documentation
```
- Viết một đoạn mô tả (500 từ) về RED-CNN
- Giải thích tại sao RED-CNN phù hợp cho LDCT denoising
```

---

## 📚 Tài liệu tham khảo cho sinh viên

### Phải đọc:
1. ✅ `RED-CNN_TUTORIAL_VI.md` - Tutorial chi tiết
2. ✅ `RED-CNN_QUICK_REFERENCE.md` - Quick reference

### Nên đọc:
3. 📄 RED-CNN paper (Chen et al., 2017)
4. 📖 Project documentation: https://eeulig.github.io/ldct-benchmark/

### Code để explore:
- `ldctbench/methods/redcnn/network.py`
- `ldctbench/methods/redcnn/Trainer.py`
- `ldctbench/methods/base.py`

---

## 💡 Tips cho người hướng dẫn

### Chuẩn bị trước:
- [ ] Test chạy visualization script
- [ ] Prepare ví dụ ảnh CT (low-dose vs high-dose)
- [ ] Setup WandB account (nếu cần)
- [ ] Print out Quick Reference cho sinh viên

### Trong khi hướng dẫn:
- ✅ Khuyến khích sinh viên hỏi ngay khi chưa hiểu
- ✅ Pause thường xuyên để check understanding
- ✅ Cho ví dụ thực tế, dễ hiểu
- ✅ Live coding thay vì chỉ giảng lý thuyết

### Điều chỉnh:
- Nếu sinh viên chưa biết về CNN: Bổ sung 30 phút về CNN basics
- Nếu sinh viên đã biết nhiều về DL: Skip phần basics, focus vào advanced topics

---

## ✅ Checklist đánh giá hiểu bài

Sinh viên cần có khả năng:
- [ ] Giải thích được vấn đề LDCT denoising
- [ ] Vẽ được sơ đồ kiến trúc RED-CNN
- [ ] Tính toán được dimensions sau mỗi layer
- [ ] Giải thích được vai trò của skip connections
- [ ] Đọc và hiểu code trong network.py
- [ ] Chạy được training script
- [ ] Modify được hyperparameters
- [ ] Interpret được training metrics

---

**Chúc buổi hướng dẫn thành công! 🎓**
