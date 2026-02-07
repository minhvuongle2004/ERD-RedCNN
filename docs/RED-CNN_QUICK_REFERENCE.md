# RED-CNN Quick Reference

## 🎯 Mục đích
**Input**: Low-dose CT (nhiễu) → **Output**: Denoised CT (sạch)

## 🏗️ Kiến trúc tóm tắt

```
Input (1×128×128)
  ↓
[Encoder: 5 Conv layers + ReLU]
  - Conv1-5: kernel=5×5, channels=96
  - 3 skip connections được lưu
  ↓
Bottleneck (96×108×108)
  ↓
[Decoder: 5 TransConv layers]
  - TransConv1-5: kernel=5×5
  - Add skip connections
  ↓
Output (1×128×128)
```

## 📐 Layer dimensions

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Conv1 | 1×128×128 | 96×124×124 | kernel=5, stride=1 |
| Conv2 | 96×124×124 | 96×120×120 | kernel=5, stride=1 |
| Conv3 | 96×120×120 | 96×116×116 | kernel=5, stride=1 |
| Conv4 | 96×116×116 | 96×112×112 | kernel=5, stride=1 |
| Conv5 | 96×112×112 | 96×108×108 | kernel=5, stride=1 |
| TConv1 | 96×108×108 | 96×112×112 | kernel=5, stride=1 |
| TConv2 | 96×112×112 | 96×116×116 | kernel=5, stride=1 |
| TConv3 | 96×116×116 | 96×120×120 | kernel=5, stride=1 |
| TConv4 | 96×120×120 | 96×124×124 | kernel=5, stride=1 |
| TConv5 | 96×124×124 | 1×128×128 | kernel=5, stride=1 |

**Skip connections**:
- residual_1: after input (1×128×128) → added before output
- residual_2: after Conv2 (96×120×120) → added after TConv3
- residual_3: after Conv4 (96×112×112) → added after TConv1

## ⚙️ Training config

```yaml
Loss: MSELoss
Optimizer: Adam
Learning rate: 9.58e-05
Batch size: 73
Patch size: 128×128
Max iterations: 92,994
Validation: every 1000 iterations
Metrics: SSIM, PSNR, RMSE
```

## 📊 Forward pass formula

```
Encoder:
  r1 = x
  x = ReLU(Conv1(x))
  x = ReLU(Conv2(x))
  r2 = x
  x = ReLU(Conv3(x))
  x = ReLU(Conv4(x))
  r3 = x
  x = ReLU(Conv5(x))

Decoder:
  x = TConv1(x)
  x = x + r3
  x = TConv2(ReLU(x))
  x = TConv3(ReLU(x))
  x = x + r2
  x = TConv4(ReLU(x))
  x = TConv5(ReLU(x))
  x = x + r1
  
  return x
```

## 💻 Code snippets

### Load và sử dụng model
```python
import torch
from ldctbench.methods.redcnn.network import Model

# Load checkpoint
checkpoint = torch.load("best_SSIM.pt")
model = Model(args)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Inference
with torch.no_grad():
    output = model(input_image)
```

### Training từ config
```bash
python app.py --config configs/redcnn.yaml
```

### Modify architecture
```python
# Thay đổi số channels
model = Model(args, out_ch=48)  # default=96

# Xem architecture
print(model)
```

## 📁 File structure
```
ldctbench/methods/redcnn/
├── network.py      # Model architecture
├── Trainer.py      # Training logic
├── argparser.py    # Arguments (empty)
└── __init__.py

configs/
└── redcnn.yaml     # Hyperparameters
```

## 🔑 Key concepts

**Skip connections**: Giữ lại thông tin chi tiết, giúp gradient flow

**MSE Loss**: `(1/N) * Σ(pred - target)²`

**Bottleneck**: Layer nén nhất (108×108), chứa features quan trọng nhất

**Residual learning**: Network học phần khác biệt thay vì toàn bộ mapping

## 🐛 Common issues

1. **Out of memory**: Giảm batch size hoặc patch size
2. **Slow training**: Kiểm tra GPU utilization, tăng num_workers
3. **Poor performance**: Kiểm tra data normalization, tăng iterations
4. **Dimension mismatch**: Đảm bảo input là bội của patch size

## 📈 Metrics interpretation

- **SSIM**: 0.8-0.9 là tốt, >0.9 là rất tốt
- **PSNR**: >30 dB là acceptable, >35 dB là good
- **RMSE**: Càng thấp càng tốt, phụ thuộc vào data range

## 🚀 Tips

1. **Hyperparameter tuning**: Bắt đầu với config mặc định
2. **Data augmentation**: Random crop, flip để tăng diversity
3. **Checkpoint**: Luôn save theo SSIM (best metric cho medical images)
4. **Validation**: Monitor overfitting bằng validation metrics
5. **Inference**: Batch inference để tăng tốc độ

---

**Tham khảo chi tiết**: Xem `RED-CNN_TUTORIAL_VI.md`
