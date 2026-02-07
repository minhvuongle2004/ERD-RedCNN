# 🚀 Hướng dẫn chi tiết: Trình tự train RED-CNN

## 📋 Tổng quan

Hướng dẫn này sẽ giải thích **từng bước một** cách train model RED-CNN, bao gồm:
- File nào chạy đầu tiên
- Code nào dẫn đến file nào
- Giải thích từng bước

---

## 🎯 Trình tự tổng quan

```
BƯỚC 1: Chuẩn bị dữ liệu
   ↓
BƯỚC 2: Chạy file train.py
   ↓
BƯỚC 3: Load Trainer class
   ↓
BƯỚC 4: Trainer khởi tạo Model và DataLoader
   ↓
BƯỚC 5: Training loop
```

---

## 📂 BƯỚC 1: Chuẩn bị dữ liệu (Trước khi train)

### File: `ldctbench/data/prepare_dataset.py`

**Mục đích**: Chuẩn bị dataset từ file DICOM gốc

**Khi nào chạy**: **TRƯỚC KHI TRAIN** (chỉ cần chạy 1 lần)

**Cách chạy**:
```bash
python -m ldctbench.data.prepare_dataset \
    --datafolder /path/to/ldct-data \
    --train_frac 0.7 \
    --val_frac 0.2 \
    --test_frac 0.1
```

**File này làm gì?**:
1. Đọc file DICOM từ thư mục `ldct-data`
2. Chuyển đổi sang định dạng phù hợp
3. Chia thành train/val/test
4. Lưu metadata (min, max values để normalize)
5. Tạo file CSV chứa đường dẫn đến các ảnh

**Kết quả**: 
- Dataset đã được chuẩn bị sẵn
- Có file CSV chứa danh sách ảnh train/val/test

**Lưu ý**: 
- ⚠️ Chỉ cần chạy 1 lần
- ⚠️ Phải chạy TRƯỚC khi train
- ✅ Sau khi chạy xong, có thể bỏ qua bước này

---

## 🎬 BƯỚC 2: Bắt đầu training - File chính

### File: `ldctbench/scripts/train.py`

**Đây là file CHÍNH để bắt đầu train!**

**Cách chạy**:
```bash
python -m ldctbench.scripts.train \
    --trainer redcnn \
    --datafolder /path/to/ldct-data \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001
```

**Hoặc dùng config file**:
```bash
python -m ldctbench.scripts.train --config configs/redcnn.yaml
```

---

## 🔍 Chi tiết từng bước trong `train.py`

### Bước 2.1: Hàm `main()` được gọi đầu tiên

**Vị trí trong code**: Dòng 162-217

```python
def main():
    # Bước 1: Parse arguments
    parser = make_parser()  # ← Gọi hàm từ ldctbench.utils.argparser
    args = parser.parse_args()
    args = use_config(args)  # ← Load config file nếu có
    
    # Bước 2: Kiểm tra datafolder
    if not hasattr(args, "datafolder") or not args.datafolder:
        # Lấy từ environment variable hoặc raise error
        ...
    
    # Bước 3: Xử lý dryrun mode
    if args.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
    
    # Bước 4: Gọi hàm train()
    train(args)  # ← Điểm quan trọng: Gọi hàm train()
```

**Giải thích**:
- ✅ `main()` là điểm bắt đầu khi chạy file
- ✅ Parse arguments từ command line hoặc config file
- ✅ Kiểm tra datafolder (nơi chứa dữ liệu)
- ✅ Gọi hàm `train(args)` để bắt đầu training

---

### Bước 2.2: Hàm `train(args)` - Quy trình chính

**Vị trí trong code**: Dòng 42-156

#### 2.2.1: Thiết lập Random Seed

```python
if args.seed is None:
    args.seed = np.random.randint(1, 10000)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
```

**Giải thích**: Đảm bảo kết quả có thể tái tạo (reproducible)

---

#### 2.2.2: Thiết lập GPU/CPU

```python
# Xử lý nhiều GPU, 1 GPU, hoặc CPU
if isinstance(args.devices, list) and len(args.devices) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in args.devices])
    args.devices = list(range(len(args.devices)))
elif isinstance(args.devices, list) and len(args.devices) == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices[0])
    args.devices = 0
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    args.devices = 0

device = torch.device("cuda" if args.cuda else "cpu")
```

**Giải thích**: 
- ✅ Chọn GPU nào để dùng
- ✅ Tạo device object (cuda hoặc cpu)

---

#### 2.2.3: Khởi tạo Weights & Biases (WandB)

```python
while True:
    try:
        if hasattr(args, "wandbtag") and args.wandbtag:
            wandb.init(project="ldct-benchmark", config=args, tags=[args.wandbtag])
        else:
            wandb.init(project="ldct-benchmark", config=args)
        break
    except Exception as e:
        print(f"{e} ... retrying...")
        time.sleep(10)
```

**Giải thích**: 
- ✅ Khởi tạo WandB để theo dõi training
- ✅ Retry nếu kết nối mạng bị lỗi

---

#### 2.2.4: Load Trainer Class (QUAN TRỌNG!)

**Vị trí trong code**: Dòng 125-148

```python
# Import module trainer dựa trên args.trainer
# Ví dụ: args.trainer="redcnn" 
# → import ldctbench.methods.redcnn.Trainer
try:
    trainer_module = importlib.import_module(
        "ldctbench.methods.{}.Trainer".format(args.trainer)
    )
except ModuleNotFoundError:
    raise ValueError(
        "Trainer {0} not known and module methods.{0}.Trainer not found".format(
            args.trainer
        )
    )

# Lấy class "Trainer" từ module
trainer_class = getattr(trainer_module, "Trainer")

# Khởi tạo instance của Trainer
trainer = trainer_class(args, device)  # ← Điểm quan trọng!
```

**Giải thích**:
- ✅ **Dynamic import**: Import module dựa trên tên trainer
- ✅ Nếu `args.trainer="redcnn"` → Import `ldctbench.methods.redcnn.Trainer`
- ✅ Lấy class `Trainer` từ module
- ✅ Khởi tạo instance: `trainer = Trainer(args, device)`
- ✅ **Điểm quan trọng**: Tất cả logic training nằm trong class Trainer này!

**Dẫn đến file**: `ldctbench/methods/redcnn/Trainer.py`

---

#### 2.2.5: Bắt đầu training

```python
print("Start training...")
trainer.fit()  # ← Gọi phương thức fit() để bắt đầu train
```

**Giải thích**:
- ✅ Gọi phương thức `fit()` của Trainer
- ✅ Phương thức này chứa toàn bộ training loop

**Dẫn đến**: Phương thức `fit()` trong `ldctbench/methods/redcnn/Trainer.py`

---

## 🏗️ BƯỚC 3: Trainer Class - Khởi tạo

### File: `ldctbench/methods/redcnn/Trainer.py`

**File này được import và khởi tạo ở Bước 2.4**

### Bước 3.1: Class Trainer kế thừa BaseTrainer

```python
from ldctbench.methods.base import BaseTrainer
from ldctbench.utils.training_utils import setup_optimizer
from .network import Model

class Trainer(BaseTrainer):
    def __init__(self, args: Namespace, device: torch.device):
        super().__init__(args, device)  # ← Gọi __init__ của BaseTrainer
        ...
```

**Giải thích**:
- ✅ `Trainer` kế thừa từ `BaseTrainer`
- ✅ `super().__init__(args, device)` gọi `__init__` của BaseTrainer
- ✅ **Dẫn đến**: `ldctbench/methods/base.py` - BaseTrainer class

---

### Bước 3.2: BaseTrainer.__init__() làm gì?

**File**: `ldctbench/methods/base.py`

**BaseTrainer làm gì?**:
1. ✅ Tạo DataLoader (load dữ liệu train/val)
2. ✅ Setup các thư viện cần thiết
3. ✅ Tạo thư mục lưu model

**Quan trọng**: BaseTrainer tạo DataLoader từ dataset!

**Dẫn đến**: `ldctbench/data/LDCTMayo.py` - Dataset class

---

### Bước 3.3: Khởi tạo Model

**Vị trí trong code**: Dòng 31 trong Trainer.py

```python
self.model = Model(args).to(self.dev)
```

**Giải thích**:
- ✅ Tạo instance của Model class
- ✅ Chuyển model lên GPU (`.to(self.dev)`)
- ✅ **Dẫn đến**: `ldctbench/methods/redcnn/network.py` - Model class

---

### Bước 3.4: Khởi tạo Loss và Optimizer

**Vị trí trong code**: Dòng 30, 34 trong Trainer.py

```python
self.criterion = nn.MSELoss()  # Loss function
self.optimizer = setup_optimizer(args, self.model.parameters())  # Optimizer
```

**Giải thích**:
- ✅ `criterion`: MSE Loss để tính sai số
- ✅ `optimizer`: Adam optimizer (từ `setup_optimizer`)
- ✅ **Dẫn đến**: `ldctbench/utils/training_utils.py` - hàm `setup_optimizer`

---

## 🔄 BƯỚC 4: Training Loop

### File: `ldctbench/methods/base.py` - Phương thức `fit()`

**Phương thức này được gọi ở Bước 2.5**: `trainer.fit()`

**fit() làm gì?** (Dòng 136-151):
1. ✅ Vòng lặp cho đến khi đạt max_iterations
2. ✅ Mỗi iteration:
   - Training phase (train trên training set)
   - Validation phase (đánh giá trên validation set)
   - Log losses, metrics, và ảnh
3. ✅ Lưu model tốt nhất (trong save_checkpoint)

**Chi tiết training loop**:

```python
def fit(self):
    delta_seed = 0
    while self.iteration < self.args.max_iterations:
        # Set seed cho mỗi iteration (để reproducible)
        torch.manual_seed(self.args.seed + delta_seed)
        np.random.seed(self.args.seed + delta_seed)
        
        # Train và validate
        self.train()  # ← Gọi train()
        self.validate()  # ← Gọi validate()
        
        # Log losses, metrics, và ảnh
        self.log()  # ← Lưu model, log lên WandB
        
        # Reset để iteration tiếp theo
        self.losses.reset()
        self.metrics.reset()
        
        delta_seed += 1
```

**Giải thích**:
- ✅ Vòng lặp `while` cho đến khi `iteration >= max_iterations`
- ✅ Mỗi iteration:
  1. Set seed (để reproducible)
  2. `self.train()`: Train trên training set
  3. `self.validate()`: Validate trên validation set
  4. `self.log()`: Lưu model tốt nhất, log lên WandB
  5. Reset losses và metrics
- ✅ `self.log()` gọi `save_checkpoint()` để lưu model tốt nhất

---

### Bước 4.1: Training - `train()`

**File**: `ldctbench/methods/base.py`

**train() làm gì?**:

```python
def train(self):
    self.model.train()  # Chế độ training
    for batch in tqdm(self.dataloader["train"], desc="Train: "):
        batch = {
            k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
        }
        self.train_step(batch)  # ← Gọi train_step() cho mỗi batch
    self.losses.summarize("train")
```

**train_step() làm gì?** (Dòng 36-47):

```python
def train_step(self, batch):
    inputs, targets = batch["x"], batch["y"]  # Lấy ảnh nhiễu và sạch
    
    outputs = self.model(inputs)  # ← Model xử lý ảnh nhiễu
    loss = self.criterion(outputs, targets)  # ← Tính loss
    
    self.optimizer.zero_grad()  # Xóa gradient cũ
    loss.backward()  # Tính gradient
    self.optimizer.step()  # Cập nhật weights
    
    self.iteration += 1
    self.losses.push(loss, "train")  # Lưu loss để tính trung bình
```

**Giải thích**:
- ✅ `self.model.train()`: Chuyển model sang chế độ training
- ✅ `self.dataloader["train"]`: DataLoader cho training (đã tạo ở __init__)
- ✅ Vòng lặp qua từng batch:
  1. Chuyển batch lên GPU: `.to(self.dev)`
  2. Gọi `train_step()`:
     - Forward: `model(inputs)` → outputs
     - Tính loss: `criterion(outputs, targets)`
     - Backward: `loss.backward()` → tính gradient
     - Update: `optimizer.step()` → cập nhật weights
  3. Lưu loss để tính trung bình
- ✅ `losses.summarize("train")`: Tính trung bình loss và log

**Model forward pass dẫn đến**: `ldctbench/methods/redcnn/network.py` - phương thức `forward()`

---

### Bước 4.2: Model.forward() - Xử lý ảnh

**File**: `ldctbench/methods/redcnn/network.py`

**forward() làm gì?**:

```python
def forward(self, x):
    # Encoder
    residual_1 = x
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    residual_2 = out
    out = self.relu(self.conv3(out))
    out = self.relu(self.conv4(out))
    residual_3 = out
    out = self.relu(self.conv5(out))
    
    # Decoder
    out = self.tconv1(out)
    out += residual_3  # Skip connection
    out = self.tconv2(self.relu(out))
    out = self.tconv3(self.relu(out))
    out += residual_2  # Skip connection
    out = self.tconv4(self.relu(out))
    out = self.tconv5(self.relu(out))
    out += residual_1  # Skip connection
    
    return out
```

**Giải thích**:
- ✅ **Encoder**: 5 layers conv, mỗi layer giảm kích thước
- ✅ **Decoder**: 5 layers transpose conv, mỗi layer tăng kích thước
- ✅ **Skip connections**: Cộng thêm residual từ encoder
- ✅ Kết quả: Ảnh sạch (đã khử nhiễu)

---

### Bước 4.3: Validation - `validate()`

**File**: `ldctbench/methods/base.py`

**validate() làm gì?** (Dòng 72-83):

```python
def validate(self):
    self.images = {}  # Lưu ảnh để log lên WandB
    self.model.eval()  # Chế độ evaluation
    for batch_idx, batch in enumerate(
        tqdm(self.dataloader["val"], desc="Validate: ")
    ):
        batch = {
            k: Variable(v).to(self.dev, non_blocking=True) for k, v in batch.items()
        }
        self.val_step(batch_idx, batch)  # ← Gọi val_step() cho mỗi batch
    self.losses.summarize("val")  # Tính trung bình loss
    self.metrics.summarize()  # Tính metrics (PSNR, SSIM, RMSE)
```

**val_step() làm gì?** (Dòng 49-61):

```python
@torch.no_grad()  # ← Không tính gradient (tiết kiệm bộ nhớ)
def val_step(self, batch_idx, batch):
    inputs, targets = batch["x"], batch["y"]
    
    outputs = self.model(inputs)  # Forward pass
    loss = self.criterion(outputs, targets)  # Tính loss
    
    self.losses.push(loss, "val")  # Lưu loss
    self.metrics.push(targets, outputs)  # Lưu để tính metrics
    
    # Log ảnh lên WandB (chỉ một số batch đầu)
    if batch_idx < self.args.valsamples:
        self.log_wandb_images(
            {"low dose": inputs, "prediction": outputs, "high dose": targets}
        )
```

**Giải thích**:
- ✅ `self.model.eval()`: Chế độ evaluation (không update weights)
- ✅ `@torch.no_grad()`: Decorator - không tính gradient (nhanh hơn, tiết kiệm bộ nhớ)
- ✅ `self.dataloader["val"]`: DataLoader cho validation
- ✅ Vòng lặp qua từng batch:
  1. Forward: `model(inputs)` → outputs
  2. Tính loss và metrics
  3. Log ảnh lên WandB (nếu là batch đầu)
- ✅ `losses.summarize("val")`: Tính trung bình loss
- ✅ `metrics.summarize()`: Tính PSNR, SSIM, RMSE

---

## 📊 Sơ đồ luồng hoàn chỉnh

```
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 1: Chuẩn bị dữ liệu (TRƯỚC KHI TRAIN)                  │
└─────────────────────────────────────────────────────────────┘
                    ↓
    python -m ldctbench.data.prepare_dataset
                    ↓
    Tạo dataset train/val/test
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 2: Chạy file chính                                     │
└─────────────────────────────────────────────────────────────┘
                    ↓
    python -m ldctbench.scripts.train --trainer redcnn
                    ↓
    ┌──────────────────────────────────────┐
    │ train.py                              │
    │  ├─ main()                            │
    │  │   ├─ Parse arguments               │
    │  │   └─ train(args)                  │
    │  │                                    │
    │  └─ train(args)                       │
    │      ├─ Setup seed, GPU               │
    │      ├─ Init WandB                    │
    │      ├─ Import Trainer class          │ ← Bước 3
    │      │   └─ ldctbench.methods.redcnn  │
    │      │       .Trainer                 │
    │      │                                │
    │      └─ trainer.fit()                 │ ← Bước 4
    └──────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 3: Trainer khởi tạo                                   │
└─────────────────────────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────┐
    │ Trainer.__init__()                   │
    │  ├─ super().__init__()               │
    │  │   └─ BaseTrainer.__init__()       │
    │  │       ├─ Tạo DataLoader           │
    │  │       │   └─ LDCTMayo dataset     │
    │  │       └─ Setup thư mục           │
    │  │                                   │
    │  ├─ self.model = Model(args)         │
    │  │   └─ network.py                  │
    │  │                                   │
    │  ├─ self.criterion = MSE()          │
    │  └─ self.optimizer = Adam()          │
    └──────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ BƯỚC 4: Training Loop                                        │
└─────────────────────────────────────────────────────────────┘
                    ↓
    ┌──────────────────────────────────────┐
    │ fit() - Vòng lặp iterations          │
    │  │                                   │
    │  ├─ while iteration < max_iter:      │
    │  │   ├─ train()                      │
    │  │   │   ├─ for batch in loader:     │
    │  │   │   │   ├─ train_step(batch)    │
    │  │   │   │   │   ├─ output = model(x)│
    │  │   │   │   │   │   └─ network.     │
    │  │   │   │   │   │       forward()   │
    │  │   │   │   │   ├─ loss = MSE()     │
    │  │   │   │   │   ├─ loss.backward()  │
    │  │   │   │   │   └─ optimizer.step()  │
    │  │   │   └─ losses.summarize()       │
    │  │   │                               │
    │  │   ├─ validate()                    │
    │  │   │   ├─ for batch in loader:     │
    │  │   │   │   ├─ val_step(batch)      │
    │  │   │   │   │   ├─ output = model() │
    │  │   │   │   │   ├─ loss, metrics    │
    │  │   │   │   │   └─ log images       │
    │  │   │   └─ metrics.summarize()      │
    │  │   │                               │
    │  │   ├─ log()                         │
    │  │   │   ├─ save_checkpoint()        │
    │  │   │   └─ Log lên WandB            │
    │  │   └─ Reset losses/metrics          │
    │  └─ End training                      │
    └──────────────────────────────────────┘
```

---

## 📝 Tóm tắt các file quan trọng

### 1. File chính để chạy
```
ldctbench/scripts/train.py
```
- ✅ File này chạy đầu tiên
- ✅ Parse arguments
- ✅ Import và khởi tạo Trainer
- ✅ Gọi trainer.fit()

---

### 2. Trainer class
```
ldctbench/methods/redcnn/Trainer.py
```
- ✅ Kế thừa BaseTrainer
- ✅ Khởi tạo Model, Loss, Optimizer
- ✅ Chứa logic training (từ BaseTrainer)

---

### 3. Base Trainer
```
ldctbench/methods/base.py
```
- ✅ Tạo DataLoader
- ✅ Chứa phương thức fit(), train_epoch(), val_epoch()
- ✅ Lưu model, log WandB

---

### 4. Model architecture
```
ldctbench/methods/redcnn/network.py
```
- ✅ Định nghĩa Model class
- ✅ Phương thức forward() xử lý ảnh

---

### 5. Dataset
```
ldctbench/data/LDCTMayo.py
```
- ✅ Dataset class để load ảnh
- ✅ Được dùng bởi DataLoader

---

### 6. Utilities
```
ldctbench/utils/training_utils.py
```
- ✅ Hàm setup_optimizer()
- ✅ Các hàm tiện ích khác

---

## 🎯 Checklist khi train

### Trước khi train
```
□ 1. Đã chạy prepare_dataset.py (chuẩn bị dữ liệu)
□ 2. Có dataset train/val/test
□ 3. Có GPU hoặc CPU
□ 4. Đã cài đặt thư viện (PyTorch, wandb, ...)
```

### Khi train
```
□ 1. Chạy: python -m ldctbench.scripts.train --trainer redcnn
□ 2. Kiểm tra log xem có lỗi không
□ 3. Kiểm tra WandB xem loss có giảm không
□ 4. Kiểm tra model có được lưu không
```

---

## 💡 Ví dụ chạy đầy đủ

### Bước 1: Chuẩn bị dữ liệu (1 lần duy nhất)
```bash
python -m ldctbench.data.prepare_dataset \
    --datafolder ./ldct-data \
    --train_frac 0.7 \
    --val_frac 0.2 \
    --test_frac 0.1
```

### Bước 2: Train model
```bash
python -m ldctbench.scripts.train \
    --trainer redcnn \
    --datafolder ./ldct-data \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.0001 \
    --devices 0
```

### Hoặc dùng config file
```bash
python -m ldctbench.scripts.train --config configs/redcnn.yaml
```

---

## 🔍 Debug: Làm sao biết đang ở bước nào?

### Kiểm tra log

**Khi chạy train.py, bạn sẽ thấy**:
```
Start training...  ← Đã vào hàm train()
Initializing Trainer...  ← Đang khởi tạo Trainer
Loading dataset...  ← Đang load dữ liệu
Epoch 1/100...  ← Đã vào training loop
```

### Kiểm tra WandB

- Mở WandB dashboard
- Xem loss có giảm không
- Xem ảnh output có đẹp không

---

## 🎓 Kết luận

**Trình tự train RED-CNN**:

1. ✅ **Chuẩn bị dữ liệu** (1 lần): `prepare_dataset.py`
2. ✅ **Chạy file chính**: `train.py`
3. ✅ **Trainer khởi tạo**: Load Model, DataLoader, Optimizer
4. ✅ **Training loop**: fit() → train_epoch() → val_epoch()
5. ✅ **Model forward**: network.forward() xử lý ảnh
6. ✅ **Lưu model**: Model tốt nhất được lưu

**Quan trọng nhất**:
- ✅ File chính: `ldctbench/scripts/train.py`
- ✅ Trainer: `ldctbench/methods/redcnn/Trainer.py`
- ✅ Model: `ldctbench/methods/redcnn/network.py`

**Chúc bạn train thành công!** 🚀

---

➡️ **Xem thêm**: 
- `CAU_HOI_17.md` - Cần chuẩn bị gì để train?
- `CAU_HOI_21.md` - Cách chạy training như thế nào?

