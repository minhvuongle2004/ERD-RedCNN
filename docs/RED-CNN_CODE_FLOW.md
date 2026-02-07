## Quy trình tạo mô hình RED-CNN trong project

Tài liệu này mô tả **các bước cụ thể để tạo ra mô hình RED-CNN bằng code**,  
theo đúng **quy tắc kiến trúc RED-CNN**, và chỉ rõ **mỗi quy tắc nằm ở đâu trong code**.

---

## 1. Xác định “quy tắc” kiến trúc RED-CNN

Trước khi viết code, cần thống nhất các quy tắc sau (đây là “design spec” của RED-CNN):

- **Quy tắc 1 – Dạng Encoder–Decoder đối xứng**  
  - Encoder: 5 lớp `Conv2d` liên tiếp, kernel \(5 \times 5\), `stride=1`, `padding=0`.  
  - Decoder: 5 lớp `ConvTranspose2d` đối xứng với encoder, cũng kernel \(5 \times 5\), `stride=1`, `padding=0`.  
  - **Trong code**: xem file `ldctbench/methods/redcnn/network.py`, phần `__init__`:
    - `self.conv1`–`self.conv5` và `self.tconv1`–`self.tconv5`.

- **Quy tắc 2 – Số kênh feature cố định trong phần thân mạng**  
  - Từ `conv1` đến `tconv4`: số kênh = `out_ch` (mặc định 96).  
  - Input có 1 kênh, output cũng 1 kênh.  
  - **Trong code**: tham số `out_ch=96` ở `__init__`, `self.conv1 = nn.Conv2d(1, out_ch, ...)`,  
    và `self.tconv5 = nn.ConvTranspose2d(out_ch, 1, ...)`.

- **Quy tắc 3 – Skip connections (residual connections)**  
  - Lưu lại:
    - `residual_1`: ảnh input gốc.  
    - `residual_2`: feature sau `conv2`.  
    - `residual_3`: feature sau `conv4`.  
  - Ở decoder:
    - Cộng `residual_3` sau `tconv1`.  
    - Cộng `residual_2` sau `tconv3`.  
    - Cộng `residual_1` ở output cuối.  
  - **Trong code**: phần `forward` của `Model` trong `network.py`.

- **Quy tắc 4 – Residual learning cho khử nhiễu**  
  - Mạng học phần “chênh lệch” so với ảnh gốc, nên **output cuối = mạng(x) + x**.  
  - **Trong code**: dòng `out += residual_1` ngay trước `return out` trong `forward`.

> Khi dạy sinh viên, bạn có thể bắt đầu từ 4 quy tắc này như “bản thiết kế”,  
> sau đó dẫn các bạn sang xem lớp `Model` trong `network.py` để thấy code tuân thủ đúng các quy tắc.

---

## 2. Bước 1 – Tạo lớp `Model` kế thừa `nn.Module`

- **Mục tiêu**: Định nghĩa “vỏ” của mô hình RED-CNN.
- **Trong code** (`ldctbench/methods/redcnn/network.py`):

```python
class Model(nn.Module):
    """RED-CNN Model..."""

    def __init__(self, args, out_ch=96):
        super(Model, self).__init__()
        ...

    def forward(self, x):
        ...
```

- **Quy tắc tuân thủ**:
  - Kế thừa `nn.Module` để dùng được trong PyTorch.
  - Có hai phần chính: `__init__` (khai báo layers) và `forward` (định nghĩa luồng xử lý).

---

## 3. Bước 2 – Khai báo các lớp encoder (5 Conv2d)

- **Mục tiêu**: Tạo phần “nén thông tin” của RED-CNN theo đúng spec.
- **Trong code** (`__init__` của `Model`):

```python
self.conv1 = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)
self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
```

- **Quy tắc RED-CNN thể hiện ở đây**:
  - **Conv1**: input 1 kênh → `out_ch` kênh (ảnh CT grayscale).  
  - **Conv2–Conv5**: giữ nguyên số kênh = `out_ch`.  
  - Tất cả đều kernel \(5 \times 5\), `stride=1`, `padding=0` → mỗi lớp làm **giảm kích thước ảnh 4 pixel mỗi chiều**.

---

## 4. Bước 3 – Khai báo các lớp decoder (5 ConvTranspose2d)

- **Mục tiêu**: Tạo phần “khôi phục ảnh” đối xứng với encoder.
- **Trong code** (`__init__` của `Model`):

```python
self.tconv1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.tconv2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.tconv3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.tconv4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
```

- **Quy tắc RED-CNN thể hiện ở đây**:
  - Dùng `ConvTranspose2d` để tăng kích thước không gian, đối xứng với phần encoder.  
  - `tconv1`–`tconv4`: giữ số kênh = `out_ch`.  
  - `tconv5`: đưa về lại **1 kênh** để ra ảnh CT grayscale.

---

## 5. Bước 4 – Thêm activation `ReLU`

- **Mục tiêu**: Tạo phi tuyến giữa các lớp tích chập.
- **Trong code**:

```python
self.relu = nn.ReLU()
```

- **Cách dùng**: Trong `forward`, mọi output của conv / tconv (trừ một số chỗ) đều đi qua `self.relu(...)`.  
  Điều này phù hợp với kiến trúc RED-CNN gốc: **Conv/TransConv + ReLU**.

---

## 6. Bước 5 – Viết phần encoder trong `forward` (gồm residual_1, residual_2, residual_3)

- **Mục tiêu**: Cài đặt đúng luồng forward cho nửa đầu mạng và tạo các skip connections.
- **Trong code** (`forward` của `Model`):

```python
# encoder
residual_1 = x
out = self.relu(self.conv1(x))
out = self.relu(self.conv2(out))
residual_2 = out
out = self.relu(self.conv3(out))
out = self.relu(self.conv4(out))
residual_3 = out
out = self.relu(self.conv5(out))
```

- **Quy tắc RED-CNN thể hiện ở đây**:
  - **Lưu residual_1**: ảnh input gốc (dùng cho residual learning ở cuối).  
  - **Lưu residual_2**: sau Conv2 → skip connection giữa Conv2 và TConv3.  
  - **Lưu residual_3**: sau Conv4 → skip connection giữa Conv4 và TConv1.  
  - Conv + ReLU xếp nối tiếp, không có pooling → đúng spec của RED-CNN.

---

## 7. Bước 6 – Viết phần decoder trong `forward` + skip connections

- **Mục tiêu**: Cài đặt nửa sau mạng, đảm bảo skip connections và residual learning.
- **Trong code** (`forward` tiếp tục):

```python
# decoder
out = self.tconv1(out)
out += residual_3
out = self.tconv2(self.relu(out))
out = self.tconv3(self.relu(out))
out += residual_2
out = self.tconv4(self.relu(out))
out = self.tconv5(self.relu(out))
out += residual_1
return out
```

- **Quy tắc RED-CNN thể hiện ở đây**:
  - **Skip connection 1**: `out += residual_3` (sâu hơn).  
  - **Skip connection 2**: `out += residual_2` (nông hơn).  
  - **Residual learning**: `out += residual_1` làm bước cuối cùng → mô hình học phần “sửa lỗi” trên ảnh gốc.  
  - Decoder luôn dùng TransConv + ReLU, trừ trước lần cộng cuối (giữ đúng hành vi paper).

---

## 8. Bước 7 – Để mô hình hoạt động trong toàn project

Sau khi tuân thủ các bước 1–6 để định nghĩa kiến trúc RED-CNN trong `network.py`,  
cần thêm 2 mảnh ghép để **mô hình thực sự được “dùng” để khử nhiễu**:

- **(a) Cấu hình chọn đúng phương pháp RED-CNN**
  - **File**: `configs/redcnn.yaml`.  
  - **Quy tắc**: `method: redcnn` để hệ thống biết phải dùng `Trainer` và `Model` của RED-CNN.

- **(b) Trainer khởi tạo đúng model**
  - **File**: `ldctbench/methods/redcnn/Trainer.py`.  
  - **Dòng quan trọng**:

```python
self.model = Model(args).to(self.dev)
```

  - Dòng này lấy kiến trúc bạn đã định nghĩa trong `network.py` và đưa vào pipeline huấn luyện/inference.

---

## 9. Bước 8 – Cách model “bị chấm điểm” (tính loss MSE)

- **Mục tiêu**: Hiểu đoạn code **đánh giá kết quả** của mô hình (model làm tốt tới đâu).
- **Trong code** (`ldctbench/methods/redcnn/Trainer.py`):

```python
self.criterion = nn.MSELoss()
```

- **Ý nghĩa**:
  - `criterion` là “hàm chấm điểm” (loss function).  
  - `MSELoss` (Mean Squared Error) đo **độ lệch bình phương trung bình từng pixel** giữa:
    - `outputs` = ảnh mô hình dự đoán sau khi đi qua `Model.forward`,  
    - `targets` = ảnh CT liều chuẩn (ground truth).
- **Cách dùng (khái niệm)**:
  - Trong vòng lặp train (được định nghĩa trong `BaseTrainer`), mỗi batch sẽ làm:

```python
outputs = self.model(inputs)          # 1) Cho ảnh nhiễu đi qua model (dùng forward trong network.py)
loss = self.criterion(outputs, targets)  # 2) Tính xem khác ảnh chuẩn bao nhiêu (MSE)
```

- Bạn có thể giải cho sinh viên:  
  “MSE càng nhỏ thì ảnh dự đoán càng gần ảnh chuẩn, model đang làm càng tốt.”

---

## 10. Bước 9 – Cập nhật trọng số để model dần khử nhiễu tốt hơn

- **Mục tiêu**: Hiểu bước **model tự điều chỉnh các tham số** để lần sau sai ít hơn.
- Trong `Trainer.__init__`:

```python
self.optimizer = setup_optimizer(args, self.model.parameters())
```

- **Ý nghĩa**:
  - `optimizer` là “người” quyết định **cập nhật trọng số thế nào** sau khi biết model đang sai bao nhiêu (loss).  
  - Hàm `setup_optimizer` sẽ tạo ra optimizer (thường là Adam) dựa trên config trong `configs/redcnn.yaml`
    (học rate, beta, v.v.).

- **Chu trình train (khái niệm, nằm trong BaseTrainer)**:

```python
outputs = self.model(inputs)                 # 1) forward: ảnh đi qua RED-CNN
loss = self.criterion(outputs, targets)      # 2) tính loss (MSE)

self.optimizer.zero_grad()                   # 3) xóa gradient cũ
loss.backward()                              # 4) backprop: tính gradient cho từng trọng số
self.optimizer.step()                        # 5) cập nhật trọng số theo gradient
```

- Bạn có thể nói với sinh viên:
  - Bước 1: model “đoán” ảnh sạch từ ảnh nhiễu.  
  - Bước 2: so sánh với ảnh chuẩn để biết sai bao nhiêu (loss).  
  - Bước 3–5: dùng loss đó để chỉnh lại mọi “nút vặn” (trọng số Conv/TConv) → lần sau đoán tốt hơn.

---

## 11. Tóm tắt ngắn gọn theo kiểu “recipe”

Để **tự tay tạo ra mô hình RED-CNN đúng chuẩn** trong code, sinh viên có thể ghi nhớ:

1. Tạo `class Model(nn.Module)` với `__init__` và `forward`.  
2. Trong `__init__`:
   - Khai báo 5 `Conv2d` (encoder) và 5 `ConvTranspose2d` (decoder) theo quy tắc RED-CNN.  
   - Thêm một `ReLU`.  
3. Trong `forward`:
   - Encoder: Conv + ReLU, lưu `residual_1`, `residual_2`, `residual_3`.  
   - Decoder: TransConv + ReLU, cộng lần lượt `residual_3`, `residual_2`, cuối cùng cộng `residual_1` rồi trả về.  
4. Đảm bảo `configs/redcnn.yaml` đặt `method: redcnn` và `Trainer` khởi tạo `Model(args)`.  
5. `Trainer` dùng `nn.MSELoss()` để chấm điểm model (loss) và `optimizer` để cập nhật trọng số sau mỗi vòng train.

Khi giảng, bạn có thể yêu cầu sinh viên **đọc lại 4 quy tắc đầu tiên**,  
rồi mở `network.py` và check xem code đã tuân thủ đủ từng quy tắc hay chưa.

