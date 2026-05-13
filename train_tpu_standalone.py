"""
TRAIN_TPU_STANDALONE.PY
========================
Bản script đặc nhiệm để huấn luyện EDR-REDNet trên 8 lõi TPU (Kaggle TPU VM).
Sử dụng torch_xla để tối ưu hóa tốc độ.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

# Import các thành phần torch_xla
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
except ImportError:
    print("❌ Lỗi: Không tìm thấy thư viện torch_xla. Hãy chắc chắn bạn đang dùng TPU VM.")

from ldctbench.methods.edrrednet.network import Model
from ldctbench.methods.edrrednet.loss import CharbonnierLoss
from ldctbench.data.LDCTMayo import MayoDataset
from argparse import Namespace

# ─── CẤU HÌNH VARIANT C ──────────────────────────────────────────────────────
CONFIG = {
    "seed": 1339,
    "lr": 9.583e-05,
    "mbs": 8,            # 8 per core (tổng batch size = 64)
    "max_iters": 31000,
    "num_edge_blocks": 2,
    "use_sobel_input": True,
    "patchsize": 128,
    "data_norm": "meanstd"
}

def train_map_fn(index, flags):
    # Mỗi lõi TPU (trong 8 lõi) sẽ chạy hàm này song song
    torch.manual_seed(CONFIG["seed"])
    
    # 1. Khởi tạo Device TPU
    device = xm.xla_device()
    
    # 2. Khởi tạo Dataset & Dataloader (Dùng DistributedSampler cho 8 lõi)
    dataset = MayoDataset(
        datafolder=flags['data_path'],
        split="train_set",
        patchsize=CONFIG["patchsize"],
        norm=CONFIG["data_norm"]
    )
    
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=CONFIG["mbs"],
        sampler=train_sampler,
        num_workers=2,
        drop_last=True
    )
    
    # 3. Khởi tạo Model & Loss
    args = Namespace(
        num_edge_blocks=CONFIG["num_edge_blocks"],
        use_sobel_input=CONFIG["use_sobel_input"]
    )
    model = Model(args).to(device)
    
    # Variant C chỉ dùng Charbonnier Loss (alpha=0)
    criterion = CharbonnierLoss().to(device)
    
    # 4. Optimizer (Quan trọng: LR phải nhân với world_size)
    lr = CONFIG["lr"] * xm.xrt_world_size()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 5. Vòng lặp huấn luyện
    model.train()
    iteration = 0
    start_time = time.time()
    
    # ParallelLoader giúp nạp dữ liệu vào TPU cực nhanh
    train_device_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
    
    xm.master_print(f"🚀 Bắt đầu huấn luyện Variant C trên 8 lõi TPU...")
    
    while iteration < CONFIG["max_iters"]:
        for batch in train_device_loader:
            ldct = batch["x"]
            ndct = batch["y"]
            
            optimizer.zero_grad()
            pred = model(ldct)
            loss = criterion(pred, ndct)
            
            loss.backward()
            
            # xm.optimizer_step thực hiện đồng bộ hóa giữa các lõi
            xm.optimizer_step(optimizer)
            
            iteration += 1
            
            # Chỉ in log ở lõi chính (lõi số 0) để không bị loạn màn hình
            if iteration % 100 == 0:
                elapsed = time.time() - start_time
                xm.master_print(f"Iter {iteration}/{CONFIG['max_iters']} | Loss: {loss.item():.6f} | Time: {elapsed:.1f}s")
                start_time = time.time()
            
            # Lưu checkpoint mỗi 5000 vòng
            if iteration % 5000 == 0:
                xm.save(model.state_dict(), f"variantC_seed{CONFIG['seed']}_iter{iteration}.pt")
            
            if iteration >= CONFIG["max_iters"]:
                break
                
    # Lưu bản cuối cùng
    xm.save(model.state_dict(), f"variantC_seed{CONFIG['seed']}_final.pt")
    xm.master_print("✅ Huấn luyện hoàn tất!")

if __name__ == "__main__":
    # Nhận đường dẫn data từ argument
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/input/mayo-clinic-ldct-ndct-dataset"
    
    flags = {'data_path': data_path}
    
    # Kích hoạt 8 lõi TPU
    xmp.spawn(train_map_fn, args=(flags,), nprocs=8, start_method='fork')
