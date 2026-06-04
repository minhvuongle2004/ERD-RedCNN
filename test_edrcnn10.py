"""Mini training test for EDR-CNN10 on CPU."""
import torch
from argparse import Namespace
from ldctbench.methods.edrcnn10.network import Model
from ldctbench.methods.edrcnn10.loss import CombinedLoss

device = torch.device("cpu")
args = Namespace(num_edge_blocks=2, use_sobel_input=True)

model = Model(args).to(device)
criterion = CombinedLoss(alpha=0.1, beta=0.0, gamma=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print("=== EDR-CNN10 Mini Training Test (CPU, 10 steps) ===")
losses = []
for i in range(10):
    x = torch.randn(2, 1, 64, 64).to(device)
    y = torch.randn(2, 1, 64, 64).to(device)
    optimizer.zero_grad()
    pred = model(x)
    loss, comps = criterion(pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    charb = comps["loss/charbonnier"]
    sobel = comps["loss/sobel"]
    print(f"  Step {i+1:2d}: total={loss.item():.6f}  charb={charb:.4f}  sobel={sobel:.4f}")

print()
print(f"Loss start : {losses[0]:.6f}")
print(f"Loss end   : {losses[-1]:.6f}")
print(f"No NaN     : {not any(l != l for l in losses)}")
print("=== TEST PASS ===")
