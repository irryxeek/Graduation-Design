import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/Graduation-Design')

from ro_retrieval.data.dataset import ROMultiVarDataset
from ro_retrieval.model.unet import EnhancedConditionalUNet1D
from ro_retrieval.model.diffusion import DiffusionSchedule
from torch.utils.data import DataLoader

device = torch.device('cuda')
print(f'Device: {device}')

# 数据
dataset = ROMultiVarDataset('Data/Processed_ATP/train_x.npy', 'Data/Processed_ATP/train_y.npy')
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
print(f'Dataset: {len(dataset)} samples')

# 模型
model = EnhancedConditionalUNet1D(in_channels=2, cond_channels=1, out_channels=2, use_cross_attention=True).to(device)
schedule = DiffusionSchedule(1000, device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

# 训练 2 个 epoch
for epoch in range(2):
    model.train()
    losses = []
    for i, (condition, x_0) in enumerate(loader):
        condition = condition.to(device)
        x_0 = x_0.to(device)

        t = torch.randint(0, 1000, (x_0.shape[0], 1), device=device).long()
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise)
        noise_pred = model(x_t, t, condition)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        if i % 50 == 0:
            print(f'Epoch {epoch+1}, Batch {i}/{len(loader)}, Loss: {loss.item():.6f}')

    avg_loss = sum(losses) / len(losses)
    print(f'Epoch {epoch+1} avg loss: {avg_loss:.6f}')

print('Training completed successfully!')
