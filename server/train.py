# server/train.py
import os, torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make sure model directory exists
os.makedirs('server/model', exist_ok=True)

# Generate synthetic data
data = np.sin(np.linspace(0, 100, 10000)) + 0.1 * np.random.randn(10000)
T, F = 50, 1
X = np.array([data[i:i+T] for i in range(len(data) - T)]).reshape(-1, T, F).astype(np.float32)

loader = DataLoader(TensorDataset(torch.Tensor(X), torch.Tensor(X)), batch_size=64, shuffle=True)

# LSTM Autoencoder
class LSTMAutoencoder(nn.Module):
    def __init__(self, T, F, latent=16):
        super().__init__()
        self.encoder = nn.LSTM(F, latent, batch_first=True)
        self.decoder = nn.LSTM(latent, F, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        out, _ = self.decoder(h)
        return out

model = LSTMAutoencoder(T, F)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train
for epoch in range(20):
    total, loss_sum = 0, 0
    for xb, _ in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), xb)
        loss.backward()
        opt.step()
        total += xb.size(0)
        loss_sum += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}: Loss = {loss_sum / total:.6f}")

# Save
torch.save(model.state_dict(), 'server/model/autoencoder.pth')
print("✅ Model saved.")
