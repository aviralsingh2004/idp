# server/infer.py
import json, time, numpy as np, torch
import torch.nn as nn

T, F = 50, 1

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
model.load_state_dict(torch.load('server/model/autoencoder.pth'))
model.eval()

while True:
    data = np.sin(np.linspace(time.time(), time.time() + 5, T)) + 0.05 * np.random.randn(T)
    x = torch.Tensor(data.reshape(1, T, F))
    with torch.no_grad():
        recon = model(x)
        mse = float(((recon - x) ** 2).mean().item())
    print(json.dumps({"anomalyScore": mse}), flush=True)
    time.sleep(0.5)
