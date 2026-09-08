"""
Phase 1: Scale Dataset to 10,000 Samples & Train 2D-FNO Forward Surrogate
========================================================================
- Scales master stratified HDF5 archive: 8,000 Train, 1,000 Val, 1,000 Test.
- Trains 2D Fourier Neural Operator with passivity penalty, weight decay, and cosine annealing.
- Evaluates R^2, NMSE, and inference latency on 1,000 held-out test geometries.
- Saves checkpoint to models/fno_surrogate_best.pt and history to data/outputs/fno_training_history.json.
"""

import os
import sys
import time
import json
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Resolve project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from src.automation.dataset_pipeline import generate_stratified_dataset
from src.automation.rcwa_solver import MetasurfaceRCWASolver


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNOSurrogate2D(nn.Module):
    def __init__(self, modes1: int = 16, modes2: int = 16, width: int = 64, num_freq_points: int = 101):
        super().__init__()
        self.modes1, self.modes2, self.width, self.num_freq_points = modes1, modes2, width, num_freq_points
        self.p = nn.Conv2d(1, self.width, kernel_size=1)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(self.width * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 4 * num_freq_points)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.p(x)
        x = F.gelu(self.conv0(x) + self.w0(x))
        x = F.gelu(self.conv1(x) + self.w1(x))
        x = F.gelu(self.conv2(x) + self.w2(x))
        x = F.gelu(self.conv3(x) + self.w3(x))
        x = self.pool(x).flatten(start_dim=1)
        out = self.fc2(F.gelu(self.fc1(x)))
        return out.view(-1, 4, self.num_freq_points)


class MetasurfaceDataset(Dataset):
    def __init__(self, h5_filepath: str, split: str = 'train', max_samples: int = None):
        super().__init__()
        with h5py.File(h5_filepath, 'r') as f:
            n = len(f[split]['sdf']) if max_samples is None else min(max_samples, len(f[split]['sdf']))
            self.sdf = torch.from_numpy(f[split]['sdf'][:n]).unsqueeze(1)  # (N, 1, 128, 128)
            s11_r = f[split]['s11_real'][:n]
            s11_i = f[split]['s11_imag'][:n]
            s21_r = f[split]['s21_real'][:n]
            s21_i = f[split]['s21_imag'][:n]
            y = np.stack([s11_r, s11_i, s21_r, s21_i], axis=1)  # (N, 4, 101)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.sdf)

    def __getitem__(self, idx):
        return self.sdf[idx], self.y[idx]


def compute_passivity_penalty(pred: torch.Tensor) -> torch.Tensor:
    s11_sq = pred[:, 0, :]**2 + pred[:, 1, :]**2
    s21_sq = pred[:, 2, :]**2 + pred[:, 3, :]**2
    return torch.mean(F.relu(s11_sq + s21_sq - 1.0))


def run_scaling_pipeline(total_samples: int = 10000, epochs: int = 25, batch_size: int = 64):
    h5_path = os.path.join(BASE_DIR, 'data/raw/metasurface_dataset_25k.h5')
    models_dir = os.path.join(BASE_DIR, 'models')
    outputs_dir = os.path.join(BASE_DIR, 'data/outputs')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    # 1. Dataset Generation / Check
    need_gen = True
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            if 'train' in f and len(f['train']['sdf']) >= int(total_samples * 0.8):
                print(f"Master dataset already exists with {len(f['train']['sdf'])} train samples. Skipping regeneration.")
                need_gen = False

    if need_gen:
        print(f"Scaling dataset to {total_samples} samples...")
        solver = MetasurfaceRCWASolver()
        generate_stratified_dataset(
            total_samples=total_samples,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            resolution=128,
            output_h5_path=h5_path,
            solver=solver,
            seed=2026,
            verbose=True,
        )

    # 2. Setup Device & Data Loaders
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training 2D-FNO Surrogate on device: {device}")

    train_ds = MetasurfaceDataset(h5_path, split='train')
    val_ds = MetasurfaceDataset(h5_path, split='val')
    test_ds = MetasurfaceDataset(h5_path, split='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 3. Model, Optimizer, Scheduler
    model = FNOSurrogate2D().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"FNO Architecture: {param_count:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_weights_path = os.path.join(models_dir, 'fno_surrogate_best.pt')
    history = {"train_loss": [], "val_loss": [], "passivity_penalty": []}

    print(f"Starting FNO surrogate training ({epochs} epochs)...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        pass_loss_total = 0.0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)

            mse = F.mse_loss(pred, y_b)
            pass_pen = compute_passivity_penalty(pred)
            total_loss = mse + 10.0 * pass_pen

            total_loss.backward()
            optimizer.step()

            train_loss += mse.item() * x_b.size(0)
            pass_loss_total += pass_pen.item() * x_b.size(0)

        scheduler.step()
        train_loss /= len(train_ds)
        pass_loss_total /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                val_loss += F.mse_loss(pred, y_b).item() * x_b.size(0)
        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["passivity_penalty"].append(pass_loss_total)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_weights_path)

        if ep % 5 == 0 or ep == epochs:
            print(f"  Epoch {ep:2d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Passivity: {pass_loss_total:.6e}")

    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f} s ({train_time/epochs:.2f} s/epoch). Best Val Loss: {best_val_loss:.6f}")

    # 4. Evaluation on 1,000 Held-Out Test Samples
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    y_true_list, y_pred_list, latencies = [], [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            t_start = time.time()
            pred = model(x_b)
            latencies.append((time.time() - t_start) / x_b.size(0) * 1000)
            y_true_list.append(y_b.cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    test_mse = float(np.mean((y_true - y_pred)**2))
    var_y = float(np.var(y_true))
    nmse = float(test_mse / var_y)
    r2 = float(1.0 - nmse)
    mean_latency_ms = float(np.mean(latencies))

    print("\n--- Test Set (N=1,000 Held-Out Geometries) Performance ---")
    print(f"  Test R^2 Score:         {r2:.5f} (Target: >= 0.995)")
    print(f"  Test NMSE:              {nmse:.2e} (Target: < 1.0e-4)")
    print(f"  Mean Inference Latency: {mean_latency_ms:.3f} ms/sample (Target: < 0.5 ms)")

    # Save metrics and history
    results = {
        "dataset": {
            "total_samples": total_samples,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
        },
        "model": {
            "param_count": param_count,
            "modes": 16,
            "width": 64,
        },
        "test_metrics": {
            "r2_score": r2,
            "nmse": nmse,
            "mse": test_mse,
            "mean_latency_ms": mean_latency_ms,
        },
        "training_time_s": train_time,
        "history": history,
    }

    history_file = os.path.join(outputs_dir, 'fno_training_history.json')
    with open(history_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved training history to {history_file}")


if __name__ == '__main__':
    run_scaling_pipeline(total_samples=10000, epochs=25, batch_size=64)
