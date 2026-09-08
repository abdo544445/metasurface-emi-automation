"""
Experiment 3: Data-Efficiency & Neural Operator Power-Law Scaling
================================================================
Trains 2D-FNO forward surrogates across varying dataset subsets:
N in [500, 1000, 2500, 5000, 8000].

Measures Test Relative L2 Error (NMSE) against a fixed 1,000-sample test partition.
Fits the empirical power-law scaling exponent:
    Error(N) = C * N^(-gamma)
Demonstrating that the surrogate acts as a generalizable operator learner.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from src.experiments.scale_dataset_and_fno import FNOSurrogate2D, MetasurfaceDataset, compute_passivity_penalty


def run_data_scaling_study(subset_sizes=[500, 1000, 2500, 5000, 8000], epochs=15):
    outputs_dir = os.path.join(BASE_DIR, 'data/outputs')
    h5_path = os.path.join(BASE_DIR, 'data/raw/metasurface_dataset_25k.h5')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Experiment 3 (Data Scaling Law) on device: {device}...")

    test_ds = MetasurfaceDataset(h5_path, split='test')
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    scaling_results = []

    for n_samples in subset_sizes:
        print(f"\n--- Training 2D-FNO on N = {n_samples} samples ({epochs} epochs) ---")
        train_ds = MetasurfaceDataset(h5_path, split='train', max_samples=n_samples)
        train_loader = DataLoader(train_ds, batch_size=min(64, n_samples), shuffle=True)

        model = FNOSurrogate2D().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        t0 = time.time()
        for ep in range(1, epochs + 1):
            model.train()
            for x_b, y_b in train_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                pred = model(x_b)
                loss = F.mse_loss(pred, y_b) + 10.0 * compute_passivity_penalty(pred)
                loss.backward()
                optimizer.step()
            scheduler.step()

        elapsed = time.time() - t0

        # Evaluate on fixed test set
        model.eval()
        y_true_list, y_pred_list = [], []
        with torch.no_grad():
            for x_b, y_b in test_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                y_true_list.append(y_b.cpu().numpy())
                y_pred_list.append(pred.cpu().numpy())

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        mse = float(np.mean((y_true - y_pred)**2))
        var_y = float(np.var(y_true))
        nmse = float(mse / var_y)
        r2 = float(1.0 - nmse)

        entry = {
            "num_training_samples": n_samples,
            "training_time_s": elapsed,
            "test_mse": mse,
            "test_nmse": nmse,
            "test_r2": r2,
        }
        scaling_results.append(entry)
        print(f"  N={n_samples:5d} | Test NMSE: {nmse:.3e} | R^2: {r2:.5f} | Time: {elapsed:.1f} s")

    # Fit power law: log(NMSE) = log(C) - gamma * log(N)
    N_vals = np.array([res["num_training_samples"] for res in scaling_results])
    nmse_vals = np.array([res["test_nmse"] for res in scaling_results])

    log_N = np.log(N_vals)
    log_err = np.log(nmse_vals)
    poly = np.polyfit(log_N, log_err, 1)
    gamma = float(-poly[0])
    C_const = float(np.exp(poly[1]))

    print(f"\nPower-Law Fit: Test NMSE = {C_const:.3e} * N^(-{gamma:.3f})")

    out_file = os.path.join(outputs_dir, 'exp3_data_scaling_law.json')
    payload = {
        "experiment": "Data Scaling Law",
        "power_law_fit": {"gamma": gamma, "C": C_const, "formula": f"NMSE = {C_const:.3e} * N^(-{gamma:.3f})"},
        "points": scaling_results,
    }
    with open(out_file, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"Experiment 3 successfully exported to: {out_file}")


if __name__ == '__main__':
    run_data_scaling_study()
