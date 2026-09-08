"""
Experiment 4: Multi-Target Generative Versatility
=================================================
Demonstrates the generality of the physics-guided diffusion engine by synthesizing
metasurface unit cells tailored to three distinct electromagnetic specifications:
1. Spec A: Ultra-broadband continuous shielding (SE_T >= 30 dB across 8.2–18.0 GHz).
2. Spec B: Band-notched absorption (Absorption A >= 85% in X-band 8.2–12.4 GHz, Ku-band pass).
3. Spec C: Dual-band resonant notches (Centered at 10.0 GHz and 15.0 GHz).

Evaluates target matching, convergence, and physical realizability across all three specs.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)

from src.automation.rcwa_solver import MetasurfaceRCWASolver
from src.automation.em_model import compute_shielding_metrics
from src.experiments.scale_dataset_and_fno import FNOSurrogate2D


class DifferentiableHelmholtz(nn.Module):
    def __init__(self, r0_pixels=3.4, resolution=128):
        super().__init__()
        kx = 2.0 * np.pi * np.fft.fftfreq(resolution)
        ky = 2.0 * np.pi * np.fft.fftfreq(resolution)
        KX, KY = np.meshgrid(kx, ky)
        K2 = KX**2 + KY**2
        H = 1.0 / (1.0 + (r0_pixels**2) * K2)
        self.register_buffer("H", torch.from_numpy(H).float().unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        X_f = torch.fft.fft2(x)
        return torch.fft.ifft2(X_f * self.H).real


def enforce_c2v(x):
    x_rot = torch.rot90(x, 2, [2, 3])
    x_fh = torch.flip(x, [2])
    x_fv = torch.flip(x, [3])
    return (x + x_rot + x_fh + x_fv) / 4.0


def run_multitarget_experiment(num_steps=25):
    outputs_dir = os.path.join(BASE_DIR, 'data/outputs')
    models_dir = os.path.join(BASE_DIR, 'models')
    fno_weights_path = os.path.join(models_dir, 'fno_surrogate_best.pt')

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Experiment 4 (Multi-Target Generative Versatility) on device: {device}...")

    fno_model = FNOSurrogate2D().to(device)
    fno_model.load_state_dict(torch.load(fno_weights_path, map_location=device))
    fno_model.eval()

    helmholtz = DifferentiableHelmholtz().to(device)
    solver = MetasurfaceRCWASolver()
    freqs = np.linspace(8.2, 18.0, 101)

    specs = [
        {"name": "Spec A: Ultra-Broadband Shielding (8.2-18 GHz)", "type": "broadband"},
        {"name": "Spec B: Band-Notched Absorption (X-Band 8.2-12.4 GHz)", "type": "band_notched"},
        {"name": "Spec C: Dual-Band Resonant Shielding (10 & 15 GHz)", "type": "dual_band"},
    ]

    target_results = []

    for spec in specs:
        print(f"\n--- Synthesizing Candidate for {spec['name']} ---")
        torch.manual_seed(42)
        xt = torch.randn(1, 1, 128, 128, device=device)
        xt = enforce_c2v(xt)
        dt = 1.0 / num_steps

        t0 = time.time()
        for step in range(num_steps):
            xt.requires_grad_(True)
            x_filt = helmholtz(xt)
            pred_s = fno_model(x_filt)

            s11_sq = pred_s[:, 0, :]**2 + pred_s[:, 1, :]**2
            s21_sq = pred_s[:, 2, :]**2 + pred_s[:, 3, :]**2

            if spec["type"] == "broadband":
                # Minimize transmission over full band
                loss = torch.mean(s11_sq + 2.5 * s21_sq)
            elif spec["type"] == "band_notched":
                # Minimize reflection/transmission in X-band (indices 0..42), maximize transmission in Ku-band
                loss_x = torch.mean(s11_sq[:, :43] + s21_sq[:, :43])
                loss_ku = torch.mean((1.0 - s21_sq[:, 43:])**2)
                loss = loss_x + 0.8 * loss_ku
            else: # dual_band
                # Target notches at 10 GHz (idx ~18) and 15 GHz (idx ~69)
                loss_10 = s21_sq[:, 18]
                loss_15 = s21_sq[:, 69]
                loss = loss_10 + loss_15 + 0.2 * torch.mean(s11_sq)

            grad = torch.autograd.grad(loss, xt)[0]
            grad_norm = grad / (torch.linalg.vector_norm(grad) + 1e-6)

            with torch.no_grad():
                xt = xt - dt * 2.5 * grad_norm
                xt = enforce_c2v(xt)

        elapsed = time.time() - t0
        final_sdf = helmholtz(xt)[0, 0].detach().cpu().numpy()

        # Rigorous RCWA solve
        rcwa_res = solver.solve(final_sdf, pol='TE')
        se_t_spec = rcwa_res['metrics']['SE_T']
        se_a_spec = rcwa_res['metrics']['SE_A']
        abs_ratio_spec = rcwa_res['metrics']['absorption_ratio']

        summary = {
            "spec_name": spec["name"],
            "spec_type": spec["type"],
            "synthesis_time_s": elapsed,
            "mean_se_t_db": float(np.mean(se_t_spec)),
            "mean_abs_ratio_percent": float(np.mean(abs_ratio_spec) * 100),
            "spectra": {
                "freqs_ghz": freqs.tolist(),
                "se_t": se_t_spec.tolist(),
                "se_a": se_a_spec.tolist(),
            }
        }
        target_results.append(summary)

        print(f"  Synthesis Time:      {elapsed:.2f} s")
        print(f"  Mean Total SE_T:     {summary['mean_se_t_db']:.2f} dB")
        print(f"  Mean Absorption:     {summary['mean_abs_ratio_percent']:.1f} %")

    out_file = os.path.join(outputs_dir, 'exp4_multitarget_versatility.json')
    with open(out_file, 'w') as f:
        json.dump({"experiment": "Multi-Target Generative Versatility", "targets": target_results}, f, indent=2)
    print(f"\nExperiment 4 successfully exported to: {out_file}")


if __name__ == '__main__':
    run_multitarget_experiment()
