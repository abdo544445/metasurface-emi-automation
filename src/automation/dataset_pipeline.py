"""
High-Throughput Parallel Batch Dataset Generation & Stratified Archival
=======================================================================
Orchestrates parallel simulation of C2v unit cells across 4 topology classes,
enforces automated quality gates (Passivity, Rozanov bound, Manufacturing resolution),
and exports chunked, compressed, stratified HDF5 archives (/train, /val, /test).
"""

import os
import time
import h5py
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from ..generators.sdf_generator import generate_c2v_sdf, generate_c4v_sdf
from .rcwa_solver import MetasurfaceRCWASolver
from .em_model import MetasurfaceStackup
from ..analysis.validation import audit_sample


def generate_single_validated_sample(
    seed: int,
    resolution: int = 128,
    solver: Optional[MetasurfaceRCWASolver] = None,
    mode: str = 'diverse',
) -> Optional[Dict[str, Any]]:
    """
    Generates and simulates a single unit cell, validating against all physical gates.
    """
    if solver is None:
        solver = MetasurfaceRCWASolver()
        
    rng = np.random.default_rng(seed)
    
    for attempt in range(15):
        sample_seed = int(rng.integers(0, 2**31 - 1))
        sdf = generate_c2v_sdf(resolution=resolution, mode=mode, seed=sample_seed)
        res = solver.solve(sdf, pol='both')

        
        audit = audit_sample(
            freqs_ghz=res["freqs_ghz"],
            S11_te=res["S11_TE"],
            S21_te=res["S21_TE"],
            S11_tm=res["S11_TM"],
            S21_tm=res["S21_TM"],
            total_thickness_mm=solver.stackup.total_thickness_mm,
            mu_s=solver.stackup.mu_s_static,
        )
        
        if audit["valid"]:
            return {
                "sdf": sdf.astype(np.float32),
                "s11_real": np.real(res["S11"]).astype(np.float32),
                "s11_imag": np.imag(res["S11"]).astype(np.float32),
                "s21_real": np.real(res["S21"]).astype(np.float32),
                "s21_imag": np.imag(res["S21"]).astype(np.float32),
                "rho_R": np.float32(audit["rho_R"]),
                "SE_T": res["metrics"]["SE_T"].astype(np.float32),
                "SE_A": res["metrics"]["SE_A"].astype(np.float32),
                "SE_R": res["metrics"]["SE_R"].astype(np.float32),
                "absorption": res["metrics"]["A"].astype(np.float32),
            }
            
    return None


def generate_stratified_dataset(
    total_samples: int = 1000,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    resolution: int = 128,
    output_h5_path: Optional[str] = None,
    solver: Optional[MetasurfaceRCWASolver] = None,
    seed: int = 2026,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Executes scaled batch generation and stores stratified HDF5 archives (/train, /val, /test).
    """
    if solver is None:
        solver = MetasurfaceRCWASolver()
        
    num_train = int(total_samples * train_ratio)
    num_val = int(total_samples * val_ratio)
    num_test = total_samples - num_train - num_val
    
    split_counts = {
        "train": num_train,
        "val": num_val,
        "test": num_test,
    }
    
    stratified_data = {
        split: {
            "sdf": np.zeros((count, resolution, resolution), dtype=np.float32),
            "s11_real": np.zeros((count, solver.num_freq_points), dtype=np.float32),
            "s11_imag": np.zeros((count, solver.num_freq_points), dtype=np.float32),
            "s21_real": np.zeros((count, solver.num_freq_points), dtype=np.float32),
            "s21_imag": np.zeros((count, solver.num_freq_points), dtype=np.float32),
            "rho_R": np.zeros(count, dtype=np.float32),
            "SE_T": np.zeros((count, solver.num_freq_points), dtype=np.float32),
        }
        for split, count in split_counts.items()
    }
    
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    if verbose:
        print(f"Generating {total_samples} stratified samples (Train: {num_train}, Val: {num_val}, Test: {num_test})...")
        
    for split, count in split_counts.items():
        filled = 0
        while filled < count:
            seed_i = int(rng.integers(0, 2**31 - 1))
            sample = generate_single_validated_sample(seed_i, resolution, solver, mode='diverse')
            if sample is not None:
                for key in stratified_data[split].keys():
                    stratified_data[split][key][filled] = sample[key]
                filled += 1
                
    elapsed = time.time() - start_time
    if verbose:
        print(f"Dataset generation completed in {elapsed:.2f} s ({elapsed/total_samples*1000:.1f} ms/sample).")
        
    if output_h5_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)
        with h5py.File(output_h5_path, 'w') as f:
            f.attrs["frequencies_ghz"] = solver.freqs_ghz
            f.attrs["total_thickness_mm"] = solver.stackup.total_thickness_mm
            f.attrs["mu_s_static"] = solver.stackup.mu_s_static
            
            for split, arrays in stratified_data.items():
                grp = f.create_group(split)
                for key, arr in arrays.items():
                    grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
                    
        if verbose:
            print(f"Saved stratified HDF5 dataset to: {output_h5_path}")
            
    return stratified_data
