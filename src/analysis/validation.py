"""
Data Integrity and Physics Validation Gates
===========================================
Implements mathematical and physical validation audits for metasurface designs:
1. Passivity and Energy Conservation (A(w) >= 0, R(w) + T(w) <= 1.0).
2. Rozanov Causality Figure of Merit (rho_R <= 1.0).
3. Polarization Degeneracy Symmetry Check (|S11_TE - S11_TM| < tol).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


def compute_rozanov_metric(
    freqs_ghz: np.ndarray,
    S11: np.ndarray,
    total_thickness_mm: float = 1.175,
    mu_s: float = 1.4,
) -> float:
    r"""
    Calculates the normalized Rozanov Figure of Merit (rho_R).
    
    Formula:
        \rho_R = | \int_{\lambda_min}^{\lambda_max} \ln|S11(\lambda)| d\lambda | / (2 * \pi^2 * \mu_s * d)
        
    Interpretation:
        \rho_R <= 1.0 : Physically realizable and causal.
        \rho_R > 1.0  : Violates Rozanov fundamental bound (unphysical hallucination).
        0.75 <= \rho_R <= 0.88 : Approaching theoretical optimum (Target requirement).
    """
    c0 = 299792458.0  # m/s
    freqs_hz = freqs_ghz * 1e9
    wavelengths_m = c0 / freqs_hz
    
    # Sort strictly in ascending wavelength order for numerical trapezoidal integration
    sort_indices = np.argsort(wavelengths_m)
    lam = wavelengths_m[sort_indices]
    
    # Avoid log(0) numerical singularities
    s11_mag = np.clip(np.abs(S11[sort_indices]), 1e-8, 1.0)
    integrand = np.log(s11_mag)
    
    # Trapezoidal integration across bandwidth
    integral_val = np.abs(np.trapezoid(integrand, lam) if hasattr(np, 'trapezoid') else np.trapz(integrand, lam))
    
    d_m = total_thickness_mm * 1e-3
    rozanov_limit = 2.0 * (np.pi**2) * mu_s * d_m
    
    rho_R = float(integral_val / rozanov_limit)
    return rho_R


def check_passivity(
    S11: np.ndarray,
    S21: np.ndarray,
    tol: float = 1e-4,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validates electromagnetic passivity across all frequency points:
    1. Reflection magnitude: |S11| <= 1.0 + tol
    2. Transmission magnitude: |S21| <= 1.0 + tol
    3. Power sum: |S11|^2 + |S21|^2 <= 1.0 + tol
    4. Absorption: A = 1 - |S11|^2 - |S21|^2 >= -tol
    """
    R = np.abs(S11)**2
    T = np.abs(S21)**2
    power_sum = R + T
    A = 1.0 - power_sum
    
    max_power = float(np.max(power_sum))
    min_absorption = float(np.min(A))
    
    is_passive = (max_power <= 1.0 + tol) and (min_absorption >= -tol)
    
    stats = {
        "max_power_sum": max_power,
        "min_absorption": min_absorption,
        "is_passive": bool(is_passive),
    }
    return bool(is_passive), stats


def check_c4v_symmetry(
    S11_te: np.ndarray,
    S11_tm: np.ndarray,
    tol: float = 1e-3,
) -> Tuple[bool, float]:
    """
    Validates polarization degeneracy (|S11_TE - S11_TM| < tol)
    guaranteed by C4v point-group symmetry.
    """
    max_diff = float(np.max(np.abs(S11_te - S11_tm)))
    is_symmetric = max_diff < tol
    return bool(is_symmetric), max_diff


def audit_sample(
    freqs_ghz: np.ndarray,
    S11_te: np.ndarray,
    S21_te: np.ndarray,
    S11_tm: Optional[np.ndarray] = None,
    S21_tm: Optional[np.ndarray] = None,
    total_thickness_mm: float = 1.175,
    mu_s: float = 1.4,
) -> Dict[str, Any]:
    """
    Performs full automated data integrity audit on a single simulation run.
    """
    # 1. Passivity check
    passivity_ok, passivity_stats = check_passivity(S11_te, S21_te)
    
    # 2. Rozanov causality check
    rho_R = compute_rozanov_metric(freqs_ghz, S11_te, total_thickness_mm, mu_s)
    rozanov_ok = (0.0 <= rho_R <= 1.0) and not np.isnan(rho_R) and not np.isinf(rho_R)
    
    # 3. Symmetry check (if TM data provided)
    if S11_tm is not None:
        symmetry_ok, sym_diff = check_c4v_symmetry(S11_te, S11_tm)
    else:
        symmetry_ok, sym_diff = True, 0.0
        
    all_passed = passivity_ok and rozanov_ok and symmetry_ok
    
    return {
        "valid": bool(all_passed),
        "passivity_passed": bool(passivity_ok),
        "rozanov_passed": bool(rozanov_ok),
        "symmetry_passed": bool(symmetry_ok),
        "rho_R": rho_R,
        "max_power_sum": passivity_stats["max_power_sum"],
        "min_absorption": passivity_stats["min_absorption"],
        "symmetry_max_diff": sym_diff,
    }
