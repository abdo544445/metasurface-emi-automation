"""
Electromagnetic Metasurface Model & Stackup Specification
=========================================================
Defines the multi-layer composite stackup, conductivity mapping,
complex permittivity/permeability tensors, and shielding effectiveness
deconvolution formulas for X- and Ku-band metashields (8.2 - 18.0 GHz).
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# Physical fundamental constants (SI units)
EPSILON_0 = 8.8541878128e-12  # Vacuum permittivity [F/m]
MU_0 = 1.25663706212e-6       # Vacuum permeability [H/m]
C_0 = 299792458.0              # Speed of light in vacuum [m/s]
Z_0 = 376.730313668            # Free-space characteristic impedance [Ohms]


@dataclass
class LayerSpec:
    name: str
    thickness_m: float
    eps_r: complex
    mu_r: complex
    is_patterned: bool = False


@dataclass
class MetasurfaceStackup:
    """
    Standard 5-layer metashield stackup:
    1. Layer 1: Semi-infinite Air Superstrate (eps_r=1.0, mu_r=1.0)
    2. Layer 2: Patterned MXene/Carbon Metasurface (t=25 um, sigma_film=10^4 S/m)
    3. Layer 3: Polyimide Dielectric Spacer (t=50 um, eps_r=3.5 - j0.028, mu_r=1.0)
    4. Layer 4: Carbonyl Iron / Polyurethane Composite (t=1.10 mm, eps_r=4.2 - j1.8, mu_r=1.4 - j0.6)
    5. Layer 5: Semi-infinite Air Exit Substrate (eps_r=1.0, mu_r=1.0)
    """
    # Unit cell dimensions (meters)
    px: float = 5.715e-3  # 5.715 mm
    py: float = 5.080e-3  # 5.080 mm
    
    # Layer thicknesses (meters)
    t_meta: float = 25.0e-6    # 25 um
    t_spacer: float = 50.0e-6  # 50 um
    t_mag: float = 1.10e-3     # 1.10 mm
    
    # Material properties
    sigma_film: float = 1.0e4             # 10,000 S/m
    beta_heaviside: float = 4.0           # Projection sharpness
    eps_spacer: complex = 3.5 - 1j * (3.5 * 0.008)  # eps_r = 3.5, tan_delta = 0.008
    mu_spacer: complex = 1.0 + 0j
    eps_mag: complex = 4.2 - 1j * 1.8     # Carbonyl-iron composite eps
    mu_mag: complex = 1.4 - 1j * 0.6      # Carbonyl-iron composite mu
    mu_s_static: float = 1.4              # Static permeability for Rozanov limit
    
    @property
    def total_thickness_m(self) -> float:
        return self.t_meta + self.t_spacer + self.t_mag
        
    @property
    def total_thickness_mm(self) -> float:
        return self.total_thickness_m * 1e3


def sdf_to_conductivity(
    sdf: np.ndarray,
    sigma_film: float = 1.0e4,
    beta: float = 4.0
) -> np.ndarray:
    """
    Maps continuous Signed Distance Field Phi(r) to 2D surface conductivity sigma(r)
    via smoothed differentiable Heaviside projection.
    
    Convention:
    Phi < 0 : Conductive patch (sigma -> sigma_film)
    Phi = 0 : Boundary (sigma = sigma_film / 2)
    Phi > 0 : Dielectric / Air (sigma -> 0)
    
    Formula:
    sigma(r) = sigma_film / (1 + exp(2 * beta * Phi(r)))
    """
    # Clip exponent to avoid numerical overflow in float precision
    exp_arg = np.clip(2.0 * beta * sdf, -30.0, 30.0)
    sigma = sigma_film / (1.0 + np.exp(exp_arg))
    return sigma.astype(np.float32)


def conductivity_to_relative_permittivity(
    sigma: np.ndarray,
    freq_hz: float,
    eps_host: complex = 1.0 + 0j
) -> np.ndarray:
    """
    Converts 2D conductivity distribution sigma(r) into complex relative permittivity:
    eps_r(omega, r) = eps_host - j * sigma(r) / (omega * eps_0)
    """
    omega = 2.0 * np.pi * freq_hz
    sigma_term = sigma / (omega * EPSILON_0)
    return eps_host - 1j * sigma_term


def compute_shielding_metrics(
    S11: np.ndarray,
    S21: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Deconvolutes S-parameters into Power Reflection (R), Transmission (T),
    Absorption (A), and Shielding Effectiveness metrics (SE_R, SE_A, SE_T).
    
    Definitions:
    R = |S11|^2
    T = |S21|^2
    A = 1 - R - T
    SE_R = -10 * log10(1 - R)
    SE_A = -10 * log10(T / (1 - R))
    SE_T = -10 * log10(T) = SE_R + SE_A
    Absorption Ratio = SE_A / SE_T
    """
    # Magnitudes squared
    R = np.clip(np.abs(S11)**2, 0.0, 1.0)
    T = np.clip(np.abs(S21)**2, 1e-18, 1.0)
    A = np.clip(1.0 - R - T, 0.0, 1.0)
    
    # Shielding effectiveness in dB
    # Clip (1 - R) to avoid division by zero
    one_minus_R = np.clip(1.0 - R, 1e-12, 1.0)
    
    SE_R = -10.0 * np.log10(one_minus_R)
    SE_A = -10.0 * np.log10(np.clip(T / one_minus_R, 1e-18, 1.0))
    SE_T = -10.0 * np.log10(T)
    
    # Absorption dominance ratio
    SE_T_safe = np.where(SE_T > 1e-6, SE_T, 1e-6)
    absorption_ratio = np.clip(SE_A / SE_T_safe, 0.0, 1.0)
    
    return {
        "R": R,
        "T": T,
        "A": A,
        "SE_R": SE_R,
        "SE_A": SE_A,
        "SE_T": SE_T,
        "absorption_ratio": absorption_ratio,
    }
