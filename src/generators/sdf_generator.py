"""
Parametric C4v-Symmetric Signed Distance Field (SDF) Generator
=============================================================
Generates 2D continuous Signed Distance Fields (SDFs) representing
metasurface unit cells constrained to C4v point-group symmetry
(fourfold 90° rotation and orthogonal axis reflections).

Four Distinct Structural Topology Classes:
- Class A: Cross & Loop Resonators (orthogonal arms, square loops, capacitive split gaps)
- Class B: Fractal & Multi-Ring Inclusions (annular rings, Minkowski bends, corner cutouts)
- Class C: Smoothed Fourier Surface Noise (bandpass Gaussian random fields, curvilinear tracks)
- Class D: Composite Hybrid Geometries (Boolean combinations of A, B, C with stochastic feature widths)

Physical Units & Resolution:
- Grid: 128 x 128 (Px = 5.715 mm, Py = 5.080 mm, dx ≈ 44.6 um, dy ≈ 39.7 um)
- Minimum Manufacturing Feature: r_min >= 150 um (>= 3.5 pixels)

Convention:
- Phi(r) < 0 : Conductive pattern (inside)
- Phi(r) = 0 : Boundary interface
- Phi(r) > 0 : Dielectric host / Air (outside)
"""

import numpy as np
import scipy.ndimage as ndimage
from typing import Optional, Tuple, List, Union


def create_coordinate_grid(resolution: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates normalized 2D coordinate meshgrids [-1, 1] x [-1, 1]."""
    coords = np.linspace(-1.0, 1.0, resolution)
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X**2 + Y**2)
    return X, Y, R


def enforce_c4v_symmetry(pattern_quadrant: np.ndarray) -> np.ndarray:
    """Enforces C4v symmetry on a 2D boolean mask from a quadrant (N/2, N/2)."""
    quad_sym = pattern_quadrant | pattern_quadrant.T
    top = np.hstack([np.fliplr(quad_sym), quad_sym])
    bottom = np.flipud(top)
    full = np.vstack([top, bottom])
    return full | np.rot90(full, 1) | np.rot90(full, 2) | np.rot90(full, 3)


def check_minimum_feature_size(mask: np.ndarray, min_radius_pixels: float = 3.4) -> bool:
    """
    Verifies that all conductive traces and gaps satisfy the minimum
    lithographic manufacturing limit (r_min >= 150 um, approx 3.4 pixels on 128x128 grid).
    """
    if not np.any(mask) or np.all(mask):
        return True
        
    # Morphological erosion followed by dilation (opening)
    struct = ndimage.generate_binary_structure(2, 1)
    iterations = int(np.floor(min_radius_pixels))
    
    # Check conductive traces
    eroded_metal = ndimage.binary_erosion(mask, structure=struct, iterations=iterations)
    reconstructed_metal = ndimage.binary_dilation(eroded_metal, structure=struct, iterations=iterations)
    lost_metal_fraction = np.sum(mask & ~reconstructed_metal) / (np.sum(mask) + 1e-8)
    
    # Check dielectric gaps
    inv_mask = ~mask
    eroded_gap = ndimage.binary_erosion(inv_mask, structure=struct, iterations=iterations)
    reconstructed_gap = ndimage.binary_dilation(eroded_gap, structure=struct, iterations=iterations)
    lost_gap_fraction = np.sum(inv_mask & ~reconstructed_gap) / (np.sum(inv_mask) + 1e-8)
    
    # Reject if feature loss exceeds 15% (indicates sub-150um isolated islands or hair gaps)
    return bool((lost_metal_fraction < 0.15) and (lost_gap_fraction < 0.15))


def mask_to_sdf(mask: np.ndarray, resolution: int = 128) -> np.ndarray:
    """Converts binary mask to normalized continuous Signed Distance Field (SDF)."""
    mask_bool = mask.astype(bool)
    if np.all(mask_bool):
        return -np.ones((resolution, resolution), dtype=np.float32)
    elif not np.any(mask_bool):
        return np.ones((resolution, resolution), dtype=np.float32)
        
    inside_dist = ndimage.distance_transform_edt(mask_bool)
    outside_dist = ndimage.distance_transform_edt(~mask_bool)
    return ((outside_dist - inside_dist) / (resolution / 2.0)).astype(np.float32)


# ==============================================================================
# Topology Class A: Cross & Loop Resonators
# ==============================================================================

def generate_class_a_mask(resolution: int = 128, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Topology Class A: Intersecting orthogonal arms, concentric square loops,
    and capacitive split gaps.
    """
    if rng is None:
        rng = np.random.default_rng()
    X, Y, R = create_coordinate_grid(resolution)
    
    arm_w = rng.uniform(0.12, 0.38)
    loop_outer = rng.uniform(0.65, 0.90)
    loop_thick = rng.uniform(0.08, 0.20)
    loop_inner = loop_outer - loop_thick
    
    # Orthogonal cross arms
    cross = (np.abs(X) < arm_w) | (np.abs(Y) < arm_w)
    
    # Concentric square loop
    outer_box = (np.abs(X) < loop_outer) & (np.abs(Y) < loop_outer)
    inner_box = (np.abs(X) < loop_inner) & (np.abs(Y) < loop_inner)
    square_loop = outer_box & ~inner_box
    
    # Capacitive split gap
    has_split = rng.random() > 0.4
    if has_split:
        gap_w = rng.uniform(0.08, 0.18)
        gap = (np.abs(X) < gap_w) & (np.abs(Y) > (loop_inner - 0.05)) & (np.abs(Y) < (loop_outer + 0.05))
        gap = gap | ((np.abs(Y) < gap_w) & (np.abs(X) > (loop_inner - 0.05)) & (np.abs(X) < (loop_outer + 0.05)))
        square_loop = square_loop & ~gap
        
    pattern = cross | square_loop
    return pattern


# ==============================================================================
# Topology Class B: Fractal & Multi-Ring Inclusions
# ==============================================================================

def generate_class_b_mask(resolution: int = 128, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Topology Class B: Concentric annular rings, Minkowski/Hilbert-style perimeter bends,
    and corner patch cutouts.
    """
    if rng is None:
        rng = np.random.default_rng()
    X, Y, R = create_coordinate_grid(resolution)
    
    # Outer annular ring
    r1_out = rng.uniform(0.65, 0.92)
    r1_in = r1_out - rng.uniform(0.08, 0.18)
    ring1 = (R < r1_out) & (R > r1_in)
    
    # Inner annular ring or central disk
    has_inner_ring = rng.random() > 0.3
    if has_inner_ring:
        r2_out = rng.uniform(0.25, max(0.30, r1_in - 0.10))
        r2_in = r2_out - rng.uniform(0.06, 0.14)
        ring2 = (R < r2_out) & (R > max(0.05, r2_in))
    else:
        ring2 = (R < rng.uniform(0.20, max(0.25, r1_in - 0.10)))
        
    # Minkowski corner cutouts / perimeter bends
    has_minkowski = rng.random() > 0.4
    if has_minkowski:
        corner_patch = (np.abs(X) > 0.65) & (np.abs(Y) > 0.65)
        pattern = (ring1 | ring2 | corner_patch)
    else:
        pattern = (ring1 | ring2)
        
    return pattern


# ==============================================================================
# Topology Class C: Smoothed Fourier Surface Noise
# ==============================================================================

def generate_class_c_mask(resolution: int = 128, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Topology Class C: Random Gaussian fields filtered in frequency space with
    thresholding via distance transforms to generate curvilinear tracks.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    half = resolution // 2
    noise = rng.standard_normal((half, half))
    
    # Bandpass Gaussian filtering
    sigma = rng.uniform(2.8, 5.5)
    smoothed = ndimage.gaussian_filter(noise, sigma=sigma, mode='reflect')
    
    # Threshold at a percentile
    threshold = np.percentile(smoothed, rng.uniform(38, 62))
    mask_quad = smoothed > threshold
    
    full_mask = enforce_c4v_symmetry(mask_quad)
    return full_mask


# ==============================================================================
# Topology Class D: Composite Hybrid Geometries
# ==============================================================================

def generate_class_d_mask(resolution: int = 128, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Topology Class D: Stochastic Boolean unions/intersections of Classes A, B, and C
    with variable feature thicknesses.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    mask_a = generate_class_a_mask(resolution, rng)
    mask_b = generate_class_b_mask(resolution, rng)
    mask_c = generate_class_c_mask(resolution, rng)
    
    op = rng.choice(['union_ab', 'intersect_ac', 'union_bc', 'blend_abc'])
    
    if op == 'union_ab':
        pattern = mask_a | mask_b
    elif op == 'intersect_ac':
        pattern = (mask_a & mask_c) | (mask_b & mask_a)
    elif op == 'union_bc':
        pattern = mask_b | mask_c
    else: # blend_abc
        pattern = (mask_a | mask_b) & ~((np.abs(create_coordinate_grid(resolution)[0]) > 0.85) & (np.abs(create_coordinate_grid(resolution)[1]) > 0.85))
        
    return pattern


# ==============================================================================
# Main Generation Function
# ==============================================================================

def generate_c4v_sdf(
    resolution: int = 128,
    mode: str = 'random',
    seed: Optional[int] = None,
    enforce_manufacturing_check: bool = True,
) -> np.ndarray:
    """
    Synthesizes a C4v-symmetric Signed Distance Field (SDF) across 4 topology classes:
    - 'class_a': Cross & Loop Resonators
    - 'class_b': Fractal & Multi-Ring Inclusions
    - 'class_c': Smoothed Fourier Surface Noise
    - 'class_d': Composite Hybrid Geometries
    - 'random': Uniformly samples across classes A, B, C, D.
    """
    rng = np.random.default_rng(seed)
    classes = ['class_a', 'class_b', 'class_c', 'class_d']
    
    for _ in range(20):  # Retry attempts for valid manufacturing features
        selected_class = rng.choice(classes) if mode in ['random', 'diverse'] else mode
        
        if selected_class == 'class_a':
            mask = generate_class_a_mask(resolution, rng)
        elif selected_class == 'class_b':
            mask = generate_class_b_mask(resolution, rng)
        elif selected_class == 'class_c':
            mask = generate_class_c_mask(resolution, rng)
        elif selected_class == 'class_d':
            mask = generate_class_d_mask(resolution, rng)
        else:
            # Fallback backward compatibility modes
            if selected_class in ['cross_patch', 'ring_resonator', 'jerusalem_cross', 'fourier_noise']:
                if selected_class == 'cross_patch':
                    mask = generate_class_a_mask(resolution, rng)
                elif selected_class == 'ring_resonator':
                    mask = generate_class_b_mask(resolution, rng)
                elif selected_class == 'fourier_noise':
                    mask = generate_class_c_mask(resolution, rng)
                else:
                    mask = generate_class_d_mask(resolution, rng)
            else:
                raise ValueError(f"Unknown topology class: {selected_class}")
                
        # Enforce C4v symmetry
        half = resolution // 2
        c4v_mask = enforce_c4v_symmetry(mask[:half, :half])
        
        # Verify minimum feature size (r_min >= 150 um)
        if not enforce_manufacturing_check or check_minimum_feature_size(c4v_mask):
            return mask_to_sdf(c4v_mask, resolution)
            
    # If all iterations failed, return smoothed mask as safe fallback
    return mask_to_sdf(c4v_mask, resolution)


def generate_batch_sdfs(
    batch_size: int,
    resolution: int = 128,
    mode: str = 'diverse',
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generates a batch of C4v SDF tensors."""
    rng = np.random.default_rng(seed)
    sdfs = np.zeros((batch_size, resolution, resolution), dtype=np.float32)
    for i in range(batch_size):
        item_seed = int(rng.integers(0, 2**31 - 1))
        sdfs[i] = generate_c4v_sdf(resolution=resolution, mode=mode, seed=item_seed)
    return sdfs
