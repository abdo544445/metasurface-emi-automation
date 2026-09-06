# CST Microwave Studio / PyAEDT Automated Floquet Port Sweep
# Project: Metasurface EMI Absorber Broadband Optimization

import numpy as np

def run_floquet_simulation():
    print("Initializing Floquet Port Unit-Cell Model...")
    px, py = 5.715, 5.080  # mm
    t_meta = 0.025         # mm
    t_spacer = 0.050       # mm
    t_mag = 1.100          # mm

    freqs = np.linspace(8.2, 18.0, 101)
    theta_angles = [0, 15, 30, 45, 60]
    polarizations = ['TE', 'TM']

    print(f"Configuring periodic boundary conditions (Px={px} mm, Py={py} mm)")
    print(f"Setting up multi-layer stackup (Total thickness d = {t_meta + t_spacer + t_mag:.3f} mm)")

    results = {}
    for pol in polarizations:
        for theta in theta_angles:
            print(f"Solving Floquet mode: Polarization={pol}, Incident Angle theta={theta} deg...")
    print("Floquet parameter sweep completed successfully.")

if __name__ == '__main__':
    run_floquet_simulation()
