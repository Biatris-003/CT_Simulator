import numpy as np
from skimage.transform import radon


def generate_spectrum_physics(kVp, mA, Cu=0.0, Al=0.0):
    """Generate a simple X-ray spectrum and return energies, spectrum and total_i0.

    This is a lightweight, approximate model intended for simulation/visualisation.
    """
    kVp = float(kVp)
    mA = float(mA)

    energies = np.arange(2.0, kVp + 1.0, 1.0)
    if energies.size == 0:
        return energies, np.array([]), 0.0

    # Simple empirical attenuation proxies for Al and Cu
    mu_al = 40.0 * (energies / 10.0) ** (-3.2) + 0.17
    mu_cu = np.where(
        energies < 8.98,
        28000.0 / (energies ** 3) + 0.15,
        220.0 * (energies / 10.0) ** (-3.0) + 0.15,
    )

    # Kramers-like continuous bremsstrahlung shape
    intensities = (kVp - energies) / energies
    intensities = np.clip(intensities, 0.0, None)

    # Add characteristic peaks for high kVp
    if kVp > 69.5:
        scale = ((kVp - 69.5) / 30.0) ** 1.6
        k_alpha = 1.5 * scale * np.exp(-0.5 * ((energies - 59.0) / 1.0) ** 2)
        k_beta = 0.4 * scale * np.exp(-0.5 * ((energies - 67.5) / 1.0) ** 2)
        intensities += k_alpha + k_beta

    # Intrinsic and added filtration
    att_intrinsic = mu_al * 2.7 * 0.15
    att_added = (mu_al * 2.7 * (Al / 10.0)) + (mu_cu * 8.96 * (Cu / 10.0))

    final_intensities = intensities * np.exp(-att_intrinsic) * np.exp(-att_added)
    # Scale by current (mA) and an arbitrary factor for visibility
    final_intensities = final_intensities * mA * 6.5e4

    total_i0 = float(np.sum(final_intensities))
    return energies, final_intensities, total_i0

def generate_physics_sinogram(mu_map, total_i0, user_step_angle, dx=0.1):
    # --- 1. توليد الـ Reference ---
    angles_ref = np.arange(0, 180, 1.0)
    ideal_sino_ref = radon(mu_map, theta=angles_ref) * dx
    
    np.random.seed(42) # حطي الـ seed هنا قبل أول نويز
    intensity_ref = total_i0 * np.exp(-ideal_sino_ref)
    noisy_intensity_ref = np.random.poisson(intensity_ref).astype(np.float32)
    noisy_intensity_ref[noisy_intensity_ref <= 0] = 1.0
    noisy_sino_ref = -np.log(noisy_intensity_ref / total_i0)

    # --- 2. توليد الـ Variable ---
    angles_var = np.arange(0, 180, user_step_angle)
    ideal_sino_var = radon(mu_map, theta=angles_var) * dx
    
    np.random.seed(42) # وعيدي الـ seed هنا تاني قبل تاني نويز
    intensity_var = total_i0 * np.exp(-ideal_sino_var)
    noisy_intensity_var = np.random.poisson(intensity_var).astype(np.float32)
    noisy_intensity_var[noisy_intensity_var <= 0] = 1.0
    noisy_sino_var = -np.log(noisy_intensity_var / total_i0)

    return noisy_sino_ref, noisy_sino_var, angles_ref, angles_var