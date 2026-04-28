"""Utilities for building a 3-material CT phantom and attenuation map.

This module converts the Shepp-Logan phantom into three material classes:
- Air / background
- Soft tissue
- Bone

The attenuation coefficients depend on the selected kVp, while the spectrum
current (mA) is kept as a separate display/input value for the acquisition.
"""

from __future__ import annotations

import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def get_mu_for_material(material_id: int, kvp: float) -> float:
    """Return a simple kVp-dependent linear attenuation coefficient."""
    kvp = max(float(kvp), 1.0)
    e_eff = kvp * 0.4

    if material_id == 1:  # Soft tissue
        return 0.2 * (60.0 / e_eff) ** 0.5
    if material_id == 2:  # Bone
        return 0.6 * (60.0 / e_eff) ** 2.5 + 0.1
    return 0.0002  # Air / background


def build_three_material_phantom(size: int = 512):
    """Build a 3-material phantom map with material IDs 0, 1, and 2."""
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (size, size), anti_aliasing=True)

    material_map = np.zeros_like(phantom, dtype=np.uint8)
    material_map[(phantom > 0.1) & (phantom <= 0.9)] = 1  # Soft tissue
    material_map[phantom > 0.9] = 2  # Bone
    return material_map


def build_three_material_mu_map(size: int = 512, kvp: float = 120.0):
    """Build a 3-material attenuation map from the Shepp-Logan phantom."""
    material_map = build_three_material_phantom(size=size)

    mu_air = get_mu_for_material(0, kvp)
    mu_soft_tissue = get_mu_for_material(1, kvp)
    mu_bone = get_mu_for_material(2, kvp)

    mu_map = np.zeros_like(material_map, dtype=np.float32)
    mu_map[material_map == 0] = mu_air
    mu_map[material_map == 1] = mu_soft_tissue
    mu_map[material_map == 2] = mu_bone

    return material_map, mu_map
