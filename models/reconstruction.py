"""
Reconstruction algorithms for CT imaging

This module contains various reconstruction algorithms including:
- Simple BP (Simple Back Projection) reconstruction
- Sparse FBP (Filtered Back Projection) reconstruction
- Dense FBP reconstruction
"""

import numpy as np
from scipy import ndimage
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom


class SparseReconstruction:
    """Handles sparse CT reconstruction using BP and FBP"""
    
    @staticmethod
    def generate_sparse_projections(num_projections=36, angle_step=10):
        """
        Generate sparse projection angles.
        
        Args:
            num_projections (int): Number of projections to generate
            angle_step (float): Angular step size in degrees
            
        Returns:
            np.ndarray: Array of projection angles in degrees
        """
        angles = np.arange(0, 360, angle_step)[:num_projections]
        return angles
    
    @staticmethod
    def compute_raw_sinogram(phantom, angles):
        """
        Compute raw projections (sinogram) for given angles using Radon transform.
        
        **Use this when you have a loaded phantom (e.g., from main window):**
        - Takes pre-loaded phantom as input (don't create a new one)
        - Computes line integrals through the phantom
        - Returns raw sinogram without spectrum effects
        
        Args:
            phantom (np.ndarray): 2D phantom image (already loaded from main window)
            angles (np.ndarray): Array of projection angles in degrees
            
        Returns:
            tuple: (sinogram, angles) - raw projection data and angles used
        """
        sinogram = radon(phantom, theta=angles, circle=True)
        return sinogram, angles
    
    @staticmethod
    def compute_sparse_sinogram(phantom, angles):
        """
        Compute projections (sinogram) for sparse angles - DEPRECATED.
        Use compute_raw_sinogram instead for consistency.
        
        Args:
            phantom (np.ndarray): 2D phantom image
            angles (np.ndarray): Array of projection angles
            
        Returns:
            tuple: (sinogram, angles) - projection data and angles used
        """
        return SparseReconstruction.compute_raw_sinogram(phantom, angles)
    
    @staticmethod
    def simple_bp_reconstruction(sinogram, angles):
        """
        Reconstruct image using Simple Back Projection (SBP - unfiltered).
        Creates blurry reconstruction without filtering.
        
        Args:
            sinogram (np.ndarray): Projection data
            angles (np.ndarray): Projection angles in degrees
            
        Returns:
            np.ndarray: Reconstructed image (blurry)
        """
        # Use iradon with no filter (equivalent to simple backprojection)
        reconstructed = iradon(sinogram, theta=angles, filter_name=None)
        return reconstructed
    
    @staticmethod
    def fbp_reconstruction(sinogram, angles, filter_name='ramp'):
        """
        Reconstruct image using Filtered Back Projection (FBP).
        
        Args:
            sinogram (np.ndarray): Projection data
            angles (np.ndarray): Projection angles in degrees
            filter_name (str): Type of filter ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann')
            
        Returns:
            np.ndarray: Reconstructed image
        """
        reconstructed = iradon(sinogram, theta=angles, filter_name=filter_name)
        return reconstructed
    
    @staticmethod
    def apply_high_pass_filter(image, kernel_size=5, strength=1.0):
        """
        Apply high-pass filter to reduce noise in reconstructed image.
        
        Args:
            image (np.ndarray): Reconstructed image
            kernel_size (int): Size of high-pass kernel (must be odd)
            strength (float): Strength of high-pass filter (0.0 to 1.0)
            
        Returns:
            np.ndarray: High-pass filtered image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian blur
        blurred = ndimage.gaussian_filter(image, sigma=kernel_size/4)
        
        # High-pass: original - blurred
        high_pass = image - blurred
        
        # Blend with original: result = original + strength * high_pass
        filtered = image + strength * high_pass
        
        return filtered
    
    @staticmethod
    def sparse_fbp(num_projections=36, angle_step=10, noise_level=0.0, 
                   filter_name='ramp', method='fbp', high_pass=False, 
                   hp_strength=1.0, image_size=None):
        """
        Full sparse reconstruction pipeline using BP or FBP.
        
        Args:
            num_projections (int): Number of sparse projections (e.g., 36)
            angle_step (float): Angular step size in degrees (e.g., 10)
            noise_level (float): Gaussian noise standard deviation
            filter_name (str): FBP filter type ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann')
            method (str): 'sbp' for simple BP or 'fbp' for filtered BP
            high_pass (bool): Whether to apply high-pass filter
            hp_strength (float): High-pass filter strength (0.0 to 1.0)
            image_size (tuple): Output image size (uses phantom size if None)
            
        Returns:
            tuple: (phantom, sinogram, reconstructed, angles)
        """
        # 1. Generate phantom
        phantom = shepp_logan_phantom()
        if image_size:
            from skimage.transform import resize
            phantom = resize(phantom, image_size)
        
        # 2. Generate sparse angles
        angles = SparseReconstruction.generate_sparse_projections(
            num_projections, angle_step
        )
        
        # 3. Compute sparse projections
        sinogram, angles = SparseReconstruction.compute_sparse_sinogram(
            phantom, angles
        )
        
        # 4. Add optional noise
        if noise_level > 0:
            sinogram += np.random.normal(0, noise_level, sinogram.shape)
        
        # 5. Reconstruct with chosen method
        if method.lower() == 'sbp':
            reconstructed = SparseReconstruction.simple_bp_reconstruction(
                sinogram, angles
            )
        else:  # fbp
            reconstructed = SparseReconstruction.fbp_reconstruction(
                sinogram, angles, filter_name=filter_name
            )
        
        # 6. Apply high-pass filter if requested
        if high_pass:
            reconstructed = SparseReconstruction.apply_high_pass_filter(
                reconstructed, strength=hp_strength
            )
        
        return phantom, sinogram, reconstructed, angles
    
    @staticmethod
    def dense_fbp(num_projections=360, angle_step=1, noise_level=0.0, 
                  filter_name='ramp', method='fbp', high_pass=False,
                  hp_strength=1.0):
        """
        Dense reconstruction for comparison (good quality).
        
        Args:
            num_projections (int): Number of projections (typically 360)
            angle_step (float): Angular step size in degrees
            noise_level (float): Gaussian noise standard deviation
            filter_name (str): FBP filter type
            method (str): 'sbp' or 'fbp'
            high_pass (bool): Whether to apply high-pass filter
            hp_strength (float): High-pass filter strength
            
        Returns:
            tuple: (phantom, sinogram, reconstructed, angles)
        """
        # Generate phantom
        phantom = shepp_logan_phantom()
        
        # Generate dense angles
        angles = np.arange(0, 360, angle_step)[:num_projections]
        
        # Compute projections
        sinogram = radon(phantom, theta=angles, circle=True)
        
        # Add optional noise
        if noise_level > 0:
            sinogram += np.random.normal(0, noise_level, sinogram.shape)
        
        # Reconstruct
        if method.lower() == 'sbp':
            reconstructed = SparseReconstruction.simple_bp_reconstruction(
                sinogram, angles
            )
        else:  # fbp
            reconstructed = SparseReconstruction.fbp_reconstruction(
                sinogram, angles, filter_name=filter_name
            )
        
        # Apply high-pass filter if requested
        if high_pass:
            reconstructed = SparseReconstruction.apply_high_pass_filter(
                reconstructed, strength=hp_strength
            )
        
        return phantom, sinogram, reconstructed, angles
    
    @staticmethod
    def create_material_map(phantom, material_densities=None):
        """
        Convert phantom image to material attenuation map.
        
        This is realistic because in real CT:
        - Each voxel contains a material with known density
        - Attenuation coefficient depends on material density and energy
        
        Args:
            phantom (np.ndarray): 2D phantom image (intensity values 0-1 or 0-255)
            material_densities (dict): Maps intensity ranges to attenuation coefficients
                                      e.g., {(0.0, 0.2): 0.02, (0.2, 0.8): 0.1, ...}
                                      Default: maps phantom values to realistic HU attenuation
            
        Returns:
            np.ndarray: 2D attenuation map (same shape as phantom)
        """
        if phantom.max() > 1.0:
            # Normalize to 0-1 range
            phantom_norm = phantom / phantom.max()
        else:
            phantom_norm = phantom.copy()
        
        if material_densities is None:
            # Default: map phantom intensity to material attenuation
            # Low values (dark) = air/tissue ~ 0.01-0.02 cm^-1
            # Medium values = soft tissue ~ 0.15-0.20 cm^-1  
            # High values (bright) = bone ~ 0.30-0.40 cm^-1
            attenuation_map = (
                0.01 +  # baseline for air/low density
                phantom_norm * 0.35  # scale by intensity to match density variation
            )
        else:
            # Apply material densities based on intensity ranges
            attenuation_map = np.zeros_like(phantom_norm)
            for intensity_range, mu_value in material_densities.items():
                mask = (phantom_norm >= intensity_range[0]) & (phantom_norm < intensity_range[1])
                attenuation_map[mask] = mu_value
        
        return attenuation_map
    
    @staticmethod
    def compute_sinogram_from_attenuation(attenuation_map, angles):
        """
        Generate sinogram directly from attenuation map using Radon transform.
        
        **This is the REALISTIC approach for real CT systems:**
        - In real CT: You DON'T have a "phantom" image
        - You have: attenuation coefficients for materials at each location
        - The Radon transform computes line integrals through these coefficients
        - This represents: total attenuation along each X-ray path
        
        The formula: I_detected = I_0 * e^(-∫μ(x,y) dl)
        where ∫μ(x,y) dl is the line integral (computed by Radon transform)
        
        Args:
            attenuation_map (np.ndarray): 2D array of linear attenuation coefficients
            angles (np.ndarray): Projection angles in degrees
            
        Returns:
            tuple: (sinogram, angles) where sinogram[detector_pixel, angle] contains
                   the integrated attenuation along that ray
        """
        # Compute line integrals through the attenuation map
        # Each element in sinogram = sum of attenuation coefficients along that ray
        sinogram = radon(attenuation_map, theta=angles, circle=True)
        
        return sinogram, angles
    
    @staticmethod
    def apply_spectrum_to_sinogram(sinogram, spectrum, energies, kVp=100, mA=1):
        """
        Apply realistic X-ray spectrum effects to raw sinogram.
        
        **Real CT System Process:**
        1. Raw sinogram is computed from phantom (line integrals)
        2. Different photon energies are attenuated differently
        3. Low-energy photons are attenuated more (beam hardening)
        4. Detected intensity depends on spectrum and tube current
        
        Args:
            sinogram (np.ndarray): Raw sinogram from Radon transform
            spectrum (np.ndarray): Photon fluence for each energy (from SpectraToolDialog)
            energies (np.ndarray): Photon energies in keV (from SpectraToolDialog)
            kVp (float): Tube voltage in kV (from SpectraToolDialog)
            mA (float): Tube current in mA (from SpectraToolDialog)
            
        Returns:
            np.ndarray: Detected sinogram with spectrum effects applied
        """
        if spectrum is None or len(spectrum) == 0:
            # If no spectrum provided, just scale by current
            return sinogram * mA
        
        # Normalize spectrum to probability distribution
        spectrum_norm = spectrum / np.sum(spectrum)
        
        # Energy-dependent attenuation: higher energies penetrate better
        # Calculate average penetration factor across spectrum
        # Photons with E < 20 keV are heavily attenuated
        # Photons with E > 50 keV penetrate well
        energy_weights = np.ones_like(energies)
        energy_weights[energies < 20] = 0.3  # Low energy: high attenuation
        energy_weights[(energies >= 20) & (energies < 50)] = 0.7  # Medium
        energy_weights[energies >= 50] = 0.95  # High energy: low attenuation
        
        # Weighted average penetration across spectrum
        avg_penetration = np.sum(spectrum_norm * energy_weights)
        
        # Convert attenuation line integrals to detected photon counts
        # I_detected = I_0 * e^(-μ*d) ≈ I_0 * (1 - μ*d) for small μ*d
        # But use exponential for realistic beam hardening
        try:
            detected_sinogram = mA * avg_penetration * np.exp(-sinogram / 100.0)
        except:
            # Fallback if exponential fails
            detected_sinogram = sinogram * mA * avg_penetration
        
        return detected_sinogram
    
    @staticmethod
    def process_with_spectrum(loaded_phantom, angles, spectrum=None, energies=None,
                              kVp=100, mA=1, filter_name='ramp', method='fbp',
                              noise_level=0.0, high_pass=False, hp_strength=1.0):
        """
        **MAIN WORKFLOW FOR REALISTIC CT RECONSTRUCTION:**
        
        Process a loaded phantom with applied spectrum settings using attenuation map.
        This is the realistic approach that mimics real CT systems:
        
        1. Takes a pre-loaded phantom (from main window)
        2. Converts phantom to ATTENUATION MAP (material-based, depends on density)
        3. Computes raw sinogram from attenuation map (line integrals through materials)
        4. Applies X-ray spectrum effects based on kVp, mA, and spectrum
        5. Reconstructs image using selected algorithm
        
        **Why this is realistic:**
        - Real CT: You scan materials with known attenuation coefficients (μ)
        - NOT the intensity - the ATTENUATION (how much X-rays are absorbed)
        - Attenuation depends on MATERIAL TYPE and DENSITY, NOT on kVp/mA
        - Spectrum (kVp, mA) affects DETECTED INTENSITY, not attenuation coefficients
        - Low-energy photons are attenuated more (beam hardening)
        
        **Phantom → Attenuation Map → Sinogram → Detected Sinogram → Reconstruction**
        
        Args:
            loaded_phantom (np.ndarray): Pre-loaded phantom from main window (self.fantom)
            angles (np.ndarray): Projection angles in degrees
            spectrum (np.ndarray): Photon fluence array (from chosen_spectrum callback)
            energies (np.ndarray): Photon energy bins in keV (from chosen_spectrum callback)
            kVp (float): Tube voltage in kV (from chosen_spectrum callback)
            mA (float): Tube current in mA (from chosen_spectrum callback)
            filter_name (str): FBP filter type ('ramp', 'shepp-logan', etc.)
            method (str): Reconstruction method ('fbp' or 'sbp')
            noise_level (float): Optional Gaussian noise level
            high_pass (bool): Apply high-pass filter
            hp_strength (float): High-pass filter strength
            
        Returns:
            dict: {
                'phantom': loaded phantom (for reference),
                'attenuation_map': material-based attenuation map,
                'raw_sinogram': sinogram from attenuation map (before spectrum effects),
                'detected_sinogram': sinogram with spectrum effects applied,
                'reconstructed': final reconstructed image,
                'angles': projection angles used
            }
        """
        # Step 1: Generate angles if not provided
        if angles is None:
            angles = SparseReconstruction.generate_sparse_projections(36, 10)
        
        # Step 2: Convert loaded phantom to ATTENUATION MAP
        # This maps phantom intensity to material attenuation coefficients (μ)
        # Based on material density, NOT on spectrum settings
        attenuation_map = SparseReconstruction.create_material_map(loaded_phantom)
        
        # Step 3: Compute raw sinogram from ATTENUATION MAP (line integrals through materials)
        # This is what the detector would measure before any spectrum effects
        raw_sinogram, angles = SparseReconstruction.compute_sinogram_from_attenuation(
            attenuation_map, angles
        )
        
        # Step 4: Add optional noise to raw sinogram
        if noise_level > 0:
            raw_sinogram = raw_sinogram + np.random.normal(0, noise_level, raw_sinogram.shape)
        
        # Step 5: Apply spectrum effects if spectrum is available
        # This models beam hardening, energy-dependent attenuation, etc.
        if spectrum is not None and energies is not None:
            detected_sinogram = SparseReconstruction.apply_spectrum_to_sinogram(
                raw_sinogram, spectrum, energies, kVp, mA
            )
        else:
            # If no spectrum provided, just scale by mA
            detected_sinogram = raw_sinogram * mA
        
        # Step 6: Reconstruct using selected method on DETECTED sinogram
        if method.lower() == 'sbp':
            reconstructed = SparseReconstruction.simple_bp_reconstruction(
                detected_sinogram, angles
            )
        else:  # fbp
            reconstructed = SparseReconstruction.fbp_reconstruction(
                detected_sinogram, angles, filter_name=filter_name
            )
        
        # Step 7: Apply high-pass filter if requested
        if high_pass:
            reconstructed = SparseReconstruction.apply_high_pass_filter(
                reconstructed, strength=hp_strength
            )
        
        return {
            'phantom': loaded_phantom,
            'attenuation_map': attenuation_map,
            'raw_sinogram': raw_sinogram,
            'detected_sinogram': detected_sinogram,
            'reconstructed': reconstructed,
            'angles': angles
        }


class ComparisonReconstruction:
    """Compare sparse vs dense reconstruction"""
    
    @staticmethod
    def compare_sparse_vs_dense(num_sparse_proj=36, sparse_step=10, 
                                num_dense_proj=360, dense_step=1, 
                                noise_level=0.0, method='fbp', filter_name='ramp',
                                high_pass=False, hp_strength=1.0):
        """
        Generate both sparse and dense reconstructions for comparison.
        
        Args:
            num_sparse_proj (int): Number of sparse projections
            sparse_step (float): Sparse angle step
            num_dense_proj (int): Number of dense projections (360 for full coverage)
            dense_step (float): Dense angle step (1 for complete coverage)
            noise_level (float): Noise level for both
            method (str): 'sbp' or 'fbp'
            filter_name (str): Filter type for FBP
            high_pass (bool): Whether to apply high-pass filter
            hp_strength (float): High-pass filter strength
            
        Returns:
            dict: Dictionary containing both reconstruction results
        """
        sparse_results = SparseReconstruction.sparse_fbp(
            num_sparse_proj, sparse_step, noise_level, filter_name, 
            method, high_pass, hp_strength
        )
        
        dense_results = SparseReconstruction.dense_fbp(
            num_dense_proj, dense_step, noise_level, filter_name,
            method, high_pass, hp_strength
        )
        
        return {
            'sparse': {
                'phantom': sparse_results[0],
                'sinogram': sparse_results[1],
                'reconstructed': sparse_results[2],
                'angles': sparse_results[3],
                'num_projections': num_sparse_proj,
                'angle_step': sparse_step
            },
            'dense': {
                'phantom': dense_results[0],
                'sinogram': dense_results[1],
                'reconstructed': dense_results[2],
                'angles': dense_results[3],
                'num_projections': num_dense_proj,
                'angle_step': dense_step
            }
        }
    
    @staticmethod
    def compute_reconstruction_error(original, reconstructed):
        """
        Compute reconstruction error metrics.
        
        Args:
            original (np.ndarray): Original phantom image
            reconstructed (np.ndarray): Reconstructed image
            
        Returns:
            dict: Error metrics (NMSE, PSNR)
        """
        from skimage.metrics import mean_squared_error
        
        # Normalize MSE (NMSE)
        mse = mean_squared_error(original, reconstructed)
        original_power = np.mean(original ** 2)
        nmse = mse / original_power if original_power != 0 else mse
        
        # Peak Signal-to-Noise Ratio (PSNR)
        max_val = np.max(original)
        min_val = np.min(original)
        peak = max_val - min_val
        psnr = 20 * np.log10(peak / np.sqrt(mse)) if mse > 0 else float('inf')
        
        return {
            'nmse': nmse,
            'psnr': psnr
        }
