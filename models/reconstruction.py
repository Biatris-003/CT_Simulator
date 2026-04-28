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
    
    

class ComparisonReconstruction:
    """Compare sparse vs dense reconstruction"""

    @staticmethod
    def reconstruct_fbp_from_sinograms(full_sinogram, sparse_sinogram, full_angles, sparse_angles, original=None, filter_name='ramp'):
        """Reconstruct full and sparse FBP images from supplied sinograms.

        Args:
            full_sinogram (np.ndarray): Dense/full sinogram
            sparse_sinogram (np.ndarray): Sparse sinogram
            full_angles (np.ndarray): Projection angles for the full sinogram
            sparse_angles (np.ndarray): Projection angles for the sparse sinogram
            original (np.ndarray | None): Optional reference image for NMSE
            filter_name (str): FBP filter name

        Returns:
            dict: full/sparse reconstructions, angles, and optional NMSE metrics
        """
        full_recon = SparseReconstruction.fbp_reconstruction(full_sinogram, full_angles, filter_name=filter_name)
        sparse_recon = SparseReconstruction.fbp_reconstruction(sparse_sinogram, sparse_angles, filter_name=filter_name)

        result = {
            'full_recon': full_recon,
            'sparse_recon': sparse_recon,
            'full_angles': full_angles,
            'sparse_angles': sparse_angles,
        }

        if original is not None:
            full_err = ComparisonReconstruction.compute_reconstruction_error(original, full_recon)
            sparse_err = ComparisonReconstruction.compute_reconstruction_error(original, sparse_recon)
            result['full_nmse'] = full_err['nmse']
            result['sparse_nmse'] = sparse_err['nmse']
            result['full_psnr'] = full_err['psnr']
            result['sparse_psnr'] = sparse_err['psnr']

        return result
    
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
        from skimage.metrics import mean_squared_error
        
        # 1. أهم خطوة: توحيد المقياس (Min-Max Normalization)
        # بنخلي قيم الصورتين بين 0 و 1 عشان الطرح يكون عادل
        def normalize(img):
            denom = (img.max() - img.min())
            return (img - img.min()) / denom if denom != 0 else img

        org_norm = normalize(original)
        rec_norm = normalize(reconstructed)
        
        # 2. حساب الـ MSE على الصور المتوحدة
        mse = mean_squared_error(org_norm, rec_norm)
        original_power = np.mean(org_norm ** 2)
        nmse = mse / original_power if original_power != 0 else mse
        
        # 3. حساب الـ PSNR
        # بما إننا وحدنا لـ 0-1، يبقى الـ peak هو 1
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        # 4. حساب الـ Error Map
        # دلوقتي الـ emap هتبين الـ Artifacts مش فرق الإضاءة
        error_map = (org_norm - rec_norm) ** 2
        
        return {
            'nmse': nmse,
            'psnr': psnr,
            'emap': error_map
        }
