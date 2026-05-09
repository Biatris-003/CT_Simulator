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
    def compute_reconstruction_error(original, reconstructed):
        from skimage.metrics import mean_squared_error
        
        def normalize(img):
            denom = (img.max() - img.min())
            return (img - img.min()) / denom if denom != 0 else img

        org_norm = normalize(original)
        rec_norm = normalize(reconstructed)
        
        mse = mean_squared_error(org_norm, rec_norm)
        original_power = np.mean(org_norm ** 2)
        nmse = mse / original_power if original_power != 0 else mse
        
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        error_map = (org_norm - rec_norm) ** 2
        
        return {
            'nmse': nmse,
            'psnr': psnr,
            'emap': error_map
        }
