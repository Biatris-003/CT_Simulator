"""
Iterative Reconstruction Technique (SIRT) - Numerically Stable Implementation.

This implements the proper SIRT algorithm with numerical stability improvements:
1. Normalized sinogram input
2. Normalized backprojection (divided by number of angles)
3. Small damping factor to control convergence
4. Value clipping to keep results physically meaningful
5. Decreasing step size over iterations
6. Proper forward/backward projection consistency
"""

import numpy as np
from skimage.transform import radon, iradon


class IterativeReconstruction:
    """Numerically stable SIRT reconstruction with proper iterative loop"""

    @staticmethod
    def fbp_reconstruction(sinogram, angles, filter_name='ramp'):
        """
        FBP reconstruction for initial guess.
        
        Args:
            sinogram (np.ndarray): Input sinogram
            angles (np.ndarray): Projection angles in degrees
            filter_name (str): Filter type ('ramp', 'shepp-logan', etc.)
            
        Returns:
            np.ndarray: Reconstructed image
        """
        return iradon(sinogram, theta=angles, filter_name=filter_name)

    @staticmethod
    def sirt_reconstruction(sinogram, angles, iterations=10, damping_factor=0.03, 
                           verbose=False):
        """
        SIRT (Simultaneous Iterative Reconstruction Technique) - Numerically Stable.
        
        Proper SIRT algorithm with stability improvements:
        
        1. Normalize sinogram
        2. x_0 = FBP(sinogram)
        3. For each iteration n:
           - Forward project: p_n = Radon(x_n)
           - Compute error: err = sinogram - p_n
           - Back-project: correction = Radon^T(err) / num_angles
           - Decreasing step: lambda_n = damping_factor / (1 + 0.1*n)
           - Clip to physical range: x = clip(x_n + lambda_n * correction, 0, inf)
        
        Args:
            sinogram (np.ndarray): Measured sinogram
            angles (np.ndarray): Projection angles in degrees
            iterations (int): Number of iterations (1-100, default 10)
            damping_factor (float): Initial step size (0.01-0.1, default 0.03)
                Smaller values = more stable but slower
            verbose (bool): Print convergence info per iteration
            
        Returns:
            np.ndarray: Reconstructed image
        """
        # Ensure valid parameters
        iterations = max(1, min(int(iterations), 100))
        damping_factor = np.clip(float(damping_factor), 0.001, 0.2)
        angles = np.asarray(angles)
        sinogram = sinogram.astype(np.float32)
        
        # =====================================================
        # REQUIREMENT 4: Normalize input sinogram
        # =====================================================
        sino_max = np.max(np.abs(sinogram)) + 1e-8
        sinogram_normalized = sinogram / sino_max
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIRT Reconstruction Started")
            print(f"{'='*70}")
            print(f"Image shape: {sinogram.shape}")
            print(f"Number of angles: {len(angles)}")
            print(f"Iterations: {iterations}")
            print(f"Initial damping factor: {damping_factor}")
            print(f"Sinogram normalized (max={sino_max:.4e})")
            print(f"{'='*70}")
        
        # =====================================================
        # REQUIREMENT 8: Initialize with FBP
        # =====================================================
        x = IterativeReconstruction.fbp_reconstruction(
            sinogram_normalized, angles, filter_name='ramp'
        )
        x = x.astype(np.float32)
        
        # =====================================================
        # REQUIREMENT 3: Clip initial FBP to physical range
        # =====================================================
        x = np.clip(x, 0, None)
        
        if verbose:
            print(f"\nFBP Initial Guess:")
            print(f"  min={x.min():.6f}, max={x.max():.6f}, "
                  f"mean={x.mean():.6f}, std={x.std():.6f}")
        
        # =====================================================
        # SIRT ITERATION LOOP
        # =====================================================
        for iter_n in range(iterations):
            
            # =====================================================
            # REQUIREMENT 5: Forward projection with consistent settings
            # =====================================================
            forward_proj = radon(x, theta=angles, circle=True)
            
            # Compute error in sinogram space
            error_sino = sinogram_normalized - forward_proj
            
            # =====================================================
            # REQUIREMENT 5: Back-projection with consistent settings
            # =====================================================
            correction = iradon(
                error_sino,
                theta=angles,
                filter_name=None,
                circle=True,
                output_size=x.shape[0]
            )
            correction = correction.astype(np.float32)
            
            # =====================================================
            # REQUIREMENT 1: Normalize backprojection by number of angles
            # =====================================================
            correction = correction / len(angles)
            
            # =====================================================
            # REQUIREMENT 6: Decreasing step size over iterations
            # =====================================================
            lambda_n = damping_factor / (1.0 + iter_n * 0.1)
            
            # =====================================================
            # Update with damped correction
            # =====================================================
            x = x + lambda_n * correction
            
            # =====================================================
            # REQUIREMENT 3: Clip to physical range (non-negative)
            # =====================================================
            x = np.clip(x, 0, None)
            
            # =====================================================
            # REQUIREMENT 7: Debug prints for monitoring convergence
            # =====================================================
            if verbose:
                error_norm = np.linalg.norm(error_sino)
                correction_norm = np.linalg.norm(correction)
                step_norm = np.linalg.norm(lambda_n * correction)
                
                print(f"Iter {iter_n + 1:3d} | λ={lambda_n:.6f} | "
                      f"x:[{x.min():.6f}, {x.max():.6f}] | "
                      f"err_norm={error_norm:.6e} | "
                      f"corr_norm={correction_norm:.6e} | "
                      f"step_norm={step_norm:.6e}")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"SIRT Reconstruction Completed")
            print(f"Final image:")
            print(f"  min={x.min():.6f}, max={x.max():.6f}, "
                  f"mean={x.mean():.6f}, std={x.std():.6f}")
            print(f"{'='*70}\n")
        
        return x

    @staticmethod
    def reconstruct_ils_from_sinograms(full_sinogram, sparse_sinogram, 
                                      full_angles, sparse_angles, 
                                      iterations=10, damping_factor=0.03, 
                                      original=None):
        """
        Reconstruct from both full and sparse sinograms using stable SIRT.
        
        Args:
            full_sinogram (np.ndarray): Full/dense sinogram (180° or 360°)
            sparse_sinogram (np.ndarray): Sparse sinogram (variable angle)
            full_angles (np.ndarray): Angles for full sinogram
            sparse_angles (np.ndarray): Angles for sparse sinogram
            iterations (int): Number of SIRT iterations (1-100, default 10)
            damping_factor (float): Step size (0.01-0.1, default 0.03)
            original (np.ndarray, optional): Reference image for metrics
            
        Returns:
            dict: Reconstruction results with NMSE/PSNR metrics
        """
        # SIRT reconstruction on full sinogram
        full_recon = IterativeReconstruction.sirt_reconstruction(
            full_sinogram, full_angles, 
            iterations=iterations, 
            damping_factor=damping_factor,
            verbose=False
        )
        
        # SIRT reconstruction on sparse sinogram
        sparse_recon = IterativeReconstruction.sirt_reconstruction(
            sparse_sinogram, sparse_angles, 
            iterations=iterations, 
            damping_factor=damping_factor,
            verbose=False
        )
        
        result = {
            'full_recon': full_recon,
            'sparse_recon': sparse_recon,
            'full_angles': full_angles,
            'sparse_angles': sparse_angles,
        }
        
        # Compute metrics if reference provided
        if original is not None:
            from models.reconstruction import ComparisonReconstruction
            
            full_err = ComparisonReconstruction.compute_reconstruction_error(original, full_recon)
            sparse_err = ComparisonReconstruction.compute_reconstruction_error(original, sparse_recon)
            
            result['full_nmse'] = full_err['nmse']
            result['sparse_nmse'] = sparse_err['nmse']
            result['full_psnr'] = full_err['psnr']
            result['sparse_psnr'] = sparse_err['psnr']
        
        return result