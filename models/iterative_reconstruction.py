"""
Iterative Reconstruction Technique (SIRT) - Numerically Stable Implementation.

This implements the proper SIRT algorithm with numerical stability improvements:
1. Normalized sinogram input
2. Normalized backprojection (divided by number of angles)
3. Small damping factor to control convergence
4. Value clipping to keep results physically meaningful
5. Decreasing step size over iterations
6. Proper forward/backward projection consistency

PERFORMANCE FIX:
- Sinogram and reconstruction are downsampled to SIRT_SIZE before iterating,
  then upsampled back to the original resolution.  This gives a 4-16x speedup
  with negligible quality loss because the iterative loop only needs enough
  resolution to converge.
"""

import numpy as np
from skimage.transform import radon, iradon, resize as sk_resize


# ---------------------------------------------------------------------------
# Tunable constant: reduce to 128 for even faster (but coarser) iterations.
# ---------------------------------------------------------------------------
SIRT_SIZE = 256


def _downsample_sinogram(sinogram: np.ndarray, target_rows: int) -> np.ndarray:
    """Resize sinogram detector axis (rows) to target_rows, keep angle axis."""
    if sinogram.shape[0] == target_rows:
        return sinogram
    return sk_resize(
        sinogram,
        (target_rows, sinogram.shape[1]),
        anti_aliasing=True,
    ).astype(np.float32)


def _upsample_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a square 2-D image back to target_size × target_size."""
    if image.shape[0] == target_size:
        return image
    return sk_resize(
        image,
        (target_size, target_size),
        anti_aliasing=True,
    ).astype(np.float32)


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
                            initial_guess=None, verbose=False):
        """
        SIRT (Simultaneous Iterative Reconstruction Technique) - Numerically Stable.

        Proper SIRT algorithm with stability improvements AND internal downsampling
        for speed (images are downsampled to SIRT_SIZE before iterating, then
        upsampled back to the original resolution before returning).

        Args:
            sinogram (np.ndarray): Measured sinogram
            angles (np.ndarray): Projection angles in degrees
            iterations (int): Number of iterations (1-100, default 10)
            damping_factor (float): Initial step size (0.01-0.1, default 0.03)
                Smaller values = more stable but slower
            initial_guess (np.ndarray, optional): Pre-computed initial FBP.
                If None, will compute it internally.
            verbose (bool): Print convergence info per iteration

        Returns:
            np.ndarray: Reconstructed image (same spatial size as input sinogram rows)
        """
        # ── Validate parameters ──────────────────────────────────────────────
        iterations = max(1, min(int(iterations), 100))
        damping_factor = np.clip(float(damping_factor), 0.001, 0.2)
        angles = np.asarray(angles)
        sinogram = sinogram.astype(np.float32)

        # Remember original detector / image size so we can upsample at the end
        original_size = sinogram.shape[0]

        # ── PERFORMANCE: downsample sinogram + initial guess ─────────────────
        work_sino = _downsample_sinogram(sinogram, SIRT_SIZE)
        work_size = work_sino.shape[0]   # == SIRT_SIZE (or original if already small)

        # ── Normalize sinogram ───────────────────────────────────────────────
        sino_max = np.max(np.abs(work_sino)) + 1e-8
        sinogram_normalized = work_sino / sino_max

        if verbose:
            print(f"\n{'='*70}")
            print(f"SIRT Reconstruction Started")
            print(f"{'='*70}")
            print(f"Original sinogram shape : {sinogram.shape}")
            print(f"Working sinogram shape  : {work_sino.shape}")
            print(f"Number of angles        : {len(angles)}")
            print(f"Iterations              : {iterations}")
            print(f"Initial damping factor  : {damping_factor}")
            print(f"Sinogram normalized (max={sino_max:.4e})")
            print(f"{'='*70}")

        # ── Initialize with FBP or use provided initial guess ────────────────
        if initial_guess is not None:
            # Downsample the provided guess to the working size
            x = _upsample_image(initial_guess.astype(np.float32), work_size)
        else:
            x = IterativeReconstruction.fbp_reconstruction(
                sinogram_normalized, angles, filter_name='ramp'
            )
            x = x.astype(np.float32)

        # Clip initial guess to physical range (non-negative)
        x = np.clip(x, 0, None)

        if verbose:
            print(f"\nInitial Guess (working size {work_size}x{work_size}):")
            print(f"  min={x.min():.6f}, max={x.max():.6f}, "
                  f"mean={x.mean():.6f}, std={x.std():.6f}")

        # ── SIRT ITERATION LOOP ──────────────────────────────────────────────
        for iter_n in range(iterations):

            # Forward projection
            forward_proj = radon(x, theta=angles, circle=True)

            # Error in sinogram space
            error_sino = sinogram_normalized - forward_proj

            # Back-projection (unfiltered)
            correction = iradon(
                error_sino,
                theta=angles,
                filter_name=None,
                circle=True,
                output_size=x.shape[0],
            )
            correction = correction.astype(np.float32)

            # Normalize backprojection by number of angles
            correction = correction / len(angles)

            # Decreasing step size
            lambda_n = damping_factor / (1.0 + iter_n * 0.1)

            # Update
            x = x + lambda_n * correction

            # Clip to physical range
            x = np.clip(x, 0, None)

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
            print(f"Final image (working size):")
            print(f"  min={x.min():.6f}, max={x.max():.6f}, "
                  f"mean={x.mean():.6f}, std={x.std():.6f}")
            print(f"{'='*70}\n")

        # ── PERFORMANCE: upsample result back to original resolution ─────────
        x = _upsample_image(x, original_size)

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
        # Full SIRT reconstruction
        full_recon = IterativeReconstruction.sirt_reconstruction(
            full_sinogram, full_angles,
            iterations=iterations,
            damping_factor=damping_factor,
            verbose=False,
        )

        # Pre-compute FBP initial guess for sparse (avoids recomputing inside sirt_reconstruction)
        sino_max = np.max(np.abs(sparse_sinogram)) + 1e-8
        sinogram_normalized = sparse_sinogram / sino_max
        fbp_guess = IterativeReconstruction.fbp_reconstruction(
            sinogram_normalized, sparse_angles, filter_name='ramp'
        )
        fbp_guess = np.clip(fbp_guess.astype(np.float32), 0, None)

        # Sparse SIRT reconstruction using pre-computed FBP guess
        sparse_recon = IterativeReconstruction.sirt_reconstruction(
            sparse_sinogram, sparse_angles,
            iterations=iterations,
            damping_factor=damping_factor,
            initial_guess=fbp_guess,
            verbose=False,
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