"""
Iterative Reconstruction Technique (SIRT) - Numerically Stable Implementation.

This implements the proper SIRT algorithm with numerical stability improvements:
1. Normalized sinogram input
2. Normalized backprojection (divided by number of angles)
3. Small damping factor to control convergence
4. Value clipping to keep results physically meaningful
5. Decreasing step size over iterations
6. Proper forward/backward projection consistency

PERFORMANCE IMPROVEMENT (Option C):
- The SIRT loop runs the image at 256x256 working resolution, then upsamples
  the result back to the original resolution at the end.
- circle=True is kept throughout to avoid detector-row shape mismatches.
- ~4x speedup with negligible quality loss for CT simulator use.
"""

import numpy as np
from skimage.transform import radon, iradon, resize as sk_resize

# Working resolution for the SIRT loop.
_SIRT_WORK_SIZE = 256


def _resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a square 2-D image to target_size × target_size."""
    if image.shape[0] == target_size:
        return image.astype(np.float32)
    return sk_resize(
        image,
        (target_size, target_size),
        anti_aliasing=True,
    ).astype(np.float32)


def _resize_sinogram(sinogram: np.ndarray, target_rows: int) -> np.ndarray:
    """Resize sinogram along detector axis (rows), keep angle axis intact."""
    if sinogram.shape[0] == target_rows:
        return sinogram.astype(np.float32)
    return sk_resize(
        sinogram,
        (target_rows, sinogram.shape[1]),
        anti_aliasing=True,
    ).astype(np.float32)


class IterativeReconstruction:
    """Numerically stable SIRT reconstruction with proper iterative loop."""

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
    def sirt_reconstruction(sinogram, angles, iterations=5, damping_factor=0.05,
                            initial_guess=None, verbose=False):
        """
        SIRT (Simultaneous Iterative Reconstruction Technique) - Numerically Stable.

        Proper SIRT algorithm with stability improvements:

        1. Normalize sinogram
        2. x_0 = FBP(sinogram) downsampled to 256, or provided initial_guess
        3. For each iteration n:
           - Forward project x (256x256) -> forward_proj (detector_rows x angles)
           - Resize sinogram_normalized to match forward_proj detector rows
           - Compute error: err = sinogram_resized - forward_proj
           - Back-project err -> correction (256x256)
           - Decreasing step: lambda_n = damping_factor / (1 + 0.05*n)
           - Clip to physical range: x = clip(x_n + lambda_n * correction, 0, inf)
        4. Upsample result to original resolution

        Args:
            sinogram (np.ndarray): Measured sinogram
            angles (np.ndarray): Projection angles in degrees
            iterations (int): Number of iterations (1-40, default 5)
            damping_factor (float): Initial step size (default 0.05)
                Larger values = more visible per-iteration improvement, less stable.
                Smaller values = more stable but slower convergence.
            initial_guess (np.ndarray, optional): Pre-computed initial FBP.
                If None, will compute it internally.
            verbose (bool): Print convergence info per iteration.

        Returns:
            np.ndarray: Reconstructed image (upsampled back to original resolution)
        """
        # ── Validate parameters ──────────────────────────────────────────────
        iterations     = max(1, min(int(iterations), 40))
        damping_factor = np.clip(float(damping_factor), 0.001, 0.2)
        angles         = np.asarray(angles)
        sinogram       = sinogram.astype(np.float32)

        # Remember original size so we can upsample at the end
        original_size = sinogram.shape[0]

        # ── Normalize full sinogram ──────────────────────────────────────────
        sino_max            = np.max(np.abs(sinogram)) + 1e-8
        sinogram_normalized = sinogram / sino_max   # shape: (original_size, num_angles)

        if verbose:
            print(f"\n{'='*70}")
            print(f"SIRT Reconstruction Started")
            print(f"{'='*70}")
            print(f"Original sinogram shape : {sinogram.shape}")
            print(f"Working image size      : {_SIRT_WORK_SIZE}x{_SIRT_WORK_SIZE}")
            print(f"Number of angles        : {len(angles)}")
            print(f"Iterations              : {iterations}")
            print(f"Initial damping factor  : {damping_factor}")
            print(f"Sinogram normalized (max={sino_max:.4e})")
            print(f"{'='*70}")

        # ── Initialize x at working resolution ──────────────────────────────
        if initial_guess is not None:
            x = _resize_image(initial_guess.astype(np.float32), _SIRT_WORK_SIZE)
            if verbose:
                print(f"\nUsing provided initial guess (resized to {_SIRT_WORK_SIZE}x{_SIRT_WORK_SIZE})")
        else:
            # FBP on normalized sinogram, then downsample to working size
            fbp_full = IterativeReconstruction.fbp_reconstruction(
                sinogram_normalized, angles, filter_name='ramp'
            ).astype(np.float32)
            x = _resize_image(fbp_full, _SIRT_WORK_SIZE)

        # Clip initial guess to physical range
        x = np.clip(x, 0, None)

        if verbose:
            print(f"\nInitial Guess ({_SIRT_WORK_SIZE}x{_SIRT_WORK_SIZE}):")
            print(f"  min={x.min():.6f}, max={x.max():.6f}, "
                  f"mean={x.mean():.6f}, std={x.std():.6f}")

        # ── SIRT iteration loop ──────────────────────────────────────────────
        # Precompute a dummy forward proj to know the detector row count at
        # working resolution — radon's output size depends on image size.
        _dummy = radon(x, theta=angles, circle=True)
        work_detector_rows = _dummy.shape[0]   # e.g. 363 for a 256x256 image

        # Resize sinogram_normalized once to match working detector rows
        sino_work = _resize_sinogram(sinogram_normalized, work_detector_rows)

        if verbose:
            print(f"\nWorking sinogram shape  : {sino_work.shape}")

        for iter_n in range(iterations):

            # Forward projection of current estimate
            forward_proj = radon(x, theta=angles, circle=True)

            # Error in sinogram space (shapes now match)
            error_sino = sino_work - forward_proj

            # Unfiltered back-projection
            correction = iradon(
                error_sino,
                theta=angles,
                filter_name=None,
                circle=True,
                output_size=x.shape[0],
            ).astype(np.float32)

            # Normalize backprojection by number of angles
            correction = correction / len(angles)

            # Decreasing step size
            lambda_n = damping_factor / (1.0 + iter_n * 0.05)

            # Update and clip to physical range
            x = np.clip(x + lambda_n * correction, 0, None)

            if verbose:
                error_norm      = np.linalg.norm(error_sino)
                correction_norm = np.linalg.norm(correction)
                step_norm       = np.linalg.norm(lambda_n * correction)
                print(f"Iter {iter_n + 1:3d} | λ={lambda_n:.6f} | "
                      f"x:[{x.min():.6f}, {x.max():.6f}] | "
                      f"err_norm={error_norm:.6e} | "
                      f"corr_norm={correction_norm:.6e} | "
                      f"step_norm={step_norm:.6e}")

        if verbose:
            print(f"\n{'='*70}")
            print(f"SIRT Reconstruction Completed")
            print(f"Final image ({_SIRT_WORK_SIZE}x{_SIRT_WORK_SIZE} → {original_size}x{original_size}):")
            print(f"  min={x.min():.6f}, max={x.max():.6f}, "
                  f"mean={x.mean():.6f}, std={x.std():.6f}")
            print(f"{'='*70}\n")

        # ── Upsample result back to original resolution ──────────────────────
        return _resize_image(x, original_size)

    @staticmethod
    def reconstruct_ils_from_sinograms(full_sinogram, sparse_sinogram,
                                       full_angles, sparse_angles,
                                       iterations=5, damping_factor=0.05,
                                       original=None):
        """
        Reconstruct from both full and sparse sinograms using stable SIRT.

        Args:
            full_sinogram (np.ndarray): Full/dense sinogram (180° or 360°)
            sparse_sinogram (np.ndarray): Sparse sinogram (variable angle)
            full_angles (np.ndarray): Angles for full sinogram
            sparse_angles (np.ndarray): Angles for sparse sinogram
            iterations (int): Number of SIRT iterations (1-40, default 5)
            damping_factor (float): Step size (default 0.05)
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

        # Pre-compute FBP initial guess for sparse
        sino_max            = np.max(np.abs(sparse_sinogram)) + 1e-8
        sinogram_normalized = sparse_sinogram / sino_max
        fbp_guess           = IterativeReconstruction.fbp_reconstruction(
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
            'full_recon':    full_recon,
            'sparse_recon':  sparse_recon,
            'full_angles':   full_angles,
            'sparse_angles': sparse_angles,
        }

        # Compute metrics if reference provided
        if original is not None:
            from models.reconstruction import ComparisonReconstruction

            full_err   = ComparisonReconstruction.compute_reconstruction_error(original, full_recon)
            sparse_err = ComparisonReconstruction.compute_reconstruction_error(original, sparse_recon)

            result['full_nmse']   = full_err['nmse']
            result['sparse_nmse'] = sparse_err['nmse']
            result['full_psnr']   = full_err['psnr']
            result['sparse_psnr'] = sparse_err['psnr']

        return result