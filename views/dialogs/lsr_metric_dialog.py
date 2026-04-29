"""
LSR (Least Squares Reconstruction) Metric Dialog

Displays LSR reconstructions comparing full (dense 360°) vs sparse (variable angle)
with error metrics (NMSE, PSNR) and iterations control.
"""

import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QSlider,
    QSizePolicy,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models.iterative_reconstruction import IterativeReconstruction
from models.reconstruction import ComparisonReconstruction
from models.spectra_physics import generate_physics_sinogram
from views import style


class LSRMetricDialog(QDialog):
    """Dialog for comparing LSR reconstructions (full 360° vs sparse with step angle)"""
    
    def __init__(
        self,
        parent=None,
        original=None,
        total_i0=None,
        step_angle=1,
        full_sino=None,
        sparse_sino=None,
        full_angles=None,
        sparse_angles=None,
        iterations=10,
    ):
        """
        Initialize LSR Metric Dialog.
        
        Args:
            parent: Parent window
            original: Reference mu_map image
            total_i0: Total photon intensity
            step_angle: Sparse projection angle step (1-10°)
            full_sino: Full sinogram (360°, 1° steps)
            sparse_sino: Sparse sinogram (variable angle)
            full_angles: Full projection angles
            sparse_angles: Sparse projection angles
            iterations: Initial number of SIRT iterations (1-100, default 10)
        """
        super().__init__(parent)
        self.setWindowTitle("Compare LSR Metric (Iterative Least Squares)")
        self.resize(1400, 950)

        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        # Store data
        self.original = original
        self.total_i0 = total_i0
        self.step_angle = int(step_angle)
        self.full_sino = full_sino
        self.sparse_sino = sparse_sino
        self.full_angles = full_angles
        self.sparse_angles = sparse_angles
        self.iterations = int(iterations)
        self.damping_factor = 0.03  # Standard SIRT damping factor
        
        # Reconstructions and metrics
        self.full_recon = None
        self.sparse_recon = None
        self.sparse_nmse = 0.0
        self.sparse_psnr = 0.0
        self.error_map = None

        # =================================================
        # MAIN LAYOUT
        # =================================================
        layout = QVBoxLayout(self)
        main_row = QHBoxLayout()
        layout.addLayout(main_row)

        # =================================================
        # COLUMN 1: FULL LSR (360° angles)
        # =================================================
        col_full = QVBoxLayout()

        self.full_group = QGroupBox("Full LSR Reconstruction (360°)")
        self.full_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_layout = QVBoxLayout(self.full_group)
        self.fig_full, self.ax_full = plt.subplots(figsize=(6, 5), facecolor="#1E1E2E")
        self.canvas_full = FigureCanvas(self.fig_full)
        self.canvas_full.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_layout.addWidget(self.canvas_full)
        self.fig_full.subplots_adjust(left=0, right=1, top=1, bottom=0)
        col_full.addWidget(self.full_group, 3)

        self.full_sino_group = QGroupBox("Full Sinogram (360°, 1° steps)")
        self.full_sino_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_sino_layout = QVBoxLayout(self.full_sino_group)
        self.fig_full_sino, self.ax_full_sino = plt.subplots(facecolor="#1E1E2E")
        self.canvas_full_sino = FigureCanvas(self.fig_full_sino)
        self.canvas_full_sino.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_sino_layout.addWidget(self.canvas_full_sino)
        self.fig_full_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
        col_full.addWidget(self.full_sino_group, 2)

        main_row.addLayout(col_full, 1)

        # =================================================
        # COLUMN 2: SPARSE LSR (with step angle & iterations)
        # =================================================
        col_sparse = QVBoxLayout()

        self.sparse_group = QGroupBox("Sparse LSR Reconstruction (Step Angle)")
        self.sparse_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sparse_layout = QVBoxLayout(self.sparse_group)
        self.fig_sparse, self.ax_sparse = plt.subplots(figsize=(6, 5), facecolor="#1E1E2E")
        self.canvas_sparse = FigureCanvas(self.fig_sparse)
        self.canvas_sparse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sparse_layout.addWidget(self.canvas_sparse)
        self.fig_sparse.subplots_adjust(left=0, right=1, top=1, bottom=0)
        col_sparse.addWidget(self.sparse_group, 3)

        self.sparse_sino_group = QGroupBox("Sparse Sinogram")
        self.sparse_sino_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sparse_sino_layout = QVBoxLayout(self.sparse_sino_group)
        self.fig_sparse_sino, self.ax_sparse_sino = plt.subplots(facecolor="#1E1E2E")
        self.canvas_sparse_sino = FigureCanvas(self.fig_sparse_sino)
        self.canvas_sparse_sino.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sparse_sino_layout.addWidget(self.canvas_sparse_sino)
        self.fig_sparse_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
        col_sparse.addWidget(self.sparse_sino_group, 2)

        main_row.addLayout(col_sparse, 1)

        # =================================================
        # COLUMN 3: ERROR MAP & METRICS
        # =================================================
        col_nmse = QVBoxLayout()

        self.nmse_group = QGroupBox("LSR Error Analysis\n(Sparse vs Full)")
        self.nmse_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nmse_layout = QVBoxLayout(self.nmse_group)

        self.fig_nmse, self.ax_nmse = plt.subplots(figsize=(5, 5), facecolor="#1E1E2E")
        self.canvas_nmse = FigureCanvas(self.fig_nmse)
        self.canvas_nmse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        nmse_layout.addWidget(self.canvas_nmse)
        self.fig_nmse.subplots_adjust(left=0, right=0.88, top=1, bottom=0)

        col_nmse.addWidget(self.nmse_group, 1)

        main_row.addLayout(col_nmse, 1)

        # =================================================
        # CONTROLS SECTION
        # =================================================
        
        # Row 1: Step Angle Slider
        controls_row1 = QHBoxLayout()
        layout.addLayout(controls_row1)

        controls_row1.addWidget(QLabel("Step Angle:"))

        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(1, 10)
        self.step_slider.setValue(self.step_angle)
        self.step_slider.setSingleStep(1)
        self.step_slider.setPageStep(1)

        self.step_label = QLabel(f"{self.step_angle}°")

        # Only connect to sliderReleased for efficiency
        self.step_slider.sliderReleased.connect(self._on_step_slider_released)

        controls_row1.addWidget(self.step_slider)
        controls_row1.addWidget(self.step_label)

        # Row 2: Iterations Slider (affects SPARSE reconstruction only)
        iterations_row = QHBoxLayout()
        layout.addLayout(iterations_row)

        iterations_row.addWidget(QLabel("Iterations (affects Sparse):"))

        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(1, 100)
        self.iter_slider.setValue(self.iterations)
        self.iter_slider.setSingleStep(1)
        self.iter_slider.setPageStep(5)

        self.iter_label = QLabel(str(self.iterations))

        # Connect valueChanged ONLY for label update (no heavy computation)
        self.iter_slider.valueChanged.connect(
            lambda v: self.iter_label.setText(f"{v}")
        )
        # Only connect to sliderReleased for actual recomputation
        self.iter_slider.sliderReleased.connect(self._on_iter_slider_released)

        iterations_row.addWidget(self.iter_slider)
        iterations_row.addWidget(self.iter_label)
        iterations_row.addStretch()

        # Initial computation and render
        if self.original is not None and self.total_i0 is not None:
            self._recompute_and_render()
        else:
            self._render_all()

    # =====================================================
    # SLIDER HANDLERS (Called only when slider is released)
    # =====================================================

    def _on_step_slider_released(self):
        """Step angle slider released"""
        new_step_angle = self.step_slider.value()
        
        # Only proceed if value actually changed
        if new_step_angle == self.step_angle:
            return
        
        self.step_angle = new_step_angle
        self.step_label.setText(f"{self.step_angle}°")
        self._recompute_and_render()
        
        # Sync back to parent
        if self.parent() and hasattr(self.parent(), 'sync_step_angle_from_dialog'):
            self.parent().sync_step_angle_from_dialog(self.step_angle)

    def _on_iter_slider_released(self):
        """Iterations slider released"""
        new_iterations = self.iter_slider.value()
        
        # Only proceed if value actually changed
        if new_iterations == self.iterations:
            return
        
        self.iterations = new_iterations
        self.iter_label.setText(str(self.iterations))
        
        # Only recompute SPARSE reconstruction with new iterations
        self._recompute_sparse_reconstruction()
        
        # Update metrics (sparse vs full)
        self._recompute_metrics()
        
        # Render everything
        self._render_all()
        
        # Sync back to parent
        if self.parent() and hasattr(self.parent(), 'sync_iterations_from_dialog'):
            self.parent().sync_iterations_from_dialog(self.iterations)

    # =====================================================
    # SYNC METHODS FROM MAIN WINDOW
    # =====================================================

    def sync_step_angle_from_main(self, step_angle):
        """Sync step_angle from main window to this dialog"""
        new_step_angle = int(step_angle)
        
        # Only proceed if value actually changed
        if new_step_angle == self.step_angle:
            return
        
        self.step_angle = new_step_angle
        self.step_slider.blockSignals(True)
        self.step_slider.setValue(self.step_angle)
        self.step_slider.blockSignals(False)
        self.step_label.setText(f"{self.step_angle}°")
        self._recompute_and_render()

    def sync_iterations_from_main(self, iterations):
        """Sync iterations from main window to this dialog"""
        new_iterations = int(iterations)
        
        # Only proceed if value actually changed
        if new_iterations == self.iterations:
            return
        
        self.iterations = new_iterations
        self.iter_slider.blockSignals(True)
        self.iter_slider.setValue(self.iterations)
        self.iter_slider.blockSignals(False)
        self.iter_label.setText(str(self.iterations))
        self._recompute_sparse_reconstruction()
        self._recompute_metrics()
        self._render_all()

    # =====================================================
    # RECOMPUTATION METHODS
    # =====================================================

    def _recompute_and_render(self):
        """
        Full recomputation: regenerate sinograms + reconstruct both + render.
        
        Called when step_angle changes.
        """
        if self.original is None or self.total_i0 is None:
            self._render_all()
            return

        # Regenerate sinograms for new step angle
        self._recompute_sinograms()

        # Recompute both reconstructions
        self._recompute_reconstructions()

        # Render all visualizations
        self._render_all()

    def _recompute_sinograms(self):
        """
        Regenerate sinograms for current step_angle.
        
        Calls generate_physics_sinogram() to create full and sparse sinograms.
        """
        if self.original is None or self.total_i0 is None:
            return

        full_sino, sparse_sino, full_angles, sparse_angles = generate_physics_sinogram(
            self.original,
            self.total_i0,
            user_step_angle=self.step_angle,
        )

        self.full_sino = full_sino
        self.sparse_sino = sparse_sino
        self.full_angles = full_angles
        self.sparse_angles = sparse_angles

    def _recompute_reconstructions(self):
        """
        Recompute both FULL and SPARSE SIRT reconstructions with current iterations.
        """
        if self.full_sino is None or self.sparse_sino is None:
            return

        # Perform SIRT reconstruction on both
        recon_results = IterativeReconstruction.reconstruct_ils_from_sinograms(
            self.full_sino,
            self.sparse_sino,
            self.full_angles,
            self.sparse_angles,
            iterations=self.iterations,
            damping_factor=self.damping_factor,
            original=self.original,
        )

        self.full_recon = recon_results['full_recon']
        self.sparse_recon = recon_results['sparse_recon']

        # Extract metrics
        self.sparse_nmse = recon_results.get('sparse_nmse', 0.0)
        self.sparse_psnr = recon_results.get('sparse_psnr', 0.0)

        # Compute error map (sparse vs full)
        if self.full_recon is not None and self.sparse_recon is not None:
            metrics = ComparisonReconstruction.compute_reconstruction_error(
                self.full_recon, self.sparse_recon
            )
            self.error_map = metrics['emap']

    def _recompute_sparse_reconstruction(self):
        """
        Recompute SPARSE reconstruction ONLY (using proper SIRT).
        
        Called when iterations slider changes.
        """
        if self.sparse_sino is None:
            return

        # Recompute sparse SIRT with proper iterative loop
        self.sparse_recon = IterativeReconstruction.sirt_reconstruction(
            self.sparse_sino,
            self.sparse_angles,
            iterations=self.iterations,
            damping_factor=self.damping_factor,
            verbose=False
        )

    def _recompute_metrics(self):
        """
        Recompute error map and metrics after sparse reconstruction update.
        """
        if self.full_recon is None or self.sparse_recon is None:
            return

        metrics = ComparisonReconstruction.compute_reconstruction_error(
            self.full_recon, self.sparse_recon
        )
        self.error_map = metrics['emap']
        self.sparse_nmse = metrics['nmse']
        self.sparse_psnr = metrics['psnr']

    # =====================================================
    # RENDERING METHODS
    # =====================================================

    def _render_all(self):
        """Master render method - updates all visualizations."""
        self._render_reconstructions()
        self._render_sinograms()
        self._render_error_map()

    def _render_reconstructions(self):
        """Render full and sparse LSR reconstruction images."""
        if self.full_recon is not None:
            self.ax_full.clear()
            self.ax_full.set_facecolor("black")
            self.ax_full.imshow(self.full_recon, cmap="gray")
            self.ax_full.set_title(f"Full LSR @ {self.iterations} iter (360°)", 
                                  color="white", fontsize=10)
            self.ax_full.axis("off")
            self.fig_full.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_full.draw_idle()

        if self.sparse_recon is not None:
            self.ax_sparse.clear()
            self.ax_sparse.set_facecolor("black")
            self.ax_sparse.imshow(self.sparse_recon, cmap="gray")
            self.ax_sparse.set_title(f"Sparse LSR @ {self.step_angle}° @ {self.iterations} iter", 
                                    color="white", fontsize=10)
            self.ax_sparse.axis("off")
            self.fig_sparse.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_sparse.draw_idle()

    def _render_sinograms(self):
        """Render full and sparse sinograms."""
        if self.full_sino is not None:
            self.ax_full_sino.clear()
            self.ax_full_sino.set_facecolor("black")
            self.ax_full_sino.imshow(self.full_sino, cmap="gray", aspect="auto")
            self.ax_full_sino.set_title(f"Full Sinogram (360°, 1° steps)", 
                                       color="white", fontsize=9)
            self.ax_full_sino.axis("off")
            self.fig_full_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_full_sino.draw_idle()

        if self.sparse_sino is not None:
            self.ax_sparse_sino.clear()
            self.ax_sparse_sino.set_facecolor("black")
            self.ax_sparse_sino.imshow(self.sparse_sino, cmap="gray", aspect="auto")
            self.ax_sparse_sino.set_title(f"Sparse Sinogram ({self.step_angle}° steps)", 
                                         color="white", fontsize=9)
            self.ax_sparse_sino.axis("off")
            self.fig_sparse_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_sparse_sino.draw_idle()

    def _render_error_map(self):
        """Render error map comparing sparse vs full LSR."""
        self.ax_nmse.clear()
        self.ax_nmse.set_facecolor("black")

        if self.error_map is not None:
            # Display error map with hot colormap
            vmin = 0.0
            vmax = float(np.max(self.error_map))
            if vmax <= 0.0:
                vmax = 1.0

            im = self.ax_nmse.imshow(self.error_map, cmap="hot", vmin=vmin, vmax=vmax)
            self.ax_nmse.set_title("Error Map\n(Sparse vs Full LSR)", 
                                  color="white", fontsize=10)
            self.ax_nmse.axis("off")

            # Display metrics text
            info_text = f"NMSE: {self.sparse_nmse:.4f}\nPSNR: {self.sparse_psnr:.2f} dB"
            self.ax_nmse.text(0.5, -0.15, info_text, color="#FF4800", fontsize=10,
                             ha="center", va="top", transform=self.ax_nmse.transAxes,
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555'))

            # Add colorbar
            if hasattr(self, 'cbar_nmse'):
                try:
                    self.cbar_nmse.remove()
                except Exception:
                    pass
                try:
                    if hasattr(self, 'cbar_nmse_ax'):
                        self.cbar_nmse_ax.remove()
                        delattr(self, 'cbar_nmse_ax')
                except Exception:
                    pass

            divider = make_axes_locatable(self.ax_nmse)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cbar_nmse_ax = cax
            self.cbar_nmse = self.fig_nmse.colorbar(im, cax=cax)
            self.cbar_nmse.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(self.cbar_nmse.ax.axes, 'yticklabels'), color='white')
        else:
            self.ax_nmse.text(0.5, 0.5, "Error map unavailable", color="white",
                            ha="center", va="center", transform=self.ax_nmse.transAxes)
            self.ax_nmse.axis("off")

        self.fig_nmse.subplots_adjust(left=0, right=0.88, top=1, bottom=0)
        self.canvas_nmse.draw_idle()