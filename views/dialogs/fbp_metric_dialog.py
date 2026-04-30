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

from models.reconstruction import ComparisonReconstruction
from models.spectra_physics import generate_physics_sinogram
from views import style

class FBPMetricDialog(QDialog):
    def __init__(
        self,
        parent=None,
        original=None,
        total_i0=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compare FBP Metric")
        self.resize(1400, 950)

        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        self.original = original
        self.total_i0 = total_i0
        self.step_angle = 1  # Local state for this dialog only
        self.full_recon = None
        self.sparse_recon = None
        self.full_nmse = None
        self.sparse_nmse = None
        self.error_map = None

        # =================================================
        # MAIN LAYOUT
        # =================================================
        layout = QVBoxLayout(self)

        main_row = QHBoxLayout()
        layout.addLayout(main_row)

        # =================================================
        # COLUMN 1: FULL
        # =================================================
        col_full = QVBoxLayout()

        self.full_group = QGroupBox("Full FBP")
        self.full_group.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_layout = QVBoxLayout(self.full_group)
        self.fig_full, self.ax_full = plt.subplots(figsize=(6, 5), facecolor="#1E1E2E")
        self.canvas_full = FigureCanvas(self.fig_full)
        self.canvas_full.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        full_layout.addWidget(self.canvas_full)
        self.fig_full.subplots_adjust(left=0, right=1, top=1, bottom=0)
        col_full.addWidget(self.full_group, 3)

        self.full_sino_group = QGroupBox("Full Sinogram")
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
        # COLUMN 2: SPARSE
        # =================================================
        col_sparse = QVBoxLayout()

        self.sparse_group = QGroupBox("Sparse FBP")
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
        # COLUMN 3: NMSE
        # =================================================
        col_nmse = QVBoxLayout()

        self.nmse_group = QGroupBox("NMSE Comparison")
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
        # CONTROLS
        # =================================================
        controls_row = QHBoxLayout()
        layout.addLayout(controls_row)

        controls_row.addWidget(QLabel("Step Angle"))

        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(1, 10)
        self.step_slider.setValue(self.step_angle)
        self.step_slider.setSingleStep(1)
        self.step_slider.setPageStep(1)

        self.step_label = QLabel(f"{self.step_angle}°")

        # Only connect to sliderReleased for efficiency
        self.step_slider.sliderReleased.connect(self._on_step_slider_released)

        controls_row.addWidget(self.step_slider)
        controls_row.addWidget(self.step_label)

        # =================================================
        # FOOTER
        # =================================================
        footer = QHBoxLayout()
        footer.addStretch()
        layout.addLayout(footer)

        # Initial computation and render
        if self.original is not None and self.total_i0 is not None:
            self._recompute_and_render()
        else:
            self._render()

    def _on_step_slider_released(self):
        """Step angle slider released - affects THIS DIALOG ONLY"""
        new_step_angle = self.step_slider.value()
        
        # Only proceed if value actually changed
        if new_step_angle == self.step_angle:
            return
        
        self.step_angle = new_step_angle
        self.step_label.setText(f"{self.step_angle}°")
        self._recompute_and_render()

    def _recompute_and_render(self):
        """Recompute everything based on current step_angle"""
        if self.original is None or self.total_i0 is None:
            self._render()
            return

        # Generate sinograms for current step angle
        full_sino, sparse_sino, full_angles, sparse_angles = generate_physics_sinogram(
            self.original,
            self.total_i0,
            user_step_angle=self.step_angle,
        )

        # Store sinograms
        self.full_sino = full_sino
        self.sparse_sino = sparse_sino
        self.full_angles = full_angles
        self.sparse_angles = sparse_angles

        # Compute FBP reconstructions
        recon_results = ComparisonReconstruction.reconstruct_fbp_from_sinograms(
            full_sino,
            sparse_sino,
            full_angles,
            sparse_angles,
            original=self.original,
            filter_name="ramp",
        )

        self.full_recon = recon_results['full_recon']
        self.sparse_recon = recon_results['sparse_recon']
        
        # Compute metrics
        metrics = ComparisonReconstruction.compute_reconstruction_error(
            self.full_recon, self.sparse_recon
        )
        self.sparse_nmse = metrics['nmse']
        self.sparse_psnr = metrics['psnr']
        self.error_map = metrics['emap']
        
        # Render all visualizations
        self._render()

    def _render(self):
        """Render all visualizations"""
        if self.full_recon is not None:
            self.ax_full.clear()
            self.ax_full.set_facecolor("black")
            self.ax_full.imshow(self.full_recon, cmap="gray")
            self.ax_full.axis("off")
            self.fig_full.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_full.draw_idle()

        if self.sparse_recon is not None:
            self.ax_sparse.clear()
            self.ax_sparse.set_facecolor("black")
            self.ax_sparse.imshow(self.sparse_recon, cmap="gray")
            self.ax_sparse.axis("off")
            self.fig_sparse.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_sparse.draw_idle()

        # Render sinograms if available
        if hasattr(self, 'full_sino') and self.full_sino is not None:
            self.ax_full_sino.clear()
            self.ax_full_sino.set_facecolor('black')
            self.ax_full_sino.imshow(self.full_sino, cmap='gray', aspect='auto')
            self.ax_full_sino.set_title(f'Full Sinogram (360°)', color='white')
            self.ax_full_sino.axis('off')
            self.fig_full_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_full_sino.draw_idle()

        if hasattr(self, 'sparse_sino') and self.sparse_sino is not None:
            self.ax_sparse_sino.clear()
            self.ax_sparse_sino.set_facecolor('black')
            self.ax_sparse_sino.imshow(self.sparse_sino, cmap='gray', aspect='auto')
            self.ax_sparse_sino.set_title(f'Sparse Sinogram ({self.step_angle}°)', color='white')
            self.ax_sparse_sino.axis('off')
            self.fig_sparse_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
            self.canvas_sparse_sino.draw_idle()

        # Render error map
        if hasattr(self, 'error_map') and self.error_map is not None:
            self.ax_nmse.clear()
            self.ax_nmse.set_facecolor("black")

            vmin = 0.0
            vmax = float(np.max(self.error_map))
            if vmax <= 0.0:
                vmax = 1.0

            im = self.ax_nmse.imshow(self.error_map, cmap="hot", vmin=vmin, vmax=vmax)
            self.ax_nmse.set_title("NMSE Visualization\n(Sparse vs Full)", color="white", fontsize=10)
            self.ax_nmse.axis("off")
            self.fig_nmse.subplots_adjust(left=0, right=0.88, top=1, bottom=0)

            info_text = f"NMSE: {self.sparse_nmse:.4f}\nPSNR: {self.sparse_psnr:.2f} dB"
            self.ax_nmse.text(0.5, -0.15, info_text, color="#FF4800", fontsize=10,
                             ha="center", va="top", transform=self.ax_nmse.transAxes,
                             bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555'))

            # Colorbar
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

        self.canvas_full.draw_idle()
        self.canvas_sparse.draw_idle()
        self.canvas_nmse.draw_idle()