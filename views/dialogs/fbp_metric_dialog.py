import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog, QHBoxLayout, QVBoxLayout, QLabel, QSlider, QSizePolicy, QGroupBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from models.reconstruction import ComparisonReconstruction
from models.spectra_physics import generate_physics_sinogram, generate_spectrum_physics
from models.phantom_material_map import build_three_material_mu_map
from views import style

class FBPMetricDialog(QDialog):
    def __init__(self, phantom_size=512, kVp=100, mA=2, Cu=0.0, Al=0.0, step_angle=1, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Compare FBP Metric (Frozen Snapshot @ {kVp}kVp, {mA}mA)")
        self.resize(1400, 950)

        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        # 1. ABSOLUTE ISOLATION: Build private memory from scratch
        _, original_map = build_three_material_mu_map(size=phantom_size, kvp=kVp)
        _, _, total_i0 = generate_spectrum_physics(kVp, mA, Cu, Al)
        
        self.original = np.array(original_map, copy=True, dtype=np.float32)
        self.total_i0 = float(total_i0)
        self.kVp, self.mA, self.Cu, self.Al = kVp, mA, Cu, Al
        self.step_angle = step_angle  
        
        self.full_recon, self.sparse_recon = None, None
        self.sparse_nmse, self.sparse_psnr = 0.0, 0.0
        self.error_map = None

        layout = QVBoxLayout(self)
        main_row = QHBoxLayout(); layout.addLayout(main_row)

        # COLUMN 1: FULL
        col_full = QVBoxLayout()
        self.full_group = QGroupBox("Full FBP")
        full_layout = QVBoxLayout(self.full_group)
        self.fig_full = Figure(figsize=(6, 5), facecolor="#1E1E2E")
        self.ax_full = self.fig_full.add_subplot(111)
        self.canvas_full = FigureCanvas(self.fig_full)
        full_layout.addWidget(self.canvas_full)
        col_full.addWidget(self.full_group, 3)

        self.full_sino_group = QGroupBox("Full Sinogram")
        full_sino_layout = QVBoxLayout(self.full_sino_group)
        self.fig_full_sino = Figure(facecolor="#1E1E2E")
        self.ax_full_sino = self.fig_full_sino.add_subplot(111)
        self.canvas_full_sino = FigureCanvas(self.fig_full_sino)
        full_sino_layout.addWidget(self.canvas_full_sino)
        col_full.addWidget(self.full_sino_group, 2)
        main_row.addLayout(col_full, 1)

        # COLUMN 2: SPARSE
        col_sparse = QVBoxLayout()
        self.sparse_group = QGroupBox("Sparse FBP")
        sparse_layout = QVBoxLayout(self.sparse_group)
        self.fig_sparse = Figure(figsize=(6, 5), facecolor="#1E1E2E")
        self.ax_sparse = self.fig_sparse.add_subplot(111)
        self.canvas_sparse = FigureCanvas(self.fig_sparse)
        sparse_layout.addWidget(self.canvas_sparse)
        col_sparse.addWidget(self.sparse_group, 3)

        self.sparse_sino_group = QGroupBox("Sparse Sinogram")
        sparse_sino_layout = QVBoxLayout(self.sparse_sino_group)
        self.fig_sparse_sino = Figure(facecolor="#1E1E2E")
        self.ax_sparse_sino = self.fig_sparse_sino.add_subplot(111)
        self.canvas_sparse_sino = FigureCanvas(self.fig_sparse_sino)
        sparse_sino_layout.addWidget(self.canvas_sparse_sino)
        col_sparse.addWidget(self.sparse_sino_group, 2)
        main_row.addLayout(col_sparse, 1)

        # COLUMN 3: NMSE
        col_nmse = QVBoxLayout()
        self.nmse_group = QGroupBox("NMSE Comparison")
        nmse_layout = QVBoxLayout(self.nmse_group)
        self.fig_nmse = Figure(figsize=(5, 5), facecolor="#1E1E2E")
        self.ax_nmse = self.fig_nmse.add_subplot(111)
        self.canvas_nmse = FigureCanvas(self.fig_nmse)
        nmse_layout.addWidget(self.canvas_nmse)
        col_nmse.addWidget(self.nmse_group, 1)
        main_row.addLayout(col_nmse, 1)

        # CONTROLS
        controls_row = QHBoxLayout(); layout.addLayout(controls_row)
        controls_row.addWidget(QLabel("Step Angle"))
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(1, 10); self.step_slider.setValue(self.step_angle); self.step_slider.setSingleStep(1)
        self.step_label = QLabel(f"{self.step_angle}°")
        self.step_slider.sliderReleased.connect(self._on_step_slider_released)
        controls_row.addWidget(self.step_slider); controls_row.addWidget(self.step_label)

        self._recompute_and_render()

    def _on_step_slider_released(self):
        if self.step_slider.value() == self.step_angle: return
        self.step_angle = self.step_slider.value()
        self.step_label.setText(f"{self.step_angle}°")
        self._recompute_and_render()

    def _recompute_and_render(self):
        # 2. STATE ENFORCEMENT: Force backend modules to reset global variables to THIS dialog's state
        generate_spectrum_physics(self.kVp, self.mA, self.Cu, self.Al)

        full_sino, sparse_sino, full_angles, sparse_angles = generate_physics_sinogram(
            self.original, self.total_i0, user_step_angle=self.step_angle,
        )

        self.full_sino = np.array(full_sino, copy=True)
        self.sparse_sino = np.array(sparse_sino, copy=True)
        self.full_angles = np.array(full_angles, copy=True)
        self.sparse_angles = np.array(sparse_angles, copy=True)

        recon_results = ComparisonReconstruction.reconstruct_fbp_from_sinograms(
            self.full_sino, self.sparse_sino, self.full_angles, self.sparse_angles,
            original=self.original, filter_name="ramp",
        )

        self.full_recon = np.array(recon_results['full_recon'], copy=True)
        self.sparse_recon = np.array(recon_results['sparse_recon'], copy=True)
        
        metrics = ComparisonReconstruction.compute_reconstruction_error(self.full_recon, self.sparse_recon)
        self.sparse_nmse = float(metrics['nmse'])
        self.sparse_psnr = float(metrics['psnr'])
        self.error_map = np.array(metrics['emap'], copy=True)
        
        self._render()

    def _render(self):
        if self.full_recon is not None:
            self.ax_full.clear(); self.ax_full.set_facecolor("black")
            self.ax_full.imshow(self.full_recon, cmap="gray"); self.ax_full.axis("off")
            self.fig_full.subplots_adjust(left=0, right=1, top=1, bottom=0); self.canvas_full.draw_idle()

        if self.sparse_recon is not None:
            self.ax_sparse.clear(); self.ax_sparse.set_facecolor("black")
            self.ax_sparse.imshow(self.sparse_recon, cmap="gray"); self.ax_sparse.axis("off")
            self.fig_sparse.subplots_adjust(left=0, right=1, top=1, bottom=0); self.canvas_sparse.draw_idle()

        if hasattr(self, 'full_sino') and self.full_sino is not None:
            self.ax_full_sino.clear(); self.ax_full_sino.set_facecolor('black')
            self.ax_full_sino.imshow(self.full_sino, cmap='gray', aspect='auto'); self.ax_full_sino.axis('off')
            self.fig_full_sino.subplots_adjust(left=0, right=1, top=1, bottom=0); self.canvas_full_sino.draw_idle()

        if hasattr(self, 'sparse_sino') and self.sparse_sino is not None:
            self.ax_sparse_sino.clear(); self.ax_sparse_sino.set_facecolor('black')
            self.ax_sparse_sino.imshow(self.sparse_sino, cmap='gray', aspect='auto'); self.ax_sparse_sino.axis('off')
            self.fig_sparse_sino.subplots_adjust(left=0, right=1, top=1, bottom=0); self.canvas_sparse_sino.draw_idle()

        if hasattr(self, 'error_map') and self.error_map is not None:
            self.ax_nmse.clear(); self.ax_nmse.set_facecolor("black")
            im = self.ax_nmse.imshow(self.error_map, cmap="hot", vmin=0.0, vmax=float(np.max(self.error_map)) if np.max(self.error_map) > 0 else 1.0)
            self.ax_nmse.axis("off")
            self.ax_nmse.text(0.5, -0.15, f"NMSE: {self.sparse_nmse:.4f}\nPSNR: {self.sparse_psnr:.2f} dB", color="#FF4800", fontsize=10, ha="center", va="top", transform=self.ax_nmse.transAxes, bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555'))

            if hasattr(self, 'cbar_nmse_ax'): self.cbar_nmse_ax.remove(); delattr(self, 'cbar_nmse_ax')
            cax = make_axes_locatable(self.ax_nmse).append_axes("right", size="5%", pad=0.05)
            self.cbar_nmse_ax = cax; self.cbar_nmse = self.fig_nmse.colorbar(im, cax=cax)
            self.cbar_nmse.ax.yaxis.set_tick_params(color='white')
            for label in self.cbar_nmse.ax.get_yticklabels(): label.set_color('white')

        self.fig_nmse.subplots_adjust(left=0, right=0.88, top=1, bottom=0); self.canvas_nmse.draw_idle()