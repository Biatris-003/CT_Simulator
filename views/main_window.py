import sys
import os
import datetime

# =========================
# PYTHON STANDARD LIBRARIES
# =========================
import numpy as np
import matplotlib.pyplot as plt

# =========================
# PYQT5
# =========================
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QLabel,
    QSlider,
    QSizePolicy
)

from PyQt5.QtCore import Qt

# =========================
# MATPLOTLIB QT BACKEND
# =========================
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# =========================
# SCIKIT-IMAGE
# =========================
from skimage.transform import radon, iradon

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.phantom_material_map import build_three_material_phantom, build_three_material_mu_map
from models.spectra_physics import generate_spectrum_physics, generate_physics_sinogram
from models.reconstruction import SparseReconstruction, ComparisonReconstruction
from models.iterative_reconstruction import IterativeReconstruction
from views import style
from views.dialogs.spectrum_workspace_dialog import SpectrumWorkspaceDialog
from views.dialogs.fbp_metric_dialog import FBPMetricDialog
from views.dialogs.lsr_metric_dialog import LSRMetricDialog


class SimulatorCTLabApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # ================= WINDOW =================
        self.setWindowTitle("CTlab Simulator")
        self.resize(1600, 1000)

        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        # ================= DATA =================
        self.fantom = None
        self.material_phantom = None

        self.q = None
        self.E0 = None
        self._cached_spectrum_key = None
        self._cached_mu_map = None
        self._cached_total_i0 = None
        self._cached_full_sino = None
        self._cached_sparse_sino = None
        self._cached_full_angles = None
        self._cached_sparse_angles = None
        self._cached_step_angle = None
        self._cached_sparse_lsr = None  # Cache for sparse LSR reconstruction
        self._cached_full_lsr = None     # Cache for full LSR reconstruction
        self._cached_lsr_iterations = None

        self.kVp = 100
        self.mA = 2
        self.Cu = 0.0
        self.Al = 0.0

        self.step_angle = 1
        self.iterations = 10

        # ================= CENTRAL =================
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout(central_widget)

        # ================= BUILD UI =================
        self._build_main_workspace()

        # ================= INITIAL PHANTOM =================
        self.load_phantom()
        self._refresh_workspace()

    # =====================================================
    # MAIN WORKSPACE
    # =====================================================
    def _build_main_workspace(self):
        workspace_grid = QGridLayout()
        self.main_layout.addLayout(workspace_grid)

        # =================================================
        # ROW 1
        # =================================================

        # ---------- Original Phantom ----------
        phantom_group = QGroupBox("Original Phantom")
        phantom_layout = QVBoxLayout(phantom_group)

        self.fig_phantom, self.ax_phantom = plt.subplots(facecolor="#1E1E2E")
        self.canvas_phantom = FigureCanvas(self.fig_phantom)
        phantom_layout.addWidget(self.canvas_phantom)

        workspace_grid.addWidget(phantom_group, 0, 1)

        # ---------- NMSE FBP vs Dense ----------
        fbp_nmse_group = QGroupBox("NMSE: FBP vs Dense")
        fbp_nmse_layout = QVBoxLayout(fbp_nmse_group)

        self.fig_fbp_nmse, self.ax_fbp_nmse = plt.subplots(facecolor="#1E1E2E")
        self.canvas_fbp_nmse = FigureCanvas(self.fig_fbp_nmse)
        fbp_nmse_layout.addWidget(self.canvas_fbp_nmse)
        self.canvas_fbp_nmse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        workspace_grid.addWidget(fbp_nmse_group, 1, 0)


        # ---------- NMSE LSR vs Dense ----------
        lsr_nmse_group = QGroupBox("NMSE: LSR (Sparse vs Full)")
        lsr_nmse_layout = QVBoxLayout(lsr_nmse_group)

        self.fig_lsr_nmse, self.ax_lsr_nmse = plt.subplots(facecolor="#1E1E2E")
        self.canvas_lsr_nmse = FigureCanvas(self.fig_lsr_nmse)
        lsr_nmse_layout.addWidget(self.canvas_lsr_nmse)

        workspace_grid.addWidget(lsr_nmse_group, 1, 2)

        # =================================================
        # ROW 2
        # =================================================

        # ---------- FBP Reconstruction ----------
        fbp_group = QGroupBox("FBP Reconstruction (Sparse)")
        fbp_layout = QVBoxLayout(fbp_group)

        self.fig_fbp_full, self.ax_fbp_full = plt.subplots(facecolor="#1E1E2E")
        self.canvas_fbp_full = FigureCanvas(self.fig_fbp_full)
        self.canvas_fbp_full.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fbp_layout.addWidget(self.canvas_fbp_full)

        workspace_grid.addWidget(fbp_group, 0, 0)

        # ---------- Iterative Least Squares Reconstruction ----------
        ls_group = QGroupBox("LSR Reconstruction (Sparse)")
        ls_layout = QVBoxLayout(ls_group)

        self.fig_lsr_sparse, self.ax_lsr_sparse = plt.subplots(facecolor="#1E1E2E")
        self.canvas_lsr_sparse = FigureCanvas(self.fig_lsr_sparse)
        self.canvas_lsr_sparse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ls_layout.addWidget(self.canvas_lsr_sparse)

        workspace_grid.addWidget(ls_group, 0, 2)

        # =================================================
        # CONTROL PANEL (RIGHT SIDE)
        # =================================================

        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        # ---------- Spectrum Workspace Button ----------
        self.btn_open_spectrum = QPushButton("Open Spectrum Workspace")
        self.btn_open_spectrum.clicked.connect(self.show_spectrum_workspace)
        controls_layout.addWidget(self.btn_open_spectrum)

        # ---------- FBP Metric ----------
        self.btn_compare_fbp = QPushButton("Compare FBP Metric")
        self.btn_compare_fbp.clicked.connect(self.show_fbp_metric_dialog)
        controls_layout.addWidget(self.btn_compare_fbp)

        # ---------- LSR Metric ----------
        self.btn_compare_lsr = QPushButton("Compare LSR Metric")
        self.btn_compare_lsr.clicked.connect(self.show_lsr_metric_dialog)
        controls_layout.addWidget(self.btn_compare_lsr)

        # ---------- FBP vs LSM ----------
        self.btn_compare_fbp_lsm = QPushButton("Compare FBP vs LSM Metric")
        # self.btn_compare_fbp_lsm.clicked.connect(...)
        controls_layout.addWidget(self.btn_compare_fbp_lsm)

        # ---------- Step Angle ----------
        controls_layout.addWidget(QLabel("Step Angle"))

        step_angle_row = QHBoxLayout()

        self.step_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_angle_slider.setRange(1, 10)
        self.step_angle_slider.setValue(self.step_angle)
        self.step_angle_slider.setSingleStep(1)
        self.step_angle_slider.setPageStep(1)

        self.step_angle_label = QLabel(f"{self.step_angle}°")

        self.step_angle_slider.valueChanged.connect(
            lambda v: self.step_angle_label.setText(f"{v}°")
        )
        self.step_angle_slider.sliderReleased.connect(self._apply_step_angle_from_main)

        step_angle_row.addWidget(self.step_angle_slider)
        step_angle_row.addWidget(self.step_angle_label)

        controls_layout.addLayout(step_angle_row)


        # ---------- Iterations ----------
        controls_layout.addWidget(QLabel("Iterations"))

        iter_row = QHBoxLayout()

        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(1, 100)
        self.iter_slider.setValue(self.iterations)
        self.iter_slider.setSingleStep(1)
        self.iter_slider.setPageStep(1)

        self.iter_label = QLabel(str(self.iterations))

        self.iter_slider.valueChanged.connect(
            lambda v: self.iter_label.setText(str(v))
        )
        self.iter_slider.sliderReleased.connect(self._apply_iterations_from_main)

        iter_row.addWidget(self.iter_slider)
        iter_row.addWidget(self.iter_label)

        controls_layout.addLayout(iter_row)

        controls_layout.addStretch()

        # Controls on right side spanning both rows
        workspace_grid.addWidget(controls_group, 1, 1)

        # =================================================
        # GRID STRETCH
        # =================================================
        for row in range(2):
            workspace_grid.setRowStretch(row, 1)

        for col in range(3):
            workspace_grid.setColumnStretch(col, 1)

        # =================================================
        # STABLE FIGURE LAYOUTS
        # =================================================
        all_figs = [
            self.fig_phantom,
            self.fig_lsr_nmse,
            self.fig_fbp_nmse,
            self.fig_fbp_full,
            self.fig_lsr_sparse,
        ]

        for fig in all_figs:
            fig.subplots_adjust(
                left=0.05,
                right=0.95,
                top=0.92,
                bottom=0.08
            )

    def load_phantom(self):
        self.material_phantom = build_three_material_phantom(size=512)
        self.fantom = self.material_phantom

        self.ax_phantom.clear()

        # custom colormap: Air, Soft tissue, Bone
        custom_cmap = ListedColormap(["black", "gray", "white"])

        image = self.ax_phantom.imshow(
            self.material_phantom,
            cmap=custom_cmap,
            vmin=0,
            vmax=2
        )

        # self.ax_phantom.set_title("3-Material Phantom", color="white")
        self.ax_phantom.axis("off")

        colorbar = self.fig_phantom.colorbar(
            image,
            ax=self.ax_phantom,
            fraction=0.046,
            pad=0.04
        )

        colorbar.set_ticks([0, 1, 2])
        colorbar.set_ticklabels(["Air", "Soft tissue", "Bone"])

        self.canvas_phantom.draw()

    def _render_sinograms(self, mu_map, total_i0):
        """Update cached sinograms for FBP and LSR display"""
        if mu_map is None or total_i0 is None:
            return

        # recompute whenever the step angle changes or the cache is empty
        if (
            self._cached_sparse_sino is None
            or self._cached_full_sino is None
            or self._cached_step_angle != self.step_angle
        ):
            full_sino, sparse_sino, full_angles, sparse_angles = generate_physics_sinogram(
                mu_map,
                total_i0,
                user_step_angle=self.step_angle,
            )
            self._cached_full_sino = full_sino
            self._cached_sparse_sino = sparse_sino
            self._cached_full_angles = full_angles
            self._cached_sparse_angles = sparse_angles
            self._cached_step_angle = self.step_angle
            # Invalidate LSR cache when sinograms change
            self._cached_sparse_lsr = None
            self._cached_full_lsr = None

    def _render_sparse_fbp_only(self):
        """Render FBP reconstruction (sparse) in main window"""
        if self._cached_sparse_sino is None or self._cached_sparse_angles is None:
            return

        sparse_fbp = SparseReconstruction.fbp_reconstruction(
            self._cached_sparse_sino,
            self._cached_sparse_angles,
            filter_name="ramp",
        )

        full_fbp = None
        if self._cached_full_sino is not None and self._cached_full_angles is not None:
            full_fbp = SparseReconstruction.fbp_reconstruction(
                self._cached_full_sino,
                self._cached_full_angles,
                filter_name="ramp",
            )

        self.ax_fbp_full.clear()
        self.ax_fbp_full.set_facecolor("black")
        self.ax_fbp_full.imshow(sparse_fbp, cmap="gray")
        self.ax_fbp_full.set_title(f"Sparse FBP @ mA: {self.mA}, kVp: {self.kVp}", color="white")
        self.ax_fbp_full.axis("off")
        self.canvas_fbp_full.draw_idle()

        self._render_fbp_nmse(full_fbp, sparse_fbp)

    # def _render_sparse_lsr_only(self):
    #     """Render LSR reconstruction (sparse) in main window"""
    #     if self._cached_sparse_sino is None or self._cached_sparse_angles is None:
    #         return

    #     # Compute sparse LSR with current iterations
    #     sparse_lsr = IterativeReconstruction.sirt_reconstruction_fast(
    #         self._cached_sparse_sino,
    #         self._cached_sparse_angles,
    #         iterations=self.iterations,
    #         damping_factor=0.5,
    #         verbose=False
    #     )

    #     # Compute full LSR for reference (cache it)
    #     if self._cached_full_lsr is None:
    #         self._cached_full_lsr = IterativeReconstruction.sirt_reconstruction_fast(
    #             self._cached_full_sino,
    #             self._cached_full_angles,
    #             iterations=self.iterations,
    #             damping_factor=0.5,
    #             verbose=False
    #         )

    #     self.ax_lsr_sparse.clear()
    #     self.ax_lsr_sparse.set_facecolor("black")
    #     self.ax_lsr_sparse.imshow(sparse_lsr, cmap="gray")
    #     self.ax_lsr_sparse.set_title(f"Sparse LSR @ mA: {self.mA}, kVp: {self.kVp}, iter: {self.iterations}", 
    #                                  color="white")
    #     self.ax_lsr_sparse.axis("off")
    #     self.canvas_lsr_sparse.draw_idle()

    #     self._render_lsr_nmse(self._cached_full_lsr, sparse_lsr)

    def _render_sparse_lsr_only(self):
        """Render LSR reconstruction (sparse) in main window"""
        if self._cached_sparse_sino is None or self._cached_sparse_angles is None:
            return

        # Compute sparse LSR with current iterations using PROPER SIRT
        sparse_lsr = IterativeReconstruction.sirt_reconstruction(
            self._cached_sparse_sino,
            self._cached_sparse_angles,
            iterations=self.iterations,
            damping_factor=0.5,
            verbose=False
        )

        # Compute full LSR for reference (cache it)
        if self._cached_full_lsr is None:
            self._cached_full_lsr = IterativeReconstruction.sirt_reconstruction(
                self._cached_full_sino,
                self._cached_full_angles,
                iterations=self.iterations,
                damping_factor=0.5,
                verbose=False
            )

        self.ax_lsr_sparse.clear()
        self.ax_lsr_sparse.set_facecolor("black")
        self.ax_lsr_sparse.imshow(sparse_lsr, cmap="gray")
        self.ax_lsr_sparse.set_title(f"Sparse LSR @ mA: {self.mA}, kVp: {self.kVp}, iter: {self.iterations}", 
                                     color="white")
        self.ax_lsr_sparse.axis("off")
        self.canvas_lsr_sparse.draw_idle()

        self._render_lsr_nmse(self._cached_full_lsr, sparse_lsr)

    def _render_fbp_nmse(self, full_fbp, sparse_fbp):
        """Render FBP NMSE comparison"""
        self.ax_fbp_nmse.clear()
        self.ax_fbp_nmse.set_facecolor("black")

        if full_fbp is None or sparse_fbp is None:
            self.ax_fbp_nmse.text(0.5, 0.5, "NMSE unavailable", color="white",
                                  ha="center", va="center", transform=self.ax_fbp_nmse.transAxes)
            self.ax_fbp_nmse.axis("off")
            self.canvas_fbp_nmse.draw_idle()
            return

        metrics = ComparisonReconstruction.compute_reconstruction_error(full_fbp, sparse_fbp)
        error_map = metrics["emap"]
        nmse = metrics["nmse"]
        psnr = metrics["psnr"]

        vmax = float(np.max(error_map))
        if vmax <= 0.0:
            vmax = 1.0

        im = self.ax_fbp_nmse.imshow(error_map, cmap="hot", vmin=0.0, vmax=vmax)
        self.ax_fbp_nmse.axis("off")

        self.ax_fbp_nmse.text(
            0.5,
            -0.12,
            f"NMSE: {nmse:.4f}\nPSNR: {psnr:.2f} dB",
            color="#FF4800",
            fontsize=10,
            ha="center",
            va="top",
            transform=self.ax_fbp_nmse.transAxes,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555')
        )

        if hasattr(self, 'cbar_fbp_nmse'):
            try:
                self.cbar_fbp_nmse.remove()
            except Exception:
                pass
            try:
                if hasattr(self, 'cbar_fbp_nmse_ax'):
                    self.cbar_fbp_nmse_ax.remove()
                    delattr(self, 'cbar_fbp_nmse_ax')
            except Exception:
                pass

        divider = make_axes_locatable(self.ax_fbp_nmse)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar_fbp_nmse_ax = cax
        self.cbar_fbp_nmse = self.fig_fbp_nmse.colorbar(im, cax=cax)
        self.canvas_fbp_nmse.draw_idle()

    def _render_lsr_nmse(self, full_lsr, sparse_lsr):
        """Render LSR NMSE comparison (Sparse vs Full)"""
        self.ax_lsr_nmse.clear()
        self.ax_lsr_nmse.set_facecolor("black")

        if full_lsr is None or sparse_lsr is None:
            self.ax_lsr_nmse.text(0.5, 0.5, "NMSE unavailable", color="white",
                                  ha="center", va="center", transform=self.ax_lsr_nmse.transAxes)
            self.ax_lsr_nmse.axis("off")
            self.canvas_lsr_nmse.draw_idle()
            return

        metrics = ComparisonReconstruction.compute_reconstruction_error(full_lsr, sparse_lsr)
        error_map = metrics["emap"]
        nmse = metrics["nmse"]
        psnr = metrics["psnr"]

        vmax = float(np.max(error_map))
        if vmax <= 0.0:
            vmax = 1.0

        im = self.ax_lsr_nmse.imshow(error_map, cmap="hot", vmin=0.0, vmax=vmax)
        self.ax_lsr_nmse.axis("off")

        self.ax_lsr_nmse.text(
            0.5,
            -0.12,
            f"NMSE: {nmse:.4f}\nPSNR: {psnr:.2f} dB",
            color="#FF4800",
            fontsize=10,
            ha="center",
            va="top",
            transform=self.ax_lsr_nmse.transAxes,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555')
        )

        if hasattr(self, 'cbar_lsr_nmse'):
            try:
                self.cbar_lsr_nmse.remove()
            except Exception:
                pass
            try:
                if hasattr(self, 'cbar_lsr_nmse_ax'):
                    self.cbar_lsr_nmse_ax.remove()
                    delattr(self, 'cbar_lsr_nmse_ax')
            except Exception:
                pass

        divider = make_axes_locatable(self.ax_lsr_nmse)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar_lsr_nmse_ax = cax
        self.cbar_lsr_nmse = self.fig_lsr_nmse.colorbar(im, cax=cax)
        self.canvas_lsr_nmse.draw_idle()

    def _refresh_workspace(self, *args, sync_dialog=True):
        if self.material_phantom is None:
            return

        self.step_angle = self.step_angle_slider.value()
        self.step_angle_label.setText(f"{self.step_angle}°")

        spectrum_key = (self.kVp, self.mA, self.Cu, self.Al)
        if self._cached_spectrum_key != spectrum_key or self._cached_mu_map is None:
            _, self._cached_mu_map = build_three_material_mu_map(
                size=self.material_phantom.shape[0],
                kvp=self.kVp,
            )
            _, _, self._cached_total_i0 = generate_spectrum_physics(
                self.kVp,
                self.mA,
                self.Cu,
                self.Al,
            )
            self._cached_spectrum_key = spectrum_key

        self._render_sinograms(self._cached_mu_map, self._cached_total_i0)
        self._render_sparse_fbp_only()
        self._render_sparse_lsr_only()

        if sync_dialog and getattr(self, "spectrum_dialog", None) is not None:
            self.spectrum_dialog.sync_step_angle_from_main(self.step_angle)

    def _apply_step_angle_from_main(self):
        self.step_angle = self.step_angle_slider.value()
        self.step_angle_label.setText(f"{self.step_angle}°")
        self._refresh_workspace()

    def _apply_iterations_from_main(self):
        """Handle iterations slider release - update LSR in main window"""
        self.iterations = self.iter_slider.value()
        self.iter_label.setText(str(self.iterations))
        # Refresh LSR display only (fast operation)
        self._render_sparse_lsr_only()

    def sync_step_angle_from_dialog(self, step_angle):
        self.step_angle = int(step_angle)
        self.step_angle_slider.blockSignals(True)
        self.step_angle_slider.setValue(self.step_angle)
        self.step_angle_slider.blockSignals(False)
        self.step_angle_label.setText(f"{self.step_angle}°")
        self._refresh_workspace(sync_dialog=False)

    def show_spectrum_workspace(self):
        if getattr(self, "spectrum_dialog", None) is None:
            self.spectrum_dialog = SpectrumWorkspaceDialog(self, self.material_phantom)

        # Keep the workspace step angle aligned with the main UI before showing it
        self.spectrum_dialog.sync_step_angle_from_main(self.step_angle)
        self.spectrum_dialog.show()
        self.spectrum_dialog.raise_()
        self.spectrum_dialog.activateWindow()

    def preview_spectrum(self, q, energies, kvp, ma, cu, al):
        self.q = q
        self.E0 = energies
        self.kVp = kvp
        self.mA = ma
        self.Cu = cu
        self.Al = al
        self._refresh_workspace(sync_dialog=False)

    def show_fbp_metric_dialog(self):
        if self._cached_full_sino is None or self._cached_sparse_sino is None:
            self._refresh_workspace(sync_dialog=False)

        _, mu_map = build_three_material_mu_map(
            size=self.material_phantom.shape[0],
            kvp=self.kVp,
        )

        recon_results = ComparisonReconstruction.reconstruct_fbp_from_sinograms(
            self._cached_full_sino,
            self._cached_sparse_sino,
            self._cached_full_angles,
            self._cached_sparse_angles,
            original=mu_map,
            filter_name="ramp",
        )

        self.fbp_dialog = FBPMetricDialog(
            self,
            original=mu_map,
            total_i0=self._cached_total_i0,
            step_angle=self.step_angle,
            full_recon=recon_results['full_recon'],
            sparse_recon=recon_results['sparse_recon'],
            full_nmse=recon_results.get('full_nmse'),
            sparse_nmse=recon_results.get('sparse_nmse'),
        )
        self.fbp_dialog.show()
        self.fbp_dialog.raise_()
        self.fbp_dialog.activateWindow()

    def show_lsr_metric_dialog(self):
        """Open LSR Metric Dialog with full and sparse sinograms"""
        if self._cached_full_sino is None or self._cached_sparse_sino is None:
            self._refresh_workspace(sync_dialog=False)

        _, mu_map = build_three_material_mu_map(
            size=self.material_phantom.shape[0],
            kvp=self.kVp,
        )

        self.lsr_dialog = LSRMetricDialog(
            self,
            original=mu_map,
            total_i0=self._cached_total_i0,
            step_angle=self.step_angle,
            full_sino=self._cached_full_sino,
            sparse_sino=self._cached_sparse_sino,
            full_angles=self._cached_full_angles,
            sparse_angles=self._cached_sparse_angles,
            iterations=self.iterations,
        )

        self.lsr_dialog.show()
        self.lsr_dialog.raise_()
        self.lsr_dialog.activateWindow()

    def chosen_spectrum(self, q, energies, kvp, ma, cu, al):
        self.preview_spectrum(q, energies, kvp, ma, cu, al)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimulatorCTLabApp()
    window.show()
    sys.exit(app.exec())