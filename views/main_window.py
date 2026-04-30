import sys
import os

# =========================
# PYTHON STANDARD LIBRARIES
# =========================
import numpy as np

# =========================
# PYQT5
# =========================
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QPushButton, QLabel, QSlider, QSizePolicy
)
from PyQt5.QtCore import Qt

# =========================
# MATPLOTLIB QT BACKEND (Object-Oriented Only)
# =========================
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure 
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

        self.setWindowTitle("CTlab Simulator")
        self.resize(1600, 1000)
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

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
        self._cached_sparse_fbp = None
        self._cached_full_fbp = None
        self._cached_sparse_lsr = None
        self._cached_full_lsr = None
        self._cached_full_lsr_key = None   

        self.kVp = 100
        self.mA = 2
        self.Cu = 0.0
        self.Al = 0.0
        self.step_angle = 1
        self.iterations = 10

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QVBoxLayout(central_widget)
        self._build_main_workspace()
        self.load_phantom()

    def _build_main_workspace(self):
        workspace_grid = QGridLayout()
        self.main_layout.addLayout(workspace_grid)

        # ---------- Original Phantom ----------
        phantom_group = QGroupBox("Original Phantom")
        phantom_layout = QVBoxLayout(phantom_group)
        self.fig_phantom = Figure(facecolor="#1E1E2E")
        self.ax_phantom = self.fig_phantom.add_subplot(111)
        self.canvas_phantom = FigureCanvas(self.fig_phantom)
        phantom_layout.addWidget(self.canvas_phantom)
        workspace_grid.addWidget(phantom_group, 0, 1)

        # ---------- NMSE FBP vs Dense ----------
        fbp_nmse_group = QGroupBox("NMSE: FBP")
        fbp_nmse_layout = QVBoxLayout(fbp_nmse_group)
        self.fig_fbp_nmse = Figure(facecolor="#1E1E2E")
        self.ax_fbp_nmse = self.fig_fbp_nmse.add_subplot(111)
        self.canvas_fbp_nmse = FigureCanvas(self.fig_fbp_nmse)
        self.canvas_fbp_nmse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fbp_nmse_layout.addWidget(self.canvas_fbp_nmse)
        workspace_grid.addWidget(fbp_nmse_group, 1, 0)

        # ---------- NMSE LSR vs Dense ----------
        lsr_nmse_group = QGroupBox("NMSE: LSR")
        lsr_nmse_layout = QVBoxLayout(lsr_nmse_group)
        self.fig_lsr_nmse = Figure(facecolor="#1E1E2E")
        self.ax_lsr_nmse = self.fig_lsr_nmse.add_subplot(111)
        self.canvas_lsr_nmse = FigureCanvas(self.fig_lsr_nmse)
        lsr_nmse_layout.addWidget(self.canvas_lsr_nmse)
        workspace_grid.addWidget(lsr_nmse_group, 1, 2)

        # ---------- FBP Reconstruction ----------
        fbp_group = QGroupBox("FBP Reconstruction (Sparse)")
        fbp_layout = QVBoxLayout(fbp_group)
        self.fig_fbp_full = Figure(facecolor="#1E1E2E")
        self.ax_fbp_full = self.fig_fbp_full.add_subplot(111)
        self.canvas_fbp_full = FigureCanvas(self.fig_fbp_full)
        self.canvas_fbp_full.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fbp_layout.addWidget(self.canvas_fbp_full)
        workspace_grid.addWidget(fbp_group, 0, 0)

        # ---------- LSR Reconstruction ----------
        ls_group = QGroupBox("LSR Reconstruction (Sparse)")
        ls_layout = QVBoxLayout(ls_group)
        self.fig_lsr_sparse = Figure(facecolor="#1E1E2E")
        self.ax_lsr_sparse = self.fig_lsr_sparse.add_subplot(111)
        self.canvas_lsr_sparse = FigureCanvas(self.fig_lsr_sparse)
        self.canvas_lsr_sparse.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        ls_layout.addWidget(self.canvas_lsr_sparse)
        workspace_grid.addWidget(ls_group, 0, 2)

        # ================= CONTROL PANEL =================
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)

        self.btn_open_spectrum = QPushButton("Open Spectrum Workspace")
        self.btn_open_spectrum.clicked.connect(self.show_spectrum_workspace)
        controls_layout.addWidget(self.btn_open_spectrum)

        self.btn_compare_fbp = QPushButton("FBP Reconstruction Analysis")
        self.btn_compare_fbp.clicked.connect(self.show_fbp_metric_dialog)
        controls_layout.addWidget(self.btn_compare_fbp)

        self.btn_compare_lsr = QPushButton("LSR Reconstruction Analysis")
        self.btn_compare_lsr.clicked.connect(self.show_lsr_metric_dialog)
        controls_layout.addWidget(self.btn_compare_lsr)

        self.btn_compare_fbp_lsm = QPushButton("Run Simulation")
        self.btn_compare_fbp_lsm.clicked.connect(self.compare_fbp_vs_lsm)
        controls_layout.addWidget(self.btn_compare_fbp_lsm)

        controls_layout.addWidget(QLabel("Step Angle"))
        step_angle_row = QHBoxLayout()
        self.step_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_angle_slider.setRange(1, 10)
        self.step_angle_slider.setValue(self.step_angle)
        self.step_angle_slider.setSingleStep(1)
        self.step_angle_slider.setPageStep(1)
        self.step_angle_label = QLabel(f"{self.step_angle}°")
        self.step_angle_slider.valueChanged.connect(self._on_step_angle_changed)
        step_angle_row.addWidget(self.step_angle_slider)
        step_angle_row.addWidget(self.step_angle_label)
        controls_layout.addLayout(step_angle_row)

        controls_layout.addWidget(QLabel("Iterations"))
        iter_row = QHBoxLayout()
        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(1, 40)
        self.iter_slider.setValue(self.iterations)
        self.iter_slider.setSingleStep(1)
        self.iter_slider.setPageStep(1)
        self.iter_label = QLabel(str(self.iterations))
        self.iter_slider.valueChanged.connect(self._on_iterations_changed)
        iter_row.addWidget(self.iter_slider)
        iter_row.addWidget(self.iter_label)
        controls_layout.addLayout(iter_row)

        metrics_grid = QGridLayout()
        metrics_grid.addWidget(QLabel(""), 0, 0)

        fbp_header = QLabel("FBP")
        lsr_header = QLabel("LSR")
        fbp_header.setAlignment(Qt.AlignCenter)
        lsr_header.setAlignment(Qt.AlignCenter)
        metrics_grid.addWidget(fbp_header, 0, 1)
        metrics_grid.addWidget(lsr_header, 0, 2)

        nmse_label = QLabel("NMSE")
        psnr_label = QLabel("PSNR")
        nmse_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        psnr_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        metrics_grid.addWidget(nmse_label, 1, 0)
        metrics_grid.addWidget(psnr_label, 2, 0)

        self.fbp_nmse_value_label = QLabel("--")
        self.lsr_nmse_value_label = QLabel("--")
        self.fbp_psnr_value_label = QLabel("-- dB")
        self.lsr_psnr_value_label = QLabel("-- dB")
        for label in (
            self.fbp_nmse_value_label,
            self.lsr_nmse_value_label,
            self.fbp_psnr_value_label,
            self.lsr_psnr_value_label,
        ):
            label.setAlignment(Qt.AlignCenter)

        metrics_grid.addWidget(self.fbp_nmse_value_label, 1, 1)
        metrics_grid.addWidget(self.lsr_nmse_value_label, 1, 2)
        metrics_grid.addWidget(self.fbp_psnr_value_label, 2, 1)
        metrics_grid.addWidget(self.lsr_psnr_value_label, 2, 2)

        controls_layout.addLayout(metrics_grid)
        controls_layout.addStretch()

        workspace_grid.addWidget(controls_group, 1, 1)

        for row in range(2): workspace_grid.setRowStretch(row, 1)
        for col in range(3): workspace_grid.setColumnStretch(col, 1)

        for fig in [self.fig_phantom, self.fig_lsr_nmse, self.fig_fbp_nmse, self.fig_fbp_full, self.fig_lsr_sparse]:
            fig.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)

    def load_phantom(self):
        self.material_phantom = build_three_material_phantom(size=512)
        self.fantom = self.material_phantom

        self.ax_phantom.clear()
        custom_cmap = ListedColormap(["black", "gray", "white"])
        image = self.ax_phantom.imshow(self.material_phantom, cmap=custom_cmap, vmin=0, vmax=2)
        self.ax_phantom.axis("off")

        colorbar = self.fig_phantom.colorbar(image, ax=self.ax_phantom, fraction=0.046, pad=0.04)
        colorbar.set_ticks([0, 1, 2])
        colorbar.set_ticklabels(["Air", "Soft tissue", "Bone"])

        colorbar.ax.yaxis.set_tick_params(color='white')
        for label in colorbar.ax.get_yticklabels(): label.set_color('white')
        self.canvas_phantom.draw()

    def _render_sinograms(self, mu_map, total_i0):
        if mu_map is None or total_i0 is None: return

        if (self._cached_sparse_sino is None or self._cached_full_sino is None or self._cached_step_angle != self.step_angle):
            QApplication.processEvents()
            full_sino, sparse_sino, full_angles, sparse_angles = generate_physics_sinogram(
                mu_map, total_i0, user_step_angle=self.step_angle,
            )
            self._cached_full_sino = full_sino
            self._cached_sparse_sino = sparse_sino
            self._cached_full_angles = full_angles
            self._cached_sparse_angles = sparse_angles
            self._cached_step_angle = self.step_angle

            self._cached_sparse_fbp = None
            self._cached_full_fbp = None
            self._cached_sparse_lsr = None

    def _render_sparse_fbp_only(self):
        if self._cached_sparse_sino is None or self._cached_sparse_angles is None: return

        if self._cached_sparse_fbp is None:
            QApplication.processEvents()
            self._cached_sparse_fbp = SparseReconstruction.fbp_reconstruction(
                self._cached_sparse_sino, self._cached_sparse_angles, filter_name="ramp",
            )

        if self._cached_full_fbp is None:
            QApplication.processEvents()
            self._cached_full_fbp = SparseReconstruction.fbp_reconstruction(
                self._cached_full_sino, self._cached_full_angles, filter_name="ramp",
            )

        self.ax_fbp_full.clear(); self.ax_fbp_full.set_facecolor("black")
        self.ax_fbp_full.imshow(self._cached_sparse_fbp, cmap="gray"); self.ax_fbp_full.axis("off")
        self.ax_fbp_full.set_title(f"Sparse FBP @ mA: {self.mA}, kVp: {self.kVp}", color="white")
        self.canvas_fbp_full.draw_idle()

        self._render_fbp_nmse(self._cached_full_fbp, self._cached_sparse_fbp)

    def _render_sparse_lsr_only(self):
        if self._cached_sparse_sino is None or self._cached_sparse_angles is None: return

        QApplication.processEvents()
        sparse_lsr = IterativeReconstruction.sirt_reconstruction(
            self._cached_sparse_sino, self._cached_sparse_angles,
            iterations=self.iterations, damping_factor=0.03, verbose=False,
        )

        full_lsr_key = (self.iterations, self._cached_spectrum_key)
        if self._cached_full_lsr is None or self._cached_full_lsr_key != full_lsr_key:
            QApplication.processEvents()
            self._cached_full_lsr = IterativeReconstruction.sirt_reconstruction(
                self._cached_full_sino, self._cached_full_angles,
                iterations=self.iterations, damping_factor=0.03, verbose=False,
            )
            self._cached_full_lsr_key = full_lsr_key

        self.ax_lsr_sparse.clear(); self.ax_lsr_sparse.set_facecolor("black")
        self.ax_lsr_sparse.imshow(sparse_lsr, cmap="gray"); self.ax_lsr_sparse.axis("off")
        self.ax_lsr_sparse.set_title(f"Sparse LSR @ mA: {self.mA}, kVp: {self.kVp}, iter: {self.iterations}", color="white")
        self.canvas_lsr_sparse.draw_idle()

        self._render_lsr_nmse(self._cached_full_lsr, sparse_lsr)

    def _render_fbp_nmse(self, full_fbp, sparse_fbp):
        self.ax_fbp_nmse.clear(); self.ax_fbp_nmse.set_facecolor("black")

        if self.material_phantom is None or sparse_fbp is None:
            self.fbp_nmse_value_label.setText("--")
            self.fbp_psnr_value_label.setText("-- dB")
            return

        metrics = ComparisonReconstruction.compute_reconstruction_error(self.material_phantom, sparse_fbp)
        im = self.ax_fbp_nmse.imshow(metrics["emap"], cmap="hot", vmin=0.0, vmax=float(np.max(metrics["emap"])) if np.max(metrics["emap"]) > 0 else 1.0)
        self.ax_fbp_nmse.axis("off")

        self.ax_fbp_nmse.text(0.5, -0.12, f"NMSE: {metrics['nmse']:.4f}\nPSNR: {metrics['psnr']:.2f} dB\n(Original vs FBP)",
            color="#FF4800", fontsize=10, ha="center", va="top", transform=self.ax_fbp_nmse.transAxes, bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555'))
        self.fbp_nmse_value_label.setText(f"{metrics['nmse']:.4f}")
        self.fbp_psnr_value_label.setText(f"{metrics['psnr']:.2f} dB")

        if hasattr(self, 'cbar_fbp_nmse_ax'): self.cbar_fbp_nmse_ax.remove(); delattr(self, 'cbar_fbp_nmse_ax')
        cax = make_axes_locatable(self.ax_fbp_nmse).append_axes("right", size="5%", pad=0.05)
        self.cbar_fbp_nmse_ax = cax; self.cbar_fbp_nmse = self.fig_fbp_nmse.colorbar(im, cax=cax)
        self.cbar_fbp_nmse.ax.yaxis.set_tick_params(color='white')
        for label in self.cbar_fbp_nmse.ax.get_yticklabels(): label.set_color('white')
            
        self.canvas_fbp_nmse.draw_idle()

    def _render_lsr_nmse(self, full_lsr, sparse_lsr):
        self.ax_lsr_nmse.clear(); self.ax_lsr_nmse.set_facecolor("black")

        if self.material_phantom is None or sparse_lsr is None:
            self.lsr_nmse_value_label.setText("--")
            self.lsr_psnr_value_label.setText("-- dB")
            return

        metrics = ComparisonReconstruction.compute_reconstruction_error(self.material_phantom, sparse_lsr)
        im = self.ax_lsr_nmse.imshow(metrics["emap"], cmap="hot", vmin=0.0, vmax=float(np.max(metrics["emap"])) if np.max(metrics["emap"]) > 0 else 1.0)
        self.ax_lsr_nmse.axis("off")

        self.ax_lsr_nmse.text(0.5, -0.12, f"NMSE: {metrics['nmse']:.4f}\nPSNR: {metrics['psnr']:.2f} dB\n(Original vs LSR)",
            color="#FF4800", fontsize=10, ha="center", va="top", transform=self.ax_lsr_nmse.transAxes, bbox=dict(facecolor='black', alpha=0.6, edgecolor='#555555'))
        self.lsr_nmse_value_label.setText(f"{metrics['nmse']:.4f}")
        self.lsr_psnr_value_label.setText(f"{metrics['psnr']:.2f} dB")

        if hasattr(self, 'cbar_lsr_nmse_ax'): self.cbar_lsr_nmse_ax.remove(); delattr(self, 'cbar_lsr_nmse_ax')
        cax = make_axes_locatable(self.ax_lsr_nmse).append_axes("right", size="5%", pad=0.05)
        self.cbar_lsr_nmse_ax = cax; self.cbar_lsr_nmse = self.fig_lsr_nmse.colorbar(im, cax=cax)
        self.cbar_lsr_nmse.ax.yaxis.set_tick_params(color='white')
        for label in self.cbar_lsr_nmse.ax.get_yticklabels(): label.set_color('white')
            
        self.canvas_lsr_nmse.draw_idle()

    def _refresh_workspace(self, *args):
        if self.material_phantom is None: return

        self.step_angle = self.step_angle_slider.value()
        self.step_angle_label.setText(f"{self.step_angle}°")

        spectrum_key = (self.kVp, self.mA, self.Cu, self.Al)
        if self._cached_spectrum_key != spectrum_key or self._cached_mu_map is None:
            QApplication.processEvents()
            _, self._cached_mu_map = build_three_material_mu_map(size=self.material_phantom.shape[0], kvp=self.kVp)
            _, _, self._cached_total_i0 = generate_spectrum_physics(self.kVp, self.mA, self.Cu, self.Al)
            self._cached_spectrum_key = spectrum_key
            self._cached_full_lsr = None
            self._cached_full_lsr_key = None

        self._render_sinograms(self._cached_mu_map, self._cached_total_i0)

    def compare_fbp_vs_lsm(self):
        QApplication.processEvents()
        self._refresh_workspace()
        QApplication.processEvents()
        self._render_sparse_fbp_only()
        QApplication.processEvents()
        self._render_sparse_lsr_only()

    def _on_step_angle_changed(self):
        self.step_angle = self.step_angle_slider.value()
        self.step_angle_label.setText(f"{self.step_angle}°")
        self.compare_fbp_vs_lsm()

    def _on_iterations_changed(self):
        self.iterations = self.iter_slider.value()
        self.iter_label.setText(str(self.iterations))
        self.compare_fbp_vs_lsm()

    def show_spectrum_workspace(self):
        if getattr(self, "spectrum_dialog", None) is None:
            self.spectrum_dialog = SpectrumWorkspaceDialog(
                self, self.material_phantom, initial_step_angle=self.step_angle,
            )
        self.spectrum_dialog.show()
        self.spectrum_dialog.raise_()
        self.spectrum_dialog.activateWindow()

    def preview_spectrum(self, q, energies, kvp, ma, cu, al, step_angle=None):
        new_step_angle = step_angle if step_angle is not None else self.step_angle

        if (kvp == self.kVp and ma == self.mA and cu == self.Cu and al == self.Al and new_step_angle == self.step_angle): return

        self.q = q
        self.E0 = energies
        self.kVp = kvp
        self.mA = ma
        self.Cu = cu
        self.Al = al
        self.step_angle = new_step_angle

        self.step_angle_slider.blockSignals(True)
        self.step_angle_slider.setValue(self.step_angle)
        self.step_angle_slider.blockSignals(False)
        self.step_angle_label.setText(f"{self.step_angle}°")

        self._cached_spectrum_key = None
        self._cached_mu_map = None
        self._cached_total_i0 = None
        self._cached_full_sino = None
        self._cached_sparse_sino = None
        self._cached_sparse_fbp = None
        self._cached_full_fbp = None
        self._cached_sparse_lsr = None
        self._cached_full_lsr = None
        self._cached_full_lsr_key = None

        self._refresh_workspace()
        self.compare_fbp_vs_lsm()

    def show_fbp_metric_dialog(self):
        # ABSOLUTE PARAMETER ISOLATION: Pass ONLY primitive numbers to the dialog
        dialog = FBPMetricDialog(
            phantom_size=self.material_phantom.shape[0],
            kVp=self.kVp,
            mA=self.mA,
            Cu=self.Cu,
            Al=self.Al,
            step_angle=self.step_angle,
            parent=self # Keep reference alive
        )
        
        # Force OS to treat this as an independent desktop window
        dialog.setWindowFlags(Qt.Window)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def show_lsr_metric_dialog(self):
        # ABSOLUTE PARAMETER ISOLATION: Pass ONLY primitive numbers to the dialog
        dialog = LSRMetricDialog(
            phantom_size=self.material_phantom.shape[0],
            kVp=self.kVp,
            mA=self.mA,
            Cu=self.Cu,
            Al=self.Al,
            step_angle=self.step_angle,
            iterations=self.iterations,
            parent=self # Keep reference alive
        )
        
        # Force OS to treat this as an independent desktop window
        dialog.setWindowFlags(Qt.Window)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def chosen_spectrum(self, q, energies, kvp, ma, cu, al, step_angle=None):
        self.preview_spectrum(q, energies, kvp, ma, cu, al, step_angle)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimulatorCTLabApp()
    window.show()
    sys.exit(app.exec())