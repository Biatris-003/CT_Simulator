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
from models.spectra_physics import generate_physics_sinogram

class LSRMetricDialog(QDialog):
    def __init__(
        self,
        parent=None,
        original=None,
        total_i0=None,
        step_angle=1,
        full_recon=None,
        sparse_recon=None,
        full_nmse=None,
        sparse_nmse=None,
        full_sino=None,
        sparse_sino=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Compare LSR Metric (Least Squares Reconstruction)")
        self.resize(1400, 950)

        # Apply styles if available
        # style.apply_matplotlib_theme()
        # self.setStyleSheet(style.MODERN_STYLE)

        self.original = original
        self.total_i0 = total_i0
        self.step_angle = int(step_angle)
        self.full_recon = full_recon
        self.sparse_recon = sparse_recon
        self.full_sino = full_sino
        self.sparse_sino = sparse_sino
        self.full_nmse = full_nmse
        self.sparse_nmse = sparse_nmse
        self.sparse_psnr = 0.0
        self.error_map = None

        # =================================================
        # MAIN LAYOUT
        # =================================================
        layout = QVBoxLayout(self)
        main_row = QHBoxLayout()
        layout.addLayout(main_row)

        # =================================================
        # COLUMN 1: FULL LSR
        # =================================================
        col_full = QVBoxLayout()

        self.full_group = QGroupBox("Full LSR Reconstruction")
        full_layout = QVBoxLayout(self.full_group)
        self.fig_full, self.ax_full = plt.subplots(figsize=(6, 5), facecolor="#1E1E2E")
        self.canvas_full = FigureCanvas(self.fig_full)
        full_layout.addWidget(self.canvas_full)
        col_full.addWidget(self.full_group, 3)

        self.full_sino_group = QGroupBox("Full Sinogram")
        full_sino_layout = QVBoxLayout(self.full_sino_group)
        self.fig_full_sino, self.ax_full_sino = plt.subplots(facecolor="#1E1E2E")
        self.canvas_full_sino = FigureCanvas(self.fig_full_sino)
        full_sino_layout.addWidget(self.canvas_full_sino)
        col_full.addWidget(self.full_sino_group, 2)

        main_row.addLayout(col_full, 1)

        # =================================================
        # COLUMN 2: SPARSE LSR
        # =================================================
        col_sparse = QVBoxLayout()

        self.sparse_group = QGroupBox("Sparse LSR Reconstruction")
        sparse_layout = QVBoxLayout(self.sparse_group)
        self.fig_sparse, self.ax_sparse = plt.subplots(figsize=(6, 5), facecolor="#1E1E2E")
        self.canvas_sparse = FigureCanvas(self.fig_sparse)
        sparse_layout.addWidget(self.canvas_sparse)
        col_sparse.addWidget(self.sparse_group, 3)

        self.sparse_sino_group = QGroupBox("Sparse Sinogram")
        sparse_sino_layout = QVBoxLayout(self.sparse_sino_group)
        self.fig_sparse_sino, self.ax_sparse_sino = plt.subplots(facecolor="#1E1E2E")
        self.canvas_sparse_sino = FigureCanvas(self.fig_sparse_sino)
        sparse_sino_layout.addWidget(self.canvas_sparse_sino)
        col_sparse.addWidget(self.sparse_sino_group, 2)

        main_row.addLayout(col_sparse, 1)

        # =================================================
        # COLUMN 3: NMSE / METRICS
        # =================================================
        col_nmse = QVBoxLayout()
        self.nmse_group = QGroupBox("LSR Error Analysis")
        nmse_layout = QVBoxLayout(self.nmse_group)

        self.fig_nmse, self.ax_nmse = plt.subplots(figsize=(5, 5), facecolor="#1E1E2E")
        self.canvas_nmse = FigureCanvas(self.fig_nmse)
        nmse_layout.addWidget(self.canvas_nmse)
        col_nmse.addWidget(self.nmse_group, 1)

        main_row.addLayout(col_nmse, 1)

        # =================================================
        # CONTROLS (Step Angle)
        # =================================================
        # =================================================
        # CONTROLS (Step Angle)
        # =================================================
        controls_row = QHBoxLayout()
        layout.addLayout(controls_row)

        controls_row.addWidget(QLabel("Step Angle:"))

        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(1, 10)
        self.step_slider.setValue(self.step_angle)
        self.step_slider.setSingleStep(1)
        self.step_slider.setPageStep(1)

        self.step_label = QLabel(f"{self.step_angle}°")

        self.step_slider.valueChanged.connect(
            lambda v: self.step_label.setText(f"{v}°")
        )
        self.step_slider.sliderReleased.connect(self._recompute_from_step)

        controls_row.addWidget(self.step_slider)
        controls_row.addWidget(self.step_label)


        # =================================================
        # CONTROLS (Iterations) - NEW SLIDER BELOW
        # =================================================
        iterations_row = QHBoxLayout()
        layout.addLayout(iterations_row)

        iterations_row.addWidget(QLabel("Iterations:"))

        self.iter_slider = QSlider(Qt.Orientation.Horizontal)
        self.iter_slider.setRange(1, 100)   # Adjust max as needed
        self.iter_slider.setValue(10)       # Default value
        self.iter_slider.setSingleStep(1)
        self.iter_slider.setPageStep(5)

        self.iter_label = QLabel("10")

        # Only updates label, no reconstruction function connected
        self.iter_slider.valueChanged.connect(
            lambda v: self.iter_label.setText(f"{v}")
        )

        iterations_row.addWidget(self.iter_slider)
        iterations_row.addWidget(self.iter_label)

        if self.original is not None and self.total_i0 is not None:
            self._recompute_sinograms()

        # Final Render Call
        self._render_all()

    def _recompute_from_step(self):
        self.step_angle = self.step_slider.value()
        self.step_label.setText(f"{self.step_angle}°")
        self._recompute_sinograms()
        self._render_all()

    def _recompute_sinograms(self):
        if self.original is None or self.total_i0 is None:
            return

        full_sino, sparse_sino, _, _ = generate_physics_sinogram(
            self.original,
            self.total_i0,
            user_step_angle=self.step_angle,
        )
        self.full_sino = full_sino
        self.sparse_sino = sparse_sino

    def _render_all(self):
        """Updates the canvases with current data."""
        # Helper to clear and format axes
        for ax, fig in [(self.ax_full, self.fig_full), (self.ax_sparse, self.fig_sparse), 
                        (self.ax_full_sino, self.fig_full_sino), (self.ax_sparse_sino, self.fig_sparse_sino)]:
            ax.clear()
            ax.set_facecolor("black")
            ax.axis("off")
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Keep these areas empty for now (future LSR recon/error visualizations).
        for ax in [self.ax_full, self.ax_sparse, self.ax_full_sino, self.ax_sparse_sino, self.ax_nmse]:
            ax.clear()
            ax.set_facecolor("black")
            ax.axis("off")

        # Show only sinograms in their placeholders.
        if self.full_sino is not None:
            self.ax_full_sino.imshow(self.full_sino, cmap="gray", aspect="auto")
            self.ax_full_sino.set_title("Full Sinogram", color="white")

        if self.sparse_sino is not None:
            self.ax_sparse_sino.imshow(self.sparse_sino, cmap="gray", aspect="auto")
            self.ax_sparse_sino.set_title("Sparse Sinogram", color="white")

        # Refresh all canvases
        self.canvas_full.draw_idle()
        self.canvas_sparse.draw_idle()
        self.canvas_full_sino.draw_idle()
        self.canvas_sparse_sino.draw_idle()
        self.canvas_nmse.draw_idle()