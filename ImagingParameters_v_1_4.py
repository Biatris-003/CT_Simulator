import sys
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QLineEdit, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D

class ImagingParametersDialog(QDialog):
    def __init__(self, parent_app, current_beam_geom, current_reco, ma):
        super().__init__(parent_app)  # type: ignore
        self.parent_app = parent_app
        self.current_beam_geom = current_beam_geom
        self.current_reco = current_reco
        self.ma = ma
        
        self.setWindowTitle("Select Scan Parameters")
        self.resize(1050, 700)
        import style
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        main_layout = QHBoxLayout(self)

        # Left controls + scan preview
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 7)

        # Right graphs
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 3)

        # Set up Parameter Selection Panel
        param_group = QGroupBox("Parameter selection")
        param_grid = QGridLayout()
        param_group.setLayout(param_grid)
        
        # Fields
        self.fields = {}
        
        def add_field(name, row, col, default=""):
            lbl = QLabel(name)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            edit = QLineEdit(default)
            param_grid.addWidget(lbl, row*2, col)
            param_grid.addWidget(edit, row*2+1, col)
            self.fields[name] = edit
            return edit
            
        add_field("Min angle", 0, 0, "0")
        add_field("Step angle", 0, 1, "1")
        self.max_angle_edit = add_field("Max angle", 0, 2, "360")
        self.img_vol_edit = add_field("Image volume (pix.)", 0, 3, "256")
        
        self.det_w_edit = add_field("Detector width (pix.)", 1, 0, "512")
        add_field("Detector element size (pix.)", 1, 1, "1")
        self.noise_edit = add_field("Noise", 1, 2, "0.001")
        add_field("Scan time (s)", 1, 3, "10")
        
        add_field("Regularization parameter", 2, 0, "0.1")
        add_field("Iterations", 2, 1, "10")
        add_field("SOD", 2, 2, "500")
        add_field("ODD", 2, 3, "500")
        add_field("Gradient descent step size", 2, 4, "0.01")
        
        self.max_angle_edit.textChanged.connect(self.update_max_angle)
        self.det_w_edit.textChanged.connect(self.update_detector_width)
        self.noise_edit.textChanged.connect(self.update_noise)
        
        btn_layout = QHBoxLayout()
        self.btn_preview = QPushButton("Run scan preview")
        self.btn_preview.clicked.connect(self.run_preview)
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.on_ok)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_preview)
        btn_layout.addWidget(self.btn_ok)
        param_grid.addLayout(btn_layout, 4, 3, 1, 2)
        
        left_layout.addWidget(param_group, 2)
        
        # Scan preview Panel
        scan_group = QGroupBox("Scan preview")
        scan_layout = QVBoxLayout()
        self.fig_preview, self.ax_preview = plt.subplots(facecolor='#1E1E2E')
        self.canvas_preview = FigureCanvas(self.fig_preview)
        scan_layout.addWidget(self.canvas_preview)
        scan_group.setLayout(scan_layout)
        left_layout.addWidget(scan_group, 5)
        
        # Right Parameter Visualization Panels
        self.fig_vis, (self.ax_angles, self.ax_det, self.ax_noise) = plt.subplots(3, 1, facecolor='#1E1E2E')
        self.ax_angles.set_title('Scan angles', color='white')
        self.ax_det.set_title('Detector', color='white')
        self.ax_noise.set_title('Noise', color='white')
        for ax in (self.ax_angles, self.ax_det, self.ax_noise):
            ax.set_facecolor('#282A36')
            ax.tick_params(colors='white')
        self.canvas_vis = FigureCanvas(self.fig_vis)
        
        vis_group = QGroupBox("Parameter visualization")
        vis_layout = QVBoxLayout()
        vis_layout.addWidget(self.canvas_vis)
        vis_group.setLayout(vis_layout)
        right_layout.addWidget(vis_group)
        
        self.apply_visibility()
        self.init_plots()

    def try_float(self, field_name):
        try:
            return float(self.fields[field_name].text())
        except ValueError:
            return 0.0

    def apply_visibility(self):
        # Hide parameters based on current_reco and current_beam_geom
        pass

    def init_plots(self):
        # Implement initial plot
        # For preview:
        from skimage.data import shepp_logan_phantom
        from skimage.transform import resize
        self.phantom_img = resize(shepp_logan_phantom(), (256, 256))
        
        self.ax_preview.clear()
        self.ax_preview.imshow(self.phantom_img, cmap='gray', extent=[-70, 70, 70, -70])
        self.ax_preview.axis('off')
        self.canvas_preview.draw()
        
        self.update_max_angle()
        self.update_detector_width()
        self.update_noise()

    def update_max_angle(self):
        min_ang = self.try_float("Min angle")
        max_ang = self.try_float("Max angle")
        step = self.try_float("Step angle")
        if step <= 0: step = 5
        
        t = np.arange(min_ang, max_ang + step, step)
        sine = np.sin(np.deg2rad(t))
        cosine = np.cos(np.deg2rad(t))
        
        self.ax_angles.clear()
        self.ax_angles.plot(sine, cosine, 'w.')
        self.ax_angles.set_xlim([-1.5, 1.5])
        self.ax_angles.set_ylim([1.5, -1.5]) # reversed
        self.ax_angles.set_title('Scan angles', color='white')
        self.canvas_vis.draw()

    def update_detector_width(self):
        det_width = self.try_float("Detector width (pix.)")
        img_vol = self.try_float("Image volume (pix.)")
        if img_vol <= 0: img_vol = 256
        
        self.ax_det.clear()
        self.ax_det.imshow(self.phantom_img, cmap='gray', extent=[-70, 70, 70, -70])
        
        F = 150 / img_vol
        rect = Rectangle((70, -F*det_width/2), 20, F*det_width, edgecolor='w', facecolor='none')
        self.ax_det.add_patch(rect)
        
        poly = Polygon([[-70, F*det_width/2], [-70, -F*det_width/2], [70, -F*det_width/2], [70, F*det_width/2]], 
                       edgecolor='y', facecolor='w', alpha=0.5)
        self.ax_det.add_patch(poly)
        
        self.ax_det.set_xlim([-150, 150])
        self.ax_det.set_ylim([150, -150])
        self.ax_det.set_title('Detector', color='white')
        self.canvas_vis.draw()

    def update_noise(self):
        noise_val = self.try_float("Noise")
        img = self.phantom_img + np.random.normal(0, noise_val, self.phantom_img.shape)
        self.ax_noise.clear()
        self.ax_noise.imshow(img, cmap='gray', extent=[-150, 150, 150, -150])
        self.ax_noise.set_title('Noise', color='white')
        self.canvas_vis.draw()

    def run_preview(self):
        # Animate the scan preview
        pass

    def on_ok(self):
        p = {k: self.try_float(k) for k in self.fields.keys()}
        
        if self.parent_app and hasattr(self.parent_app, 'chosen_imaging_parameters'):
            self.parent_app.chosen_imaging_parameters(
                p["Max angle"], p["Min angle"], p["Step angle"],
                p["Image volume (pix.)"], p["Detector width (pix.)"],
                p["Detector element size (pix.)"], p["SOD"],
                p["ODD"], p["Noise"], p["Iterations"],
                p["Gradient descent step size"], p["Scan time (s)"],
                p["Regularization parameter"], self.current_beam_geom, self.current_reco
            )
        self.accept()
