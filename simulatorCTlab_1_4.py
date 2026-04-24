import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QPushButton, QLabel, 
                             QSlider, QListWidget, QCheckBox, QSpinBox, 
                             QGroupBox, QFrame)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import datetime

from BeamGeometry_v_1_4 import BeamGeometryDialog
from ImagingParameters_v_1_4 import ImagingParametersDialog
from ReconstructionAlgo_v_1_4 import ReconstructionAlgoDialog
from spectraTool_v_1_4 import SpectraToolDialog
from spectralParameters_v_1_4 import SpectralParametersDialog

class SimulatorCTLabApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CTlab Simulator v1.4")
        self.resize(1600, 900)
        import style
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        # Main Central Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Base parameters
        self.current_beam_geom = "Parallel"
        self.current_reco = "Filtered backprojection (FBP)"
        self.ma = 100

        # Left Column for Controls
        self.left_column = QVBoxLayout()
        self.main_layout.addLayout(self.left_column, 1)

        # Middle Column for Main Imaging / Visualizations
        self.middle_column = QVBoxLayout()
        self.main_layout.addLayout(self.middle_column, 2)

        # Right Column for Spectral / Histograms / Info
        self.right_column = QVBoxLayout()
        self.main_layout.addLayout(self.right_column, 1)

        self.init_ui()

    def init_ui(self):
        self.init_left_column()
        self.init_middle_column()
        self.init_right_column()

    def init_left_column(self):


        # Start Simulation
        start_sim_group = QGroupBox("START SIMULATION")
        start_sim_layout = QVBoxLayout()
        self.btn_load_phantom = QPushButton("Load Phantom")
        self.btn_load_phantom.clicked.connect(self.load_phantom)
        start_sim_layout.addWidget(self.btn_load_phantom)
        start_sim_group.setLayout(start_sim_layout)
        self.left_column.addWidget(start_sim_group)

        # View Axes (UIAxes6 equivalent for start/phantom)
        self.fig_start, self.ax_start = plt.subplots(facecolor='#1E1E2E')
        self.canvas_start = FigureCanvas(self.fig_start)
        self.left_column.addWidget(self.canvas_start)

        # Set Parameters
        param_group = QGroupBox("Set Parameters")
        param_layout = QVBoxLayout()
        self.btn_beam_geom = QPushButton("Beam Geometry")
        self.btn_beam_geom.clicked.connect(self.show_beam_geometry_dialog)
        self.btn_img_params = QPushButton("Imaging Parameters")
        self.btn_img_params.clicked.connect(self.show_imaging_parameters_dialog)
        param_layout.addWidget(self.btn_beam_geom)
        param_layout.addWidget(self.btn_img_params)
        param_group.setLayout(param_layout)
        self.left_column.addWidget(param_group)

        # Info listbox
        self.param_info_listbox = QListWidget()
        self.left_column.addWidget(QLabel("Parameters Info"))
        self.left_column.addWidget(self.param_info_listbox)

    def init_middle_column(self):
        # Upper Axes: Polychromatic Reconstruction
        self.fig_poly, self.ax_poly = plt.subplots(facecolor='#1E1E2E')
        self.canvas_poly = FigureCanvas(self.fig_poly)
        lbl_poly = QLabel("Polychromatic Reconstruction")
        lbl_poly.setObjectName("HeaderLabel")
        self.middle_column.addWidget(lbl_poly)
        self.middle_column.addWidget(self.canvas_poly, 1)

        # Lower Axes: Spectral Reconstruction
        self.fig_spectral, self.ax_spectral = plt.subplots(facecolor='#1E1E2E')
        self.canvas_spectral = FigureCanvas(self.fig_spectral)
        lbl_spec = QLabel("Spectral Reconstruction")
        lbl_spec.setObjectName("HeaderLabel")
        self.middle_column.addWidget(lbl_spec)
        self.middle_column.addWidget(self.canvas_spectral, 1)

        # X-Ray spectrum panel
        self.fig_xray, self.ax_xray = plt.subplots(facecolor='#1E1E2E')
        self.canvas_xray = FigureCanvas(self.fig_xray)
        
        xray_group = QGroupBox("X-Ray Spectrum")
        xray_layout = QVBoxLayout()
        x_btn_layout = QHBoxLayout()
        self.btn_gen_spectrum = QPushButton("Generate Spectrum")
        self.btn_gen_spectrum.clicked.connect(self.show_spectra_tool)
        self.btn_load_spectrum = QPushButton("Load Spectrum")
        x_btn_layout.addWidget(self.btn_gen_spectrum)
        x_btn_layout.addWidget(self.btn_load_spectrum)
        xray_layout.addLayout(x_btn_layout)
        xray_layout.addWidget(self.canvas_xray, 1)
        xray_group.setLayout(xray_layout)
        
        self.middle_column.addWidget(xray_group, 1)

    def init_right_column(self):
        # Buttons...
        # Spectral Reconstruction
        spec_recon_group = QGroupBox("Spectral Reconstruction")
        spec_recon_layout = QVBoxLayout()
        self.btn_spec_reconstruct = QPushButton("Spectral Reconstruction")
        self.btn_spec_params = QPushButton("Spectral Parameters")
        self.btn_spec_params.clicked.connect(self.show_spectral_params_dialog)
        spec_recon_layout.addWidget(self.btn_spec_reconstruct)
        spec_recon_layout.addWidget(self.btn_spec_params)
        spec_recon_group.setLayout(spec_recon_layout)
        self.right_column.addWidget(spec_recon_group)

        # Many Buttons...
        self.btn_scan = QPushButton("Scan")
        self.btn_reconstruct_algo = QPushButton("Reconstruction Algorithm")
        self.btn_reconstruct_algo.clicked.connect(self.show_reconstruction_algo_dialog)
        self.btn_reconstruct = QPushButton("Reconstruction")
        self.right_column.addWidget(self.btn_scan)
        self.right_column.addWidget(self.btn_reconstruct_algo)
        self.right_column.addWidget(self.btn_reconstruct)

        # Histogram 
        self.fig_hist, self.ax_hist = plt.subplots(facecolor='#1E1E2E')
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.right_column.addWidget(QLabel("HU-Value Histogram"))
        self.right_column.addWidget(self.canvas_hist)

        # Visualization Info Listbox
        self.vis_info_listbox = QListWidget()
        self.right_column.addWidget(QLabel("Visualization Info"))
        self.right_column.addWidget(self.vis_info_listbox)

    def load_phantom(self):
        from skimage.data import shepp_logan_phantom
        from skimage.transform import resize

        self.fantom = shepp_logan_phantom()
        self.fantom = resize(self.fantom, (1024, 1024))
        
        # Clear axes
        self.ax_start.clear()
        self.ax_hist.clear()
        
        # Plot phantom
        self.ax_start.imshow(self.fantom, cmap='gray')
        self.ax_start.axis('off')
        self.canvas_start.draw()

        # Plot histogram
        self.ax_hist.hist(self.fantom.ravel(), bins=256, color=[0.50,0.62,0.67], alpha=0.4)
        self.canvas_hist.draw()
        
        # Add tracking info
        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.vis_info_listbox.addItem(f"{t} <Shepp-Logan phantom loaded>")
        
        # Enable other buttons
        # TODO: Add button states like self.btn_load_spectrum.setEnabled(True)

    def show_beam_geometry_dialog(self):
        self.ready_lamp.setStyleSheet("background-color: red; color: white;")
        dlg = BeamGeometryDialog(self)
        dlg.exec()
        self.ready_lamp.setStyleSheet("background-color: green; color: white;")

    def chosen_beamgeom(self, beamgeometry):
        self.current_beam_geom = beamgeometry
        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.param_info_listbox.addItem(f"{t} <Reconstruction geometry: {self.current_beam_geom}>")

    def cancel_beamgeom_selection(self):
        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.param_info_listbox.addItem(f"{t} <Reconstruction geometry setting canceled...>")

    def show_reconstruction_algo_dialog(self):
        self.ready_lamp.setStyleSheet("background-color: red; color: white;")
        dlg = ReconstructionAlgoDialog(self)
        dlg.exec()
        self.ready_lamp.setStyleSheet("background-color: green; color: white;")

    def chosen_reconstruction(self, reco):
        self.current_reco = reco
        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.param_info_listbox.addItem(f"{t} <Reconstruction algorithm: {self.current_reco}>")

    def cancel_reconstruction_selection(self):
        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.param_info_listbox.addItem(f"{t} <Reconstruction algorithm setting canceled...>")

    def show_imaging_parameters_dialog(self):
        self.ready_lamp.setStyleSheet("background-color: red; color: white;")
        dlg = ImagingParametersDialog(self, self.current_beam_geom, self.current_reco, self.ma)
        dlg.exec()
        self.ready_lamp.setStyleSheet("background-color: green; color: white;")

    def chosen_imaging_parameters(self, max_ang, min_ang, step_ang, img_vol, det_w, det_el,
                                  sod, odd, noise, iters, grad, scantime, reg, geom, reco):
        self.minAngle = min_ang
        self.stepAngle = step_ang
        self.maxAngle = max_ang
        self.imageVolume = img_vol
        self.detectorWidth = det_w
        self.detectorElementsize = det_el
        self.imageNoise = noise
        self.iterations = iters
        self.alpha = grad
        self.lambda_reg = reg
        self.time = scantime
        self.SOD = sod
        self.ODD = odd
        self.current_beam_geom = geom
        self.current_reco = reco

        # Enable Scan Button (mock)
        self.btn_scan.setEnabled(True)

        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        
        # Add basic logs
        self.param_info_listbox.addItem(f"{t} <Min angle:{self.minAngle}>")
        self.param_info_listbox.addItem(f"{t} <Step:{self.stepAngle}>")
        self.param_info_listbox.addItem(f"{t} <Max angle:{self.maxAngle}>")
        self.param_info_listbox.addItem(f"{t} <Image volume:{self.imageVolume}>")
        self.param_info_listbox.addItem(f"{t} <Detector width:{self.detectorWidth}>")
        self.param_info_listbox.addItem(f"{t} <Detector element size:{self.detectorElementsize}>")
        self.param_info_listbox.addItem(f"{t} <Image noise:{self.imageNoise}>")
        self.param_info_listbox.addItem(f"{t} <Scan time:{self.time}>")
        
        if self.current_beam_geom == "Fanflat":
            self.param_info_listbox.addItem(f"{t} <SOD:{self.SOD}>")
            self.param_info_listbox.addItem(f"{t} <ODD:{self.ODD}>")
        
        if self.current_reco == "Least squares":
            self.param_info_listbox.addItem(f"{t} <Iterations:{self.iterations}>")
            
        elif self.current_reco == "Tikhonov Regularization":
            self.param_info_listbox.addItem(f"{t} <Regularization parameter:{self.lambda_reg}>")

    def show_spectra_tool(self):
        dlg = SpectraToolDialog(self)
        dlg.exec()

    def chosen_spectrum(self, q, energies, kvp, ma, cu, al):
        self.q = q
        self.E0 = energies
        self.kVp = kvp
        self.mA = ma
        self.Cu = cu
        self.Al = al
        
        # Plot in main UI
        self.ax_xray.clear()
        self.ax_xray.set_facecolor('black')
        self.ax_xray.tick_params(colors='white')
        self.ax_xray.plot(energies, q, 'w-')
        self.ax_xray.set_title("X-RAY SPECTRUM", color="white")
        self.canvas_xray.draw()

        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.param_info_listbox.addItem(f"{t} <Tube current: {self.mA}mA>")
        self.param_info_listbox.addItem(f"{t} <Tube voltage: {self.kVp}kVp>")
        self.param_info_listbox.addItem(f"{t} <Al filter thickness: {self.Al}mm>")
        self.param_info_listbox.addItem(f"{t} <Cu filter thickness: {self.Cu}mm>")
        
        self.vis_info_listbox.addItem(f"{t} <Spectrum Max Energy: {energies.max()} keV>")
        self.vis_info_listbox.addItem(f"{t} <Spectrum Min Energy: {energies.min()} keV>")

    def show_spectral_params_dialog(self):
        if hasattr(self, 'q') and hasattr(self, 'E0'):
            dlg = SpectralParametersDialog(self, self.q, self.E0.min(), self.E0.max())
            dlg.exec()
        else:
            print("Generate spectrum first!")

    def chosen_spectral_parameters(self, bins, bin_size, energy_windows):
        self.bin = bins
        self.binSize = bin_size
        self.energyWindows = energy_windows
        
        t = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        self.param_info_listbox.addItem(f"{t} <Bin size: {self.binSize}keV>")
        self.param_info_listbox.addItem(f"{t} <Number of bins: {self.bin}>")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SimulatorCTLabApp()
    window.show()
    sys.exit(app.exec())
