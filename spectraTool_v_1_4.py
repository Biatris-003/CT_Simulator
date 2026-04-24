import sys
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QLineEdit, QPushButton, QGroupBox, QSlider, QSpinBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class SpectraToolDialog(QDialog):
    def __init__(self, parent_app=None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.setWindowTitle("Generate X-ray Spectra")
        self.setMinimumSize(800, 600)
        self.resize(800, 600)
        
        import style
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        # Spectra Axes
        self.fig_spectra, self.ax_spectra = plt.subplots(facecolor='#1E1E2E')
        self.ax_spectra.set_facecolor('#282A36')
        self.ax_spectra.tick_params(colors='white')
        self.ax_spectra.set_title('Spectrum', color='white')
        self.ax_spectra.set_xlabel('Photon energy (keV)', color='white')
        self.ax_spectra.set_ylabel('Fluence/mm^2/mAs', color='white')
        
        self.canvas_spectra = FigureCanvas(self.fig_spectra)
        left_layout.addWidget(self.canvas_spectra)

        # Exposure & Voltage Panel
        ev_group = QGroupBox("Exposure & Voltage")
        ev_layout = QGridLayout()
        
        ev_layout.addWidget(QLabel("kV"), 0, 0)
        self.kv_slider = QSlider(Qt.Orientation.Horizontal)
        self.kv_slider.setRange(0, 140)
        self.kv_slider.setValue(100)
        self.kv_label = QLabel("100")
        self.kv_slider.valueChanged.connect(lambda v: self.kv_label.setText(str(v)))
        ev_layout.addWidget(self.kv_slider, 1, 0)
        ev_layout.addWidget(self.kv_label, 1, 1)

        ev_layout.addWidget(QLabel("mA"), 2, 0)
        self.ma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ma_slider.setRange(0, 5)
        self.ma_slider.setValue(2)
        self.ma_label = QLabel("2")
        self.ma_slider.valueChanged.connect(lambda v: self.ma_label.setText(str(v)))
        ev_layout.addWidget(self.ma_slider, 3, 0)
        ev_layout.addWidget(self.ma_label, 3, 1)
        ev_group.setLayout(ev_layout)
        right_layout.addWidget(ev_group)

        # Filtration Panel
        filt_group = QGroupBox("Filtration")
        filt_layout = QGridLayout()
        filt_layout.addWidget(QLabel("Aluminium (mm)"), 0, 0)
        self.edit_al = QLineEdit("0.0")
        filt_layout.addWidget(self.edit_al, 0, 1)
        
        filt_layout.addWidget(QLabel("Copper (mm)"), 1, 0)
        self.edit_cu = QLineEdit("0.0")
        filt_layout.addWidget(self.edit_cu, 1, 1)
        filt_group.setLayout(filt_layout)
        right_layout.addWidget(filt_group)

        # Operate Panel
        op_group = QGroupBox("Operate")
        op_layout = QVBoxLayout()
        self.btn_gen = QPushButton("Generate spectrum")
        self.btn_gen.clicked.connect(self.on_generate)
        self.btn_save = QPushButton("Save spectrum")
        self.btn_save.clicked.connect(self.on_save)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        op_layout.addWidget(self.btn_gen)
        op_layout.addWidget(self.btn_save)
        op_layout.addWidget(self.btn_close)
        op_group.setLayout(op_layout)
        right_layout.addWidget(op_group)

        self.q = None
        self.kVp = 0
        self.mA = 0
        self.Al = 0.0
        self.Cu = 0.0

    def on_generate(self):
        self.kVp = self.kv_slider.value()
        self.mA = self.ma_slider.value()
        try:
            self.Al = float(self.edit_al.text())
            self.Cu = float(self.edit_cu.text())
        except ValueError:
            pass

        # Realistic spectrum generation (Tungsten target)
        energies = np.arange(2.0, self.kVp + 1.0, 1.0)
        
        # Improved mass attenuation coeff approx (cm^2/g) w/ Cu K-edge at ~8.98 keV
        mu_al = 40.0 * (energies / 10.0)**(-3.2) + 0.17
        mu_cu = np.where(energies < 8.98,
                         28000.0 / (energies**3) + 0.15,
                         220.0 * (energies / 10.0)**(-3.0) + 0.15)
        
        # Photon fluence spectrum (Kramer's rule: N(E) proportional to (kVp - E)/E)
        intensities = (self.kVp - energies) / energies
        intensities = np.clip(intensities, 0, None)
        
        # Add characteristic Tungsten peaks if kVp > 69.5
        if self.kVp > 69.5:
            scale = ((self.kVp - 69.5) / 30.0) ** 1.6
            # Add Gaussians for K-alpha (~59 keV) and K-beta (~67.5 keV)
            k_alpha = 1.5 * scale * np.exp(-0.5 * ((energies - 59.0)/1.0)**2)
            k_beta = 0.4 * scale * np.exp(-0.5 * ((energies - 67.5)/1.0)**2)
            intensities += k_alpha + k_beta

        # Intrinsic filtration to form the base spectrum (approx 1.5mm Al)
        att_intrinsic = mu_al * 2.7 * 0.15
        base_spectrum = intensities * np.exp(-att_intrinsic)
        
        # Prevent base spectrum from underflowing to identically 0 so min_e matches MATLAB when added filters are 0
        base_spectrum[base_spectrum == 0.0] = 1e-300
        
        # Added filtration 
        att_added = (mu_al * 2.7 * (self.Al / 10.0)) + (mu_cu * 8.96 * (self.Cu / 10.0))
        final_intensities = base_spectrum * np.exp(-att_added)
        
        # Scale to realistic values (~5e4 for 100kV, 1mA)
        intensities = final_intensities * self.mA * 6.5e4

        self.q = intensities
        self.ax_spectra.clear()
        self.ax_spectra.plot(energies, self.q, color='#0072BD', linewidth=2.5)
        self.ax_spectra.set_title('Spectrum', color='white')
        self.ax_spectra.set_xlabel('Photon energy (keV)', color='white')
        self.ax_spectra.set_ylabel('Fluence/mm^2/mAs', color='white')
        
        limY = float(np.max(intensities)) if len(intensities) > 0 else 1.0
        if limY <= 0.0: limY = 1.0
        
        # Set limits appropriately
        self.ax_spectra.set_xlim([0, max(150, int(self.kVp))])
        limY_plot = limY * 1.1
        self.ax_spectra.set_ylim([0, limY_plot])
        
        # MATLAB eliminates exactly zero values from vector
        nonzero = energies[intensities > 0.0]
        min_e = nonzero.min() if len(nonzero) > 0 else 2.0
        max_e = self.kVp
        
        self.ax_spectra.text(energies.max() * 0.4, limY_plot * 0.9, f"Min energy: {int(min_e)} keV", color='white')
        self.ax_spectra.text(energies.max() * 0.4, limY_plot * 0.8, f"Max energy: {int(max_e)} keV", color='white')
        self.canvas_spectra.draw()

        if self.parent_app and hasattr(self.parent_app, 'chosen_spectrum'):
            # energies and q are passed. MATLAB passes q as array
            self.parent_app.chosen_spectrum(self.q, energies, self.kVp, self.mA, self.Cu, self.Al)
            
        self.btn_gen.setEnabled(False)

    def on_save(self):
        # Implement saving if needed
        pass

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = SpectraToolDialog()
    dlg.show()
    sys.exit(app.exec())
