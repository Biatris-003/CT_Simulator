import sys
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QPushButton, QGroupBox, QSlider)
from PyQt5.QtCore import Qt

# =========================
# MATPLOTLIB QT BACKEND (Object-Oriented Only - NO PLT)
# =========================
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from views import style


class SpectraToolDialog(QDialog):
    def __init__(self, parent_app=None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.setWindowTitle("Generate X-ray Spectra")
        self.setMinimumSize(800, 600)
        self.resize(800, 600)
        
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)

        # Spectra Axes (Upgraded to Object-Oriented)
        self.fig_spectra = Figure(facecolor='#1E1E2E')
        self.ax_spectra = self.fig_spectra.add_subplot(111)
        
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
        self.kv_slider.sliderReleased.connect(self.update_preview)
        ev_layout.addWidget(self.kv_slider, 1, 0)
        ev_layout.addWidget(self.kv_label, 1, 1)

        ev_layout.addWidget(QLabel("mA"), 2, 0)
        self.ma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ma_slider.setRange(0, 5)
        self.ma_slider.setValue(2)
        self.ma_label = QLabel("2")
        self.ma_slider.valueChanged.connect(lambda v: self.ma_label.setText(str(v)))
        self.ma_slider.sliderReleased.connect(self.update_preview)
        ev_layout.addWidget(self.ma_slider, 3, 0)
        ev_layout.addWidget(self.ma_label, 3, 1)
        ev_group.setLayout(ev_layout)
        right_layout.addWidget(ev_group)

        # Operate Panel
        op_group = QGroupBox("Operate")
        op_layout = QVBoxLayout()
        self.btn_gen = QPushButton("Generate spectrum")
        self.btn_gen.clicked.connect(self.on_generate)
        op_layout.addWidget(self.btn_gen)
        op_group.setLayout(op_layout)
        right_layout.addWidget(op_group)

        self.q = None
        self.kVp = 0
        self.mA = 0
        self.Cu = 0.0
        self.Al = 0.0

        self.update_preview()

    def _generate_spectrum_data(self, kVp, mA, Cu=0.0, Al=0.0):
        """Generate spectrum arrays for the current acquisition settings."""
        kVp = int(kVp)
        mA = int(mA)

        energies = np.arange(2.0, kVp + 1.0, 1.0)
        if len(energies) == 0:
            return energies, np.array([])

        mu_al = 40.0 * (energies / 10.0) ** (-3.2) + 0.17
        mu_cu = np.where(
            energies < 8.98,
            28000.0 / (energies ** 3) + 0.15,
            220.0 * (energies / 10.0) ** (-3.0) + 0.15,
        )

        intensities = (kVp - energies) / energies
        intensities = np.clip(intensities, 0, None)

        if kVp > 69.5:
            scale = ((kVp - 69.5) / 30.0) ** 1.6
            k_alpha = 1.5 * scale * np.exp(-0.5 * ((energies - 59.0) / 1.0) ** 2)
            k_beta = 0.4 * scale * np.exp(-0.5 * ((energies - 67.5) / 1.0) ** 2)
            intensities += k_alpha + k_beta

        att_intrinsic = mu_al * 2.7 * 0.15
        base_spectrum = intensities * np.exp(-att_intrinsic)
        base_spectrum[base_spectrum == 0.0] = 1e-300

        att_added = (mu_al * 2.7 * (Al / 10.0)) + (mu_cu * 8.96 * (Cu / 10.0))
        final_intensities = base_spectrum * np.exp(-att_added)

        final_intensities = final_intensities * mA * 6.5e4
        return energies, final_intensities

    def _render_spectrum(self, energies, intensities):
        self.ax_spectra.clear()
        self.ax_spectra.set_facecolor('#282A36')
        self.ax_spectra.tick_params(colors='white')
        self.ax_spectra.set_title('Spectrum', color='white')
        self.ax_spectra.set_xlabel('Photon energy (keV)', color='white')
        self.ax_spectra.set_ylabel('Fluence/mm^2/mAs', color='white')

        if len(energies) > 0 and len(intensities) > 0:
            self.ax_spectra.plot(energies, intensities, color='#0072BD', linewidth=2.5)
            limY = float(np.max(intensities)) if np.max(intensities) > 0 else 1.0
            self.ax_spectra.set_xlim([0, max(150, int(self.kVp))])
            self.ax_spectra.set_ylim([0, limY * 1.1])

            nonzero = energies[intensities > 0.0]
            min_e = nonzero.min() if len(nonzero) > 0 else 2.0
            max_e = self.kVp
            self.ax_spectra.text(energies.max() * 0.4, limY * 0.99, f"Min energy: {int(min_e)} keV", color='white')
            self.ax_spectra.text(energies.max() * 0.4, limY * 0.88, f"Max energy: {int(max_e)} keV", color='white')

        self.canvas_spectra.draw_idle()

    def update_preview(self, *args):
        """Live-update the spectrum plot and notify the main window."""
        self.kVp = self.kv_slider.value()
        self.mA = self.ma_slider.value()

        energies, intensities = self._generate_spectrum_data(self.kVp, self.mA, self.Cu, self.Al)
        self.energies = energies
        self.q = intensities
        self._render_spectrum(energies, intensities)

        if self.parent_app and hasattr(self.parent_app, 'preview_spectrum'):
            # Use deep copies to protect data if passed back to main app
            self.parent_app.preview_spectrum(np.copy(self.q), np.copy(energies), self.kVp, self.mA, self.Cu, self.Al)

    def on_generate(self):
        self.Al = 0.0
        self.Cu = 0.0
        self.update_preview()

        if self.parent_app and hasattr(self.parent_app, 'chosen_spectrum'):
            # Use deep copies to protect data if passed back to main app
            self.parent_app.chosen_spectrum(np.copy(self.q), np.copy(self.energies), self.kVp, self.mA, self.Cu, self.Al)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = SpectraToolDialog()
    dlg.show()
    sys.exit(app.exec())