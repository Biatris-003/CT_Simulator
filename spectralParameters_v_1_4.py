import sys
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, 
                             QLabel, QLineEdit, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class SpectralParametersDialog(QDialog):
    def __init__(self, parent_app, q, minE, maxE):
        super().__init__(parent_app)  # type: ignore
        self.parent_app = parent_app
        self.q = q
        self.minE = minE
        self.maxE = maxE

        self.setWindowTitle("Spectral imaging parameters")
        self.setMinimumSize(700, 450)
        self.resize(700, 450)
        
        import style
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        main_layout = QHBoxLayout(self)

        # Plot Panel
        plot_group = QGroupBox("Binned Spectrum")
        plot_layout = QVBoxLayout()
        self.fig_spectra, self.ax_spectra = plt.subplots(facecolor='#1E1E2E')
        self.ax_spectra.set_facecolor('#282A36')
        self.ax_spectra.tick_params(colors='white')
        self.ax_spectra.set_title('Spectrum', color='white')
        self.ax_spectra.set_xlabel('Photon energy (keV)', color='white')
        self.ax_spectra.set_ylabel('Fluence/mm^2/mAs', color='white')
        self.canvas_spectra = FigureCanvas(self.fig_spectra)
        plot_layout.addWidget(self.canvas_spectra)
        plot_group.setLayout(plot_layout)
        main_layout.addWidget(plot_group, 2)

        # Right control panel
        right_layout = QVBoxLayout()
        
        bin_group = QGroupBox("Bin settings")
        bin_layout = QGridLayout()
        bin_layout.addWidget(QLabel("Number of Bins"), 0, 0)
        self.num_bins_edit = QLineEdit("10")
        bin_layout.addWidget(self.num_bins_edit, 0, 1)
        
        bin_layout.addWidget(QLabel("Bin size (keV)"), 1, 0)
        self.bin_size_edit = QLineEdit("10")
        bin_layout.addWidget(self.bin_size_edit, 1, 1)
        bin_group.setLayout(bin_layout)
        right_layout.addWidget(bin_group)

        op_group = QGroupBox("Operate")
        op_layout = QHBoxLayout()
        self.btn_ok = QPushButton("Ok")
        self.btn_close = QPushButton("Close")
        self.btn_ok.clicked.connect(self.on_ok)
        self.btn_close.clicked.connect(self.close)
        op_layout.addWidget(self.btn_ok)
        op_layout.addWidget(self.btn_close)
        op_group.setLayout(op_layout)
        right_layout.addWidget(op_group)
        right_layout.addStretch()

        main_layout.addLayout(right_layout, 1)

    def on_ok(self):
        try:
            num_bins = int(self.num_bins_edit.text())
            bin_size = float(self.bin_size_edit.text())
        except ValueError:
            return

        res = (self.maxE - self.minE) % bin_size
        k = np.arange(self.minE, self.maxE + bin_size, bin_size)
        if res != 0:
            k = np.append(k, k.max() + res)

        self.ax_spectra.clear()
        
        # We need an energies array to correctly plot q. We simulate it as minE to maxE
        if getattr(self.parent_app, 'E0', None) is not None:
            energies = self.parent_app.E0
            q_val = self.parent_app.q
        else:
            energies = np.arange(self.minE, self.maxE + 1)
            q_val = self.q if self.q is not None else np.zeros_like(energies)

        self.ax_spectra.plot(energies, q_val, color='white', linewidth=2)
        
        q_max = float(np.max(q_val)) if len(q_val) > 0 else 0
        limY = q_max if q_max > 0 else 1.0
        self.ax_spectra.set_ylim([0, limY])

        for line_x in k:
            self.ax_spectra.axvline(x=line_x, color='white', linestyle=':')
            self.ax_spectra.axvspan(line_x, line_x + bin_size, facecolor='#809EA8', alpha=0.4)

        self.ax_spectra.set_title('Spectrum', color='white')
        self.ax_spectra.set_xlabel('Photon energy (keV)', color='white')
        self.ax_spectra.set_ylabel('Fluence/mm^2/mAs', color='white')
        self.canvas_spectra.draw()

        if self.parent_app and hasattr(self.parent_app, 'chosen_spectral_parameters'):
            self.parent_app.chosen_spectral_parameters(num_bins, bin_size, k)

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = SpectralParametersDialog(None, None, 10, 100)
    dlg.show()
    sys.exit(app.exec())
