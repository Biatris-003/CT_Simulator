import datetime
import numpy as np

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from models.phantom_material_map import (
    get_mu_for_material,
    build_three_material_phantom,
    build_three_material_mu_map,
)
from models.spectra_physics import generate_spectrum_physics, generate_physics_sinogram
from models.reconstruction import SparseReconstruction
from views import style

class SpectrumWorkspaceDialog(QDialog):
    def __init__(self, parent_app=None, phantom_material_map=None):
        super().__init__(parent_app)

        self.parent_app = parent_app
        self.phantom_material_map = (
            phantom_material_map
            if phantom_material_map is not None
            else build_three_material_phantom(size=1024)
        )

        self.setWindowTitle("Spectrum Workspace")
        self.resize(1200, 750)

        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)

        self.q = None
        self.energies = None
        self._cached_spectrum_key = None
        self._cached_mu_map_key = None
        self._cached_mu_map = None
        self._cached_total_i0 = None
        self._cached_noisy_sino = None
        self._cached_noisy_angles = None
        self._cached_sparse_fbp = None
        self.kVp = 100
        self.mA = 2
        self.step_angle = int(getattr(parent_app, "step_angle", 1))
        self.Cu = 0.0
        self.Al = 0.0

        main_layout = QVBoxLayout(self)

        # ================= TOP ROW =================
        top_row = QHBoxLayout()
        main_layout.addLayout(top_row)

        # Spectrum
        spectrum_group = QGroupBox("X-Ray Spectrum")
        spectrum_layout = QVBoxLayout(spectrum_group)
        self.fig_spectrum, self.ax_spectrum = plt.subplots(facecolor="#1E1E2E")
        self.canvas_spectrum = FigureCanvas(self.fig_spectrum)
        spectrum_layout.addWidget(self.canvas_spectrum)
        top_row.addWidget(spectrum_group, 1)

        # Mu Map
        mu_group = QGroupBox("3-Material Mu Map")
        mu_layout = QVBoxLayout(mu_group)
        self.fig_mu, self.ax_mu = plt.subplots(facecolor="#1E1E2E")
        self.canvas_mu = FigureCanvas(self.fig_mu)
        mu_layout.addWidget(self.canvas_mu)
        top_row.addWidget(mu_group, 1)

        # Tube Settings (KEEP ONLY ONE)
        controls_group = QGroupBox("Tube Settings")
        controls_layout = QVBoxLayout(controls_group)

        # kV slider
        kv_row = QHBoxLayout()
        kv_row.addWidget(QLabel("kV"))
        self.kv_slider = QSlider(Qt.Orientation.Horizontal)
        self.kv_slider.setRange(40, 140)
        self.kv_slider.setValue(self.kVp)
        self.kv_value_label = QLabel(str(self.kVp))
        self.kv_slider.sliderReleased.connect(self._refresh_workspace)

        kv_row.addWidget(self.kv_slider)
        kv_row.addWidget(self.kv_value_label)
        controls_layout.addLayout(kv_row)

        # mA slider
        ma_row = QHBoxLayout()
        ma_row.addWidget(QLabel("mA"))
        self.ma_slider = QSlider(Qt.Orientation.Horizontal)
        self.ma_slider.setRange(1, 10)
        self.ma_slider.setValue(self.mA)
        self.ma_value_label = QLabel(str(self.mA))
        self.ma_slider.setSingleStep(1)   # 👈 step = 1
        self.ma_slider.setPageStep(1)     # 👈 optional (for keyboard/page keys)
        self.ma_slider.sliderReleased.connect(self._refresh_workspace)

        ma_row.addWidget(self.ma_slider)
        ma_row.addWidget(self.ma_value_label)
        controls_layout.addLayout(ma_row)

        # Step-angle slider (1..10 deg), updates on release
        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step Angle"))
        self.step_slider = QSlider(Qt.Orientation.Horizontal)
        self.step_slider.setRange(1, 10)
        self.step_slider.setValue(self.step_angle)
        self.step_slider.setSingleStep(1)
        self.step_slider.setPageStep(1)
        self.step_value_label = QLabel(str(self.step_angle))
        self.step_slider.valueChanged.connect(
            lambda v: self.step_value_label.setText(str(v))
        )
        self.step_slider.sliderReleased.connect(self._apply_step_angle_from_dialog)
        step_row.addWidget(self.step_slider)
        step_row.addWidget(self.step_value_label)
        controls_layout.addLayout(step_row)

        top_row.addWidget(controls_group, 1)

        # ================= BOTTOM ROW =================
        bottom_row = QHBoxLayout()
        main_layout.addLayout(bottom_row)

        # Noisy Sinogram
        noisy_group = QGroupBox("Noisy Sinogram")
        noisy_layout = QVBoxLayout(noisy_group)
        self.fig_noisy_sino, self.ax_noisy_sino = plt.subplots(facecolor="#1E1E2E")
        self.canvas_noisy_sino = FigureCanvas(self.fig_noisy_sino)
        noisy_layout.addWidget(self.canvas_noisy_sino)
        bottom_row.addWidget(noisy_group, 1)

        # Sparse FBP Reconstruction
        sparse_fbp_group = QGroupBox("Sparse FBP Reconstruction")
        sparse_fbp_layout = QVBoxLayout(sparse_fbp_group)
        self.fig_sparse_fbp, self.ax_sparse_fbp = plt.subplots(facecolor="#1E1E2E")
        self.canvas_sparse_fbp = FigureCanvas(self.fig_sparse_fbp)
        sparse_fbp_layout.addWidget(self.canvas_sparse_fbp)
        bottom_row.addWidget(sparse_fbp_group, 1)

        # ================= FOOTER =================
        footer = QHBoxLayout()
        footer.addStretch()
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        footer.addWidget(self.btn_close)

        main_layout.addLayout(footer)

        # initial render
        self._refresh_workspace()

    def _generate_spectrum_data(self, kVp, mA, Cu=0.0, Al=0.0):
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
            self.ax_spectrum.clear()
            self.ax_spectrum.set_facecolor("black")

            # --- تعديل موضع وحجم الرسمة يدوياً ---
            # [left, bottom, width, height] (بقيم تتراوح من 0 لـ 1)
            # رفعنا الـ bottom لـ 0.25 عشان الكلمة اللي تحت (X-label) تظهر بوضوح
            # وصغرنا الـ height لـ 0.65 عشان الرسمة تصغر شوية
            self.ax_spectrum.set_position([0.15, 0.25, 0.75, 0.65])

            # ثبات الشكل في المنتصف
            self.ax_spectrum.set_anchor('C')

            # الشكل العام
            self.ax_spectrum.tick_params(axis='both', colors="white", labelsize=8)
            self.ax_spectrum.set_title("X-Ray Spectrum", color="white", fontsize=10)
            self.ax_spectrum.set_xlabel("Photon energy (keV)", color="white", fontsize=9)
            self.ax_spectrum.set_ylabel("Fluence/mm²/mAs", color="white", fontsize=9)
            self.ax_spectrum.grid(True, color='gray', linestyle='--', alpha=0.3)

            if len(energies) > 0 and len(intensities) > 0:
                self.ax_spectrum.plot(energies, intensities, color="#0004FF", linewidth=1.5)

                import matplotlib.ticker as ticker

                # تنسيق الأرقام العلمية (Scientific Notation)
                formatter = ticker.ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-1, 1))
                self.ax_spectrum.yaxis.set_major_formatter(formatter)

                # تلوين الـ 10^x باللون الأبيض
                self.ax_spectrum.yaxis.get_offset_text().set_color("white")
                self.ax_spectrum.yaxis.get_offset_text().set_size(8)

                # ضبط الحدود
                lim_y = float(np.max(intensities)) if np.max(intensities) > 0 else 1.0
                self.ax_spectrum.set_xlim([0, max(150, int(self.kVp))])
                self.ax_spectrum.set_ylim([0, lim_y * 1.2])

                # النصوص التوضيحية للطاقة
                nonzero = energies[intensities > 0.0]
                min_e = nonzero.min() if len(nonzero) > 0 else 2.0
                max_e = self.kVp

                self.ax_spectrum.text(
                    0.95, 0.95, f"Min E: {int(min_e)} keV",
                    color="cyan", transform=self.ax_spectrum.transAxes,
                    ha='right', fontsize=8
                )
                self.ax_spectrum.text(
                    0.95, 0.88, f"Max E: {int(max_e)} keV",
                    color="cyan", transform=self.ax_spectrum.transAxes,
                    ha='right', fontsize=8
                )

            # تحديث الكانفاس بدون tight_layout
            self.canvas_spectrum.draw_idle()

    def _render_mu_map(self, kvp, mu_map=None):
        if mu_map is None:
            material_map = self.phantom_material_map
            mu_map = np.zeros_like(material_map, dtype=np.float32)
            mu_map[material_map == 0] = get_mu_for_material(0, kvp)
            mu_map[material_map == 1] = get_mu_for_material(1, kvp)
            mu_map[material_map == 2] = get_mu_for_material(2, kvp)

        self.ax_mu.clear()
        self.ax_mu.set_facecolor("black")

        # --- توسيط الرسمة في النص بالظبط ---
        # حددنا أبعاد متساوية للهوامش [left, bottom, width, height]
        self.ax_mu.set_position([0.1, 0.1, 0.8, 0.8])

        # تثبيت الصورة والنسبة (Aspect Ratio)
        self.ax_mu.set_aspect('equal')
        self.ax_mu.set_anchor('C')

        image = self.ax_mu.imshow(mu_map, cmap="hot", vmin=0, vmax=1.2)

        contrast_text = "High Contrast" if kvp < 100 else "Low Contrast"
        self.ax_mu.set_title(contrast_text, color="white", fontsize=10)
        self.ax_mu.axis("off")

        # إدارة الـ Colorbar لضمان عدم التكرار أو تدمير التوسيط
        if not hasattr(self, "mu_colorbar") or self.mu_colorbar is None:
            self.mu_colorbar = self.fig_mu.colorbar(
                image, ax=self.ax_mu, fraction=0.046, pad=0.04
            )
            self.mu_colorbar.set_label(r'$\mu (cm^{-1})$', color="white")
            self.mu_colorbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(self.mu_colorbar.ax.get_yticklabels(), color='white')
        else:
            self.mu_colorbar.update_normal(image)

        self.canvas_mu.draw_idle()

    def _render_sinograms(self, mu_map, total_i0, step_angle=1.0):
        # Compute sinograms
        _, noisy_sino, _, noisy_angles = generate_physics_sinogram(mu_map, total_i0, user_step_angle=step_angle)

        self._cached_noisy_sino = noisy_sino
        self._cached_noisy_angles = noisy_angles

        self.ax_noisy_sino.clear()
        self.ax_noisy_sino.set_facecolor("black")
        self.ax_noisy_sino.imshow(noisy_sino, cmap="gray", aspect="auto")
        self.ax_noisy_sino.set_title(
            f"Noisy Sinogram ({self.step_angle}°, {self.kVp} kVp, {self.mA} mA)",
            color="white",
            fontsize=9,
        )
        self.ax_noisy_sino.axis("off")
        self.fig_noisy_sino.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas_noisy_sino.draw_idle()

        self._cached_sparse_fbp = SparseReconstruction.fbp_reconstruction(
            noisy_sino,
            noisy_angles,
            filter_name="ramp",
        )

        self.ax_sparse_fbp.clear()
        self.ax_sparse_fbp.set_facecolor("black")
        self.ax_sparse_fbp.imshow(self._cached_sparse_fbp, cmap="gray")
        self.ax_sparse_fbp.set_title(
            f"Sparse FBP Reconstruction ({self.step_angle}°)",
            color="white",
            fontsize=9,
        )
        self.ax_sparse_fbp.axis("off")
        self.fig_sparse_fbp.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas_sparse_fbp.draw_idle()

    def _refresh_workspace(self, *args, notify_parent=True):
        self.kVp = self.kv_slider.value()
        self.mA = self.ma_slider.value()
        self.step_angle = self.step_slider.value()
        self.kv_value_label.setText(str(self.kVp))
        self.ma_value_label.setText(str(self.mA))
        self.step_value_label.setText(str(self.step_angle))

        spectrum_key = (self.kVp, self.mA, self.Cu, self.Al)
        if self._cached_spectrum_key != spectrum_key or self.energies is None:
            self.energies, self.q, self._cached_total_i0 = generate_spectrum_physics(
                self.kVp,
                self.mA,
                self.Cu,
                self.Al,
            )
            self._cached_spectrum_key = spectrum_key

        mu_key = (self.kVp, self.phantom_material_map.shape[0])
        if self._cached_mu_map_key != mu_key or self._cached_mu_map is None:
            _, self._cached_mu_map = build_three_material_mu_map(
                size=self.phantom_material_map.shape[0],
                kvp=self.kVp,
            )
            self._cached_mu_map_key = mu_key

        self._render_spectrum(self.energies, self.q)
        self._render_mu_map(self.kVp, self._cached_mu_map)

        # render sinograms using mu_map and total_i0
        self._render_sinograms(self._cached_mu_map, self._cached_total_i0, step_angle=self.step_angle)

        if notify_parent and self.parent_app and hasattr(self.parent_app, "sync_step_angle_from_dialog"):
            self.parent_app.sync_step_angle_from_dialog(self.step_angle)

        if self.parent_app and hasattr(self.parent_app, "preview_spectrum"):
            self.parent_app.preview_spectrum(self.q, self.energies, self.kVp, self.mA, self.Cu, self.Al)

        # timestamp = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")
        # print(f"{timestamp} Spectrum workspace updated: {self.kVp} kVp, {self.mA} mA")

    def sync_step_angle_from_main(self, step_angle):
        self.step_angle = int(step_angle)
        self.step_slider.blockSignals(True)
        self.step_slider.setValue(self.step_angle)
        self.step_slider.blockSignals(False)
        self.step_value_label.setText(str(self.step_angle))
        self._refresh_workspace(notify_parent=False)

    def _apply_step_angle_from_dialog(self):
        self.step_angle = self.step_slider.value()
        self.step_value_label.setText(str(self.step_angle))
        self._refresh_workspace()