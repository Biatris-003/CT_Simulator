"""
Sparse Reconstruction Dialog

UI for performing sparse CT reconstruction using FBP algorithm.
Demonstrates the limitations of FBP with sparse sampling.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QLabel, QSpinBox, QDoubleSpinBox, QPushButton, 
                             QGroupBox, QComboBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from models.reconstruction import SparseReconstruction, ComparisonReconstruction
from views import style


class SparseReconstructionDialog(QDialog):
    """Dialog for sparse reconstruction using FBP"""
    
    def __init__(self, parent_app=None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.setWindowTitle("Sparse Reconstruction using FBP")
        self.setMinimumSize(1400, 900)
        self.resize(1400, 900)
        
        style.apply_matplotlib_theme()
        self.setStyleSheet(style.MODERN_STYLE)
        
        main_layout = QVBoxLayout(self)
        
        # Parameters Panel
        param_group = QGroupBox("Sparse Reconstruction Parameters")
        param_layout = QGridLayout()
        
        # Number of projections
        param_layout.addWidget(QLabel("Number of Projections:"), 0, 0)
        self.num_projections = QSpinBox()
        self.num_projections.setRange(4, 360)
        self.num_projections.setValue(36)
        self.num_projections.setToolTip("Number of sparse projections (e.g., 36)")
        param_layout.addWidget(self.num_projections, 0, 1)
        
        # Angle step
        param_layout.addWidget(QLabel("Angle Step (degrees):"), 0, 2)
        self.angle_step = QDoubleSpinBox()
        self.angle_step.setRange(1.0, 90.0)
        self.angle_step.setValue(10.0)
        self.angle_step.setToolTip("Angular step size between projections (e.g., 10°)")
        param_layout.addWidget(self.angle_step, 0, 3)
        
        # Spectrum info label
        param_layout.addWidget(QLabel("Spectrum:"), 0, 4)
        self.spectrum_info = QLabel("None")
        self.spectrum_info.setStyleSheet("color: #FFD700; font-weight: bold;")
        param_layout.addWidget(self.spectrum_info, 0, 5)
        
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_compare = QPushButton("Compare with Dense")
        self.btn_compare.clicked.connect(self.run_comparison)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_compare)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)
        
        # Visualization
        self.fig = plt.figure(figsize=(16, 10), facecolor='#1E1E2E')
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas, 1)
        
        # Connect listeners for linking angle_step and num_projections
        self.num_projections.valueChanged.connect(self.on_num_projections_changed)
        self.angle_step.valueChanged.connect(self.on_angle_step_changed)
        self.linking_enabled = True
        
    def on_num_projections_changed(self, value):
        """When num_projections changes, automatically update angle_step"""
        if not self.linking_enabled:
            return
        
        self.linking_enabled = False
        # Calculate angle_step = 360 / num_projections
        if value > 0:
            new_step = 360.0 / value
            # Clamp to valid range
            new_step = max(1.0, min(90.0, new_step))
            self.angle_step.setValue(new_step)
        self.linking_enabled = True
    
    def on_angle_step_changed(self, value):
        """When angle_step changes, automatically update num_projections"""
        if not self.linking_enabled:
            return
        
        self.linking_enabled = False
        # Calculate num_projections = 360 / angle_step
        if value > 0:
            new_projections = int(360.0 / value)
            # Clamp to valid range
            new_projections = max(4, min(360, new_projections))
            self.num_projections.setValue(new_projections)
        self.linking_enabled = True

        
    def run_comparison(self):
        """
        Perform sparse reconstruction on loaded phantom with spectrum settings.
        
        **Realistic Workflow:**
        1. Takes phantom already loaded in main window (self.parent_app.fantom)
        2. Converts phantom to ATTENUATION MAP (material-based, depends on density)
        3. Computes raw sinogram from attenuation map
        4. Uses spectrum settings from SpectraToolDialog (kVp, mA, spectrum)
        5. Applies spectrum effects for detected sinogram
        6. Reconstructs using FBP
        """
        # Check if phantom is loaded
        if not hasattr(self.parent_app, 'fantom'):
            print("ERROR: No phantom loaded! Load phantom first.")
            return
        
        # Get parameters
        num_sparse = self.num_projections.value()
        angle_sparse = self.angle_step.value()
        
        # Get loaded phantom from main window
        phantom = self.parent_app.fantom
        
        # Generate sparse angles
        angles_sparse = SparseReconstruction.generate_sparse_projections(num_sparse, angle_sparse)
        angles_dense = SparseReconstruction.generate_sparse_projections(360, 1)
        
        # Get spectrum settings from main window if available
        spectrum = getattr(self.parent_app, 'q', None)
        energies = getattr(self.parent_app, 'E0', None)
        kVp = getattr(self.parent_app, 'kVp', 100)
        mA = getattr(self.parent_app, 'mA', 1)
        
        # Update spectrum info display
        if spectrum is not None and energies is not None:
            self.spectrum_info.setText(f"Active (kVp={kVp}, mA={mA})")
            self.spectrum_info.setStyleSheet("color: #00FF00; font-weight: bold;")
        else:
            self.spectrum_info.setText("None")
            self.spectrum_info.setStyleSheet("color: #FFD700; font-weight: bold;")
        
        # Process sparse reconstruction with spectrum
        sparse_results = SparseReconstruction.process_with_spectrum(
            loaded_phantom=phantom,
            angles=angles_sparse,
            spectrum=spectrum,
            energies=energies,
            kVp=kVp,
            mA=mA,
            filter_name='ramp',
            method='fbp',
            noise_level=0.0,
            high_pass=False,
            hp_strength=0.0
        )
        
        # Process dense reconstruction with spectrum
        dense_results = SparseReconstruction.process_with_spectrum(
            loaded_phantom=phantom,
            angles=angles_dense,
            spectrum=spectrum,
            energies=energies,
            kVp=kVp,
            mA=mA,
            filter_name='ramp',
            method='fbp',
            noise_level=0.0,
            high_pass=False,
            hp_strength=0.0
        )
        
        # Calculate error metrics
        # NMSE and PSNR between original phantom and full sinogram reconstruction
        metrics_phantom_vs_dense = ComparisonReconstruction.compute_reconstruction_error(
            phantom, dense_results['reconstructed']
        )
        
        # NMSE and PSNR between sparse and dense reconstructions
        metrics_sparse_vs_dense = ComparisonReconstruction.compute_reconstruction_error(
            sparse_results['reconstructed'], dense_results['reconstructed']
        )
        
        # Clear figure
        self.fig.clear()
        
        # Create custom layout: 3 columns with varying rows
        # Column 0: Sparse sinogram (row 0-2), Sparse recon (row 3-5)
        # Column 1: Full sinogram (row 0-2), Full recon (row 3-5)
        # Column 2: Attenuation map (row 0-1), Original phantom (row 2-3), Full recon (row 4-5)
        # NMSE/PSNR text between images
        
        gs = self.fig.add_gridspec(6, 3, hspace=0.25, wspace=0.3)
        
        # Column 0: Sparse sinogram and reconstruction
        ax_sparse_sino = self.fig.add_subplot(gs[0:2, 0])
        ax_sparse_recon = self.fig.add_subplot(gs[3:6, 0])
        
        # Column 1: Full sinogram and reconstruction
        ax_dense_sino = self.fig.add_subplot(gs[0:2, 1])
        ax_dense_recon = self.fig.add_subplot(gs[3:6, 1])
        
        # Column 2: Attenuation map, Original phantom, Full recon (from dense)
        ax_att_map = self.fig.add_subplot(gs[0:2, 2])
        ax_phantom = self.fig.add_subplot(gs[2:4, 2])
        ax_dense_recon_col2 = self.fig.add_subplot(gs[4:6, 2])
        
        # Set background colors
        for ax in [ax_sparse_sino, ax_sparse_recon, ax_dense_sino, ax_dense_recon, 
                   ax_att_map, ax_phantom, ax_dense_recon_col2]:
            ax.set_facecolor('#282A36')
        
        # ===== COLUMN 0: SPARSE RECONSTRUCTION =====
        # Sparse Sinogram (top)
        ax_sparse_sino.imshow(sparse_results['detected_sinogram'], cmap='gray', aspect='auto')
        ax_sparse_sino.set_title(
            f"Sparse Sinogram ({num_sparse}@{angle_sparse}°)\nkVp={kVp}, mA={mA}",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_sparse_sino.set_xlabel('Detector', color='white', fontsize=8)
        ax_sparse_sino.set_ylabel('Projection', color='white', fontsize=8)
        ax_sparse_sino.tick_params(colors='white', labelsize=7)
        
        # Sparse Reconstruction (bottom)
        ax_sparse_recon.imshow(sparse_results['reconstructed'], cmap='gray')
        ax_sparse_recon.set_title(
            f"Sparse Recon ({num_sparse} projections)",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_sparse_recon.axis('off')
        
        # ===== COLUMN 1: FULL/DENSE RECONSTRUCTION =====
        # Full Sinogram (top)
        ax_dense_sino.imshow(dense_results['detected_sinogram'], cmap='gray', aspect='auto')
        ax_dense_sino.set_title(
            f"Full Sinogram (360@1°)\nkVp={kVp}, mA={mA}",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_dense_sino.set_xlabel('Detector', color='white', fontsize=8)
        ax_dense_sino.set_ylabel('Projection', color='white', fontsize=8)
        ax_dense_sino.tick_params(colors='white', labelsize=7)
        
        # Full Reconstruction (bottom) 
        ax_dense_recon.imshow(dense_results['reconstructed'], cmap='gray')
        
        # Display NMSE and PSNR between sparse and full reconstructions IN BETWEEN
        nmse_sv = metrics_sparse_vs_dense['nmse']
        psnr_sv = metrics_sparse_vs_dense['psnr']
        metrics_text = f"Sparse vs Full:\nNMSE: {nmse_sv:.4f}\nPSNR: {psnr_sv:.2f} dB"
        ax_dense_recon.text(0.5, 0.5, metrics_text, 
                           transform=ax_dense_recon.transAxes,
                           fontsize=9, color='#FFD700', fontweight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax_dense_recon.set_title(
            f"Full Recon (360°) - Reference",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_dense_recon.axis('off')
        
        # ===== COLUMN 2: REFERENCE IMAGES & METRICS =====
        # Attenuation Map (top)
        att_map = sparse_results['attenuation_map']
        im_att = ax_att_map.imshow(att_map, cmap='viridis')
        ax_att_map.set_title(
            f"Attenuation Map (μ)",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_att_map.axis('off')
        
        # Original Phantom (middle)
        ax_phantom.imshow(phantom, cmap='gray')
        ax_phantom.set_title(
            'Original Phantom',
            color='white', fontsize=10, fontweight='bold'
        )
        ax_phantom.axis('off')
        
        # Full Reconstruction from dense sinogram (bottom) with metrics overlay
        ax_dense_recon_col2.imshow(dense_results['reconstructed'], cmap='gray')
        
        # Display NMSE and PSNR between phantom and full reconstruction IN BETWEEN
        nmse_pv = metrics_phantom_vs_dense['nmse']
        psnr_pv = metrics_phantom_vs_dense['psnr']
        metrics_text2 = f"Phantom vs Full:\nNMSE: {nmse_pv:.4f}\nPSNR: {psnr_pv:.2f} dB"
        ax_dense_recon_col2.text(0.5, 0.5, metrics_text2,
                                transform=ax_dense_recon_col2.transAxes,
                                fontsize=9, color='#FFD700', fontweight='bold',
                                ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
        
        ax_dense_recon_col2.set_title(
            f"Dense Recon (360°)",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_dense_recon_col2.axis('off')
        
        self.canvas.draw()