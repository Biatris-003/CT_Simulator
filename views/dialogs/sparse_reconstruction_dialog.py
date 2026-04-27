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
        self.setMinimumSize(1300, 800)
        self.resize(1300, 800)
        
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
        
        # Clear figure
        self.fig.clear()
        
        # Create custom layout: 2 rows, 4 columns
        # Row 0: Attenuation maps and sinograms
        # Row 1: Reconstructions
        gs = self.fig.add_gridspec(2, 4, hspace=0.4, wspace=0.2)
        
        ax_att_map = self.fig.add_subplot(gs[0, 0])
        ax_sparse_sino = self.fig.add_subplot(gs[0, 1])
        ax_dense_sino = self.fig.add_subplot(gs[0, 2])
        ax_spare_colorbar = self.fig.add_subplot(gs[0, 3])
        
        ax_sparse_recon = self.fig.add_subplot(gs[1, 0])
        ax_phantom = self.fig.add_subplot(gs[1, 1])
        ax_dense_recon = self.fig.add_subplot(gs[1, 2:4])
        
        for ax in [ax_att_map, ax_sparse_sino, ax_dense_sino, ax_sparse_recon, ax_phantom, ax_dense_recon]:
            ax.set_facecolor('#282A36')
        
        ax_spare_colorbar.set_facecolor('#282A36')
        ax_spare_colorbar.axis('off')
        
        # Plot 0: Attenuation Map (top-left)
        att_map = sparse_results['attenuation_map']
        im_att = ax_att_map.imshow(att_map, cmap='viridis')
        ax_att_map.set_title(
            f"Attenuation Map (μ)\nMaterial-based, density-dependent",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_att_map.axis('off')
        
        # Plot 1: Sparse Sinogram (top-left-center) - detected sinogram with spectrum effects
        ax_sparse_sino.imshow(sparse_results['detected_sinogram'], cmap='gray', aspect='auto')
        ax_sparse_sino.set_title(
            f"Sparse Sinogram ({num_sparse}@{angle_sparse}°)\nkVp={kVp}, mA={mA}",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_sparse_sino.set_xlabel('Detector', color='white', fontsize=8)
        ax_sparse_sino.set_ylabel('Projection', color='white', fontsize=8)
        ax_sparse_sino.tick_params(colors='white', labelsize=7)
        
        # Plot 2: Dense Sinogram (top-right-center) - detected sinogram with spectrum effects
        ax_dense_sino.imshow(dense_results['detected_sinogram'], cmap='gray', aspect='auto')
        ax_dense_sino.set_title(
            f"Full Sinogram (360@1°)\nkVp={kVp}, mA={mA}",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_dense_sino.set_xlabel('Detector', color='white', fontsize=8)
        ax_dense_sino.set_ylabel('Projection', color='white', fontsize=8)
        ax_dense_sino.tick_params(colors='white', labelsize=7)
        
        # Plot 3: Sparse Reconstruction (bottom-left)
        ax_sparse_recon.imshow(sparse_results['reconstructed'], cmap='gray')
        ax_sparse_recon.set_title(
            f"Sparse FBP ({num_sparse} projections)",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_sparse_recon.axis('off')
        
        # Plot 4: Original Phantom (bottom-left-center)
        ax_phantom.imshow(phantom, cmap='gray')
        ax_phantom.set_title(
            'Original Phantom',
            color='white', fontsize=10, fontweight='bold'
        )
        ax_phantom.axis('off')
        
        # Plot 5: Dense Reconstruction (bottom-right)
        ax_dense_recon.imshow(dense_results['reconstructed'], cmap='gray')
        ax_dense_recon.set_title(
            f"Dense FBP (360°) - Reference",
            color='white', fontsize=10, fontweight='bold'
        )
        ax_dense_recon.axis('off')
        
        self.canvas.draw()