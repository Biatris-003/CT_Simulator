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
        """Compare sparse vs dense FBP reconstruction"""
        num_sparse = self.num_projections.value()
        angle_sparse = self.angle_step.value()
        
        # Generate comparison (always use FBP with 'ramp' filter, no high-pass)
        results = ComparisonReconstruction.compare_sparse_vs_dense(
            num_sparse_proj=num_sparse,
            sparse_step=angle_sparse,
            num_dense_proj=360,
            dense_step=1,
            noise_level=0.0,
            method='fbp',
            filter_name='ramp',
            high_pass=False,
            hp_strength=0.0
        )
        
        # Clear figure
        self.fig.clear()
        
        # Create custom layout: 2 rows, 3 columns
        # Row 0: Sinograms (top) - left and right
        # Row 1: Reconstructions (bottom) - left, center (phantom), right
        gs = self.fig.add_gridspec(2, 3, hspace=0.45, wspace=0.15)
        
        ax_sparse_sino = self.fig.add_subplot(gs[0, 0])
        ax_dense_sino = self.fig.add_subplot(gs[0, 2])
        ax_sparse_recon = self.fig.add_subplot(gs[1, 0])
        ax_phantom = self.fig.add_subplot(gs[1, 1])
        ax_dense_recon = self.fig.add_subplot(gs[1, 2])
        
        for ax in [ax_sparse_sino, ax_dense_sino, ax_sparse_recon, ax_phantom, ax_dense_recon]:
            ax.set_facecolor('#282A36')
        
        sparse_data = results['sparse']
        dense_data = results['dense']
        
        # Plot 1: Sparse Sinogram (top-left)
        ax_sparse_sino.imshow(sparse_data['sinogram'], cmap='gray', aspect='auto')
        ax_sparse_sino.set_title(
            f"Sparse Sinogram ({sparse_data['num_projections']}@{sparse_data['angle_step']}°)",
            color='white', fontsize=11, fontweight='bold'
        )
        ax_sparse_sino.set_xlabel('Detector', color='white', fontsize=9)
        ax_sparse_sino.set_ylabel('Projection', color='white', fontsize=9)
        ax_sparse_sino.tick_params(colors='white', labelsize=8)
        
        # Plot 2: Dense Sinogram (top-right)
        ax_dense_sino.imshow(dense_data['sinogram'], cmap='gray', aspect='auto')
        ax_dense_sino.set_title(
            f"Full Sinogram (360@1°)",
            color='white', fontsize=11, fontweight='bold'
        )
        ax_dense_sino.set_xlabel('Detector', color='white', fontsize=9)
        ax_dense_sino.set_ylabel('Projection', color='white', fontsize=9)
        ax_dense_sino.tick_params(colors='white', labelsize=8)
        
        # Plot 3: Sparse Reconstruction (bottom-left)
        ax_sparse_recon.imshow(sparse_data['reconstructed'], cmap='gray')
        ax_sparse_recon.set_title(
            f"Sparse FBP ({sparse_data['num_projections']} projections)",
            color='white', fontsize=11, fontweight='bold'
        )
        ax_sparse_recon.axis('off')
        
        # Plot 4: Original Phantom (bottom-center)
        ax_phantom.imshow(sparse_data['phantom'], cmap='gray')
        ax_phantom.set_title(
            'Original Phantom',
            color='white', fontsize=11, fontweight='bold'
        )
        ax_phantom.axis('off')
        
        # Plot 5: Dense Reconstruction (bottom-right) with metrics
        ax_dense_recon.imshow(dense_data['reconstructed'], cmap='gray')
        ax_dense_recon.set_title(
            f"Dense FBP (360°)",
            color='white', fontsize=11, fontweight='bold'
        )
        ax_dense_recon.axis('off')
        
        # Compute error metrics
        error_metrics = ComparisonReconstruction.compute_reconstruction_error(
            dense_data['reconstructed'], sparse_data['reconstructed']
        )
        
        # Display metrics below the dense reconstruction
        nmse = error_metrics['nmse']
        psnr = error_metrics['psnr']
        metrics_text = f'NMSE: {nmse:.4f}\nPSNR: {psnr:.2f} dB'
        ax_dense_recon.text(0.5, -0.08, transform=ax_dense_recon.transAxes,
                           s=metrics_text, ha='center', va='top', color='white', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='#282A36', alpha=0.9))
        
        self.canvas.draw()