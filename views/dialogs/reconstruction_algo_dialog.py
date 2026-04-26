import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QPushButton)
from PyQt5.QtCore import Qt
from views import style


class ReconstructionAlgoDialog(QDialog):
    def __init__(self, parent_app=None):
        super().__init__(parent_app)
        self.parent_app = parent_app
        self.setWindowTitle("Select Reconstruction Algorithm")
        self.setMinimumSize(450, 200)
        self.resize(450, 200)
        
        self.setStyleSheet(style.MODERN_STYLE)
        
        layout = QVBoxLayout()
        
        # Dropdown layout
        h_combo_layout = QHBoxLayout()
        lbl = QLabel("Reconstruction algorithm:")
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet("font-weight: bold;")
        
        self.combo = QComboBox()
        self.combo.addItems([
            "Filtered backprojection (FBP)", 
            "Least squares", 
            "Tikhonov Regularization"
        ])
        
        h_combo_layout.addWidget(lbl)
        h_combo_layout.addWidget(self.combo)
        layout.addSpacing(20)
        layout.addLayout(h_combo_layout)
        layout.addSpacing(20)
        
        # Buttons layout
        h_btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_cancel = QPushButton("Cancel")
        
        self.btn_ok.clicked.connect(self.on_ok)
        self.btn_cancel.clicked.connect(self.on_cancel)
        
        h_btn_layout.addStretch()
        h_btn_layout.addWidget(self.btn_ok)
        h_btn_layout.addWidget(self.btn_cancel)
        h_btn_layout.addStretch()
        layout.addLayout(h_btn_layout)
        
        self.setLayout(layout)

    def on_ok(self):
        if self.parent_app and hasattr(self.parent_app, 'chosen_reconstruction'):
            self.parent_app.chosen_reconstruction(self.combo.currentText())
        self.accept()

    def on_cancel(self):
        if self.parent_app and hasattr(self.parent_app, 'cancel_reconstruction_selection'):
            self.parent_app.cancel_reconstruction_selection()
        self.reject()

if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    dlg = ReconstructionAlgoDialog()
    dlg.show()
    sys.exit(app.exec())
