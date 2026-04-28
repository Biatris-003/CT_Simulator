"""
CT Simulator - Entry Point

Main application launcher for the CT Simulator using MVC architecture.
"""

import sys
import os
# Ensure project package path is on sys.path so imports like `views.*` work
sys.path.insert(0, os.path.dirname(__file__))
from PyQt5.QtWidgets import QApplication
from views.main_window import SimulatorCTLabApp


def main():
    """Initialize and run the CT Simulator application."""
    app = QApplication(sys.argv)
    window = SimulatorCTLabApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
