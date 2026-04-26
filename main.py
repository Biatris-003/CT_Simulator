"""
CT Simulator - Entry Point

Main application launcher for the CT Simulator using MVC architecture.
"""

import sys
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
