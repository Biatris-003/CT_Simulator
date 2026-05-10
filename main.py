import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from PyQt5.QtWidgets import QApplication
from views.main_window import SimulatorCTLabApp


def main():
    app = QApplication(sys.argv)
    window = SimulatorCTLabApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
