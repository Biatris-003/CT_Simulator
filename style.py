# style.py

def apply_matplotlib_theme():
    import matplotlib.pyplot as plt
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#282A36',
        'figure.facecolor': '#1E1E2E',
        'grid.color': '#44475A',
        'grid.linestyle': '--',
        'grid.alpha': 0.7,
        'axes.grid': True,
        'axes.edgecolor': '#6272A4',
        'axes.linewidth': 1.0,
        'lines.linewidth': 2.5,
        'text.color': '#F8F8F2',
        'axes.labelcolor': '#F8F8F2',
        'xtick.color': '#F8F8F2',
        'ytick.color': '#F8F8F2'
    })

MODERN_STYLE = """
/* Premium Dracula Dark Theme Options */

QWidget {
    background-color: #1E1E2E;
    color: #F8F8F2;
    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', sans-serif;
    font-size: 11pt;
}

/* Push Buttons */
QPushButton {
    background-color: #44475A;
    color: #F8F8F2;
    border: 2px solid #6272A4;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 11pt;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #6272A4;
    border-color: #8BE9FD;
    color: #FFFFFF;
}
QPushButton:pressed {
}

/* Group Boxes (Panels) */
QGroupBox {
    border: 1px solid #44475A;
    border-radius: 6px;
    margin-top: 14px;
    font-weight: 600;
    padding-top: 10px;
    color: #8BE9FD;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #8BE9FD;
    background-color: transparent;
}

/* Sliders */
QSlider::groove:horizontal {
    border: 1px solid #6272A4;
    height: 6px;
    background: #282A36;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #8BE9FD;
    border: 1px solid #8BE9FD;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #FFFFFF;
    border: 1px solid #FFFFFF;
}

/* Lists */
QListWidget {
    background-color: #282A36;
    border: 1px solid #44475A;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #6272A4;
    color: #F8F8F2;
}

QListWidget::item:hover {
    background-color: #44475A;
}

/* Checkboxes */
QCheckBox {
    spacing: 5px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 3px;
    border: 1px solid #6272A4;
    background-color: #282A36;
}

QCheckBox::indicator:checked {
    background-color: #8BE9FD;
    border: 1px solid #8BE9FD;
}
"""
