"""
Configuration constants for CT Simulator application
"""

# Application Settings
APP_NAME = "CTlab Simulator"
APP_VERSION = "1.4"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"

# Window Settings
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 900

# Default Parameters
DEFAULT_BEAM_GEOMETRY = "Parallel"
DEFAULT_RECONSTRUCTION_ALGO = "Filtered backprojection (FBP)"
DEFAULT_MA = 100

# Image Settings
DEFAULT_IMAGE_VOLUME = 256
DEFAULT_DETECTOR_WIDTH = 512
DEFAULT_NOISE = 0.001

# Spectral Settings
DEFAULT_MIN_ANGLE = 0
DEFAULT_MAX_ANGLE = 360
DEFAULT_STEP_ANGLE = 1

# Reconstruction Algorithms
RECONSTRUCTION_ALGORITHMS = [
    "Filtered backprojection (FBP)",
    "Least squares",
    "Tikhonov Regularization"
]

# Beam Geometries
BEAM_GEOMETRIES = ["Parallel", "Fanflat"]

# Color Theme
THEME_BACKGROUND = "#1E1E2E"
THEME_FOREGROUND = "#F8F8F2"
