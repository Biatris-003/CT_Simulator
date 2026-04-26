# CT_Simulator Project - MVC Architecture

## Project Structure

This project follows the Model-View-Controller (MVC) architectural pattern for better organization, maintainability, and scalability.

### Directory Layout

```
CT_Simulator/
├── main.py                          # Application entry point
├── config.py                        # Configuration constants and settings
│
├── models/                          # Business Logic & Data Models
│   ├── __init__.py
│   ├── beam_geometry.py            # Beam geometry calculations
│   ├── imaging_parameters.py       # Imaging parameter models
│   ├── reconstruction_algo.py      # Reconstruction algorithms
│   ├── spectral_parameters.py      # Spectral data models
│   └── spectra_tool.py             # Spectra generation logic
│
├── views/                           # User Interface Components
│   ├── __init__.py
│   ├── style.py                    # Theme styling and appearance
│   ├── main_window.py              # Main application window
│   └── dialogs/                    # Dialog windows
│       ├── __init__.py
│       ├── beam_geometry_dialog.py
│       ├── imaging_parameters_dialog.py
│       ├── reconstruction_algo_dialog.py
│       ├── spectral_parameters_dialog.py
│       └── spectra_tool_dialog.py
│
├── controllers/                     # Application Control & Event Handling
│   ├── __init__.py
│   └── app_controller.py           # Main application controller
│
└── utils/                           # Utility Functions
    ├── __init__.py
    └── extract_m.py                # MATLAB code extraction utility
```

## Architecture Overview

### Models Layer
- **Purpose**: Contains all business logic and data models
- **Files**: `beam_geometry.py`, `imaging_parameters.py`, `reconstruction_algo.py`, etc.
- **Responsibility**: 
  - Handle calculations and simulations
  - Manage data transformations
  - Independent from UI

### Views Layer
- **Purpose**: All user interface components
- **Files**: `style.py`, `main_window.py`, dialog files
- **Responsibility**:
  - Display information to users
  - Capture user input
  - Communicate with controllers

### Controllers Layer
- **Purpose**: Manage communication between models and views
- **Files**: `app_controller.py`
- **Responsibility**:
  - Handle user events
  - Coordinate data flow
  - Update views when data changes

### Utils Layer
- **Purpose**: Helper functions and utilities
- **Files**: `extract_m.py` and other utility modules
- **Responsibility**:
  - Provide reusable functions
  - Handle I/O operations
  - Support extraction and conversion tasks

## Running the Application

To run the CT Simulator:

```bash
python main.py
```

## Dependencies

- PyQt5
- NumPy
- Matplotlib
- scikit-image

Install dependencies:
```bash
pip install PyQt5 numpy matplotlib scikit-image
```

## Design Benefits

✅ **Separation of Concerns** - Logic, UI, and control are clearly isolated
✅ **Maintainability** - Easy to find and modify specific functionality
✅ **Scalability** - Simple to add new features without affecting existing code
✅ **Testability** - Models can be tested independently of the UI
✅ **Code Reusability** - Common logic is centralized in models and utils

## Future Enhancements

- Extract business logic from dialog classes to models
- Add service layer for external API calls
- Implement proper MVC pattern with signal/slot connections
- Add unit tests for all model components
- Create a data persistence layer (database/file storage)
