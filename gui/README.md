# X-ray Simulator GUI

A comprehensive graphical user interface for the X-ray diffraction simulation package. This GUI provides full access to all simulation modules with GPU-accelerated visualizations, preset management, and script export capabilities.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Main Window Layout](#main-window-layout)
5. [Panels Reference](#panels-reference)
6. [Inspectors Reference](#inspectors-reference)
7. [Workflows](#workflows)
8. [Preset System](#preset-system)
9. [Script Export](#script-export)
10. [Keyboard Shortcuts](#keyboard-shortcuts)
11. [Troubleshooting](#troubleshooting)

---

## Requirements

### Core Dependencies
- Python 3.8+
- PySide6 >= 6.5 (Qt6 bindings)
- NumPy
- Matplotlib (for colormaps)

### Optional Dependencies
- **VisPy >= 0.14** - Required for 3D visualization
- **CuPy** - Required for GPU acceleration and GPU monitoring
- **PyOpenGL >= 3.1** - Required for VisPy
- **h5py** - For loading HDF5 files
- **Pillow** - For loading TIFF images
- **fabio** - For loading CBF crystallographic files

### Installation

```bash
# Core dependencies
pip install pyside6 numpy matplotlib

# Optional: 3D visualization
pip install vispy pyopengl

# Optional: GPU acceleration (requires CUDA)
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12

# Optional: File format support
pip install h5py pillow fabio
```

---

## Quick Start

### Running the GUI

From the project root directory:

```bash
python -m gui
```

Or directly:

```bash
python gui/main.py
```

### First Steps

1. **Load a Crystal Structure**: Click on "Crystal" in the Object Browser, then use the CIF loader in the Inspector panel
2. **Configure Sample**: Set sample dimensions in the Sample inspector
3. **Set Beam Parameters**: Configure energy and beam profile
4. **Run Simulation**: Click the "Run" button in the toolbar or use `Ctrl+R`
5. **View Results**: Check the Detector tab in the bottom panel

---

## Main Window Layout

```
+-----------------------------------------------------------------------------+
|  Menu Bar: File | Simulation | View | Tools | Help                          |
+-----------------------------------------------------------------------------+
|  Toolbar: [Run] [Stop] [Save] [Load] | GPU: 24GB/24GB                       |
+-------------+-----------------------------------------------+---------------+
|             |                                               |               |
|  Object     |         Central 3D Viewport                   |  Inspector    |
|  Browser    |         (VisPy Canvas)                        |  Panel        |
|             |                                               |               |
|  - Crystal  |   Sample visualization with atoms,            | Context-      |
|  - Sample   |   detector plane, and beam                    | sensitive     |
|  - Beam     |                                               | property      |
|  - Detector |   [View: XY|XZ|YZ|3D] [Ortho/Persp]           | editor        |
|  - Stage    +-----------------------------------------------+               |
|  - Optics   |                                               |               |
|  - Defects  |  Tabbed: [Detector] [Analysis] [Scan] [Log]   +---------------+
|  - Deform.  |                                               |  Console      |
|  - Analysis |  2D detector image / analysis plots           |  (Logging)    |
+-------------+-----------------------------------------------+---------------+
```

### Panel Arrangement

- **Left Panel**: Object Browser - tree view of all simulation components
- **Center Top**: 3D Viewport - interactive visualization of sample and setup
- **Center Bottom**: Tabbed panels for detector view, analysis, scan progress, and logs
- **Right Panel**: Inspector - context-sensitive property editor
- **Bottom Right**: Console - logging output with filtering

All panels are **dockable** and can be rearranged, floated, or hidden via the View menu.

---

## Panels Reference

### Object Browser

The Object Browser displays all simulation components in a hierarchical tree:

| Object | Description | Status Indicators |
|--------|-------------|-------------------|
| Crystal | Crystal structure | Green = loaded, Gray = empty |
| Sample | Sample geometry | Green = generated |
| Beam | X-ray beam | Blue = configured |
| Detector | Detector setup | Green = has data |
| Stage | Goniometer motors | Shows motor count |
| Optics | Optical components | Shows component count |
| Defects | Crystal defects | Shows defect types |
| Deformation | Strain fields | Green = field loaded |
| Analysis | Analysis tools | - |

**Interactions:**
- Single-click: Select object and show its inspector
- Double-click: Expand/collapse children
- Right-click: Context menu with object-specific actions

### Inspector Panel

The Inspector panel displays editable properties for the selected object. Property types include:

| Widget | Property Type | Example |
|--------|--------------|---------|
| Spinbox | Numeric values | Energy, dimensions |
| Slider | Bounded values | Polarization rate |
| Dropdown | Enumerated choices | Beam shape |
| Checkbox | Boolean flags | Log scale |
| Vector3 | 3D coordinates | hkl indices |
| File Path | File selection | CIF file |
| Color | Color picker | Visualization colors |

### Console Panel

Displays logging output from all simulation modules:

- **Log Levels**: DEBUG, INFO, WARNING, ERROR (filterable)
- **Search**: Filter logs by keyword
- **Export**: Save logs to file
- **Clear**: Clear log history

### Bottom Tabs

| Tab | Purpose |
|-----|---------|
| Detector | 2D detector image with colormap, zoom, and pixel inspection |
| Analysis | Analysis results and plots |
| Scan | Scan progress and intermediate results |
| Log | Alternative log view |

---

## Inspectors Reference

### Crystal Inspector

Configure crystal structure and orientation.

| Section | Controls |
|---------|----------|
| **Structure** | CIF file loader, manual lattice parameters (a, b, c, α, β, γ) |
| **Orientation** | Primary hkl, secondary hkl, lab frame alignment vectors |
| **Rotation** | Rotation axis, angle, apply button |
| **Information** | Space group, volume, d-spacing calculator |

**Workflow:**
1. Load CIF file or enter lattice parameters manually
2. Set crystallographic orientation using hkl indices
3. Apply additional rotations if needed
4. Verify with d-spacing calculator

### Sample Inspector

Configure sample geometry and generation.

| Section | Controls |
|---------|----------|
| **Dimensions** | Lx, Ly, Lz in Angstroms |
| **Type** | Single crystal / Polycrystalline |
| **Temperature** | Enable, distribution type, parameters |
| **Generation** | Generate button, progress indicator |
| **Transform** | Rotate, translate, zero position |

**Workflow:**
1. Set sample dimensions appropriate for your simulation
2. Choose crystal type (single for DFXM, poly for powder)
3. Optionally enable temperature effects
4. Click "Generate" to create the atomic positions

### Beam Inspector

Configure X-ray beam parameters.

| Section | Controls |
|---------|----------|
| **Energy** | Energy (eV), wavelength display (auto-calculated) |
| **Shape** | Rectangular / Circular dropdown |
| **Size** | Ny, Nz samples; Ly, Lz dimensions |
| **Profile** | Uniform / Gaussian; waist parameters |
| **Polarization** | Polarization rate slider (0-1) |
| **Presets** | Cu Kα, Mo Kα, Ag Kα quick buttons |

**Workflow:**
1. Set beam energy (or use a preset)
2. Choose beam shape and dimensions
3. Select profile type (Gaussian for focused beams)
4. Set polarization rate (0.99 for synchrotron)

### Detector Inspector

Configure detector geometry and display.

| Section | Controls |
|---------|----------|
| **Shape** | Ny, Nz pixels |
| **Pixel Size** | Size in micrometers |
| **Position** | Distance, 2θ, η angles |
| **Display** | Intensity/Amplitude/Phase; colormap; log scale |
| **Data** | Load from file, save to file buttons |

**Workflow:**
1. Set detector dimensions and pixel size
2. Position detector at appropriate 2θ angle
3. After simulation, adjust display settings for visualization
4. Save detector data for further analysis

### Stage Inspector

Control goniometer motors.

| Section | Controls |
|---------|----------|
| **Motor Table** | Name, type, current value, resolution |
| **Movement** | Absolute / Relative toggle |
| **Actions** | Apply, Zero, Home buttons |
| **Display** | Rotation matrix, translation vector |

**Standard Motors:**
- **omega (ω)**: Sample rotation around vertical axis
- **chi (χ)**: Sample tilt
- **phi (φ)**: Sample rotation around surface normal
- **theta (θ)**: Incident angle (coupled to 2θ in some modes)
- **x, y, z**: Linear translations

### Optics Inspector

Build optical component stack.

| Section | Controls |
|---------|----------|
| **Component List** | Ordered list of optical elements |
| **Add Component** | Dropdown with component types |
| **Component Editor** | Parameters for selected component |
| **Actions** | Add, Remove, Move Up, Move Down |

**Available Components:**
- **Free Space**: Propagation distance
- **CRL (Compound Refractive Lens)**: Focal length, aperture, material
- **Bragg Magnifier**: Magnification factor, crystal parameters
- **Aperture**: Size, shape (rectangular/circular)
- **Angular Filter**: Acceptance angle

### Defects Inspector

Configure crystal defects (tabbed interface).

| Tab | Controls |
|-----|----------|
| **Stacking Faults** | Number, offset, normal vector, spacing, Burgers vector |
| **Cracks** | Vertex table, visualization toggle |
| **Point Defects** | Type (vacancy/interstitial), fraction, species |
| **Dislocations** | OpenDiS file import, visualization |

### Deformation Inspector

Import and apply deformation fields.

| Section | Controls |
|---------|----------|
| **Import Mode** | Field (array) vs FE Mesh |
| **File Selection** | Browse for field file |
| **Field Type** | Displacement / Strain / Rotation |
| **Transform** | Scale, rotate, translate |
| **Actions** | Apply to sample, clear field |

**Supported Formats:**
- NumPy arrays (.npy, .npz)
- HDF5 files (.h5, .hdf5)
- VTK files (.vtk, .vtu)
- Abaqus output (.odb)

### Analysis Inspector

Configure and run analysis tools.

| Section | Controls |
|---------|----------|
| **Analysis Type** | FFT Distance Dependence / Detector Integration |
| **FFT Parameters** | Distance array, plot prefix |
| **Integration** | Data type, axis, bins, aggregator |
| **Output** | Embedded plot display, save options |

---

## Workflows

### Workflow 1: Basic DFXM Simulation

**Goal:** Simulate a dark-field X-ray microscopy image of a crystal with defects.

1. **Load Preset**
   - File → Load Preset → "DFXM Standard"
   - Or manually configure each component

2. **Load Crystal Structure**
   - Select "Crystal" in Object Browser
   - Click "Load CIF" and select your structure file
   - Set orientation: primary hkl = [1,1,1], secondary = [1,-1,0]

3. **Configure Sample**
   - Select "Sample" in Object Browser
   - Set dimensions: Lx=50000, Ly=50000, Lz=10000 Å
   - Click "Generate"

4. **Set Beam Parameters**
   - Select "Beam"
   - Set energy to 17000 eV
   - Profile: uniform, Size: 256×256 samples

5. **Position Detector**
   - Select "Detector"
   - Set distance to 5m (5000000 Å)
   - Set 2θ to appropriate Bragg angle

6. **Add Optics (Optional)**
   - Select "Optics"
   - Add CRL with appropriate focal length

7. **Run Simulation**
   - Click "Run" in toolbar
   - Monitor progress in status bar
   - View results in Detector tab

8. **Analyze Results**
   - Adjust colormap and scale in Detector view
   - Use Analysis inspector for further processing
   - Save results: File → Export → Detector Image

### Workflow 2: Rocking Curve Scan

**Goal:** Perform a θ-2θ scan to measure a rocking curve.

1. **Set Up Base Configuration**
   - Follow steps 1-5 from Workflow 1
   - Position detector at Bragg peak

2. **Open Scan Wizard**
   - Tools → Scan Wizard (or `Ctrl+Shift+S`)

3. **Select Motors (Page 1)**
   - Add "theta" motor to scan list
   - Click Next

4. **Define Range (Page 2)**
   - Motor: theta
   - Start: -0.5°
   - End: +0.5°
   - Steps: 101
   - Click Next

5. **Configure Output (Page 3)**
   - Mode: Live Updates (to watch progress)
   - Check "Save intermediate images"
   - Select output directory
   - Click Finish

6. **Monitor Scan**
   - Watch Scan tab for progress
   - Detector view updates at each point
   - Pause/Resume as needed

7. **View Results**
   - Scan completes with summary
   - Results saved to output directory
   - Analysis tools available for rocking curve fitting

### Workflow 3: Loading Experimental Data

**Goal:** Load and visualize experimental detector data.

1. **Open Load Dialog**
   - File → Load Data → Detector Pixels
   - Or use `Ctrl+L`

2. **Select File**
   - Browse to your data file
   - Supported: .npy, .npz, .h5, .tiff

3. **Configure Options**
   - Select dataset key (for npz/h5)
   - Check "Apply to detector object"
   - Click Preview to verify

4. **Load Data**
   - Click "Load"
   - Data appears in Detector view
   - Adjust colormap/scale as needed

5. **Compare with Simulation**
   - Check "Load as comparison overlay" for side-by-side
   - Use Analysis tools for quantitative comparison

### Workflow 4: Creating and Using Presets

**Goal:** Save current configuration for reuse.

1. **Configure Simulation**
   - Set up all components as desired

2. **Open Preset Dialog**
   - File → Save Preset
   - Or use `Ctrl+Shift+P`

3. **Save Preset**
   - Go to "Save Preset" tab
   - Enter name and category
   - Add description
   - Select components to include
   - Click "Save Preset"

4. **Load Preset Later**
   - File → Load Preset
   - Browse to your preset
   - Double-click or click "Load"

5. **Export as Script**
   - File → Export → Python Script
   - Creates standalone Python file
   - Can run without GUI

### Workflow 5: Multi-dimensional Scan

**Goal:** Perform a 2D scan varying two motors.

1. **Open Scan Wizard**
   - Tools → Scan Wizard

2. **Select Multiple Motors**
   - Add "omega" and "chi" to scan list
   - Order determines nesting (first = outer loop)

3. **Define Ranges**
   - omega: -1° to +1°, 21 steps
   - chi: -0.5° to +0.5°, 11 steps
   - Total: 21 × 11 = 231 points

4. **Configure Output**
   - Mode: Batch (recommended for large scans)
   - Enable "Generate summary plots"
   - Select output directory

5. **Run Scan**
   - Click Finish to start
   - Progress shown in Scan tab
   - Can run overnight for large scans

6. **Analyze 2D Data**
   - Summary plots generated automatically
   - Use Analysis inspector for custom analysis
   - Load intermediate images as needed

---

## Preset System

### Built-in Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| DFXM Standard | Dark-field microscopy setup | Crystal defect imaging |
| Laue Diffraction | White-beam Laue | Crystal orientation |
| Powder Diffraction | Cu Kα powder setup | Phase identification |
| Bragg Coherent | BCDI configuration | Nanocrystal strain mapping |

### User Presets

User presets are stored in:
- **Windows:** `%USERPROFILE%\.xray_simulator\presets\`
- **macOS/Linux:** `~/.xray_simulator/presets/`

### Preset File Format

Presets are JSON files with the following structure:

```json
{
  "name": "My Preset",
  "category": "User",
  "description": "Description of this configuration",
  "date": "2025-01-01T12:00:00",
  "version": "1.0",
  "parameters": {
    "crystal": { ... },
    "sample": { ... },
    "beam": { ... },
    "detector": { ... },
    "stage": { ... },
    "optics": { ... }
  }
}
```

---

## Script Export

The GUI can export configurations as standalone Python scripts.

### Export Options

- **File → Export → Python Script**
- Select components to include
- Optionally include simulation execution code

### Generated Script Structure

```python
#!/usr/bin/env python3
"""
X-ray Diffraction Simulation Script
Generated by X-ray Simulator GUI
"""

from Crystal import Crystal
from Sample import Sample
from Beam import Beam
from Detector import Detector
# ... imports

# Crystal Setup
crystal = Crystal("path/to/structure.cif")
crystal.set_orientation([1,1,1], [1,-1,0])

# Sample Setup
sample = Sample(crystal, Lx=50000, Ly=50000, Lz=10000)
sample.generate()

# Beam Setup
beam = Beam(sample, energy=17000)
beam.shape = "rectangular"
# ... more configuration

# Run Simulation
beam.atomic_direct_interaction()

# Display Results
detector.plot()
```

---

## Keyboard Shortcuts

### Global

| Shortcut | Action |
|----------|--------|
| `Ctrl+N` | New simulation |
| `Ctrl+O` | Open project |
| `Ctrl+S` | Save project |
| `Ctrl+Shift+S` | Save project as |
| `Ctrl+Q` | Quit |

### Simulation

| Shortcut | Action |
|----------|--------|
| `Ctrl+R` | Run simulation |
| `Ctrl+.` | Stop simulation |
| `Ctrl+Shift+R` | Run with options |

### View

| Shortcut | Action |
|----------|--------|
| `F11` | Toggle fullscreen |
| `Ctrl+1` | Reset layout |
| `Ctrl+2` | Show all panels |
| `Ctrl+3` | Hide panels |

### Tools

| Shortcut | Action |
|----------|--------|
| `Ctrl+Shift+S` | Open Scan Wizard |
| `Ctrl+Shift+P` | Open Preset Dialog |
| `Ctrl+L` | Load Data Dialog |
| `Ctrl+E` | Export Script |

### 3D Viewport

| Shortcut/Action | Result |
|-----------------|--------|
| Left drag | Rotate view |
| Right drag | Pan view |
| Scroll wheel | Zoom |
| Middle click | Reset view |
| `X` | View along X axis |
| `Y` | View along Y axis |
| `Z` | View along Z axis |
| `P` | Toggle perspective/orthographic |

### Detector View

| Shortcut/Action | Result |
|-----------------|--------|
| Scroll wheel | Zoom |
| Left drag | Pan |
| Right click | Context menu |
| `F` | Fit to view |
| `1` | Reset zoom to 100% |

---

## Troubleshooting

### GUI Won't Start

**Problem:** `ModuleNotFoundError: No module named 'PySide6'`

**Solution:**
```bash
pip install pyside6
```

---

**Problem:** Window appears but is blank/black

**Solution:**
- Update graphics drivers
- Try: `QT_QUICK_BACKEND=software python -m gui`

---

### 3D Viewport Issues

**Problem:** "VisPy not available" message

**Solution:**
```bash
pip install vispy pyopengl
```

---

**Problem:** 3D view is slow or laggy

**Solutions:**
- Reduce sample size (fewer atoms)
- Disable atom visualization for large samples
- Update graphics drivers
- Try different VisPy backend: set `VISPY_BACKEND=pyqt5`

---

### GPU/CUDA Issues

**Problem:** "CuPy not available" - GPU features disabled

**Solution:**
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

---

**Problem:** CUDA out of memory

**Solutions:**
- Reduce sample size
- Use chunked processing (enabled by default)
- Close other GPU applications
- Use batch mode for large scans

---

### Simulation Issues

**Problem:** Simulation runs but detector shows no signal

**Solutions:**
- Verify crystal orientation matches Bragg condition
- Check beam energy for correct wavelength
- Verify detector is positioned at diffraction peak
- Check 2θ angle calculation

---

**Problem:** Simulation is very slow

**Solutions:**
- Enable GPU acceleration (requires CuPy)
- Reduce sample size or beam samples
- Use batch mode for scans
- Check GPU memory (monitor in toolbar)

---

### File Loading Issues

**Problem:** Can't load HDF5 files

**Solution:**
```bash
pip install h5py
```

---

**Problem:** Can't load TIFF files

**Solution:**
```bash
pip install pillow
```

---

**Problem:** CIF file won't load

**Solutions:**
- Verify CIF file is valid
- Check file encoding (should be UTF-8)
- Try a different CIF file to test

---

### Display Issues

**Problem:** Fonts are too small on high-DPI display

**Solution:** The GUI should auto-detect DPI. If not:
```bash
# Windows
set QT_SCALE_FACTOR=1.5
python -m gui

# Linux/macOS
QT_SCALE_FACTOR=1.5 python -m gui
```

---

**Problem:** Dark theme colors are wrong

**Solution:**
- This may occur with custom Qt themes
- Try: `QT_STYLE_OVERRIDE=fusion python -m gui`

---

## Getting Help

- **Documentation:** This file and inline tooltips
- **Issues:** Report bugs at the project repository
- **Logs:** Check Console panel for error details
- **Debug mode:** Run with `--debug` flag for verbose output

---

## Version History

- **1.0** - Initial release with full module support
  - All 11 modules exposed via inspectors
  - GPU-accelerated 3D visualization
  - Preset system with 4 built-in configurations
  - Script export functionality
  - Live and batch scan modes
