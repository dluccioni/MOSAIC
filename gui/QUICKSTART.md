# X-ray Simulator GUI - Quick Start Guide

## Running the GUI

```bash
cd "x:\Dresselhaus Lab\Code\Xray-Simulator"
python -m gui
```

## Essential Workflow

```
1. Load Crystal    →  Object Browser → Crystal → Load CIF
2. Generate Sample →  Object Browser → Sample → Generate
3. Configure Beam  →  Object Browser → Beam → Set energy
4. Run Simulation  →  Toolbar → Run (or Ctrl+R)
5. View Results    →  Bottom Panel → Detector tab
```

## Common Tasks

| Task | How To |
|------|--------|
| Load preset | File → Load Preset → Select preset |
| Save configuration | File → Save Preset |
| Run simulation | Toolbar "Run" or `Ctrl+R` |
| Stop simulation | Toolbar "Stop" or `Ctrl+.` |
| Scan multiple angles | Tools → Scan Wizard |
| Export as script | File → Export → Python Script |
| Load detector data | File → Load Data → Detector Pixels |
| Change colormap | Detector tab → Colormap dropdown |
| Toggle log scale | Detector tab → Log checkbox |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+R` | Run simulation |
| `Ctrl+.` | Stop simulation |
| `Ctrl+S` | Save project |
| `Ctrl+L` | Load data |
| `Ctrl+Shift+S` | Scan wizard |
| `F11` | Fullscreen |

## 3D Viewport Controls

| Mouse | Action |
|-------|--------|
| Left drag | Rotate |
| Right drag | Pan |
| Scroll | Zoom |
| Middle click | Reset view |

## Detector View Controls

| Mouse | Action |
|-------|--------|
| Scroll | Zoom |
| Drag | Pan |
| Hover | Show pixel value |

## Quick Presets

| Preset | Energy | Use Case |
|--------|--------|----------|
| DFXM Standard | 17 keV | Defect imaging |
| Laue Diffraction | 20 keV | Orientation |
| Powder Diffraction | 8 keV (Cu Kα) | Phase ID |
| Bragg Coherent | 9 keV | Strain mapping |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| 3D view not working | `pip install vispy pyopengl` |
| No GPU acceleration | `pip install cupy-cuda11x` |
| Can't load HDF5 | `pip install h5py` |
| Fonts too small | Set `QT_SCALE_FACTOR=1.5` |

## File Locations

| Files | Location |
|-------|----------|
| User presets | `~/.xray_simulator/presets/` |
| Built-in presets | `gui/presets/` |
| Logs | Console panel or `~/.xray_simulator/logs/` |

---

For detailed documentation, see [README.md](README.md)
