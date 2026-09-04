# MOSAIC

**Modular Optics Simulation for Atomistic Imaging of Crystals**

A GPU-accelerated Python toolkit for forward simulation of coherent X-ray microscopy and diffraction experiments directly from atomistic models, with full wave optics support. MOSAIC handles the complete workflow from crystal structure to detector image, including defect insertion, deformation fields, coherent beam propagation, and automated experimental scans. Made by Dorian Luccioni at Stanford University.

Repository: https://github.com/dluccioni/MOSAIC

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Graphical Interface](#graphical-interface)
- [Dependencies](#dependencies)
- [Architecture Overview](#architecture-overview)
- [Detailed Modules](#detailed-modules)
  - [Crystal](#1-crystal-module)
  - [Sample](#2-sample-module)
  - [Beam](#3-beam-module)
  - [Detector](#4-detector-module)
  - [Stage](#5-stage-module)
  - [Optics](#6-optics-module)
  - [Defects](#7-defects-module)
  - [Deformation](#8-deformation-module)
  - [Analysis](#9-analysis-module)
  - [Experiment](#10-experiment-module)
- [Usage Examples](#usage-examples)
- [Example Scripts](#example-scripts)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

**MOSAIC** models the complete X-ray diffraction experiment pipeline:

```
Crystal → Sample → [Defects/Deformation] → Stage → Beam → Detector → Analysis
                                                     ↓
                                                  Optics
```

1. **Crystal** - Load crystallographic data from CIF files and manipulate the lattice orientation.
2. **Sample** - Replicate the crystal into a volumetric sample using a chunked storage model for scalability.
3. **Defects/Deformation** - Apply crystallographic defects (stacking faults, cracks, dislocations) or continuous displacement fields from FE simulations.
4. **Stage** - Position and rotate the sample using a configurable goniometer with motor coupling.
5. **Beam** - Define the X-ray source and compute kinematic scattering with GPU acceleration.
6. **Optics** - Propagate the wavefield through optical elements (CRLs, Bragg magnifiers, apertures).
7. **Detector** - Capture the complex field on a 2D detector with flexible positioning.
8. **Analysis/Experiment** - Automate scans and post-process results.

---

## Key Features

### Core Capabilities
- **GPU Acceleration** via CuPy with automatic NumPy fallback for CPU-only systems
- **Chunked Data Model** for unlimited sample sizes with controlled memory usage
- **Multi-GPU & Pinned Memory** support for high-throughput scattering calculations
- **Kinematic Scattering** with Waasmaier-Kirfel f0 and Cromer-Liberman anomalous scattering factors
- **Polarization Support** for accurate intensity calculations

### Sample Generation
- **Single Crystal** generation from CIF files with arbitrary orientation
- **Polycrystalline Samples** with Voronoi grain tessellation
- **Textured Polycrystals** with controllable texture axis and angular spread
- **MD/DFT Import** for atomistic simulation data (LAMMPS trajectories)
- **Einstein Model** thermal displacements with per-species parameters

### Defects & Deformation
- **Stacking Faults** with configurable Burgers vectors and fault planes
- **Cracks** defined via convex hull geometry
- **Point Defects** (vacancies, substitutions, interstitials)
- **Dislocation Networks** imported from OpenDiS discrete dislocation dynamics simulations
- **FE Mesh Import** from COMSOL, ABAQUS, and ANSYS with automatic element interpolation

### Optics & Detection
- **Bragg Magnifier** simulation with asymmetric crystal optics
- **Compound Refractive Lenses** (CRL) focusing
- **Angular Filters** for analyzer crystal simulation
- **Ring Detector Geometry** for powder diffraction
- **Fresnel Propagation** through arbitrary optical stacks

### Experiment Automation
- **N-Dimensional Scans** over any combination of motors
- **Motor Coupling** for correlated motion (e.g., theta-2theta scans)
- **Automated Output** with per-step detector images and metadata

---

## Installation

### Standard Installation

```bash
git clone https://github.com/dluccioni/MOSAIC.git
cd MOSAIC
pip install numpy cupy pymatgen matplotlib scipy pandas cffi h5py pyvista
```

### GPU Acceleration (Recommended)

For NVIDIA GPU support, install CuPy with the appropriate CUDA version:

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### CPU-Only Installation

The simulator automatically falls back to NumPy when CuPy is unavailable:

```bash
pip install numpy pymatgen matplotlib scipy pandas cffi h5py
```

### Graphical Interface

A PySide6 GUI wraps the same modules. From the repository root:

```bash
pip install PySide6
python -m gui
```

See [gui/QUICKSTART.md](gui/QUICKSTART.md) for the workflow.

---

## Dependencies

| Package      | Purpose                                        | Required |
|--------------|------------------------------------------------|----------|
| `numpy`      | Core array operations                          | Yes      |
| `cupy`       | GPU acceleration (CUDA)                        | No*      |
| `pymatgen`   | CIF parsing and crystallography utilities      | Yes      |
| `matplotlib` | Visualization and plotting                     | Yes      |
| `scipy`      | Numerical geometry (ConvexHull, interpolation) | Yes      |
| `pandas`     | Tabular handling of scattering factor data     | Yes      |
| `cffi`       | C extension compilation for CPU kernels        | Yes      |
| `h5py`       | HDF5 storage for large experiments             | No       |
| `pyvista`    | 3D visualization of dislocation networks       | No       |
| `PySide6`    | Graphical interface (`python -m gui`)          | No       |

*Falls back to NumPy automatically if not installed.

---

## Architecture Overview

### Chunked Processing Model

Large samples are automatically divided into chunks to manage memory:

- **Chunk Volume** controls memory per chunk (default: 12.5M atoms/chunk)
- **Streaming** processes chunks sequentially on GPU with accumulation
- **Disk Storage** allows samples larger than available RAM

---

## Detailed Modules

### 1. Crystal Module

**File:** `Crystal.py` | **Class:** `crystal`

Handles crystallographic data loading and lattice manipulation.

#### Key Methods

| Method | Description |
|--------|-------------|
| `get_lattice_from_cif()` | Load crystal structure from CIF file |
| `align_axes(orientation, alignment)` | Align crystal axes to lab frame |
| `rotate_crystal(rotation_matrix)` | Apply rotation to lattice |
| `get_dhkl(target_plane)` | Compute interplanar spacing for (hkl) |
| `get_rotation(axis, angle)` | Generate rotation matrix |
| `to_conventional()` / `to_primitive()` | Convert between cell types |

The lattice matrices hold the Cartesian vectors **a**, **b**, **c** as their
**rows** (pymatgen's convention), whether or not the crystal has been rotated.

A set of Miller indices names two different vectors, which coincide only for
cubic and other orthogonal cells: as a plane `(hkl)` it means the plane normal,
`inv(lattice_matrix_conventional) @ [h,k,l]`; as a direction `[uvw]` it means
`lattice_matrix_conventional.T @ [u,v,w]`. For α-quartz the two are 30° apart
for `(100)`. `get_cartesian_from_indices` returns either, and `align_axes` reads
its indices as plane normals by default, which is what a reflection means; pass
`index_type="direction"` for real-space directions.

#### Example

```python
from Crystal import crystal
import numpy as np

# Load silicon from CIF
xtal = crystal('databases/lattice/Si.cif')
xtal.get_lattice_from_cif()

# Align [1,1,0] to Z-axis and [1,-1,0] to Y-axis
xtal.align_axes(np.array([[1,1,0], [1,-1,0]]).T)

# Rotate 45 degrees about [0,0,1]
R = xtal.get_rotation([0,0,1], np.deg2rad(45))
xtal.rotate_crystal(R)

# Get d-spacing for (111) reflection
d111 = xtal.get_dhkl(np.array([1,1,1]))
```

---

### 2. Sample Module

**File:** `Sample.py` | **Class:** `sample`

Creates volumetric samples with chunked storage for scalability.

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_sample(dimensions, offset, chunk_volume)` | Define sample volume |
| `generate_sample_single(crystal)` | Generate single crystal sample |
| `generate_sample_poly(crystal, n_grains, ...)` | Generate polycrystalline sample |
| `import_atomic_data(filepath, elements)` | Import MD/DFT atomic positions |
| `set_temperature_einstein(T, mass_amu, theta_E_K)` | Configure thermal displacements |
| `load_chunk_positions(chunk_idx)` | Load positions for GPU processing |
| `plot_sample()` / `plot_grains()` | Visualize sample structure |

#### Example

```python
from Sample import sample

# Create sample container
samp = sample("output/")
samp.create_sample(
    dimensions=[2000, 2000, 500],  # Angstroms
    chunk_volume=12500000          # atoms per chunk
)

# Generate single crystal
samp.generate_sample_single(xtal, use_gpu=True)

# Or generate polycrystal with 8 grains
samp.generate_sample_poly(
    xtal,
    n_grains=8,
    voronoi_method="random",
    orientation_mode="textured",
    texture_axis=(0, 0, 1),
    texture_spread_deg=5,
    use_gpu=True
)

# Enable thermal displacements (Einstein model)
samp.set_temperature_einstein(300, mass_amu=28.085, theta_E_K=645.0)
samp.enable_temp = True
```

---

### 3. Beam Module

**File:** `Beam.py` | **Class:** `beam`

Defines the X-ray source and computes scattering.

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_beam(energy, beam_shape, beam_size, beam_profile)` | Configure source |
| `atomic_direct_interaction(sample, detector, stage, ...)` | Main scattering + transmission |
| `atomic_scattering_kinematic(sample, detector, stage)` | GPU kinematic scattering |
| `wavefield_propagation(detector, optics)` | Propagate through optics stack |
| `set_phase_tolerance(tol)` | Set phase accuracy threshold (default 1e-3 rad if never set) |
| `set_wavefield(array)` | Supply a custom complex wavefield; shape must match `beam_samples`; call AFTER `create_beam` (re-creating or reloading the beam rebuilds the built-in profile) |

#### Scattering Physics

- **Form factors:** Waasmaier-Kirfel 9-parameter f0 coefficients
- **Anomalous scattering:** Cromer-Liberman f' and f'' from Henke tables
- **Polarization:** Configurable perpendicular/parallel polarization ratio

#### Example

```python
from Beam import beam

bx = beam("output/")
bx.create_beam(
    energy=10000,                    # eV
    beam_shape="rectangular",
    beam_size=(3000.0, 3000.0),      # micrometers
    beam_samples=(512, 512),
    beam_profile="uniform",
    pol_perp_rate=0.5                # 50% perpendicular polarization
)

bx.set_phase_tolerance(1e-8)

# Compute scattering
bx.atomic_direct_interaction(
    samp, det, stg,
    scattering=True,
    sc_kwargs={"remove_forward": False, "analyser_mode": "top-hat"},
    transmission=False,
    use_gpu=True
)
```

---

### 4. Detector Module

**File:** `Detector.py` | **Class:** `detector`

Models 2D area detectors with flexible positioning.

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_detector(shape, pixel_size, geometry)` | Define detector |
| `position_detector_absolute(distance, two_theta, eta)` | Absolute positioning |
| `position_detector_relative(distance, two_theta, eta)` | Relative positioning |
| `input_pixel_values(values)` | Store complex field values |
| `plot_detector(type, scaling, cmap)` | Plot intensity/amplitude/phase |
| `plot_detector_angles(...)` | Plot in angular coordinates |
| `coordinate_conversion(...)` | Convert between coordinate systems |

#### Geometry Types

- **Rectangular:** Standard flat panel detector
- **Ring:** Cylindrical geometry for powder diffraction

#### Example

```python
from Detector import detector
import numpy as np

det = detector("output/")

# Rectangular detector
det.create_detector(
    shape=np.array([512, 512]),
    pixel_size=np.array([10, 10])  # micrometers
)

# Position at 2500mm, 88.27 degrees 2theta
det.position_detector_absolute(2500, 88.2769, 0)

# Ring detector for powder diffraction
det_ring = detector()
det_ring.create_detector(
    shape=np.array([300, 425]),
    pixel_size=np.array([2000, 7500]),
    geometry="ring"
)
```

---

### 5. Stage Module

**File:** `Stage.py` | **Class:** `stage`

Goniometer and sample positioning with motor coupling.

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_stage(motor_name, motor_type, motor_axis)` | Define motor configuration |
| `set_motor_value_absolute(motor_value_abs)` | Set absolute motor positions |
| `set_motor_value_relative(motor_value_rel)` | Set relative motor positions |
| `get_rotation()` | Get cumulative rotation matrix |
| `get_translation()` | Get cumulative translation vector |

#### Default Motor Configuration

| Motor | Type | Axis | Description |
|-------|------|------|-------------|
| mu (μ) | rotation | [0,-1,0] | Sample rotation about -Y |
| phi (φ) | rotation | [0,-1,0] | Sample rotation about -Y (nested in mu) |
| chi (χ) | rotation | [-1,0,0] | Sample rotation about -X |
| omega (ω) | rotation | [0,0,-1] | Sample rotation about -Z |
| x | translation | [1,0,0] | X translation |
| y | translation | [0,1,0] | Y translation |
| z | translation | [0,0,1] | Z translation |

A standard Busing-Levy four-circle configuration (motors `omega`, `chi`, `phi` about lab z, x, z plus `x`,`y`,`z` translations, composing R = R_z(omega) R_x(chi) R_z(phi)) is available via `create_stage(convention='busing-levy')`; the detector arm supplies 2theta separately.

#### Example

```python
from Stage import stage

stg = stage("output/")
stg.create_stage()  # Default mu-phi-chi-omega + xyz

# Rotate to Bragg condition
stg.set_motor_value_relative(
    motor_value_rel=[-44.1384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
)
```

---

### 6. Optics Module

**File:** `Optics.py` | **Class:** `optics`

Wavefield propagation through optical element stacks.

#### Key Methods

| Method | Description |
|--------|-------------|
| `add_free_space(length_mm)` | Add drift space |
| `add_CRL_box(number, focal_length, thickness, mu_per_m=0.0, radius_of_curvature_m=inf)` | Compound refractive lenses; optional parabolic absorption: intensity T(r)=exp(-mu·N·r²/R) (off by default) |
| `add_bragg_magnifier_2b(Mx, My)` | Asymmetric Bragg magnifier |
| `add_angular_filter(half_angle, shape, rolloff)` | Analyzer crystal |
| `add_aperture(width, shape)` | Hard aperture |
| `plot_stack_3d()` | Visualize optics configuration |

#### Example

```python
from Optics import optics

op = optics("output/")

# Bragg magnifier with 3x magnification
op.add_bragg_magnifier_2b(3.0, 3.0)

# Or CRL focusing
op.add_CRL_box(number=2, focal_length=200.0, thickness=1.0)

# Propagate wavefield
bx.wavefield_propagation(det, op, use_gpu=True)
```

---

### 7. Defects Module

**File:** `Defects.py` | **Class:** `defects`

Insert crystallographic defects and import dislocation networks.

#### Key Methods

| Method | Description |
|--------|-------------|
| `add_stacking_faults(...)` | Create planar stacking faults |
| `add_cracks(vertices)` | Create crack via convex hull |
| `add_point_defects(...)` | Vacancies, substitutions, interstitials |
| `import_dislocation_network(filepath, crystal, ...)` | Import OpenDiS data |
| `clip_dislocation_network_to_sample(sample)` | Clip network to sample bounds |
| `generate_nodal_field(...)` | Compute displacement field from dislocations |
| `visualize_dislocation_network(...)` | 3D visualization with PyVista |

#### Example

```python
from Defects import defects

dft = defects("output/")

# Add stacking fault
dft.add_stacking_faults(
    fault_number=20,
    fault_offset=np.array([0, 0, 0]),
    fault_normal=np.array([1, 1, 1]),
    fault_spacing=1.428 * xtal.get_dhkl(np.array([1, 1, 1])),
    burgers_vector=1/6 * np.array([1, 1, -2]),
    fault_extent=[1, -1],
    fault_extent_sigma=0.02
)
dft.stacking_faults.generate_global_positions(samp, xtal)
dft.stacking_faults.apply_to_sample(samp)

# Add crack
crack_z = samp.dimensions[2]
vertices = np.array([
    [-3.5, -150, 0], [-3.5, 150, 0], [3.5, 150, 0], [3.5, -150, 0],
    [-3.5, -150, crack_z], [-3.5, 150, crack_z], [3.5, 150, crack_z], [3.5, -150, crack_z]
]) - np.array([[0, 0, crack_z/2]])
dft.add_cracks(vertices)
dft.cracks.apply_to_sample(samp)
```

---

### 8. Deformation Module

**File:** `Deformation.py` | **Class:** `deformation`

Apply continuous displacement fields from FE simulations.

#### Key Methods

| Method | Description |
|--------|-------------|
| `import_fe_nodal_field(filepath, preset)` | Load nodal coordinates + displacements |
| `import_fe_connectivity(filepath, preset)` | Load element connectivity |
| `clip_fe_mesh_to_sample(sample)` | Clip mesh to sample bounds |
| `apply_fe_nodal_field(sample)` | Interpolate and apply deformation |
| `import_deformation_field(filepath, preset)` | Load deformation gradient tensor field |
| `apply_deformation_chunked(sample)` | Apply F tensor to atomic positions |

#### Import Presets

| Preset | Description |
|--------|-------------|
| `comsol_nodes_txt` | COMSOL nodal displacement export |
| `comsol_mesh_txt` | COMSOL mesh connectivity |
| `abaqus_dat` | ABAQUS output database |
| `ansys_csv` | ANSYS CSV export |
| `generic_xyzu` | Generic X,Y,Z,Ux,Uy,Uz format |
| `generic_tet4_ws` | Generic tetrahedral mesh |

#### Example

```python
from Deformation import deformation

dfm = deformation("output/")

# Import COMSOL results
dfm.import_fe_nodal_field("Disp.txt", preset="comsol_nodes_txt")
dfm.import_fe_connectivity("Mesh.mphtxt", preset="comsol_mesh_txt")

# Apply to sample
dfm.clip_fe_mesh_to_sample(samp)
dfm.apply_fe_nodal_field(samp)
```

---

### 9. Analysis Module

**File:** `Analysis.py` | **Class:** `analysis`

Post-processing and visualization utilities.

#### Key Methods

| Method | Description |
|--------|-------------|
| `distance_fft_dependance(...)` | Detector distance sweep with FFT analysis |
| `integrate_detector_along_axis(...)` | 1D integration of detector data |
| `surf_plot(X, Y, Z, ...)` | 3D surface plot |
| `line_plot(x, y, ...)` | 2D line plot |

#### Example

```python
from Analysis import analysis

ana = analysis("output/figures/")

# FFT analysis at multiple distances
X, Y, Z_amp, Z_pha = ana.distance_fft_dependance(
    samp, bx, stg, det,
    distance_array=np.arange(500, 8500, 500),
    plot_prefix="Diamond"
)

# Integrate detector along 2theta
centers, values = ana.integrate_detector_along_axis(
    det,
    data_type="Intensity",
    axis="2theta",
    system="angular",
    bins=200
)
```

---

### 10. Experiment Module

**File:** `Experiment.py` | **Class:** `experiment`

Automated N-dimensional motor scans.

#### Key Methods

| Method | Description |
|--------|-------------|
| `scan_nD(sample, beam, detector, stage, ranges, stepsizes, motors, ...)` | Execute scan |
| `plot_geometry_3d()` | Visualize experimental geometry |

#### Motor Coupling

Coupled motors move together with a defined relationship:
```python
couplings = {"phi": [("two_theta", "1:-2")]}  # two_theta = -2 * delta_phi
```

#### Example

```python
from Experiment import experiment

exp = experiment("output/")

# 1D rocking curve
result = exp.scan_nD(
    sample=samp,
    beam=bx,
    detector=det,
    stage=stg,
    optics=op,
    ranges=[(-6, +6)],
    stepsizes=[0.1],
    motors=["phi"],
    degrees=True,
    scan_mode="relative",
    per_step_outputs=("Amplitude",),
    adi_kwargs={"scattering": True, "transmission": False},
    prop_kwargs={"use_gpu": True},
    save_dir="output/rocking_curve/"
)

# 2D scan with motor coupling (theta-2theta)
result = exp.scan_nD(
    sample=samp, beam=bx, detector=det, stage=stg, optics=op,
    ranges=[(-5, +1)],
    stepsizes=[0.1],
    motors=["phi"],
    couplings={"phi": [("two_theta", "1:-2")]},
    save_dir="output/theta_2theta/"
)
```

---

## Usage Examples

### 1. Basic Single Crystal Diffraction

```python
import numpy as np
from Crystal import crystal
from Sample import sample
from Detector import detector
from Beam import beam
from Stage import stage

# Load crystal structure
xtal = crystal('databases/lattice/Si.cif')
xtal.get_lattice_from_cif()
xtal.align_axes(np.array([[1,1,-2], [1,-1,0]]).T)

# Create sample
samp = sample("output/")
samp.create_sample([2000, 2000, 500], chunk_volume=12500000)
samp.generate_sample_single(xtal, use_gpu=True)

# Configure stage
stg = stage("output/")
stg.create_stage()
stg.set_motor_value_relative([-44.1384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Configure detector
det = detector("output/")
det.create_detector(np.array([256, 256]), np.array([10, 10]))
det.position_detector_absolute(2500, 88.2769, 0)

# Configure beam and compute scattering
bx = beam("output/")
bx.create_beam(10000, beam_shape="rectangular", beam_size=(3000.0, 3000.0))
bx.atomic_direct_interaction(samp, det, stg, scattering=True, use_gpu=True)

# Visualize
det.plot_detector(type="Amplitude", scaling="linear", cmap="viridis")
```

### 2. Stacking Faults and Cracks

```python
from Defects import defects

# Create defects container
dft = defects("output/")

# Add stacking faults on {111} planes
dft.add_stacking_faults(
    fault_number=20,
    fault_offset=np.array([0, 0, 0]),
    fault_normal=np.array([1, 1, 1]),
    fault_spacing=1.428 * xtal.get_dhkl(np.array([1, 1, 1])),
    burgers_vector=1/6 * np.array([1, 1, -2]),
    fault_extent=[1, -1],
    fault_extent_sigma=0.02
)
dft.stacking_faults.generate_global_positions(samp, xtal)
dft.stacking_faults.apply_to_sample(samp)

# Add crack geometry
crack_z = samp.dimensions[2]
crack_vertices = np.array([
    [-3.5, -30, 0], [-3.5, 30, 0], [3.5, 30, 0], [3.5, -30, 0],
    [-3.5, -30, crack_z], [-3.5, 30, crack_z], [3.5, 30, crack_z], [3.5, -30, crack_z]
]) - np.array([[0, 0, crack_z/2]])
dft.add_cracks(crack_vertices)
dft.cracks.apply_to_sample(samp)
```

### 3. Polycrystalline Sample with Texture

```python
# Create polycrystalline sample
samp = sample("output/")
samp.create_sample([2000, 2000, 500], chunk_volume=12500000, sample_type="poly")

# Generate with textured grain orientations
samp.generate_sample_poly(
    xtal,
    n_grains=25,
    voronoi_method="random",
    randomness_seed=42,
    orientation_mode="textured",
    texture_axis=(0.0, 0.0, 1.0),  # Texture along Z
    texture_spread_deg=5,          # 5 degree spread
    use_gpu=True
)

# Visualize grain structure
samp.plot_grains(elev=90, azim=0)

# Or define custom grain orientations
samp.input_grain_orientation(np.stack([
    np.eye(3),
    samp.get_rotation(np.array([1, 0, 0]), np.deg2rad(5.0))
]))
samp.generate_sample_poly(xtal, n_grains=2, use_gpu=True)
```

### 4. MD/LAMMPS Data Import

```python
# Import LAMMPS trajectory
samp = sample("output/")
samp.import_atomic_data(
    "path/to/dump.lammpstrj",
    elements=["C"]  # Element symbols in order
)
samp.center_atomic_data()
samp.write_sample_metadata()

# Reload saved sample later
samp2 = sample("output/")
samp2.read_sample_metadata()
```

### 5. DDD Dislocation Network Import (OpenDiS)

Complete workflow for importing and applying discrete dislocation dynamics data:

```python
from Crystal import crystal
from Sample import sample
from Defects import defects
from Deformation import deformation
from Detector import detector
from Beam import beam
from Stage import stage

# Load crystal structure
xtal = crystal('databases/lattice/C.cif')
xtal.get_lattice_from_cif()
xtal.rotate_crystal(xtal.get_rotation([0, 1, 0], np.deg2rad(-44.1384)))

# Create sample volume
samp = sample("output/")
samp.create_sample([30000, 30000, 1200], chunk_volume=12500000, sample_type="single")
samp.generate_sample_single(xtal, use_gpu=True)

# Import OpenDiS dislocation network
dft = defects("output/")
summary = dft.import_dislocation_network(
    filepath="path/to/config.2900.data",
    crystal=xtal,
    burgers_family="fcc_110_over_2"  # FCC <110>/2 Burgers vectors
)
print(f"Loaded: {summary['segment_count']} segments, {summary['node_count']} nodes")

# Clip network to sample bounds
dft.clip_dislocation_network_to_sample(samp)

# Visualize the dislocation network
dft.visualize_dislocation_network(
    sample=samp,
    mode='pyvista',
    color_mode='signed_burgers',
    length_filter_percentile=100.0,
    elev=90.0,
    azim=0.0
)

# Generate displacement field from dislocations.
# mode="direct" (default) sums the Barnett segment field at every node;
# mode="LR+SR" runs the Bertin (2019) spectral solver on the periodic grid
# with the analytic near-field correction, and accepts an anisotropic
# stiffness={"cubic": (c11, c12, c44)}. Pad the box for the spectral modes.
dft.generate_nodal_field(
    mu=26e9 * (1e-10),      # Shear modulus (Pa converted to Angstrom units)
    nu=0.33,                 # Poisson ratio
    grid_shape=(256, 256, 32),
    core_radius=5.0,
    mode="direct",
    write_directory="output/",
    nodes_filename="opendis_nodes_fe.npy",
    conn_filename="opendis_tet4.npy",
    file_format="npy",
    use_gpu=True
)

# Apply displacement field via FE interpolation
dfm = deformation("output/")
dfm.import_fe_nodal_field("output/opendis_nodes_fe.npy", preset="generic_xyzu", use_gpu=True)
dfm.import_fe_connectivity("output/opendis_tet4.npy", preset="generic_tet4_ws")
dfm.clip_fe_mesh_to_sample(samp)
dfm.apply_fe_nodal_field(samp)

# Configure detector and beam
det = detector("output/")
det.create_detector(np.array([128, 128]), np.array([220, 220]))
det.position_detector_absolute(2500, 88.2769, 0)

stg = stage("output/")
stg.create_stage()

bx = beam("output/")
bx.create_beam(10000, beam_shape="rectangular", beam_size=(100000.0, 100000.0))
bx.set_phase_tolerance(1e-8)

# Compute scattering
bx.atomic_direct_interaction(
    samp, det, stg,
    scattering=True,
    sc_kwargs={"remove_forward": False, "analyser_mode": "top-hat", "analyser_acceptance_angle_rad": 125e-4},
    use_gpu=True
)

det.plot_detector(type="Amplitude", scaling="linear", cmap="viridis")
```

### 6. FE Deformation Field Application

```python
from Deformation import deformation

# Create sample
samp = sample("output/")
samp.create_sample([2000, 2000, 500], chunk_volume=12500000)
samp.generate_sample_single(xtal, use_gpu=True)

# Import COMSOL FE results
dfm = deformation("output/")
dfm.import_fe_nodal_field("path/to/Disp.txt", preset="comsol_nodes_txt")
dfm.import_fe_connectivity("path/to/Mesh.mphtxt", preset="comsol_mesh_txt")

# Apply to sample
dfm.clip_fe_mesh_to_sample(samp)
dfm.apply_fe_nodal_field(samp)

# Visualize deformed sample exterior
samp.plot_sample_exterior(voxels=100)
```

### 7. Transmission Calculations (Beer-Lambert)

```python
# Study transmission vs thickness
depths = np.array([200, 1000, 5000, 10000, 25000])  # Angstroms
transmittance = np.zeros(depths.shape)

for i, depth in enumerate(depths):
    # Create sample with varying thickness
    samp = sample("output/")
    samp.create_sample([depth, 500, 500], chunk_volume=12500000)
    samp.generate_sample_single(xtal, use_gpu=True)

    # Configure detector at sample exit
    det = detector("output/")
    det.create_detector(np.array([256, 256]), np.array([1, 1]))
    det.position_detector_relative(samp.dimensions[0]/2, 0, 0)

    # Compute transmission only
    bx = beam("output/")
    bx.create_beam(10000, beam_size=(3000.0, 3000.0))
    bx.precompute_depth_ein_all_chunks(samp, stg, use_gpu=True)
    bx.atomic_direct_interaction(
        samp, det, stg,
        scattering=False,
        transmission=True,
        use_gpu=True
    )

    transmittance[i] = np.mean(det.pixel_intensity)

# Plot Beer-Lambert curve
plt.plot(depths, transmittance)
plt.xlabel("Sample Thickness (Angstrom)")
plt.ylabel("Transmittance")
```

### 8. Optics Propagation (Bragg Magnifier)

```python
from Optics import optics

# Compute initial scattering
bx.atomic_direct_interaction(samp, det, stg, scattering=True, use_gpu=True)
det.plot_detector(type="Amplitude", cmap="viridis")  # Before optics

# Configure Bragg magnifier
op = optics("output/")
op.add_bragg_magnifier_2b(3.0, 3.0)  # 3x magnification in both directions

# Propagate wavefield through optics
bx.wavefield_propagation(det, op, use_gpu=True, save_field=True)
det.plot_detector(type="Amplitude", cmap="viridis")  # After magnification
```

### 9. Experimental Scans with Motor Coupling

```python
from Experiment import experiment

exp = experiment("output/")

# Simple rocking curve (phi scan)
result = exp.scan_nD(
    sample=samp,
    beam=bx,
    detector=det,
    stage=stg,
    optics=op,
    ranges=[(-6, +6)],
    stepsizes=[0.1],
    motors=["phi"],
    degrees=True,
    scan_mode="relative",
    per_step_outputs=("Amplitude",),
    adi_kwargs={
        "scattering": True,
        "sc_kwargs": {"remove_forward": False, "analyser_mode": "tophat"},
        "transmission": False
    },
    prop_kwargs={"use_gpu": True},
    show_plots=False,
    save_dir="output/rocking_curve/"
)

# Theta-2theta scan with motor coupling
result = exp.scan_nD(
    sample=samp, beam=bx, detector=det, stage=stg, optics=op,
    ranges=[(-5, +1)],
    stepsizes=[0.1],
    motors=["phi"],
    couplings={"phi": [("two_theta", "1:-2")]},  # 2theta moves -2x phi
    per_step_outputs=("Amplitude",),
    save_dir="output/theta_2theta/"
)

# 2D scan (mu vs phi)
result = exp.scan_nD(
    sample=samp, beam=bx, detector=det, stage=stg, optics=op,
    ranges=[(-4, +4), (-5, +0)],
    stepsizes=[0.25, 0.2],
    motors=["mu", "phi"],
    per_step_outputs=("Amplitude",),
    save_dir="output/2d_scan/"
)
```

---

## Example Scripts

### Complete Examples (`scripts/complete/`)

| Script | Description |
|--------|-------------|
| `Tutorial.py` | End-to-end workflow demonstration |
| `SCD_gpu.py` | Single crystal diffraction with GPU |
| `NPD_gpu.py` | Powder (nanocrystalline) diffraction |

---

## Troubleshooting

### CUDA / GPU Issues

**CuPy not found:**
```
ImportError: No module named 'cupy'
```
Install CuPy with the correct CUDA version, or run in CPU-only mode (automatic fallback).

**Out of GPU memory:**
- Reduce `chunk_volume` when creating samples
- Reduce `beam_samples` in beam configuration
- Process fewer chunks at once

### Memory Issues

**Out of RAM:**
- Use smaller `chunk_volume` (fewer atoms per chunk)
- Enable chunked processing for large samples
- Use disk-backed sample storage

### Common Errors

**"CIF file not found":**
- Verify the path to your CIF file
- Check that pymatgen is installed correctly

**"No scattering computed":**
- Ensure sample is generated before calling `atomic_direct_interaction`
- Check that detector is positioned to capture the diffracted beam
- Verify stage angles place sample in Bragg condition

---

## Documentation

- This README is the reference for the scripting API: one section per module with the arguments and an example for every user-facing method.
- [gui/QUICKSTART.md](gui/QUICKSTART.md) covers the graphical interface, presets, and the peak-alignment workflow.
- The physical model, the coordinate conventions, the numerical scheme of the scattering kernel, and the validation benchmarks are described in the accompanying paper (see Citation).
- Every public function carries a docstring; `help(Beam.beam.atomic_direct_interaction)` and the like give the full argument lists.

## Citation

If MOSAIC contributes to a publication, please cite the software release and the paper:

> D. Luccioni and L. Dresselhaus-Marais, *MOSAIC: End-to-end atomistic forward simulation of coherent X-ray experiments with GPU-accelerated wave optics*, J. Appl. Cryst. (submitted, 2026).

A machine-readable citation is provided in [CITATION.cff](CITATION.cff); GitHub's "Cite this repository" button reads it. The archived release (version 1.0.0) will carry a Zenodo DOI.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Dorian Luccioni

---

## Acknowledgments

- **Pymatgen** - Crystal structure parsing and crystallography utilities
- **CuPy** - GPU-accelerated array operations
- **NumPy / SciPy / Matplotlib** - Scientific computing and visualization
- **PyVista** - 3D visualization of dislocation networks
- **cffi** - High-performance C extension compilation
