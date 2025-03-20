---

# Xray-Simulator – Wave Optics Forward Simulation

This repository contains a set of Python classes and utilities for simulating diffraction and scattering from crystalline samples, modeling defects (stacking faults, cracks), and visualizing detector data. The code is modular and is split across multiple files, each serving a different purpose:

1. **Beam.py** – Models beam-sample interaction for forward scattering.
2. **Crystal.py** – Provides functionality to read crystal structures (from CIF files), manipulate them, and calculate crystallographic properties.
3. **Sample.py** – Defines volumetric samples, subdivides them into “chunks,” and organizes large atomic position data in a memory-friendly manner.
4. **Detector.py** – Models a 2D detector, including positions, orientation, and pixel data.
5. **Analysis.py** – Offers post-processing and visualization routines (e.g., generating 2D/3D plots, performing FFT analyses).
6. **Defects.py** – Implements classes for inserting stacking faults and cracks into a sample.

Below is an overview of each component, followed by installation and usage instructions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Detailed Explanations](#detailed-explanations)
  - [1. Crystal Module](#1-crystal-module)
  - [2. Sample Module](#2-sample-module)
  - [3. Beam Module](#3-beam-module)
  - [4. Detector Module](#4-detector-module)
  - [5. Analysis Module](#5-analysis-module)
  - [6. Defects Module](#6-defects-module)
- [Examples](#examples)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

**Xray-Simulator** is a toolkit aimed at simulating X-ray (or electron) diffraction patterns from materials. The workflow typically proceeds as follows:

1. **Crystal**: A crystallographic structure is loaded from a file (e.g., CIF).
2. **Sample**: A volumetric sample is defined, subdivided into “chunks,” each containing a subset of atomic coordinates.
3. **Defects**: Stacking faults or cracks can be introduced to modify atomic positions.
4. **Beam**: A beam interacts with the sample, computing the scattering amplitude on a virtual 2D detector.
5. **Detector**: The detector is positioned and oriented; the scattering intensity, phase, or amplitude is simulated.
6. **Analysis**: The resulting patterns can be analyzed or visualized, including transformations, FFT analyses, etc.

---

## Features

- **Crystallography Tools**: Read crystal data from CIF files, switch between primitive/conventional cells, apply rotations, and compute d-spacings.
- **Large Sample Handling**: Subdivide large samples into “chunks,” each of which can be processed independently to handle memory constraints.
- **GPU Acceleration**: Optional use of CUDA via [CuPy](https://cupy.dev/) for faster computation on compatible GPUs.
- **Defects Modeling**: Insert stacking faults, cracks, or other structural defects into the sample geometry.
- **Detector Simulation**: Flexible 2D detector geometry and orientation with direct computation of scattering signals.
- **Analysis Utilities**: Quick generation of 2D and 3D plots, line plots, and Fourier transforms (FFT).

---

## Installation

1. **Clone** this repository:

   ```bash
   git clone https://github.com/YourOrganization/DorianCode2.git
   cd DorianCode2
   ```

2. **Install Dependencies** (see next section). You may use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

   *Note*: The `requirements.txt` file should list all needed packages, e.g., `numpy`, `cupy`, `pymatgen`, `matplotlib`, etc.

3. **Optional**: If you have an NVIDIA GPU and wish to accelerate computations, ensure [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is installed, and then install the appropriate CuPy version. For example:

   ```bash
   pip install cupy-cuda11x
   ```

---

## Dependencies

- **Python 3.7+**
- **NumPy** for array operations.
- **CuPy** *(optional)* for GPU acceleration.
- **Matplotlib** for plotting.
- **pymatgen** for reading crystal structures (CIF files).
- **scipy** for certain geometry routines (e.g., `ConvexHull`).
- **pandas** for tabular data handling when building scattering factor arrays.
- **cffi** for compiling and calling C/C++ routines on CPU.

---

## Quick Start

1. **Load a Crystal**:  
   ```python
   from Crystal import crystal
   
   my_crystal = crystal("path/to/structure.cif")
   my_crystal.get_lattice_from_cif()   # Reads & processes lattice
   ```
   
2. **Prepare a Sample**:  
   ```python
   from Sample import sample
   my_sample = sample()
   my_sample.create_sample(dimensions=[1000, 1000, 1000], offset=[500, 500, 500])
   my_sample.generate_sample(my_crystal, use_gpu=False)  # Subdivides into chunks
   ```
   
3. **Insert Defects** (optional):  
   ```python
   from Defects import defects
   my_defects = defects()
   # Add stacking faults
   my_defects.add_stacking_faults(
       fault_number=2,
       fault_offset=[10.0, 0.0, 0.0],
       fault_normal=[0.0, 0.0, 1.0],
       interfault_spacing=2.0,
       burgers_vector=[0.1, 0.0, 0.0],
       fault_orientation=[1, -1],  # Each plane can have ± orientation
       fault_gap=0.05
   )
   # Generate stacking faults globally
   my_defects.stacking_faults.generate_global_positions(my_sample, my_crystal, plotting=False)
   my_defects.stacking_faults.apply_to_sample(my_sample, use_gpu=False)
   ```
   
4. **Define a Beam**:  
   ```python
   from Beam import beam
   my_beam = beam()
   my_beam.create_beam(energy=8000.0, eV=True, direction=[1.0, 0.0, 0.0])  # 8 keV X-rays
   ```
   
5. **Set Up a Detector**:  
   ```python
   from Detector import detector
   my_detector = detector()
   my_detector.create_detector(shape=(512, 512), pixel_size=(1.0, 1.0))  # Arbitrary units
   my_detector.position_detector_absolute(distance=1000.0, two_theta=0.0, nu=0.0)
   ```
   
6. **Compute Scattering**:  
   ```python
   my_beam.atomic_direct_scattering(my_sample, my_detector, use_gpu=False)
   # Now my_detector.pixel_values contains the complex field
   ```
   
7. **Analysis**:  
   ```python
   from Analysis import analysis
   an = analysis()
   # Plot the intensity
   fig_int, ax_int = my_detector.plot_detector(type="Intensity")
   fig_int.show()
   ```

---

## Detailed Explanations

### 1. Crystal Module

- **File**: `Crystal.py`
- **Class**: `crystal`
- **Purpose**:  
  - Reads a crystal structure from a CIF file using `pymatgen`.
  - Converts between primitive and conventional cells.
  - Allows for custom rotations and alignment of the crystal lattice.
  - Computes crystallographic properties such as d-spacing for `(h,k,l)` planes.

**Key Methods**  
- `get_lattice_from_cif()`: Populates internal lattice data from a `.cif` file.  
- `align_axes(...)`: Rotates the crystal so user-specified directions align with global axes.  
- `get_dhkl((h,k,l))`: Returns the interplanar spacing `d_hkl`.  

### 2. Sample Module

- **File**: `Sample.py`
- **Class**: `sample`
- **Purpose**:  
  - Defines a parallelepiped region in 3D space that acts as a “container” for the crystal.
  - Subdivides the sample into chunks, each storing positions/species data in `.npy` files.
  - Loads/writes chunk data with `load_chunk_positions(...)`, `write_chunk_positions(...)`.
  - Takes a `crystal` object and “generates” the sample by replicating the crystal lattice throughout the sample volume.

**Key Methods**  
- `create_sample(dimensions, offset, chunk_volume)`: Initialize sample shape/size.  
- `generate_sample(my_crystal, ...)`: Build the atomic positions for the entire sample in chunks.  
- `load_chunk_positions(i)`: Returns a NumPy or CuPy array of atomic positions for the ith chunk.  

### 3. Beam Module

- **File**: `Beam.py`
- **Class**: `beam`
- **Purpose**:  
  - Manages beam energy, wavelength, and direction.
  - Provides CPU and GPU scattering kernels to compute the field at each detector pixel.
  - Integrates scattering from each chunk of the sample and sums the result into the detector array.

**Key Methods**  
- `create_beam(energy, eV, direction)`: Specifies beam parameters.  
- `atomic_direct_scattering(sample, detector, offset=0, use_gpu=True)`:  
  Main entry point that loops over all sample chunks, calculates the scattering contribution, and populates the detector’s pixel values.

### 4. Detector Module

- **File**: `Detector.py`
- **Class**: `detector`
- **Purpose**:  
  - Represents a 2D detection plane with a specified number of pixels (`shape`) and pixel size.
  - Can be positioned absolutely or relatively in space using rotations (`two_theta`, `nu`).
  - Stores the complex scattering field (real, imag) or amplitude/phase/intensity.

**Key Methods**  
- `create_detector(shape, pixel_size)`: Initializes the pixel grid.  
- `position_detector_absolute(...)`: Places the detector in space at a given distance and angles.  
- `input_pixel_values(...)`: Stores the complex field in `_pixel_values`, then automatically computes phase, amplitude, intensity.  
- `plot_detector(type="Intensity")`: Plots the chosen quantity (Intensity, Phase, or Amplitude).

### 5. Analysis Module

- **File**: `Analysis.py`
- **Class**: `analysis`
- **Purpose**:  
  - Contains post-processing routines for visualizing detector data over different sample-to-detector distances.
  - Generates 3D surface plots or line plots (e.g., for amplitude/phase FFTs).

**Key Methods**  
- `line_plot(x, y, ...)`: Produces simple line plots.  
- `surf_plot(X, Y, Z, ...)`: Produces 3D surface plots.  
- `distance_fft_dependance(...)`: Iterates over multiple detector positions and logs the changes in diffraction pattern via FFT.

### 6. Defects Module

- **File**: `Defects.py`
- **Class**: `defects`
- **Purpose**:  
  - Allows modeling of stacking faults or cracks in the crystal sample.
  - Each defect is represented by a sub-class (`stacking_fault`, `crack`), which modifies atomic positions or removes them.

**Stacking Faults**  
- `add_stacking_faults(...)`: Defines geometry of fault planes, spacing, and Burgers vectors.  
- `apply_to_sample(...)`: Loops through sample chunks, displacing atoms that lie beyond each fault plane.

**Cracks**  
- `add_cracks(...)`: Defines a convex hull that represents a crack volume.  
- `apply_crack_chunk(...)`: Removes all atoms that lie within the crack’s convex hull.

---

## Examples

Example scripts showing end-to-end usage (loading a CIF, generating a sample, inserting defects, simulating the beam, and plotting the detector) can be found in an `examples/` folder (if provided). A minimal pseudo-code example is illustrated under [Quick Start](#quick-start).

---

## License

Dorian Luccioni

---

## Acknowledgments

- **Pymatgen** library for initial crystal data import.
- **CuPy** for GPU acceleration.
- **NumPy / SciPy / Matplotlib** for scientific computing and plotting.
- **cffi** for bridging Python with high-performance C code.

---