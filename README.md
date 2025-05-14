---

# Xray‑Simulator – Wave Optics Forward Simulation (Production Release 2025‑05)

This repository provides a Python toolkit for forward simulation of X‑ray (and electron) diffraction,
including crystal generation, defect insertion, beam–sample interaction, detector modelling, optics propagation and experimental
scan routines.  The codebase is fully modular, GPU‑accelerated where possible, and written for clarity, extensibility and
high‑throughput workflows.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Quick Start](#quick-start)
- [Detailed Modules](#detailed-modules)
  - [1. Crystal](#1-crystal-module)
  - [2. Sample](#2-sample-module)
  - [3. Beam](#3-beam-module)
  - [4. Detector](#4-detector-module)
  - [5. Stage](#5-stage-module)
  - [6. Optics](#6-optics-module)
  - [7. Defects](#7-defects-module)
  - [8. Deformation](#8-deformation-module)
  - [9. Analysis](#9-analysis-module)
  - [10. Experiment](#10-experiment-module)
- [Examples](#examples)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

**Xray‑Simulator** models the entire diffraction experiment:

1. **Crystal** – load crystallographic data (e.g. CIF) and manipulate the lattice.
2. **Sample** – replicate the crystal into a volumetric sample split into disk‑friendly *chunks*.
3. **Defects / Deformation** – apply stacking faults, cracks or arbitrary displacement fields.
4. **Beam & Optics** – define the coherent source and an optional optics stack (free space, CRLs, apertures).
5. **Stage** – position/rotate the sample relative to the beam using goniometer motors.
6. **Detector** – place a flexible 2‑D detector in laboratory space.
7. **Experiment** – orchestrate 1‑D/2‑D/3‑D scans over motor ranges.
8. **Analysis** – visualise and interrogate the resulting complex field or intensity maps.

---

## Key Features

- **GPU Acceleration** via *CuPy* (automatic CPU fallback).
- **Chunked Data Model** for unlimited sample sizes.
- **Advanced Defect & Deformation** engines (stacking faults, cracks, custom displacement fields).
- **Multi‑GPU & Pinned‑Memory** support for high‑throughput scattering.
- **Beamline Stage & Optics** simulation (goniometer, CRLs, apertures, custom components).
- **Experiment Scans**: native 1‑D/2‑D/3‑D motor scans producing HDF5/NumPy datasets.
- **Comprehensive Analysis** helpers (FFT, line/surface plots, reciprocal‑space maps).

---

> **Tip**  If you only need CPU execution, install **`cupy`** without the CUDA wheel:
> ```bash
> pip install cupy    # uses OpenMP backend
> ```

---

## Dependencies

| Package        | Purpose                                   |
| -------------- | ----------------------------------------- |
| `numpy`        | Core array maths                          |
| `cupy` *(opt)* | CUDA acceleration                         |
| `pymatgen`     | CIF reading & crystallography utilities   |
| `matplotlib`   | Plotting                                  |
| `scipy`        | Numerical geometry (e.g. `ConvexHull`)    |
| `pandas`       | Tabular handling of scattering factors    |
| `cffi`         | Compiling high‑performance C kernels      |
| `h5py` *(opt)* | Large‑scale experiment storage            |

---

## Quick Start

```python
# 1.  Crystal
from Crystal import crystal
xtal = crystal("data/Si.cif")
xtal.get_lattice_from_cif()

# 2.  Sample
from Sample import sample
samp = sample()
samp.create_sample(dimensions=[1000,1000,1000], offset=[0,0,0])
samp.generate_sample(xtal, use_gpu=False)

# 3.  Defects (optional)
from Defects import defects
dft = defects()
dft.add_stacking_faults(fault_number=1, fault_offset=[5,0,0], fault_normal=[0,0,1])
dft.stacking_faults.generate_global_positions(samp, xtal)
dft.stacking_faults.apply_to_sample(samp)

# 4.  Beam & Optics
from Beam import beam
bx = beam()
bx.create_beam(energy=8000.0, eV=True)

from Optics import optics
op = optics()
op.add_free_space(50.0)            # 50 mm drift
op.add_CRL_box(number=2, focal_length=200.0, thickness=1.0)

# 5.  Stage
from Stage import stage
stg = stage()
stg.create_stage()                 # goniometer μ–φ–χ–ω + xyz
stg.set_motor_value_absolute(['omega'], [30], degrees=True)

# 6.  Detector
from Detector import detector
det = detector()
det.create_detector(shape=(512,512), pixel_size=(1.0,1.0))
det.position_detector_absolute(distance=1000.0, two_theta=0.0, nu=0.0)

# 7.  Scattering
bx.atomic_scattering_kinematic(samp, det, stg, use_gpu=True)
det.plot_detector(type="Intensity")
```

---

## Detailed Modules

### 1 · Crystal Module
*File `Crystal.py`, class `crystal`*  
Read CIFs, convert between primitive/conventional cells, rotate/align axes and compute d‑spacings.

### 2 · Sample Module
*File `Sample.py`, class `sample`*  
Defines a 3‑D parallelepiped container, subdivided into *chunks* for scalable memory usage and GPU streaming.

### 3 · Beam Module
*File `Beam.py`, class `beam`*  
Handles source energy/wavelength, computes kinematic/dynamical scattering on CPU or multi‑GPU back ends.

### 4 · Detector Module
*File `Detector.py`, class `detector`*  
Flexible rectangular detector with per‑pixel complex field storage and plotting utilities.

### 5 · Stage Module
*File `Stage.py`, class `stage`*  
Hierarchical goniometer abstraction supporting coupled rotational and translational motors, motor resolution clipping and live axis retrieval.

### 6 · Optics Module
*File `Optics.py`, class `optics`*  
Simple ray‑parameterised optics stack (free space, compound refractive lens boxes, hard apertures, custom elements) for wave‑front propagation.

### 7 · Defects Module
*File `Defects.py`, class `defects`*  
Insert stacking faults and cracks; each represented by dedicated sub‑classes with CPU/GPU implementations.

### 8 · Deformation Module
*File `Deformation.py`, class `deformation`*  
Apply continuous displacement fields (strain, bending, thermal noise) directly to atomic positions **(experimental)**.

### 9 · Analysis Module
*File `Analysis.py`, class `analysis`*  
Post‑processing helpers for line/surface plots, FFTs, detector distance sweeps and reciprocal‑space analysis.

### 10 · Experiment Module
*File `Experiment.py`, class `experiment`*  
Declarative scan engine generating 1‑D/2‑D/3‑D motor trajectories, persisting metadata, and automating scattering/analysis loops.

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