# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import os
import gc
import json
from Logging import logging
try:
    import cupy as cp
except ImportError:
    cp = None
from cffi import FFI
import threading

# -----------------------------------------------------------------------------
# Multi-GPU Configuration (environment variables)
# -----------------------------------------------------------------------------
SAMPLE_STREAMS_PER_GPU = int(os.environ.get("SAMPLE_STREAMS_PER_GPU", "4"))
SAMPLE_WRITER_THREADS = int(os.environ.get("SAMPLE_WRITER_THREADS", "3"))


def _get_gpu_count():
    """Return number of available CUDA GPUs, or 0 if none."""
    try:
        return cp.cuda.runtime.getDeviceCount() if cp is not None else 0
    except Exception:
        return 0

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class sample(logging):
    
    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "generate_sample_single",
        "generate_sample_poly",
        "import_atomic_data",
        "create_sample",
        "read_sample_metadata",
        "write_sample_metadata",
        "zero_sample",
        "zero_sample_position",
        "zero_sample_rotation",
        "rotate_sample_relative",
        "translate_sample_relative",
        "plot_sample",
        "plot_sample_exterior",
        "plot_grains",
        "build_cell_list_gpu",
    )
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=os.getcwd()):
        """
        Initialize core state and ensure the working directory exists.

        Compiles the CFFI intersection routine, sets default attributes for the
        sample, and guarantees that the target directory is present on disk.

        Args:
            directory (str, optional): Directory where chunk files and metadata
                will be read from and written to. Defaults to the current
                working directory.

        Note:
            Geometry and data are not created here. Use `create_sample`,
            `import_atomic_data`, or `generate_sample_single` to populate files.
        """
        super().__init__(log_name="sample")
        self.directory = directory
        self._dimensions = None
        self._offset = None
        self._rotation = None
        self._chunk_volume = None
        self._chunk_total = None 
        self._matrix = None
        self._corners = None
        self._sample_type = None
        self._grain_seeds = None
        self._grain_orientations = None
        self._grain_count = None

        # Temperature/displacement configuration (disabled by default)
        self.enable_temp = False
        self.temp_params = ['gaussian', 0.25, 1, 40]

        # Thermal expansion configuration (disabled by default)
        self._thermal_expansion_enabled = False
        self._thermal_expansion_alpha = None       # Isotropic: single float (1/K)
        self._thermal_expansion_alpha_xyz = None   # Anisotropic: [αx, αy, αz] (1/K)
        self._thermal_expansion_T_ref = 300.0      # Reference temperature (K)

        # Streaming mode configuration (disabled by default)
        # When enabled, chunks are generated on-demand during simulation
        self._streaming_mode = False
        self._streaming_material = None  # Store material reference for on-demand generation
        self._streaming_flush_size = None  # flush_size for virtual chunking
        self._streaming_geom_atom_counts = None  # atom counts per geometric chunk
        self._streaming_file_chunk_ranges = None  # mapping from file chunks to geometric chunks
        self._streaming_use_gpu = False  # GPU preference from generate_sample_* call

        # Alloy mode configuration (disabled by default)
        # When enabled, atom species are randomly assigned from a user-provided list
        self._alloy_species = None          # list of element symbols, e.g. ["Fe", "Co"]
        self._alloy_concentrations = None   # list of probabilities, e.g. [0.5, 0.5]
        self._alloy_rng = None              # np.random.Generator for reproducibility
        self._alloy_lock = None             # threading.Lock for multi-GPU safety

        # Streaming GPU cache (None when not initialized)
        # These are uploaded to GPU once and reused across chunk generation calls
        self._streaming_gpu_seeds_cp = None       # (G, 3) CuPy array of grain seeds
        self._streaming_gpu_rotations_cp = None   # (G, 3, 3) CuPy array of rotation matrices
        self._streaming_gpu_lattice_cp = None     # Pre-uploaded lattice for get_atomic_data
        self._streaming_gpu_offset_cp = None      # Pre-uploaded offset
        self._streaming_gpu_dim_half_cp = None    # Pre-uploaded dimensions/2

        # Default file basenames for chunked outputs
        self._default_filenames = np.array([
            "atomic_positions.npy",
            "atomic_species.npy",
            "sample_metadata.npy"
        ])  # sample_metadata will be a struct

        # Compile the SAT-based intersection function once for reuse
        self._ffi_object, self._intersect_function = self.compile_parallelepipeds_intersect_batch_cffi()

        # Ensure the working directory exists
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
    def create_sample(self, dimensions, offset=[0, 0, 0], chunk_volume=12500000, sample_type="single", streaming=False):
        """
        Create an axis-aligned sample box and precompute helpers.

        Sets dimensions, offset, an identity rotation, and a diagonal matrix
        representation. Also computes the 8 sample corners centered about
        `offset` for downstream geometric operations.

        Args:
            dimensions (array-like of float): Length-3 iterable giving box
                lengths along x, y, z.
            offset (array-like of float, optional): Length-3 center offset of
                the sample in the same units as `dimensions`. Defaults to
                [0, 0, 0].
            chunk_volume (int or float, optional): Target approximate number of
                sites per output chunk. This is stored and later used to pick
                chunking in `get_chunk_positions`/`generate_sample_single`. Defaults to
                12_500_000.
            sample_type (str, optional): "single" (default) or "poly". Controls
                whether subsequent generation is single crystal or polycrystal.
            streaming (bool, optional): If True, enables streaming mode where
                chunks are generated on-demand during simulation rather than
                persisted to disk. This allows samples larger than disk capacity.
                Defaults to False.

        Returns:
            None
        """
        # Clear any existing GPU cache from previous streaming session
        if hasattr(self, '_streaming_gpu_seeds_cp') and self._streaming_gpu_seeds_cp is not None:
            self._clear_streaming_gpu_cache()

        # Cache numeric forms in single precision for consistency
        self._dimensions = np.array(dimensions, dtype=np.float32)
        self._offset = np.array(offset, dtype=np.float32)

        # Start from no rotation; store chunk_volume as a scalar
        self._rotation = np.eye(3, dtype=np.float32)
        self._chunk_volume = np.array(chunk_volume, dtype=np.float32)

        # Build diagonal matrix and precompute corners in sample frame
        self._matrix = np.diag(self.dimensions)

        # Corners are unit-cube corners scaled by dimensions and shifted by offset
        # Slightly rewritten for small overhead reduction (no functional change)
        self._corners = (self.get_unit_corners() @ self.matrix) - (self.dimensions * 0.5) + self.offset

        # Sample type and poly state
        self._sample_type = sample_type

        # Streaming mode configuration
        self._streaming_mode = bool(streaming)
        if streaming:
            self._streaming_material = None  # Will be set during generate_*
            
    def read_sample_metadata(self):
        """
        Load JSON metadata from disk and restore core state.

        Reads `sample_metadata.json` from `self.directory` (or from a provided
        override path in the writer) and restores `_dimensions`, `_offset`,
        `_rotation`, `_chunk_total`, and `_sample_type` if present.

        Raises:
            FileNotFoundError: If the JSON metadata file does not exist.

        Returns:
            None
        """
        # Compose metadata path and validate its existence
        metadata_filename = os.path.join(self.directory, "sample_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        # Parse JSON and restore internal arrays as NumPy types
        with open(metadata_filename, "r") as f:
            sample_metadata = json.load(f)

        # Convert lists back to NumPy arrays where applicable
        if sample_metadata["dimensions"] is not None:
            self._dimensions = np.array(sample_metadata["dimensions"], dtype=np.float32)
        if sample_metadata["offset"] is not None:
            self._offset = np.array(sample_metadata["offset"], dtype=np.float32)
        if sample_metadata["rotation"] is not None:
            self._rotation = np.array(sample_metadata["rotation"], dtype=np.float32)
        if sample_metadata["chunk_total"] is not None:
            self._chunk_total = int(sample_metadata["chunk_total"])
        if sample_metadata["sample_type"] is not None:
            self._sample_type = sample_metadata["sample_type"]
        if sample_metadata.get("alloy_species") is not None:
            self._alloy_species = sample_metadata["alloy_species"]
        if sample_metadata.get("alloy_concentrations") is not None:
            self._alloy_concentrations = sample_metadata["alloy_concentrations"]

    ## Alloy helpers
    # -------------------------------------
    def _setup_alloy(self, alloy_species, alloy_concentrations, alloy_seed):
        """
        Validate and store alloy parameters for the current generation run.

        Args:
            alloy_species (list[str] | None): Element symbols for random
                assignment. None disables alloy mode.
            alloy_concentrations (list[float] | None): Per-species probabilities.
                Must sum to 1. None uses equal probabilities.
            alloy_seed (int | None): RNG seed for reproducibility.
        """
        if alloy_species is not None:
            alloy_species = list(alloy_species)
            if alloy_concentrations is None:
                n = len(alloy_species)
                alloy_concentrations = [1.0 / n] * n
            else:
                alloy_concentrations = list(alloy_concentrations)
                if len(alloy_concentrations) != len(alloy_species):
                    raise ValueError(
                        f"alloy_concentrations length ({len(alloy_concentrations)}) "
                        f"must match alloy_species length ({len(alloy_species)})"
                    )
                if abs(sum(alloy_concentrations) - 1.0) > 1e-6:
                    raise ValueError(
                        f"alloy_concentrations must sum to 1.0, got {sum(alloy_concentrations):.6f}"
                    )
            self._alloy_species = alloy_species
            self._alloy_concentrations = alloy_concentrations
            self._alloy_rng = np.random.default_rng(alloy_seed)
            self._alloy_lock = threading.Lock()
        else:
            self._alloy_species = None
            self._alloy_concentrations = None
            self._alloy_rng = None
            self._alloy_lock = None

    def _teardown_alloy(self):
        """Clear alloy state after generation is complete."""
        self._alloy_rng = None
        self._alloy_lock = None

    def _build_species(self, base_species, n_tiles, mask=None):
        """
        Build a species array, optionally randomizing for alloy mode.

        Tiles ``base_species`` by ``n_tiles``, applies ``mask`` if given,
        then (if alloy mode is active) replaces every element with a random
        draw from ``_alloy_species`` weighted by ``_alloy_concentrations``.

        Args:
            base_species (array-like): Per-unit-cell species labels.
            n_tiles (int): Number of unit-cell repeats.
            mask (np.ndarray | None): Boolean mask to select valid atoms.

        Returns:
            np.ndarray: 1-D array of species strings.
        """
        spc = np.tile(base_species, n_tiles)
        if mask is not None:
            spc = spc[mask]
        if self._alloy_rng is not None:
            with self._alloy_lock:
                spc = np.array(
                    self._alloy_rng.choice(
                        self._alloy_species,
                        size=len(spc),
                        p=self._alloy_concentrations,
                    )
                )
        return spc

    ## Data Handling Functions
    # -------------------------------------
    # Generate sample
    def write_chunk_positions(self, data, chunk_num, override_directory=None):
        """
        Write a positions array for a specific chunk to disk.

        Saves a (N, 3) positions array as `atomic_positions_<chunk_num>.npy`
        either under `self.directory` or `override_directory` if provided.

        Args:
            data (np.ndarray): Array of shape (N, 3) with atomic positions.
            chunk_num (int): 1-based chunk index used in the output filename.
            override_directory (str, optional): Alternate directory root for
                output. If None, uses `self.directory`.

        Returns:
            None
        """
        # Compose the chunked filename based on the default basename
        base, ext = os.path.splitext(self._default_filenames[0])
        chunk_filename = f"{base}_{chunk_num}{ext}"

        # Persist the array in the appropriate directory
        if override_directory is not None:
            np.save(os.path.join(override_directory, chunk_filename), data)
        else:
            np.save(os.path.join(self.directory, chunk_filename), data)
    
    def write_chunk_species(self, data, chunk_num, override_directory=None):
        """
        Write a species array for a specific chunk to disk.

        Saves a 1-D species array as `atomic_species_<chunk_num>.npy` either
        under `self.directory` or `override_directory` if provided.

        Args:
            data (array-like): 1-D array of species labels or ids with length N.
            chunk_num (int): 1-based chunk index used in the output filename.
            override_directory (str, optional): Alternate directory root for
                output. If None, uses `self.directory`.

        Returns:
            None
        """
        # Compose the chunked filename based on the default basename
        base, ext = os.path.splitext(self._default_filenames[1])
        chunk_filename = f"{base}_{chunk_num}{ext}"

        # Persist the array in the appropriate directory
        if override_directory is not None:
            np.save(os.path.join(override_directory, chunk_filename), data)
        else:
            np.save(os.path.join(self.directory, chunk_filename), data)
            
    def write_sample_metadata(self, override_directory=None):
        """
        Serialize critical fields to a JSON metadata file on disk.

        Writes `sample_metadata.json` containing `dimensions`, `offset`,
        `rotation`, `chunk_total`, and `sample_type`.

        Args:
            override_directory (str, optional): If provided, write the JSON to
                this directory instead of `self.directory`.

        Returns:
            None
        """
        # Convert NumPy arrays to Python lists so JSON can handle them
        sample_metadata = {
            "dimensions": self._dimensions.tolist() if self._dimensions is not None else None,
            "offset": self._offset.tolist() if self._offset is not None else None,
            "rotation": self._rotation.tolist() if self._rotation is not None else None,
            "chunk_total": int(self._chunk_total) if self._chunk_total is not None else None,
            "sample_type": self._sample_type if self._sample_type is not None else None,
            "alloy_species": self._alloy_species if self._alloy_species is not None else None,
            "alloy_concentrations": self._alloy_concentrations if self._alloy_concentrations is not None else None,
        }

        # Choose the output directory and filename
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "sample_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "sample_metadata.json")

        # Write a nicely formatted JSON file for human inspection and versioning
        with open(metadata_filename, "w") as f:
            json.dump(sample_metadata, f, indent=4)
        print(f"Metadata written to {metadata_filename} in JSON format.")
    
    def load_chunk_positions(self, chunk_number, use_gpu=True):
        """
        Load a chunk's positions from disk, optionally on GPU.

        If `use_gpu` is True and CuPy is available, returns a `cp.ndarray`.
        Otherwise, returns an `np.ndarray`. If `self.enable_temp` is True,
        temperature-based displacements are applied via `apply_temperature`.

        In streaming mode, positions are generated on-demand instead of being
        loaded from disk.

        Args:
            chunk_number (int): 1-based chunk index to load.
            use_gpu (bool, optional): If True and CuPy is available, load using
                `cp.load` and return a GPU array. Defaults to True.

        Returns:
            np.ndarray or cp.ndarray: Array of shape (N, 3) containing positions.

        Raises:
            FileNotFoundError: If the chunk file does not exist (raised by the
                underlying loader).
            ValueError: If temperature application is enabled and an unknown
                distribution was configured.
            RuntimeError: In streaming mode, if material reference is not set.
        """
        # STREAMING MODE: generate on-demand instead of loading from disk
        if getattr(self, "_streaming_mode", False):
            return self._generate_chunk_on_demand(chunk_number, use_gpu, return_positions=True)

        # Compose the on-disk filename for this chunk
        base, ext = os.path.splitext(self._default_filenames[0])
        positions_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, positions_filename)

        # Load on GPU if requested and available; else load on CPU
        if use_gpu and (cp is not None):
            positions = cp.load(full_path)
        else:
            positions = np.load(full_path)

        # Optionally apply thermal effects according to configured model
        if self.enable_temp is True:
            # Determine temperature from temp_params (index 1 is sigma/T_K)
            # For gaussian mode, use reference temperature for expansion calculation
            distribution = self.temp_params[0]
            if distribution.lower() in ('gaussian', 'normal'):
                T_K = getattr(self, '_thermal_expansion_T_ref', 300.0)
            else:
                T_K = float(self.temp_params[1])

            # Apply thermal expansion first (scales equilibrium positions)
            if getattr(self, '_thermal_expansion_enabled', False):
                positions = self.apply_thermal_expansion(positions, T_K)

            # Apply thermal vibrations (random displacements around scaled positions)
            # Note: 'sigma' now means 'temperature_K' when distribution='einstein' or 'debye'
            positions = self.apply_temperature(
                positions,
                distribution=distribution,
                sigma=self.temp_params[1],
                max_displacement=self.temp_params[2],
                seed=self.temp_params[3],
                chunk_number=chunk_number  # enables per-species masses if configured
            )
        return positions

    def load_chunk_species(self, chunk_number, use_gpu=True):
        """
        Load a chunk's species array from disk, optionally on GPU.

        If `use_gpu` is True and CuPy is available, returns a `cp.ndarray`.
        Otherwise, returns an `np.ndarray`.

        In streaming mode, species are generated on-demand instead of being
        loaded from disk.

        Args:
            chunk_number (int): 1-based chunk index to load.
            use_gpu (bool, optional): If True and CuPy is available, load using
                `cp.load` and return a GPU array. Defaults to True.

        Returns:
            np.ndarray or cp.ndarray: 1-D species array corresponding to the
            positions in the same chunk.

        Raises:
            FileNotFoundError: If the chunk file does not exist (raised by the
                underlying loader).
            RuntimeError: In streaming mode, if material reference is not set.
        """
        # STREAMING MODE: generate on-demand instead of loading from disk
        if getattr(self, "_streaming_mode", False):
            return self._generate_chunk_on_demand(chunk_number, use_gpu, return_positions=False)

        # Compose the on-disk filename for this chunk
        base, ext = os.path.splitext(self._default_filenames[1])
        species_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, species_filename)

        # Load on GPU if requested and available; else load on CPU
        if use_gpu and (cp is not None):
            return cp.load(full_path)
        else:
            return np.load(full_path)

    # -------------------------------------
    # Streaming mode helpers
    # -------------------------------------
    def _get_chunk_atom_count(self, material, chunk_position, chunk_dimensions):
        """
        Count atoms in a geometric chunk without storing positions.

        Uses the same logic as get_atomic_data but only returns the count,
        which is faster for computing chunk mappings.

        Args:
            material: Crystal material object.
            chunk_position: (3,) array of chunk origin.
            chunk_dimensions: (3,) array of chunk size in lattice units.

        Returns:
            int: Number of atoms in this geometric chunk.
        """
        # Get lattice positions
        lattice_positions_np = self.get_lattice_positions(
            material, chunk_position, chunk_dimensions, use_gpu=False
        )
        n_lattice = lattice_positions_np.shape[0]
        if n_lattice == 0:
            return 0

        n_atoms_per_cell = len(material.species)
        lattice_atom_cartesian_np = material.lattice_atom_cartesian.astype(np.float32)

        # Expand to atom sites
        atomic_positions_S = (
            lattice_positions_np[:, np.newaxis, :].astype(np.float32) +
            lattice_atom_cartesian_np[np.newaxis, :, :]
        ).reshape(-1, 3)

        # In-box mask
        mask = (
            (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= self.dimensions[0]) &
            (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= self.dimensions[1]) &
            (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= self.dimensions[2])
        )

        return int(np.sum(mask))

    def _compute_streaming_chunk_mapping(self, material, flush_size):
        """
        Pre-compute mapping from file chunks to geometric chunks.

        This allows streaming mode to produce the same number of chunks
        as non-streaming mode (which accumulates atoms up to flush_size).

        Args:
            material: Crystal material object.
            flush_size (int): Number of atoms per file chunk.
        """
        num_geom = self._chunk_positions.shape[0]
        flush_size = int(flush_size)

        # Count atoms in each geometric chunk
        geom_counts = []
        for i in range(num_geom):
            count = self._get_chunk_atom_count(
                material, self._chunk_positions[i], self._chunk_dimensions
            )
            geom_counts.append(count)

        self._streaming_geom_atom_counts = np.array(geom_counts, dtype=np.int64)
        self._streaming_flush_size = flush_size

        # Build file chunk ranges
        # Each range is (start_geom_idx, end_geom_idx, start_offset, end_offset)
        # where offsets are within the respective geometric chunks
        ranges = []
        current_file_chunk_start_geom = 0
        current_file_chunk_start_offset = 0
        accumulated = 0

        for i, count in enumerate(geom_counts):
            accumulated += count

            while accumulated >= flush_size:
                # This file chunk ends somewhere in geometric chunk i
                overflow = accumulated - flush_size
                end_offset = count - overflow

                ranges.append((
                    current_file_chunk_start_geom,
                    i,
                    current_file_chunk_start_offset,
                    end_offset
                ))

                # Start next file chunk from where we left off in geometric chunk i
                current_file_chunk_start_geom = i
                current_file_chunk_start_offset = end_offset
                accumulated = overflow

        # Handle tail (remaining atoms that don't fill a full flush_size)
        if accumulated > 0:
            last_geom_idx = num_geom - 1
            ranges.append((
                current_file_chunk_start_geom,
                last_geom_idx,
                current_file_chunk_start_offset,
                geom_counts[last_geom_idx] if last_geom_idx >= 0 else 0
            ))

        self._streaming_file_chunk_ranges = ranges
        self._chunk_total = len(ranges)

        self._log("normal", f"[sample] Streaming mode: {num_geom} geometric chunks mapped to {len(ranges)} file chunks")

    def _generate_geometric_chunk(self, geom_idx, use_gpu=True):
        """
        Generate atoms for a single geometric chunk.

        Dispatches to single-crystal or polycrystal generation as appropriate.
        Uses GPU acceleration when available for better performance.

        Args:
            geom_idx (int): 0-based geometric chunk index.
            use_gpu (bool): Whether to use GPU acceleration. Defaults to True.

        Returns:
            tuple: (positions, species) NumPy arrays for this geometric chunk.
        """
        material = self._streaming_material
        chunk_pos = self._chunk_positions[geom_idx]
        chunk_dims = self._chunk_dimensions

        if self._sample_type == "poly":
            return self._generate_poly_geometric_chunk(geom_idx, use_gpu=use_gpu)
        else:
            # Single crystal: pass use_gpu to get_atomic_data
            # Note: get_atomic_data returns numpy arrays when use_gpu=True but return_on_gpu=False (default)
            return self.get_atomic_data(material, chunk_pos, chunk_dims, use_gpu=use_gpu)

    def _generate_poly_geometric_chunk(self, geom_idx, use_gpu=True):
        """
        Generate polycrystal atoms for a single geometric chunk.

        Applies Voronoi grain assignment and per-grain rotations.
        Uses GPU acceleration when available for better performance.

        Args:
            geom_idx (int): 0-based geometric chunk index.
            use_gpu (bool): Whether to use GPU acceleration. Defaults to True.

        Returns:
            tuple: (positions, species) NumPy arrays for this geometric chunk.
        """
        material = self._streaming_material
        chunk_pos = self._chunk_positions[geom_idx]
        chunk_dims = self._chunk_dimensions

        # Determine if GPU path is available
        gpu_available = (use_gpu and cp is not None)
        gpu_count = 0
        if gpu_available:
            try:
                gpu_count = int(cp.cuda.runtime.getDeviceCount())
                gpu_available = (gpu_count > 0)
            except Exception as e:
                self._log("debug", f"[sample] GPU detection failed: {type(e).__name__}: {e}")
                gpu_available = False

        # Try GPU path first
        if gpu_available:
            try:
                return self._generate_poly_geometric_chunk_gpu(material, chunk_pos, chunk_dims)
            except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e:
                # GPU memory or runtime error -> fall through to CPU path
                self._log("normal", f"[sample] GPU streaming chunk generation failed (OOM/runtime), falling back to CPU: {e}")
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            except Exception as e:
                # Unexpected error - log it and re-raise to help debugging
                self._log("normal", f"[sample] GPU streaming chunk generation failed with unexpected error: {type(e).__name__}: {e}")
                raise

        # CPU fallback path
        return self._generate_poly_geometric_chunk_cpu(material, chunk_pos, chunk_dims)

    def _generate_poly_geometric_chunk_gpu(self, material, chunk_pos, chunk_dims):
        """
        GPU-accelerated polycrystal chunk generation.

        Internal helper for _generate_poly_geometric_chunk.

        Args:
            material: Material object with lattice data.
            chunk_pos: Chunk position array.
            chunk_dims: Chunk dimensions array.

        Returns:
            tuple: (positions, species) NumPy arrays.

        Raises:
            CuPy exceptions on GPU errors (caught by caller).
        """
        # Initialize GPU cache if needed
        if self._streaming_gpu_seeds_cp is None:
            if not self._init_streaming_gpu_cache():
                raise RuntimeError("Failed to initialize GPU cache")

        seeds_cp = self._streaming_gpu_seeds_cp
        R_cp = self._streaming_gpu_rotations_cp

        # Generate atoms on GPU
        pos_cp, mask_cp, site_count = self.get_atomic_data(
            material,
            chunk_pos,
            chunk_dims,
            use_gpu=True,
            return_on_gpu=True,
            lattice_atom_cartesian_cp=self._streaming_gpu_lattice_cp,
            offset_gpu=self._streaming_gpu_offset_cp,
            dim_half_gpu=self._streaming_gpu_dim_half_cp
        )

        if pos_cp.size == 0:
            return np.zeros((0, 3), dtype=np.float32), np.array([], dtype=object)

        # GPU Voronoi assignment (memory-safe streaming)
        grain_labels = self._voronoi_assign_gpu_streaming(pos_cp, seeds_cp)

        # Species array (on CPU for memory efficiency, matching non-streaming path)
        mask_np = mask_cp.get()
        spc_sample = self._build_species(material.species, site_count, mask_np)

        # Batched per-grain rotation using einsum
        # R_per_atom[i] = R_cp[grain_labels[i]] for each atom i
        R_per_atom = R_cp[grain_labels]  # (N, 3, 3)
        pos_rotated = cp.einsum('nij,nj->ni', R_per_atom, pos_cp)

        # Transfer results to CPU
        pos_np = pos_rotated.get().astype(np.float32)

        # Cleanup per-chunk GPU memory (keep invariant cache)
        del pos_cp, mask_cp, grain_labels, R_per_atom, pos_rotated
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

        return pos_np, spc_sample

    def _generate_poly_geometric_chunk_cpu(self, material, chunk_pos, chunk_dims):
        """
        CPU fallback for polycrystal chunk generation.

        Internal helper for _generate_poly_geometric_chunk.

        Args:
            material: Material object with lattice data.
            chunk_pos: Chunk position array.
            chunk_dims: Chunk dimensions array.

        Returns:
            tuple: (positions, species) NumPy arrays.
        """
        seeds_np = np.asarray(self._grain_seeds, dtype=np.float32)
        R = np.asarray(self._grain_orientations, dtype=np.float32)
        G = int(seeds_np.shape[0])

        # Generate base atoms (CPU path)
        pos_np, spc_np = self.get_atomic_data(material, chunk_pos, chunk_dims, use_gpu=False)

        if pos_np.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32), np.array([], dtype=spc_np.dtype if spc_np.size > 0 else object)

        # Voronoi assignment (CPU method)
        grain_labels = self._voronoi_min_index_cpu(pos_np, seeds_np)

        # Apply per-grain rotations
        pos_parts = []
        spc_parts = []
        for g in range(G):
            mask_g = (grain_labels == g)
            if not np.any(mask_g):
                continue
            pos_g = (pos_np[mask_g, :] @ R[g].T).astype(np.float32)
            pos_parts.append(pos_g)
            spc_parts.append(spc_np[mask_g])

        if pos_parts:
            return np.concatenate(pos_parts, axis=0), np.concatenate(spc_parts, axis=0)
        return np.zeros((0, 3), dtype=np.float32), np.array([], dtype=spc_np.dtype if spc_np.size > 0 else object)

    def _init_streaming_gpu_cache(self):
        """
        Initialize GPU cache for streaming mode invariants.

        Uploads seeds, rotation matrices, and get_atomic_data invariants
        to GPU once for reuse across all chunk generations.

        Returns:
            bool: True if GPU cache initialized successfully, False otherwise.
        """
        if cp is None:
            return False

        try:
            # Check GPU availability
            if int(cp.cuda.runtime.getDeviceCount()) < 1:
                return False

            material = self._streaming_material
            if material is None:
                return False

            # Upload invariants to GPU
            if self._grain_seeds is not None:
                self._streaming_gpu_seeds_cp = cp.asarray(self._grain_seeds, dtype=cp.float32)
            if self._grain_orientations is not None:
                self._streaming_gpu_rotations_cp = cp.asarray(self._grain_orientations, dtype=cp.float32)
            self._streaming_gpu_lattice_cp = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
            self._streaming_gpu_offset_cp = cp.asarray(self.offset, dtype=cp.float32)
            self._streaming_gpu_dim_half_cp = cp.asarray(self.dimensions * 0.5, dtype=cp.float32)

            return True

        except Exception:
            self._clear_streaming_gpu_cache()
            return False

    def _clear_streaming_gpu_cache(self):
        """
        Clear GPU cache for streaming mode invariants.

        Frees GPU memory used by cached seeds, rotations, and other invariants.
        Safe to call even if cache was never initialized.
        """
        self._streaming_gpu_seeds_cp = None
        self._streaming_gpu_rotations_cp = None
        self._streaming_gpu_lattice_cp = None
        self._streaming_gpu_offset_cp = None
        self._streaming_gpu_dim_half_cp = None

        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

    def _generate_chunk_on_demand(self, chunk_number, use_gpu=True, return_positions=True):
        """
        Generate a file chunk on-demand for streaming mode.

        Uses the pre-computed mapping from file chunks to geometric chunks to
        generate atoms that match what non-streaming mode would produce.

        Args:
            chunk_number (int): 1-based file chunk index.
            use_gpu (bool): Whether to use GPU and/or return GPU arrays.
            return_positions (bool): If True, return positions; if False, return species.

        Returns:
            Positions array (N, 3) or species array (N,) depending on return_positions.

        Raises:
            RuntimeError: If streaming mode is active but material reference is not set.
            ValueError: If chunk_number is out of range.
        """
        if self._streaming_material is None:
            raise RuntimeError("Streaming mode requires material; call generate_sample_single/poly first")

        if self._streaming_file_chunk_ranges is None:
            raise RuntimeError("Streaming chunk mapping not computed; call generate_sample_single/poly first")

        file_chunk_idx = chunk_number - 1
        if file_chunk_idx < 0 or file_chunk_idx >= len(self._streaming_file_chunk_ranges):
            raise ValueError(f"chunk_number {chunk_number} out of range [1, {len(self._streaming_file_chunk_ranges)}]")

        # Get the mapping for this file chunk
        start_geom, end_geom, start_offset, end_offset = self._streaming_file_chunk_ranges[file_chunk_idx]

        # In streaming mode, use the GPU setting from sample generation, not from caller
        effective_use_gpu = getattr(self, '_streaming_use_gpu', use_gpu)

        # Generate atoms from all contributing geometric chunks
        all_pos = []
        all_spc = []

        for geom_idx in range(start_geom, end_geom + 1):
            pos, spc = self._generate_geometric_chunk(geom_idx, use_gpu=effective_use_gpu)

            if pos.shape[0] == 0:
                continue

            # Apply slicing for boundary geometric chunks
            if geom_idx == start_geom and geom_idx == end_geom:
                # This file chunk is entirely within one geometric chunk
                pos = pos[start_offset:end_offset]
                spc = spc[start_offset:end_offset]
            elif geom_idx == start_geom:
                # First geometric chunk: take from start_offset to end
                pos = pos[start_offset:]
                spc = spc[start_offset:]
            elif geom_idx == end_geom:
                # Last geometric chunk: take from beginning to end_offset
                pos = pos[:end_offset]
                spc = spc[:end_offset]
            # Middle geometric chunks: take all atoms (no slicing needed)

            if pos.shape[0] > 0:
                all_pos.append(pos)
                all_spc.append(spc)

        # Concatenate results
        if all_pos:
            positions = np.concatenate(all_pos, axis=0)
            species = np.concatenate(all_spc, axis=0)
        else:
            positions = np.zeros((0, 3), dtype=np.float32)
            species = np.array([], dtype=object)

        # Apply temperature effects if enabled (same logic as load_chunk_positions)
        if return_positions and self.enable_temp and positions.shape[0] > 0:
            distribution = self.temp_params[0]
            if distribution.lower() in ('gaussian', 'normal'):
                T_K = getattr(self, '_thermal_expansion_T_ref', 300.0)
            else:
                T_K = float(self.temp_params[1])

            if getattr(self, '_thermal_expansion_enabled', False):
                positions = self.apply_thermal_expansion(positions, T_K)

            positions = self.apply_temperature(
                positions,
                distribution=distribution,
                sigma=self.temp_params[1],
                max_displacement=self.temp_params[2],
                seed=self.temp_params[3],
                chunk_number=chunk_number
            )

        result = positions if return_positions else species

        # Convert to GPU if requested
        if use_gpu and (cp is not None):
            return cp.asarray(result)
        return result

    # -------------------------------------

    # -------------------------------------
    # KNN search
    def write_chunk_nn_indices(self, index_list, chunk_num, override_directory=None):
        """
        Write neighbor index lists for a chunk to a compact NPZ.

        Produces ``nearest_neighbors_indices_<chunk_num>.npz`` containing:
        - ``flat_idx``: concatenated neighbor indices for all atoms.
        - ``offsets``: start positions for each atom's neighbor list
          (length n_atoms + 1; offsets[i+1] - offsets[i] = neighbors of atom i).

        Args:
            index_list (list[np.ndarray]): Ragged list where ``index_list[i]`` is
                a 1-D integer array of neighbor indices for atom ``i``.
            chunk_num (int): 1-based chunk number used in the output filename.
            override_directory (str | None): If provided, write to this directory
                instead of ``self.directory``.

        Returns:
            None
        """
        # Compose output path
        base_name = "nearest_neighbors_indices"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        # Number of atoms equals number of sub-arrays
        n_atoms = len(index_list)

        # Build lengths and offsets to delimit each atom's sub-array
        lengths = [arr.size for arr in index_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        # Flatten the ragged structure into a single array
        if n_atoms > 0:
            flat_idx = np.concatenate(index_list)
        else:
            # Empty case: produce valid, empty arrays
            flat_idx = np.zeros(0, dtype=np.int32)

        # Save compressed representation
        np.savez(save_path, flat_idx=flat_idx, offsets=offsets)
        

    def write_chunk_nn_phase(self, phase_list, chunk_num, override_directory=None):
        """
        Write neighbor phases for a chunk to a compact NPZ.

        Produces ``nearest_neighbors_phase_<chunk_num>.npz`` containing:
        - ``flat_phase``: concatenated float phases for all atoms' neighbors.
        - ``offsets``: start positions for each atom's neighbor list.

        Args:
            phase_list (list[np.ndarray]): Ragged list where ``phase_list[i]`` is
                a 1-D float array (float32 recommended) of phases for atom ``i``.
            chunk_num (int): 1-based chunk number used in the output filename.
            override_directory (str | None): If provided, write to this directory
                instead of ``self.directory``.

        Returns:
            None
        """
        # Compose output path
        base_name = "nearest_neighbors_phase"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(phase_list)

        # Offsets delimit each atom's sub-array inside the flattened vector
        lengths = [arr.size for arr in phase_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        # Flatten ragged list
        if n_atoms > 0:
            flat_phase = np.concatenate(phase_list)
        else:
            flat_phase = np.zeros(0, dtype=np.float32)

        # Persist to disk
        np.savez(save_path, flat_phase=flat_phase, offsets=offsets)

    def write_chunk_nn_scatter(self, scatter_list, chunk_num, override_directory=None):
        """
        Write neighbor wavevectors for a chunk to a compact NPZ.

        Each element of ``scatter_list`` has shape ``(N_i, 3)`` with columns
        ``[kx, ky, kz]``. This function writes:
        - ``flat_kx``, ``flat_ky``, ``flat_kz``: concatenated components.
        - ``offsets``: start indices per atom so the ragged lists can be rebuilt.

        Args:
            scatter_list (list[np.ndarray]): Ragged list where ``scatter_list[i]``
                has shape ``(N_i, 3)`` containing neighbor wavevectors for atom ``i``.
            chunk_num (int): 1-based chunk number used in the output filename.
            override_directory (str | None): If provided, write to this directory
                instead of ``self.directory``.

        Returns:
            None
        """
        # Compose output path
        base_name = "nearest_neighbors_scatter"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(scatter_list)

        # Number of neighbors per atom and offsets into the flattened arrays
        lengths = [arr.shape[0] for arr in scatter_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        # Split and flatten kx, ky, kz components
        if n_atoms > 0:
            flat_kx = np.concatenate([arr[:, 0] for arr in scatter_list])
            flat_ky = np.concatenate([arr[:, 1] for arr in scatter_list])
            flat_kz = np.concatenate([arr[:, 2] for arr in scatter_list])
        else:
            flat_kx = np.zeros(0, dtype=np.float32)
            flat_ky = np.zeros(0, dtype=np.float32)
            flat_kz = np.zeros(0, dtype=np.float32)

        # Persist to disk
        np.savez(save_path, flat_kx=flat_kx, flat_ky=flat_ky, flat_kz=flat_kz, offsets=offsets)
        
    def write_chunk_nn_species(self, species_list, chunk_num, override_directory=None):
        """
        Write neighbor species for a chunk to a compact NPZ.

        Produces ``nearest_neighbors_species_<chunk_num>.npz`` with:
        - ``flat_species``: concatenated neighbor species values.
        - ``offsets``: start positions per atom (length n_atoms + 1).

        Note:
            ``flat_species`` dtype may be numeric or string depending on input.
            For maximum portability, prefer fixed-length dtypes over object arrays.

        Args:
            species_list (list[np.ndarray]): Ragged list where ``species_list[i]``
                is a 1-D array of species values for atom ``i``.
            chunk_num (int): 1-based chunk number used in the output filename.
            override_directory (str | None): If provided, write to this directory
                instead of ``self.directory``.

        Returns:
            None
        """
        # Compose output path
        base_name = "nearest_neighbors_species"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(species_list)

        # Offsets delimit each atom's species slice inside the flattened vector
        lengths = [arr.size for arr in species_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        # Concatenate ragged species values
        if n_atoms > 0:
            flat_species = np.concatenate(species_list)
        else:
            # Empty case: choose a safe numeric dtype
            flat_species = np.array([], dtype=species_list[0].dtype if n_atoms > 0 else np.int32)

        # Persist to disk
        np.savez(save_path, flat_species=flat_species, offsets=offsets)

    def write_chunk_nn_dist(self, dist_list, chunk_num, override_directory=None):
        """
        Write neighbor distances for a chunk to a compact NPZ.

        Produces ``nearest_neighbors_dist_<chunk_num>.npz`` containing:
        - ``flat_dist``: concatenated float distances (Angstrom) for all atoms' neighbors.
        - ``offsets``: start positions for each atom's neighbor list.

        Args:
            dist_list (list[np.ndarray]): Ragged list where ``dist_list[i]`` is
                a 1-D float array (float32) of distances in Angstrom for atom ``i``.
            chunk_num (int): 1-based chunk number used in the output filename.
            override_directory (str | None): If provided, write to this directory
                instead of ``self.directory``.
        """
        base_name = "nearest_neighbors_dist"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(dist_list)
        lengths = [arr.size for arr in dist_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if n_atoms > 0:
            flat_dist = np.concatenate(dist_list)
        else:
            flat_dist = np.zeros(0, dtype=np.float32)

        np.savez(save_path, flat_dist=flat_dist, offsets=offsets)

    def load_chunk_nn_indices(self, chunk_num):
        """
        Load neighbor indices for a chunk.

        Reads ``nearest_neighbors_indices_<chunk_num>.npz`` and returns the
        flattened indices and offsets.

        Args:
            chunk_num (int): 1-based chunk number to load.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(flat_idx, offsets)`` where
            ``flat_idx`` is a 1-D int array of all neighbor indices and
            ``offsets`` is a 1-D int64 array of length n_atoms + 1.

        Raises:
            FileNotFoundError: If the NPZ file is missing.
        """
        # Build expected path and verify existence
        base_name = "nearest_neighbors_indices"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN indices file not found: {full_path}")

        # Load arrays from NPZ
        with np.load(full_path) as data:
            flat_idx = data['flat_idx']
            offsets = data['offsets']

        return flat_idx, offsets


    def load_chunk_nn_phase(self, chunk_num):
        """
        Load neighbor phases for a chunk.

        Reads ``nearest_neighbors_phase_<chunk_num>.npz`` and returns the
        flattened phases and offsets.

        Args:
            chunk_num (int): 1-based chunk number to load.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(flat_phase, offsets)`` where
            ``flat_phase`` is a 1-D float array and ``offsets`` is a 1-D
            int64 array of length n_atoms + 1.

        Raises:
            FileNotFoundError: If the NPZ file is missing.
        """
        # Build expected path and verify existence
        base_name = "nearest_neighbors_phase"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN phase file not found: {full_path}")

        # Load arrays from NPZ
        with np.load(full_path) as data:
            flat_phase = data['flat_phase']
            offsets = data['offsets']

        return flat_phase, offsets

    def load_chunk_nn_scatter(self, chunk_num):
        """
        Load neighbor wavevectors for a chunk.

        Reads ``nearest_neighbors_scatter_<chunk_num>.npz`` and returns the
        flattened ``kx``, ``ky``, ``kz`` arrays and the offsets vector.

        Args:
            chunk_num (int): 1-based chunk number to load.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                ``(flat_kx, flat_ky, flat_kz, offsets)``.

        Raises:
            FileNotFoundError: If the NPZ file is missing.
        """
        # Build expected path and verify existence
        base_name = "nearest_neighbors_scatter"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN scatter file not found: {full_path}")

        # Load arrays from NPZ
        with np.load(full_path) as data:
            flat_kx = data['flat_kx']
            flat_ky = data['flat_ky']
            flat_kz = data['flat_kz']
            offsets = data['offsets']

        return flat_kx, flat_ky, flat_kz, offsets
    
    def load_chunk_nn_species(self, chunk_num):
        """
        Load neighbor species for a chunk.

        Reads ``nearest_neighbors_species_<chunk_num>.npz`` and returns the
        flattened species array and offsets.

        Example:
            To reconstruct the ragged lists::

                flat_spc, offsets = self.load_chunk_nn_species(chunk_num)
                species_list = [flat_spc[offsets[i]:offsets[i+1]]
                                for i in range(offsets.size - 1)]

        Args:
            chunk_num (int): 1-based chunk number to load.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(flat_species, offsets)``.

        Raises:
            FileNotFoundError: If the NPZ file is missing.
        """
        # Build expected path and verify existence
        base_name = "nearest_neighbors_species"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN species file not found: {full_path}")

        # allow_pickle=True to support non-numeric species types if present
        with np.load(full_path, allow_pickle=True) as data:
            flat_species = data['flat_species']
            offsets = data['offsets']

        return flat_species, offsets

    def load_chunk_nn_dist(self, chunk_num):
        """
        Load neighbor distances for a chunk.

        Reads ``nearest_neighbors_dist_<chunk_num>.npz`` and returns the
        flattened distances (Angstrom) and offsets.

        Args:
            chunk_num (int): 1-based chunk number to load.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(flat_dist, offsets)`` where
            ``flat_dist`` is a 1-D float32 array of distances in Angstrom and
            ``offsets`` is a 1-D int64 array of length n_atoms + 1.

        Raises:
            FileNotFoundError: If the NPZ file is missing.
        """
        base_name = "nearest_neighbors_dist"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN dist file not found: {full_path}")

        with np.load(full_path) as data:
            flat_dist = data['flat_dist']
            offsets = data['offsets']

        return flat_dist, offsets
    # -------------------------------------

    # -------------------------------------
    # MD sample
    def import_atomic_data(self, import_file, element_list, header_lines=9, ID_column=1, position_columns=[2,3,4], scale=1e-10, flush_size=100000000, override_directory=None):
        """
        Import a large text file of atoms and write chunked .npy outputs.

        Reads atomic positions and species from a text file, skipping a header,
        and streams them into fixed-size binary chunks for positions/species.
        Also computes the axis-aligned bounding box to infer sample dimensions
        and offset, and records the number of written chunks.

        The input line is split on whitespace. The species identifier is read
        from ``ID_column`` (0-based column index) and is expected to be a
        1-based integer ID; it is mapped to ``element_list[id-1]``. Positions
        are taken from ``position_columns`` and multiplied by
        ``scale/1e-10`` so they end up in the same units as the rest of the
        code (angstroms by default). With the default ``scale=1e-10``, values
        are unchanged.

        Args:
            import_file (str): Path to the input text file.
            element_list (list[str]): Map from 1-based species IDs in the file
                to element symbols or species labels.
            header_lines (int, optional): Number of header lines to skip.
                Defaults to 9.
            ID_column (int, optional): 0-based column index containing the
                1-based species ID. Defaults to 1.
            position_columns (list[int], optional): 0-based indices of x, y, z
                position columns. Defaults to [2, 3, 4].
            scale (float, optional): Conversion factor from input units to meters.
                Values are multiplied by ``scale/1e-10`` to convert to angstroms.
                Defaults to 1e-10.
            flush_size (int, optional): Number of atoms per written chunk.
                Defaults to 100_000_000.
            override_directory (str | None, optional): Directory to write files
                to instead of ``self.directory``. Defaults to None.

        Returns:
            None

        Raises:
            FileNotFoundError: If ``import_file`` does not exist.
            ValueError: If parsing fails due to malformed lines.

        Notes:
            - After completion, ``_chunk_total``, ``_dimensions``, ``_offset``,
              and ``_rotation`` are updated.
            - This function streams lines to bound memory usage on very large files.
        """
        from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

        chunk_num = 0  # running chunk counter

        # Track bounding box while streaming the file
        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')

        # Thread pool for async disk writes (mirrors generate_sample_single pattern)
        def _write_chunk(idx, pos_arr, spc_arr):
            self.write_chunk_positions(pos_arr, idx, override_directory=override_directory)
            self.write_chunk_species(spc_arr, idx, override_directory=override_directory)
            return idx

        writer_pool = ThreadPoolExecutor(
            max_workers=max(1, SAMPLE_WRITER_THREADS),
            thread_name_prefix="import_writer"
        )
        pending_writes = []

        # Determine which columns to read and their order in the output array
        scale_factor = float(scale / 1e-10)
        read_species = (ID_column != 0)
        if read_species:
            cols_to_read = list(position_columns) + [ID_column]
            element_arr = np.array(element_list)
        else:
            cols_to_read = list(position_columns)

        try:
            # Open and iterate in batches of up to flush_size rows
            with open(import_file, "r") as f:
                # Skip header lines at the top of the file
                for _ in range(header_lines):
                    next(f)

                while True:
                    # Read up to flush_size rows with C-optimized parsing
                    try:
                        raw_data = np.loadtxt(
                            f,
                            max_rows=flush_size,
                            usecols=cols_to_read,
                            dtype=np.float64,
                            ndmin=2
                        )
                    except (StopIteration, ValueError):
                        break

                    if raw_data.size == 0:
                        break

                    n_atoms = raw_data.shape[0]

                    # Extract and scale positions (first 3 columns correspond to position_columns)
                    data_arr = (raw_data[:, :3] * scale_factor).astype(np.float32)

                    # Vectorized bounding box update
                    chunk_min = data_arr.min(axis=0)
                    chunk_max = data_arr.max(axis=0)
                    x_min = min(x_min, chunk_min[0])
                    y_min = min(y_min, chunk_min[1])
                    z_min = min(z_min, chunk_min[2])
                    x_max = max(x_max, chunk_max[0])
                    y_max = max(y_max, chunk_max[1])
                    z_max = max(z_max, chunk_max[2])

                    # Map species IDs to element labels
                    if read_species:
                        species_ids = raw_data[:, 3].astype(np.int32)
                        species_arr = element_arr[species_ids - 1]
                    else:
                        species_arr = np.full(n_atoms, element_list[0])

                    # Submit async write (copy to isolate from next iteration)
                    chunk_num += 1
                    pending_writes.append(
                        writer_pool.submit(_write_chunk, chunk_num, data_arr.copy(), species_arr.copy())
                    )

            # Wait for all writes to complete before setting metadata
            wait(pending_writes, return_when=ALL_COMPLETED)
        finally:
            writer_pool.shutdown(wait=True)

        # Record how many chunks were written to disk
        self._chunk_total = chunk_num

        # Infer sample box dimensions from the bounding box
        self._dimensions = np.array([x_max - x_min,
                                     y_max - y_min,
                                     z_max - z_min], dtype=np.float32)

        # Center (offset) is the mid-point of the bounding box
        self._offset = np.array([(x_min + x_max) / 2.0,
                                 (y_min + y_max) / 2.0,
                                 (z_min + z_max) / 2.0], dtype=np.float32)

        # Set default rotation to identity
        self._rotation = np.eye(3)
    # -------------------------------------

    ## Static Functions
    # -------------------------------------
    # General
    @staticmethod
    def get_unit_corners():
        """
        Return the 8 corners of the unit cube as an (8, 3) float32 array.

        The ordering matches bit-coded vertices used elsewhere:
        index i has bits (x,y,z) taken from (i&1, (i>>1)&1, (i>>2)&1).
        This convention is consistent with the SAT C code for rebuilding corners.

        Returns:
            np.ndarray: Array of shape (8, 3) with values in {0, 1}.
        """
        # Corners of a unit axis-aligned box in the order expected by SAT code
        unit_corners = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]], dtype=np.float32)
        return unit_corners
    
    @staticmethod
    def get_rotation(axis, angle):
        """
        Compute a 3x3 rotation matrix for a rotation about an axis.

        The input axis is normalized inside this function. The rotation follows
        the right-hand rule.

        Args:
            axis (array-like): Length-3 vector representing the rotation axis.
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        # Normalize axis to ensure a proper rotation basis
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        d = 1.0 - c
        x, y, z = axis
        return np.array([[c + d*x*x,     d*x*y - z*s,   d*x*z + y*s],
                         [d*y*x + z*s,   c + d*y*y,     d*y*z - x*s],
                         [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]])
    
    @staticmethod
    def get_flat_grid(dimensions, use_gpu=False):
        """
        Create a flat grid of integer coordinates as an (N, 3) array.

        The grid spans [0..d0-1] x [0..d1-1] x [0..d2-1] in row-major order
        without materializing a full 3D tensor. Returned dtype is float32 to
        match downstream GPU code.

        Args:
            dimensions (array-like): Length-3 iterable of integers (d0, d1, d2).
            use_gpu (bool, optional): If True and CuPy is available, return a
                ``cp.ndarray``; otherwise return a ``np.ndarray``. Defaults to False.

        Returns:
            np.ndarray or cp.ndarray: Array of shape (d0*d1*d2, 3), dtype float32.

        Notes:
            This function uses repeat/tile to generate coordinates efficiently.
        """
        d0 = int(dimensions[0])
        d1 = int(dimensions[1])
        d2 = int(dimensions[2])

        if use_gpu and (cp is not None):
            # GPU path: build i, j, k using cp repeat/tile and stack
            ii = cp.repeat(cp.arange(d0, dtype=cp.float32), d1 * d2)
            jj = cp.tile(cp.repeat(cp.arange(d1, dtype=cp.float32), d2), d0)
            kk = cp.tile(cp.arange(d2, dtype=cp.float32), d0 * d1)
            return cp.stack((ii, jj, kk), axis=1)
        else:
            # CPU path: equivalent construction using NumPy
            ii = np.repeat(np.arange(d0, dtype=np.float32), d1 * d2)
            jj = np.tile(np.repeat(np.arange(d1, dtype=np.float32), d2), d0)
            kk = np.tile(np.arange(d2, dtype=np.float32), d0 * d1)
            return np.stack((ii, jj, kk), axis=1)

    def set_temperature_einstein(
        self,
        T_K,
        mass_amu=None,
        theta_E_K=None,
        species_mass_amu=None,
        species_theta_E_K=None,
        max_displacement=None,
        seed=40
    ):
        """
        Configure Einstein-model thermal displacements and enable temperature.

        Sets ``enable_temp=True`` and programs ``temp_params`` to use the
        Einstein model, where the random displacement variance is derived from
        temperature and oscillator frequency.

        You may specify either global values (``mass_amu``, ``theta_E_K``) or
        per-species dictionaries (``species_mass_amu``, ``species_theta_E_K``).
        If per-species dicts are provided and a ``chunk_number`` is later passed
        into ``apply_temperature``, the per-atom mass and theta_E will be looked
        up from the species data stored on disk.

        Args:
            T_K (float): Target temperature in kelvin.
            mass_amu (float | None): Global atomic mass in amu, used if per-species
                masses are not provided. Optional.
            theta_E_K (float | None): Global Einstein temperature (K), used if
                per-species values are not provided. Optional.
            species_mass_amu (dict[str, float] | None): Map species label -> mass.
            species_theta_E_K (dict[str, float] | None): Map species label -> theta_E.
            max_displacement (float | None): Optional per-axis clip in position units.
            seed (int, optional): Random seed for reproducibility. Defaults to 40.

        Returns:
            None

        Notes:
            - Default position unit is angstrom; override with
              ``set_position_unit_in_m`` if positions are in a different unit.
            - This function only configures parameters; displacements are applied
              when ``apply_temperature`` is called.
        """
        # Enable temperature-driven displacements via the Einstein model
        self.enable_temp = True
        self.temp_params = ['einstein', float(T_K), max_displacement, seed]

        # Save optional global fallbacks
        if mass_amu is not None:
            self._temp_mass_amu = float(mass_amu)
        if theta_E_K is not None:
            self._temp_theta_E_K = float(theta_E_K)

        # Save optional per-species overrides
        if species_mass_amu is not None:
            self._temp_species_mass_amu = dict(species_mass_amu)
        if species_theta_E_K is not None:
            self._temp_species_theta_E_K = dict(species_theta_E_K)

        # Default position unit is angstrom (1e-10 m) unless user overrides later
        if not hasattr(self, '_position_unit_in_m'):
            self._position_unit_in_m = 1.0e-10

    def set_temperature_debye(
        self,
        T_K,
        mass_amu=None,
        theta_D_K=None,
        max_displacement=None,
        seed=40
    ):
        """
        Configure Debye-model thermal displacements and enable temperature.

        Sets ``enable_temp=True`` and programs ``temp_params`` to use the
        Debye model, where the random displacement variance is derived from
        the Debye function integral for more accurate low-temperature behavior.

        Args:
            T_K (float): Target temperature in kelvin.
            mass_amu (float | None): Atomic mass in amu. Defaults to 28.0 if not provided.
            theta_D_K (float | None): Debye temperature (K). Defaults to 300.0 if not provided.
            max_displacement (float | None): Optional per-axis clip in position units.
            seed (int, optional): Random seed for reproducibility. Defaults to 40.

        Returns:
            None

        Notes:
            - Default position unit is angstrom; override with
              ``set_position_unit_in_m`` if positions are in a different unit.
            - The Debye model provides more accurate thermal displacement at low
              temperatures (T << theta_D) compared to the Einstein model.
            - At high temperatures, Debye and Einstein models converge.
        """
        # Enable temperature-driven displacements via the Debye model
        self.enable_temp = True
        self.temp_params = ['debye', float(T_K), max_displacement, seed]

        # Save optional global parameters
        if mass_amu is not None:
            self._temp_mass_amu = float(mass_amu)
        if theta_D_K is not None:
            self._temp_theta_D_K = float(theta_D_K)

        # Default position unit is angstrom (1e-10 m) unless user overrides later
        if not hasattr(self, '_position_unit_in_m'):
            self._position_unit_in_m = 1.0e-10

    def set_thermal_expansion(
        self,
        alpha=None,
        alpha_xyz=None,
        T_ref=300.0
    ):
        """
        Configure thermal expansion coefficient(s).

        Thermal expansion scales positions by factor (1 + α * ΔT) where ΔT = T - T_ref.
        Can use either isotropic (single α) or anisotropic (αx, αy, αz) expansion.

        Args:
            alpha (float | None): Isotropic linear expansion coefficient (1/K).
                Typical values: ~1e-5 for metals, ~1e-6 for ceramics.
            alpha_xyz (array-like | None): Anisotropic expansion [αx, αy, αz] (1/K).
                If provided, overrides isotropic alpha.
            T_ref (float): Reference temperature in Kelvin. Default 300 K (room temp).
                Expansion is calculated as ΔT = T_current - T_ref.

        Returns:
            None

        Notes:
            - Thermal expansion is applied during chunk loading if enable_temp is True.
            - Expansion is applied before thermal vibrations.
            - For most materials, α is in the range 1e-6 to 1e-5 per Kelvin.
        """
        self._thermal_expansion_enabled = True
        self._thermal_expansion_T_ref = float(T_ref)

        if alpha_xyz is not None:
            self._thermal_expansion_alpha_xyz = np.array(alpha_xyz, dtype=np.float64)
            self._thermal_expansion_alpha = None
        elif alpha is not None:
            self._thermal_expansion_alpha = float(alpha)
            self._thermal_expansion_alpha_xyz = None

    def apply_thermal_expansion(self, positions, T_K):
        """
        Apply thermal expansion to positions.

        Scales positions relative to sample center by (1 + α * ΔT) where ΔT = T_K - T_ref.
        Supports both isotropic and anisotropic expansion.

        The expansion formula is: p_new = center + (p_old - center) * scale
        This ensures atoms expand uniformly from the sample center, not from the origin.

        Args:
            positions (np.ndarray or cp.ndarray): Array of shape (N, 3).
            T_K (float): Current temperature in Kelvin.

        Returns:
            np.ndarray or cp.ndarray: Scaled positions of the same shape and backend.

        Notes:
            - If thermal expansion is not enabled, returns positions unchanged.
            - Expansion is relative to self.offset (sample center), not origin.
            - Isotropic: all coordinates scaled by same factor.
            - Anisotropic: each axis (x, y, z) scaled by different factor.
        """
        if not getattr(self, '_thermal_expansion_enabled', False):
            return positions

        T_ref = getattr(self, '_thermal_expansion_T_ref', 300.0)
        dT = T_K - T_ref

        # Determine backend (NumPy vs CuPy)
        use_cp = (cp is not None) and isinstance(positions, cp.ndarray)
        xp = cp if use_cp else np

        # Expand from the sample CENTER (self.offset), not the origin
        # This ensures uniform expansion regardless of sample position
        center = xp.asarray(self.offset, dtype=xp.float64)

        alpha_xyz = getattr(self, '_thermal_expansion_alpha_xyz', None)
        alpha = getattr(self, '_thermal_expansion_alpha', None)

        if alpha_xyz is not None:
            # Anisotropic: different scaling per axis
            alpha_arr = xp.asarray(alpha_xyz, dtype=xp.float64)
            scale = 1.0 + alpha_arr * dT  # Shape (3,)
            return center + (positions - center) * scale
        elif alpha is not None:
            # Isotropic: uniform scaling
            scale = 1.0 + alpha * dT
            return center + (positions - center) * scale

        return positions

    def set_position_unit_in_m(self, unit_in_m):
        """
        Set the conversion factor from the position unit to meters.

        This affects the Einstein-model conversion from mean-square displacement
        in meters^2 to the position units of your arrays.

        Args:
            unit_in_m (float): Number of meters represented by 1 position unit
                in your arrays (e.g., 1e-10 for angstrom, 1e-9 for nanometer).

        Returns:
            None
        """
        # Store the unit scale so apply_temperature can convert correctly
        self._position_unit_in_m = float(unit_in_m)

    def apply_temperature(
        self,
        positions,
        distribution='gaussian',
        sigma=0.25,
        max_displacement=1,
        seed=40,
        chunk_number=None
    ):
        """
        Apply random displacements to positions using a chosen distribution.

        Three modes are supported:

        1) ``distribution='gaussian'``:
           - ``sigma`` is the standard deviation in position units.
        2) ``distribution='einstein'``:
           - ``sigma`` is interpreted as temperature in kelvin.
           - The per-atom Gaussian width is computed from the Einstein model:
             <x^2> = (hbar / (2 m omega)) * coth(hbar*omega / (2 k_B T)),
             with omega = k_B * theta_E / hbar.
           - If per-species mass and theta_E were configured with
             ``set_temperature_einstein`` and ``chunk_number`` is provided, the
             correct per-atom parameters are used by reading species for that chunk.
        3) ``distribution='debye'``:
           - ``sigma`` is interpreted as temperature in kelvin.
           - Uses the Debye model with the Debye function integral:
             <u^2> = (9 hbar^2 / m k_B theta_D) * [D_3(theta_D/T) * (T/theta_D) + 1/4]
           - More accurate than Einstein at low temperatures (T << theta_D).
           - Configure with ``set_temperature_debye``.

        Args:
            positions (np.ndarray or cp.ndarray): Array of shape (N, 3).
            distribution (str, optional): 'gaussian', 'einstein', or 'debye'.
                Defaults to 'gaussian'.
            sigma (float, optional): Stddev (gaussian) or temperature K (einstein/debye).
                Defaults to 0.25.
            max_displacement (float | None, optional): Optional per-axis clip on
                the displacement magnitude. Defaults to 1.
            seed (int | None, optional): Random seed for reproducibility. Set to
                None to avoid reseeding. Defaults to 40.
            chunk_number (int | None, optional): Chunk index used to fetch
                per-atom species on CPU for per-species Einstein parameters.

        Returns:
            np.ndarray or cp.ndarray: Displaced positions of the same shape and backend.

        Raises:
            ValueError: If an unknown ``distribution`` is provided.

        Notes:
            - Backend (NumPy vs CuPy) is inferred from ``positions`` type.
            - If ``T_K <= 0`` in Einstein/Debye mode, only zero-point motion is applied.
        """
        # Select backend based on input array type
        use_cp = (cp is not None) and isinstance(positions, cp.ndarray)
        xp = cp if use_cp else np

        # Seed the RNG for deterministic noise if a seed is provided
        if seed is not None:
            if use_cp:
                cp.random.seed(int(seed))
            else:
                np.random.seed(int(seed))

        # Mode 1: plain Gaussian displacements with a single sigma
        if isinstance(distribution, str) and distribution.lower() in ('gaussian', 'normal'):
            displacements = xp.random.normal(loc=0.0, scale=float(sigma), size=positions.shape)
            if (max_displacement is not None) and (max_displacement > 0.0):
                xp.clip(displacements, -max_displacement, max_displacement, out=displacements)
            return positions + displacements

        # Mode 2: Einstein-model temperature-driven displacements
        if isinstance(distribution, str) and distribution.lower() in ('einstein', 'temperature', 'kelvin'):
            # Treat sigma as temperature in kelvin in this mode
            T_K = float(sigma)

            # Conversion from position units to meters (default assumes angstroms)
            pos_unit_m = getattr(self, '_position_unit_in_m', 1e-10)

            # Physical constants (SI)
            k_B = 1.380649e-23
            hbar = 1.054571817e-34
            amu_to_kg = 1.66053906660e-27

            N = int(positions.shape[0])

            # Resolve per-atom mass and theta_E (per-species if configured)
            masses_amu = None
            thetaE_K = None

            have_per_species = (
                hasattr(self, '_temp_species_mass_amu') and
                hasattr(self, '_temp_species_theta_E_K') and
                (self._temp_species_mass_amu is not None) and
                (self._temp_species_theta_E_K is not None) and
                (chunk_number is not None)
            )

            if have_per_species:
                # Load species on CPU for this chunk and map to arrays of m and theta_E
                species = self.load_chunk_species(chunk_number, use_gpu=False)
                masses_amu = np.empty(N, dtype=np.float64)
                thetaE_K = np.empty(N, dtype=np.float64)

                # Fallbacks if a species key is missing
                m_default = getattr(self, '_temp_mass_amu', 28.0)
                th_default = getattr(self, '_temp_theta_E_K', 300.0)

                for i, sp in enumerate(species):
                    key = str(sp)
                    masses_amu[i] = self._temp_species_mass_amu.get(key, m_default)
                    thetaE_K[i] = self._temp_species_theta_E_K.get(key, th_default)
            else:
                # Use global values (with safe fallbacks) for all atoms
                m_default = getattr(self, '_temp_mass_amu', 28.0)
                th_default = getattr(self, '_temp_theta_E_K', 300.0)
                masses_amu = np.full(N, m_default, dtype=np.float64)
                thetaE_K = np.full(N, th_default, dtype=np.float64)

            # Move to the active backend and compute omega from theta_E
            m_kg = xp.asarray(masses_amu, dtype=xp.float64) * amu_to_kg
            thetaE_K = xp.asarray(thetaE_K, dtype=xp.float64)
            omega = (k_B * thetaE_K) / hbar

            # Compute coth term robustly; handle T_K <= 0 by zero-point motion
            if T_K <= 0.0:
                coth_z = xp.ones_like(omega, dtype=xp.float64)
            else:
                z = (hbar * omega) / (2.0 * k_B * T_K)
                small = z < 1.0e-6
                coth_series = (1.0 / z) + (z / 3.0)
                coth_exact = 1.0 / xp.tanh(z)
                coth_z = xp.where(small, coth_series, coth_exact)

            # Mean-square displacement in meters^2, then convert to position units
            msd_m2 = (hbar / (2.0 * m_kg * omega)) * coth_z
            sigma_units = xp.sqrt(msd_m2) / pos_unit_m  # per-atom sigma in position units

            # Draw per-atom Gaussian noise and cast to match input dtype
            rand = xp.random.standard_normal(size=positions.shape)
            displacements = rand * sigma_units.reshape(-1, 1)
            displacements = displacements.astype(positions.dtype, copy=False)

            # Optionally clip each coordinate
            if (max_displacement is not None) and (max_displacement > 0.0):
                xp.clip(displacements, -max_displacement, max_displacement, out=displacements)

            return positions + displacements

        # Mode 3: Debye-model temperature-driven displacements
        if isinstance(distribution, str) and distribution.lower() == 'debye':
            # Treat sigma as temperature in kelvin in this mode
            T_K = float(sigma)

            # Conversion from position units to meters (default assumes angstroms)
            pos_unit_m = getattr(self, '_position_unit_in_m', 1e-10)

            # Physical constants (SI)
            k_B = 1.380649e-23
            hbar = 1.054571817e-34
            amu_to_kg = 1.66053906660e-27

            N = int(positions.shape[0])

            # Get global mass and Debye temperature (with safe fallbacks)
            m_default = getattr(self, '_temp_mass_amu', 28.0)
            theta_D = getattr(self, '_temp_theta_D_K', 300.0)

            # Convert mass to kg
            m_kg = m_default * amu_to_kg

            # Debye model mean-square displacement formula:
            # <u^2> = (9 hbar^2 / m k_B theta_D) * [D_3(x)/x + 1/4]
            # where x = theta_D / T and D_3(x) = (3/x^3) * integral_0^x t^3/(e^t - 1) dt
            #
            # Simplified form:
            # <u^2> = (9 hbar^2 / m k_B theta_D) * [phi(x) + 1/4]
            # where phi(x) = D_3(x)/x = (3/x^4) * integral_0^x t^3/(e^t - 1) dt

            # Compute the prefactor (units: m^2 / K)
            prefactor = (9.0 * hbar * hbar) / (m_kg * k_B * theta_D)

            # Compute the Debye integral contribution
            if T_K <= 0.0:
                # Zero-point motion only: phi(infinity) -> 0, so just 1/4 term
                phi_term = 0.0
            else:
                x = theta_D / T_K
                # Compute phi(x) = (3/x^4) * integral_0^x t^3/(e^t - 1) dt
                # Use scipy.integrate.quad for numerical integration
                from scipy.integrate import quad

                def debye_integrand(t):
                    if t < 1e-10:
                        # Taylor expansion near t=0: t^3/(e^t - 1) ~ t^2
                        return t * t
                    return (t * t * t) / (np.exp(t) - 1.0)

                if x > 50.0:
                    # For very large x, use the known limit: integral -> pi^4/15
                    # phi(x) = (3/x^4) * (pi^4/15) for x -> infinity
                    integral_val = (np.pi ** 4) / 15.0
                else:
                    integral_val, _ = quad(debye_integrand, 0, x, limit=100)

                phi_term = (3.0 / (x ** 4)) * integral_val

            # Mean-square displacement in m^2
            msd_m2 = prefactor * (phi_term + 0.25)

            # Convert to position units (sigma per atom)
            sigma_units = np.sqrt(msd_m2) / pos_unit_m

            # Draw Gaussian noise and scale by sigma
            rand = xp.random.standard_normal(size=positions.shape)
            displacements = rand * sigma_units
            displacements = displacements.astype(positions.dtype, copy=False)

            # Optionally clip each coordinate
            if (max_displacement is not None) and (max_displacement > 0.0):
                xp.clip(displacements, -max_displacement, max_displacement, out=displacements)

            return positions + displacements

        # Anything else is unsupported
        raise ValueError("Unknown distribution: {}".format(distribution))
    # -------------------------------------
        
    # -------------------------------------
    # Sample generation
    @staticmethod    
    def compile_parallelepipeds_intersect_batch_cffi():
        """
        Compile the CFFI SAT intersection batch function.

        Builds and verifies a small C module implementing a 15-axis Separating
        Axis Theorem (SAT) test for parallelepipeds. The compiled module
        exposes:

            int check_parallelepipeds_intersect_batch(
                const double *all_pts1,
                const double *pts2,
                double eps,
                int n,
                int *out_intersect
            );

        where each shape is represented by 8 corners (24 doubles). The function
        returns an FFI object and the compiled module handle.

        Returns:
            tuple[FFI, <cffi.verifier.VerifiedModule>]: ``(ffi_obj, compiled_module)``

        Notes:
            - The corner ordering must match ``get_unit_corners``.
            - ``eps`` is used to skip near-degenerate axes in the SAT test.
        """
        # C source implementing dot/cross helpers, projection, interval overlap,
        # single-shape SAT, and a batch wrapper over n shapes.
        c_source = r'''
        #include <math.h>
        #include <stdlib.h> // for malloc/free if needed
        // Dot product
        static double dot3(const double *a, const double *b){
            return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
        }

        // Cross product out = a x b
        static void cross3(const double *a, const double *b, double *out){
            out[0] = a[1]*b[2] - a[2]*b[1];
            out[1] = a[2]*b[0] - a[0]*b[2];
            out[2] = a[0]*b[1] - a[1]*b[0];
        }

        // Norm of a 3D vector
        static double norm3(const double *v){
            return sqrt(dot3(v,v));
        }

        // Project 8 points onto an axis
        // out[0] = min, out[1] = max of the projection
        static void project_points(const double *pts8x3, const double *axis, double eps, double *out){
            double axis_len = norm3(axis);
            if(axis_len < eps){
                // Degenerate axis -> all points project to zero
                out[0] = 0.0; 
                out[1] = 0.0;
                return;
            }
            double ax[3] = { axis[0]/axis_len, axis[1]/axis_len, axis[2]/axis_len };

            double val = dot3(pts8x3, ax); // first corner
            double minv = val, maxv = val;
            for(int i=1; i<8; i++){
                val = dot3(pts8x3 + 3*i, ax);
                if(val < minv) minv = val;
                if(val > maxv) maxv = val;
            }
            out[0] = minv;
            out[1] = maxv;
        }

        // Check if intervals [a0,a1] and [b0,b1] overlap
        static int intervals_overlap(const double *a, const double *b){
            // If one interval is strictly to the left of the other, no overlap
            if(a[1] < b[0] || b[1] < a[0]) 
                return 0;
            return 1;
        }

        // single_intersect: checks intersection for one pair of parallelepipeds
        // pts1, pts2 each has 8 corners -> 24 doubles
        static int single_intersect(const double *pts1, const double *pts2, double eps)
        {
            // 1) Identify shape1 edges from the known corner ordering
            //    c1 = pts1[0], e1 = pts1[1] - pts1[0], e2 = pts1[2] - pts1[0], e3 = pts1[3] - pts1[0].
            double c1[3]  = { pts1[0], pts1[1], pts1[2] };
            double e1[3]  = { pts1[3] - c1[0], pts1[4] - c1[1], pts1[5] - c1[2] };
            double e2[3]  = { pts1[6] - c1[0], pts1[7] - c1[1], pts1[8] - c1[2] };
            double e3[3]  = { pts1[9] - c1[0], pts1[10] - c1[1], pts1[11] - c1[2] };

            // 2) Identify shape2 edges similarly
            double c2[3]  = { pts2[0], pts2[1], pts2[2] };
            double f1[3]  = { pts2[3] - c2[0], pts2[4] - c2[1], pts2[5] - c2[2] };
            double f2[3]  = { pts2[6] - c2[0], pts2[7] - c2[1], pts2[8] - c2[2] };
            double f3[3]  = { pts2[9] - c2[0], pts2[10] - c2[1], pts2[11] - c2[2] };

            // 3) Rebuild all 8 corners for shape1
            //    shape1[i] = c1 + alpha1 * e1 + alpha2 * e2 + alpha3 * e3,
            //    where alphaN is either 0 or 1. The corner ordering matches get_unit_corners().
            double shape1[24];
            for(int i=0; i<8; i++){
                int a1 = (i & 1) ? 1 : 0; // bit 0
                int a2 = (i & 2) ? 1 : 0; // bit 1
                int a3 = (i & 4) ? 1 : 0; // bit 2
                shape1[3*i + 0] = c1[0] + a1*e1[0] + a2*e2[0] + a3*e3[0];
                shape1[3*i + 1] = c1[1] + a1*e1[1] + a2*e2[1] + a3*e3[1];
                shape1[3*i + 2] = c1[2] + a1*e1[2] + a2*e2[2] + a3*e3[2];
            }

            // 4) Rebuild all 8 corners for shape2
            double shape2[24];
            for(int i=0; i<8; i++){
                int a1 = (i & 1) ? 1 : 0;
                int a2 = (i & 2) ? 1 : 0;
                int a3 = (i & 4) ? 1 : 0;
                shape2[3*i + 0] = c2[0] + a1*f1[0] + a2*f2[0] + a3*f3[0];
                shape2[3*i + 1] = c2[1] + a1*f1[1] + a2*f2[1] + a3*f3[1];
                shape2[3*i + 2] = c2[2] + a1*f1[2] + a2*f2[2] + a3*f3[2];
            }

            // 5) Compute the 15 candidate axes:
            //    -- 3 face normals from shape1
            //    -- 3 face normals from shape2
            //    -- 9 cross products of edges from shape1 x edges from shape2

            // shape1 face normals
            double n1[3], n2[3], n3[3];
            cross3(e1, e2, n1);
            cross3(e2, e3, n2);
            cross3(e3, e1, n3);

            // shape2 face normals
            double m1[3], m2[3], m3[3];
            cross3(f1, f2, m1);
            cross3(f2, f3, m2);
            cross3(f3, f1, m3);

            double edges1[3][3] = {{e1[0], e1[1], e1[2]},
                                   {e2[0], e2[1], e2[2]},
                                   {e3[0], e3[1], e3[2]}};
            double edges2[3][3] = {{f1[0], f1[1], f1[2]},
                                   {f2[0], f2[1], f2[2]},
                                   {f3[0], f3[1], f3[2]}};

            double axes[15][3];
            int axisCount = 0;

            // shape1 face normals
            axes[axisCount][0] = n1[0]; axes[axisCount][1] = n1[1]; axes[axisCount][2] = n1[2]; axisCount++;
            axes[axisCount][0] = n2[0]; axes[axisCount][1] = n2[1]; axes[axisCount][2] = n2[2]; axisCount++;
            axes[axisCount][0] = n3[0]; axes[axisCount][1] = n3[1]; axes[axisCount][2] = n3[2]; axisCount++;

            // shape2 face normals
            axes[axisCount][0] = m1[0]; axes[axisCount][1] = m1[1]; axes[axisCount][2] = m1[2]; axisCount++;
            axes[axisCount][0] = m2[0]; axes[axisCount][1] = m2[1]; axes[axisCount][2] = m2[2]; axisCount++;
            axes[axisCount][0] = m3[0]; axes[axisCount][1] = m3[1]; axes[axisCount][2] = m3[2]; axisCount++;

            // cross products of edges
            for(int i=0; i<3; i++){
                for(int j=0; j<3; j++){
                    double c12[3];
                    cross3(edges1[i], edges2[j], c12);
                    double len_c12 = norm3(c12);
                    if(len_c12 > eps){  // skip near-degenerate
                        axes[axisCount][0] = c12[0];
                        axes[axisCount][1] = c12[1];
                        axes[axisCount][2] = c12[2];
                        axisCount++;
                    }
                }
            }

            // 6) Run the SAT test
            double proj1[2], proj2[2];
            for(int a=0; a<axisCount; a++){
                project_points(shape1, axes[a], eps, proj1);
                project_points(shape2, axes[a], eps, proj2);
                if(!intervals_overlap(proj1, proj2)){
                    // Found a separating axis -> no intersection
                    return 0;
                }
            }
            // No separating axis found => shapes intersect
            return 1;
        }

        // --------------------------------------------------------------------
        // BATCH function: parallelepipeds_intersect for n parallelepipeds
        // all_pts1: length 24*n (each block of 8 corners = 24 floats)
        // pts2    : just one shape of 8 corners = 24 floats
        // out_intersect[i] = 0 or 1
        // --------------------------------------------------------------------
        int check_parallelepipeds_intersect_batch(
            const double *all_pts1,
            const double *pts2,
            double eps,
            int n,
            int *out_intersect
        )
        {
            for(int i=0; i<n; i++){
                const double *shape_i = all_pts1 + 24*i; 
                out_intersect[i] = single_intersect(shape_i, pts2, eps);
            }
            return 0; // success
        }
        '''
        # Define the C function signature for the FFI layer
        ffi_obj = FFI()
        ffi_obj.cdef("""int check_parallelepipeds_intersect_batch(
            const double *all_pts1,
            const double *pts2,
            double eps,
            int n,
            int *out_intersect);
        """)

        # Compile and link the C code at runtime with optimization enabled
        C_mod = ffi_obj.verify(c_source, extra_compile_args=["-O3"], libraries=[])

        # Return both the FFI handle and the compiled module (used to call the function)
        return ffi_obj, C_mod
    # -------------------------------------
    
    # -------------------------------------
    # KNN search
    @staticmethod
    def build_cell_list_count_kernel():
        """
        Build and return the CUDA kernel that counts items per cell.

        Compiles a CUDA C kernel that:
        1) Computes a cell index for each point given an axis-aligned bounding
           box and uniform cubic cell size.
        2) Atomically increments the per-cell count.
        3) Writes the per-point cell index for use in a second pass.

        Returns:
            cupy.cuda.function.Function: Compiled CUDA kernel function
            ``cell_list_count_kernel`` ready to launch.

        Notes:
            - Requires CuPy with NVCC toolchain available.
            - The kernel expects positions as a flat float32 array shaped (N, 3).
        """
        _cell_list_count_kernel = r'''
        extern "C" __global__
        void cell_list_count_kernel(const float* __restrict__ positions,
                                    const float* __restrict__ bounding_box_min,
                                    const float* __restrict__ inv_cell_size,
                                    const int nx, const int ny, const int nz,
                                    int* __restrict__ cell_indices_out,
                                    int* __restrict__ cell_counts,
                                    const int N)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= N) return;

            // Read particle position
            float px = positions[3*idx + 0];
            float py = positions[3*idx + 1];
            float pz = positions[3*idx + 2];

            // Shift relative to bounding_box_min
            px -= bounding_box_min[0];
            py -= bounding_box_min[1];
            pz -= bounding_box_min[2];

            // Compute cell indices in each dimension
            int cx = (int)floorf(px * inv_cell_size[0]);
            int cy = (int)floorf(py * inv_cell_size[1]);
            int cz = (int)floorf(pz * inv_cell_size[2]);

            // Clamp to valid cell range just in case of numeric issues
            if (cx < 0) cx = 0; else if (cx >= nx) cx = nx - 1;
            if (cy < 0) cy = 0; else if (cy >= ny) cy = ny - 1;
            if (cz < 0) cz = 0; else if (cz >= nz) cz = nz - 1;

            // 1D cell index
            int cell_id = cz * (nx * ny) + cy * nx + cx;

            cell_indices_out[idx] = cell_id;

            // Use atomicAdd to increment cell_counts
            atomicAdd(&cell_counts[cell_id], 1);
        }
        '''
        # Build the raw CUDA module and fetch the kernel entry point
        kernel_module = cp.RawModule(
            code=_cell_list_count_kernel,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return kernel_module.get_function('cell_list_count_kernel')

    @staticmethod
    def build_cell_list_fill_kernel():
        """
        Build and return the CUDA kernel that fills sorted cell lists.

        Compiles a CUDA C kernel that:
        1) Uses the per-point cell index to place each point into a compact,
           cell-contiguous array using atomic adds on a per-cell write pointer.
        2) Writes both sorted positions and the original point indices.

        Returns:
            cupy.cuda.function.Function: Compiled CUDA kernel function
            ``cell_list_fill_kernel`` ready to launch.

        Notes:
            - Expects that ``cell_offsets`` already contains the starting offsets
              for each cell (usually the exclusive prefix sum of counts).
        """
        _cell_list_fill_kernel = r'''
        extern "C" __global__
        void cell_list_fill_kernel(const float* __restrict__ positions,
                                const int* __restrict__ cell_indices,
                                const int* __restrict__ cell_offsets,
                                float* __restrict__ sorted_positions,
                                int* __restrict__ sorted_indices,
                                const int N)
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= N) return;

            int cell_id = cell_indices[idx];

            // The offset for this cell is cell_offsets[cell_id].
            // We then use an atomicAdd to find the correct slot.
            int pos = atomicAdd((int*)&cell_offsets[cell_id], 1);

            sorted_positions[3*pos + 0] = positions[3*idx + 0];
            sorted_positions[3*pos + 1] = positions[3*idx + 1];
            sorted_positions[3*pos + 2] = positions[3*idx + 2];
            sorted_indices[pos] = idx;  // keep track of original (unsorted) index
        }
        '''
        # Build the raw CUDA module and fetch the kernel entry point
        kernel_module = cp.RawModule(
            code=_cell_list_fill_kernel,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return kernel_module.get_function('cell_list_fill_kernel')
    # -------------------------------------
        
    ## Main Functions
    # -------------------------------------
    # General
    def zero_sample_position(self, use_gpu=True):
        """
        Center all atom positions by subtracting the current offset.

        Reloads each chunk, subtracts ``self.offset`` from every position, and
        writes them back to disk. Finally sets ``self._offset`` to zeros.

        Args:
            use_gpu (bool, optional): If True and CuPy is available, load and
                process chunks on GPU before saving back to CPU. Defaults to True.

        Raises:
            ValueError: If ``_offset`` or ``_chunk_total`` is not initialized.

        Returns:
            None
        """
        # Validate that offset and chunk count are known
        if self._offset is None:
            raise ValueError("Offset is not initialized. Please set self._offset or load metadata first.")

        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate sample or import atoms first.")
        
        # Cache offset as float32 for consistent arithmetic
        offset_np = self.offset.astype(np.float32)

        for i in range(self.chunk_total):
            # Load the i-th chunk (1-based chunk files)
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                # GPU path: subtract offset on device, then bring to host for writing
                positions_chunk -= cp.array(offset_np)
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                # CPU path
                positions_chunk -= offset_np
                self.write_chunk_positions(positions_chunk, i + 1)

        # Reset offset to the origin
        self._offset = np.zeros(3, dtype=np.float32)
        print("All atomic positions re-centered. Offset is now [0, 0, 0].")
        
    def zero_sample_rotation(self, use_gpu=True):
        """
        Remove the current global rotation from all atom positions.

        Reloads each chunk, right-multiplies all positions by ``self._rotation.T``
        (the inverse for an orthonormal rotation), writes them back, and then
        resets ``self._rotation`` to identity.

        Args:
            use_gpu (bool, optional): If True and CuPy is available, perform the
                rotation on GPU before saving to CPU. Defaults to True.

        Raises:
            ValueError: If ``_rotation`` or ``_chunk_total`` is not initialized.

        Returns:
            None
        """
        # Must have a rotation to undo
        if self._rotation is None:
            raise ValueError("No sample rotation matrix is set. Please initialize or load it first.")
        
        # Inverse of rotation is its transpose for orthonormal matrices
        R_inv = self._rotation.T.astype(np.float32)
        
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                # GPU path: do matrix multiply on device
                R_inv_cp = cp.asarray(R_inv)
                positions_chunk = positions_chunk @ R_inv_cp
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                # CPU path
                positions_chunk = positions_chunk @ R_inv
                self.write_chunk_positions(positions_chunk, i + 1)
        
        # Reset rotation to identity
        self._rotation = np.eye(3, dtype=np.float32)
        print("All atomic positions de-rotated. Sample rotation is now the identity matrix.")
        
    def zero_sample(self, use_gpu=True):
        """
        Center and de-rotate the sample in-place.

        Calls ``zero_sample_position`` followed by ``zero_sample_rotation``.

        Args:
            use_gpu (bool, optional): If True and CuPy is available, enable GPU
                acceleration for both steps. Defaults to True.

        Returns:
            None
        """
        # First center positions at the origin, then remove any global rotation
        self.zero_sample_position(use_gpu=use_gpu)
        self.zero_sample_rotation(use_gpu=use_gpu)
        
    def rotate_sample_relative(self, axis, dangle, degrees=True, use_gpu=True):
        """
        Apply an additional rotation to all atoms and update state.

        Computes a rotation matrix for the given axis and angle, applies it to
        every chunk, writes the results, and left-multiplies the stored
        ``_rotation`` by the new rotation.

        Args:
            axis (array-like): 3-vector for the rotation axis.
            dangle (float): Angle of rotation. Interpreted as degrees if
                ``degrees=True``, otherwise radians.
            degrees (bool, optional): If True, convert ``dangle`` from degrees
                to radians. Defaults to True.
            use_gpu (bool, optional): If True and CuPy is available, perform the
                rotation on GPU before saving to CPU. Defaults to True.

        Raises:
            ValueError: If ``_chunk_total`` is not initialized.

        Returns:
            None
        """
        # Convert to radians if specified in degrees
        if degrees:
            dangle = np.deg2rad(dangle)
        
        # Build the rotation matrix and ensure float32 for consistency
        R = self.get_rotation(axis, dangle).astype(np.float32)
        
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                # GPU path
                R_cp = cp.asarray(R)
                positions_chunk = positions_chunk @ R_cp
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                # CPU path
                positions_chunk = positions_chunk @ R
                self.write_chunk_positions(positions_chunk, i + 1)
        
        # Update the stored global rotation
        self._rotation = R @ self._rotation
        print(f"Sample rotated by {dangle:.4f} radians about axis {axis}. "
              f"Updated sample rotation matrix:\n{self._rotation}")

    def translate_sample_relative(self, offset_vector, use_gpu=True):  # update this to use dx, dy, dz
        """
        Translate the sample by adding an offset to all atom positions.

        Reloads each chunk, adds ``offset_vector`` to every position, writes the
        results, and updates ``self._offset`` accordingly.

        Args:
            offset_vector (array-like): Length-3 translation vector in position units.
            use_gpu (bool, optional): If True and CuPy is available, perform the
                addition on GPU before saving to CPU. Defaults to True.

        Raises:
            ValueError: If ``_chunk_total`` is not initialized.

        Returns:
            None
        """
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        # Normalize input to float32 ndarray for consistent math
        offset_np = np.array(offset_vector, dtype=np.float32)
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                # GPU path
                positions_chunk += cp.asarray(offset_np)
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                # CPU path
                positions_chunk += offset_np
                self.write_chunk_positions(positions_chunk, i + 1)
        
        # Update the stored offset
        if self._offset is None:
            self._offset = offset_np
        else:
            self._offset += offset_np
        
        print(f"Sample translated by {offset_vector}. New offset is {self._offset}.")
    # -------------------------------------
    
    # -------------------------------------
    # Sample generation
    def _gpu_stream_chunk(self, material, chunk_position, chunk_dimensions, stream):
        """
        Generate, filter, and offset a chunk on a given CUDA stream.

        Builds atomic positions for a single geometric chunk entirely on GPU:
        1) Computes lattice positions in the sample frame.
        2) Broadcasts unit-cell atom offsets and flattens to sites.
        3) Applies an in-box mask against [0, dimensions].
        4) Shifts to centered coordinates using offset - 0.5 * dimensions.

        Args:
            material: Object holding lattice info with attributes
                ``lattice_matrix`` (3x3), ``lattice_atom_cartesian`` (A, 3).
            chunk_position (array-like): Length-3 chunk origin in sample frame.
            chunk_dimensions (array-like): Length-3 integer extents in unit cells.
            stream (cupy.cuda.Stream): CUDA stream to enqueue the work on.

        Returns:
            tuple[cp.ndarray, cp.ndarray]:
                - pos_sel_cp: (M, 3) float32 positions in centered sample frame.
                - idx_sel_cp: (M,) int64 indices into the flattened lattice sites.

        Raises:
            RuntimeError: If CuPy is not available.

        Notes:
            This function only enqueues work on the provided stream and does not
            synchronize. The caller decides when to synchronize.
        """
        if cp is None:
            raise RuntimeError("CuPy is not available but _gpu_stream_chunk was called")

        with stream:
            # Crystal -> sample transforms
            lattice_matrix_cp = cp.asarray(material.lattice_matrix.T, dtype=cp.float32)
            chunk_position_cp = cp.asarray(chunk_position, dtype=cp.float32)

            # Lattice points of this chunk in sample frame
            lattice_positions_C = self.get_flat_grid(chunk_dimensions, use_gpu=True)
            lattice_positions_S = lattice_positions_C @ lattice_matrix_cp + chunk_position_cp

            # Unit-cell atom offsets, broadcast over lattice points, then flatten
            atom_uc = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
            atomic_positions = (lattice_positions_S[:, cp.newaxis, :] +
                                atom_uc[cp.newaxis, :, :]).reshape(-1, 3)

            # In-box mask using [0, dimensions]
            dims = cp.asarray(self.dimensions, dtype=cp.float32)
            mask = ((atomic_positions[:, 0] >= 0) & (atomic_positions[:, 0] <= dims[0]) &
                    (atomic_positions[:, 1] >= 0) & (atomic_positions[:, 1] <= dims[1]) &
                    (atomic_positions[:, 2] >= 0) & (atomic_positions[:, 2] <= dims[2]))

            # Compact on GPU
            pos_sel_cp = atomic_positions[mask, :]

            # Offset shift (centered sample)
            offset_cp = cp.asarray(self.offset, dtype=cp.float32)
            pos_sel_cp = pos_sel_cp + (offset_cp - 0.5 * dims)

            # Selected flattened indices (used to map species on CPU via modulo)
            idx_sel_cp = cp.where(mask)[0].astype(cp.int64, copy=False)

        return pos_sel_cp, idx_sel_cp
    
    def get_chunk_positions(self, material):
        """
        Compute candidate chunk origins and dimensions that intersect the sample.

        Workflow:
          1) Transform the sample corners into the lattice frame and estimate the
             number of lattice units spanned by the sample.
          2) Choose a base uniform chunk size from chunk_volume and lattice_volume,
             then adjust so chunks tile the spanned region.
          3) Build candidate chunk origins, convert to the sample frame, and form
             their 8-corner boxes.
          4) AABB prefilter against [0, dimensions], then run SAT via CFFI on the
             survivors to keep only true intersections.

        Args:
            material: An object with fields:
                - lattice_matrix: 3x3 matrix (crystal-to-sample). Transposed inside.
                - lattice_volume: Scalar volume of one lattice unit cell.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - chunk_positions (float32, shape (M, 3)): Origins of accepted
                  chunks in the sample frame.
                - chunk_dimensions (float32, shape (3,)): Integer-like chunk
                  sizes in lattice units for each axis.

        Notes:
            Uses an AABB prefilter for speed and then a robust SAT test through
            :meth:`parallelepipeds_intersect_cffi`.
        """
        lattice_matrix = material.lattice_matrix.T
        lattice_volume = material.lattice_volume

        inv_lattice_matrix = np.linalg.inv(lattice_matrix)

        # Corners of sample in lattice frame for span measurement
        corners_in_lattice = self.corners @ inv_lattice_matrix

        # Number of lattice units spanning the sample
        lattice_units = np.ceil(
            np.max(corners_in_lattice, axis=0) - np.min(corners_in_lattice, axis=0)
        )

        # Default chunk size in unit cells for each axis
        base_cells = np.floor((self.chunk_volume / lattice_volume) ** (1.0 / 3.0))
        chunk_dimensions = np.zeros(lattice_units.shape, dtype=np.float32) + base_cells

        # Ensure chunk_dimensions are not smaller than the sample span along axes
        size_check = lattice_units > chunk_dimensions
        if not np.all(size_check):
            # Keep dims that are not smaller than sample size, adjust others
            tmp = np.min((chunk_dimensions, lattice_units), axis=0)
            chunk_dimensions[~size_check] = tmp[~size_check]
            # Rebalance remaining dims
            remaining = np.sum(size_check)
            if remaining > 0:
                scale = ((self.chunk_volume / lattice_volume) /
                        np.prod(chunk_dimensions[~size_check])) ** (1.0 / remaining)
                chunk_dimensions[size_check] = np.floor(scale)
                # Ensure aligned division of lattice_units by chunk_dimensions
                chunk_dimensions[size_check] = np.floor(
                    lattice_units[size_check] /
                    np.ceil(lattice_units[size_check] / chunk_dimensions[size_check])
                )

        # How many chunks along each axis
        chunk_units = np.ceil(lattice_units / chunk_dimensions).astype(np.int64)

        # Candidate chunk origins in crystal frame
        chunk_positions_C = self.get_flat_grid(chunk_units, use_gpu=False) * chunk_dimensions

        # Convert to sample frame and center relative to sample [0..dimensions]
        adj_val = (lattice_units * 0.5) - (self.dimensions @ inv_lattice_matrix * 0.5)
        chunk_positions_S = (chunk_positions_C - adj_val) @ lattice_matrix  # (N, 3)

        # Precompute corner offsets once: 8x3 offsets in sample frame for one chunk
        u8 = self.get_unit_corners().astype(np.float32)
        corner_offsets_S = (u8 * chunk_dimensions.astype(np.float32)) @ lattice_matrix.astype(np.float32)

        # All chunk corners in sample frame: (N, 8, 3)
        chunk_corners_S = chunk_positions_S[:, np.newaxis, :] + corner_offsets_S[np.newaxis, :, :]

        # AABB prefilter against [0, dimensions]
        sample_min = np.zeros(3, dtype=np.float32)
        sample_max = self.dimensions.astype(np.float32)

        cc_min = chunk_corners_S.min(axis=1)
        cc_max = chunk_corners_S.max(axis=1)

        aabb_mask = (
            (cc_max[:, 0] >= sample_min[0]) & (cc_min[:, 0] <= sample_max[0]) &
            (cc_max[:, 1] >= sample_min[1]) & (cc_min[:, 1] <= sample_max[1]) &
            (cc_max[:, 2] >= sample_min[2]) & (cc_min[:, 2] <= sample_max[2])
        )

        # No candidates survived the AABB test
        if not np.any(aabb_mask):
            return chunk_positions_S[:0, :], chunk_dimensions

        # SAT on survivors only
        sample_corners_S = (self.get_unit_corners() @ self.matrix)  # 8x3 in sample frame
        sat_mask_sub = self.parallelepipeds_intersect_cffi(
            self._intersect_function,
            self._ffi_object,
            chunk_corners_S[aabb_mask, :, :],
            sample_corners_S,
            eps=1e-12
        )

        # Reconstruct full mask and select survivors
        full_mask = np.zeros(chunk_positions_S.shape[0], dtype=bool)
        full_mask[np.flatnonzero(aabb_mask)] = sat_mask_sub

        chunk_positions_S = chunk_positions_S[full_mask, :]
        return chunk_positions_S.astype(np.float32, copy=False), chunk_dimensions.astype(np.float32, copy=False)
        
    def parallelepipeds_intersect_cffi(self, compiled_code, ffi_object, pts1, pts2, eps=1e-12):
        """
        Run a batched SAT intersection test via the verified CFFI module.

        Converts input arrays to contiguous float64 buffers, passes them to the
        C function, and returns a boolean mask of intersections.

        Args:
            compiled_code: Module returned by ``compile_parallelepipeds_intersect_batch_cffi``
                exposing ``check_parallelepipeds_intersect_batch``.
            ffi_object: CFFI FFI instance used to create C views of NumPy buffers.
            pts1 (np.ndarray): Array of shape (N, 8, 3) or equivalent with the 8
                corners of N boxes in the same frame as ``pts2``.
            pts2 (np.ndarray): Array of shape (8, 3) with corners of the reference box.
            eps (float, optional): Axis-length threshold used by the C code to
                skip near-degenerate separating axes. Defaults to 1e-12.

        Returns:
            np.ndarray: Boolean array of shape (N,) where True means the
            corresponding parallelepiped intersects ``pts2``.
        """
        # Ensure contiguous float64 buffers for direct C access
        pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
        pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
        n = int(pts1.shape[0])

        # Output buffer for integer flags from C (0 or 1)
        results_int = np.zeros(n, dtype=np.int32)

        # Build CFFI pointers without copying
        c_all = ffi_object.from_buffer("double[]", pts1)
        c_arr2 = ffi_object.from_buffer("double[]", pts2)
        c_out = ffi_object.cast("int *", results_int.ctypes.data)

        # Call the verified C function
        compiled_code.check_parallelepipeds_intersect_batch(
            c_all, c_arr2, float(eps), n, c_out
        )
        # Convert 0/1 flags to boolean mask
        return results_int == 1

    def get_lattice_positions(self, material, chunk_position, chunk_dimensions, use_gpu=True):
        """
        Compute lattice point positions in the sample frame for one chunk.

        Args:
            material: Object with ``lattice_matrix`` (3x3). Transposed internally.
            chunk_position (array-like): Length-3 origin for this chunk in the
                sample frame.
            chunk_dimensions (array-like): Integer-like extents (cells) per axis.
            use_gpu (bool, optional): If True and CuPy is available, returns a
                ``cp.ndarray``; otherwise returns a ``np.ndarray``. Defaults to True.

        Returns:
            np.ndarray or cp.ndarray: Array of shape (d0*d1*d2, 3), dtype float32,
            with lattice points transformed into the sample frame.
        """
        lattice_matrix = material.lattice_matrix.T

        if use_gpu and (cp is not None):
            # GPU path
            lattice_matrix_cp = cp.asarray(lattice_matrix, dtype=cp.float32)
            chunk_position_cp = cp.asarray(chunk_position, dtype=cp.float32)

            lattice_positions_C = self.get_flat_grid(chunk_dimensions, use_gpu=True)
            lattice_positions_S = lattice_positions_C @ lattice_matrix_cp + chunk_position_cp
            return lattice_positions_S

        else:
            # CPU path (single-precision for consistency)
            lattice_matrix_np = lattice_matrix.astype(np.float32)
            chunk_position_np = np.array(chunk_position, dtype=np.float32)

            lattice_positions_C = self.get_flat_grid(chunk_dimensions, use_gpu=False)
            lattice_positions_S = lattice_positions_C @ lattice_matrix_np + chunk_position_np
            return lattice_positions_S

    def get_atomic_data(
        self,
        material,
        chunk_position,
        chunk_dimensions,
        use_gpu=True,
        stream=None,
        return_on_gpu=False,
        lattice_atom_cartesian_cp=None,
        offset_gpu=None,
        dim_half_gpu=None
    ):
        """
        Build atom positions and species for a single geometric chunk.

        On GPU:
            - Expands lattice points by the unit-cell atom offsets,
              masks to [0, dimensions], applies centering shift (offset - 0.5*dimensions),
              and optionally returns GPU arrays without host copies.
        On CPU:
            - Performs the equivalent operations using NumPy and returns host arrays.

        Args:
            material: Object with fields:
                - lattice_atom_cartesian: (A, 3) array of unit-cell atom offsets.
                - species: 1-D array-like with A species labels for a unit cell.
            chunk_position (array-like): Length-3 origin in the sample frame.
            chunk_dimensions (array-like): Integer-like extents (cells) per axis.
            use_gpu (bool, optional): Enable CuPy path if available. Defaults to True.
            stream (cupy.cuda.Stream | None, optional): CUDA stream for queuing work.
            return_on_gpu (bool, optional): If True, returns GPU arrays
                (positions_cp, mask_cp, site_count) for deferred host processing.
            lattice_atom_cartesian_cp (cp.ndarray | None, optional): Preloaded
                device array of ``material.lattice_atom_cartesian`` to avoid
                re-uploading per chunk.
            offset_gpu (cp.ndarray | None, optional): Preloaded device copy of
                ``self.offset``.
            dim_half_gpu (cp.ndarray | None, optional): Preloaded device copy of
                ``self.dimensions * 0.5``.

        Returns:
            tuple:
                If ``return_on_gpu`` is False (default):
                    (positions_np, species_np)
                If ``return_on_gpu`` is True:
                    (positions_cp, mask_cp, site_count)

        Notes:
            This method does not synchronize the provided stream; callers should
            synchronize as needed when ``return_on_gpu`` is True.
        """
        use_gpu = (use_gpu and (cp is not None))

        if use_gpu:
            # Use provided stream or default null stream
            s = stream if (stream is not None) else cp.cuda.Stream.null

            with s:
                # Preload invariant device arrays if not provided
                if lattice_atom_cartesian_cp is None:
                    lattice_atom_cartesian_cp = cp.asarray(
                        material.lattice_atom_cartesian, dtype=cp.float32
                    )
                if offset_gpu is None:
                    offset_gpu = cp.asarray(self.offset, dtype=cp.float32)
                if dim_half_gpu is None:
                    dim_half_gpu = cp.asarray(self.dimensions * 0.5, dtype=cp.float32)

                # Lattice positions in this chunk on GPU
                lattice_positions_cp = self.get_lattice_positions(
                    material, chunk_position, chunk_dimensions, use_gpu=True
                )  # uses current stream

                # Expand to atom sites, then flatten
                atomic_positions_S = (
                    lattice_positions_cp[:, cp.newaxis, :] + lattice_atom_cartesian_cp[cp.newaxis, :, :]
                ).reshape(-1, 3)

                # In-box mask against [0, dimensions]
                dims_gpu = cp.asarray(self.dimensions, dtype=cp.float32)
                mask = (
                    (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= dims_gpu[0]) &
                    (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= dims_gpu[1]) &
                    (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= dims_gpu[2])
                )

                # Filter and apply center/offset
                atomic_positions_S = atomic_positions_S[mask, :]
                atomic_positions_S += (offset_gpu - dim_half_gpu)

                if return_on_gpu:
                    # Defer host copies; caller will handle mask/species tiling
                    site_count = int(lattice_positions_cp.shape[0])
                    return atomic_positions_S, mask, site_count

                # Host path: bring back positions and mask, then map species on CPU
                mask_np = mask.get()
                positions_np = atomic_positions_S.get()

                atomic_species = self._build_species(material.species, int(lattice_positions_cp.shape[0]), mask_np)

                # Free temporary device allocations
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
                return positions_np.astype(np.float32, copy=False), atomic_species

        # CPU branch (unchanged semantics)
        lattice_atom_cartesian_np = material.lattice_atom_cartesian.astype(np.float32)
        lattice_positions_np = self.get_lattice_positions(
            material, chunk_position, chunk_dimensions, use_gpu=False
        )

        atomic_positions_S = (
            lattice_positions_np[:, np.newaxis, :].astype(np.float32) +
            lattice_atom_cartesian_np[np.newaxis, :, :]
        ).reshape(-1, 3)

        # In-box mask on CPU
        mask = (
            (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= self.dimensions[0]) &
            (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= self.dimensions[1]) &
            (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= self.dimensions[2])
        )

        # Apply mask and center/offset shift
        atomic_positions_S = atomic_positions_S[mask, :].astype(np.float32)
        atomic_species = self._build_species(material.species, lattice_positions_np.shape[0], mask)

        offset_np = self.offset.astype(np.float32)
        dim_half_np = (self.dimensions * 0.5).astype(np.float32)
        atomic_positions_S += (offset_np - dim_half_np)
        gc.collect()
        return atomic_positions_S, atomic_species

    def generate_sample_single(
        self,
        material,
        flush_size=100000000,
        use_gpu=True,
        gpu_streams=4,
        writer_threads=3,
        n_gpus=None,
        alloy_species=None,
        alloy_concentrations=None,
        alloy_seed=None
    ):
        """
        Generate and persist the sample to disk in fixed-size chunks.

        The function:
          - Computes geometric chunks once using :meth:`get_chunk_positions`.
          - Streams generated atoms into CPU buffers of length ``flush_size``.
          - Writes chunked ``.npy`` files for positions and species via a
            thread pool to overlap I/O.
          - Runs a multi-stream GPU path if available; otherwise, or upon GPU
            failure, it falls back to a pure-CPU path without losing progress.
          - Supports multi-GPU acceleration when n_gpus > 1.

        Args:
            material: Object with lattice and unit-cell definitions used by
                :meth:`get_atomic_data`.
            flush_size (int, optional): Number of atoms per on-disk chunk.
                Defaults to 100_000_000.
            use_gpu (bool, optional): Enable the GPU generation path if CuPy and
                a CUDA device are available. Defaults to True.
            gpu_streams (int, optional): Number of concurrent CUDA streams to
                pipeline GPU work per GPU. Defaults to 4.
            writer_threads (int, optional): Number of I/O worker threads for
                writing chunks to disk. Defaults to 3.
            n_gpus (int, optional): Number of GPUs to use. Default (None) uses
                all available GPUs. Set to 1 to force single-GPU mode.
            alloy_species (list[str] | None, optional): Element symbols for
                random alloy assignment (e.g. ``["Fe", "Co"]``). Each atom site
                is randomly assigned one of these species instead of the CIF
                species. None (default) uses the CIF species.
            alloy_concentrations (list[float] | None, optional): Per-species
                probabilities (must sum to 1.0). Must match the length of
                ``alloy_species``. None defaults to equal probabilities.
            alloy_seed (int | None, optional): Seed for the random number
                generator used for alloy species assignment. None (default)
                gives non-deterministic results.

        Returns:
            None

        Notes:
            On success, updates ``self._chunk_total`` with the number of files
            written. Progress already written is preserved if the GPU path
            encounters an error and falls back to CPU.
        """
        # Set up alloy mode if requested
        self._setup_alloy(alloy_species, alloy_concentrations, alloy_seed)

        # Build geometric chunks once
        self._chunk_positions, self._chunk_dimensions = self.get_chunk_positions(material)
        num_geom = int(self.chunk_positions.shape[0])

        # Early out if there is nothing to do
        if num_geom == 0:
            self._chunk_total = 0
            return

        if self._streaming_mode:
            self._streaming_use_gpu = use_gpu  # Remember GPU preference for streaming
            self._streaming_material = material
            self._compute_streaming_chunk_mapping(material, flush_size)
            return

        flush_size = int(flush_size)

        from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

        # Thread pool for disk writes
        def _write_chunk(idx, pos_arr, spc_arr):
            self.write_chunk_positions(pos_arr, idx)
            self.write_chunk_species(spc_arr, idx)
            return idx

        writer_pool = ThreadPoolExecutor(
            max_workers=max(1, int(writer_threads)),
            thread_name_prefix="writer"
        )
        pending_writes = []

        # CPU-side streaming buffers shared by both GPU and CPU paths
        buf_pos = None
        buf_spc = None
        fill = 0
        file_chunk_index = 0

        def _accumulate_to_buffers(pos_np, spc_np):
            nonlocal buf_pos, buf_spc, fill, file_chunk_index, pending_writes
            n = int(pos_np.shape[0])
            if n == 0:
                return
            if buf_pos is None:
                buf_pos = np.empty((flush_size, 3), dtype=pos_np.dtype)
                buf_spc = np.empty((flush_size,), dtype=np.asarray(spc_np).dtype)

            start = 0
            while start < n:
                space = flush_size - fill
                take = (n - start) if (n - start) < space else space

                buf_pos[fill:fill + take] = pos_np[start:start + take]
                buf_spc[fill:fill + take] = spc_np[start:start + take]

                fill += take
                start += take

                if fill == flush_size:
                    file_chunk_index += 1
                    # Copy slices to isolate from subsequent overwrites
                    pending_writes.append(
                        writer_pool.submit(_write_chunk, file_chunk_index, buf_pos.copy(), buf_spc.copy())
                    )
                    fill = 0  # reset

        def _flush_tail():
            nonlocal fill, file_chunk_index, pending_writes
            if fill > 0:
                file_chunk_index += 1
                pending_writes.append(
                    writer_pool.submit(_write_chunk, file_chunk_index, buf_pos[:fill].copy(), buf_spc[:fill].copy())
                )
                fill = 0

        # 3) Decide if we can and should use GPU, and how many
        gpu_ok = False
        available_gpus = 0
        if use_gpu and (cp is not None):
            try:
                available_gpus = int(cp.cuda.runtime.getDeviceCount())
                gpu_ok = (available_gpus > 0)
            except Exception:
                gpu_ok = False

        # Determine number of GPUs to use
        if n_gpus is None:
            use_n_gpus = max(1, available_gpus)
        else:
            use_n_gpus = min(int(n_gpus), available_gpus) if available_gpus > 0 else 0

        # GPU path
        drained_count = 0  # number of geom-chunks fully drained into CPU buffers

        # Multi-GPU path: distribute chunks across multiple GPUs
        if gpu_ok and use_n_gpus > 1:
            try:
                n_streams = max(1, int(gpu_streams))
                buf_lock = threading.Lock()

                # Round-robin distribute chunks to GPUs
                shards = [[] for _ in range(use_n_gpus)]
                for i in range(num_geom):
                    shards[i % use_n_gpus].append(i)

                # Shared state for multi-GPU accumulation
                gpu_errors = [None] * use_n_gpus

                def gpu_worker(dev_id, my_chunks):
                    """Process assigned chunks on a specific GPU."""
                    nonlocal drained_count
                    try:
                        cp.cuda.Device(dev_id).use()

                        # Per-GPU stream ring
                        streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

                        # Preload GPU invariants once per device
                        lattice_cp = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
                        offset_cp = cp.asarray(self.offset, dtype=cp.float32)
                        dim_half_cp = cp.asarray(self.dimensions * 0.5, dtype=cp.float32)

                        inflight = []
                        enq_idx = 0

                        def _enqueue_local(chunk_i, s):
                            with s:
                                pos_cp, mask_cp, site_count = self.get_atomic_data(
                                    material,
                                    self.chunk_positions[chunk_i, :],
                                    self._chunk_dimensions,
                                    use_gpu=True,
                                    stream=s,
                                    return_on_gpu=True,
                                    lattice_atom_cartesian_cp=lattice_cp,
                                    offset_gpu=offset_cp,
                                    dim_half_gpu=dim_half_cp
                                )
                            ev = cp.cuda.Event()
                            ev.record(s)
                            inflight.append({
                                "event": ev,
                                "pos_cp": pos_cp,
                                "mask_cp": mask_cp,
                                "site_count": site_count
                            })

                        def _drain_local():
                            nonlocal drained_count
                            task = inflight.pop(0)
                            task["event"].synchronize()

                            pos_np = task["pos_cp"].get()
                            mask_np = task["mask_cp"].get()

                            spc_np = self._build_species(material.species, task["site_count"], mask_np)

                            # Thread-safe accumulation
                            with buf_lock:
                                _accumulate_to_buffers(pos_np, spc_np)
                                drained_count += 1

                            del pos_np, mask_np, spc_np, task

                        # Fill-drain loop for this GPU's chunks
                        while (enq_idx < len(my_chunks)) or inflight:
                            while (enq_idx < len(my_chunks)) and (len(inflight) < n_streams):
                                s = streams[enq_idx % n_streams]
                                _enqueue_local(my_chunks[enq_idx], s)
                                enq_idx += 1

                            if inflight:
                                _drain_local()

                        # Cleanup this GPU's memory
                        try:
                            cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass

                    except Exception as e:
                        gpu_errors[dev_id] = e

                # Launch one thread per GPU
                threads = []
                for dev_id in range(use_n_gpus):
                    if shards[dev_id]:
                        t = threading.Thread(
                            target=gpu_worker,
                            args=(dev_id, shards[dev_id]),
                            name=f"GPU-{dev_id}"
                        )
                        t.start()
                        threads.append(t)

                # Wait for all GPUs to complete
                for t in threads:
                    t.join()

                # Check for errors (if any GPU failed, fall back to CPU for remaining)
                if any(e is not None for e in gpu_errors):
                    gpu_ok = False

            except Exception:
                try:
                    for dev_id in range(use_n_gpus):
                        cp.cuda.Device(dev_id).use()
                        cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                gpu_ok = False

        # Single-GPU path
        elif gpu_ok:
            try:
                n_streams = max(1, int(gpu_streams))
                streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

                # Preload invariants once
                lattice_atom_cartesian_cp = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
                offset_gpu = cp.asarray(self.offset, dtype=cp.float32)
                dim_half_gpu = cp.asarray(self.dimensions * 0.5, dtype=cp.float32)

                inflight = []  # list of dicts: {event, pos_cp, mask_cp, site_count}
                enqueue_idx = 0  # next geom-chunk index to enqueue

                def _enqueue(i, s):
                    with s:
                        pos_cp, mask_cp, site_count = self.get_atomic_data(
                            material,
                            self.chunk_positions[i, :],
                            self._chunk_dimensions,
                            use_gpu=True,
                            stream=s,
                            return_on_gpu=True,
                            lattice_atom_cartesian_cp=lattice_atom_cartesian_cp,
                            offset_gpu=offset_gpu,
                            dim_half_gpu=dim_half_gpu
                        )
                    ev = cp.cuda.Event()
                    ev.record(s)
                    inflight.append({
                        "event": ev,
                        "pos_cp": pos_cp,
                        "mask_cp": mask_cp,
                        "site_count": site_count
                    })

                def _drain_one():
                    nonlocal drained_count
                    task = inflight.pop(0)
                    task["event"].synchronize()

                    # Bring results to host
                    pos_np = task["pos_cp"].get()
                    mask_np = task["mask_cp"].get()

                    # Build species vector on host
                    spc_np = self._build_species(material.species, task["site_count"], mask_np)

                    _accumulate_to_buffers(pos_np, spc_np)
                    drained_count += 1

                    # Cleanup local references
                    del pos_np, mask_np, spc_np, task

                # Fill-drain loop
                while (enqueue_idx < num_geom) or inflight:
                    # Enqueue up to ring capacity
                    while (enqueue_idx < num_geom) and (len(inflight) < n_streams):
                        s = streams[enqueue_idx % n_streams]
                        _enqueue(enqueue_idx, s)
                        enqueue_idx += 1

                    # Drain at least one task if any are inflight
                    if inflight:
                        _drain_one()

                # Free GPU memory pool
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError, RuntimeError, ValueError) as _gpu_err:
                # GPU failure -> fall back to CPU for remaining work
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                gpu_ok = False  # signal CPU fallback

            except Exception:
                # Any other unexpected GPU-side error -> CPU fallback
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                gpu_ok = False

        # CPU path (or GPU fallback remainder)
        if not gpu_ok:
            start_i = drained_count  # redo from first not-drained chunk
            for i in range(start_i, num_geom):
                pos_np, spc_np = self.get_atomic_data(
                    material,
                    self.chunk_positions[i, :],
                    self._chunk_dimensions,
                    use_gpu=False
                )
                _accumulate_to_buffers(pos_np, spc_np)
                del pos_np, spc_np

        # Flush trailing partial buffer and finish
        _flush_tail()
        wait(pending_writes, return_when=ALL_COMPLETED)
        writer_pool.shutdown(wait=True)

        # Update metadata
        self._chunk_total = int(file_chunk_index)
        self._teardown_alloy()
        return

    def input_voronoi_seed(self, seeds):
        """
        Set user-provided Voronoi seed map.

        Args:
            seeds (array-like): shape (G, 3) array of seed positions in the same
                coordinate frame as saved atomic positions (i.e., with the sample
                centered at `offset - 0.5*dimensions`).

        Returns:
            np.ndarray: stored seeds (G, 3), float32
        """
        seeds = np.asarray(seeds, dtype=np.float32).reshape(-1, 3)
        if seeds.shape[0] == 0:
            raise ValueError("input_voronoi_seed: empty seed array")
        self._grain_seeds = seeds
        self._grain_count = int(seeds.shape[0])
        return self._grain_seeds

    def input_grain_orientation(self, orientation_matrices):
        """
        Set user-provided grain orientation map.

        Args:
            orientation_matrices (array-like): shape (G, 3, 3). Each 3x3 is a
                rotation/transform matrix applied to the crystal lattice of each grain.

        Returns:
            np.ndarray: stored orientation matrices (G, 3, 3), float32
        """
        R = np.asarray(orientation_matrices, dtype=np.float32)
        if R.ndim != 3 or R.shape[1:] != (3, 3):
            raise ValueError("input_grain_orientation expects array of shape (G, 3, 3)")
        if (self._grain_count is not None) and (int(R.shape[0]) != int(self._grain_count)):
            raise ValueError("orientation count does not match number of seeds")
        self._grain_orientations = R
        self._grain_count = int(R.shape[0])
        return self._grain_orientations

    def generate_voronoi_seeds(self, n_grains, method="uniform", random_seed=None):
        """
        Generate Voronoi seeds inside the sample box.

        Two methods:
        - 'uniform': build a near-cubic grid, place one seed per cell center,
                    then jitter each within its cell without crossing borders.
        - 'random' : sample uniform i.i.d. seed positions in the box.

        Args:
            n_grains (int): number of grains (seeds).
            method (str): 'uniform' (default) or 'random'.
            random_seed (int|None): RNG seed for reproducibility.

        Returns:
            np.ndarray: seeds (G, 3) in world coordinates, dtype float32.
        """
        if n_grains is None or int(n_grains) <= 0:
            raise ValueError("generate_voronoi_seeds: n_grains must be positive")

        rng = np.random.RandomState(None if random_seed is None else int(random_seed))

        dims = np.asarray(self.dimensions, dtype=np.float32)
        box_min = (self.offset - 0.5 * dims).astype(np.float32)
        box_max = box_min + dims

        if method not in ("uniform", "random"):
            method = "uniform"

        if method == "uniform":
            # Choose integer grid dims close to cube root
            g = int(np.round(n_grains ** (1.0 / 3.0)))
            if g < 1: g = 1
            # Expand to reach or exceed n_grains
            nx = g
            ny = g
            nz = int(np.ceil(float(n_grains) / float(nx * ny)))
            while nx * ny * nz < n_grains:
                # Grow the smallest dimension
                if nx <= ny and nx <= nz:
                    nx += 1
                elif ny <= nx and ny <= nz:
                    ny += 1
                else:
                    nz += 1

            hx, hy, hz = dims[0] / nx, dims[1] / ny, dims[2] / nz
            # cell centers
            cx = box_min[0] + (np.arange(nx, dtype=np.float32) + 0.5) * hx
            cy = box_min[1] + (np.arange(ny, dtype=np.float32) + 0.5) * hy
            cz = box_min[2] + (np.arange(nz, dtype=np.float32) + 0.5) * hz
            X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
            seeds = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)

            # jitter within each cell but keep uniqueness per cell
            jitter_scale = np.array([0.45 * hx, 0.45 * hy, 0.45 * hz], dtype=np.float32)
            jitter = (rng.rand(seeds.shape[0], 3).astype(np.float32) * 2.0 - 1.0) * jitter_scale
            seeds = seeds + jitter

            # Trim to exact count if grid overshoots
            if seeds.shape[0] > n_grains:
                sel = rng.choice(seeds.shape[0], size=int(n_grains), replace=False)
                seeds = seeds[sel, :]

        else:
            # Pure random uniform sampling
            seeds = rng.uniform(low=box_min, high=box_max, size=(int(n_grains), 3)).astype(np.float32)

        self._grain_seeds = seeds
        self._grain_count = int(seeds.shape[0])
        return self._grain_seeds

    @staticmethod
    def _random_rotation_matrix(rng):
        """
        Return a Shoemake-style random rotation matrix (uniform on SO(3)).
        """
        u1, u2, u3 = rng.rand(3)
        q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
        q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
        q3 = np.sqrt(u1)       * np.sin(2.0 * np.pi * u3)
        q4 = np.sqrt(u1)       * np.cos(2.0 * np.pi * u3)
        # quaternion to rotation
        x, y, z, w = q1, q2, q3, q4
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)]
        ], dtype=np.float32)
        return R

    @staticmethod
    def _align_rotation_from_to(a, b):
        """
        Return a rotation aligning vector a to b (both 3, any length).
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-20)
        b = b / (np.linalg.norm(b) + 1e-20)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c > 1.0: c = 1.0
        if c < -1.0: c = -1.0
        s = np.linalg.norm(v)
        if s < 1e-12:
            # vectors are parallel (or antiparallel)
            if c > 0:
                return np.eye(3, dtype=np.float32)
            # 180 deg: pick any orthogonal axis
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = axis - a * np.dot(axis, a)
            axis = axis / (np.linalg.norm(axis) + 1e-20)
            return sample.get_rotation(axis, np.pi).astype(np.float32)
        kmat = np.array([[    0, -v[2],  v[1]],
                        [ v[2],     0, -v[0]],
                        [-v[1],  v[0],     0]], dtype=np.float64)
        R = np.eye(3) + kmat + (kmat @ kmat) * ((1.0 - c) / (s * s + 1e-20))
        return R.astype(np.float32)

    def _orientation_matrices_from_mode(self, n_grains, mode="random",
                                        texture_axis=(0.0, 0.0, 1.0),
                                        spread_deg=5.0,
                                        random_seed=None):
        """
        Build per-grain orientation matrices.
        """
        rng = np.random.RandomState(None if random_seed is None else int(random_seed))
        R = np.zeros((int(n_grains), 3, 3), dtype=np.float32)
        if mode not in ("random", "textured"):
            mode = "random"

        if mode == "random":
            for g in range(int(n_grains)):
                R[g] = self._random_rotation_matrix(rng)
            return R

        # textured: align lattice z to texture_axis, then apply small-angle Gaussian tilt and random twist
        t = np.asarray(texture_axis, dtype=np.float64)
        if np.linalg.norm(t) < 1e-12:
            t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        t = t / np.linalg.norm(t)
        R0 = self._align_rotation_from_to(np.array([0.0, 0.0, 1.0], dtype=np.float64), t).astype(np.float32)

        # build two orthonormal vectors spanning plane perp to t
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(tmp, t)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        e1 = tmp - t * np.dot(tmp, t)
        e1 = e1 / (np.linalg.norm(e1) + 1e-20)
        e2 = np.cross(t, e1)

        spread_rad = float(spread_deg) * np.pi / 180.0
        for g in range(int(n_grains)):
            # random twist about t
            twist = rng.uniform(0.0, 2.0*np.pi)
            Rt = sample.get_rotation(t, twist).astype(np.float32)

            # small tilt around axis in plane perp to t
            phi = rng.uniform(0.0, 2.0*np.pi)
            axis = (np.cos(phi) * e1 + np.sin(phi) * e2)
            angle = abs(rng.normal(loc=0.0, scale=spread_rad))
            Rtilt = sample.get_rotation(axis, angle).astype(np.float32)

            R[g] = (Rtilt @ R0 @ Rt).astype(np.float32)
        return R

    @staticmethod
    def _rotate_material_like(material, R):
        """
        Construct a lightweight material-like object with rotated lattice.
        """
        # R is 3x3 in sample frame
        mat = type("MatLike", (), {})()
        # rotate lattice vectors (columns) -> R @ lattice_matrix
        mat.lattice_matrix = (R @ np.asarray(material.lattice_matrix, dtype=np.float32)).astype(np.float32)
        # rotate unit cell atom offsets
        mat.lattice_atom_cartesian = (np.asarray(material.lattice_atom_cartesian, dtype=np.float32) @ R.T).astype(np.float32)
        # copy-through scalars/arrays
        mat.lattice_volume = getattr(material, "lattice_volume", None)
        mat.species = np.asarray(getattr(material, "species", []))
        return mat

    @staticmethod
    def _voronoi_min_index_cpu(positions_np, seeds_np):
        """
        Return argmin seed index for each position (CPU; streaming over seeds).
        """
        N = positions_np.shape[0]
        G = seeds_np.shape[0]
        min_d2 = np.full((N,), np.inf, dtype=np.float64)
        min_idx = np.full((N,), -1, dtype=np.int32)
        # loop seeds to keep memory O(N)
        for g in range(G):
            d = positions_np - seeds_np[g, :][None, :]
            d2 = np.sum(d * d, axis=1, dtype=np.float64)
            mask = d2 < min_d2
            min_idx[mask] = g
            min_d2[mask] = d2[mask]
        return min_idx

    @staticmethod
    def _voronoi_min_index_gpu(positions_cp, seeds_cp, seed_tile=512, pos_tile=None):
        """
        Argmin over seeds using GEMM tiles:
        d2 = ||p||^2[:,None] + ||s||^2[None,:] - 2 * P @ S^T
        Tiles over seeds and positions to control memory.
        """
        N = int(positions_cp.shape[0])
        G = int(seeds_cp.shape[0])

        # tile positions too if N is huge
        if pos_tile is None or pos_tile <= 0:
            pos_tile = N

        best_d2 = cp.full((N,), cp.inf, dtype=cp.float32)
        best_idx = cp.full((N,), -1, dtype=cp.int32)

        # Precompute norms once
        p2 = cp.sum(positions_cp * positions_cp, axis=1).astype(cp.float32)  # (N,)
        s2 = cp.sum(seeds_cp * seeds_cp, axis=1).astype(cp.float32)          # (G,)

        for p0 in range(0, N, pos_tile):
            p1 = min(p0 + pos_tile, N)
            P = positions_cp[p0:p1, :]                    # (nP, 3)
            p2_blk = p2[p0:p1]                            # (nP,)
            # local best for this position tile
            lbest_d2 = cp.full((p1 - p0,), cp.inf, dtype=cp.float32)
            lbest_idx = cp.full((p1 - p0,), -1, dtype=cp.int32)

            for s0 in range(0, G, seed_tile):
                s1 = min(s0 + seed_tile, G)
                S = seeds_cp[s0:s1, :]                    # (nS, 3)
                # (nP, nS) via cuBLAS
                PS = P @ S.T                              # float32 GEMM
                # d2_block = p2 + s2 - 2·PS (broadcasted)
                d2_blk = (p2_blk[:, None] + s2[s0:s1][None, :] - 2.0 * PS)

                # argmin across seeds in this tile
                idx_local = cp.argmin(d2_blk, axis=1)             # (nP,)
                d2_local  = d2_blk[cp.arange(p1 - p0), idx_local]

                # keep better results
                mask = d2_local < lbest_d2
                lbest_d2 = cp.where(mask, d2_local, lbest_d2)
                lbest_idx = cp.where(mask, (idx_local + s0).astype(cp.int32), lbest_idx)

            # commit tile results to global
            better = lbest_d2 < best_d2[p0:p1]
            best_d2[p0:p1] = cp.where(better, lbest_d2, best_d2[p0:p1])
            best_idx[p0:p1] = cp.where(better, lbest_idx, best_idx[p0:p1])

        return best_idx

    @staticmethod
    def _voronoi_assign_gpu_bounded(positions_cp, seeds_cp, memory_fraction=0.5):
        """
        Memory-bounded GPU Voronoi assignment using correct memory calculation.

        Guarantees GPU memory usage stays below memory_fraction of total GPU memory
        regardless of N (positions) or G (grains/seeds).

        Args:
            positions_cp: (N, 3) CuPy array of atom positions.
            seeds_cp: (G, 3) CuPy array of Voronoi seed positions.
            memory_fraction: Fraction of total GPU memory to use (default 0.5).

        Returns:
            cp.ndarray: (N,) int32 array of grain indices for each position.
        """
        N = int(positions_cp.shape[0])
        G = int(seeds_cp.shape[0])

        # Get available GPU memory
        try:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            budget_bytes = int(memory_fraction * total_bytes)
        except Exception:
            budget_bytes = 2 * 1024 * 1024 * 1024  # Default 2GB

        # Fixed seed_tile (controls inner loop granularity)
        seed_tile = min(512, G)

        # Calculate pos_tile based on ACTUAL memory requirements:
        # Per iteration allocates: d2_blk + PS + intermediates
        # = pos_tile × seed_tile × (4 + 4 + 4) bytes ≈ pos_tile × seed_tile × 16 bytes
        # Plus: lbest_d2 (pos_tile × 4) + lbest_idx (pos_tile × 4) ≈ pos_tile × 8
        # Total per iteration ≈ pos_tile × (seed_tile × 16 + 8)
        bytes_per_pos_iteration = seed_tile * 16 + 8

        # Reserve memory for global arrays: best_d2 + best_idx = N × 8 bytes
        # And input arrays: positions (N × 12) + seeds (G × 12)
        reserved_bytes = N * 8 + N * 12 + G * 12
        available_bytes = max(budget_bytes - reserved_bytes, budget_bytes // 2)

        # Calculate safe pos_tile
        pos_tile = max(32768, available_bytes // max(bytes_per_pos_iteration, 1))
        pos_tile = min(pos_tile, N)  # Don't exceed actual positions

        return sample._voronoi_min_index_gpu(positions_cp, seeds_cp,
                                              seed_tile=seed_tile, pos_tile=pos_tile)

    @staticmethod
    def _voronoi_assign_gpu_streaming(positions_cp, seeds_cp, memory_fraction=0.5):
        """
        Fully streaming GPU Voronoi for arbitrarily large samples.

        Processes positions in sub-chunks, never allocating more than
        memory_fraction of GPU memory regardless of N or G.

        Args:
            positions_cp: (N, 3) CuPy array of atom positions.
            seeds_cp: (G, 3) CuPy array of Voronoi seed positions.
            memory_fraction: Fraction of total GPU memory to use (default 0.5).

        Returns:
            cp.ndarray: (N,) int32 array of grain indices for each position.
        """
        N = int(positions_cp.shape[0])
        G = int(seeds_cp.shape[0])

        try:
            free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
            budget_bytes = int(memory_fraction * total_bytes)
        except Exception:
            budget_bytes = 2 * 1024 * 1024 * 1024  # Default 2GB

        seed_tile = min(512, G)

        # For streaming: we can only hold a portion of positions at a time
        # Each sub-chunk needs: positions (sub_N × 12) + best arrays (sub_N × 8)
        # Plus working memory for GEMM: sub_N × seed_tile × 16
        bytes_per_pos = 12 + 8 + seed_tile * 16  # ~8KB per position for seed_tile=512

        # Reserve for seeds array
        seeds_bytes = G * 12
        available_bytes = max(budget_bytes - seeds_bytes, budget_bytes // 2)

        # Calculate sub-chunk size
        sub_chunk_size = max(32768, available_bytes // max(bytes_per_pos, 1))
        sub_chunk_size = min(sub_chunk_size, N)

        # If sub_chunk covers all positions, use the faster non-streaming version
        if sub_chunk_size >= N:
            return sample._voronoi_assign_gpu_bounded(positions_cp, seeds_cp, memory_fraction)

        # Process in sub-chunks, accumulating results
        result_idx = cp.empty((N,), dtype=cp.int32)

        for start in range(0, N, sub_chunk_size):
            end = min(start + sub_chunk_size, N)
            pos_sub = positions_cp[start:end]

            # Compute Voronoi for this sub-chunk using the bounded version
            idx_sub = sample._voronoi_min_index_gpu(pos_sub, seeds_cp,
                                                     seed_tile=seed_tile,
                                                     pos_tile=end - start)
            result_idx[start:end] = idx_sub

            # Free intermediate memory
            del pos_sub, idx_sub
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass

        return result_idx

    def generate_sample_poly(
        self,
        material,
        n_grains=None,
        voronoi_method="uniform",
        randomness_seed=None,
        orientation_mode="random",
        texture_axis=(0.0, 0.0, 1.0),
        texture_spread_deg=5.0,
        flush_size=100000000,
        use_gpu=True,
        gpu_streams=4,
        grain_workers=None,
        writer_threads=3,
        n_gpus=None,
        alloy_species=None,
        alloy_concentrations=None,
        alloy_seed=None
    ):
        """
        Generate and persist a polycrystalline sample using Voronoi grains.

        Each grain gets:
        - a Voronoi cell (from seeds provided or generated),
        - an orientation (random or textured),
        - its own lattice generation over the sample,
        - masking to the Voronoi region.

        Supports multi-GPU acceleration when n_gpus > 1.

        Args:
            material: object with lattice/unit-cell, see `generate_sample_single`.
            n_grains (int|None): if provided, defines or overrides number of grains.
                If seeds already exist via `input_voronoi_seed`, this may be None.
                If neither is set, defaults to 8.
            voronoi_method (str): 'uniform' (default) or 'random' when generating seeds.
            randomness_seed (int|None): RNG seed for reproducibility of seeds and, if
                needed, random orientations.
            orientation_mode (str): 'random' (default) or 'textured'.
            texture_axis (3,): for 'textured' mode, principal texture axis.
            texture_spread_deg (float): Gaussian stddev (degrees) of misorientation
                cone around `texture_axis` (small angle approx).
            flush_size (int): atoms per on-disk chunk.
            use_gpu (bool): enable GPU path if CuPy + device available.
            gpu_streams (int): CUDA streams for GPU path per GPU.
            grain_workers (int|None): CPU worker threads; default=min(G, os.cpu_count()).
            writer_threads (int): I/O worker threads.
            n_gpus (int, optional): Number of GPUs to use. Default (None) uses
                all available GPUs. Set to 1 to force single-GPU mode.
            alloy_species (list[str] | None, optional): Element symbols for
                random alloy assignment (e.g. ``["Fe", "Co"]``). Each atom site
                is randomly assigned one of these species instead of the CIF
                species. None (default) uses the CIF species.
            alloy_concentrations (list[float] | None, optional): Per-species
                probabilities (must sum to 1.0). Must match the length of
                ``alloy_species``. None defaults to equal probabilities.
            alloy_seed (int | None, optional): Seed for the random number
                generator used for alloy species assignment. None (default)
                gives non-deterministic results.

        Returns:
            None (writes chunked arrays and metadata to disk).
        """
        # Set up alloy mode if requested
        self._setup_alloy(alloy_species, alloy_concentrations, alloy_seed)

        self._sample_type = "poly"

        # Seeds
        if self._grain_seeds is None:
            G = int(8 if (n_grains is None) else n_grains)
            self.generate_voronoi_seeds(G, method=voronoi_method, random_seed=randomness_seed)
        else:
            G = int(self._grain_seeds.shape[0])
            if n_grains is not None and int(n_grains) != G:
                raise ValueError("n_grains does not match existing seed map")
        seeds_np = np.asarray(self._grain_seeds, dtype=np.float32)

        # Orientations
        if self._grain_orientations is None:
            R = self._orientation_matrices_from_mode(
                G,
                mode=orientation_mode,
                texture_axis=texture_axis,
                spread_deg=texture_spread_deg,
                random_seed=randomness_seed
            )
            self._grain_orientations = R
        else:
            R = np.asarray(self._grain_orientations, dtype=np.float32)
            if R.shape[0] != G:
                raise ValueError("orientation map size does not match number of seeds")

        # Compute chunk geometry (needed for both streaming and disk modes)
        self._chunk_positions, self._chunk_dimensions = self.get_chunk_positions(material)
        num_geom = int(self.chunk_positions.shape[0])

        # STREAMING MODE: store material reference, compute mapping, and exit without writing files
        if self._streaming_mode:
            self._streaming_use_gpu = use_gpu  # Remember GPU preference for streaming
            self._streaming_material = material
            self._compute_streaming_chunk_mapping(material, flush_size)
            return

        # Writer and global accumulation buffers (same pattern as generate_sample_single)
        from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
        import threading

        flush_size = int(flush_size)

        def _write_chunk(idx, pos_arr, spc_arr):
            self.write_chunk_positions(pos_arr, idx)
            self.write_chunk_species(spc_arr, idx)
            return idx

        writer_pool = ThreadPoolExecutor(
            max_workers=max(1, int(writer_threads)),
            thread_name_prefix="writer"
        )
        pending_writes = []

        buf_pos = None
        buf_spc = None
        fill = 0
        file_chunk_index = 0
        lock = threading.Lock()

        def _accumulate_to_buffers(pos_np, spc_np):
            nonlocal buf_pos, buf_spc, fill, file_chunk_index, pending_writes
            n = int(pos_np.shape[0])
            if n == 0:
                return
            with lock:
                if buf_pos is None:
                    buf_pos = np.empty((flush_size, 3), dtype=pos_np.dtype)
                    buf_spc = np.empty((flush_size,), dtype=np.asarray(spc_np).dtype)

                start = 0
                while start < n:
                    space = flush_size - fill
                    take = (n - start) if (n - start) < space else space

                    buf_pos[fill:fill + take] = pos_np[start:start + take]
                    buf_spc[fill:fill + take] = spc_np[start:start + take]

                    fill += take
                    start += take

                    if fill == flush_size:
                        file_chunk_index += 1
                        pending_writes.append(
                            writer_pool.submit(_write_chunk, file_chunk_index, buf_pos.copy(), buf_spc.copy())
                        )
                        fill = 0  # reset

        def _flush_tail():
            nonlocal fill, file_chunk_index, pending_writes
            with lock:
                if fill > 0:
                    file_chunk_index += 1
                    pending_writes.append(
                        writer_pool.submit(_write_chunk, file_chunk_index, buf_pos[:fill].copy(), buf_spc[:fill].copy())
                    )
                    fill = 0

        # 4) Decide if GPU is available, and how many
        gpu_ok = False
        available_gpus = 0
        if use_gpu and (cp is not None):
            try:
                available_gpus = int(cp.cuda.runtime.getDeviceCount())
                gpu_ok = (available_gpus > 0)
            except Exception:
                gpu_ok = False

        # Determine number of GPUs to use
        if n_gpus is None:
            use_n_gpus = max(1, available_gpus)
        else:
            use_n_gpus = min(int(n_gpus), available_gpus) if available_gpus > 0 else 0

        # Single-pass generation: generate atoms once per chunk, compute Voronoi once,
        # Get geometric chunks using the base (unrotated) material
        chunk_positions, chunk_dims = self.get_chunk_positions(material)
        num_geom_chunks = int(chunk_positions.shape[0])

        # Multi-GPU path: distribute chunks across multiple GPUs
        if gpu_ok and use_n_gpus > 1:
            try:
                # Round-robin distribute chunks to GPUs
                shards = [[] for _ in range(use_n_gpus)]
                for i in range(num_geom_chunks):
                    shards[i % use_n_gpus].append(i)

                # Shared state for multi-GPU accumulation
                gpu_errors = [None] * use_n_gpus

                def gpu_worker_poly(dev_id, my_chunks):
                    """Process assigned chunks on a specific GPU."""
                    try:
                        cp.cuda.Device(dev_id).use()

                        # Pre-allocate invariants on this GPU
                        seeds_cp = cp.asarray(seeds_np, dtype=cp.float32)
                        R_cp = cp.asarray(R, dtype=cp.float32)
                        lattice_cp = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
                        offset_cp = cp.asarray(self.offset, dtype=cp.float32)
                        dim_half_cp = cp.asarray(self.dimensions * 0.5, dtype=cp.float32)

                        for chunk_idx in my_chunks:
                            pos_cp, mask_cp, site_count = self.get_atomic_data(
                                material,
                                chunk_positions[chunk_idx, :],
                                chunk_dims,
                                use_gpu=True,
                                return_on_gpu=True,
                                lattice_atom_cartesian_cp=lattice_cp,
                                offset_gpu=offset_cp,
                                dim_half_gpu=dim_half_cp
                            )

                            if pos_cp.size == 0:
                                continue

                            # Voronoi assignment (streaming for memory safety)
                            grain_labels = self._voronoi_assign_gpu_streaming(pos_cp, seeds_cp)

                            # Species array on CPU
                            mask_np = mask_cp.get()
                            spc_sample = self._build_species(material.species, site_count, mask_np)

                            # Partition by grain and apply rotations
                            for g in range(G):
                                mask_g = (grain_labels == g)
                                if not bool(cp.any(mask_g)):
                                    continue

                                pos_g_unrotated = pos_cp[mask_g, :]
                                pos_g_rotated = pos_g_unrotated @ R_cp[g].T
                                pos_np = pos_g_rotated.get().astype(np.float32)

                                mask_g_np = mask_g.get()
                                spc_g = spc_sample[mask_g_np]

                                # Thread-safe accumulation (lock is inside _accumulate_to_buffers)
                                _accumulate_to_buffers(pos_np, spc_g)

                            # Cleanup after each chunk
                            del pos_cp, mask_cp, grain_labels
                            try:
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                pass

                        # Cleanup this GPU's invariants
                        del seeds_cp, R_cp, lattice_cp, offset_cp, dim_half_cp
                        try:
                            cp.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass

                    except Exception as e:
                        gpu_errors[dev_id] = e

                # Launch one thread per GPU
                threads = []
                for dev_id in range(use_n_gpus):
                    if shards[dev_id]:
                        t = threading.Thread(
                            target=gpu_worker_poly,
                            args=(dev_id, shards[dev_id]),
                            name=f"GPU-{dev_id}"
                        )
                        t.start()
                        threads.append(t)

                # Wait for all GPUs to complete
                for t in threads:
                    t.join()

                # Check for errors
                if any(e is not None for e in gpu_errors):
                    gpu_ok = False

            except Exception:
                try:
                    for dev_id in range(use_n_gpus):
                        cp.cuda.Device(dev_id).use()
                        cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                gpu_ok = False

        # Single-GPU path (original implementation)
        elif gpu_ok:
            # GPU path: single-pass with memory-bounded Voronoi
            try:
                # Pre-allocate invariants on GPU
                seeds_cp = cp.asarray(seeds_np, dtype=cp.float32)
                R_cp = cp.asarray(R, dtype=cp.float32)  # (G, 3, 3) rotation matrices
                lattice_atom_cartesian_cp = cp.asarray(material.lattice_atom_cartesian, dtype=cp.float32)
                offset_gpu = cp.asarray(self.offset, dtype=cp.float32)
                dim_half_gpu = cp.asarray(self.dimensions * 0.5, dtype=cp.float32)

                for chunk_idx in range(num_geom_chunks):
                    # Generate atoms ONCE per chunk using unrotated material
                    pos_cp, mask_cp, site_count = self.get_atomic_data(
                        material,
                        chunk_positions[chunk_idx, :],
                        chunk_dims,
                        use_gpu=True,
                        return_on_gpu=True,
                        lattice_atom_cartesian_cp=lattice_atom_cartesian_cp,
                        offset_gpu=offset_gpu,
                        dim_half_gpu=dim_half_gpu
                    )

                    if pos_cp.size == 0:
                        continue

                    # Compute Voronoi membership ONCE for all atoms in this chunk
                    # This is O(N_chunk * G) instead of O(G * N_chunk * G)
                    # Uses streaming to guarantee memory stays bounded for any G or N
                    grain_labels = self._voronoi_assign_gpu_streaming(pos_cp, seeds_cp)

                    # Species array for this chunk (on CPU for memory efficiency)
                    mask_np = mask_cp.get()
                    spc_sample = self._build_species(material.species, site_count, mask_np)

                    # Partition atoms by grain and apply rotations
                    for g in range(G):
                        mask_g = (grain_labels == g)
                        if not bool(cp.any(mask_g)):
                            continue

                        # Extract grain atoms and apply rotation ON GPU
                        pos_g_unrotated = pos_cp[mask_g, :]  # (N_g, 3)
                        pos_g_rotated = pos_g_unrotated @ R_cp[g].T  # Apply grain rotation
                        pos_np = pos_g_rotated.get().astype(np.float32)

                        # Extract species for this grain
                        mask_g_np = mask_g.get()
                        spc_g = spc_sample[mask_g_np]

                        _accumulate_to_buffers(pos_np, spc_g)

                    # Memory cleanup after each chunk
                    del pos_cp, mask_cp, grain_labels
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        pass

                # Final cleanup of GPU invariants
                del seeds_cp, R_cp, lattice_atom_cartesian_cp, offset_gpu, dim_half_gpu
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass

            except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError, RuntimeError, ValueError):
                # GPU error -> fallback to CPU path
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
                gpu_ok = False  # Fall through to CPU path

        if not gpu_ok:
            # CPU path: single-pass with chunk-level parallelism
            def _process_chunk_cpu(chunk_idx):
                # Generate atoms ONCE per chunk using unrotated material
                pos_np, spc_np = self.get_atomic_data(
                    material,
                    chunk_positions[chunk_idx, :],
                    chunk_dims,
                    use_gpu=False
                )

                if pos_np.shape[0] == 0:
                    return

                # Compute Voronoi membership ONCE for all atoms in this chunk
                grain_labels = self._voronoi_min_index_cpu(pos_np, seeds_np)

                # Partition atoms by grain and apply rotations
                for g in range(G):
                    mask_g = (grain_labels == g)
                    if not np.any(mask_g):
                        continue

                    # Extract grain atoms and apply rotation
                    pos_g_unrotated = pos_np[mask_g, :]
                    pos_g_rotated = (pos_g_unrotated @ R[g].T).astype(np.float32)
                    spc_g = spc_np[mask_g]

                    _accumulate_to_buffers(pos_g_rotated, spc_g)

            # Process chunks (can be parallelized with threads if needed)
            if grain_workers is None or int(grain_workers) <= 0:
                try:
                    grain_workers = min(num_geom_chunks, os.cpu_count() or 1)
                except Exception:
                    grain_workers = min(num_geom_chunks, 4)

            if num_geom_chunks > 1 and grain_workers > 1:
                with ThreadPoolExecutor(max_workers=int(grain_workers), thread_name_prefix="chunk") as pool:
                    futs = [pool.submit(_process_chunk_cpu, i) for i in range(num_geom_chunks)]
                    wait(futs, return_when=ALL_COMPLETED)
            else:
                for i in range(num_geom_chunks):
                    _process_chunk_cpu(i)

        # Flush and finalize
        _flush_tail()
        wait(pending_writes, return_when=ALL_COMPLETED)
        writer_pool.shutdown(wait=True)

        # update metadata
        self._chunk_total = int(file_chunk_index)
        self._teardown_alloy()
        return
    # -------------------------------------

    # -------------------------------------
    # KNN search
    def build_cell_list_gpu(self, positions, r_cut):
        """
        Build a GPU cell list for neighbor searches with a cubic cutoff.

        Two-pass algorithm on GPU:
          1) Count pass assigns each point to a cell and atomically increments
             per-cell counts.
          2) Fill pass writes positions and original indices into cell-compacted
             arrays using per-cell write pointers.

        Args:
            positions (cp.ndarray): Array of shape (N, 3) with float32 positions
                on the device.
            r_cut (float): Desired real-space cutoff; used as the cubic cell size.

        Returns:
            tuple:
                - sorted_positions (cp.ndarray, shape (N, 3), float32)
                - sorted_indices (cp.ndarray, shape (N,), int32)
                - cell_start (cp.ndarray, shape (num_cells,), int32) exclusive starts
                - cell_end (cp.ndarray, shape (num_cells,), int32) exclusive ends
                - bounding_box_min (cp.ndarray, shape (3,), float32)
                - cell_size (float)
                - nx, ny, nz (int): Number of cells per axis.

        Notes:
            Requires CuPy, a CUDA device, and kernels built by
            :meth:`build_cell_list_count_kernel` and :meth:`build_cell_list_fill_kernel`.
        """
        N = positions.shape[0]
        if N == 0:
            # Return trivial arrays for an empty set
            return (cp.zeros((0,3), dtype=cp.float32),
                    cp.zeros((0,), dtype=cp.int32),
                    cp.zeros((0,), dtype=cp.int32),
                    cp.zeros((0,), dtype=cp.int32),
                    cp.array([0,0,0], dtype=cp.float32),
                    1.0, 0, 0, 0)

        # 1) Compute bounding box on GPU
        min_corner = cp.min(positions, axis=0)
        max_corner = cp.max(positions, axis=0)
        box_size = max_corner - min_corner

        # 2) Decide cell size = r_cut (cube cells)
        cell_size = max(r_cut, 1.0)

        # 3) Number of cells in each dimension
        nx = int(cp.ceil(box_size[0] / cell_size)) if box_size[0] > 0 else 1
        ny = int(cp.ceil(box_size[1] / cell_size)) if box_size[1] > 0 else 1
        nz = int(cp.ceil(box_size[2] / cell_size)) if box_size[2] > 0 else 1
        num_cells = nx*ny*nz

        # 4) Prepare arrays
        cell_indices = cp.zeros((N,), dtype=cp.int32)
        cell_counts  = cp.zeros((num_cells,), dtype=cp.int32)

        # We'll need a 3-element array for inv_cell_size
        inv_cell_size = cp.array([1.0/cell_size, 1.0/cell_size, 1.0/cell_size], dtype=cp.float32)

        threads_per_block = 256
        blocks = (N + threads_per_block - 1) // threads_per_block

        # 5) First pass: count how many points go into each cell
        cell_list_count_kernel = self.build_cell_list_count_kernel()
        cell_list_count_kernel(
            (blocks,), (threads_per_block,),
            (
                positions,
                min_corner,
                inv_cell_size,   # Must be a pointer of length 3
                nx, ny, nz,
                cell_indices,
                cell_counts,
                N
            )
        )

        # 6) Compute prefix sum (exclusive scan) of cell_counts
        cell_offsets = cp.cumsum(cell_counts, dtype=cp.int32)
        # shift right by 1 so cell_start[i] is the start of cell i
        cell_start = cp.zeros_like(cell_offsets)
        cell_start[1:] = cell_offsets[:-1]
        cell_start[0]  = 0
        cell_end = cell_offsets  # end = exclusive offset

        # 7) Create arrays for the sorted output
        sorted_positions = cp.zeros_like(positions)
        sorted_indices   = cp.zeros((N,), dtype=cp.int32)

        # 8) Second pass: fill compacted arrays using an atomic write pointer
        cell_offsets_copy = cp.array(cell_start, copy=True)

        cell_list_fill_kernel = self.build_cell_list_fill_kernel()
        cell_list_fill_kernel(
            (blocks,), (threads_per_block,),
            (
                positions,
                cell_indices,
                cell_offsets_copy,
                sorted_positions,
                sorted_indices,
                N
            )
        )

        return (sorted_positions,
                sorted_indices,
                cell_start,
                cell_end,
                min_corner,
                cell_size,
                nx, ny, nz)
    # -------------------------------------
        
    # -------------------------------------
    # Plotting
    def plot_sample(self, elev=0, azim=0):
        """
        Plot all chunks of the sample as a 3D scatter.

        Loads each chunk from disk on CPU, then plots the points using Matplotlib.

        Args:
            elev (float, optional): Elevation angle in degrees for the 3D view.
                Defaults to 0.
            azim (float, optional): Azimuth angle in degrees for the 3D view.
                Defaults to 0.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot]:
                The created figure and 3D axes.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_proj_type('ortho')
        ax1.axis('equal')
        plt.tight_layout()

        for i in range(self.chunk_total):
            positions_chunk_np = self.load_chunk_positions(i + 1, use_gpu=False)
            ax1.scatter(
                positions_chunk_np[:, 0],
                positions_chunk_np[:, 1],
                positions_chunk_np[:, 2],
                c='b', marker='.'
            )
        return fig, ax1
    
    def plot_sample_exterior(
        self,
        voxels=100,
        voxel_size=None,
        engine="auto",
        decimate=1,
        max_quads=500000,
        face_alpha=0.15,
        show_edges=True,
        elev=20,
        azim=-60,
        figsize=(8, 8)
    ):
        """
        Plot exterior voxels as complete cubes (6 faces per voxel) instead of only
        plotting boundary faces.

        Steps:
        1) Build an occupancy grid by streaming atom chunks.
        2) Identify exterior voxels (occupied cells touching an empty neighbor or
            lying on the domain boundary).
        3) Assemble 6 quads per selected exterior voxel, honoring decimation and
            max_quads budget.
        4) Render as a Poly3DCollection.

        Notes:
            - decimate applies at the voxel level (keep every nth exterior voxel).
            - max_quads caps total faces; since each voxel contributes 6 faces,
            the effective voxel cap is floor(max_quads/6).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection

        # validation
        if self._chunk_total is None or int(self._chunk_total) <= 0:
            raise ValueError("No on-disk chunks found. Generate or import sample first.")
        if self._dimensions is None or self._offset is None:
            raise ValueError("Sample dimensions/offset are not initialized.")

        # choose backend
        use_gpu = False
        if engine not in ("auto", "gpu", "cpu"):
            engine = "auto"
        if engine in ("auto", "gpu") and (cp is not None):
            try:
                use_gpu = (int(cp.cuda.runtime.getDeviceCount()) > 0)
            except Exception:
                use_gpu = False

        # grid setup
        dims = np.asarray(self.dimensions, dtype=np.float32)
        if np.any(dims <= 0):
            raise ValueError("Invalid sample dimensions.")
        box_min = np.asarray(self.offset - 0.5 * dims, dtype=np.float32)
        box_max = box_min + dims

        if voxel_size is not None and float(voxel_size) > 0.0:
            nx, ny, nz = np.ceil(dims / float(voxel_size)).astype(np.int64)
        else:
            scale = float(voxels) / float(dims.max())
            nx, ny, nz = np.ceil(dims * scale).astype(np.int64)

        nx = int(max(nx, 1))
        ny = int(max(ny, 1))
        nz = int(max(nz, 1))
        hx, hy, hz = dims[0] / nx, dims[1] / ny, dims[2] / nz

        if use_gpu:
            occ = cp.zeros((nx, ny, nz), dtype=cp.bool_)
            box_min_xp = cp.asarray(box_min, dtype=cp.float32)
        else:
            occ = np.zeros((nx, ny, nz), dtype=np.bool_)
            box_min_xp = box_min  # numpy

        # stream chunks -> fill voxel occupancy
        def _accumulate_voxels_gpu(pos_cp):
            if pos_cp.size == 0:
                return
            idxf = (pos_cp - box_min_xp) / cp.asarray([hx, hy, hz], dtype=cp.float32)
            idx = cp.floor(idxf).astype(cp.int64)
            idx[:, 0] = cp.clip(idx[:, 0], 0, nx - 1)
            idx[:, 1] = cp.clip(idx[:, 1], 0, ny - 1)
            idx[:, 2] = cp.clip(idx[:, 2], 0, nz - 1)
            lin = (idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2]).astype(cp.int64)
            lin = cp.unique(lin)
            occ.ravel()[lin] = True

        def _accumulate_voxels_cpu(pos_np):
            if pos_np.size == 0:
                return
            idxf = (pos_np - box_min) / np.array([hx, hy, hz], dtype=np.float32)
            idx = np.floor(idxf).astype(np.int64)
            idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
            idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
            idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
            lin = (idx[:, 0] * (ny * nz) + idx[:, 1] * nz + idx[:, 2]).astype(np.int64)
            lin = np.unique(lin)
            occ.ravel()[lin] = True

        for i in range(int(self.chunk_total)):
            try:
                pos = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            except Exception:
                if use_gpu:
                    # fallback to CPU for this chunk
                    pos = self.load_chunk_positions(i + 1, use_gpu=False)
                    _accumulate_voxels_cpu(pos)
                    continue
                else:
                    raise
            if use_gpu and isinstance(pos, cp.ndarray):
                _accumulate_voxels_gpu(pos)
                try:
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    pass
            else:
                _accumulate_voxels_cpu(pos)

        # identify exterior voxels (occupied and touching empty or domain)
        if use_gpu:
            lib = cp
        else:
            lib = np

        o = occ
        ext_mask = lib.zeros_like(o, dtype=bool)

        # neighbors along x
        ext_mask[:-1, :, :] |= o[:-1, :, :] & ~o[1:, :, :]
        ext_mask[1:,  :, :] |= o[1:,  :, :] & ~o[:-1, :, :]
        # neighbors along y
        ext_mask[:, :-1, :] |= o[:, :-1, :] & ~o[:, 1:, :]
        ext_mask[:, 1:,  :] |= o[:, 1:,  :] & ~o[:, :-1, :]
        # neighbors along z
        ext_mask[:, :, :-1] |= o[:, :, :-1] & ~o[:, :, 1:]
        ext_mask[:, :, 1:]  |= o[:, :, 1:]  & ~o[:, :, :-1]

        # domain boundaries (occupied at boundary is exterior)
        ext_mask[0,   :, :] |= o[0,   :, :]
        ext_mask[-1,  :, :] |= o[-1,  :, :]
        ext_mask[:, 0,  :] |= o[:, 0,  :]
        ext_mask[:, -1, :] |= o[:, -1, :]
        ext_mask[:, :, 0] |= o[:, :, 0]
        ext_mask[:, :, -1] |= o[:, :, -1]

        # gather exterior voxel indices to host
        ext_idx = lib.argwhere(ext_mask)
        if use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            ext_idx = ext_idx.get()
        ext_idx = np.asarray(ext_idx, dtype=np.int64)

        # decimate at voxel level
        if int(decimate) > 1 and ext_idx.shape[0] > 0:
            ext_idx = ext_idx[::int(decimate), :]

        # enforce face budget via voxel cap (6 faces per voxel)
        total_voxels_before = int(ext_idx.shape[0])
        total_quads_before = 6 * total_voxels_before
        if max_quads is not None and int(max_quads) >= 0 and total_voxels_before > 0:
            voxel_cap = int(max_quads) // 6
            if voxel_cap > 0 and total_voxels_before > voxel_cap:
                sel = np.linspace(0, total_voxels_before - 1, voxel_cap, dtype=np.int64)
                ext_idx = ext_idx[sel, :]
            elif voxel_cap == 0:
                ext_idx = ext_idx[:0, :]

        M = int(ext_idx.shape[0])
        if M == 0:
            # Nothing exterior; draw empty box for reference
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.set_xlim([box_min[0], box_max[0]])
            ax.set_ylim([box_min[1], box_max[1]])
            ax.set_zlim([box_min[2], box_max[2]])
            ax.view_init(elev=elev, azim=azim)
            ax.set_proj_type("ortho")
            ax.set_title("No exterior voxels detected")
            return fig, ax, {
                "grid_shape": (nx, ny, nz),
                "voxel_size": (hx, hy, hz),
                "quads_kept": 0,
                "quads_total_before_decimate": int(total_quads_before),
            }

        # build 6 faces per selected voxel (vectorized on CPU)
        i = ext_idx[:, 0]
        j = ext_idx[:, 1]
        k = ext_idx[:, 2]

        x0 = box_min[0] + i.astype(np.float64) * hx
        x1 = x0 + hx
        y0 = box_min[1] + j.astype(np.float64) * hy
        y1 = y0 + hy
        z0 = box_min[2] + k.astype(np.float64) * hz
        z1 = z0 + hz

        # helper to stack corners for a face
        def _face(a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3):
            v0 = np.stack([a0, b0, c0], axis=1)
            v1 = np.stack([a1, b1, c1], axis=1)
            v2 = np.stack([a2, b2, c2], axis=1)
            v3 = np.stack([a3, b3, c3], axis=1)
            return np.stack([v0, v1, v2, v3], axis=1)

        # six faces per voxel
        f_x0 = _face(x0, y0, z0,  x0, y1, z0,  x0, y1, z1,  x0, y0, z1)
        f_x1 = _face(x1, y0, z0,  x1, y0, z1,  x1, y1, z1,  x1, y1, z0)
        f_y0 = _face(x0, y0, z0,  x1, y0, z0,  x1, y0, z1,  x0, y0, z1)
        f_y1 = _face(x0, y1, z0,  x0, y1, z1,  x1, y1, z1,  x1, y1, z0)
        f_z0 = _face(x0, y0, z0,  x0, y1, z0,  x1, y1, z0,  x1, y0, z0)
        f_z1 = _face(x0, y0, z1,  x1, y0, z1,  x1, y1, z1,  x0, y1, z1)

        quads = np.concatenate([f_x0, f_x1, f_y0, f_y1, f_z0, f_z1], axis=0).astype(np.float32)

        # safety trim in case of rounding up
        if quads.shape[0] > int(max_quads):
            step = int(np.ceil(quads.shape[0] / max(1, int(max_quads))))
            quads = quads[::step, :]

        # plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")

        quads_plot = quads.get() if hasattr(quads, "get") else quads
        poly = _Poly3DCollection(quads_plot, linewidths=(0.2 if show_edges else 0.0))
        poly.set_edgecolor("k" if show_edges else (0, 0, 0, 0))
        poly.set_facecolor((0.7, 0.8, 1.0, float(face_alpha)) if face_alpha > 0 else (0, 0, 0, 0))
        ax.add_collection3d(poly)

        ax.set_xlim([box_min[0], box_max[0]])
        ax.set_ylim([box_min[1], box_max[1]])
        ax.set_zlim([box_min[2], box_max[2]])
        try:
            ax.set_box_aspect(dims)
        except Exception:
            pass
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        plt.tight_layout()

        info = {
            "grid_shape": (nx, ny, nz),
            "voxel_size": (hx, hy, hz),
            "quads_kept": int(quads.shape[0]),
            "quads_total_before_decimate": int(total_quads_before),
        }
        return fig, ax, info
    
    def plot_grains(
        self,
        voxels=120,
        voxel_size=None,
        engine="auto",
        decimate=1,
        max_quads=800000,
        face_alpha=0.35,
        show_edges=False,
        show_seeds=True,
        label_orientations=True,
        elev=20,
        azim=-60,
        figsize=(12, 8),
        cmap_name="tab20"
    ):
        """
        Plot Voronoi grains (cells) as colored surfaces inside the sample box.

        The volume is discretized to a regular grid; each voxel center is assigned
        to the nearest Voronoi seed; inter-voxel boundaries between different grain
        ids are surfaced and colored by grain id. If there are < 10 grains and
        label_orientations=True, a label is placed near each seed showing the
        grain's number. Each orientation is then displayed in a legend as Euler ZYX 
        (yaw, pitch, roll) in degrees.

        Args:
            voxels (int): Target voxels across the longest box edge (ignored if
                voxel_size provided). Higher gives finer cell surfaces.
            voxel_size (float | None): If given, edge length of voxels in position units.
            engine (str): "auto" (default), "gpu", or "cpu" for the seed assignment step.
            decimate (int): Keep every nth face during plotting. 1 = keep all.
            max_quads (int): Global cap on number of quads to draw.
            face_alpha (float): Face opacity in [0,1].
            show_edges (bool): Draw polygon edges.
            show_seeds (bool): Plot seed markers.
            label_orientations (bool): If True and grain_count < 10, annotate Euler angles.
            elev, azim (float): Matplotlib 3D view angles in degrees.
            figsize (tuple): Figure size (width, height).
            cmap_name (str): Name of matplotlib colormap for grain colors.

        Returns:
            (fig, ax, info) where:
                - fig: matplotlib Figure
                - ax:  3D Axes
                - info: dict with details (grid_shape, voxel_size, grains, faces_kept, engine_used)

        Raises:
            ValueError: If seeds are missing or sample geometry is not initialized.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection as _Poly3DCollection
        import matplotlib.cm as cm
        import matplotlib.patches as mpatches

        # validations
        if not hasattr(self, "_dimensions") or self._dimensions is None or not hasattr(self, "_offset") or self._offset is None:
            raise ValueError("Sample dimensions/offset not initialized. Create or load metadata first.")
        if not hasattr(self, "_grain_seeds") or self._grain_seeds is None:
            raise ValueError("No Voronoi seeds present. Call generate_voronoi_seeds(...) or input_voronoi_seed(...).")

        seeds_np = np.asarray(self._grain_seeds, dtype=np.float32).reshape(-1, 3)
        G = int(seeds_np.shape[0])

        use_gpu = False
        if engine not in ("auto", "gpu", "cpu"):
            engine = "auto"
        if engine in ("auto", "gpu") and (cp is not None):
            try:
                use_gpu = (int(cp.cuda.runtime.getDeviceCount()) > 0)
            except Exception:
                use_gpu = False

        dims = np.asarray(self.dimensions, dtype=np.float32)
        box_min = (self.offset - 0.5 * dims).astype(np.float32)
        box_max = box_min + dims

        # grid setup (nx, ny, nz)
        if voxel_size is not None and float(voxel_size) > 0.0:
            nx, ny, nz = np.ceil(dims / float(voxel_size)).astype(np.int64)
        else:
            scale = float(voxels) / float(dims.max())
            nx, ny, nz = np.ceil(dims * scale).astype(np.int64)

        nx = int(max(nx, 1))
        ny = int(max(ny, 1))
        nz = int(max(nz, 1))
        hx, hy, hz = dims[0] / nx, dims[1] / ny, dims[2] / nz

        # nearest seed argmin (streaming over seeds)
        def _argmin_seeds_cpu(positions_np, seeds_np_local):
            N = int(positions_np.shape[0])
            min_d2 = np.full((N,), np.inf, dtype=np.float64)
            min_idx = np.full((N,), -1, dtype=np.int32)
            for g in range(seeds_np_local.shape[0]):
                d = positions_np - seeds_np_local[g][None, :]
                d2 = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1] + d[:, 2] * d[:, 2]
                mask = d2 < min_d2
                min_idx[mask] = g
                min_d2[mask] = d2[mask]
            return min_idx

        def _argmin_seeds_gpu(positions_cp, seeds_np_local):
            seeds_cp = cp.asarray(seeds_np_local, dtype=cp.float32)
            N = int(positions_cp.shape[0])
            min_d2 = cp.full((N,), cp.inf, dtype=cp.float32)
            min_idx = cp.full((N,), -1, dtype=cp.int32)
            for g in range(int(seeds_cp.shape[0])):
                d = positions_cp - seeds_cp[g, :][None, :]
                d2 = cp.sum(d * d, axis=1, dtype=cp.float32)
                mask = d2 < min_d2
                min_idx = cp.where(mask, g, min_idx)
                min_d2 = cp.where(mask, d2, min_d2)
            return min_idx

        # voxel centers to Voronoi id per cell
        if use_gpu:
            centers_ijk = self.get_flat_grid((nx, ny, nz), use_gpu=True)
            centers = centers_ijk * cp.asarray([hx, hy, hz], dtype=cp.float32) + \
                    (cp.asarray(box_min, dtype=cp.float32) + cp.asarray([0.5*hx, 0.5*hy, 0.5*hz], dtype=cp.float32))
            try:
                gids_flat = _argmin_seeds_gpu(centers, seeds_np)
            except Exception:
                centers_np = centers.get()
                gids_flat = _argmin_seeds_cpu(centers_np, seeds_np)
                use_gpu = False
            else:
                gids_flat = gids_flat.get()
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        else:
            centers_ijk = self.get_flat_grid((nx, ny, nz), use_gpu=False)
            centers = centers_ijk * np.array([hx, hy, hz], dtype=np.float32) + \
                    (box_min + np.array([0.5*hx, 0.5*hy, 0.5*hz], dtype=np.float32))
            gids_flat = _argmin_seeds_cpu(centers, seeds_np)

        labels = gids_flat.reshape(nx, ny, nz)

        # face assembly utilities
        def _group_quads_by_grain(gids, quads_arr):
            out = {}
            if gids.size == 0:
                return out
            u = np.unique(gids)
            for gr in u:
                sel = (gids == gr)
                out[int(gr)] = quads_arr[sel, :, :]
            return out

        def _merge_group(dst, src):
            for k, v in src.items():
                if k in dst:
                    dst[k] = np.concatenate([dst[k], v], axis=0)
                else:
                    dst[k] = v

        def _faces_x_different(labels3d):
            m = labels3d[:-1, :, :] != labels3d[1:, :, :]
            i0, j0, k0 = np.where(m)
            if i0.size == 0:
                return {}
            g = labels3d[i0, j0, k0]
            xp = box_min[0] + (i0.astype(np.float64) + 1.0) * hx
            y0 = box_min[1] + j0.astype(np.float64) * hy
            y1 = y0 + hy
            z0 = box_min[2] + k0.astype(np.float64) * hz
            z1 = z0 + hz
            v0 = np.stack([xp, y0, z0], axis=1)
            v1 = np.stack([xp, y1, z0], axis=1)
            v2 = np.stack([xp, y1, z1], axis=1)
            v3 = np.stack([xp, y0, z1], axis=1)
            quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
            return _group_quads_by_grain(g, quads)

        def _faces_y_different(labels3d):
            m = labels3d[:, :-1, :] != labels3d[:, 1:, :]
            i0, j0, k0 = np.where(m)
            if i0.size == 0:
                return {}
            g = labels3d[i0, j0, k0]
            yp = box_min[1] + (j0.astype(np.float64) + 1.0) * hy
            x0 = box_min[0] + i0.astype(np.float64) * hx
            x1 = x0 + hx
            z0 = box_min[2] + k0.astype(np.float64) * hz
            z1 = z0 + hz
            v0 = np.stack([x0, yp, z0], axis=1)
            v1 = np.stack([x1, yp, z0], axis=1)
            v2 = np.stack([x1, yp, z1], axis=1)
            v3 = np.stack([x0, yp, z1], axis=1)
            quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
            return _group_quads_by_grain(g, quads)

        def _faces_z_different(labels3d):
            m = labels3d[:, :, :-1] != labels3d[:, :, 1:]
            i0, j0, k0 = np.where(m)
            if i0.size == 0:
                return {}
            g = labels3d[i0, j0, k0]
            zp = box_min[2] + (k0.astype(np.float64) + 1.0) * hz
            x0 = box_min[0] + i0.astype(np.float64) * hx
            x1 = x0 + hx
            y0 = box_min[1] + j0.astype(np.float64) * hy
            y1 = y0 + hy
            v0 = np.stack([x0, y0, zp], axis=1)
            v1 = np.stack([x1, y0, zp], axis=1)
            v2 = np.stack([x1, y1, zp], axis=1)
            v3 = np.stack([x0, y1, zp], axis=1)
            quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
            return _group_quads_by_grain(g, quads)

        def _faces_domain_boundaries(labels3d):
            out = {}
            # x min
            if nx > 0:
                i = 0
                j_idx, k_idx = np.indices((ny, nz))
                g = labels3d[i, j_idx.ravel(), k_idx.ravel()]
                x = np.full(g.size, box_min[0], dtype=np.float64)
                y0 = box_min[1] + j_idx.ravel().astype(np.float64) * hy
                y1 = y0 + hy
                z0 = box_min[2] + k_idx.ravel().astype(np.float64) * hz
                z1 = z0 + hz
                v0 = np.stack([x, y0, z0], axis=1)
                v1 = np.stack([x, y1, z0], axis=1)
                v2 = np.stack([x, y1, z1], axis=1)
                v3 = np.stack([x, y0, z1], axis=1)
                quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
                _merge_group(out, _group_quads_by_grain(g, quads))
            # x max
            if nx > 0:
                i = nx - 1
                j_idx, k_idx = np.indices((ny, nz))
                g = labels3d[i, j_idx.ravel(), k_idx.ravel()]
                x = np.full(g.size, box_min[0] + nx * hx, dtype=np.float64)
                y0 = box_min[1] + j_idx.ravel().astype(np.float64) * hy
                y1 = y0 + hy
                z0 = box_min[2] + k_idx.ravel().astype(np.float64) * hz
                z1 = z0 + hz
                v0 = np.stack([x, y0, z0], axis=1)
                v1 = np.stack([x, y0, z1], axis=1)
                v2 = np.stack([x, y1, z1], axis=1)
                v3 = np.stack([x, y1, z0], axis=1)
                quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
                _merge_group(out, _group_quads_by_grain(g, quads))
            # y min
            if ny > 0:
                j = 0
                i_idx, k_idx = np.indices((nx, nz))
                g = labels3d[i_idx.ravel(), j, k_idx.ravel()]
                y = np.full(g.size, box_min[1], dtype=np.float64)
                x0 = box_min[0] + i_idx.ravel().astype(np.float64) * hx
                x1 = x0 + hx
                z0 = box_min[2] + k_idx.ravel().astype(np.float64) * hz
                z1 = z0 + hz
                v0 = np.stack([x0, y, z0], axis=1)
                v1 = np.stack([x1, y, z0], axis=1)
                v2 = np.stack([x1, y, z1], axis=1)
                v3 = np.stack([x0, y, z1], axis=1)
                quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
                _merge_group(out, _group_quads_by_grain(g, quads))
            # y max
            if ny > 0:
                j = ny - 1
                i_idx, k_idx = np.indices((nx, nz))
                g = labels3d[i_idx.ravel(), j, k_idx.ravel()]
                y = np.full(g.size, box_min[1] + ny * hy, dtype=np.float64)
                x0 = box_min[0] + i_idx.ravel().astype(np.float64) * hx
                x1 = x0 + hx
                z0 = box_min[2] + k_idx.ravel().astype(np.float64) * hz
                z1 = z0 + hz
                v0 = np.stack([x0, y, z0], axis=1)
                v1 = np.stack([x0, y, z1], axis=1)
                v2 = np.stack([x1, y, z1], axis=1)
                v3 = np.stack([x1, y, z0], axis=1)
                quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
                _merge_group(out, _group_quads_by_grain(g, quads))
            # z min
            if nz > 0:
                k = 0
                i_idx, j_idx = np.indices((nx, ny))
                g = labels3d[i_idx.ravel(), j_idx.ravel(), k]
                z = np.full(g.size, box_min[2], dtype=np.float64)
                x0 = box_min[0] + i_idx.ravel().astype(np.float64) * hx
                x1 = x0 + hx
                y0 = box_min[1] + j_idx.ravel().astype(np.float64) * hy
                y1 = y0 + hy
                v0 = np.stack([x0, y0, z], axis=1)
                v1 = np.stack([x1, y0, z], axis=1)
                v2 = np.stack([x1, y1, z], axis=1)
                v3 = np.stack([x0, y1, z], axis=1)
                quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
                _merge_group(out, _group_quads_by_grain(g, quads))
            # z max
            if nz > 0:
                k = nz - 1
                i_idx, j_idx = np.indices((nx, ny))
                g = labels3d[i_idx.ravel(), j_idx.ravel(), k]
                z = np.full(g.size, box_min[2] + nz * hz, dtype=np.float64)
                x0 = box_min[0] + i_idx.ravel().astype(np.float64) * hx
                x1 = x0 + hx
                y0 = box_min[1] + j_idx.ravel().astype(np.float64) * hy
                y1 = y0 + hy
                v0 = np.stack([x0, y0, z], axis=1)
                v1 = np.stack([x0, y1, z], axis=1)
                v2 = np.stack([x1, y1, z], axis=1)
                v3 = np.stack([x1, y0, z], axis=1)
                quads = np.stack([v0, v1, v2, v3], axis=1).astype(np.float32)
                _merge_group(out, _group_quads_by_grain(g, quads))
            return out

        faces_by_grain = {}
        _merge_group(faces_by_grain, _faces_x_different(labels))
        _merge_group(faces_by_grain, _faces_y_different(labels))
        _merge_group(faces_by_grain, _faces_z_different(labels))
        _merge_group(faces_by_grain, _faces_domain_boundaries(labels))

        # Decimate and cap faces
        def _count_faces(d):
            return int(sum(arr.shape[0] for arr in d.values()))
        total_faces = _count_faces(faces_by_grain)

        if int(decimate) > 1:
            step = int(decimate)
            for gk in list(faces_by_grain.keys()):
                faces_by_grain[gk] = faces_by_grain[gk][::step, :, :]

        total_faces = _count_faces(faces_by_grain)
        if max_quads is not None and int(max_quads) >= 0 and total_faces > int(max_quads):
            step = int(np.ceil(total_faces / float(int(max_quads))))
            for gk in list(faces_by_grain.keys()):
                faces_by_grain[gk] = faces_by_grain[gk][::step, :, :]
            total_faces = _count_faces(faces_by_grain)

        # ---- plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        try:
            ax.set_box_aspect(dims)
        except Exception:
            pass
        ax.set_xlim([box_min[0], box_max[0]])
        ax.set_ylim([box_min[1], box_max[1]])
        ax.set_zlim([box_min[2], box_max[2]])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")

        cmap = cm.get_cmap(cmap_name, max(G, 1))
        def _color_for_gid(gid):
            return cmap(int(gid) % max(G, 1))

        # draw faces per grain
        for gk, quads in faces_by_grain.items():
            if quads.size == 0:
                continue
            poly = _Poly3DCollection(quads, linewidths=(0.2 if show_edges else 0.0))
            poly.set_edgecolor("k" if show_edges else (0, 0, 0, 0))
            base = _color_for_gid(gk)
            rgba = (base[0], base[1], base[2], float(face_alpha))
            poly.set_facecolor(rgba)
            ax.add_collection3d(poly)

        # seed markers
        if show_seeds:
            ax.scatter(seeds_np[:, 0], seeds_np[:, 1], seeds_np[:, 2], c="k", s=12, marker="o", depthshade=False, label="seeds")

        # arrows for [100],[010],[001] from each seed using orientation matrices
        # arrow length scaled to box size
        arrow_len = 0.08 * float(np.max(dims))
        have_R = hasattr(self, "_grain_orientations") and (self._grain_orientations is not None) \
                and (int(np.shape(self._grain_orientations)[0]) == G)
        I3 = np.eye(3, dtype=np.float32)

        for gid in range(G):
            p = seeds_np[gid]
            Rg = np.asarray(self._grain_orientations[gid], dtype=np.float32).reshape(3, 3) if have_R else I3

            # Sample-frame directions for crystal axes
            d100 = (Rg @ np.array([1.0, 0.0, 0.0], dtype=np.float32))
            d010 = (Rg @ np.array([0.0, 1.0, 0.0], dtype=np.float32))
            d001 = (Rg @ np.array([0.0, 0.0, 1.0], dtype=np.float32))

            ax.quiver(float(p[0]), float(p[1]), float(p[2]),
                    float(d100[0]), float(d100[1]), float(d100[2]),
                    length=arrow_len, normalize=True, color="r", linewidth=1.0)
            ax.quiver(float(p[0]), float(p[1]), float(p[2]),
                    float(d010[0]), float(d010[1]), float(d010[2]),
                    length=arrow_len, normalize=True, color="g", linewidth=1.0)
            ax.quiver(float(p[0]), float(p[1]), float(p[2]),
                    float(d001[0]), float(d001[1]), float(d001[2]),
                    length=arrow_len, normalize=True, color="b", linewidth=1.0)

        def _euler_zyx_from_R(R):
            r00, r01, r02 = R[0, 0], R[0, 1], R[0, 2]
            r10, r11, r12 = R[1, 0], R[1, 1], R[1, 2]
            r20, r21, r22 = R[2, 0], R[2, 1], R[2, 2]
            pitch = np.arcsin(np.clip(-r20, -1.0, 1.0))
            cpitch = np.cos(pitch)
            if abs(cpitch) > 1e-7:
                roll = np.arctan2(r21, r22)
                yaw = np.arctan2(r10, r00)
            else:
                roll = 0.0
                yaw = np.arctan2(-r01, r11)
            return yaw, pitch, roll

        if label_orientations and (G < 10):
            handles = []
            labels_out = []

            for gid in range(G):
                p = seeds_np[gid]

                # in-plot numeric label (1-based)
                ax.text(float(p[0]), float(p[1]), float(p[2] + 0.02 * dims[2]),
                        f"{gid + 1}", fontsize=9, color="k", ha="center", va="bottom")

                # legend line with Euler angles
                Rg = np.asarray(self._grain_orientations[gid], dtype=np.float64).reshape(3, 3) if have_R else np.eye(3, dtype=np.float64)
                yaw, pitch, roll = _euler_zyx_from_R(Rg)
                yd = float(np.degrees(yaw))
                pd = float(np.degrees(pitch))
                rd = float(np.degrees(roll))
                grain_color = _color_for_gid(gid)
                handle = mpatches.Patch(facecolor=grain_color, edgecolor="k", linewidth=0.5)
                label_txt = f"{gid + 1}: yaw={yd:.1f}, pitch={pd:.1f}, roll={rd:.1f}"
                handles.append(handle)
                labels_out.append(label_txt)

            # park the legend outside on the right; reserve space
            plt.tight_layout(rect=[0.0, 0.0, 0.5, 1.0])
            ax.legend(handles=handles, labels=labels_out,
                    loc="center left", bbox_to_anchor=(1.15, 0.5),
                    frameon=True, title="Grain orientation (deg)", fontsize=8)
        else:
            plt.tight_layout()
        
        return fig, ax
    # -------------------------------------
    
    ## Properties
    @property
    def dimensions(self):
        """
        Return the dimensions array (length 3).
        """
        if self._dimensions is None:
            print("self._dimensions has not been initialized yet")
        return self._dimensions

    @property
    def offset(self):
        """
        Return the offset array (length 3).
        """
        if self._offset is None:
            print("self._offset has not been initialized yet")
        return self._offset
    
    @property
    def rotation(self):
        """
        Return the rotation matrix (3x3).
        """
        if self._rotation is None:
            print("self._rotation has not been initialized yet")
        return self._rotation

    @property
    def chunk_volume(self):
        """
        Return the chunk volume.
        """
        if self._chunk_volume is None:
            print("self._chunk_volume has not been initialized yet")
        return self._chunk_volume

    @property
    def matrix(self):
        """
        Return the sample matrix (3x3).
        """
        if self._matrix is None:
            self._matrix = np.diag(self.dimensions)
        return self._matrix

    @property
    def corners(self):
        """
        Return the corners of the sample parallelepiped (8x3).
        """
        if self._corners is None:
            self._corners = (self.get_unit_corners() @ self.matrix) - (self.dimensions * 0.5) + self.offset
        return self._corners
    
    @property
    def chunk_positions(self):
        """
        Return the array of chunk positions (Nx3).
        """
        if self._chunk_positions is None:
            print("self._chunk_positions has not been initialized yet")
        return self._chunk_positions
    
    @property
    def chunk_dimensions(self):
        """
        Return the chunk dimensions (in lattice units).
        """
        if self._chunk_dimensions is None:
            print("self._chunk_dimensions has not been initialized yet")
        return self._chunk_dimensions
    
    @property
    def chunk_total(self):
        """
        Return the total number of chunks in the sample.
        """
        if self._chunk_total is None:
            print("self._chunk_total has not been initialized yet")
        return self._chunk_total

    @property
    def sample_type(self):
        """
        Return the current sample type: 'single' or 'poly'.
        """
        if not hasattr(self, "_sample_type"):
            self._sample_type = "single"
        return self._sample_type

    @property
    def streaming_mode(self):
        """
        Return True if streaming mode is enabled.

        In streaming mode, chunks are generated on-demand during simulation
        rather than being loaded from disk files.
        """
        return getattr(self, "_streaming_mode", False)