# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import os
import gc
import json
try:
    import cupy as cp
except ImportError:
    cp = None
from cffi import FFI

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class sample:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=os.getcwd()):
        """Initialize core state and ensure the working directory exists.

        Compiles the CFFI intersection routine, sets default attributes for the
        sample, and guarantees that the target directory is present on disk.

        Args:
            directory (str, optional): Directory where chunk files and metadata
                will be read from and written to. Defaults to the current
                working directory.

        Notes:
            Geometry and data are not created here. Use `create_sample`,
            `import_atomic_data`, or `generate_sample` to populate files.
        """
        # Core directory and lazily initialized fields
        self.directory = directory
        self._dimensions = None
        self._offset = None
        self._rotation = None
        self._chunk_volume = None
        self._chunk_total = None 
        self._matrix = None
        self._corners = None

        # Temperature/displacement configuration (disabled by default)
        self.enable_temp = False
        self.temp_params = ['gaussian', 0.25, 1, 40]

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
            
    def create_sample(self, dimensions, offset=[0, 0, 0], chunk_volume=12500000):
        """Create an axis-aligned sample box and precompute helpers.

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
                chunking in `get_chunk_positions`/`generate_sample`. Defaults to
                12_500_000.

        Returns:
            None
        """
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
        
    def read_sample_metadata(self):
        """Load JSON metadata from disk and restore core state.

        Reads `sample_metadata.json` from `self.directory` (or from a provided
        override path in the writer) and restores `_dimensions`, `_offset`,
        `_rotation`, and `_chunk_total` if present.

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
        
    ## Data Handling Functions
    # -------------------------------------
    # Generate sample
    def write_chunk_positions(self, data, chunk_num, override_directory=None):
        """Write a positions array for a specific chunk to disk.

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
        """Write a species array for a specific chunk to disk.

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
        """Serialize critical fields to a JSON metadata file on disk.

        Writes `sample_metadata.json` containing `dimensions`, `offset`,
        `rotation`, and `chunk_total`. NumPy arrays are converted to lists so
        they are JSON serializable.

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
        """Load a chunk's positions from disk, optionally on GPU.

        If `use_gpu` is True and CuPy is available, returns a `cp.ndarray`.
        Otherwise, returns an `np.ndarray`. If `self.enable_temp` is True,
        temperature-based displacements are applied via `apply_temperature`.

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
        """
        # Compose the on-disk filename for this chunk
        base, ext = os.path.splitext(self._default_filenames[0])
        positions_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, positions_filename)

        # Load on GPU if requested and available; else load on CPU
        if use_gpu and (cp is not None):
            positions = cp.load(full_path)
        else:
            positions = np.load(full_path)

        # Optionally apply thermal displacements according to configured model
        if self.enable_temp is True:
            # Note: 'sigma' now means 'temperature_K' when distribution='einstein'
            positions = self.apply_temperature(
                positions,
                distribution=self.temp_params[0],
                sigma=self.temp_params[1],
                max_displacement=self.temp_params[2],
                seed=self.temp_params[3],
                chunk_number=chunk_number  # enables per-species masses if configured
            )
        return positions

    def load_chunk_species(self, chunk_number, use_gpu=True):
        """Load a chunk's species array from disk, optionally on GPU.

        If `use_gpu` is True and CuPy is available, returns a `cp.ndarray`.
        Otherwise, returns an `np.ndarray`.

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
        """
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
    
    # -------------------------------------
    # KNN search
    def write_chunk_nn_indices(self, index_list, chunk_num, override_directory=None):
        """Write neighbor index lists for a chunk to a compact NPZ.

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
        """Write neighbor phases for a chunk to a compact NPZ.

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
        """Write neighbor wavevectors for a chunk to a compact NPZ.

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
        """Write neighbor species for a chunk to a compact NPZ.

        Produces ``nearest_neighbors_species_<chunk_num>.npz`` with:
        - ``flat_species``: concatenated neighbor species values.
        - ``offsets``: start positions per atom (length n_atoms + 1).

        Notes:
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
        
    def load_chunk_nn_indices(self, chunk_num):
        """Load neighbor indices for a chunk.

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
        """Load neighbor phases for a chunk.

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
        """Load neighbor wavevectors for a chunk.

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
        """Load neighbor species for a chunk.

        Reads ``nearest_neighbors_species_<chunk_num>.npz`` and returns the
        flattened species array and offsets.

        Example:
            To reconstruct the ragged lists:

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
    # -------------------------------------

    # -------------------------------------
    # MD sample   
    def import_atomic_data(self, import_file, element_list, header_lines=9, ID_column=1, position_columns=[2,3,4], scale=1e-10, flush_size=100000000, override_directory=None):
        """Import a large text file of atoms and write chunked .npy outputs.

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
        chunk_num = 0  # running chunk counter

        # Track bounding box while streaming the file
        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')

        # Open and iterate in batches of up to flush_size lines
        with open(import_file, "r") as f:
            # Skip header lines at the top of the file
            for _ in range(header_lines):
                next(f)

            while True:
                lines = []
                # Collect up to flush_size lines for this batch
                for _ in range(flush_size):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)

                # End when we have no more lines
                if not lines:
                    break

                # Parse positions/species from the batch
                data_arr = np.zeros((len(lines), 3), dtype=np.float32)
                species_arr = []
                for i, line in enumerate(lines):
                    split_line = line.strip().split()

                    # Map 1-based species ID to label via provided element_list
                    species_arr.append(element_list[int(split_line[ID_column]) - 1])

                    # Convert coordinates; default keeps angstrom units unchanged
                    data_arr[i, 0] = float(split_line[position_columns[0]]) * float(scale / 1e-10)
                    data_arr[i, 1] = float(split_line[position_columns[1]]) * float(scale / 1e-10)
                    data_arr[i, 2] = float(split_line[position_columns[2]]) * float(scale / 1e-10)

                    # Update bounding box online to avoid a second pass
                    if data_arr[i, 0] < x_min: x_min = data_arr[i, 0]
                    if data_arr[i, 0] > x_max: x_max = data_arr[i, 0]
                    if data_arr[i, 1] < y_min: y_min = data_arr[i, 1]
                    if data_arr[i, 1] > y_max: y_max = data_arr[i, 1]
                    if data_arr[i, 2] < z_min: z_min = data_arr[i, 2]
                    if data_arr[i, 2] > z_max: z_max = data_arr[i, 2]

                # Bump the chunk index and persist this batch
                chunk_num += 1
                self.write_chunk_positions(data_arr, chunk_num, override_directory=override_directory)
                self.write_chunk_species(species_arr, chunk_num, override_directory=override_directory)

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
        """Return the 8 corners of the unit cube as an (8, 3) float32 array.

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
        """Compute a 3x3 rotation matrix for a rotation about an axis.

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
        """Create a flat grid of integer coordinates as an (N, 3) array.

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
        """Configure Einstein-model thermal displacements and enable temperature.

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

    def set_position_unit_in_m(self, unit_in_m):
        """Set the conversion factor from the position unit to meters.

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
        """Apply random displacements to positions using a chosen distribution.

        Two modes are supported:

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

        Args:
            positions (np.ndarray or cp.ndarray): Array of shape (N, 3).
            distribution (str, optional): 'gaussian' or 'einstein'. Defaults to 'gaussian'.
            sigma (float, optional): Stddev (gaussian) or temperature K (einstein).
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
            - If ``T_K <= 0`` in Einstein mode, only zero-point motion is applied.
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

        # Anything else is unsupported
        raise ValueError("Unknown distribution: {}".format(distribution))
    # -------------------------------------
        
    # -------------------------------------
    # Sample generation
    @staticmethod    
    def compile_parallelepipeds_intersect_batch_cffi():
        """Compile the CFFI SAT intersection batch function.

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
        """Build and return the CUDA kernel that counts items per cell.

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
        """Build and return the CUDA kernel that fills sorted cell lists.

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
        """Center all atom positions by subtracting the current offset.

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
        """Remove the current global rotation from all atom positions.

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
        """Center and de-rotate the sample in-place.

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
        """Apply an additional rotation to all atoms and update state.

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
        """Translate the sample by adding an offset to all atom positions.

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
        """Generate, filter, and offset a chunk on a given CUDA stream.

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
        """Compute candidate chunk origins and dimensions that intersect the sample.

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
        """Run a batched SAT intersection test via the verified CFFI module.

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
        """Compute lattice point positions in the sample frame for one chunk.

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
        """Build atom positions and species for a single geometric chunk.

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

                atomic_species = np.tile(material.species, int(lattice_positions_cp.shape[0]))
                atomic_species = atomic_species[mask_np]

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

        atomic_species = np.tile(material.species, lattice_positions_np.shape[0])

        # In-box mask on CPU
        mask = (
            (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= self.dimensions[0]) &
            (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= self.dimensions[1]) &
            (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= self.dimensions[2])
        )

        # Apply mask and center/offset shift
        atomic_positions_S = atomic_positions_S[mask, :].astype(np.float32)
        atomic_species = atomic_species[mask]

        offset_np = self.offset.astype(np.float32)
        dim_half_np = (self.dimensions * 0.5).astype(np.float32)
        atomic_positions_S += (offset_np - dim_half_np)
        gc.collect()
        return atomic_positions_S, atomic_species

    def generate_sample(
        self,
        material,
        flush_size=100000000,
        use_gpu=True,
        gpu_streams=4,
        writer_threads=3
    ):
        """Generate and persist the sample to disk in fixed-size chunks.

        The function:
          - Computes geometric chunks once using :meth:`get_chunk_positions`.
          - Streams generated atoms into CPU buffers of length ``flush_size``.
          - Writes chunked ``.npy`` files for positions and species via a
            thread pool to overlap I/O.
          - Runs a multi-stream GPU path if available; otherwise, or upon GPU
            failure, it falls back to a pure-CPU path without losing progress.

        Args:
            material: Object with lattice and unit-cell definitions used by
                :meth:`get_atomic_data`.
            flush_size (int, optional): Number of atoms per on-disk chunk.
                Defaults to 100_000_000.
            use_gpu (bool, optional): Enable the GPU generation path if CuPy and
                a CUDA device are available. Defaults to True.
            gpu_streams (int, optional): Number of concurrent CUDA streams to
                pipeline GPU work. Defaults to 4.
            writer_threads (int, optional): Number of I/O worker threads for
                writing chunks to disk. Defaults to 3.

        Returns:
            None

        Notes:
            On success, updates ``self._chunk_total`` with the number of files
            written. Progress already written is preserved if the GPU path
            encounters an error and falls back to CPU.
        """
        # 0) Build geometric chunks once
        self._chunk_positions, self._chunk_dimensions = self.get_chunk_positions(material)
        num_geom = int(self.chunk_positions.shape[0])

        # Early out if there is nothing to do
        if num_geom == 0:
            self._chunk_total = 0
            return

        flush_size = int(flush_size)

        from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED

        # 1) Thread pool for disk writes
        def _write_chunk(idx, pos_arr, spc_arr):
            self.write_chunk_positions(pos_arr, idx)
            self.write_chunk_species(spc_arr, idx)
            return idx

        writer_pool = ThreadPoolExecutor(
            max_workers=max(1, int(writer_threads)),
            thread_name_prefix="writer"
        )
        pending_writes = []

        # 2) CPU-side streaming buffers shared by both GPU and CPU paths
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

        # 3) Decide if we can and should use GPU
        gpu_ok = False
        if use_gpu and (cp is not None):
            try:
                devcount = int(cp.cuda.runtime.getDeviceCount())
                gpu_ok = (devcount > 0)
            except Exception:
                gpu_ok = False

        # 4) GPU path (with safe fallback)
        drained_count = 0  # number of geom-chunks fully drained into CPU buffers
        if gpu_ok:
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
                    spc_all = np.tile(material.species, task["site_count"])
                    spc_np = spc_all[mask_np]

                    _accumulate_to_buffers(pos_np, spc_np)
                    drained_count += 1

                    # Cleanup local references
                    del pos_np, mask_np, spc_all, spc_np, task

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

        # 5) CPU path (or GPU fallback remainder)
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

        # 6) Flush trailing partial buffer and finish
        _flush_tail()
        wait(pending_writes, return_when=ALL_COMPLETED)
        writer_pool.shutdown(wait=True)

        # 7) Update metadata
        self._chunk_total = int(file_chunk_index)
        return
    # -------------------------------------
    
    # -------------------------------------
    # KNN search
    def build_cell_list_gpu(self, positions, r_cut):
        """Build a GPU cell list for neighbor searches with a cubic cutoff.

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
        """Plot all chunks of the sample as a 3D scatter.

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
