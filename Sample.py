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
    def __init__(self,directory=os.getcwd()):
        self.directory = directory
        self._dimensions = None
        self._offset = None
        self._rotation = None
        self._chunk_volume = None
        self._chunk_total = None 
        self._matrix = None
        self._corners = None
        self.enable_temp = False
        self.temp_params = ['gaussian',0.25,1,40]
        self._default_filenames = np.array([
            "atomic_positions.npy",
            "atomic_species.npy",
            "sample_metadata.npy"
        ])  # sample_metadata will be a struct
        self._ffi_object, self._intersect_function = self.compile_parallelepipeds_intersect_batch_cffi()
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
    def create_sample(self, dimensions, offset=[0,0,0], chunk_volume=12500000):
        self._dimensions = np.array(dimensions, dtype=np.float32)
        self._offset = np.array(offset, dtype=np.float32)
        self._rotation = np.eye(3, dtype=np.float32)
        self._chunk_volume = np.array(chunk_volume, dtype=np.float32)
        self._matrix = np.diag(self.dimensions)
        # Slightly rewritten for small overhead reduction (no functional change)
        self._corners = (self.get_unit_corners() @ self.matrix) - (self.dimensions * 0.5) + self.offset
        
    def read_sample_metadata(self):
        """
        Reads the metadata JSON file from disk and restores
        this sample object's state.
        """
        metadata_filename = os.path.join(self.directory, "sample_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            sample_metadata = json.load(f)

        # Convert lists back to NumPy arrays
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
        base, ext = os.path.splitext(self._default_filenames[0])
        chunk_filename = f"{base}_{chunk_num}{ext}"
        if override_directory is not None:
            np.save(os.path.join(override_directory, chunk_filename), data)
        else:
            np.save(os.path.join(self.directory, chunk_filename), data)
    
    def write_chunk_species(self, data, chunk_num, override_directory=None):
        base, ext = os.path.splitext(self._default_filenames[1])
        chunk_filename = f"{base}_{chunk_num}{ext}"
        if override_directory is not None:
            np.save(os.path.join(override_directory, chunk_filename), data)
        else:
            np.save(os.path.join(self.directory, chunk_filename), data)
            
    def write_sample_metadata(self, override_directory=None):
        """
        Serializes the sample object's critical internal fields to disk 
        as human-readable JSON so that the state can be restored later.
        """
        # Convert NumPy arrays to Python lists so JSON can handle them
        sample_metadata = {
            "dimensions": self._dimensions.tolist() if self._dimensions is not None else None,
            "offset": self._offset.tolist() if self._offset is not None else None,
            "rotation": self._rotation.tolist() if self._rotation is not None else None,
            "chunk_total": int(self._chunk_total) if self._chunk_total is not None else None,
        }

        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "sample_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "sample_metadata.json")

        # Write as nicely formatted JSON
        with open(metadata_filename, "w") as f:
            json.dump(sample_metadata, f, indent=4)
        print(f"Metadata written to {metadata_filename} in JSON format.")
    
    def load_chunk_positions(self, chunk_number, use_gpu=True):
        """
        Load positions from disk. If use_gpu=True and cupy is available, return a cp.ndarray.
        Otherwise, return an np.ndarray.
        """
        base, ext = os.path.splitext(self._default_filenames[0])
        positions_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, positions_filename)
        if use_gpu and (cp is not None):
            positions = cp.load(full_path)
        else:
            positions = np.load(full_path)
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
        """
        Load species from disk. If use_gpu=True and cupy is available, return a cp.ndarray.
        Otherwise, return an np.ndarray.
        """
        base, ext = os.path.splitext(self._default_filenames[1])
        species_filename = f"{base}_{chunk_number}{ext}"
        full_path = os.path.join(self.directory, species_filename)
        if use_gpu and (cp is not None):
            return cp.load(full_path)
        else:
            return np.load(full_path)
    # -------------------------------------
    
    # -------------------------------------
    # KNN search
    def write_chunk_nn_indices(self, index_list, chunk_num, override_directory=None):
        """
        Write a ragged list of neighbor indices for each atom in a chunk.

        Parameters
        ----------
        index_list : list of 1D arrays
            index_list[i] has shape (num_neighbors_i,) storing integer neighbor indices for atom i.
        chunk_num : int
            The chunk number to write.
        override_directory : str, optional
            If provided, write to this directory instead of self.directory.
        """
        base_name = "nearest_neighbors_indices"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        # Number of atoms = number of sub-arrays
        n_atoms = len(index_list)

        # (1) Compute lengths of each sub-array
        lengths = [arr.size for arr in index_list]
        # (2) Offsets array: size n_atoms + 1, with a cumsum
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        # (3) Concatenate all the sub-arrays in a single pass
        if n_atoms > 0:
            flat_idx = np.concatenate(index_list)
        else:
            # Handle empty case
            flat_idx = np.zeros(0, dtype=np.int32)

        # (4) Write to NPZ
        np.savez(save_path, flat_idx=flat_idx, offsets=offsets)
        
    def write_chunk_nn_phase(self, phase_list, chunk_num, override_directory=None):
        """
        Write a ragged list of float phases for each atom in a chunk.

        Parameters
        ----------
        phase_list : list of 1D np.ndarray(float32)
            phase_list[i] has shape (num_neighbors_i,) containing the phases for that atom.
        chunk_num : int
            The chunk number to write.
        override_directory : str, optional
            If provided, write to this directory instead of self.directory.
        """
        base_name = "nearest_neighbors_phase"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(phase_list)
        lengths = [arr.size for arr in phase_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if n_atoms > 0:
            flat_phase = np.concatenate(phase_list)
        else:
            flat_phase = np.zeros(0, dtype=np.float32)

        np.savez(save_path, flat_phase=flat_phase, offsets=offsets)

    def write_chunk_nn_scatter(self, scatter_list, chunk_num, override_directory=None):
        """
        Write a ragged list of wavevectors (kx, ky, kz) for each atom in a chunk.
        Each element in scatter_list is an array of shape (N_neighbors_i, 3).

        Parameters
        ----------
        scatter_list : list of arrays
            scatter_list[i] has shape (N_neighbors_i, 3), storing [kx, ky, kz].
        chunk_num : int
            The chunk number to write.
        override_directory : str, optional
            If provided, write to this directory instead of self.directory.
        """
        base_name = "nearest_neighbors_scatter"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(scatter_list)
        # Each element in scatter_list has shape (num_neighbors_i, 3)
        lengths = [arr.shape[0] for arr in scatter_list]  # neighbors per atom
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        if n_atoms > 0:
            # Flatten kx, ky, kz parts
            flat_kx = np.concatenate([arr[:, 0] for arr in scatter_list])
            flat_ky = np.concatenate([arr[:, 1] for arr in scatter_list])
            flat_kz = np.concatenate([arr[:, 2] for arr in scatter_list])
        else:
            flat_kx = np.zeros(0, dtype=np.float32)
            flat_ky = np.zeros(0, dtype=np.float32)
            flat_kz = np.zeros(0, dtype=np.float32)

        # Save to NPZ
        np.savez(save_path, flat_kx=flat_kx, flat_ky=flat_ky, flat_kz=flat_kz, offsets=offsets)
        
    def write_chunk_nn_species(self, species_list, chunk_num, override_directory=None):
        """
        Write a ragged list of neighbor species for each atom in a chunk.

        This mirrors the pattern used by write_chunk_nn_phase / write_chunk_nn_scatter:
        we flatten the species arrays into a single array and store an offsets array.

        Parameters
        ----------
        species_list : list of 1D arrays (could be string dtype, int dtype, etc.)
            species_list[i] has shape (num_neighbors_i,) storing the species of each neighbor
            for atom i. The dtype can be anything numpy supports (str, int, object), but note
            that some dtypes (e.g., object) may be less portable than numeric or fixed-length
            string arrays.
        chunk_num : int
            The chunk number to write.
        override_directory : str, optional
            If provided, write to this directory instead of self.directory.
        """
        base_name = "nearest_neighbors_species"
        filename = f"{base_name}_{chunk_num}.npz"
        if override_directory is not None:
            save_path = os.path.join(override_directory, filename)
        else:
            save_path = os.path.join(self.directory, filename)

        n_atoms = len(species_list)
        lengths = [arr.size for arr in species_list]
        offsets = np.zeros(n_atoms + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lengths)

        # Concatenate all sub-arrays
        if n_atoms > 0:
            # Make sure they can be concatenated.  If they are strings or mixed types,
            # you might want to ensure they share a compatible dtype.  We'll assume so:
            flat_species = np.concatenate(species_list)
        else:
            # Handle empty case
            flat_species = np.array([], dtype=species_list[0].dtype if n_atoms>0 else np.int32)

        # Save to NPZ
        np.savez(save_path, flat_species=flat_species, offsets=offsets)
        
    def load_chunk_nn_indices(self, chunk_num):
        """
        Load the flat nearest-neighbor index array and offsets for a chunk.
        Returns (flat_idx, offsets).
        """
        base_name = "nearest_neighbors_indices"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN indices file not found: {full_path}")

        with np.load(full_path) as data:
            flat_idx = data['flat_idx']  # shape (total_size,)
            offsets = data['offsets']    # shape (n_atoms+1,)

        return flat_idx, offsets


    def load_chunk_nn_phase(self, chunk_num):
        """
        Load the flat nearest-neighbor phase array and offsets for a chunk.
        Returns (flat_phase, offsets).
        """
        base_name = "nearest_neighbors_phase"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN phase file not found: {full_path}")

        with np.load(full_path) as data:
            flat_phase = data['flat_phase']  # shape (total_size,)
            offsets = data['offsets']        # shape (n_atoms+1,)

        return flat_phase, offsets

    def load_chunk_nn_scatter(self, chunk_num):
        """
        Load the flat nearest-neighbor wavevector arrays and offsets for a chunk.
        Now stores kx, ky, kz in separate arrays.

        Parameters
        ----------
        chunk_num : int
            The chunk number to load.

        Returns
        -------
        flat_kx : np.ndarray
            Concatenated kx values for all atoms' neighbors.
        flat_ky : np.ndarray
            Concatenated ky values for all atoms' neighbors.
        flat_kz : np.ndarray
            Concatenated kz values for all atoms' neighbors.
        offsets : np.ndarray
            The offsets array of shape (n_atoms+1,).
        """
        base_name = "nearest_neighbors_scatter"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN scatter file not found: {full_path}")

        with np.load(full_path) as data:
            flat_kx = data['flat_kx']
            flat_ky = data['flat_ky']
            flat_kz = data['flat_kz']
            offsets = data['offsets']

        return flat_kx, flat_ky, flat_kz, offsets
    
    def load_chunk_nn_species(self, chunk_num):
        """
        Load the flat nearest-neighbor species array and offsets for a chunk.
        Returns (flat_species, offsets).

        You can reconstruct the ragged species_list by something like:

            (flat_spc, offsets) = load_chunk_nn_species(...)
            species_list = []
            for i in range(offsets.size - 1):
                start = offsets[i]
                end   = offsets[i+1]
                species_list.append(flat_spc[start:end])

        Parameters
        ----------
        chunk_num : int
            Which chunk to load.

        Returns
        -------
        flat_species : np.ndarray
            The concatenated neighbor species for all atoms in this chunk.
        offsets : np.ndarray, shape (n_atoms+1,)
            offsets[i] is the start index of the i-th atom's neighbor-species in flat_species.
        """
        base_name = "nearest_neighbors_species"
        filename = f"{base_name}_{chunk_num}.npz"
        full_path = os.path.join(self.directory, filename)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"NN species file not found: {full_path}")

        with np.load(full_path, allow_pickle=True) as data:
            flat_species = data['flat_species']
            offsets = data['offsets']

        return flat_species, offsets
    # -------------------------------------

    # -------------------------------------
    # MD sample   
    def import_atomic_data(self, import_file, element_list, header_lines=9, ID_column=1, position_columns=[2,3,4], scale=1e-10, flush_size=100000000, override_directory=None):
        """
        Reads the atoms from a large text file, skipping the first 9 lines, and
        chunks them into binary .npy files of size flush_size in the desired folder.
        
        The atomic positions are assumed to be in columns 3,4,5 of each line (1-based indexing).
        Also recovers 'dimensions', 'offset', and 'chunk_total' from the bounding box
        of these atomic positions.
        """
        chunk_num = 0
        # Track min/max in x,y,z to calculate dimensions and offset afterward
        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')
        with open(import_file, "r") as f:
            # Skip the first 9 lines
            for _ in range(header_lines):
                next(f)
            while True:
                # Read up to flush_size lines at a time
                lines = []
                for _ in range(flush_size):
                    line = f.readline()
                    if not line:
                        break
                    lines.append(line)
                # If no lines were read, we're at EOF
                if not lines:
                    break
                # Parse positions from columns 3,4,5
                data_arr = np.zeros((len(lines), 3), dtype=np.float32)
                species_arr = []
                for i, line in enumerate(lines):
                    split_line = line.strip().split()
                    species_arr.append(element_list[int(split_line[ID_column])-1])
                    data_arr[i, 0] = float(split_line[position_columns[0]])*float(scale/1e-10)
                    data_arr[i, 1] = float(split_line[position_columns[1]])*float(scale/1e-10)
                    data_arr[i, 2] = float(split_line[position_columns[2]])*float(scale/1e-10)
                    # Update bounding box
                    if data_arr[i, 0] < x_min: x_min = data_arr[i, 0]
                    if data_arr[i, 0] > x_max: x_max = data_arr[i, 0]
                    if data_arr[i, 1] < y_min: y_min = data_arr[i, 1]
                    if data_arr[i, 1] > y_max: y_max = data_arr[i, 1]
                    if data_arr[i, 2] < z_min: z_min = data_arr[i, 2]
                    if data_arr[i, 2] > z_max: z_max = data_arr[i, 2]
                # Increment chunk number and save the positions
                chunk_num += 1
                self.write_chunk_positions(data_arr, chunk_num, override_directory=override_directory)
                self.write_chunk_species(species_arr, chunk_num, override_directory=override_directory)
        # Record how many chunks were created
        self._chunk_total = chunk_num
        # Infer dimensions from bounding box
        self._dimensions = np.array([x_max - x_min, 
                                    y_max - y_min, 
                                    z_max - z_min], dtype=np.float32)
        # Offset is the midpoint of the bounding box (center)
        self._offset = np.array([(x_min + x_max) / 2.0,
                                (y_min + y_max) / 2.0,
                                (z_min + z_max) / 2.0], dtype=np.float32)
        self._rotation = np.eye(3)
    # -------------------------------------

    ## Static Functions
    # -------------------------------------
    # General
    @staticmethod
    def get_unit_corners():
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
    def get_rotation(axis,angle):
        """
        Return the 3x3 rotation matrix for rotation by 'angle' radians
        around the (normalized) 'axis'.
        """
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
        Create a 3D grid of integer coordinates as an (N, 3) array without
        materializing three full 3D arrays. Returns float32 for parity with GPU.
        """
        d0 = int(dimensions[0])
        d1 = int(dimensions[1])
        d2 = int(dimensions[2])

        if use_gpu and (cp is not None):
            ii = cp.repeat(cp.arange(d0, dtype=cp.float32), d1 * d2)
            jj = cp.tile(cp.repeat(cp.arange(d1, dtype=cp.float32), d2), d0)
            kk = cp.tile(cp.arange(d2, dtype=cp.float32), d0 * d1)
            return cp.stack((ii, jj, kk), axis=1)
        else:
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
        Convenience: configure physical mapping (Einstein model) and set temp_params
        so that enable_temp + temp_params drive temperature-based displacements.

        Typical usage:
            s.set_temperature_einstein(
                T_K=300,
                mass_amu=28.0855,
                theta_E_K=400.0,
                max_displacement=3.0  # optional per-axis clip, in position units
            )
            s.enable_temp = True

        For per-species control, pass dictionaries:
            species_mass_amu = {'Si': 28.0855, 'C': 12.011}
            species_theta_E_K = {'Si': 645.0, 'C': 2230.0}
        """
        # Configure temp_params to use the new 'einstein' mode.
        self.enable_temp = True
        self.temp_params = ['einstein', float(T_K), max_displacement, seed]

        # Save global fallbacks if provided
        if mass_amu is not None:
            self._temp_mass_amu = float(mass_amu)
        if theta_E_K is not None:
            self._temp_theta_E_K = float(theta_E_K)

        # Save optional per-species maps
        if species_mass_amu is not None:
            self._temp_species_mass_amu = dict(species_mass_amu)
        if species_theta_E_K is not None:
            self._temp_species_theta_E_K = dict(species_theta_E_K)

        # Default position unit is angstrom. Override with set_position_unit_in_m if needed.
        if not hasattr(self, '_position_unit_in_m'):
            self._position_unit_in_m = 1.0e-10

    def set_position_unit_in_m(self, unit_in_m):
        """
        Set the conversion from one position unit to meters.
        Default is 1e-10 (angstrom). Set to 1e-9 for nanometer units, etc.
        """
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
        Apply random displacements to 'positions' according to:
        - 'gaussian': same behavior as before; 'sigma' is stddev in position units.
        - 'einstein': 'sigma' now means temperature in kelvin; we compute the
            physically-based Gaussian width from the Einstein model:
                <x^2> = (hbar / (2 m omega)) * coth(hbar*omega / (2 k_B T))
            where omega = k_B * theta_E / hbar. You configure m (amu) and theta_E (K)
            globally or per species using 'set_temperature_einstein' (see helper below).
        The optional 'max_displacement' still clips each coordinate if provided.
        A random 'seed' is used for reproducibility.
        """
        # Select backend
        use_cp = (cp is not None) and isinstance(positions, cp.ndarray)
        xp = cp if use_cp else np

        # Seed RNG
        if seed is not None:
            if use_cp:
                cp.random.seed(int(seed))
            else:
                np.random.seed(int(seed))

        # Fast path: keep original Gaussian behavior
        if isinstance(distribution, str) and distribution.lower() in ('gaussian', 'normal'):
            displacements = xp.random.normal(loc=0.0, scale=float(sigma), size=positions.shape)
            if (max_displacement is not None) and (max_displacement > 0.0):
                xp.clip(displacements, -max_displacement, max_displacement, out=displacements)
            return positions + displacements

        # Temperature-driven displacement using the Einstein model
        if isinstance(distribution, str) and distribution.lower() in ('einstein', 'temperature', 'kelvin'):
            # 'sigma' carries T in kelvin in this mode
            T_K = float(sigma)

            # Position unit scale: default assumes arrays are in angstroms (1e-10 m)
            pos_unit_m = getattr(self, '_position_unit_in_m', 1e-10)

            # Physical constants (SI)
            k_B = 1.380649e-23
            hbar = 1.054571817e-34
            amu_to_kg = 1.66053906660e-27

            N = int(positions.shape[0])

            # Resolve masses and Einstein temperatures (either per species or global)
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
                # Load species for this chunk on CPU; map to arrays of m and theta_E
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
                # Use global values (with safe fallbacks)
                m_default = getattr(self, '_temp_mass_amu', 28.0)
                th_default = getattr(self, '_temp_theta_E_K', 300.0)
                masses_amu = np.full(N, m_default, dtype=np.float64)
                thetaE_K = np.full(N, th_default, dtype=np.float64)

            # Move to correct backend
            m_kg = xp.asarray(masses_amu, dtype=xp.float64) * amu_to_kg
            thetaE_K = xp.asarray(thetaE_K, dtype=xp.float64)

            # omega from theta_E: omega = k_B * theta_E / hbar
            omega = (k_B * thetaE_K) / hbar

            # Handle coth safely for very small and very large arguments
            # z = hbar * omega / (2 k_B T)
            if T_K <= 0.0:
                # Zero temperature -> zero-point motion only: coth(z)->1
                coth_z = xp.ones_like(omega, dtype=xp.float64)
            else:
                z = (hbar * omega) / (2.0 * k_B * T_K)
                # Use series near zero to avoid 1/tanh underflow
                small = z < 1.0e-6
                coth_series = (1.0 / z) + (z / 3.0)
                coth_exact = 1.0 / xp.tanh(z)
                coth_z = xp.where(small, coth_series, coth_exact)

            # <x^2> in meters^2
            msd_m2 = (hbar / (2.0 * m_kg * omega)) * coth_z
            # Convert to position units (angstroms if pos_unit_m = 1e-10)
            sigma_units = xp.sqrt(msd_m2) / pos_unit_m  # shape (N,)

            # Draw Gaussian displacements with per-atom sigma
            rand = xp.random.standard_normal(size=positions.shape)
            displacements = rand * sigma_units.reshape(-1, 1)
            # Match dtype of input
            displacements = displacements.astype(positions.dtype, copy=False)

            if (max_displacement is not None) and (max_displacement > 0.0):
                xp.clip(displacements, -max_displacement, max_displacement, out=displacements)

            return positions + displacements

        # Unknown distribution keyword
        raise ValueError("Unknown distribution: {}".format(distribution))
    # -------------------------------------
        
    # -------------------------------------
    # Sample generation
    @staticmethod    
    def compile_parallelepipeds_intersect_batch_cffi():
        '''
        C++ code using 15-axis SAT method for determining if a set of cornerpoints intersects
        with another, made to run a batch operation of corner points against a single reference.
        '''
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
        ffi_obj = FFI()
        ffi_obj.cdef("""int check_parallelepipeds_intersect_batch(
            const double *all_pts1,
            const double *pts2,
            double eps,
            int n,
            int *out_intersect);
        """)
        C_mod = ffi_obj.verify(c_source, extra_compile_args=["-O3"], libraries=[])
        return ffi_obj, C_mod
    # -------------------------------------
    
    # -------------------------------------
    # KNN search
    @staticmethod
    def build_cell_list_count_kernel():
        '''
        '''
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
        # Build raw module
        kernel_module = cp.RawModule(
            code=_cell_list_count_kernel,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return kernel_module.get_function('cell_list_count_kernel')

    @staticmethod
    def build_cell_list_fill_kernel():
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
        # Build raw module
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
        Re-loads each chunk of atomic positions, subtracts the current self.offset
        from every position (centering them), and writes them back out.
        Finally sets self.offset to [0,0,0].
        """
        if self._offset is None:
            raise ValueError("Offset is not initialized. Please set self._offset or load metadata first.")

        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate sample or import atoms first.")
        
        offset_np = self.offset.astype(np.float32)

        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                positions_chunk -= cp.array(offset_np)
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk -= offset_np
                self.write_chunk_positions(positions_chunk, i + 1)

        self._offset = np.zeros(3, dtype=np.float32)
        print("All atomic positions re-centered. Offset is now [0, 0, 0].")
        
    def zero_sample_rotation(self, use_gpu=True):
        """
        Re-loads each chunk of atomic positions, rotates all chunks by the inverse
        of the current self._rotation, and writes them back out.
        Finally sets self._rotation to the 3x3 identity matrix.
        """
        if self._rotation is None:
            raise ValueError("No sample rotation matrix is set. Please initialize or load it first.")
        
        R_inv = self._rotation.T.astype(np.float32)
        
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                R_inv_cp = cp.asarray(R_inv)
                positions_chunk = positions_chunk @ R_inv_cp
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk = positions_chunk @ R_inv
                self.write_chunk_positions(positions_chunk, i + 1)
        
        self._rotation = np.eye(3, dtype=np.float32)
        print("All atomic positions de-rotated. Sample rotation is now the identity matrix.")
        
    def zero_sample(self, use_gpu=True):
        self.zero_sample_position(use_gpu=use_gpu)
        self.zero_sample_rotation(use_gpu=use_gpu)
        
    def rotate_sample_relative(self, axis, dangle, degrees=True, use_gpu=True):
        """
        Re-loads each chunk of atomic positions, rotates it according to self.get_rotation(axis, dangle),
        writes them back out, and then updates self._rotation by left-multiplying with the new rotation.
        """
        if degrees:
            dangle = np.deg2rad(dangle)
        
        R = self.get_rotation(axis, dangle).astype(np.float32)
        
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                R_cp = cp.asarray(R)
                positions_chunk = positions_chunk @ R_cp
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk = positions_chunk @ R
                self.write_chunk_positions(positions_chunk, i + 1)
        
        self._rotation = R @ self._rotation
        print(f"Sample rotated by {dangle:.4f} radians about axis {axis}. "
              f"Updated sample rotation matrix:\n{self._rotation}")

    def translate_sample_relative(self, offset_vector, use_gpu=True): # update this to use dx, dy, dz
        """
        Re-loads each chunk of atomic positions, adds the offset_vector to every position,
        and writes them back out.
        Finally adds offset_vector to self._offset.
        """
        if self._chunk_total is None:
            raise ValueError("Chunk total is not initialized. Please generate or import sample data first.")
        
        offset_np = np.array(offset_vector, dtype=np.float32)
        
        for i in range(self.chunk_total):
            positions_chunk = self.load_chunk_positions(i + 1, use_gpu=use_gpu)
            if cp is not None and isinstance(positions_chunk, cp.ndarray):
                positions_chunk += cp.asarray(offset_np)
                positions_chunk_cpu = positions_chunk.get()
                self.write_chunk_positions(positions_chunk_cpu, i + 1)
            else:
                positions_chunk += offset_np
                self.write_chunk_positions(positions_chunk, i + 1)
        
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
        Compute the selected atomic positions for one geometric chunk entirely on a
        given CUDA stream, and return:
        pos_sel_cp : cp.ndarray of shape (M, 3), float32, positions after offset shift
        idx_sel_cp : cp.ndarray of shape (M,), int64, indices into the flattened atoms
                    (used to map to species on CPU without building a huge tile)
        This does not synchronize; it only enqueues work on 'stream'.
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
        Compute candidate chunk positions and dimensions, prefilter with a cheap
        AABB test against the sample box [0..dimensions], then run SAT via CFFI
        only on the survivors. This greatly reduces SAT calls.
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

        # Reconstruct full mask
        full_mask = np.zeros(chunk_positions_S.shape[0], dtype=bool)
        full_mask[np.flatnonzero(aabb_mask)] = sat_mask_sub

        chunk_positions_S = chunk_positions_S[full_mask, :]
        return chunk_positions_S.astype(np.float32, copy=False), chunk_dimensions.astype(np.float32, copy=False)
        
    def parallelepipeds_intersect_cffi(self, compiled_code, ffi_object, pts1, pts2, eps=1e-12):
        """
        Faster CFFI bridge: avoid Python lists and extra copies by passing buffers
        directly to C. Input arrays are made contiguous float64 and referenced
        through ffi.from_buffer.
        """
        pts1 = np.ascontiguousarray(pts1, dtype=np.float64)
        pts2 = np.ascontiguousarray(pts2, dtype=np.float64)
        n = int(pts1.shape[0])

        results_int = np.zeros(n, dtype=np.int32)

        c_all = ffi_object.from_buffer("double[]", pts1)
        c_arr2 = ffi_object.from_buffer("double[]", pts2)
        c_out = ffi_object.cast("int *", results_int.ctypes.data)

        compiled_code.check_parallelepipeds_intersect_batch(
            c_all, c_arr2, float(eps), n, c_out
        )
        return results_int == 1

    def get_lattice_positions(self, material, chunk_position, chunk_dimensions, use_gpu=True):
        '''
        Gets the location of lattice points in the sample frame in a given chunk.
        
        If use_gpu=True and cupy is installed, returns a cp.ndarray.
        Otherwise returns a np.ndarray.
        '''
        lattice_matrix = material.lattice_matrix.T

        if use_gpu and (cp is not None):
            # GPU path
            lattice_matrix_cp = cp.asarray(lattice_matrix, dtype=cp.float32)
            chunk_position_cp = cp.asarray(chunk_position, dtype=cp.float32)

            lattice_positions_C = self.get_flat_grid(chunk_dimensions, use_gpu=True)
            lattice_positions_S = lattice_positions_C @ lattice_matrix_cp + chunk_position_cp
            return lattice_positions_S

        else:
            # CPU path
            # Ensure single-precision on CPU
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
        Gets atom positions and species for one geometric chunk.

        New optional args:
        - stream: cp.cuda.Stream to enqueue GPU work on.
        - return_on_gpu: if True, keeps positions/mask on GPU and returns them
            along with a site_count so the caller can finish on the host later.
        - lattice_atom_cartesian_cp, offset_gpu, dim_half_gpu:
            optional preloaded cp arrays to avoid re-uploading per chunk.

        Returns:
        if return_on_gpu is False (default): (positions_np, species_np)
        if return_on_gpu is True:  (positions_cp, mask_cp, site_count)
        """
        use_gpu = (use_gpu and (cp is not None))

        if use_gpu:
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

                # Host path identical to original behavior
                mask_np = mask.get()
                positions_np = atomic_positions_S.get()

                atomic_species = np.tile(material.species, int(lattice_positions_cp.shape[0]))
                atomic_species = atomic_species[mask_np]

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

        mask = (
            (atomic_positions_S[:, 0] >= 0) & (atomic_positions_S[:, 0] <= self.dimensions[0]) &
            (atomic_positions_S[:, 1] >= 0) & (atomic_positions_S[:, 1] <= self.dimensions[1]) &
            (atomic_positions_S[:, 2] >= 0) & (atomic_positions_S[:, 2] <= self.dimensions[2])
        )

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
        gpu_streams=6,
        writer_threads=3
    ):
        """
        Generate and write the sample in fixed-size chunks.
        - If use_gpu is True and a CUDA device is available, runs a multi-stream GPU pipeline.
        - If CUDA is unavailable or any GPU error occurs (OOM, runtime error),
        falls back to a pure-CPU pipeline without losing progress already written.
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
        """
        Build a cell list on GPU for the given positions and cutoff r_cut.
        Returns:
        sorted_positions (N, 3) [cp.float32]
        sorted_indices   (N,)    [cp.int32]
        cell_start       (num_cells,) [cp.int32]
        cell_end         (num_cells,) [cp.int32]
        bounding_box_min (3,)    [cp.float32]
        cell_size        (float)
        nx, ny, nz       (int)   # number of cells in each dimension
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

        # 8) We'll do a second pass to fill sorted_positions using an atomicAdd on cell_offsets_copy
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
