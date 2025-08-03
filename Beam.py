# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import json
import os
import sys
import gc
import threading
try:
    import cupy as cp
except ImportError:
    cp = None
from cffi import FFI
import databases.scattering
import importlib.resources as pkg_resources

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class beam:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization    
    def __init__(self, directory=os.getcwd()):
        """
        Initialize a new `beam` object with default or user-specified directory.

        Args:
            directory (str, optional): The file path to store/read beam-related
                metadata. Defaults to the current working directory.
        """
        self.directory = directory
        self._direction = None
        self._energy = None
        self._wavelength = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        # Constants
        self._h = 6.62607015e-34
        self._c = 299792458
        self._q = 1.602176634e-19
        self._hq = self._h / self._q

    def create_beam(self, energy, eV=True,
                    direction=np.array([1.0, 0.0, 0.0]),
                    beam_shape="rectangular",
                    beam_size=(1000.0, 1000.0)):
        """
        Create a beam of specified energy and direction, with a user-specified
        cross-section shape and dimensions (in Angstroms).

        Args:
            energy (float): Beam energy value.
            eV (bool, optional): If True, interpret `energy` in eV; otherwise,
                interpret in Joules. Defaults to True.
            direction (np.ndarray, optional): A 3-element array specifying the
                beam's propagation direction. Defaults to [1.0, 0.0, 0.0].
            beam_shape (str, optional): Shape of the beam cross-section. Current
                valid options include "rectangular" and "circular", though only
                "rectangular" is used in binning. Defaults to "rectangular".
            beam_size (tuple of float, optional): A 2D size tuple (size_y, size_z)
                in Angstroms for the cross-section. Defaults to (1000.0, 1000.0).
        """
        self._direction = direction / np.linalg.norm(direction)
        if not eV:
            energy = energy / self._q
        self._energy = energy
        self._wavelength = self._hq * self._c / self._energy
        self._beam_shape = beam_shape.lower()
        self._beam_size = beam_size
        self._kx_scalar = self._direction[0] * (2.0 * np.pi / self._wavelength)
        self._ky_scalar = self._direction[1] * (2.0 * np.pi / self._wavelength)
        self._kz_scalar = self._direction[2] * (2.0 * np.pi / self._wavelength)
        
    def read_beam_metadata(self):
        """
        Read beam metadata from a JSON file in the current directory, restoring
        the beam's internal state (energy, wavelength, direction, shape, and size).

        Raises:
            FileNotFoundError: If the metadata JSON file does not exist.
        """
        metadata_filename = os.path.join(self.directory, "beam_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            beam_metadata = json.load(f)

        # Convert lists back to NumPy arrays
        if beam_metadata["direction"] is not None:
            self._direction = np.array(beam_metadata["direction"], dtype=np.float32)
        self._energy = beam_metadata["energy"]
        self._wavelength = beam_metadata["wavelength"]

        # If older metadata, might not have shape/size; handle gracefully
        self._beam_shape = beam_metadata.get("beam_shape", "rectangular")
        self._beam_size  = tuple(beam_metadata.get("beam_size", (0.0, 0.0)))

        print(f"Beam metadata loaded from {metadata_filename}.")

    ## Data Handling Functions    
    def write_beam_metadata(self, override_directory=None):
        """
        Serialize the beam's internal state to a JSON file for future restoration.

        Args:
            override_directory (str, optional): If provided, this directory is used
                to store the JSON file. Otherwise, the directory specified during
                initialization (self.directory) is used.
        """
        beam_metadata = {
            "direction"   : self._direction.tolist() if self._direction is not None else None,
            "energy"      : self._energy,
            "wavelength"  : self._wavelength,
            "beam_shape"  : self._beam_shape,
            "beam_size"   : list(self._beam_size)
        }

        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "beam_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "beam_metadata.json")

        with open(metadata_filename, "w") as f:
            json.dump(beam_metadata, f, indent=4)
        print(f"Beam metadata written to {metadata_filename} in JSON format.")

    ## Static Functions
    # -------------------------------------
    # General
    @staticmethod
    def make_orthonormal_basis(direction):
        """
        Generate two orthonormal vectors e1, e2 that are orthogonal to the input direction.

        Args:
            direction (np.ndarray): A 3-element array representing a beam direction.

        Returns:
            tuple of np.ndarray: Two 3-element arrays e1, e2 that are orthonormal
            to each other and to `direction`.
        """
        d = direction / np.linalg.norm(direction)

        # pick an arbitrary vector not parallel to d
        if abs(d[0]) < 0.9:
            temp = np.array([1,0,0], dtype=np.float32)
        else:
            temp = np.array([0,1,0], dtype=np.float32)

        # e1 = d x temp
        e1 = np.cross(d, temp)
        e1 /= np.linalg.norm(e1)

        # e2 = d x e1
        e2 = np.cross(d, e1)
        e2 /= np.linalg.norm(e2)

        return e1.astype(np.float32), e2.astype(np.float32)
    
    @staticmethod
    def allocate_pinned_array(np_array, dtype=np.float32):
        """
        Allocate pinned (page-locked) host memory of the same shape as 'np_array'
        and copy its contents.

        Returns
        -------
        pinned_arr : np.ndarray
            A NumPy array backed by pinned memory. Safe to pass to cp.array(...).
        """
        if dtype is None:
            dtype = np_array.dtype
        shape = np_array.shape
        n_elems = np.prod(shape)
        
        # Allocate pinned block using CuPy
        memptr = cp.cuda.alloc_pinned_memory(
            n_elems * np.dtype(dtype).itemsize
        )
        # Build a NumPy array around that pinned memory
        pinned_arr = np.ndarray(shape=shape, dtype=dtype, buffer=memptr)
        # Copy data into pinned array
        pinned_arr[...] = np_array
        return pinned_arr
    
    @staticmethod
    def parse_f0_db_all(database_name='f0_WaasKirf.dat'):
        """
        Load f0 scattering form factor parameters for all elements from the specified database.

        Args:
            database_name (str, optional): Name of the resource file in `databases.scattering`
                containing the Waasmaier-Kirfel f0 parameters. Defaults to 'f0_WaasKirf.dat'.

        Returns:
            dict: A dictionary mapping element symbols to an array of 11 parameters
            [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5].
        """
        db_dict = {}
        db_file = pkg_resources.open_text(databases.scattering, database_name)
        element = None
        for line in db_file:
            if line.startswith('#S'):
                element = line.split()[2].strip()
            elif (not line.startswith('#')) and element is not None:
                params = np.fromiter((float(x) for x in line.split()), dtype=np.float32)
                if params.size == 11:
                    db_dict[element] = params
        return db_dict

    @staticmethod
    def parse_f1f2_db_all(database_name='f1f2_CromerLiberman.dat'):
        """
        Load anomalous scattering factors f1, f2 for all elements from the specified database.

        Args:
            database_name (str, optional): Name of the resource file in `databases.scattering`
                containing the Cromer-Liberman f1, f2 data. Defaults to 'f1f2_CromerLiberman.dat'.

        Returns:
            dict: A dictionary mapping element symbols to a NumPy array of shape (N, 3)
            with columns [Energy(eV), f1, f2], for each element.
        """
        f1f2_dict = {}
        db_file = pkg_resources.open_text(databases.scattering, database_name)
        element = None
        param_list = []
        for line in db_file:
            if line.startswith('#S'):
                if element is not None and len(param_list) > 0:
                    f1f2_dict[element] = np.array(param_list, dtype=np.float32)
                element = line.split()[2].strip()
                param_list = []
            elif not line.startswith('#') and element is not None:
                row_vals = [float(val) for val in line.split()]
                if len(row_vals) == 3:
                    param_list.append(row_vals)
        if element is not None and len(param_list) > 0:
            f1f2_dict[element] = np.array(param_list, dtype=np.float32)
        return f1f2_dict

    @staticmethod
    def get_f1f2_from_params(energy, f1f2_table):
        """
        Interpolate f1 + i*f2 at a given energy using a table of [E, f1, f2].

        Args:
            energy (float): The energy (in eV) at which to interpolate.
            f1f2_table (np.ndarray): A (N, 3) array of [E, f1, f2] values.

        Returns:
            complex: The complex anomalous scattering factor (f1 + i*f2) at the given energy.
        """
        E = energy
        energies = f1f2_table[:, 0]
        idx = np.searchsorted(energies, E)
        if idx >= len(energies):
            idx = len(energies) - 1
        if idx == 0:
            idx = 1

        E0, f10, f20 = energies[idx - 1], f1f2_table[idx - 1, 1], f1f2_table[idx - 1, 2]
        E1, f11, f21 = energies[idx], f1f2_table[idx, 1], f1f2_table[idx, 2]
        denom = (E1 - E0) if (E1 > E0) else 1e-20

        w = (E - E0) / denom
        f1 = f10 + (f11 - f10)*w
        f2 = f20 + (f21 - f20)*w
        return f1 + 1j*f2
    
    @staticmethod
    def _build_f0_zero_dict(db_dict_f0_all):
        """
        Compute f0(0) for each element from the Waasmaier-Kirfel parameters.

        Args:
            db_dict_f0_all (dict): Dictionary mapping element symbols to f0 parameters
                [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5].

        Returns:
            dict: A dictionary {element_symbol: f0(0) value}.
        """
        f0_0_dict = {}
        for el, params in db_dict_f0_all.items():
            # params layout = [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]
            a1, a2, a3, a4, a5, c = params[0], params[1], params[2], params[3], params[4], params[5]
            # f0(0) = c + sum(a1..a5)
            val = float(c + a1 + a2 + a3 + a4 + a5)
            f0_0_dict[el] = val
        return f0_0_dict
    # -------------------------------------
    
    # -------------------------------------
    # Direct Scattering
    @staticmethod
    def compile_compute_scattering_cffi():
        """
        Compile or verify a CFFI module for CPU-based scattering calculations.

        Returns:
            tuple: (ffi_obj, C_mod), where ffi_obj is the CFFI FFI object and
            C_mod is the compiled C module offering `compute_scattering_cffi(...)`.
        """
        c_source = r'''
        #include <math.h>
        #include <stddef.h>

        static inline float get_f0_value(
            float Q_val,
            const float* params
        )
        {
            // params layout: [a1,a2,a3,a4,a5, c, b1,b2,b3,b4,b5]
            // f0(Q) = c + sum_{i=1..5}( a_i * exp(-b_i * (k^2)) )
            // where k = 0.25 * Q_val * 1.0e-10 / pi
            const float PI_F = 3.14159265358979323846f;
            const float K_SCALE_FACTOR = 0.25f * 1.0e-10f / PI_F;
            float ktmp  = K_SCALE_FACTOR * Q_val;
            float ktmp2 = ktmp * ktmp;

            float f0_val = params[5]; // c
            // accumulate the 5 terms
            for(int i=0; i<5; i++){
                float ai = params[i];
                float bi = params[6 + i];
                f0_val += ai * expf(-bi * ktmp2);
            }
            return f0_val;
        }

        void compute_scattering_cffi(
            int atom_count,
            const float *positions,    // shape=(atom_count,3)
            const float *f0_params,    // shape=(atom_count,11)
            const float *s_anom_real,  // shape=(atom_count,)
            const float *s_anom_imag,  // shape=(atom_count,)
            int Nx, 
            int Ny,
            const float *coords_x,     // shape=(Nx*Ny)
            const float *coords_y,
            const float *coords_z,
            float k_val,
            float *out_r,             // shape=(Nx*Ny)
            float *out_i              // shape=(Nx*Ny)
        )
        {
            const float PI_F = 3.14159265358979323846f;
            float wavelength_m = (2.0f * PI_F) / k_val;
            int pixel_count = Nx * Ny;

            for(int a = 0; a < atom_count; a++){
                float ax = positions[3*a + 0];
                float ay = positions[3*a + 1];
                float az = positions[3*a + 2];

                const float *f0p = &f0_params[a*11];
                float sanom_r = s_anom_real[a];
                float sanom_i = s_anom_imag[a];

                for(int p=0; p < pixel_count; p++){
                    float dx = coords_x[p] - ax;
                    float dy = coords_y[p] - ay;
                    float dz = coords_z[p] - az;

                    float r_det = sqrtf(dx*dx + dy*dy + dz*dz);
                    if(r_det == 0.0f){
                        continue;
                    }
                    float rdx = dx / r_det;
                    float tmp = 2.0f*(1.0f - rdx);
                    if(tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_val * sqrtf(tmp);

                    // Evaluate f0
                    float f0_val = get_f0_value(Q_val, f0p);
                    float real_tot = f0_val + sanom_r;
                    float imag_tot = sanom_i; // f0 is real

                    float ax_mod = fmodf(ax, wavelength_m);
                    float rdet_mod = fmodf(r_det, wavelength_m);
                    // Phase = k_val * ( (ax % λ) + (r_det % λ) )
                    float phase = k_val * (ax_mod + rdet_mod);
                    float cph = cosf(phase);
                    float sph = sinf(phase);

                    float val_r = real_tot * cph - imag_tot * sph;
                    float val_i = real_tot * sph + imag_tot * cph;

                    out_r[p] += val_r;
                    out_i[p] += val_i;
                }
            }
        }
        ''';

        ffi_obj = FFI()
        ffi_obj.cdef(
            r"""
            void compute_scattering_cffi(
                int atom_count,
                const float *positions,
                const float *f0_params,
                const float *s_anom_real,
                const float *s_anom_imag,
                int Nx,
                int Ny,
                const float *coords_x,
                const float *coords_y,
                const float *coords_z,
                float k_val,
                float *out_r,
                float *out_i
            );
            """
        )
        C_mod = ffi_obj.verify(c_source, extra_compile_args=['-O3'])
        return ffi_obj, C_mod
    
    @staticmethod
    def build_interaction_kernel():
        """
        Create a CuPy RawKernel for performing GPU-based scattering calculations
        using a shared-memory approach. Modified to accept a per-atom wavevector
        (kx, ky, kz).
        """
        _cuda_source_memtile = r'''
        #define CHUNK_SIZE 128
        extern "C" {
        __device__ __forceinline__ float2 get_f0_from_params(float Q_val, const float* params)
        {
            // params layout: [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]
            // f0(Q) = c + sum_{i=1..5}( a_i * exp(-b_i*(k^2)) )
            // k = 0.25f * Q_val * 1.0e-10f / PI

            const float PI_F = 3.14159265358979323846f;
            const float K_SCALE_FACTOR = 0.25f * 1.0e-10f / PI_F;
            float k   = K_SCALE_FACTOR * Q_val;
            float kk  = k * k;
            float f0  = params[5]; // c

            #pragma unroll
            for (int i = 0; i < 5; i++)
            {
                float ai = params[i];
                float bi = params[6 + i];
                f0 += ai * __expf(-bi * kk);
            }
            return make_float2(f0, 0.0f);
        }

        __global__ void interaction_kernal(
            const int   nAtoms,
            // Instead of a single scalar k, we take three arrays (kx, ky, kz) of length nAtoms
            const float* kx_atom,         // shape=(nAtoms,)
            const float* ky_atom,         // shape=(nAtoms,)
            const float* kz_atom,         // shape=(nAtoms,)
            const float* px,              // atom positions.x (length nAtoms)
            const float* py,              // atom positions.y
            const float* pz,              // atom positions.z
            const float2* initial_amp,    // per-atom initial amplitude
            const float2* scattering_anom,// (f1 + i f2) for each atom
            const float* f0_params,       // shape=(nAtoms, 11)
            const float* x_coords,        // length Nx*Ny
            const float* y_coords,        // length Nx*Ny
            const float* z_coords,        // length Nx*Ny
            float2*     detector_field,   // shape Nx*Ny
            const int   Nx,
            const int   Ny
        )
        {
            const float PI_F = 3.14159265358979323846f;
            const float rE_F = 2.81794092e-5f;

            // Determine which pixel this thread processes
            int pxid = blockIdx.x * blockDim.x + threadIdx.x;
            int pyid = blockIdx.y * blockDim.y + threadIdx.y;

            // Check if we are in-bounds
            bool in_bounds = (pxid < Nx && pyid < Ny);
            int pixel_index = pyid * Nx + pxid;

            float tx = 0.0f, ty = 0.0f, tz = 0.0f;
            if (in_bounds)
            {
                tx = x_coords[pixel_index];
                ty = y_coords[pixel_index];
                tz = z_coords[pixel_index];
            }

            // Accumulate the result in registers
            float2 sum_val = make_float2(0.0f, 0.0f);

            // Shared memory for tiling
            __shared__ float  s_px[CHUNK_SIZE];
            __shared__ float  s_py[CHUNK_SIZE];
            __shared__ float  s_pz[CHUNK_SIZE];
            __shared__ float2 s_amp[CHUNK_SIZE];
            __shared__ float2 s_anom[CHUNK_SIZE];
            __shared__ float  s_params[CHUNK_SIZE * 11];

            // These are new: store wavevector in tile
            __shared__ float s_kx[CHUNK_SIZE];
            __shared__ float s_ky[CHUNK_SIZE];
            __shared__ float s_kz[CHUNK_SIZE];

            int threads_in_block = blockDim.x * blockDim.y;
            int t_id = threadIdx.y * blockDim.x + threadIdx.x;

            for (int tile_start = 0; tile_start < nAtoms; tile_start += CHUNK_SIZE)
            {
                // Load a tile of atoms into shared memory
                for (int t = t_id; t < CHUNK_SIZE; t += threads_in_block)
                {
                    int atom_idx = tile_start + t;
                    if (atom_idx < nAtoms)
                    {
                        s_px[t]   = px[atom_idx];
                        s_py[t]   = py[atom_idx];
                        s_pz[t]   = pz[atom_idx];
                        s_amp[t]  = initial_amp[atom_idx];
                        s_anom[t] = scattering_anom[atom_idx];

                        // Copy wavevector
                        s_kx[t] = kx_atom[atom_idx];
                        s_ky[t] = ky_atom[atom_idx];
                        s_kz[t] = kz_atom[atom_idx];

                        // Copy 11 f0_params
                        #pragma unroll
                        for (int pi = 0; pi < 11; pi++)
                        {
                            s_params[t * 11 + pi] = f0_params[atom_idx * 11 + pi];
                        }
                    }
                }
                __syncthreads();

                // Each thread accumulates from these chunk atoms if in-bounds
                if (in_bounds)
                {
                    #pragma unroll 4
                    for (int j = 0; j < CHUNK_SIZE; j++)
                    {
                        int global_atom_idx = tile_start + j;
                        if (global_atom_idx >= nAtoms) break;

                        float dx = tx - s_px[j];
                        float dy = ty - s_py[j];
                        float dz = tz - s_pz[j];
                        float r_det = sqrtf(dx*dx + dy*dy + dz*dz);
                        if (r_det == 0.0f) continue;

                        // Compute wave number from s_kx[j], s_ky[j], s_kz[j]
                        float kix = s_kx[j];
                        float kiy = s_ky[j];
                        float kiz = s_kz[j];
                        float k_mag = sqrtf(kix*kix + kiy*kiy + kiz*kiz);
                        if (k_mag < 1.0e-20f) continue;

                        // cos(theta) ~ dot(r_hat, k_in_hat)
                        float dot_val = (dx*kix + dy*kiy + dz*kiz) / (r_det * k_mag);
                        float tmp = 2.0f*(1.0f - dot_val);
                        if (tmp < 0.0f) tmp = 0.0f;

                        float Q_val = k_mag * __fsqrt_rn(tmp);

                        // Evaluate f0
                        const float* param_ptr = &s_params[j * 11];
                        float2 f0c = get_f0_from_params(Q_val, param_ptr);

                        // Add anomalous
                        float2 s_a   = s_anom[j];
                        float2 amp_a = s_amp[j];  // per-atom initial amplitude
                        float2 s_tot = make_float2(f0c.x + s_a.x, f0c.y + s_a.y);

                        // Phase
                        float wavelength_m = (2.0f * PI_F) / k_mag;
                        float ax_mod = fmodf(s_px[j], wavelength_m);
                        float rdet_mod = fmodf(r_det, wavelength_m);
                        float phase = k_mag * (ax_mod + rdet_mod);

                        float cph, sph;
                        __sincosf(phase, &sph, &cph);

                        float2 val;
                        float real_part = amp_a.x * s_tot.x - amp_a.y * s_tot.y; // amplitude * (f0+anom)
                        float imag_part = amp_a.x * s_tot.y + amp_a.y * s_tot.x;
                        // then rotate by e^{i phase}
                        val.x = real_part * cph - imag_part * sph;
                        val.y = real_part * sph + imag_part * cph;

                        sum_val.x += val.x * rE_F;
                        sum_val.y += val.y * rE_F;
                    }
                }
                __syncthreads();
            }

            if (in_bounds)
            {
                detector_field[pixel_index].x += sum_val.x;
                detector_field[pixel_index].y += sum_val.y;
            }
        }
        }
        ''';

        # Build the raw module
        kernel_module = cp.RawModule(
            code=_cuda_source_memtile,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return kernel_module.get_function('interaction_kernal')
    # -------------------------------------
    
    # -------------------------------------
    # Dynamical
    @staticmethod
    def build_intra_neighbor_search_kernel():
        """
        Build a CuPy RawKernel for intra-chunk neighbor search (atom i -> atom i
        within the same chunk). This version records (phase, kx, ky, kz) instead of
        any scattering factors.
        """
        _intra_neighbor_search_kernel = r'''
        #include <math.h>
        
        extern "C" __global__
        void intra_neighbor_search_kernel(
            // Sorted atom data
            const float*  __restrict__ sorted_positions,   // (N,3)
            const int*    __restrict__ sorted_indices,     // (N,)

            // Cell list data
            const int*  __restrict__ cell_start,
            const int*  __restrict__ cell_end,
            const int   nx, 
            const int   ny, 
            const int   nz,

            // neighbor search
            const float  r_cut,
            const float* __restrict__ bounding_box_min,
            const float  cell_size,
            const int    max_neighbors_per_atom,

            // beam constants
            const float  k_val,
            const float  wavelength,

            // outputs
            float* __restrict__ phase_buffer,    // (N*max_neighbors_per_atom)
            float* __restrict__ kx_buffer,       // (N*max_neighbors_per_atom)
            float* __restrict__ ky_buffer,       // (N*max_neighbors_per_atom)
            float* __restrict__ kz_buffer,       // (N*max_neighbors_per_atom)
            int*   __restrict__ neighbor_idx_buffer,  // (N*max_neighbors_per_atom)
            int*   __restrict__ neighbor_counts,      // (N,)

            // total
            const int    N
        )
        {
            // 27 neighbor cell offsets
            const int neighbor_delta[27][3] = {
            {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},
            {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
            {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
            { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},
            { 0, 0,-1}, { 0, 0, 0}, { 0, 0, 1},
            { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
            { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},
            { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
            { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}
            };

            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= N) return;

            float px = sorted_positions[3*i + 0];
            float py = sorted_positions[3*i + 1];
            float pz = sorted_positions[3*i + 2];

            // Cell index for i
            float fx = (px - bounding_box_min[0]) / cell_size;
            float fy = (py - bounding_box_min[1]) / cell_size;
            float fz = (pz - bounding_box_min[2]) / cell_size;

            int cx = (int)floorf(fx);
            int cy = (int)floorf(fy);
            int cz = (int)floorf(fz);

            int neighbor_count = 0;

            for(int n=0; n<27; n++){
                int ncx = cx + neighbor_delta[n][0];
                int ncy = cy + neighbor_delta[n][1];
                int ncz = cz + neighbor_delta[n][2];

                if(ncx<0 || ncx>=nx) continue;
                if(ncy<0 || ncy>=ny) continue;
                if(ncz<0 || ncz>=nz) continue;

                int cell_id = ncz*(nx*ny) + ncy*nx + ncx;
                int start = cell_start[cell_id];
                int end   = cell_end[cell_id];

                for(int j=start; j<end; j++){
                    if(j == i) continue;

                    float qx = sorted_positions[3*j + 0];
                    float qy = sorted_positions[3*j + 1];
                    float qz = sorted_positions[3*j + 2];

                    float dx = qx - px;
                    float dy = qy - py;
                    float dz = qz - pz;
                    float dist2 = dx*dx + dy*dy + dz*dz;
                    if(dist2 <= r_cut*r_cut){
                        if(neighbor_count < max_neighbors_per_atom){
                            float dist = sqrtf(dist2);

                            // Phase = k_val * mod(distance, wavelength)
                            float mod_val = fmodf(dist, wavelength);
                            float phase_val = k_val * mod_val;

                            // Wave vector from i->j is k_val * (dx,dy,dz)/dist
                            float kx = 0.f;
                            float ky = 0.f;
                            float kz = 0.f;
                            if (dist > 1.0e-20f) {
                                kx = k_val * (dx / dist);
                                ky = k_val * (dy / dist);
                                kz = k_val * (dz / dist);
                            }

                            int widx = i*max_neighbors_per_atom + neighbor_count;
                            phase_buffer[widx]      = phase_val;
                            kx_buffer[widx]         = kx;
                            ky_buffer[widx]         = ky;
                            kz_buffer[widx]         = kz;
                            neighbor_idx_buffer[widx] = j;
                        }
                        neighbor_count++;
                    }
                }
            }
            neighbor_counts[i] = neighbor_count;
        }
        ''';

        kernel_module = cp.RawModule(
            code=_intra_neighbor_search_kernel,
            backend='nvcc',
            options=('--gpu-architecture=native','-O3','--ftz=true','--fmad=true')
        )
        return kernel_module.get_function('intra_neighbor_search_kernel')
    
    @staticmethod
    def build_inter_neighbor_search_kernel():
        """
        Build a CuPy RawKernel for inter-chunk neighbor search (between
        two distinct boundary sets). Now it stores (phase, kx, ky, kz)
        and excludes i->i or j->j neighbors.
        """
        _inter_neighbor_search_kernel = r'''
        #include <math.h>
        
        extern "C" __global__
        void inter_neighbor_search_kernel(
            const float*  positions,          // combined (N_total,3)
            const int     N_i,
            const int     N_total,

            const int*  cell_start,
            const int*  cell_end,
            const int   nx,
            const int   ny,
            const int   nz,

            const float  r_cut,
            const float* bounding_box_min,
            const float  cell_size,
            const int    max_neighbors_per_atom,

            const float  k_val,
            const float  wavelength,

            float* phase_buffer,    // shape=(N_total*max_neighbors_per_atom)
            float* kx_buffer,
            float* ky_buffer,
            float* kz_buffer,
            int*   neighbor_idx_buffer,
            int*   neighbor_counts
        )
        {
            const int neighbor_delta[27][3] = {
            {-1,-1,-1}, {-1,-1, 0}, {-1,-1, 1},
            {-1, 0,-1}, {-1, 0, 0}, {-1, 0, 1},
            {-1, 1,-1}, {-1, 1, 0}, {-1, 1, 1},
            { 0,-1,-1}, { 0,-1, 0}, { 0,-1, 1},
            { 0, 0,-1}, { 0, 0, 0}, { 0, 0, 1},
            { 0, 1,-1}, { 0, 1, 0}, { 0, 1, 1},
            { 1,-1,-1}, { 1,-1, 0}, { 1,-1, 1},
            { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
            { 1, 1,-1}, { 1, 1, 0}, { 1, 1, 1}
            };

            int idx = blockDim.x*blockIdx.x + threadIdx.x;
            if(idx >= N_total) return;

            float px = positions[3*idx + 0];
            float py = positions[3*idx + 1];
            float pz = positions[3*idx + 2];

            float fx = (px - bounding_box_min[0]) / cell_size;
            float fy = (py - bounding_box_min[1]) / cell_size;
            float fz = (pz - bounding_box_min[2]) / cell_size;

            int cx = (int)floorf(fx);
            int cy = (int)floorf(fy);
            int cz = (int)floorf(fz);

            bool is_in_i = (idx < N_i);  // chunk i or chunk j
            int neighbor_count = 0;

            for(int n=0; n<27; n++){
                int ncx = cx + neighbor_delta[n][0];
                int ncy = cy + neighbor_delta[n][1];
                int ncz = cz + neighbor_delta[n][2];

                if(ncx<0||ncx>=nx) continue;
                if(ncy<0||ncy>=ny) continue;
                if(ncz<0||ncz>=nz) continue;

                int cell_id = ncz*(nx*ny) + ncy*nx + ncx;
                int start = cell_start[cell_id];
                int end   = cell_end[cell_id];

                for(int j=start; j<end; j++){
                    if(j == idx) continue;

                    // skip i->i or j->j
                    bool neighbor_in_i = (j < N_i);
                    if(is_in_i == neighbor_in_i){
                        continue;
                    }

                    float qx = positions[3*j + 0];
                    float qy = positions[3*j + 1];
                    float qz = positions[3*j + 2];

                    float dx = qx - px;
                    float dy = qy - py;
                    float dz = qz - pz;
                    float dist2 = dx*dx + dy*dy + dz*dz;
                    if(dist2 <= r_cut*r_cut){
                        if(neighbor_count < max_neighbors_per_atom){
                            float dist = sqrtf(dist2);

                            // Phase
                            float mod_val = fmodf(dist, wavelength);
                            float phase_val = k_val*mod_val;

                            // wave vector (kx, ky, kz)
                            float kx_ = 0.f;
                            float ky_ = 0.f;
                            float kz_ = 0.f;
                            if(dist > 1.0e-20f){
                                kx_ = k_val*(dx/dist);
                                ky_ = k_val*(dy/dist);
                                kz_ = k_val*(dz/dist);
                            }

                            int widx = idx*max_neighbors_per_atom + neighbor_count;
                            phase_buffer[widx]      = phase_val;
                            kx_buffer[widx]         = kx_;
                            ky_buffer[widx]         = ky_;
                            kz_buffer[widx]         = kz_;
                            neighbor_idx_buffer[widx] = j;
                        }
                        neighbor_count++;
                    }
                }
            }
            neighbor_counts[idx] = neighbor_count;
        }
        ''';
        kernel_module = cp.RawModule(
            code=_inter_neighbor_search_kernel,
            backend='nvcc',
            options=('--gpu-architecture=native','-O3','--ftz=true','--fmad=true')
        )
        return kernel_module.get_function('inter_neighbor_search_kernel')
    
    @staticmethod
    def build_expand_paths_kernel():
        """
        Returns a CUDA kernel for expanding a set of scattering paths into the next bounce.
        This FIXED version does *not* compute the scattering factor for each neighbor
        inside the GPU kernel. Instead, it merely:

        - Computes the new wavevector (k_out).
        - Accumulates the phase exp(i*phase_ij).
        - Updates the path amplitude with that phase factor only.
        - Stores the neighbor species in out_spc[outPos].
        - Stores the neighbor's atom index in out_atomIndex[outPos].
        - Stores the neighbor position indices, wavevector, and amplitude.

        The actual scattering factor from the neighbor's species will be accounted for
        later during the sub-chunk `process_subchunk(...)`, where we build a per-path
        f0_params and anom array for the final call to build_interaction_kernel().
        """
        code = r'''
        #include <math.h>
        extern "C" {

        __device__ __forceinline__ float2 cplx_mul(const float2 a, const float2 b)
        {
            float2 r;
            r.x = a.x*b.x - a.y*b.y;
            r.y = a.x*b.y + a.y*b.x;
            return r;
        }
        __device__ __forceinline__ float2 cplx_expf(float phase)
        {
            float s, c;
            __sincosf(phase, &s, &c);
            float2 val;
            val.x = c;
            val.y = s;
            return val;
        }

        __global__
        void expand_paths_kernel(
            // Incoming paths (size = numIncomingPaths)
            const float*  in_x,    // positions.x
            const float*  in_y,
            const float*  in_z,
            const float*  in_kx,
            const float*  in_ky,
            const float*  in_kz,
            const float2* in_amp,  
            const int*    in_atomIndex,

            // neighbor info (per-atom)
            const int*    neighborStart,
            const int*    neighborCount,
            const float*  neighborPhase,
            const float*  neighborKx,
            const float*  neighborKy,
            const float*  neighborKz,
            const int*    neighborIdxAtom,
            // NEW: nearest-neighbor species array
            const int*    neighborSpc,

            // global sizes
            const int     numIncomingPaths,
            const float   k_in_mag,
            const float   wavelength,

            // output (expanded) arrays
            float*  out_x,
            float*  out_y,
            float*  out_z,
            float*  out_kx,
            float*  out_ky,
            float*  out_kz,
            float2* out_amp,
            int*    out_atomIndex,
            // NEW: store neighbor species
            int*    out_spc,

            // total capacity
            const int     maxPaths
        )
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= numIncomingPaths) return;

            float sx   = in_x[idx];
            float sy   = in_y[idx];
            float sz   = in_z[idx];
            float skx  = in_kx[idx];
            float sky  = in_ky[idx];
            float skz  = in_kz[idx];
            float2 samp= in_amp[idx];
            int   sourceAtom = in_atomIndex[idx];

            int startN = neighborStart[sourceAtom];
            int countN = neighborCount[sourceAtom];

            for(int n=0; n<countN; n++){
                int globalN = startN + n;

                // The neighbor's atom index, wavevector, species
                int nb_atomIdx = neighborIdxAtom[globalN];
                int nb_spc     = neighborSpc[globalN];  // <--- from spc_flat

                float nkx = neighborKx[globalN];
                float nky = neighborKy[globalN];
                float nkz = neighborKz[globalN];
                float phase_ij = neighborPhase[globalN];

                // Here, we NO LONGER compute the scattering factor. We only multiply
                // the old amplitude by e^{ i * phase_ij }, to keep track of phase.
                float2 eip = cplx_expf(phase_ij);
                float2 new_amp;
                new_amp.x = samp.x*eip.x - samp.y*eip.y;
                new_amp.y = samp.x*eip.y + samp.y*eip.x;

                // The new path's wavevector is (nkx, nky, nkz). The new path's "position"
                // is at the neighbor's atom location, but we store that later if needed.
                // For minimal code, we store zero for out_x,y,z or replicate (sx,sy,sz).
                // We'll just store zero for demonstration:
                
                int outPos = atomicAdd((unsigned int*)&out_atomIndex[maxPaths], 1);
                if(outPos < maxPaths){
                    out_x[outPos]      = 0.f; 
                    out_y[outPos]      = 0.f;
                    out_z[outPos]      = 0.f;
                    out_kx[outPos]     = nkx;
                    out_ky[outPos]     = nky;
                    out_kz[outPos]     = nkz;
                    out_amp[outPos]    = new_amp;
                    out_atomIndex[outPos] = nb_atomIdx;

                    // NEW: store the neighbor's species
                    out_spc[outPos] = nb_spc;
                }
            }
        }
        } // extern "C"
        '''

        kernel_module = cp.RawModule(
            code=code,
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true'),
            backend='nvcc'
        )
        return kernel_module.get_function('expand_paths_kernel')
    # -------------------------------------

    ## Main Functions
    # -------------------------------------
    # Kinematic scattering
    def cpu_scatter_chunk_cffi(self, complied_code, ffi_obj, chunk_id, sample,
                               Nx, Ny, coords_x_m, coords_y_m, coords_z_m,
                               db_dict_f0_all, db_dict_f1f2_all, k_val,
                               stage):
        """
        Compute scattering contributions for a single chunk on CPU using
        a CFFI-based routine.

        Args:
            complied_code: The verified CFFI module containing the compiled C function.
            ffi_obj (FFI): The CFFI FFI object associated with `complied_code`.
            chunk_id (int): The chunk index to process.
            sample: The sample object providing methods like
                `load_chunk_positions(...)` and `load_chunk_species(...)`.
            Nx (int): The number of pixels along the x-dimension of the detector.
            Ny (int): The number of pixels along the y-dimension of the detector.
            coords_x_m (np.ndarray): x-coordinates of the detector pixels in meters.
            coords_y_m (np.ndarray): y-coordinates of the detector pixels in meters.
            coords_z_m (np.ndarray): z-coordinates of the detector pixels in meters.
            db_dict_f0_all (dict): f0 database for all elements.
            db_dict_f1f2_all (dict): f1f2 database for all elements.
            k_val (float): Wave number (2π / wavelength).
            stage: A stage object with rotation (3x3) and translation (3,) to apply to positions.

        Returns:
            np.ndarray: A (Ny, Nx) array of complex64 representing the partial
            scattering field from this chunk.
        """
        species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
        atom_count = species_chunk_np.shape[0]
        if atom_count == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)

        # Build scattering arrays
        scattering_anom_np_real = np.zeros(atom_count, dtype=np.float32)
        scattering_anom_np_imag = np.zeros(atom_count, dtype=np.float32)
        f0_params_np = np.zeros((atom_count, 11), dtype=np.float32)

        unique_elements = pd.unique(species_chunk_np)
        for el in unique_elements:
            if el not in db_dict_f0_all:
                continue
            mask = (species_chunk_np == el)
            table = db_dict_f1f2_all.get(el, None)
            if table is not None:
                cplx = self.get_f1f2_from_params(self._energy, table)
                scattering_anom_np_real[mask] = cplx.real
                scattering_anom_np_imag[mask] = cplx.imag

            f0_params_np[mask] = db_dict_f0_all[el]

        # Load positions (in Angstrom)
        positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)

        # Stage translation and rotation
        positions_chunk = positions_chunk @ stage.rotation
        positions_chunk += stage.translation

        # Convert to contiguous before dividing to meters
        positions_chunk = np.ascontiguousarray(positions_chunk)

        # Convert to meters
        positions_chunk[:, 0] /= 1e10
        positions_chunk[:, 1] /= 1e10
        positions_chunk[:, 2] /= 1e10

        # Output arrays
        out_r = np.zeros(Nx*Ny, dtype=np.float32)
        out_i = np.zeros(Nx*Ny, dtype=np.float32)

        # Ensure everything is contiguous for CFFI
        f0_params_np          = np.ascontiguousarray(f0_params_np)
        scattering_anom_np_real = np.ascontiguousarray(scattering_anom_np_real)
        scattering_anom_np_imag = np.ascontiguousarray(scattering_anom_np_imag)

        # Convert to pointers
        positions_ptr   = ffi_obj.cast("const float *", positions_chunk.ctypes.data)
        f0_params_ptr   = ffi_obj.cast("const float *", f0_params_np.ctypes.data)
        s_anom_r_ptr    = ffi_obj.cast("const float *", scattering_anom_np_real.ctypes.data)
        s_anom_i_ptr    = ffi_obj.cast("const float *", scattering_anom_np_imag.ctypes.data)
        coords_x_ptr    = ffi_obj.cast("const float *", coords_x_m.ctypes.data)
        coords_y_ptr    = ffi_obj.cast("const float *", coords_y_m.ctypes.data)
        coords_z_ptr    = ffi_obj.cast("const float *", coords_z_m.ctypes.data)
        out_r_ptr       = ffi_obj.cast("float *", out_r.ctypes.data)
        out_i_ptr       = ffi_obj.cast("float *", out_i.ctypes.data)

        # Call the C function
        complied_code.compute_scattering_cffi(
            atom_count, positions_ptr, f0_params_ptr,
            s_anom_r_ptr, s_anom_i_ptr, Nx, Ny,
            coords_x_ptr, coords_y_ptr, coords_z_ptr,
            k_val, out_r_ptr, out_i_ptr
        )

        partial_field = (out_r + 1j*out_i).reshape((Ny, Nx)).astype(np.complex64)
        return partial_field

    def interact_beam_cpu(self, sample, measurement_positions, measurement_shape, stage):
        """
        Perform multi-threaded CPU scattering computation for all chunks in a sample.

        Args:
            sample: The sample object with methods to load species and positions by chunk.
            measurement_positions (np.ndarray or cp.ndarray): (3, Nx*Ny) array of
                detector pixel coordinates in Angstrom.
            measurement_shape (tuple of int): (Nx, Ny) specifying detector dimensions.
            stage: A stage object with rotation (3x3) and translation (3,) to apply to positions.

        Returns:
            np.ndarray: A (Ny, Nx) array of complex64 containing the summed scattering field.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        Nx, Ny = measurement_shape

        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        k_val = np.float32(2.0 * np.pi / self._wavelength)

        # Ensure measurement_positions is contiguous float32
        # Then convert to meters
        if cp is not None and isinstance(measurement_positions, cp.ndarray):
            measurement_positions = measurement_positions.get()
        coords_x_m = np.ascontiguousarray(measurement_positions[0, :].astype(np.float32) / 1e10)
        coords_y_m = np.ascontiguousarray(measurement_positions[1, :].astype(np.float32) / 1e10)
        coords_z_m = np.ascontiguousarray(measurement_positions[2, :].astype(np.float32) / 1e10)

        chunk_total = sample.chunk_total
        if chunk_total == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)

        import multiprocessing
        max_threads = multiprocessing.cpu_count()
        n_threads = min(chunk_total, max_threads)

        # Build CFFI library only once
        ffi_obj, complied_code = self.compile_compute_scattering_cffi()

        def worker(chunk_id):
            return self.cpu_scatter_chunk_cffi(
                complied_code, ffi_obj,
                chunk_id, sample, Nx, Ny,
                coords_x_m, coords_y_m, coords_z_m,
                db_dict_f0_all, db_dict_f1f2_all,
                k_val, stage
            )

        final_result = np.zeros((Ny, Nx), dtype=np.complex64)
        chunk_ids = range(1, chunk_total + 1)

        with ThreadPoolExecutor(max_workers=n_threads) as exe:
            futures = {exe.submit(worker, cid): cid for cid in chunk_ids}
            for fut in as_completed(futures):
                partial_2d = fut.result()
                final_result += partial_2d

        return final_result

    def interact_beam_gpu(self, sample, measurement_positions, measurement_shape, stage):
        """
        Perform GPU-based scattering computation for all chunks in a sample.

        Distributes chunk processing across multiple GPUs (if available) and
        aggregates the partial fields on the CPU at the end.

        Args:
            sample: The sample object with methods to load species and positions by chunk.
            measurement_positions (np.ndarray or cp.ndarray): (3, Nx*Ny) array of
                detector pixel coordinates in Angstrom.
            measurement_shape (tuple of int): (Nx, Ny) specifying detector dimensions.
            stage: A stage object with rotation (3x3) and translation (3,) to apply to positions.

        Returns:
            np.ndarray: A (Ny, Nx) complex64 array of the total scattering field.
        """
        if cp is None:
            # If Cupy not available, fallback
            print("[beam] Cupy not installed, falling back to CPU.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            print("[beam] No GPUs found, falling back to CPU.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage)

        print(f"[beam] Found {n_gpus} GPU(s).")

        # Database lookups
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        Nx, Ny = measurement_shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        x_coords = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        y_coords = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        z_coords = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)
        stage_rotation = self.allocate_pinned_array(stage.rotation)
        stage_translation = self.allocate_pinned_array(stage.translation)
        
        chunk_total = sample.chunk_total
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")

        # Divide chunk indices among GPUs
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus

        partial_results = [None] * n_gpus
        interaction_kernel = self.build_interaction_kernel()

        def gpu_worker(gpu_id, x_coords, y_coords, z_coords, chunk_indices, result_index):
            """One thread per GPU."""
            cp.cuda.Device(gpu_id).use()
            
            # Create Stage variables 
            R_stage_gpu = cp.asarray(stage_rotation, dtype=cp.float32)
            trans_stage_gpu = cp.asarray(stage_translation, dtype=cp.float32)
            
            # Create detector variables
            x_coords_gpu = cp.asarray(x_coords)
            y_coords_gpu = cp.asarray(y_coords)
            z_coords_gpu = cp.asarray(z_coords)
            detector_field_gpu = cp.zeros((Nx * Ny,), dtype=cp.complex64)

            # Use streams for concurrency on that GPU
            num_streams = 4
            streams = [cp.cuda.Stream() for _ in range(num_streams)]

            block_size = (16, 16)
            grid_size = ((Nx + block_size[0] - 1) // block_size[0],
                         (Ny + block_size[1] - 1) // block_size[1])

            for i, cidx in enumerate(chunk_indices):
                stream = streams[i % num_streams]

                # Load species on CPU, then positions on GPU
                species_chunk_np = sample.load_chunk_species(cidx, use_gpu=False)
                atom_count = species_chunk_np.shape[0]
                if atom_count == 0:
                    continue

                # Build scattering arrays on CPU
                scattering_anom_np = np.zeros(atom_count, dtype=np.complex64)
                f0_params_np       = np.zeros((atom_count, 11), dtype=np.float32)

                unique_elements = pd.unique(species_chunk_np)
                for el in unique_elements:
                    if el not in db_dict_f0_all:
                        continue
                    mask = (species_chunk_np == el)
                    # Interpolate f1,f2
                    table = db_dict_f1f2_all.get(el, None)
                    if table is not None:
                        scattering_anom_np[mask] = self.get_f1f2_from_params(self._energy, table)
                    # f0
                    f0_params_np[mask] = db_dict_f0_all[el]

                with stream:
                    # Load chunk positions on GPU
                    positions_chunk_cp = cp.array(sample.load_chunk_positions(cidx, use_gpu=True),
                                                  dtype=cp.float32)

                    # Stage translation and rotation
                    positions_chunk_cp = positions_chunk_cp @ R_stage_gpu
                    positions_chunk_cp += trans_stage_gpu

                    # Convert to meters
                    px = positions_chunk_cp[:, 0] / 1e10
                    py = positions_chunk_cp[:, 1] / 1e10
                    pz = positions_chunk_cp[:, 2] / 1e10

                    scattering_anom_cp = cp.asarray(scattering_anom_np)
                    f0_params_cp       = cp.asarray(f0_params_np)

                    # Per-atom wavevector arrays
                    # Replicate the global k_vec for all atoms in this chunk:
                    kx_cp = cp.full(atom_count, self._kx_scalar, dtype=cp.float32)
                    ky_cp = cp.full(atom_count, self._ky_scalar, dtype=cp.float32)
                    kz_cp = cp.full(atom_count, self._kz_scalar, dtype=cp.float32)
                    
                    initial_amp_cp = cp.ones(atom_count, dtype=cp.complex64)

                    interaction_kernel(
                        grid_size,
                        block_size,
                        (
                            np.int32(atom_count),
                            kx_cp,            # kx_atom
                            ky_cp,            # ky_atom
                            kz_cp,            # kz_atom
                            px,               # px
                            py,               # py
                            pz,               # pz
                            initial_amp_cp,
                            scattering_anom_cp,
                            f0_params_cp,
                            x_coords_gpu,
                            y_coords_gpu,
                            z_coords_gpu,
                            detector_field_gpu,
                            np.int32(Nx),
                            np.int32(Ny)
                        ),
                        stream=stream
                    )

                    # Periodically free memory
                    if ((i % 16 == 0) and (i != 0)) or (i == (len(chunk_indices)-1)):
                        stream.synchronize()
                        cp.get_default_memory_pool().free_all_blocks()

            # Sync all streams
            for s in streams:
                s.synchronize()

            # Copy partial back to CPU
            partial_results[result_index] = detector_field_gpu.reshape((Ny, Nx)).get()

            # Cleanup
            del x_coords_gpu, y_coords_gpu, z_coords_gpu
            del detector_field_gpu
            for s in streams:
                del s
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Launch one thread per GPU
        threads = []
        start_chunk = 1
        for gpu_id in range(n_gpus):
            my_count = chunks_per_gpu + (1 if gpu_id < remainder else 0)
            end_chunk = start_chunk + my_count
            chunk_indices = list(range(start_chunk, end_chunk))
            start_chunk = end_chunk
            t = threading.Thread(target=gpu_worker,
                                 args=(gpu_id,
                                       x_coords, y_coords, z_coords,
                                       chunk_indices, gpu_id))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Sum partial results
        for pr in partial_results:
            if pr is not None:
                final_result += pr

        return final_result
    
    def atomic_scattering_kinematic(self, sample, detector, stage, offset=None, use_gpu=True):
        """
        Compute kinematic scattering of a beam from a sample onto a detector,
        now using the actual beam direction rather than assuming +x.

        Args:
            sample: The sample object containing chunked atomic data.
            detector: The detector object with `pixel_coordinates` (3, Nx*Ny) 
                    and `shape` (Nx, Ny).
            stage: A stage object specifying rotation (3x3) and translation (3,).
            offset (float, optional): If provided, subtract from the final scattering field.
            use_gpu (bool, optional): Whether to attempt GPU mode. Defaults to True.

        Returns:
            np.ndarray: A (Ny, Nx) array of complex scattering amplitudes.
        """
        measurement_positions = detector.pixel_coordinates
        Nx, Ny = detector.shape

        # Check if we can run GPU
        if use_gpu and (cp is not None):
            # Attempt GPU path
            final_field = self.interact_beam_gpu(sample, measurement_positions, (Nx, Ny), stage)
        else:
            # CPU fallback
            if cp is None and use_gpu:
                print("[beam] Cupy not installed, running CPU mode.")
            final_field = self.interact_beam_cpu(sample, measurement_positions, (Nx, Ny), stage)

        if offset is not None:
            return final_field - offset
        else:
            return final_field
    # -------------------------------------
        
    # -------------------------------------
    # Direct transmission
    def bin_atoms_in_pixels_cpu(self, sample, Nx, Ny, e1, e2,
                                pixel_size_u, pixel_size_v,
                                stage, atomic_radius=1.7, kernel_radius=0, detector=None):
        """
        Map atoms to detector pixels on CPU. Each atom is projected onto the basis (e1, e2),
        binned into the 2D plane, and optionally convolved with a Gaussian kernel.

        Args:
            sample: The sample object providing chunked atomic data.
            Nx (int): Number of pixels in the x-direction.
            Ny (int): Number of pixels in the y-direction.
            e1 (np.ndarray): A 3-element array representing the first in-plane basis vector.
            e2 (np.ndarray): A 3-element array representing the second in-plane basis vector.
            pixel_size_u (float): Size of a pixel along e1 in Angstroms.
            pixel_size_v (float): Size of a pixel along e2 in Angstroms.
            stage: A stage object with rotation (3x3) and translation (3,).
            atomic_radius (float, optional): Approximate atomic radius in Angstroms
                to expand each atom's binning. Defaults to 1.7.
            kernel_radius (int, optional): The radius of the Gaussian kernel for
                2D convolution. If 0, no smoothing is applied. Defaults to 0.
            detector (object, optional): A detector object providing
                `pixel_coordinates`. Defaults to None.

        Returns:
            np.ndarray: A (Ny, Nx) array of complex64 indicating the real + i*imag
            form factor sums in each pixel (after optional smoothing).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing
        
        # Retrieve pixel centers from detector
        #    shape=(3, Nx*Ny)
        pix_coords = detector.pixel_coordinates
        if cp is not None and isinstance(pix_coords, cp.ndarray):
            pix_coords = pix_coords.get()  # Bring to CPU
        # Project each pixel center onto (e1, e2)
        px = pix_coords[0]
        py = pix_coords[1]
        pz = pix_coords[2]
        pixel_u = px*e1[0] + py*e1[1] + pz*e1[2]
        pixel_v = px*e2[0] + py*e2[1] + pz*e2[2]
        min_u = pixel_u.min()
        min_v = pixel_v.min()

        # Build a 2D Gaussian kernel for smoothing
        def _make_gaussian_kernel_cpu(radius):
            if radius < 1:
                return None
            sigma = radius/2.0
            y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
            g = np.exp(-(x*x + y*y)/(2.0*sigma*sigma)).astype(np.float32)
            g /= g.sum()
            return g

        def _fft_convolve2d_cpu(data2d, kernel):
            if kernel is None:
                return data2d
            s1, s2 = data2d.shape
            k1, k2 = kernel.shape
            fft_shape = (s1 + k1 - 1, s2 + k2 - 1)
            Fdata   = np.fft.fft2(data2d, s=fft_shape)
            Fkernel = np.fft.fft2(kernel, s=fft_shape)
            Fout    = Fdata * Fkernel
            conved  = np.fft.ifft2(Fout)
            start_x = (k1 - 1)//2
            start_y = (k2 - 1)//2
            conved  = conved[start_x:start_x+s1, start_y:start_y+s2]
            return conved.real.astype(data2d.dtype)

        # Pre‐load scattering DB
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero_dict     = self._build_f0_zero_dict(db_dict_f0_all)

        radius = atomic_radius
        offsets = np.array([
            [-radius*np.sqrt(2), -radius*np.sqrt(2)],
            [-radius,            0.0],
            [-radius*np.sqrt(2), +radius*np.sqrt(2)],
            [           0.0,         -radius],
            [           0.0,          0.0],
            [           0.0,         +radius],
            [+radius*np.sqrt(2), -radius*np.sqrt(2)],
            [+radius,            0.0],
            [+radius*np.sqrt(2), +radius*np.sqrt(2)]
        ], dtype=np.float32)
        oy = offsets[:,0]
        oz = offsets[:,1]

        final_map_real = np.zeros((Ny, Nx), dtype=np.float32)
        final_map_imag = np.zeros((Ny, Nx), dtype=np.float32)
        chunk_total = sample.chunk_total
        if chunk_total == 0:
            return (final_map_real + 1j*final_map_imag).astype(np.complex64)

        max_threads = min(chunk_total, multiprocessing.cpu_count())

        def worker(chunk_id):
            pos = sample.load_chunk_positions(chunk_id, use_gpu=False)
            spc = sample.load_chunk_species(chunk_id,  use_gpu=False)
            nAtoms = pos.shape[0]
            if nAtoms == 0:
                return (np.zeros((Ny, Nx), np.float32),
                        np.zeros((Ny, Nx), np.float32))
            
            # stage transform
            pos = pos @ stage.rotation
            pos += stage.translation

            # build scattering real, imag
            scattering_real = np.zeros(nAtoms, dtype=np.float32)
            scattering_imag = np.zeros(nAtoms, dtype=np.float32)
            unique_els = pd.unique(spc)
            for el in unique_els:
                mask = (spc == el)
                if el not in f0_zero_dict:
                    continue
                f0_0 = f0_zero_dict[el]
                table = db_dict_f1f2_all.get(el)
                if table is not None:
                    cplx = self.get_f1f2_from_params(self._energy, table)
                    f1_val = cplx.real
                    f2_val = cplx.imag
                else:
                    f1_val = 0.0
                    f2_val = 0.0
                scattering_real[mask] = f0_0 + f1_val
                scattering_imag[mask] = f2_val

            # project atoms onto e1,e2
            atom_u = (pos[:,0]*e1[0] + pos[:,1]*e1[1] + pos[:,2]*e1[2])
            atom_v = (pos[:,0]*e2[0] + pos[:,1]*e2[1] + pos[:,2]*e2[2])

            # expand for offsets
            expanded_u = (atom_u[:,None] + oy[None,:]).ravel()
            expanded_v = (atom_v[:,None] + oz[None,:]).ravel()
            atom_ids   = np.repeat(np.arange(nAtoms, dtype=np.uint64), offsets.shape[0])

            # bin i,j
            i = np.floor((expanded_u - min_u)/pixel_size_u).astype(np.int32)
            j = np.floor((expanded_v - min_v)/pixel_size_v).astype(np.int32)

            # clip
            mask_in = (i>=0)&(i<Nx)&(j>=0)&(j<Ny)
            if not mask_in.any():
                return (np.zeros((Ny, Nx), np.float32),
                        np.zeros((Ny, Nx), np.float32))

            i_valid = i[mask_in]
            j_valid = j[mask_in]
            atom_ids_valid = atom_ids[mask_in]

            bin_idx = (j_valid.astype(np.uint64)*Nx + i_valid.astype(np.uint64))
            encoded = (atom_ids_valid << 32) | bin_idx

            sidx = np.argsort(encoded)
            encoded_sorted = encoded[sidx]
            bin_idx_sorted = bin_idx[sidx]
            atom_sorted    = atom_ids_valid[sidx]

            keep = np.ones(encoded_sorted.size, dtype=bool)
            if encoded_sorted.size > 1:
                keep[1:] = (encoded_sorted[1:] != encoded_sorted[:-1])

            bin_idx_unique  = bin_idx_sorted[keep]
            atom_ids_unique = atom_sorted[keep]

            r_e = 2.81794092e-5 # in Angstrom
            w_real = scattering_real[atom_ids_unique]
            partial_hist_real = np.bincount(bin_idx_unique.astype(np.int64),
                                            weights=(r_e*w_real-1), minlength=Nx*Ny)
            w_imag = scattering_imag[atom_ids_unique]
            partial_hist_imag = np.bincount(bin_idx_unique.astype(np.int64),
                                            weights=(r_e*w_imag-1), minlength=Nx*Ny)

            return (partial_hist_real.reshape((Ny, Nx)),
                    partial_hist_imag.reshape((Ny, Nx)))

        from concurrent.futures import ThreadPoolExecutor
        chunk_ids = range(1, chunk_total+1)
        with ThreadPoolExecutor(max_workers=max_threads) as exe:
            futs = {exe.submit(worker, cid): cid for cid in chunk_ids}
            for fut in as_completed(futs):
                rpart, ipart = fut.result()
                final_map_real += rpart
                final_map_imag += ipart

        # Convolve
        kernel = _make_gaussian_kernel_cpu(kernel_radius)
        conv_real = _fft_convolve2d_cpu(final_map_real, kernel)
        conv_imag = _fft_convolve2d_cpu(final_map_imag, kernel)

        return (conv_real + 1j*conv_imag).astype(np.complex64)
    
    def bin_atoms_in_pixels_gpu(self, sample, Nx, Ny, e1, e2,
                                pixel_size_u, pixel_size_v,
                                stage, atomic_radius=1.7, kernel_radius=0, detector=None):
        """
        GPU-based version of binning atoms into detector pixels. Applies an optional
        Gaussian convolution for atomic finite size.

        Args:
            sample: The sample object providing chunked atomic data.
            Nx (int): Number of pixels in the x-direction.
            Ny (int): Number of pixels in the y-direction.
            e1 (np.ndarray): A 3-element array representing the first in-plane basis vector.
            e2 (np.ndarray): A 3-element array representing the second in-plane basis vector.
            pixel_size_u (float): Size of a pixel along e1 in Angstroms.
            pixel_size_v (float): Size of a pixel along e2 in Angstroms.
            stage: A stage object with rotation (3x3) and translation (3,).
            atomic_radius (float, optional): Approximate atomic radius in Angstroms.
            kernel_radius (int, optional): The radius of the Gaussian kernel for
                smoothing. If 0, no smoothing is applied.
            detector (object, optional): A detector object providing
                `pixel_coordinates`. Defaults to None.

        Returns:
            cupy.ndarray or np.ndarray: A (Ny, Nx) complex array of the binned
            form factor contributions. GPU array if CuPy is available and used,
            otherwise CPU array.
        """
        if cp is None:
            print("[beam] Cupy not installed, fallback to CPU.")
            return self.bin_atoms_in_pixels_cpu(sample, Nx, Ny, e1, e2,
                                                pixel_size_u, pixel_size_v,
                                                stage,atomic_radius=atomic_radius,
                                                kernel_radius=kernel_radius,
                                                detector=detector)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            print("[beam] No GPUs found, fallback to CPU.")
            return self.bin_atoms_in_pixels_cpu(sample, Nx, Ny, e1, e2,
                                                pixel_size_u, pixel_size_v,
                                                stage,atomic_radius=atomic_radius,
                                                kernel_radius=kernel_radius,
                                                detector=detector)

        chunk_total = sample.chunk_total
        if chunk_total == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)
        
        # Convert or ensure pixel coordinates are pinned
        pix_coords = self.allocate_pinned_array(detector.pixel_coordinates)
        e1 = self.allocate_pinned_array(e1)
        e2 = self.allocate_pinned_array(e2)
        stage_rotation = self.allocate_pinned_array(stage.rotation)
        stage_translation = self.allocate_pinned_array(stage.translation)

        # Build GPU Gaussian kernel
        def _make_gaussian_kernel_gpu(radius):
            if radius < 1:
                return None
            sigma = radius/2.0
            y = cp.arange(-radius, radius+1, dtype=cp.float32)[:, None]
            x = cp.arange(-radius, radius+1, dtype=cp.float32)[None, :]
            g = cp.exp(-(x*x + y*y)/(2.0*sigma*sigma))
            g = g.astype(cp.float32)
            g /= cp.sum(g)
            return g

        def _fft_convolve2d_gpu(data2d_gpu, kernel_gpu):
            if kernel_gpu is None:
                return data2d_gpu
            s1, s2 = data2d_gpu.shape
            k1, k2 = kernel_gpu.shape
            Fdata   = cp.fft.fft2(data2d_gpu, s=(s1+k1-1, s2+k2-1))
            Fkernel = cp.fft.fft2(kernel_gpu, s=(s1+k1-1, s2+k2-1))
            Fout    = Fdata * Fkernel
            conved  = cp.fft.ifft2(Fout)
            sx = (k1-1)//2
            sy = (k2-1)//2
            conved  = conved[sx:sx+s1, sy:sy+s2]
            return conved.real.astype(data2d_gpu.dtype)

        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero_dict     = self._build_f0_zero_dict(db_dict_f0_all)

        radius = atomic_radius
        offsets_cpu = np.array([
            [-radius*np.sqrt(2), -radius*np.sqrt(2)],
            [-radius,  0.0],
            [-radius*np.sqrt(2), +radius*np.sqrt(2)],
            [ 0.0, -radius],
            [ 0.0,  0.0],
            [ 0.0, +radius],
            [+radius*np.sqrt(2), -radius*np.sqrt(2)],
            [+radius,  0.0],
            [+radius*np.sqrt(2), +radius*np.sqrt(2)]
        ], dtype=np.float32)
        offsets_cpu = self.allocate_pinned_array(offsets_cpu)

        partial_results = [None] * n_gpus
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus

        def gpu_worker(gpu_id, chunk_indices, out_idx):
            cp.cuda.Device(gpu_id).use()
            
            offsets_gpu = cp.asarray(offsets_cpu)
            oy_gpu = offsets_gpu[:, 0]
            oz_gpu = offsets_gpu[:, 1]
            out_r = cp.zeros((Ny, Nx), dtype=cp.float32)
            out_i = cp.zeros((Ny, Nx), dtype=cp.float32)
            
            # Convert e1, e2, stage transforms to GPU arrays
            e1_gpu = cp.asarray(e1, dtype=cp.float32)
            e2_gpu = cp.asarray(e2, dtype=cp.float32)
            R_stage_gpu = cp.asarray(stage_rotation, dtype=cp.float32)
            trans_stage_gpu = cp.asarray(stage_translation, dtype=cp.float32)

            # Project pixel centers onto (u,v) basis
            px = cp.asarray(pix_coords[0])
            py = cp.asarray(pix_coords[1])
            pz = cp.asarray(pix_coords[2])
            pixel_u = px*e1_gpu[0] + py*e1_gpu[1] + pz*e1_gpu[2]
            pixel_v = px*e2_gpu[0] + py*e2_gpu[1] + pz*e2_gpu[2]

            # Compute min_u and min_v on GPU
            min_u_gpu = pixel_u.min()
            min_v_gpu = pixel_v.min()

            for cid in chunk_indices:
                pos_cpu = sample.load_chunk_positions(cid, use_gpu=False)
                spc_cpu = sample.load_chunk_species(cid,  use_gpu=False)
                nAtoms  = pos_cpu.shape[0]
                if nAtoms == 0:
                    continue

                # scattering arrays
                scattering_real = np.zeros(nAtoms, dtype=np.float32)
                scattering_imag = np.zeros(nAtoms, dtype=np.float32)
                unique_els = pd.unique(spc_cpu)
                for el in unique_els:
                    mask = (spc_cpu == el)
                    if el not in f0_zero_dict:
                        continue
                    f0_0 = f0_zero_dict[el]
                    table = db_dict_f1f2_all.get(el, None)
                    if table is not None:
                        cplx = self.get_f1f2_from_params(self._energy, table)
                        f1_val = cplx.real
                        f2_val = cplx.imag
                    else:
                        f1_val = 0.0
                        f2_val = 0.0
                    scattering_real[mask] = f0_0 + f1_val
                    scattering_imag[mask] = f2_val

                pos_gpu = cp.asarray(pos_cpu, dtype=cp.float32)
                real_gpu = cp.asarray(scattering_real, dtype=cp.float32)
                imag_gpu = cp.asarray(scattering_imag, dtype=cp.float32)

                # stage
                pos_gpu = pos_gpu @ R_stage_gpu
                pos_gpu += trans_stage_gpu

                ax = pos_gpu[:,0]
                ay = pos_gpu[:,1]
                az = pos_gpu[:,2]
                au = ax*e1_gpu[0] + ay*e1_gpu[1] + az*e1_gpu[2]
                av = ax*e2_gpu[0] + ay*e2_gpu[1] + az*e2_gpu[2]

                # expand offsets
                au_exp = (au[:,None] + oy_gpu[None,:]).ravel()
                av_exp = (av[:,None] + oz_gpu[None,:]).ravel()
                atom_ids = cp.repeat(cp.arange(nAtoms, dtype=cp.uint64), offsets_gpu.shape[0])

                # bin i, j
                i_gpu = cp.floor((au_exp - min_u_gpu)/pixel_size_u).astype(cp.int32)
                j_gpu = cp.floor((av_exp - min_v_gpu)/pixel_size_v).astype(cp.int32)

                mask_in = (i_gpu>=0)&(i_gpu<Nx)&(j_gpu>=0)&(j_gpu<Ny)
                if not mask_in.any():
                    continue

                i_valid = i_gpu[mask_in]
                j_valid = j_gpu[mask_in]
                atoms_valid = atom_ids[mask_in]

                bin_idx = (j_valid.astype(cp.uint64)*Nx + i_valid.astype(cp.uint64))
                encoded = (atoms_valid << 32) | bin_idx

                sidx = cp.argsort(encoded)
                enc_sort = encoded[sidx]
                bin_sort = bin_idx[sidx]
                atom_sort= atoms_valid[sidx]

                keep = cp.ones(enc_sort.size, dtype=cp.bool_)
                if enc_sort.size > 1:
                    diffs = (enc_sort[1:] != enc_sort[:-1])
                    keep[1:] = diffs

                bin_uniq  = bin_sort[keep]
                atom_uniq = atom_sort[keep]

                w_real = real_gpu[atom_uniq]
                w_imag = imag_gpu[atom_uniq]

                r_e = 2.81794092e-5 # in Angstrom
                hist_r = cp.bincount(bin_uniq, weights=(r_e*w_real-1), minlength=Nx*Ny)
                hist_i = cp.bincount(bin_uniq, weights=(r_e*w_imag-1), minlength=Nx*Ny)
                out_r += hist_r.reshape(Ny, Nx)
                out_i += hist_i.reshape(Ny, Nx)

                del pos_gpu, real_gpu, imag_gpu
                del i_gpu, j_gpu, mask_in, atom_ids
                del encoded, sidx, enc_sort, bin_sort, atom_sort
                del bin_uniq, atom_uniq, hist_r, hist_i
                cp.get_default_memory_pool().free_all_blocks()

            partial_results[out_idx] = (out_r, out_i)

        threads = []
        start_chunk = 1
        for gpu_id in range(n_gpus):
            my_count = chunks_per_gpu + (1 if gpu_id < remainder else 0)
            end_chunk = start_chunk + my_count
            chunk_indices = range(start_chunk, end_chunk)
            start_chunk = end_chunk
            t = threading.Thread(target=gpu_worker, args=(gpu_id, chunk_indices, gpu_id))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Sum partials
        final_real = np.zeros((Ny, Nx), dtype=np.float32)
        final_imag = np.zeros((Ny, Nx), dtype=np.float32)
        for pr in partial_results:
            if pr is not None:
                real, imag = pr
                final_real += real.get()
                final_imag += imag.get()

        # Convolve
        cp.cuda.Device(0).use()
        final_real_gpu = cp.asarray(final_real)
        final_imag_gpu = cp.asarray(final_imag)
        kernel_gpu = _make_gaussian_kernel_gpu(kernel_radius)
        final_real_gpu = _fft_convolve2d_gpu(final_real_gpu, kernel_gpu)
        final_imag_gpu = _fft_convolve2d_gpu(final_imag_gpu, kernel_gpu)

        out_gpu = final_real_gpu + 1j*final_imag_gpu
        return out_gpu.astype(cp.complex64)
    
    def atomic_transmission(self, sample, detector, stage, use_gpu=True, atomic_radius=1.7, kernel_radius=0):
        """
        Compute a projection-based transmission map by summing the atoms' form factors
        (f0(0) + f1 + i*f2) in each detector pixel, optionally convolving with a
        Gaussian kernel to account for finite atomic radii.

        Args:
            sample: The sample object holding atomic data in chunks.
            detector: The detector object with `shape` -> (Ny, Nx),
                `pixel_coordinates` (3, Nx*Ny), and `pixel_size` -> (pixel_size_u, pixel_size_v).
            stage: A stage object with rotation (3x3) and translation (3,).
            use_gpu (bool, optional): If True and GPUs are available, computation is
                performed on the GPU. Defaults to True.
            atomic_radius (float, optional): Approximate atomic radius in Angstroms. Defaults to 1.7.
            kernel_radius (int, optional): Radius of the Gaussian kernel for 2D smoothing.
                Defaults to 0.

        Returns:
            np.ndarray: A 2D array (Nx, Ny) of the real part of the map after summation
            and convolution, transposed to match the detector's indexing.
        """
        Ny, Nx = detector.shape
        pixel_size_u, pixel_size_v = detector.pixel_size
        
        # Build orthonormal basis
        e1, e2 = self.make_orthonormal_basis(self._direction)

        if use_gpu and (cp is not None):
            f0_map_complex = self.bin_atoms_in_pixels_gpu(
                sample, Nx, Ny, e1, e2,
                pixel_size_u, pixel_size_v,
                stage, atomic_radius=atomic_radius, 
                kernel_radius=kernel_radius,
                detector=detector
            )
            final_map = f0_map_complex.real/((pixel_size_u*pixel_size_v)*(1+atomic_radius)) # Factor to account for additional ampltude absorption from finite atom size
            final_map -= cp.min(final_map)
            final_map = cp.asnumpy(final_map)
        else:
            f0_map_complex = self.bin_atoms_in_pixels_cpu(
                sample, Nx, Ny, e1, e2,
                pixel_size_u, pixel_size_v,
                stage, atomic_radius=atomic_radius,
                kernel_radius=kernel_radius,
                detector=detector
            )
            final_map = f0_map_complex.real/((pixel_size_u*pixel_size_v)*(1+atomic_radius))
            final_map -= np.min(final_map)

        return final_map.T
    # -------------------------------------
    
    # -------------------------------------
    # Dynamical scattering
    def compute_intra_chunk_neighbors_gpu(
        self,
        sample,
        positions,           # cp.ndarray (N,3)
        r_cut=5.0,
        max_neighbors_per_atom=32
    ):
        """
        Perform an intra-chunk nearest neighbor search on GPU, now returning
        (phase, [kx, ky, kz], neighbor_idx) for each atom. We do NOT compute
        scattering factors anymore.
        """
        N = positions.shape[0]
        if N == 0:
            # Empty chunk => return empty structures
            return [
                (np.array([], dtype=np.float32),
                np.zeros((0,3), dtype=np.float32),
                np.array([], dtype=np.int32))
                for _ in range(N)
            ]

        # 1) Build cell list => sorted_positions, sorted_indices
        (sorted_positions,
        sorted_indices,
        cell_start,
        cell_end,
        box_min,
        cell_size,
        nx, ny, nz) = sample.build_cell_list_gpu(positions, r_cut)

        # 2) Prepare output buffers
        phase_gpu  = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        kx_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        ky_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        kz_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        idx_gpu    = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.int32)
        counts_gpu = cp.zeros((N,), dtype=cp.int32)

        # wave number
        wavelength_angs = self._wavelength
        k_val = (2.0 * np.pi) / wavelength_angs

        # 3) Launch kernel
        kernel = self.build_intra_neighbor_search_kernel()
        threads_per_block = 256
        blocks = (N + threads_per_block - 1)//threads_per_block

        kernel(
            (blocks,), (threads_per_block,),
            (
                sorted_positions,
                sorted_indices,
                cell_start,
                cell_end,
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                cp.float32(r_cut),
                box_min,
                cp.float32(cell_size),
                np.int32(max_neighbors_per_atom),
                cp.float32(k_val),
                cp.float32(wavelength_angs),
                phase_gpu,
                kx_gpu,
                ky_gpu,
                kz_gpu,
                idx_gpu,
                counts_gpu,
                np.int32(N)
            )
        )

        # 4) Move data to CPU
        phase_arr  = phase_gpu.reshape(N, max_neighbors_per_atom).get()
        kx_arr     = kx_gpu.reshape(N, max_neighbors_per_atom).get()
        ky_arr     = ky_gpu.reshape(N, max_neighbors_per_atom).get()
        kz_arr     = kz_gpu.reshape(N, max_neighbors_per_atom).get()
        idx_arr    = idx_gpu.reshape(N, max_neighbors_per_atom).get()
        counts_arr = counts_gpu.get()
        sorted_idx_arr = sorted_indices.get()

        # 5) Rebuild ragged lists in original order
        output = [None]*N
        for sorted_i in range(N):
            orig_i = sorted_idx_arr[sorted_i]
            used_count = counts_arr[sorted_i]
            used = min(used_count, max_neighbors_per_atom)
            if used <= 0:
                output[orig_i] = (
                    np.array([], dtype=np.float32),
                    np.zeros((0,3), dtype=np.float32),
                    np.array([], dtype=np.int32)
                )
                continue

            ph_sub  = phase_arr[sorted_i, :used]
            kx_sub  = kx_arr[sorted_i, :used]
            ky_sub  = ky_arr[sorted_i, :used]
            kz_sub  = kz_arr[sorted_i, :used]
            idx_sub = idx_arr[sorted_i, :used]

            # stack k vectors => shape (num_neighbors, 3)
            kvec_sub = np.vstack([kx_sub, ky_sub, kz_sub]).T
            output[orig_i] = (ph_sub, kvec_sub, idx_sub)

        return output

    def compute_inter_chunk_neighbors_gpu(
        self,
        sample,
        pos_i,         # cp.ndarray (N_i,3)
        pos_j,         # cp.ndarray (N_j,3)
        r_cut,
        max_neighbors_per_atom=32
    ):
        """
        Perform a cross-chunk neighbor search on GPU between two boundary
        sets i->j, j->i. Return a list of length (N_i + N_j) with
        (phase_array, kvec_3, neighbor_idx_array).

        The first N_i entries correspond to atoms in chunk i, the last N_j
        entries correspond to atoms in chunk j.
        """
        N_i = pos_i.shape[0]
        N_j = pos_j.shape[0]
        if N_i == 0 and N_j == 0:
            return []
        if N_i == 0 or N_j == 0:
            # one chunk empty => no cross neighbors
            blank_list = [(np.array([], dtype=np.float32),
                        np.zeros((0,3), dtype=np.float32),
                        np.array([], dtype=np.int32))
                        for _ in range(N_i + N_j)]
            return blank_list

        # Combine
        pos_comb = cp.concatenate([pos_i, pos_j], axis=0)
        N_total  = N_i + N_j

        # build cell list
        (sorted_positions,
        sorted_indices,
        cell_start,
        cell_end,
        box_min,
        cell_size,
        nx, ny, nz) = sample.build_cell_list_gpu(pos_comb, r_cut)

        # Output buffers
        phase_gpu  = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        kx_gpu     = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        ky_gpu     = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        kz_gpu     = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        idx_gpu    = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.int32)
        counts_gpu = cp.zeros((N_total,), dtype=cp.int32)

        wavelength_angs = self._wavelength
        k_val = (2.0 * np.pi)/wavelength_angs

        kernel = self.build_inter_neighbor_search_kernel()
        threads_per_block = 256
        blocks = (N_total + threads_per_block - 1)//threads_per_block

        kernel(
            (blocks,), (threads_per_block,),
            (
                sorted_positions,
                np.int32(N_i),
                np.int32(N_total),
                cell_start,
                cell_end,
                np.int32(nx),
                np.int32(ny),
                np.int32(nz),
                cp.float32(r_cut),
                box_min,
                cp.float32(cell_size),
                np.int32(max_neighbors_per_atom),
                cp.float32(k_val),
                cp.float32(wavelength_angs),
                phase_gpu,
                kx_gpu,
                ky_gpu,
                kz_gpu,
                idx_gpu,
                counts_gpu
            )
        )

        # copy to CPU
        phase_arr  = phase_gpu.reshape(N_total, max_neighbors_per_atom).get()
        kx_arr     = kx_gpu.reshape(N_total, max_neighbors_per_atom).get()
        ky_arr     = ky_gpu.reshape(N_total, max_neighbors_per_atom).get()
        kz_arr     = kz_gpu.reshape(N_total, max_neighbors_per_atom).get()
        idx_arr    = idx_gpu.reshape(N_total, max_neighbors_per_atom).get()
        counts_arr = counts_gpu.get()
        sorted_idx_arr = sorted_indices.get()

        out_list = [None]*N_total
        for sorted_i in range(N_total):
            orig_i = sorted_idx_arr[sorted_i]
            used_count = counts_arr[sorted_i]
            used = min(used_count, max_neighbors_per_atom)
            if used <= 0:
                out_list[orig_i] = (
                    np.array([], dtype=np.float32),
                    np.zeros((0,3), dtype=np.float32),
                    np.array([], dtype=np.int32)
                )
                continue

            ph_sub  = phase_arr[sorted_i, :used]
            kx_sub  = kx_arr[sorted_i, :used]
            ky_sub  = ky_arr[sorted_i, :used]
            kz_sub  = kz_arr[sorted_i, :used]
            idx_sub = idx_arr[sorted_i, :used]

            kvec_sub = np.vstack([kx_sub, ky_sub, kz_sub]).T
            out_list[orig_i] = (ph_sub, kvec_sub, idx_sub)

        return out_list
    
    def compute_nearest_neighbor_distances_passA(self, sample, r_cut, max_neighbors_per_atom):
        """
        Pass A: Intra-chunk neighbor searches for all chunks, identify boundary
        atoms, and store (phase, kx, ky, kz, neighbor_idx, neighbor_species).
        We do not compute or store scattering factors here.
        """
        boundary_dict   = {}
        all_data_memory = {}

        for cid in range(1, sample.chunk_total+1):
            chunk_positions = sample.load_chunk_positions(cid, use_gpu=True)
            chunk_species   = sample.load_chunk_species(cid, use_gpu=False)  # CPU side
            n_atoms = chunk_positions.shape[0]

            if n_atoms == 0:
                # trivial chunk
                sample.write_chunk_nn_phase([], cid)
                sample.write_chunk_nn_scatter([], cid)  # reusing "scatter" slot for k-vectors
                sample.write_chunk_nn_indices([], cid)
                ### NEW ###
                sample.write_chunk_nn_species([], cid)
                ###
                boundary_dict[cid] = {
                    "positions": cp.zeros((0,3), dtype=cp.float32),
                    "indices":   cp.zeros((0,),  dtype=cp.int32),
                    "species":   np.array([], dtype=chunk_species.dtype)
                }
                all_data_memory[cid] = []
                continue

            # (1) Intra-chunk i->i neighbors
            results_intra = self.compute_intra_chunk_neighbors_gpu(
                sample,
                chunk_positions,
                r_cut=r_cut,
                max_neighbors_per_atom=max_neighbors_per_atom
            )
            # results_intra is a list of length n_atoms, each entry:
            #   (phase_array, kvec_3, neighbor_idx_array)

            # (2) Identify boundary
            min_val = cp.min(chunk_positions, axis=0)
            max_val = cp.max(chunk_positions, axis=0)
            margin  = r_cut
            cond_min = cp.any((chunk_positions - min_val) < margin, axis=1)
            cond_max = cp.any((max_val - chunk_positions) < margin, axis=1)
            boundary_mask = (cond_min | cond_max)
            boundary_positions = chunk_positions[boundary_mask]
            boundary_indices   = cp.arange(n_atoms, dtype=cp.int32)[boundary_mask]
            boundary_mask_cpu  = boundary_mask.get()

            boundary_species   = chunk_species[boundary_mask_cpu]

            # (3) Write partial .npz-like data
            phase_list    = []
            kvector_list  = []
            idx_list      = []
            species_list  = []  ### NEW ###

            ### We build an augmented data structure (ph, kvec, idx, spc) ###
            results_intra_with_spc = [None] * n_atoms

            for i_atom, (ph, kvec_3, n_idx) in enumerate(results_intra):
                # The neighbor species for this atom come from chunk_species[n_idx].
                # n_idx is a 1D array of neighbor indices in the same chunk.
                n_spc = chunk_species[n_idx]  # CPU side indexing

                phase_list.append(ph.astype(np.float32))
                kvector_list.append(kvec_3.astype(np.float32))
                idx_list.append(n_idx.astype(np.int32))
                ### NEW ###
                species_list.append(n_spc)  # store neighbor species array
                ###

                # Also store internally for pass B merges
                results_intra_with_spc[i_atom] = (ph, kvec_3, n_idx, n_spc)

            sample.write_chunk_nn_phase(phase_list, cid)
            sample.write_chunk_nn_scatter(kvector_list, cid)  # still using 'scatter' slot
            sample.write_chunk_nn_indices(idx_list, cid)
            ### NEW ###
            sample.write_chunk_nn_species(species_list, cid)
            ###

            # Save to all_data_memory for pass B usage
            all_data_memory[cid] = results_intra_with_spc

            # Save boundary
            boundary_dict[cid] = {
                "positions": boundary_positions,
                "indices":   boundary_indices,
                "species":   boundary_species
            }

            del chunk_positions
            cp.get_default_memory_pool().free_all_blocks()

        return boundary_dict, all_data_memory


    def compute_nearest_neighbor_distances_passB(self, sample, boundary_dict, all_data_memory,
                                                r_cut, max_neighbors_per_atom):
        """
        Pass B: Inter-chunk neighbor searches among boundary atoms, storing
        (phase, kx, ky, kz, neighbor_idx, neighbor_species) cross-chunk. Merges
        results back into all_data_memory (augmented with neighbor species).
        """
        # Build bounding boxes for boundary sets
        chunk_bounds = {}
        for cid in range(1, sample.chunk_total+1):
            posB = boundary_dict[cid]["positions"]
            if posB.size == 0:
                chunk_bounds[cid] = (None, None)
                continue
            min_bb = cp.min(posB, axis=0)
            max_bb = cp.max(posB, axis=0)
            chunk_bounds[cid] = (min_bb, max_bb)

        for i in range(1, sample.chunk_total+1):
            i_bd   = boundary_dict[i]
            i_data = all_data_memory[i]  # list of (ph, kvec, idx, spc)
            pos_i  = i_bd["positions"]
            idx_i  = i_bd["indices"]
            spc_i  = i_bd["species"]  # boundary species
            if pos_i.size == 0:
                continue
            N_i = pos_i.shape[0]
            min_i, max_i = chunk_bounds[i]

            for j in range(i+1, sample.chunk_total+1):
                j_bd   = boundary_dict[j]
                j_data = all_data_memory[j]  # list of (ph, kvec, idx, spc)
                pos_j  = j_bd["positions"]
                idx_j  = j_bd["indices"]
                spc_j  = j_bd["species"]
                if pos_j.size == 0:
                    continue
                N_j = pos_j.shape[0]
                min_j, max_j = chunk_bounds[j]

                # bounding-box check (quick reject)
                if (min_i is None) or (min_j is None):
                    continue
                if cp.any((max_i + r_cut) < (min_j - r_cut)) or cp.any((max_j + r_cut) < (min_i - r_cut)):
                    continue

                # cross-chunk merges
                cross_list = self.compute_inter_chunk_neighbors_gpu(
                    sample,
                    pos_i, pos_j,
                    r_cut=r_cut,
                    max_neighbors_per_atom=max_neighbors_per_atom
                )
                # cross_list has length (N_i + N_j), each entry is (ph_new, kvec_new, idx_new)

                idx_i_cpu = idx_i.get()  # boundary -> original chunk index
                idx_j_cpu = idx_j.get()

                # first N_i => chunk i
                for local_i in range(N_i):
                    (ph_new, kvec_new, idx_new) = cross_list[local_i]
                    if ph_new.size > 0:
                        global_i = idx_i_cpu[local_i]  # the actual atom index in chunk i
                        (ph_old, kvec_old, idx_old, spc_old) = i_data[global_i]

                        ### NEW: Build neighbor species array from chunk j boundary ###
                        spc_new_list = []
                        for nb_ind in idx_new:
                            # If nb_ind < N_i => boundary i, else boundary j
                            # but since we did i->j, the neighbors for i
                            #   must come from j side in cross_list. (We skip i->i or j->j in the kernel.)
                            # So nb_ind should always fall in j's domain: nb_ind >= N_i
                            # but we can still check:
                            if nb_ind >= N_i:
                                spc_new_list.append(spc_j[nb_ind - N_i])
                            else:
                                # Edge case: if code merges i->i
                                spc_new_list.append(spc_i[nb_ind])
                        spc_new = np.array(spc_new_list, dtype=spc_old.dtype)

                        i_data[global_i] = (
                            np.concatenate([ph_old, ph_new]),
                            np.vstack([kvec_old, kvec_new]),
                            np.concatenate([idx_old, idx_new]),
                            np.concatenate([spc_old, spc_new])  # neighbor species
                        )

                # next N_j => chunk j
                for local_j in range(N_j):
                    (ph_new, kvec_new, idx_new) = cross_list[N_i + local_j]
                    if ph_new.size > 0:
                        global_j = idx_j_cpu[local_j]
                        (ph_old, kvec_old, idx_old, spc_old) = j_data[global_j]

                        ### NEW: Build neighbor species array from chunk i boundary ###
                        spc_new_list = []
                        for nb_ind in idx_new:
                            # Now neighbors for j must be from i side
                            # so nb_ind < N_i typically:
                            if nb_ind < N_i:
                                spc_new_list.append(spc_i[nb_ind])
                            else:
                                # Edge case
                                spc_new_list.append(spc_j[nb_ind - N_i])
                        spc_new = np.array(spc_new_list, dtype=spc_old.dtype)

                        j_data[global_j] = (
                            np.concatenate([ph_old, ph_new]),
                            np.vstack([kvec_old, kvec_new]),
                            np.concatenate([idx_old, idx_new]),
                            np.concatenate([spc_old, spc_new])
                        )

                del cross_list
                cp.get_default_memory_pool().free_all_blocks()

        return all_data_memory


    def compute_nearest_neighbor_distances(self, sample, r_cut=5.0, use_gpu=True, max_neighbors_per_atom=32):
        """
        Orchestrate the computation of nearest neighbors (intra-chunk and inter-chunk)
        for all atoms in the sample. We store only (phase, kx, ky, kz, neighbor_idx,
        neighbor_species) now, Provide removing any prior scattering references.

        Pass A: Intra-chunk neighbors
        Pass B: Inter-chunk neighbors
        Pass C: Save final arrays (phase, kvec, idx, species).
        """
        if (not use_gpu) or (cp is None):
            raise ValueError("GPU usage required, but CuPy is not available or use_gpu=False.")
        if sample.chunk_total is None:
            raise ValueError("No chunks found; import or generate sample first.")

        # Pass A
        boundary_dict, all_data_memory = self.compute_nearest_neighbor_distances_passA(
            sample, r_cut, max_neighbors_per_atom
        )

        # Pass B
        all_data_memory = self.compute_nearest_neighbor_distances_passB(
            sample, boundary_dict, all_data_memory, r_cut, max_neighbors_per_atom
        )

        # Pass C: re-save final arrays (phase, kvec, idx, species)
        for cid in range(1, sample.chunk_total+1):
            final_list = all_data_memory[cid]  # list of (ph_arr, kvec_3, idx_arr, spc_arr)
            phase_list    = []
            kvector_list  = []
            idx_list      = []
            species_list  = []

            for (ph_arr, kvec_3, idx_arr, spc_arr) in final_list:
                phase_list.append(ph_arr.astype(np.float32))
                kvector_list.append(kvec_3.astype(np.float32))
                idx_list.append(idx_arr.astype(np.int32))
                ### NEW ###
                species_list.append(spc_arr)  # keep user dtype or cast if needed

            sample.write_chunk_nn_phase(phase_list, cid)
            sample.write_chunk_nn_scatter(kvector_list, cid)  # re-use "scatter" slot for k vectors
            sample.write_chunk_nn_indices(idx_list, cid)
            ### NEW ###
            sample.write_chunk_nn_species(species_list, cid)

        print(f"[beam] Completed nearest-neighbor calculation with cutoff={r_cut} "
            f"for {sample.chunk_total} chunks (GPU).")
        
    def atomic_scattering_dynamical(self, sample, detector, stage, n_bounces=0,
                                    offset=None, use_gpu=True, sub_chunk_size=100_000):
        """
        Full multi-bounce GPU code that:
        1) For bounce=0: uses build_interaction_kernel to scatter from each atom (wavevector=beam, amplitude=1+0j).
        2) For bounces > 0: expansions with expand_paths_kernel, but now each path only accumulates
            a phase. We store the neighbor species in out_spc so that in process_subchunk we can
            build the correct scattering factor array for each path before calling interaction_kernel.
        3) We do sub-chunk processing for memory reasons. 
        4) Summation is accumulated in a final detector_field array on GPU, then returned to CPU.

        This replacement ensures that each path's species is used to build the correct
        scattering array in process_subchunk, fixing the original bug where a single chunk-level
        scattering array was incorrectly reused for all expanded paths.
        """
        if (not use_gpu) or (cp is None):
            raise RuntimeError("GPU-based dynamical code requires CuPy installed and use_gpu=True.")
        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            raise RuntimeError("No GPUs found for dynamical scattering.")

        chunk_total = sample.chunk_total
        if chunk_total == 0:
            final_result = np.zeros(detector.shape[::-1], dtype=np.complex64)
            if offset is not None:
                final_result -= offset
            return final_result

        print(f"[beam] Using GPU dynamical scattering with up to {n_bounces} bounce(s).")
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")

        # Build scattering DB
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        Nx, Ny = detector.shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        # Pinned detector arrays
        measurement_positions = detector.pixel_coordinates
        px_pin = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        py_pin = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        pz_pin = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)

        # Stage pinned
        R_stage_pin = self.allocate_pinned_array(stage.rotation)
        T_stage_pin = self.allocate_pinned_array(stage.translation)

        # K‐vector magnitude of the beam
        k_mag = np.float32(np.sqrt(self._kx_scalar**2 + self._ky_scalar**2 + self._kz_scalar**2))
        wavelength_angs = np.float32(self._wavelength)

        chunk_per_gpu = chunk_total // n_gpus
        remainder     = chunk_total % n_gpus

        partial_results = [None] * n_gpus

        # Build or load kernels
        interaction_kernel = self.build_interaction_kernel()
        expand_kernel      = self.build_expand_paths_kernel()  # The updated one above!

        def gpu_worker(gpu_id, chunk_list, out_idx):
            cp.cuda.Device(gpu_id).use()

            # Stage & detector on this GPU
            R_stage_gpu = cp.asarray(R_stage_pin, dtype=cp.float32)
            T_stage_gpu = cp.asarray(T_stage_pin, dtype=cp.float32)
            pxg = cp.asarray(px_pin)
            pyg = cp.asarray(py_pin)
            pzg = cp.asarray(pz_pin)

            dfield_gpu = cp.zeros((Nx*Ny,), dtype=cp.complex64)

            block2d = (16,16)
            grid2d  = ((Nx + block2d[0] - 1)//block2d[0],
                    (Ny + block2d[1] - 1)//block2d[1])

            block1d = 256

            # Helper to run the final interaction (per sub-chunk)
            def process_subchunk(start_idx, end_idx,
                                out_x_gpu, out_y_gpu, out_z_gpu,
                                out_kx_gpu, out_ky_gpu, out_kz_gpu,
                                out_amp_gpu, out_spc_gpu,
                                sub_sz):
                """
                Build a per-path scattering array from out_spc_gpu, then call build_interaction_kernel
                so that each path uses the correct scattering factor.
                """
                # Slice out the path subset
                sub_x   = out_x_gpu[start_idx:end_idx]
                sub_y   = out_y_gpu[start_idx:end_idx]
                sub_z   = out_z_gpu[start_idx:end_idx]
                sub_kx  = out_kx_gpu[start_idx:end_idx]
                sub_ky  = out_ky_gpu[start_idx:end_idx]
                sub_kz  = out_kz_gpu[start_idx:end_idx]
                sub_amp = out_amp_gpu[start_idx:end_idx]
                sub_spc = out_spc_gpu[start_idx:end_idx]

                # Move sub_spc to CPU to do Python dictionary lookups
                sub_spc_cpu = sub_spc.get()

                # Build the f0_params and anom arrays for these paths
                f0_local = np.zeros((sub_sz, 11), dtype=np.float32)
                anom_local = np.zeros(sub_sz, dtype=np.complex64)

                for i in range(sub_sz):
                    # species ID at path i
                    spc_id = sub_spc_cpu[i]
                    # Suppose you have a mapping or direct usage: spc_id -> element symbol
                    # e.g. 'C' = 6, 'O' = 8, etc. Or you store them as strings
                    # Here, we assume spc_id is an integer you can map to an element
                    # If you store them as strings, do something like:
                    #   element_symbol = sample.species_id_map[spc_id]
                    element_symbol = sample.get_symbol_from_id(spc_id)  # or similar

                    # Build f0_params
                    if element_symbol in db_dict_f0_all:
                        f0_local[i,:] = db_dict_f0_all[element_symbol]
                    # Build anomalous
                    if element_symbol in db_dict_f1f2_all:
                        cplx = self.get_f1f2_from_params(self._energy, db_dict_f1f2_all[element_symbol])
                        anom_local[i] = cplx

                f0_params_gpu = cp.asarray(f0_local)
                anom_gpu      = cp.asarray(anom_local)

                # Now call build_interaction_kernel with sub_sz = number of paths
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(sub_sz),
                        sub_kx, sub_ky, sub_kz,
                        sub_x,  sub_y,  sub_z,
                        sub_amp,
                        anom_gpu,
                        f0_params_gpu,
                        pxg, pyg, pzg,
                        dfield_gpu,
                        np.int32(Nx),
                        np.int32(Ny)
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()

            # Process each chunk in chunk_list
            for cidx in chunk_list:
                spc_host = sample.load_chunk_species(cidx, use_gpu=False)
                nA = spc_host.shape[0]
                if nA == 0:
                    continue

                # For bounce=0, we do one pass: wavevector=beam, amplitude=1
                # Build per-atom arrays
                pos_gpu = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                pos_gpu = pos_gpu @ R_stage_gpu
                pos_gpu += T_stage_gpu

                px_at = pos_gpu[:,0]/1e10
                py_at = pos_gpu[:,1]/1e10
                pz_at = pos_gpu[:,2]/1e10

                # chunk-level species => only for bounce=0
                # We'll do the same approach: build f0, anom
                f0_params_host = np.zeros((nA,11), dtype=np.float32)
                anom_host      = np.zeros(nA, dtype=np.complex64)

                unique_els = pd.unique(spc_host)
                for el in unique_els:
                    mask = (spc_host == el)
                    if el in db_dict_f0_all:
                        f0_params_host[mask] = db_dict_f0_all[el]
                    if el in db_dict_f1f2_all:
                        anom_host[mask] = self.get_f1f2_from_params(self._energy, db_dict_f1f2_all[el])

                f0_params_gpu = cp.asarray(f0_params_host)
                anom_gpu      = cp.asarray(anom_host)

                # initial wavevector = beam
                kx_atom_gpu = cp.full((nA,), self._kx_scalar, dtype=cp.float32)
                ky_atom_gpu = cp.full((nA,), self._ky_scalar, dtype=cp.float32)
                kz_atom_gpu = cp.full((nA,), self._kz_scalar, dtype=cp.float32)
                amp_atom_gpu= cp.ones((nA,), dtype=cp.complex64)

                # Direct scattering => accumulate to dfield_gpu
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(nA),
                        kx_atom_gpu, ky_atom_gpu, kz_atom_gpu,
                        px_at, py_at, pz_at,
                        amp_atom_gpu,
                        anom_gpu,
                        f0_params_gpu,
                        pxg, pyg, pzg,
                        dfield_gpu,
                        np.int32(Nx),
                        np.int32(Ny)
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()

                # If no further bounces, continue
                if n_bounces < 1:
                    cp.get_default_memory_pool().free_all_blocks()
                    continue

                # ============== For bounce >= 1 ==============
                # Load nearest-neighbor info
                ph_flat,  offs_ph = sample.load_chunk_nn_phase(cidx)
                kx_flat, ky_flat, kz_flat, offs_kv = sample.load_chunk_nn_scatter(cidx)
                idx_flat, offs_ix = sample.load_chunk_nn_indices(cidx)
                spc_flat, offs_sp = sample.load_chunk_nn_species(cidx)  # <--- neighbor species

                neighborPhase_gpu = cp.asarray(ph_flat,  dtype=cp.float32)
                neighborKx_gpu    = cp.asarray(kx_flat,  dtype=cp.float32)
                neighborKy_gpu    = cp.asarray(ky_flat,  dtype=cp.float32)
                neighborKz_gpu    = cp.asarray(kz_flat,  dtype=cp.float32)
                neighborIdx_gpu   = cp.asarray(idx_flat, dtype=cp.int32)
                neighborSpc_gpu   = cp.asarray(spc_flat, dtype=cp.int32)

                neighborStart_host = offs_ph[:-1].astype(np.int32)
                neighborCount_host = (offs_ph[1:] - offs_ph[:-1]).astype(np.int32)
                neighborStart_gpu  = cp.asarray(neighborStart_host)
                neighborCount_gpu  = cp.asarray(neighborCount_host)

                # Prepare the "in_" arrays for bounce=1 expansions
                cur_size   = nA
                in_x_gpu   = px_at.copy()
                in_y_gpu   = py_at.copy()
                in_z_gpu   = pz_at.copy()
                in_kx_gpu  = kx_atom_gpu.copy()
                in_ky_gpu  = ky_atom_gpu.copy()
                in_kz_gpu  = kz_atom_gpu.copy()
                in_amp_gpu = amp_atom_gpu.copy()
                in_idx_gpu = cp.arange(nA, dtype=cp.int32)  # atom index

                # For expansions, we have an output buffer of size sub_chunk_size
                expand_max = sub_chunk_size

                for bounce_i in range(1, n_bounces+1):
                    # Create output arrays
                    out_x_gpu   = cp.zeros((expand_max,), dtype=cp.float32)
                    out_y_gpu   = cp.zeros((expand_max,), dtype=cp.float32)
                    out_z_gpu   = cp.zeros((expand_max,), dtype=cp.float32)
                    out_kx_gpu  = cp.zeros((expand_max,), dtype=cp.float32)
                    out_ky_gpu  = cp.zeros((expand_max,), dtype=cp.float32)
                    out_kz_gpu  = cp.zeros((expand_max,), dtype=cp.float32)
                    out_amp_gpu = cp.zeros((expand_max,), dtype=cp.complex64)
                    out_idx_gpu = cp.zeros((expand_max+1,), dtype=cp.int32)
                    out_spc_gpu = cp.zeros((expand_max,), dtype=cp.int32)

                    # The last slot of out_idx_gpu is used as a global counter
                    out_idx_gpu[expand_max] = 0

                    # Launch expand_paths_kernel (bounce expansion)
                    nBlocks = (cur_size + block1d - 1)//block1d
                    expand_kernel(
                        (nBlocks,), (block1d,),
                        (
                            in_x_gpu, in_y_gpu, in_z_gpu,
                            in_kx_gpu, in_ky_gpu, in_kz_gpu,
                            in_amp_gpu,
                            in_idx_gpu,

                            neighborStart_gpu,
                            neighborCount_gpu,
                            neighborPhase_gpu,
                            neighborKx_gpu,
                            neighborKy_gpu,
                            neighborKz_gpu,
                            neighborIdx_gpu,
                            neighborSpc_gpu,   # new species array

                            np.int32(cur_size),
                            k_mag,
                            wavelength_angs,

                            out_x_gpu,
                            out_y_gpu,
                            out_z_gpu,
                            out_kx_gpu,
                            out_ky_gpu,
                            out_kz_gpu,
                            out_amp_gpu,
                            out_idx_gpu,
                            out_spc_gpu,  # store species
                            np.int32(expand_max)
                        )
                    )
                    cp.cuda.stream.get_current_stream().synchronize()

                    expansions_written = int(out_idx_gpu[expand_max].get())
                    if expansions_written == 0:
                        # no expansions => done
                        break

                    # Now sub-chunk these expansions for the final scattering pass
                    batchSize = sub_chunk_size
                    nSubBatches = (expansions_written + batchSize - 1)//batchSize
                    for sb in range(nSubBatches):
                        sbStart = sb*batchSize
                        sbEnd   = min(sbStart+batchSize, expansions_written)
                        sub_sz  = sbEnd - sbStart
                        if sub_sz <= 0:
                            continue

                        process_subchunk(sbStart, sbEnd,
                                        out_x_gpu, out_y_gpu, out_z_gpu,
                                        out_kx_gpu, out_ky_gpu, out_kz_gpu,
                                        out_amp_gpu, out_spc_gpu,
                                        sub_sz)

                    # Prepare for next bounce, if any
                    if bounce_i < n_bounces:
                        # The expansions become "in_" arrays for the next iteration
                        in_x_gpu   = out_x_gpu
                        in_y_gpu   = out_y_gpu
                        in_z_gpu   = out_z_gpu
                        in_kx_gpu  = out_kx_gpu
                        in_ky_gpu  = out_ky_gpu
                        in_kz_gpu  = out_kz_gpu
                        in_amp_gpu = out_amp_gpu
                        in_idx_gpu = out_idx_gpu
                        cur_size   = expansions_written

                cp.get_default_memory_pool().free_all_blocks()

            # Done with all chunks for this GPU
            partial_results[out_idx] = dfield_gpu.reshape((Ny, Nx)).get()

            # Cleanup
            del pxg, pyg, pzg, dfield_gpu
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Launch one thread per GPU
        threads = []
        start_chunk = 1
        for gpu_id in range(n_gpus):
            my_count = chunk_per_gpu + (1 if gpu_id < remainder else 0)
            end_chunk = start_chunk + my_count
            chunk_ids = range(start_chunk, end_chunk)
            start_chunk = end_chunk
            t = threading.Thread(target=gpu_worker, args=(gpu_id, chunk_ids, gpu_id))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Sum partial results from all GPUs
        for part in partial_results:
            if part is not None:
                final_result += part

        if offset is not None:
            final_result -= offset

        return final_result
    # -------------------------------------
    
    # -------------------------------------
    # Atomic master
    def atomic_direct_interaction(self, sample, detector, stage, scattering=True, scattering_params=[None], transmission=True, transmission_params=[0.0,0.0], use_gpu=True):
        """
        High-level method to compute both scattering and/or transmission from
        an atomic sample onto a detector.

        Args:
            sample: The sample object containing chunked atomic data.
            detector: The detector object with methods like `pixel_coordinates`,
                `shape`, and `input_pixel_values(...)`.
            stage: A stage object specifying rotation and translation transforms.
            scattering (bool, optional): If True, compute the scattering field.
            scattering_params (list, optional): A list that may include an offset
                value [offset, ...]. Defaults to [None].
            transmission (bool, optional): If True, compute the atomic transmission.
            transmission_params (list, optional): [atomic_radius, kernel_radius].
                Defaults to [0.0, 0.0].
            use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.

        Returns:
            None
        """
        Nx, Ny = detector.shape
        final_field = np.zeros((Ny,Nx)).astype(np.complex128)
        
        # Check if we can run GPU
        if use_gpu and (cp is not None):
            # Attempt GPU path
            if scattering is True:
                final_field += self.atomic_scattering_kinematic(sample, detector, stage, offset=scattering_params[0], use_gpu=use_gpu)
            if transmission is True:
                final_field += self.atomic_transmission(sample, detector, stage, use_gpu=use_gpu, atomic_radius=transmission_params[0], kernel_radius=transmission_params[0])
        else:
            # CPU fallback
            if cp is None and use_gpu:
                print("[beam] Cupy not installed, running CPU mode.")
            if scattering is True:
                final_field += self.atomic_scattering_kinematic(sample, detector, stage, offset=scattering_params[0], use_gpu=use_gpu)
            if transmission is True:
                final_field += self.atomic_transmission(sample, detector, stage, use_gpu=use_gpu, atomic_radius=transmission_params[0], kernel_radius=transmission_params[0])

        detector.input_pixel_values(final_field)
    # -------------------------------------
    
    # -------------------------------------
    # Wavefield propagation
    def wavefield_propagate(self, detector, optics):
        """
        Propagate the beam through a freespace/optical stack.
        """
    # -------------------------------------
    
