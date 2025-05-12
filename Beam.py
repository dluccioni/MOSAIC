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

        Parameters
        ----------
        energy : float
            Beam energy in eV by default (use eV=False to interpret in Joules).
        direction : np.ndarray of shape (3,)
            Propagation direction of the beam (assume collimated).
        beam_shape : str
            "rectangular", "circular", etc.  (Currently only "rectangular" used in binning.)
        beam_size : tuple of float
            (size_y, size_z) in Angstroms for the cross-section.
        """
        self._direction = direction
        if not eV:
            energy = energy / self._q
        self._energy = energy
        self._wavelength = self._hq * self._c / self._energy
        self._beam_shape = beam_shape.lower()
        self._beam_size = beam_size
        
    def read_beam_metadata(self):
        """
        Reads the metadata JSON file from disk and restores
        this beam object's state.
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
        Serializes the beam object's critical internal fields to disk 
        as human-readable JSON so that the state can be restored later.
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
    @staticmethod
    def _make_orthonormal_basis(direction):
        """
        Given a 3D direction vector (beam direction),
        return two 3D vectors e1, e2 which are orthonormal
        to each other and to 'direction'.
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
    
    # CPU kernel
    @staticmethod
    def compile_compute_scattering_cffi():
        """
        Builds (or caches) a CFFI module that implements the scattering for CPU
        in C, providing a function compute_scattering_cffi(...) to do the loops.
        Returns (ffi_obj, C_mod).
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
            float ktmp  = 0.25f * Q_val * 1.0e-10f / PI_F;
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
    
    # GPU kernel
    @staticmethod
    def build_interaction_kernel():
        """
        Returns the precompiled CuPy RawKernel object for computing scattering
        with a shared-memory approach. Only invoked if `cp` is not None.
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
            float k   = 0.25f * Q_val * 1.0e-10f / PI_F;
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
            const float k,                  // wave number = 2*pi / wavelength
            const float* px,                // atom positions.x (length nAtoms)
            const float* py,                // atom positions.y
            const float* pz,                // atom positions.z
            const float2* scattering_anom,  // (f1 + i f2) for each atom
            const float* f0_params,         // shape = (nAtoms, 11)
            const float* x_coords,          // length Nx*Ny
            const float* y_coords,          // length Nx*Ny
            const float* z_coords,          // length Nx*Ny
            float2*     detector_field,     // shape Nx*Ny
            const int   Nx,
            const int   Ny
        )
        {
            const float PI_F = 3.14159265358979323846f;
            float wavelength_m = (2.0f * PI_F) / k;
            // Determine which pixel this thread processes
            int pxid = blockIdx.x * blockDim.x + threadIdx.x;
            int pyid = blockIdx.y * blockDim.y + threadIdx.y;

            // Check if we are in-bounds
            bool in_bounds = (pxid < Nx && pyid < Ny);
            int pixel_index = pyid * Nx + pxid;

            // Coordinates for this pixel (0 if out-of-bounds)
            float tx = 0.0f, ty = 0.0f, tz = 0.0f;
            if (in_bounds)
            {
                tx = x_coords[pixel_index];
                ty = y_coords[pixel_index];
                tz = z_coords[pixel_index];
            }

            // Accumulate the result in registers
            float2 sum_val = make_float2(0.0f, 0.0f);

            // Shared memory for tiled approach
            __shared__ float  s_px[CHUNK_SIZE];
            __shared__ float  s_py[CHUNK_SIZE];
            __shared__ float  s_pz[CHUNK_SIZE];
            __shared__ float2 s_anom[CHUNK_SIZE];
            __shared__ float  s_params[CHUNK_SIZE * 11];

            int threads_in_block = blockDim.x * blockDim.y;
            // Linear thread ID in the current block
            int t_id = threadIdx.y * blockDim.x + threadIdx.x;

            // Tiling over nAtoms
            for (int tile_start = 0; tile_start < nAtoms; tile_start += CHUNK_SIZE)
            {
                // 1) Load a tile of atoms into shared memory
                for (int t = t_id; t < CHUNK_SIZE; t += threads_in_block)
                {
                    int atom_idx = tile_start + t;
                    if (atom_idx < nAtoms)
                    {
                        s_px[t]   = px[atom_idx];
                        s_py[t]   = py[atom_idx];
                        s_pz[t]   = pz[atom_idx];
                        s_anom[t] = scattering_anom[atom_idx];
                        // Copy 11 f0_params
                        #pragma unroll
                        for (int pi = 0; pi < 11; pi++)
                        {
                            s_params[t * 11 + pi] = f0_params[atom_idx * 11 + pi];
                        }
                    }
                }
                __syncthreads();

                // 2) Only in-bounds threads do the summation
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
                        if (r_det == 0.0f) {
                            // If pixel is exactly at the atom's location, skip
                            continue;
                        }

                        float rdx = dx / r_det;
                        // Q_val = k * sqrt(2*(1 - rdx))
                        float Q_val = k * __fsqrt_rn(2.0f * (1.0f - rdx));

                        // Evaluate f0
                        const float* param_ptr = &s_params[j * 11];
                        float2 f0c = get_f0_from_params(Q_val, param_ptr);

                        // Add anomalous
                        float2 s_a   = s_anom[j];
                        float2 s_tot = make_float2(f0c.x + s_a.x, f0c.y + s_a.y);

                        // Phase
                        float px_mod    = fmodf(s_px[j], wavelength_m);
                        float rdet_mod  = fmodf(r_det,   wavelength_m);
                        float phase     = k * (px_mod + rdet_mod);
                        float cph, sph;
                        __sincosf(phase, &sph, &cph);

                        float2 val;
                        val.x = s_tot.x * cph - s_tot.y * sph;
                        val.y = s_tot.x * sph + s_tot.y * cph;

                        sum_val.x += val.x;
                        sum_val.y += val.y;
                    }
                }
                __syncthreads();  // Ensure all threads finish this tile
            }

            if (in_bounds)
            {
                // Accumulate to global detector field
                //atomicAdd(&detector_field[pixel_index].x, sum_val.x);
                //atomicAdd(&detector_field[pixel_index].y, sum_val.y);
                detector_field[pixel_index].x += sum_val.x;
                detector_field[pixel_index].y += sum_val.y;
            }
        }
        }
        '''
        # Build raw module
        kernel_module = cp.RawModule(
            code=_cuda_source_memtile,
            backend='nvcc',
            options=('--gpu-architecture=sm_89', '-O3', '--ftz=true', '--fmad=true')
        )
        return kernel_module.get_function('interaction_kernal')

    @staticmethod
    def parse_f0_db_all(database_name='f0_WaasKirf.dat'):
        """
        Loads the entire f0 database for all elements in the file.
        Returns a dict: { "H": [a1,a2,a3,a4,a5,c,b1,b2,b3,b4,b5], ... }
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
        Loads the entire f1f2 database for all elements.
        Returns a dict: { "H": array([[E1,f1_1,f2_1],[E2,f1_2,f2_2],...]), ... }
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
        Given an energy (float) and the full f1,f2 table for an element
        (shape [N,3], columns: E, f1, f2), returns (f1 + i*f2)
        by linear interpolation near the requested energy.
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
        Given the dictionary of Waasmaier-Kirfel params for each element,
        compute f0(0) = c + sum_{i=1..5}(a_i) for each element.
        Returns a dict {element_name: f0_zero_value}.
        """
        f0_0_dict = {}
        for el, params in db_dict_f0_all.items():
            # params layout = [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]
            a1, a2, a3, a4, a5, c = params[0], params[1], params[2], params[3], params[4], params[5]
            # f0(0) = c + sum(a1..a5)
            val = float(c + a1 + a2 + a3 + a4 + a5)
            f0_0_dict[el] = val
        return f0_0_dict

    ## Main Functions
    # CPU function
    def cpu_scatter_chunk_cffi(self, complied_code, ffi_obj, chunk_id, sample,
                               Nx, Ny, coords_x_m, coords_y_m, coords_z_m,
                               db_dict_f0_all, db_dict_f1f2_all, k_val,
                               stage):
        """
        Use CFFI for a single chunk scattering. Return (Ny, Nx) complex64 array.
        
        NOTE: We apply the stage rotation + translation to the positions
              before converting to meters.
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
        Multi-threaded CPU approach. Splits sample chunks among threads,
        accumulates partial fields. Uses CFFI for numeric loops,
        applying stage transformations to each chunk.
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
        
    def bin_atoms_in_pixels_cpu(self, sample, Nx, Ny, e1, e2,
                                pixel_size_u, pixel_size_v,
                                stage, atomic_radius=1.7, kernel_radius=0, detector=None):
        """
        CPU approach that accounts for a ~2 radius by shifting each atom 
        around +/-2 Ang. Then it deduplicates so each (atom, pixel) is unique.

        Now we get the *actual* pixel centers from 'detector.pixel_coordinates' 
        and still use 'pixel_size_u, pixel_size_v' for binning widths.

        Returns
        -------
        final_map : (Ny, Nx) complex64
            Real + imag sum in each pixel (after optional Gaussian).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing
        
        # 1) Retrieve pixel centers from detector
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

        # 2) Optionally build a 2D Gaussian kernel for smoothing
        def _make_gaussian_kernel_cpu(radius):
            if radius < 1:
                return None
            diam = 2*radius + 1
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

        # 3) Pre‐load scattering DB
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

            w_real = scattering_real[atom_ids_unique]
            partial_hist_real = np.bincount(bin_idx_unique.astype(np.int64),
                                            weights=w_real, minlength=Nx*Ny)
            w_imag = scattering_imag[atom_ids_unique]
            partial_hist_imag = np.bincount(bin_idx_unique.astype(np.int64),
                                            weights=w_imag, minlength=Nx*Ny)

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

        # Convolve if needed
        kernel = _make_gaussian_kernel_cpu(kernel_radius)
        conv_real = _fft_convolve2d_cpu(final_map_real, kernel)
        conv_imag = _fft_convolve2d_cpu(final_map_imag, kernel)

        return (conv_real + 1j*conv_imag).astype(np.complex64)

    def interact_beam_gpu(self, sample, measurement_positions, measurement_shape, stage):
        """
        Perform beam-sample interaction using all available GPUs in parallel,
        chunk-based. Each GPU gets a subset of chunks. Summation on CPU.
        
        We also apply the stage rotation+translation on GPU before dividing by 1e10.
        """
        if cp is None:
            # If somehow called without cupy installed, fallback to CPU
            print("[beam] Cupy not installed, falling back to CPU mode.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            print("[beam] No GPUs found, falling back to CPU mode.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage)

        print(f"[beam] Found {n_gpus} GPU(s).")
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        Nx, Ny = measurement_shape
        k_val = np.float32(2.0 * np.pi / self._wavelength)

        if isinstance(measurement_positions, np.ndarray):
            measurement_positions = cp.asarray(measurement_positions, dtype=cp.float32)
        x_coords_gpu = cp.ascontiguousarray(measurement_positions[0, :].astype(cp.float32) / 1e10)
        y_coords_gpu = cp.ascontiguousarray(measurement_positions[1, :].astype(cp.float32) / 1e10)
        z_coords_gpu = cp.ascontiguousarray(measurement_positions[2, :].astype(cp.float32) / 1e10)
        
        chunk_total = sample.chunk_total
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")

        # Create Stage variables 
        R_stage_gpu = cp.asarray(stage.rotation, dtype=cp.float32)
        trans_stage_gpu = cp.asarray(stage.translation, dtype=cp.float32)

        # Divide chunk indices among GPUs
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus

        # A place to store partial results
        partial_results = [None] * n_gpus

        def gpu_worker(gpu_id, x_coords_gpu, y_coords_gpu, z_coords_gpu, chunk_indices, result_index):
            """One thread per GPU."""
            cp.cuda.Device(gpu_id).use()
            interaction_kernel = self.build_interaction_kernel()

            detector_field_gpu = cp.zeros((Nx * Ny,), dtype=cp.complex64)

            # Use streams for concurrency on that GPU
            num_streams = 4
            streams = [cp.cuda.Stream() for _ in range(num_streams)]

            block_size = (16, 16)
            grid_size = ((Nx + block_size[0] - 1) // block_size[0],
                         (Ny + block_size[1] - 1) // block_size[1])

            for i, cidx in enumerate(chunk_indices):
                stream = streams[i % num_streams]

                # Load species (CPU)
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

                    # Now convert to meters
                    px = positions_chunk_cp[:, 0] / 1e10
                    py = positions_chunk_cp[:, 1] / 1e10
                    pz = positions_chunk_cp[:, 2] / 1e10

                    scattering_anom_cp = cp.asarray(scattering_anom_np)
                    f0_params_cp       = cp.asarray(f0_params_np)

                    interaction_kernel(
                        grid_size,
                        block_size,
                        (
                            np.int32(atom_count),
                            k_val,
                            px,
                            py,
                            pz,
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
                
                if ((i % 16 == 0) and (i != 0)) or (i == (len(chunk_indices)-1)):
                    stream.synchronize()
                    cp.get_default_memory_pool().free_all_blocks()

            # Sync streams
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
                                       x_coords_gpu, y_coords_gpu, z_coords_gpu,
                                       chunk_indices, gpu_id))
            t.start()
            threads.append(t)

        # Join
        for t in threads:
            t.join()

        # Sum partial results
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)
        for pr in partial_results:
            if pr is not None:
                final_result += pr

        return final_result
    
    def bin_atoms_in_pixels_gpu(self, sample, Nx, Ny, e1, e2,
                                pixel_size_u, pixel_size_v,
                                stage, atomic_radius=1.7, kernel_radius=0, detector=None):
        """
        GPU approach that accounts for offsets, deduplicates, and
        then does 2D Gaussian convolution if requested.

        We now read the pixel centers from 'detector.pixel_coordinates'
        on the GPU to define the bounding box for binning.
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

        # 1) Project pixel centers to (u,v) on GPU
        pix_coords = detector.pixel_coordinates
        if not isinstance(pix_coords, cp.ndarray):
            pix_coords = cp.asarray(pix_coords, dtype=cp.float32)
        e1_gpu = cp.asarray(e1, dtype=cp.float32)
        e2_gpu = cp.asarray(e2, dtype=cp.float32)

        px = pix_coords[0]
        py = pix_coords[1]
        pz = pix_coords[2]
        pixel_u = px*e1_gpu[0] + py*e1_gpu[1] + pz*e1_gpu[2]
        pixel_v = px*e2_gpu[0] + py*e2_gpu[1] + pz*e2_gpu[2]
        min_u_gpu = pixel_u.min()
        min_v_gpu = pixel_v.min()

        # 2) Build GPU Gaussian kernel if needed
        def _make_gaussian_kernel_gpu(radius):
            if radius < 1:
                return None
            diam = 2*radius + 1
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

        # stage transforms on GPU
        R_stage_gpu     = cp.asarray(stage.rotation, dtype=cp.float32)
        trans_stage_gpu = cp.asarray(stage.translation, dtype=cp.float32)

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
        offsets_gpu = cp.asarray(offsets_cpu)
        oy_gpu = offsets_gpu[:,0]
        oz_gpu = offsets_gpu[:,1]

        partial_results = [None] * n_gpus
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus

        def gpu_worker(gpu_id, chunk_indices, out_idx):
            cp.cuda.Device(gpu_id).use()
            out_r = cp.zeros((Ny, Nx), dtype=cp.float32)
            out_i = cp.zeros((Ny, Nx), dtype=cp.float32)

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

                # Move to GPU
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

                hist_r = cp.bincount(bin_uniq, weights=w_real, minlength=Nx*Ny)
                hist_i = cp.bincount(bin_uniq, weights=w_imag, minlength=Nx*Ny)
                out_r += hist_r.reshape(Ny, Nx)
                out_i += hist_i.reshape(Ny, Nx)

                # cleanup
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

        # Sum partial on device 0
        cp.cuda.Device(0).use()
        final_r_gpu = cp.zeros((Ny, Nx), dtype=cp.float32)
        final_i_gpu = cp.zeros((Ny, Nx), dtype=cp.float32)
        for pr in partial_results:
            if pr is not None:
                r_gpu, i_gpu = pr
                final_r_gpu += r_gpu
                final_i_gpu += i_gpu

        # Convolve if needed
        kernel_gpu = _make_gaussian_kernel_gpu(kernel_radius)
        final_r_gpu = _fft_convolve2d_gpu(final_r_gpu, kernel_gpu)
        final_i_gpu = _fft_convolve2d_gpu(final_i_gpu, kernel_gpu)

        out_cpu = final_r_gpu.get() + 1j*final_i_gpu.get()
        return out_cpu.astype(np.complex64)
    
    def count_atoms_in_pixels(self, sample, detector, stage, use_gpu=True, atomic_radius=1.7, kernel_radius=0):
        """
        Bins atoms into each detector pixel, weighting each atom by
        f0(0) + f1 + i*f2. Then applies a 2D circular convolution (if kernel_radius>0)
        to the real and imaginary parts.

        Returns
        -------
        f0_complex_map : np.ndarray of shape (Ny, Nx), dtype=complex64
            The convolved complex sum in each pixel.
        """
        Ny, Nx = detector.shape
        pixel_size_u, pixel_size_v = detector.pixel_size
        size_y = pixel_size_u * Nx
        size_z = pixel_size_v * Ny # in Angstrom
        
        r_e = 2.81794e-5 # in Angstrom
        prefactor = 1/r_e**2
        
        # Build orthonormal basis
        e1, e2 = self._make_orthonormal_basis(self._direction)

        if use_gpu and (cp is not None):
            f0_map_complex = self.bin_atoms_in_pixels_gpu(
                sample, Nx, Ny, e1, e2,
                pixel_size_u, pixel_size_v,
                stage, atomic_radius=atomic_radius, 
                kernel_radius=kernel_radius,
                detector=detector
            )
        else:
            f0_map_complex = self.bin_atoms_in_pixels_cpu(
                sample, Nx, Ny, e1, e2,
                pixel_size_u, pixel_size_v,
                stage, atomic_radius=atomic_radius,
                kernel_radius=kernel_radius,
                detector=detector
            )
            
        # final_map = -f0_map_complex + f0_map_complex.flat[np.abs(f0_map_complex).argmax()]
        # print(f0_map_complex.flat[np.abs(f0_map_complex).argmax()])
        final_map = -f0_map_complex.real + np.max(f0_map_complex.real)
        final_map = prefactor*final_map/(size_y*size_z)
        return final_map.T

    # Main API
    def atomic_direct_scattering(self, sample, detector, stage, offset=None, transmission=False, atomic_radius=0, kernel_radius=0, use_gpu=True):
        """
        High-level entry point for beam-sample scattering.
        Now includes a 'stage' argument to apply its rotation+translation.
        
        Parameters
        ----------
        sample : object with 'chunk_total', 'load_chunk_species(i)', 'load_chunk_positions(i)'
        detector : object with 'pixel_coordinates' (3, Nx*Ny), 'shape' -> (Nx, Ny),
                   and 'input_pixel_values(...)'
        stage : stage object that provides rotation angles and translation
        offset : float
            Offset subtracted from final result
        use_gpu : bool
            If True and cupy installed with GPU(s), use GPU. Else use CPU.
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
        if transmission is True:
            final_field += self.count_atoms_in_pixels(sample, detector, stage, use_gpu=use_gpu, atomic_radius=atomic_radius, kernel_radius=kernel_radius)
            
        if offset is not None:
            detector.input_pixel_values(final_field - offset)
        else:
            detector.input_pixel_values(final_field)
            
    def field_propagate(self, detector, optics):
        """
        Propagate the beam through a freespace/optical stack
        """
