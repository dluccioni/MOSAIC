# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import cupy as cp
import pandas as pd
import pickle
import os
import sys
import gc
import threading
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '\databases')

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

    def create_beam(self, energy, eV=True, direction=np.array([1.0, 0.0, 0.0])):
        """
        Create a beam of specified energy and direction.
        energy is in eV by default (set eV=False if it's in Joules).
        """
        self._direction = direction
        if not eV:
            energy = energy / self._q
        self._energy = energy
        self._wavelength = self._hq * self._c / self._energy

    ## Data Handling Functions    
    def write_beam_metadata(self):
        """
        (Placeholder) Writes beam metadata to disk.
        """
        beam_metadata = [self._energy]

    ## Static Functions
    @staticmethod
    def build_interaction_kernel():
        """
        Returns the precompiled CuPy RawKernel object for computing scattering
        with a shared-memory approach.
        """
        # Includes both fastmath and shared memory tiling over standard (fastest)
        _cuda_source_memtile = r'''
        #define CHUNK_SIZE 128
        extern "C" {
        __device__ __forceinline__ float2 get_f0_from_params(float Q_val, const float* params)
        {
            // params layout: [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]
            // f0(Q) = c + sum_{i=1..5}( a_i * exp(-b_i*(k^2)) )
            // k = 0.25f * Q_val * 1.0e-10f / pi

            const float PI_F = 3.14159265358979323846f;
            float k   = 0.25f * Q_val * 1.0e-10f / PI_F;
            float kk  = k * k;
            float f0  = params[5]; // "c"

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
                            // If the pixel is exactly at the atom's location, skip
                            continue;
                        }

                        float rdx = dx / r_det;
                        // Q_val = k*sqrt(2*(1 - rdx))
                        float Q_val = k * __fsqrt_rn(2.0f * (1.0f - rdx));

                        // Evaluate f0
                        const float* param_ptr = &s_params[j * 11];
                        float2 f0c = get_f0_from_params(Q_val, param_ptr);

                        // Add anomalous
                        float2 s_a   = s_anom[j];
                        float2 s_tot = make_float2(f0c.x + s_a.x, f0c.y + s_a.y);

                        // Phase
                        float phase = k * (s_px[j] + r_det);

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
        mod = cp.RawModule(
            code=_cuda_source_memtile,
            backend='nvcc',
            options=('--gpu-architecture=sm_89','-O3','--ftz=true','--fmad=true')
        )
        kernel = mod.get_function('interaction_kernal')
        return kernel
    
    @staticmethod
    def parse_f0_db_all(database_name='f0_WaasKirf.dat'):
        """
        Loads the entire f0 database for all elements in the file.
        Returns a dict: { "H": [a1,a2,a3,a4,a5,c,b1,b2,b3,b4,b5], ... }
        """
        import scattering
        import importlib.resources as pkg_resources
        db_dict = {}
        db_file = pkg_resources.open_text(scattering, database_name)
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
        import scattering
        import importlib.resources as pkg_resources
        f1f2_dict = {}
        db_file = pkg_resources.open_text(scattering, database_name)
        element = None
        param_list = []
        for line in db_file:
            if line.startswith('#S'):
                # if we already had an element
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

    ## Main Functions
    def atomic_direct_scattering(self, sample, detector, offset=0):
        """
        If multigpu=True, call the multi-GPU version; otherwise fallback to single-GPU.
        """
        measurement_positions = detector.pixel_coordinates
        Nx, Ny = detector.shape
        pixel_values = self.interact_beam(sample, measurement_positions, (Nx, Ny))
        detector.input_pixel_values(pixel_values - offset)
    
    def interact_beam(self, sample, measurement_positions, measurement_shape):
        """
        Perform beam-sample interaction using multiple GPUs in parallel,
        assigning one CPU thread per GPU. Each thread processes a subset
        of sample chunks and accumulates partial scattering results, which
        are then summed on the CPU.
        """
        # Number of GPUs available
        n_gpus = cp.cuda.runtime.getDeviceCount()
        print(f"Found {n_gpus} GPU(s).")
        # Prepare constants in CPU memory (will be copied to each GPU thread)
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        Nx, Ny = measurement_shape
        k_val = np.float32(2.0 * np.pi / self._wavelength)
        # Convert measurement_positions to float32 CPU arrays
        x_coords_cpu = measurement_positions[0, :].astype(np.float32)
        y_coords_cpu = measurement_positions[1, :].astype(np.float32)
        z_coords_cpu = measurement_positions[2, :].astype(np.float32)
        chunk_total = sample.chunk_total
        print(f"Total of {chunk_total} chunk(s) to process.")
        # We will divide the chunk indices among the GPUs
        # (contiguous split in this example)
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus
        # A place to store each GPU's partial result (on CPU)
        partial_results = [None] * n_gpus

        def gpu_worker(gpu_id, chunk_indices, result_index):
            """
            Runs on a single GPU (gpu_id), processes the given chunk_indices,
            and stores the final partial detector array in partial_results[result_index].
            """
            cp.cuda.Device(gpu_id).use()  # Select GPU
            # Build the kernel inside this thread (optional; can also build globally)
            interaction_kernel = self.build_interaction_kernel()
            # Allocate GPU arrays for coordinates & output
            x_coords_gpu = cp.asarray(x_coords_cpu, dtype=cp.float32)
            y_coords_gpu = cp.asarray(y_coords_cpu, dtype=cp.float32)
            z_coords_gpu = cp.asarray(z_coords_cpu, dtype=cp.float32)
            detector_field_gpu = cp.zeros((Nx * Ny,), dtype=cp.complex64)
            # We can create a few streams to overlap chunk processing on this GPU
            num_streams = 4
            streams = [cp.cuda.Stream() for _ in range(num_streams)]
            block_size = (16, 16)
            grid_size = ((Nx + block_size[0] - 1) // block_size[0],
                         (Ny + block_size[1] - 1) // block_size[1])
            # Process assigned chunks on this GPU
            for i, chunk_id in enumerate(chunk_indices):
                stream = streams[i % num_streams]
                # Load species (CPU)
                species_chunk_np = sample.load_chunk_species(chunk_id, gpu=False)
                atom_count = species_chunk_np.shape[0]
                if atom_count == 0:
                    continue
                # Prepare arrays on CPU
                scattering_anom_np = np.zeros(atom_count, dtype=np.complex64)
                f0_params_np = np.zeros((atom_count, 11), dtype=np.float32)
                unique_elements = pd.unique(species_chunk_np)
                for el in unique_elements:
                    if el not in db_dict_f0_all:
                        continue  # skip missing
                    mask = (species_chunk_np == el)
                    # f1,f2
                    table = db_dict_f1f2_all.get(el, None)
                    if table is not None:
                        scattering_anom_np[mask] = self.get_f1f2_from_params(self._energy, table)
                    # f0
                    f0_params_np[mask] = db_dict_f0_all[el]
                # Transfer positions & scattering arrays to GPU
                with stream:
                    # positions_chunk can be loaded directly in GPU format
                    positions_chunk_cp = cp.array(sample.load_chunk_positions(chunk_id, gpu=True),dtype=cp.float32)
                    px = positions_chunk_cp[:, 0] / 1e10
                    py = positions_chunk_cp[:, 1] / 1e10
                    pz = positions_chunk_cp[:, 2] / 1e10
                    scattering_anom_cp = cp.asarray(scattering_anom_np)
                    f0_params_cp = cp.asarray(f0_params_np)
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
                            x_coords_gpu / 1e10,
                            y_coords_gpu / 1e10,
                            z_coords_gpu / 1e10,
                            detector_field_gpu,
                            np.int32(Nx),
                            np.int32(Ny),
                        ),
                        stream=stream
                    )
            # Wait for all chunks on this GPU to finish
            for s in streams:
                s.synchronize()
            # Bring partial result back to CPU
            partial_results[result_index] = detector_field_gpu.reshape((Ny, Nx)).get()
            # Cleanup GPU memory
            del x_coords_gpu, y_coords_gpu, z_coords_gpu
            del detector_field_gpu
            for s in streams:
                del s
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
        # Create and start threads
        threads = []
        start_chunk = 1
        for gpu_id in range(n_gpus):
            # Determine which chunks this GPU should handle
            my_count = chunks_per_gpu + (1 if gpu_id < remainder else 0)
            end_chunk = start_chunk + my_count
            chunk_indices = list(range(start_chunk, end_chunk))
            start_chunk = end_chunk
            t = threading.Thread(target=gpu_worker,args=(gpu_id, chunk_indices, gpu_id))
            t.start()
            threads.append(t)
        # Join threads
        for t in threads:
            t.join()
        # Sum partial results on CPU
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)
        for pr in partial_results:
            if pr is not None:
                final_result += pr
        return final_result

