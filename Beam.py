# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import cupy as cp
import pandas as pd
import pickle
import os
import gc
import threading
import sys
sys.path.insert(1, 'X://Dresselhaus Lab/Code/Phase Retreival/Wave_Optics/waveoptics_fwrd_sim/')

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class beam:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=os.getcwd()):
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
        self._hq = self._h/self._q
        
    def create_beam(self,energy,eV=True,direction=np.array([1.0,0.0,0.0])):
        """
        Create a beam of specified energy and direction.
        energy is in eV by default (set eV=False if it's in Joules).
        """
        self._direction = direction
        if not eV:
            energy = energy/self._q
        self._energy = energy
        self._wavelength = self._hq*self._c/self._energy

    ## Data Handling Functions
    def write_beam_metadata(self):
        """
        (Placeholder) Writes beam metadata to disk.
        """
        beam_metadata = [self._energy]
        # This can be fleshed out if needed.

    ## Static Functions
    @staticmethod
    def build_interaction_kernel():
        """
        Returns the pre-compiled CuPy RawKernel object for computing scattering
        contributions on the GPU. Uses a shared-memory approach and atomicAdd
        for final accumulation into detector_field.
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
                // Uses fast exp intrinsic if compiled with --use_fast_math
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

                        float rdx   = dx / r_det;
                        // Q_val = k * sqrt(2 * (1 - rdx))
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

            // 3) Write out the result with atomicAdd
            if (in_bounds)
            {
                // Can replace this with atomicAdd if there are any thread safety issues e.g.
                // atomicAdd(&detector_field[pixel_index].x, sum_val.x);
                // atomicAdd(&detector_field[pixel_index].y, sum_val.y);
                detector_field[pixel_index].x += sum_val.x;
                detector_field[pixel_index].y += sum_val.y;
            }
        }
        }
        '''
        # Includes fastmath changes over standard (intermediat)
        _cuda_source_fastmath = r'''
        extern "C" {

            __device__ __forceinline__ float2 get_f0_from_params(const float Q_val, const float* params)
            {
                // params layout: [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]
                // f0(Q) = c + sum_{i=1..5} ( a_i * exp(-b_i * (k^2)) )
                // where k = 0.25f * Q_val * 1.0e-10f / pi

                const float PI_F = 3.14159265358979323846f;
                float k   = 0.25f * Q_val * 1.0e-10f / PI_F;
                float kk  = k * k;

                // The constant term "c"
                float f0 = params[5];

                // Sum over the 5 exponent terms
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
                const float k,            // wave number = 2*pi / wavelength
                const float* px,          // atom positions.x (length nAtoms)
                const float* py,          // atom positions.y
                const float* pz,          // atom positions.z
                const float2* scattering_anom, // precomputed (f1 + i f2) for each atom
                const float* f0_params,   // shape = (nAtoms, 11)
                const float* x_coords,    // length Nx*Ny
                const float* y_coords,    // length Nx*Ny
                const float* z_coords,    // length Nx*Ny
                float2*     detector_field,  // shape Nx*Ny
                const int   Nx,
                const int   Ny
            )
            {
                // 1) Identify this thread's pixel index
                int ix = blockDim.x * blockIdx.x + threadIdx.x;
                int iy = blockDim.y * blockIdx.y + threadIdx.y;
                if (ix >= Nx || iy >= Ny) return;  // out-of-bounds pixel

                int pixel_index = iy * Nx + ix;

                // 2) Read pixel coordinates
                float tx = x_coords[pixel_index];
                float ty = y_coords[pixel_index];
                float tz = z_coords[pixel_index];

                // 3) We'll accumulate real/imag parts locally, then atomicAdd at the end
                float2 sum_val = make_float2(0.0f, 0.0f);

                // 4) Loop over all atoms
                for (int i = 0; i < nAtoms; i++)
                {
                    // Position difference
                    float dx = tx - px[i];
                    float dy = ty - py[i];
                    float dz = tz - pz[i];

                    float r_det = sqrtf(dx*dx + dy*dy + dz*dz);
                    float rdx   = dx / r_det;

                    // Q_val = k * sqrt(2 * (1 - rdx))
                    float Q_val = k * __fsqrt_rn(2.0f * (1.0f - rdx));

                    // f0 from parameters
                    const float* p  = &f0_params[i * 11];
                    float2 f0c      = get_f0_from_params(Q_val, p);

                    // Add anomalous scattering
                    float2 s_anom   = scattering_anom[i];
                    float2 s_tot    = make_float2(f0c.x + s_anom.x, f0c.y + s_anom.y);

                    // Phase = k * (px[i] + r_det)
                    float phase = k * (px[i] + r_det);

                    // Use fast intrinsics for sin/cos
                    float cph, sph;
                    __sincosf(phase, &sph, &cph);

                    // Multiply s_tot by e^(i*phase)
                    float2 val;
                    val.x = s_tot.x * cph - s_tot.y * sph;
                    val.y = s_tot.x * sph + s_tot.y * cph;

                    sum_val.x += val.x;
                    sum_val.y += val.y;
                }

                // 5) Write out to global memory with atomicAdd
                atomicAdd(&detector_field[pixel_index].x, sum_val.x);
                atomicAdd(&detector_field[pixel_index].y, sum_val.y);
            }

        }  // extern "C"
        '''
        # Standard working CUDA code (slowest)
        _cuda_source = r'''
        extern "C" {
            __device__ inline float2 get_f0_from_params(const float Q_val, const float* params)
            {
                // params = [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]
                // f0(Q) = c + sum_{i=1..5} ( a_i * exp(-b_i * k^2) )
                // k = 0.25 * Q_val * 1e-10f / pi
                float k = 0.25f * Q_val * 1e-10f / 3.14159265358979323846f;

                float f0 = params[5]; // c
                #pragma unroll
                for (int i = 0; i < 5; i++)
                {
                    float ai = params[i];
                    float bi = params[6 + i];
                    f0 += ai * expf(-bi * (k * k));
                }
                // Return real = f0, imaginary = 0
                return make_float2(f0, 0.0f);
            }

            __global__ void interaction_kernal(
                const int   nAtoms,
                const float k,               // wave number = 2*pi / wavelength
                const float* px,             // atom positions.x (length nAtoms)
                const float* py,             // atom positions.y
                const float* pz,             // atom positions.z
                const float2* scattering_anom, // precomputed (f1 + i f2) for each atom
                const float* f0_params,      // shape = (nAtoms, 11)
                const float* x_coords,       // length Nx*Ny
                const float* y_coords,       // length Nx*Ny
                const float* z_coords,       // length Nx*Ny
                float2*     detector_field,  // shape Nx*Ny
                const int   Nx,
                const int   Ny
            )
            {
                // Hard-coded beam_in_dir = (1, 0, 0)
                // so Q = k_out - k_in = k * (r_det_hat - (1, 0, 0))

                int ix = blockDim.x * blockIdx.x + threadIdx.x;
                int iy = blockDim.y * blockIdx.y + threadIdx.y;
                if (ix >= Nx || iy >= Ny) return;

                int pixel_index = iy * Nx + ix;

                // Coordinates of this pixel
                float tx = x_coords[pixel_index];
                float ty = y_coords[pixel_index];
                float tz = z_coords[pixel_index];

                // Accumulate real/imag part
                float2 sum_val = make_float2(0.0f, 0.0f);

                // Sum over all atoms
                for (int i = 0; i < nAtoms; i++)
                {
                    // Position difference to detector pixel
                    float dx = tx - px[i];
                    float dy = ty - py[i];
                    float dz = tz - pz[i];
                    float r_det = sqrtf(dx*dx + dy*dy + dz*dz);

                    // r_det_hat
                    float rdx = dx / r_det;  // x component of unit vector
                    float rdy = dy / r_det;  // y component
                    float rdz = dz / r_det;  // z component

                    // Q_val = |(r_det_hat - (1,0,0))| * k
                    float diffx = rdx - 1.0f;
                    float diffy = rdy;
                    float diffz = rdz;
                    float Q_val = sqrtf(diffx*diffx + diffy*diffy + diffz*diffz) * k;

                    // f0 from params
                    const float* p = &f0_params[i * 11];
                    float2 f0c = get_f0_from_params(Q_val, p);

                    // Add anomalous
                    float2 s_anom = scattering_anom[i];
                    float2 s_tot  = make_float2(f0c.x + s_anom.x, f0c.y + s_anom.y);

                    // Phase factor: for beam along +x, replace pz[i] with px[i].
                    // i.e. exp(i * k*(px[i] + r_det))
                    float phase  = k * (px[i] + r_det);
                    float cph    = cosf(phase);
                    float sph    = sinf(phase);

                    // Multiply s_tot * e^(i*phase)
                    float2 val;
                    val.x = s_tot.x * cph - s_tot.y * sph;
                    val.y = s_tot.x * sph + s_tot.y * cph;

                    // Accumulate
                    sum_val.x += val.x;
                    sum_val.y += val.y;
                }

                // Add contribution to global detector_field
                atomicAdd(&detector_field[pixel_index].x, sum_val.x);
                atomicAdd(&detector_field[pixel_index].y, sum_val.y);
            }
        }
        '''
        # Compile via CuPy
        mod = cp.RawModule(code=_cuda_source_memtile,backend='nvcc',options=('--gpu-architecture=sm_89','-O3'))
        # mod = cp.RawModule(code=_cuda_source,options=('--std=c++11',)) #backend='nvrtc'
        kernel = mod.get_function('interaction_kernal')
        return kernel
    
    @staticmethod
    def parse_f0_db_all(database_name='f0_WaasKirf.dat'):
        """
        Loads the entire f0 database for all elements in the file.
        Returns a dict: { "H": [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5], ... }
        """
        import databases
        import importlib.resources as pkg_resources
        db_dict = {}
        db_file = pkg_resources.open_text(databases, database_name)
        element = None
        for line in db_file:
            if line.startswith('#S'):
                element = line.split()[2].strip()
            elif not line.startswith('#') and element is not None:
                params = np.fromiter((float(x) for x in line.split()), dtype=np.float32)
                if params.size == 11:
                    db_dict[element] = params
        return db_dict

    @staticmethod
    def parse_f1f2_db_all(database_name='f1f2_CromerLiberman.dat'):
        """
        Loads the entire f1f2 database for all elements.
        Returns a dict: { "H": array([[E1, f1_1, f2_1],[E2, f1_2, f2_2],...]), ... }
        """
        import databases
        import importlib.resources as pkg_resources
        f1f2_dict = {}
        db_file = pkg_resources.open_text(databases, database_name)
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
        (shape [N, 3], columns: E, f1, f2), returns (f1 + i f2)
        by simple linear interpolation near the requested energy.
        """
        E = energy
        energies = f1f2_table[:, 0]
        idx = np.searchsorted(energies, E)
        if idx >= len(energies):
            idx = len(energies) - 1
        if idx == 0:
            idx = 1
        E0, f10, f20 = energies[idx - 1], f1f2_table[idx - 1, 1], f1f2_table[idx - 1, 2]
        E1, f11, f21 = energies[idx],     f1f2_table[idx, 1],     f1f2_table[idx, 2]
        denom = (E1 - E0) if (E1 > E0) else 1e-20
        w = (E - E0) / denom
        f1 = f10 + (f11 - f10)*w
        f2 = f20 + (f21 - f20)*w
        return f1 + 1j*f2

    ## Main Functions
    def atomic_direct_scattering(self, sample, detector, offset=0):
        """
        High-level API: compute scattering at each detector pixel,
        store the result back into the detector object.
        """
        measurement_positions = detector.pixel_coordinates
        measurement_shape = detector.shape
        pixel_values = self.interact_beam(sample, measurement_positions, measurement_shape)
        detector.input_pixel_values(pixel_values-offset)
        
    def interact_beam(self, sample, measurement_positions, measurement_shape):
        """
        Perform beam-sample interaction, distributing the computation across all GPUs,
        with 2 threads per GPU. Each thread:
         - Processes a subset of chunk indices in a round-robin fashion
         - Launches 4 asynchronous kernel calls, then forces the 5th to be synchronous
         - Accumulates into the GPU's partial detector_field via atomicAdd

        Finally, the partial fields from each GPU are summed on the host.
        """
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"Found {device_count} GPUs.")
        # Build kernel
        interaction_kernel = self.build_interaction_kernel()
        # Parse databases
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        # Prepare per-device detector fields, coordinate arrays
        # measurement_positions shape = (3, Nx*Ny)
        Nx, Ny = measurement_shape
        device_arrays = []
        for dev_id in range(device_count):
            with cp.cuda.Device(dev_id):
                x_coords_gpu = cp.array(measurement_positions[0,:], dtype=cp.float32)
                y_coords_gpu = cp.array(measurement_positions[1,:], dtype=cp.float32)
                z_coords_gpu = cp.array(measurement_positions[2,:], dtype=cp.float32)
                detector_field_gpu = cp.zeros((Nx*Ny,), dtype=cp.complex64)
                device_arrays.append((x_coords_gpu, y_coords_gpu, z_coords_gpu, detector_field_gpu))
        # Create threads
        num_threads_per_device = 2
        threads = []
        # Round-robin assignment of chunk indices to (device, thread)
        total_threads = device_count * num_threads_per_device
        chunk_total = sample.chunk_total
        for dev_id in range(device_count):
            for thread_idx in range(num_threads_per_device):
                # Build the list of chunk indices that this thread will process
                # e.g. chunks = [ i for i in range(1, chunk_total+1)
                #                 if (i-1) % total_threads == (dev_id*num_threads_per_device + thread_idx) ]
                tid_global = dev_id * num_threads_per_device + thread_idx
                chunk_indices = []
                for cidx in range(1, chunk_total + 1):
                    if (cidx - 1) % total_threads == tid_global:
                        chunk_indices.append(cidx)
                # Ensure there are chunks being sent
                if len(chunk_indices) == 0:
                    continue
                # Assign thread object
                t = threading.Thread(
                    target=self.gpu_thread_worker,
                    args=(
                        dev_id,
                        chunk_indices,
                        sample,
                        interaction_kernel,
                        db_dict_f0_all,
                        db_dict_f1f2_all,
                        device_arrays,
                        (Nx, Ny),
                        )
                    )
                threads.append(t)
        # Start all threads
        for t in threads:
            t.start()
        # Join all threads
        for t in threads:
            t.join()
        # Combine final fields on host
        combined_field = np.zeros((Ny, Nx), dtype=np.complex64)
        for dev_id in range(device_count):
            with cp.cuda.Device(dev_id):
                # device_arrays[dev_id][3] is the detector_field_gpu
                field_component = device_arrays[dev_id][3].reshape((Ny, Nx)).get()
            combined_field += field_component
        # Final cleanup
        for dev_id in range(device_count):
            with cp.cuda.Device(dev_id):
                # release references
                del device_arrays[dev_id]
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        return combined_field
    
    def gpu_thread_worker(self,dev_id,chunk_indices,sample,interaction_kernel,db_dict_f0_all,db_dict_f1f2_all,device_arrays,measurement_shape):
        """
        Each thread processes a subset of chunk indices on GPU 'dev_id'.
        We issue 4 asynchronous kernel calls, then the 5th one is made synchronous
        (by calling stream.synchronize() immediately after the launch).
        """
        cp.cuda.Device(dev_id).use()
        stream = cp.cuda.Stream()
        # Unpack references to relevant GPU arrays
        x_coords_gpu, y_coords_gpu, z_coords_gpu, detector_field_gpu = device_arrays[dev_id]
        # Precompute wave number
        k_gpu = cp.float32(2.0 * np.pi / self._wavelength)
        # Compute GPU grid size
        block_size = (16, 16)
        grid_size = ((measurement_shape[0] + block_size[0] - 1) // block_size[0],
                     (measurement_shape[1] + block_size[1] - 1) // block_size[1])
        local_count = 0
        for chunk_idx in chunk_indices:
            species_chunk_np = sample.load_chunk_species(chunk_idx, gpu=False)
            atom_count = species_chunk_np.shape[0]
            if atom_count == 0:
                # Move on to next chunk
                continue
            unique_elements = pd.unique(species_chunk_np)
            # Build scattering anomalous + f0 params on CPU
            scattering_anom_np = np.zeros(atom_count, dtype=np.complex64)
            f0_params_np = np.zeros((atom_count, 11), dtype=np.float32)
            for el in unique_elements:
                el_mask = (species_chunk_np == el)
                if el not in db_dict_f0_all:
                    # skip if not found in f0 DB
                    continue
                # Interpolate f1,f2
                table = db_dict_f1f2_all.get(el, None)
                if table is None:
                    continue
                scattering_anom_np[el_mask] = self.get_f1f2_from_params(self._energy, table)
                # f0
                f0_params_np[el_mask] = db_dict_f0_all[el]
            # Move chunk positions to GPU
            with stream:
                positions_chunk_cp = cp.array(sample.load_chunk_positions(chunk_idx, gpu=True),dtype=cp.float32)
                px = positions_chunk_cp[:, 0].copy()
                py = positions_chunk_cp[:, 1].copy()
                pz = positions_chunk_cp[:, 2].copy()
                scattering_anom_cp = cp.asarray(scattering_anom_np)
                f0_params_cp       = cp.asarray(f0_params_np)
            # Launch the kernel asynchronously on stream
            with stream:
                interaction_kernel(
                    grid_size,
                    block_size,
                    (
                        np.int32(atom_count),
                        k_gpu,
                        px / 1e10,  # convert to meters => 1A=1e-10 m
                        py / 1e10,
                        pz / 1e10,
                        scattering_anom_cp,
                        f0_params_cp,
                        x_coords_gpu / 1e10,
                        y_coords_gpu / 1e10,
                        z_coords_gpu / 1e10,
                        detector_field_gpu,
                        np.int32(measurement_shape[0]),
                        np.int32(measurement_shape[1])
                    )
                )
            local_count += 1
            # The number after the % determines the number of kernels launch per worker
            if (local_count % 5) == 0:
                stream.synchronize()
            # Cleanup references in Python
            del species_chunk_np
            del scattering_anom_np, f0_params_np
            del positions_chunk_cp, px, py, pz
            del scattering_anom_cp, f0_params_cp
            gc.collect()
        # Final sync to ensure all kernels are done
        stream.synchronize()
        # Done
        stream = None  # let Python GC handle it
