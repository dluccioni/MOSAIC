# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import json
import os
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

    def create_beam(self,
                    energy,
                    eV=True,
                    beam_shape="rectangular",
                    beam_size=(1000.0, 1000.0),
                    beam_samples=(256, 256),
                    beam_profile="uniform",
                    gaussian_waist=None):
        """
        Initialize beam with direction hard-coded to +x for performance.
        (User 'direction' is ignored.)
        """
        # Force direction to +x
        self._direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Energy/wavelength
        if not eV:
            energy = energy / self._q
        self._energy = float(energy)
        self._wavelength = self._hq * self._c / self._energy

        # k-vector (only x is needed/used)
        k = 2.0 * np.pi / self._wavelength
        self._kx_scalar = np.float32(k)
        self._ky_scalar = np.float32(0.0)
        self._kz_scalar = np.float32(0.0)

        # store shape/size/profile
        self._beam_shape   = str(beam_shape).lower()
        self._beam_size    = (float(beam_size[0]), float(beam_size[1]))
        self._beam_samples = (int(beam_samples[0]), int(beam_samples[1]))
        self._beam_profile = str(beam_profile).lower()
        self._gauss_waist  = gaussian_waist

        # transverse basis is exactly (ŷ, ẑ)
        self._beam_e1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._beam_e2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # build grid & E0(u,v)
        self._init_beam_grid()
        
    def _init_beam_grid(self):
        """
        Build the (u,v) grid centered at 0 with sizes beam_size and samples beam_samples.
        Sets:
        _beam_Ny, _beam_Nz, _beam_du, _beam_dv, _beam_uc, _beam_vc,
        _beam_u_centers, _beam_v_centers, _beam_E0_map (complex64, no phase)
        """
        Ny, Nz = self._beam_samples
        Sy, Sz = self._beam_size  # Å
        Ny = int(max(1, Ny)); Nz = int(max(1, Nz))
        self._beam_Ny, self._beam_Nz = Ny, Nz

        self._beam_du = float(Sy) / Ny  # Å per grid step in u
        self._beam_dv = float(Sz) / Nz  # Å per grid step in v
        self._beam_uc = (Ny - 1) * 0.5  # center index along u
        self._beam_vc = (Nz - 1) * 0.5  # center index along v

        u_centers = (np.arange(Ny, dtype=np.float32) - self._beam_uc) * self._beam_du
        v_centers = (np.arange(Nz, dtype=np.float32) - self._beam_vc) * self._beam_dv
        U, V = np.meshgrid(u_centers, v_centers, indexing='ij')  # (Ny,Nz)

        # support mask
        if self._beam_shape == "circular":
            ry = 0.5 * Sy
            rz = 0.5 * Sz
            mask = ((U / max(ry, 1e-9))**2 + (V / max(rz, 1e-9))**2) <= 1.0
        else:
            mask = np.ones_like(U, dtype=bool)  # rectangular support equals full grid

        # profile amplitude (no phase here)
        if self._beam_profile == "gaussian":
            wy, wz = self._gauss_waist if (self._gauss_waist is not None) else (0.5 * Sy, 0.5 * Sz)
            wy = max(float(wy), 1e-6); wz = max(float(wz), 1e-6)
            A0 = np.exp(-((U / wy) ** 2 + (V / wz) ** 2)).astype(np.float32)
            A0 *= mask.astype(np.float32)
        else:
            A0 = mask.astype(np.float32)

        self._beam_u_centers = u_centers
        self._beam_v_centers = v_centers
        # complex; phase=0
        self._beam_E0_map = (A0.astype(np.float32) + 0.0j).astype(np.complex64)
        
    def read_beam_metadata(self):
        """
        Read beam metadata from JSON and restore the beam including its transverse grid.
        Rebuilds derived quantities:
        - normalized direction
        - k‑vector components (_kx_scalar, _ky_scalar, _kz_scalar)
        - (u,v) orthonormal basis (self._beam_e1, self._beam_e2)
        - beam grid (centers, spacings, E0 profile) via _init_beam_grid()
        Backward compatible with older files that lack the beam‑grid fields.
        """
        metadata_filename = os.path.join(self.directory, "beam_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            beam_metadata = json.load(f)

        # --- core scalars --------------------------------------------------------
        direction = beam_metadata.get("direction", None)
        if direction is None:
            direction = [1.0, 0.0, 0.0]
        self._direction = np.array(direction, dtype=np.float32)
        self._direction = self._direction / np.linalg.norm(self._direction)

        self._energy     = float(beam_metadata.get("energy", self._energy if self._energy is not None else 1.0))
        self._wavelength = float(beam_metadata.get("wavelength",
                                                (self._hq * self._c / self._energy)))

        # wavevector components (derived)
        k = 2.0 * np.pi / self._wavelength
        self._kx_scalar = float(self._direction[0] * k)
        self._ky_scalar = float(self._direction[1] * k)
        self._kz_scalar = float(self._direction[2] * k)

        # --- beam‑grid primitives -----------------------------------------------
        self._beam_shape = str(beam_metadata.get("beam_shape", "rectangular")).lower()

        # ensure non‑degenerate default sizes (Å)
        default_size = (1000.0, 1000.0)
        size_list = beam_metadata.get("beam_size", default_size)
        if size_list is None or len(size_list) != 2:
            size_list = default_size
        Sy = float(size_list[0]) if float(size_list[0]) > 0.0 else default_size[0]
        Sz = float(size_list[1]) if float(size_list[1]) > 0.0 else default_size[1]
        self._beam_size = (Sy, Sz)

        # samples (Ny, Nz) – fall back to a sensible grid if missing
        samples = beam_metadata.get("beam_samples", None)
        if samples is None or (isinstance(samples, (list, tuple)) and len(samples) != 2):
            samples = (256, 256)
        Ny = int(samples[0]); Nz = int(samples[1])
        Ny = max(1, Ny); Nz = max(1, Nz)
        self._beam_samples = (Ny, Nz)

        # profile
        self._beam_profile = str(beam_metadata.get("beam_profile", "uniform")).lower()
        gw = beam_metadata.get("gaussian_waist", None)
        if gw is None:
            # if profile is gaussian but waist was not provided, default to half‑size
            if self._beam_profile == "gaussian":
                self._gauss_waist = (0.5 * Sy, 0.5 * Sz)
            else:
                self._gauss_waist = None
        else:
            # accept list/tuple/float
            if isinstance(gw, (list, tuple)) and len(gw) == 2:
                self._gauss_waist = (float(gw[0]), float(gw[1]))
            else:
                # malformed -> safe default
                self._gauss_waist = (0.5 * Sy, 0.5 * Sz) if self._beam_profile == "gaussian" else None

        # --- transverse basis and grid ------------------------------------------
        e1, e2 = self.make_orthonormal_basis(self._direction)
        self._beam_e1 = e1.astype(np.float32)
        self._beam_e2 = e2.astype(np.float32)

        # Build the beam grid and E0(u,v) based on the loaded settings
        if hasattr(self, "_init_beam_grid"):
            self._init_beam_grid()

        print(f"Beam metadata loaded from {metadata_filename}.")

    ## Data Handling Functions    
    def write_beam_metadata(self, override_directory=None):
        """
        Serialize the beam's internal state (including the beam‑grid definition)
        to a JSON file for future restoration.

        Newly saved fields:
        - beam_samples    : [Ny, Nz] on the transverse (u,v) grid
        - beam_profile    : "uniform" or "gaussian"
        - gaussian_waist  : [wy, wz] in Å for Gaussian profile (1/e^2 radii) or null
        - metadata_version: integer schema tag (>=2 when beam grid is present)
        """
        # graceful fallbacks if older attributes aren't present
        direction = self._direction.tolist() if getattr(self, "_direction", None) is not None else None
        energy    = getattr(self, "_energy", None)
        wavelength= getattr(self, "_wavelength", None)

        beam_shape   = getattr(self, "_beam_shape", "rectangular")
        beam_size    = list(getattr(self, "_beam_size", (1000.0, 1000.0)))     # Å
        beam_samples = getattr(self, "_beam_samples", None)
        if beam_samples is not None:
            beam_samples = [int(beam_samples[0]), int(beam_samples[1])]
        beam_profile = getattr(self, "_beam_profile", "uniform")
        gauss_waist  = getattr(self, "_gauss_waist", None)
        if gauss_waist is not None:
            gauss_waist = [float(gauss_waist[0]), float(gauss_waist[1])]

        beam_metadata = {
            "metadata_version": 2,
            "direction"       : direction,
            "energy"          : energy,
            "wavelength"      : wavelength,
            "beam_shape"      : beam_shape,
            "beam_size"       : beam_size,       # [size_u_Å, size_v_Å]
            "beam_samples"    : beam_samples,    # [Ny, Nz]
            "beam_profile"    : beam_profile,    # "uniform" | "gaussian"
            "gaussian_waist"  : gauss_waist      # [wy_Å, wz_Å] or null
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
        with pkg_resources.open_text(databases.scattering, database_name) as db_file:
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
        with pkg_resources.open_text(databases.scattering, database_name) as db_file:
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
        CPU scattering routine (CFFI), now matching the GPU math:

        - initial_amp is COMPLEX (passed as real & imag arrays)
        - optional forward-component removal via f0(Q)-f0(0)
        - includes classical electron radius scaling (r_e)
        """
        from cffi import FFI

        c_source = r'''
        #include <math.h>
        #include <stddef.h>

        static inline float get_f0_value(float Q_val, const float* params)
        {
            // params: [a1 a2 a3 a4 a5 c b1 b2 b3 b4 b5]
            const float PI_F = 3.14159265358979323846f;
            const float K_SCALE_FACTOR = 0.25f * 1.0e-10f / PI_F;  // 0.25 * Å / π
            const float k   = K_SCALE_FACTOR * Q_val;
            const float kk  = k*k;

            float f0_val = params[5]; // c
            for (int i=0;i<5;i++){
                const float ai = params[i];
                const float bi = params[6+i];
                f0_val += ai * expf(-bi * kk);
            }
            return f0_val;
        }

        void compute_scattering_cffi(
            int atom_count,
            const float *positions,       // (atom_count,3) [m]
            const float *f0_params,       // (atom_count,11)
            const float *f0_zero,         // (atom_count,)
            int remove_forward,           // 0/1
            const float *s_anom_real,     // (atom_count,)
            const float *s_anom_imag,     // (atom_count,)
            const float *initial_amp_r,   // (atom_count,) REAL part of entrance amp
            const float *initial_amp_i,   // (atom_count,) IMAG part of entrance amp
            int Nx, int Ny,
            const float *coords_x,        // (Nx*Ny) [m]
            const float *coords_y,
            const float *coords_z,
            float k_val,                  // 2π/λ  [rad/m]
            float *out_r, float *out_i    // (Nx*Ny)
        )
        {
            const float PI_F = 3.14159265358979323846f;
            const float rE_F = 2.81794092e-15f;  // classical electron radius [m]
            const int pixel_count = Nx*Ny;

            for (int a=0; a<atom_count; ++a)
            {
                const float ax = positions[3*a+0];
                const float ay = positions[3*a+1];
                const float az = positions[3*a+2];

                const float *f0p = &f0_params[a*11];
                const float f00  = f0_zero[a];
                const float sanr = s_anom_real[a];
                const float sani = s_anom_imag[a];

                const float amp_r = initial_amp_r[a];
                const float amp_i = initial_amp_i[a];

                for (int p=0; p<pixel_count; ++p)
                {
                    const float dx = coords_x[p] - ax;
                    const float dy = coords_y[p] - ay;
                    const float dz = coords_z[p] - az;

                    float r_det = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (r_det == 0.0f) continue;

                    float rdx = (dx / r_det);
                    float tmp = 2.0f*(1.0f - rdx);
                    if (tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_val * sqrtf(tmp);

                    float f0_val = get_f0_value(Q_val, f0p);
                    if (remove_forward != 0) { f0_val -= f00; }

                    // s_tot = f0 + s_anom (complex)
                    float s_re = (f0_val + sanr);
                    float s_im = (sani);

                    // multiply by complex entrance amplitude: s_tot *= (amp_r + i*amp_i)
                    float t_re = amp_r * s_re - amp_i * s_im;
                    float t_im = amp_r * s_im + amp_i * s_re;

                    float wavelength_m = (2.0f * PI_F) / k_val;
                    float phase = k_val * (fmodf(ax, wavelength_m) + fmodf(r_det, wavelength_m));
                    float cph = cosf(phase);
                    float sph = sinf(phase);

                    float val_r = (t_re * cph - t_im * sph) * rE_F;
                    float val_i = (t_re * sph + t_im * cph) * rE_F;

                    out_r[p] += val_r;
                    out_i[p] += val_i;
                }
            }
        }
        ''';

        ffi_obj = FFI()
        ffi_obj.cdef(r"""
            void compute_scattering_cffi(
                int atom_count,
                const float *positions,
                const float *f0_params,
                const float *f0_zero,
                int remove_forward,
                const float *s_anom_real,
                const float *s_anom_imag,
                const float *initial_amp_r,
                const float *initial_amp_i,
                int Nx, int Ny,
                const float *coords_x,
                const float *coords_y,
                const float *coords_z,
                float k_val,
                float *out_r, float *out_i
            );
        """)
        C_mod = ffi_obj.verify(c_source, extra_compile_args=['-O3'])
        return ffi_obj, C_mod
    
    @staticmethod
    def build_interaction_kernel():
        """
        CUDA kernel specialized for +x beam:
        - dotv = dx/r  (no dot-product)
        - k_mag = |kx| (no sqrt)
        - anomalous term is NOT zeroed when remove_forward==1
        Signature stays the same; call sites unchanged.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for GPU scattering kernels.")

        _cuda_source = r'''
        #define CHUNK_SIZE 128
        extern "C" {
        __device__ __forceinline__ float2 get_f0_from_params(float Q_val, const float* params)
        {
            // f0(Q) = c + Σ a_i exp(-b_i * s^2), with s = Q/(4π) in Å^-1 (see CPU path)
            const float PI_F   = 3.14159265358979323846f;
            const float K_SCALE= 0.25f * 1.0e-10f / PI_F;  // Q[m^-1] -> s[Å^-1]
            float s  = K_SCALE * Q_val;
            float ss = s * s;
            float f0 = params[5]; // c
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                float ai = params[i];
                float bi = params[6 + i];
                f0 += ai * __expf(-bi * ss);
            }
            return make_float2(f0, 0.0f);
        }

        __global__ void interaction_kernal(
            const int   nAtoms,
            const float* __restrict__ kx_atom,   // kept for signature; kx>0, ky=kz=0
            const float* __restrict__ ky_atom,   // unused for +x specialization
            const float* __restrict__ kz_atom,   // unused for +x specialization
            const float* __restrict__ px,        // atom x [m]
            const float* __restrict__ py,        // atom y [m]
            const float* __restrict__ pz,        // atom z [m]
            const float2* __restrict__ initial_amp,
            const float2* __restrict__ scattering_anom,
            const float*  __restrict__ f0_params,
            const float*  __restrict__ f0_zero,
            const float* __restrict__ x_coords,  // detector coords [m]
            const float* __restrict__ y_coords,
            const float* __restrict__ z_coords,
            float2*      __restrict__ detector_field,
            const int    Nx,
            const int    Ny,
            const int    remove_forward)
        {
            const float PI_F = 3.14159265358979323846f;
            const float rE_F = 2.81794092e-15f;

            int pxid = blockIdx.x * blockDim.x + threadIdx.x;
            int pyid = blockIdx.y * blockDim.y + threadIdx.y;
            bool in_bounds = (pxid < Nx && pyid < Ny);
            int pixel_index = pyid * Nx + pxid;

            float tx = 0.0f, ty = 0.0f, tz = 0.0f;
            if (in_bounds) {
                tx = x_coords[pixel_index];
                ty = y_coords[pixel_index];
                tz = z_coords[pixel_index];
            }

            float2 sum_val = make_float2(0.0f, 0.0f);

            __shared__ float  s_px[CHUNK_SIZE];
            __shared__ float  s_py[CHUNK_SIZE];
            __shared__ float  s_pz[CHUNK_SIZE];
            __shared__ float2 s_amp[CHUNK_SIZE];
            __shared__ float2 s_anom[CHUNK_SIZE];
            __shared__ float  s_params[CHUNK_SIZE * 11];
            __shared__ float  s_kx[CHUNK_SIZE];
            __shared__ float  s_f0z[CHUNK_SIZE];

            int threads_in_block = blockDim.x * blockDim.y;
            int t_id = threadIdx.y * blockDim.x + threadIdx.x;

            for (int tile_start = 0; tile_start < nAtoms; tile_start += CHUNK_SIZE) {
                for (int t = t_id; t < CHUNK_SIZE; t += threads_in_block) {
                    int a = tile_start + t;
                    if (a < nAtoms) {
                        s_px[t] = px[a]; s_py[t] = py[a]; s_pz[t] = pz[a];
                        s_amp[t]= initial_amp[a];
                        s_anom[t]=scattering_anom[a]; // DO NOT zero, even if remove_forward
                        s_kx[t] = kx_atom[a];         // k along +x (>=0)
                        s_f0z[t]= f0_zero[a];
                        #pragma unroll
                        for (int pi=0; pi<11; ++pi)
                            s_params[t*11 + pi] = f0_params[a*11 + pi];
                    }
                }
                __syncthreads();

                if (in_bounds) {
                    #pragma unroll 4
                    for (int j = 0; j < CHUNK_SIZE; ++j) {
                        int a = tile_start + j;
                        if (a >= nAtoms) break;

                        float dx = tx - s_px[j];
                        float dy = ty - s_py[j];
                        float dz = tz - s_pz[j];
                        float r_det = sqrtf(dx*dx + dy*dy + dz*dz);
                        if (r_det == 0.0f) continue;

                        // +x specialization:
                        float k_mag = fabsf(s_kx[j]);      // no sqrt
                        float dotv  = dx / r_det;          // cos(theta)

                        float tmp = 2.0f*(1.0f - dotv);
                        if (tmp < 0.0f) tmp = 0.0f;
                        float Q_val = k_mag * __fsqrt_rn(tmp);

                        const float* param_ptr = &s_params[j*11];
                        float2 f0c = get_f0_from_params(Q_val, param_ptr);
                        if (remove_forward) {
                            f0c.x -= s_f0z[j];  // f0(Q)-f0(0)
                        }

                        float2 s_tot = make_float2(f0c.x + s_anom[j].x, f0c.y + s_anom[j].y);

                        // phase uses only atom x (entrance) + path length
                        float wavelength_m = (2.0f * PI_F) / k_mag;
                        float ax_mod = fmodf(s_px[j], wavelength_m);
                        float rdet_mod = fmodf(r_det, wavelength_m);
                        float phase = k_mag * (ax_mod + rdet_mod);

                        float cph, sph;
                        __sincosf(phase, &sph, &cph);

                        float2 amp_a = s_amp[j];
                        float real_part = amp_a.x * s_tot.x - amp_a.y * s_tot.y;
                        float imag_part = amp_a.x * s_tot.y + amp_a.y * s_tot.x;

                        float2 val;
                        val.x = real_part * cph - imag_part * sph;
                        val.y = real_part * sph + imag_part * cph;

                        sum_val.x += val.x * rE_F;
                        sum_val.y += val.y * rE_F;
                    }
                }
                __syncthreads();
            }
            if (in_bounds) {
                detector_field[pixel_index].x += sum_val.x;
                detector_field[pixel_index].y += sum_val.y;
            }
        }
        }
        ''';

        kernel_module = cp.RawModule(
            code=_cuda_source,
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
        CUDA kernel to expand paths by adding neighbor-induced phase only.
        Unused inputs k_in_mag and wavelength removed from the signature.
        """
        code = r'''
        #include <math.h>
        extern "C" {

        __device__ __forceinline__ float2 cplx_expf(float phase)
        {
            float s, c;
            __sincosf(phase, &s, &c);
            float2 v; v.x = c; v.y = s;
            return v;
        }

        __global__
        void expand_paths_kernel(
            // Incoming paths (size = numIncomingPaths)
            const float*  in_x,
            const float*  in_y,
            const float*  in_z,
            const float*  in_kx,
            const float*  in_ky,
            const float*  in_kz,
            const float2* in_amp,
            const int*    in_atomIndex,

            // neighbor info (per-atom, flattened)
            const int*    neighborStart,
            const int*    neighborCount,
            const float*  neighborPhase,
            const float*  neighborKx,
            const float*  neighborKy,
            const float*  neighborKz,
            const int*    neighborIdxAtom,
            const int*    neighborSpc,

            // global size
            const int     numIncomingPaths,

            // outputs (capacity = maxPaths)
            float*  out_x,
            float*  out_y,
            float*  out_z,
            float*  out_kx,
            float*  out_ky,
            float*  out_kz,
            float2* out_amp,
            int*    out_atomIndex,
            int*    out_spc,

            const int     maxPaths
        )
        {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= numIncomingPaths) return;

            float2 samp = in_amp[idx];
            int    sourceAtom = in_atomIndex[idx];

            int startN = neighborStart[sourceAtom];
            int countN = neighborCount[sourceAtom];

            for (int n = 0; n < countN; ++n) {
                int gN = startN + n;

                float phase_ij = neighborPhase[gN];
                float2 eip = cplx_expf(phase_ij);
                float2 new_amp;
                new_amp.x = samp.x * eip.x - samp.y * eip.y;
                new_amp.y = samp.x * eip.y + samp.y * eip.x;

                int outPos = atomicAdd((unsigned int*)&out_atomIndex[maxPaths], 1);
                if (outPos < maxPaths) {
                    out_x[outPos]   = 0.f;
                    out_y[outPos]   = 0.f;
                    out_z[outPos]   = 0.f;
                    out_kx[outPos]  = neighborKx[gN];
                    out_ky[outPos]  = neighborKy[gN];
                    out_kz[outPos]  = neighborKz[gN];
                    out_amp[outPos] = new_amp;
                    out_atomIndex[outPos] = neighborIdxAtom[gN];
                    out_spc[outPos]  = neighborSpc[gN];
                }
            }
        }
        } // extern "C"
        '''

        mod = cp.RawModule(
            code=code,
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true'),
            backend='nvcc'
        )
        return mod.get_function('expand_paths_kernel')
    
    @staticmethod
    def _next_pow_two(n) -> int:
        """
        Return the next power‑of‑two ≥ n.

        * Accepts int, float, or NumPy scalar.
        * Works for n ≤ 2**63‑1.
        """
        # ensure a real Python int
        n_int = int(np.ceil(n))              # ← converts numpy.float64 → int
        if n_int < 1:
            return 1
        return 1 << (n_int - 1).bit_length()

    @staticmethod
    def _choose_optimal_pad(
        Nx: int, Ny: int, dx: float, dy: float,
        wavelength: float, z: float,
        safety: float = 1.1,
        enforce_pow2: bool = True,
        min_pad_factor: float = 1.0,
    ):
        """
        Compute padding for angular spectrum propagation so that wrap-around is
        avoided after a distance |z|. We model the worst-case lateral drift of any
        plane-wave component the sampled grid can represent.

        The largest *propagating* angle supported by sampling along x (y) is
            sin(theta_x_max) = min( 1,  λ / (2·dx) )
            sin(theta_y_max) = min( 1,  λ / (2·dy) )
        (derived from kx_Nyq = π/d and sinθ = kx/k with k = 2π/λ).

        Required half-padding (meters) per axis is then
            pad_x = |z| · tan(theta_x_max),  pad_y = |z| · tan(theta_y_max)
        and we convert to pixels and add a small safety factor.

        Returns
        -------
        Nx_pad, Ny_pad : int
            Final padded sizes (≥ original). If enforce_pow2, each is rounded up
            to the next power of two. A minimum scaling of min_pad_factor× is also
            enforced.
        """
        zabs = abs(float(z))
        if zabs == 0.0:
            Nx2 = max(int(np.ceil(Nx * min_pad_factor)), Nx)
            Ny2 = max(int(np.ceil(Ny * min_pad_factor)), Ny)
            if enforce_pow2:
                Nx2 = beam._next_pow_two(Nx2)
                Ny2 = beam._next_pow_two(Ny2)
            return int(Nx2), int(Ny2)

        # sampling-limited maximum angles
        srx = min(1.0, float(wavelength) / (2.0 * float(dx)))
        sry = min(1.0, float(wavelength) / (2.0 * float(dy)))
        # avoid tan(π/2)
        srx = min(srx, 0.999999)
        sry = min(sry, 0.999999)

        tanx = srx / np.sqrt(max(1e-18, 1.0 - srx * srx))
        tany = sry / np.sqrt(max(1e-18, 1.0 - sry * sry))

        pad_x_m = safety * zabs * tanx
        pad_y_m = safety * zabs * tany

        pad_x_px = int(np.ceil(pad_x_m / float(dx)))
        pad_y_px = int(np.ceil(pad_y_m / float(dy)))

        Nx2 = Nx + 2 * pad_x_px
        Ny2 = Ny + 2 * pad_y_px

        # also respect a minimum multiplicative padding if requested
        Nx2 = max(Nx2, int(np.ceil(Nx * min_pad_factor)))
        Ny2 = max(Ny2, int(np.ceil(Ny * min_pad_factor)))

        if enforce_pow2:
            Nx2 = beam._next_pow_two(Nx2)
            Ny2 = beam._next_pow_two(Ny2)

        return int(Nx2), int(Ny2)
    
    @staticmethod
    def build_propagation_multiplier_kernel():
        src = r'''
        #include <math.h>

        extern "C" __global__
        void prop_mul_kernel(
            const float* __restrict__ kx,   // length Nx   [rad/m]
            const float* __restrict__ ky,   // length Ny   [rad/m]
            const float  k,                 // 2*pi/lambda [rad/m]
            const float  z,                 // propagation [m]
            const int    Nx,
            const int    Ny,
            float2* __restrict__ F)         // spectrum (Ny*Nx), row-major
        {
            int ix = blockIdx.x * blockDim.x + threadIdx.x;
            int iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= Nx || iy >= Ny) return;

            const int idx = iy * Nx + ix;

            const float kxv = kx[ix];
            const float kyv = ky[iy];
            const float kt2 = kxv * kxv + kyv * kyv;
            const float kz2 = k * k - kt2;

            // build multiplier H(kt) = exp(+i z sqrt(k^2 - kt^2)) for propagating,
            // and = exp(-|z| sqrt(kt^2 - k^2)) for evanescent (pure decay).
            float phase = 0.0f;
            float amp   = 1.0f;

            if (kz2 >= 0.0f) {
                phase = z * sqrtf(kz2);
                amp   = 1.0f;
            } else {
                // strictly decaying, never growing (note fabsf(z))
                amp   = expf(-fabsf(z) * sqrtf(-kz2));
                phase = 0.0f;
            }

            const float cph = cosf(phase);
            const float sph = sinf(phase);

            float2 G = F[idx];
            const float hr = amp * cph;
            const float hi = amp * sph;

            float2 out;
            out.x = G.x * hr - G.y * hi;
            out.y = G.x * hi + G.y * hr;

            F[idx] = out;
        }
        ''';

        src = src.encode('ascii', 'backslashreplace').decode('ascii')

        mod = cp.RawModule(
            code    = src,
            backend = 'nvcc',
            options = ('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return mod.get_function('prop_mul_kernel')

    @staticmethod
    def compile_propagation_multiplier_cffi():
        source = r'''
        #include <math.h>
        #include <complex.h>

        void prop_mul_cpu(
            const int      Nx,
            const int      Ny,
            const float*   kx,    /* rad/m, length Nx */
            const float*   ky,    /* rad/m, length Ny */
            const float    k,     /* 2*pi/lambda */
            const float    z,     /* meters */
            float _Complex* F)    /* spectrum (Ny*Nx), row-major */
        {
            const float az = fabsf(z);
            for (int iy = 0; iy < Ny; ++iy) {
                const float kyv = ky[iy];
                for (int ix = 0; ix < Nx; ++ix) {
                    const float kxv = kx[ix];
                    const float kt2 = kxv*kxv + kyv*kyv;
                    const float kz2 = k*k - kt2;

                    float phase, amp;
                    if (kz2 >= 0.0f) {
                        phase = z * sqrtf(kz2);
                        amp   = 1.0f;
                    } else {
                        phase = 0.0f;
                        amp   = expf(-az * sqrtf(-kz2));
                    }

                    const float cph = cosf(phase);
                    const float sph = sinf(phase);
                    const float _Complex H = amp * (cph + I*sph);

                    const int idx = iy * Nx + ix;
                    F[idx] *= H;
                }
            }
        }
        ''';

        ffi = FFI()
        ffi.cdef('void prop_mul_cpu(int,int,const float*,const float*,float,float,'
                'float _Complex*);')
        lib = ffi.verify(source, extra_compile_args=['-O3'])
        return ffi, lib

    @staticmethod
    def _safe_bincount_gpu(idxs, weights, size, dtype=None):
        """
        Robust CuPy bincount with bool(...) guards around reductions.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for _safe_bincount_gpu")

        if size <= 0:
            return cp.zeros((0,), dtype=cp.float32 if dtype is None else dtype)

        if idxs is None or int(getattr(idxs, "size", 0)) == 0:
            return cp.zeros((size,), dtype=cp.float32 if dtype is None else dtype)

        idxs = cp.asarray(idxs)
        m = cp.isfinite(idxs)
        if not bool(m.all()):
            idxs = idxs[m]
            if weights is not None:
                weights = cp.asarray(weights)[m]
        idxs = idxs.astype(cp.int64, copy=False)

        if idxs.size == 0:
            return cp.zeros((size,), dtype=cp.float32 if dtype is None else dtype)

        m = (idxs >= 0) & (idxs < size)
        if not bool(m.all()):
            idxs = idxs[m]
            if weights is not None:
                weights = cp.asarray(weights)[m]

        if idxs.size == 0:
            return cp.zeros((size,), dtype=cp.float32 if dtype is None else dtype)

        if dtype is None:
            dtype = (weights.dtype if weights is not None else cp.float32)

        if weights is None:
            out = cp.bincount(idxs, minlength=size)
            return out.astype(dtype, copy=False) if out.dtype != dtype else out
        else:
            weights = cp.asarray(weights).astype(dtype, copy=False)
            return cp.bincount(idxs, weights=weights, minlength=size)
    # -------------------------------------

    ## Main Functions
    # -------------------------------------
    # Kinematic scattering
    def _compute_global_depth_bounds(self, sample, stage):
        """
        Global front-to-back bounds along the beam direction (in Å),
        after applying stage rotation/translation. Returns (s_min, s_max).
        """
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        s_min = np.float32(np.inf)
        s_max = np.float32(-np.inf)

        R = np.asarray(stage.rotation, dtype=np.float32)
        T = np.asarray(stage.translation, dtype=np.float32)

        for cid in range(1, sample.chunk_total + 1):
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
            if pos.size == 0:
                continue
            pos = pos @ R
            pos += T
            s_vals = pos @ k_hat
            cur_min = np.min(s_vals)
            cur_max = np.max(s_vals)
            if cur_min < s_min: s_min = cur_min
            if cur_max > s_max: s_max = cur_max

        if not np.isfinite(s_min) or (s_max <= s_min):
            # Fallback to a harmless 0..1 span
            return 0.0, 1.0
        return float(s_min), float(s_max)

    def cpu_scatter_chunk_cffi(self, complied_code, ffi_obj, chunk_id, sample,
                            Nx, Ny, coords_x_m, coords_y_m, coords_z_m,
                            db_dict_f0_all, db_dict_f1f2_all, k_val,
                            stage, detector=None, remove_forward_component=False,
                            initial_amp_complex=None):
        """
        CPU scattering for a single chunk. Matches GPU path:

        - Uses complex entrance amplitude per atom (initial_amp_complex).
        If None, defaults to 1+0j (or your own attenuation model).
        - Optional removal of forward component.
        - Includes r_e scaling (handled inside C kernel).
        """
        # Species & atom count
        species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
        atom_count = int(species_chunk_np.shape[0])
        if atom_count == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)

        # Per-atom anomalous and f0 params + f0(0)
        scattering_anom_np_real = np.zeros(atom_count, dtype=np.float32)
        scattering_anom_np_imag = np.zeros(atom_count, dtype=np.float32)
        f0_params_np            = np.zeros((atom_count, 11), dtype=np.float32)
        f0_zero_np              = np.zeros((atom_count,), dtype=np.float32)

        f0_zero_dict = self._build_f0_zero_dict(db_dict_f0_all)
        unique_elements = pd.unique(species_chunk_np)
        for el in unique_elements:
            el = str(el)
            if el not in db_dict_f0_all:
                continue
            mask = (species_chunk_np == el)
            # anomalous (if not removing forward we still keep anom, same as GPU)
            table = db_dict_f1f2_all.get(el, None)
            if table is not None:
                cplx = self.get_f1f2_from_params(self._energy, table)
                scattering_anom_np_real[mask] = float(cplx.real)
                scattering_anom_np_imag[mask] = float(cplx.imag)
            f0_params_np[mask] = db_dict_f0_all[el]
            f0_zero_np[mask]   = float(f0_zero_dict.get(el, 0.0))

        # Positions (Å) → stage → meters
        positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
        positions_chunk = positions_chunk @ stage.rotation
        positions_chunk += stage.translation
        positions_chunk_m = positions_chunk / 1e10

        # Entrance amplitude
        if initial_amp_complex is None:
            # default = 1+0j per atom (or use your attenuation model here)
            amp_r = np.ones((atom_count,), dtype=np.float32)
            amp_i = np.zeros((atom_count,), dtype=np.float32)
        else:
            amp_r = np.asarray(np.real(initial_amp_complex), dtype=np.float32, order='C')
            amp_i = np.asarray(np.imag(initial_amp_complex), dtype=np.float32, order='C')
            if amp_r.shape[0] != atom_count:
                raise ValueError("initial_amp_complex size mismatch for chunk {}".format(chunk_id))

        # Output buffers
        out_r = np.zeros(Nx*Ny, dtype=np.float32)
        out_i = np.zeros(Nx*Ny, dtype=np.float32)

        # Contiguous views
        positions_chunk_m = np.ascontiguousarray(positions_chunk_m)
        f0_params_np      = np.ascontiguousarray(f0_params_np)
        f0_zero_np        = np.ascontiguousarray(f0_zero_np)
        s_anom_r          = np.ascontiguousarray(scattering_anom_np_real)
        s_anom_i          = np.ascontiguousarray(scattering_anom_np_imag)
        amp_r             = np.ascontiguousarray(amp_r)
        amp_i             = np.ascontiguousarray(amp_i)

        # C pointers
        positions_ptr = ffi_obj.cast("const float *", positions_chunk_m.ctypes.data)
        f0_params_ptr = ffi_obj.cast("const float *", f0_params_np.ctypes.data)
        f0_zero_ptr   = ffi_obj.cast("const float *", f0_zero_np.ctypes.data)
        s_anom_r_ptr  = ffi_obj.cast("const float *", s_anom_r.ctypes.data)
        s_anom_i_ptr  = ffi_obj.cast("const float *", s_anom_i.ctypes.data)
        amp_r_ptr     = ffi_obj.cast("const float *", amp_r.ctypes.data)
        amp_i_ptr     = ffi_obj.cast("const float *", amp_i.ctypes.data)
        coords_x_ptr  = ffi_obj.cast("const float *", coords_x_m.ctypes.data)
        coords_y_ptr  = ffi_obj.cast("const float *", coords_y_m.ctypes.data)
        coords_z_ptr  = ffi_obj.cast("const float *", coords_z_m.ctypes.data)
        out_r_ptr     = ffi_obj.cast("float *", out_r.ctypes.data)
        out_i_ptr     = ffi_obj.cast("float *", out_i.ctypes.data)

        # Call C
        complied_code.compute_scattering_cffi(
            atom_count,
            positions_ptr,
            f0_params_ptr,
            f0_zero_ptr,
            int(remove_forward_component),
            s_anom_r_ptr,
            s_anom_i_ptr,
            amp_r_ptr,
            amp_i_ptr,
            Nx, Ny,
            coords_x_ptr, coords_y_ptr, coords_z_ptr,
            k_val,
            out_r_ptr, out_i_ptr
        )

        return (out_r + 1j*out_i).reshape((Ny, Nx)).astype(np.complex64)

    def interact_beam_cpu(
        self,
        sample,
        measurement_positions,
        measurement_shape,
        stage,
        detector=None,
        remove_forward_component=False,
        use_depth_ein=False,
        ein_cache_dir=None,
        recompute_cache=False
    ):
        """
        CPU kinematic scattering to match the GPU path:

        - Optional depth‑E_in per-atom entrance amplitude (complex), computed by
        sampling E0(u,v) and A(u,v) on the beam grid with a global depth fraction.
        - Optional forward component removal.
        - Caching of E_in per chunk (same keying scheme as GPU).
        """
        import hashlib, json
        Nx, Ny = measurement_shape

        # DBs
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        k_val = np.float32(2.0 * np.pi / self._wavelength)

        # detector coords [m]
        if cp is not None and isinstance(measurement_positions, cp.ndarray):
            measurement_positions = measurement_positions.get()
        coords_x_m = np.ascontiguousarray(measurement_positions[0, :].astype(np.float32) / 1e10)
        coords_y_m = np.ascontiguousarray(measurement_positions[1, :].astype(np.float32) / 1e10)
        coords_z_m = np.ascontiguousarray(measurement_positions[2, :].astype(np.float32) / 1e10)

        chunk_total = int(sample.chunk_total or 0)
        if chunk_total == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)

        # Precompute A(u,v) and global depth if requested
        A_beam_np = None
        s_min = s_max = None
        if use_depth_ein:
            A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=0)
            s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # cache key & dir (shared with GPU path for consistency)
        if use_depth_ein:
            key_obj = dict(
                E_eV=float(self._energy),
                lam=float(self._wavelength),
                direction=[float(x) for x in self._direction],
                stage_R=np.asarray(stage.rotation, dtype=float).round(7).tolist(),
                stage_T=[float(x) for x in np.asarray(stage.translation, dtype=float)],
                beam_size=[float(x) for x in self._beam_size],
                beam_samples=[int(self._beam_Ny), int(self._beam_Nz)],
                beam_profile=self._beam_profile,
                gauss_waist=[None if self._gauss_waist is None else float(self._gauss_waist[0]),
                            None if self._gauss_waist is None else float(self._gauss_waist[1])],
                s_min=float(s_min) if s_min is not None else None,
                s_max=float(s_max) if s_max is not None else None
            )
            key_hash = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode('utf-8')).hexdigest()
            cache_dir = ein_cache_dir or os.path.join(self.directory, "ein_cache")
            os.makedirs(cache_dir, exist_ok=True)
        else:
            key_hash = None
            cache_dir = None

        # Build an E_in sampler (CPU bilinear)
        if use_depth_ein:
            NyB, NzB = self._beam_Ny, self._beam_Nz
            du, dv = self._beam_du, self._beam_dv
            uc, vc = self._beam_uc, self._beam_vc
            e1 = self._beam_e1; e2 = self._beam_e2
            khat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
            E0_map = self._beam_E0_map.astype(np.complex64)

            def _ein_for_positions_cpu(pos_np):
                # pos_np in Å after stage transform
                au = pos_np[:, 0]*e1[0] + pos_np[:, 1]*e1[1] + pos_np[:, 2]*e1[2]
                av = pos_np[:, 0]*e2[0] + pos_np[:, 1]*e2[1] + pos_np[:, 2]*e2[2]
                iu = au / du + uc
                iv = av / dv + vc

                i0 = np.floor(iu).astype(np.int64); j0 = np.floor(iv).astype(np.int64)
                i1 = np.clip(i0 + 1, 0, NyB-1);     j1 = np.clip(j0 + 1, 0, NzB-1)
                i0 = np.clip(i0,       0, NyB-1);   j0 = np.clip(j0,       0, NzB-1)

                fu = (iu - i0).astype(np.float32); fv = (iv - j0).astype(np.float32)
                r00 = (i0 * NzB + j0); r01 = (i0 * NzB + j1)
                r10 = (i1 * NzB + j0); r11 = (i1 * NzB + j1)

                A00 = A_beam_np.ravel()[r00]; A01 = A_beam_np.ravel()[r01]
                A10 = A_beam_np.ravel()[r10]; A11 = A_beam_np.ravel()[r11]
                A_s = (A00 * (1.0 - fu)*(1.0 - fv) +
                    A01 * (1.0 - fu)*fv +
                    A10 * fu*(1.0 - fv) +
                    A11 * fu*fv).astype(np.complex64)

                E0_00 = E0_map.ravel()[r00]; E0_01 = E0_map.ravel()[r01]
                E0_10 = E0_map.ravel()[r10]; E0_11 = E0_map.ravel()[r11]
                E0_s = (E0_00 * (1.0 - fu)*(1.0 - fv) +
                        E0_01 * (1.0 - fu)*fv +
                        E0_10 * fu*(1.0 - fv) +
                        E0_11 * fu*fv).astype(np.complex64)

                # global depth fraction
                s = pos_np @ khat
                f = np.clip((s - s_min) / (s_max - s_min + 1e-12), 0.0, 1.0).astype(np.float32)

                # E_in = E0 * A^f   (robust pow; handle 0^0)
                tiny = 1e-12
                Ein = np.exp(np.log(A_s + 0j) * f) * E0_s
                mask00 = (np.abs(A_s) < tiny) & (f < tiny)
                Ein[mask00] = E0_s[mask00]
                return Ein.astype(np.complex64)

        # compile CPU CFFI kernel once
        ffi_obj, complied_code = self.compile_compute_scattering_cffi()

        # multi-thread across chunks
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_threads = min(chunk_total, multiprocessing.cpu_count())

        def worker(chunk_id):
            # Positions after stage, for E_in and for scattering
            pos_A = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
            if pos_A.size == 0:
                return np.zeros((Ny, Nx), dtype=np.complex64)

            pos_A = pos_A @ stage.rotation
            pos_A += stage.translation

            init_amp = None
            if use_depth_ein:
                cache_path = os.path.join(cache_dir, f"ein_chunk_{chunk_id}_{key_hash}.npz")
                if (not recompute_cache) and os.path.isfile(cache_path):
                    try:
                        with np.load(cache_path) as npz:
                            arr = npz["ein"]
                        if arr.shape[0] == pos_A.shape[0]:
                            init_amp = arr.astype(np.complex64, copy=False)
                    except Exception:
                        init_amp = None
                if init_amp is None:
                    init_amp = _ein_for_positions_cpu(pos_A)
                    try:
                        np.savez_compressed(cache_path, ein=init_amp)
                    except Exception:
                        pass

            return self.cpu_scatter_chunk_cffi(
                complied_code, ffi_obj, chunk_id, sample, Nx, Ny,
                coords_x_m, coords_y_m, coords_z_m,
                db_dict_f0_all, db_dict_f1f2_all, k_val, stage,
                detector, remove_forward_component,
                initial_amp_complex=init_amp
            )

        final_result = np.zeros((Ny, Nx), dtype=np.complex64)
        with ThreadPoolExecutor(max_workers=n_threads) as exe:
            futures = {exe.submit(worker, cid): cid for cid in range(1, chunk_total + 1)}
            for fut in as_completed(futures):
                final_result += fut.result()
        return final_result

    def interact_beam_gpu(
        self,
        sample,
        measurement_positions,
        measurement_shape,
        stage,
        remove_forward: bool = False,
        use_depth_ein: bool = False,
        ein_cache_dir: str | None = None,
        recompute_cache: bool = False
    ):
        if cp is None:
            print("[beam] CuPy not installed, falling back to CPU.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            print("[beam] No GPUs found, falling back to CPU.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage)

        import hashlib, json

        print(f"[beam] Found {n_gpus} GPU(s).")

        # DBs
        db_f0   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero = self._build_f0_zero_dict(db_f0)

        Nx, Ny = measurement_shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        # detector coords → meters (needed only for kernel geometry)
        x_coords = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        y_coords = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        z_coords = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)

        # stage
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        chunk_total = sample.chunk_total
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")

        # If depth-E_in: precompute beam-column A(u,v) once, and global depth bounds
        A_beam_np = None
        s_min = s_max = None
        if use_depth_ein:
            if cp is not None:
                A_beam_np = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius=0)
            else:
                A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=0)
            s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # cache key
        if use_depth_ein:
            key_obj = dict(
                E_eV=float(self._energy),
                lam=float(self._wavelength),
                direction=[float(x) for x in self._direction],
                stage_R=np.asarray(stage.rotation, dtype=float).round(7).tolist(),
                stage_T=[float(x) for x in np.asarray(stage.translation, dtype=float)],
                beam_size=[float(x) for x in self._beam_size],
                beam_samples=[int(self._beam_Ny), int(self._beam_Nz)],
                beam_profile=self._beam_profile,
                gauss_waist=[None if self._gauss_waist is None else float(self._gauss_waist[0]),
                            None if self._gauss_waist is None else float(self._gauss_waist[1])],
                s_min=float(s_min) if s_min is not None else None,
                s_max=float(s_max) if s_max is not None else None
            )
            key_hash = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode('utf-8')).hexdigest()
            cache_dir = ein_cache_dir or os.path.join(self.directory, "ein_cache")
            os.makedirs(cache_dir, exist_ok=True)
        else:
            key_hash = None
            cache_dir = None

        # split chunks
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus
        partial_results = [None] * n_gpus

        interaction_kernel = self.build_interaction_kernel()

        def gpu_worker(gpu_id, x_coords, y_coords, z_coords, chunk_indices, result_index):
            cp.cuda.Device(gpu_id).use()

            Rg = cp.asarray(R_pin, dtype=cp.float32)
            Tg = cp.asarray(T_pin, dtype=cp.float32)
            xg = cp.asarray(x_coords); yg = cp.asarray(y_coords); zg = cp.asarray(z_coords)

            dfield = cp.zeros((Nx * Ny,), dtype=cp.complex64)

            block = (16, 16)
            grid  = ((Nx + block[0] - 1) // block[0],
                    (Ny + block[1] - 1) // block[1])

            # move beam assets to device
            if use_depth_ein:
                A_gpu = cp.asarray(A_beam_np)              # (NyB,NzB)
                E0_gpu = cp.asarray(self._beam_E0_map)     # (NyB,NzB)
                NyB, NzB = A_gpu.shape
                du, dv = self._beam_du, self._beam_dv
                uc, vc = self._beam_uc, self._beam_vc
                du_g, dv_g = cp.float32(du), cp.float32(dv)
                uc_g, vc_g = cp.float32(uc), cp.float32(vc)
                e1g = cp.asarray(self._beam_e1); e2g = cp.asarray(self._beam_e2)
                khat = cp.asarray(self._direction / np.linalg.norm(self._direction), dtype=cp.float32)
                smin = cp.float32(s_min); smax = cp.float32(s_max)

            # helper: sample E0 and A at atom positions, compute E_in with global depth f
            def _ein_for_positions(pos_g):
                # (Å) positions already stage-transformed
                au = pos_g[:, 0]*e1g[0] + pos_g[:, 1]*e1g[1] + pos_g[:, 2]*e1g[2]
                av = pos_g[:, 0]*e2g[0] + pos_g[:, 1]*e2g[1] + pos_g[:, 2]*e2g[2]
                iu = au / du_g + uc_g
                iv = av / dv_g + vc_g

                i0 = cp.floor(iu).astype(cp.int64); j0 = cp.floor(iv).astype(cp.int64)
                i1 = cp.clip(i0 + 1, 0, NyB-1); j1 = cp.clip(j0 + 1, 0, NzB-1)
                i0 = cp.clip(i0, 0, NyB - 1);     j0 = cp.clip(j0, 0, NzB - 1)

                fu = (iu - i0).astype(cp.float32); fv = (iv - j0).astype(cp.float32)
                one = cp.float32(1.0)

                # bilinear sampling
                r00 = (i0 * NzB + j0).astype(cp.int64); r01 = (i0 * NzB + j1).astype(cp.int64)
                r10 = (i1 * NzB + j0).astype(cp.int64); r11 = (i1 * NzB + j1).astype(cp.int64)

                A00 = A_gpu.ravel()[r00]; A01 = A_gpu.ravel()[r01]
                A10 = A_gpu.ravel()[r10]; A11 = A_gpu.ravel()[r11]
                A_s = (A00 * (one - fu)*(one - fv) +
                    A01 * (one - fu)*fv +
                    A10 * fu*(one - fv) +
                    A11 * fu*fv).astype(cp.complex64)

                E0_00 = E0_gpu.ravel()[r00]; E0_01 = E0_gpu.ravel()[r01]
                E0_10 = E0_gpu.ravel()[r10]; E0_11 = E0_gpu.ravel()[r11]
                E0_s = (E0_00 * (one - fu)*(one - fv) +
                        E0_01 * (one - fu)*fv +
                        E0_10 * fu*(one - fv) +
                        E0_11 * fu*fv).astype(cp.complex64)

                # global depth fraction f
                s = pos_g[:, 0]*khat[0] + pos_g[:, 1]*khat[1] + pos_g[:, 2]*khat[2]
                f = cp.clip((s - smin) / (smax - smin + cp.float32(1e-12)), 0.0, 1.0).astype(cp.float32)

                # E_in = E0 * A^f  (robust log for complex; handle 0^0 → 1)
                tiny = cp.float32(1e-12)
                absA = cp.abs(A_s)
                Ein = cp.exp(cp.log(A_s + 0j) * f) * E0_s
                mask00 = (absA < tiny) & (f < tiny)
                Ein = cp.where(mask00, E0_s, Ein)
                return Ein

            for cidx in chunk_indices:
                spc = sample.load_chunk_species(cidx, use_gpu=False)
                nA = spc.shape[0]
                if nA == 0:
                    continue

                pos = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                pos = pos @ Rg; pos += Tg

                # per-atom scattering arrays
                s_anom = np.zeros(nA, np.complex64)
                f0p    = np.zeros((nA, 11), np.float32)
                f0z    = np.zeros(nA, np.float32)
                for el in pd.unique(spc):
                    if el not in db_f0:
                        continue
                    m = (spc == el)
                    f0p[m] = db_f0[el]
                    f0z[m] = f0_zero.get(el, 0.0)
                    # if not remove_forward: Old zeroing code
                    tbl = db_f1f2.get(el)
                    if tbl is not None:
                        s_anom[m] = self.get_f1f2_from_params(self._energy, tbl)

                # per-atom initial amplitude
                if use_depth_ein:
                    cache_path = os.path.join(cache_dir, f"ein_chunk_{cidx}_{key_hash}.npz")
                    initial_amp = None
                    if (not recompute_cache) and os.path.isfile(cache_path):
                        try:
                            with np.load(cache_path) as npz:
                                arr = npz["ein"]
                            if arr.shape[0] == nA:
                                initial_amp = cp.asarray(arr.astype(np.complex64))
                        except Exception:
                            initial_amp = None
                    if initial_amp is None:
                        initial_amp = _ein_for_positions(pos)
                        try:
                            np.savez_compressed(cache_path, ein=initial_amp.get())
                        except Exception:
                            pass
                else:
                    initial_amp = cp.ones(nA, dtype=cp.complex64)

                # positions to meters for kernel
                px = pos[:, 0] / 1e10; py = pos[:, 1] / 1e10; pz = pos[:, 2] / 1e10

                kx_cp = cp.full(nA, self._kx_scalar, dtype=cp.float32)
                ky_cp = cp.full(nA, self._ky_scalar, dtype=cp.float32)
                kz_cp = cp.full(nA, self._kz_scalar, dtype=cp.float32)

                s_anom_cp = cp.asarray(s_anom)
                f0_params_cp = cp.asarray(f0p)
                f0_zero_cp   = cp.asarray(f0z)

                interaction_kernel(
                    grid, block,
                    (
                        np.int32(nA),
                        kx_cp, ky_cp, kz_cp,
                        px, py, pz,
                        initial_amp,
                        s_anom_cp,
                        f0_params_cp,
                        f0_zero_cp,
                        xg, yg, zg,
                        dfield,
                        np.int32(Nx),
                        np.int32(Ny),
                        np.int32(1 if remove_forward else 0)
                    )
                )
                cp.get_default_memory_pool().free_all_blocks()

            partial_results[result_index] = dfield.reshape((Ny, Nx)).get()
            del xg, yg, zg
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # launch one thread per GPU
        threads = []
        start_chunk = 1
        for gid in range(n_gpus):
            my_count = chunks_per_gpu + (1 if gid < remainder else 0)
            end_chunk = start_chunk + my_count
            chunk_indices = list(range(start_chunk, end_chunk))
            start_chunk = end_chunk
            t = threading.Thread(target=gpu_worker,
                                args=(gid, x_coords, y_coords, z_coords, chunk_indices, gid))
            t.start(); threads.append(t)

        for t in threads: t.join()

        for pr in partial_results:
            if pr is not None:
                final_result += pr

        return final_result
    
    def atomic_scattering_kinematic(
        self,
        sample,
        detector,
        stage,
        offset=None,
        use_gpu=True,
        remove_forward: bool = False,
        use_depth_ein: bool = False,
        ein_cache_dir: str | None = None,
        recompute_cache: bool = False
    ):
        measurement_positions = detector.pixel_coordinates
        Nx, Ny = detector.shape

        if use_gpu and (cp is not None):
            final_field = self.interact_beam_gpu(
                sample,
                measurement_positions,
                (Nx, Ny),
                stage,
                remove_forward=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache
            )
        else:
            if cp is None and use_gpu:
                print("[beam] Cupy not installed, running CPU mode.")
            final_field = self.interact_beam_cpu(
                sample,
                measurement_positions,
                (Nx, Ny),
                stage,
                detector=None,
                remove_forward_component=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache
            )

        return (final_field - offset) if (offset is not None) else final_field
    # -------------------------------------
        
    # -------------------------------------
    # Direct transmission
    def _compute_beam_column_A_map_cpu(self, sample, stage, kernel_radius=0):
        """
        CPU fallback for A(u,v) on the beam grid. Returns np.complex64 (Ny,Nz).
        """
        # constants (Å)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10
        du, dv = self._beam_du, self._beam_dv
        NyB, NzB = self._beam_Ny, self._beam_Nz
        A_pix_A2 = float(du) * float(dv)
        scale = (r_e_A * lam_A) / A_pix_A2

        tau = np.zeros((NyB, NzB), np.float32)
        phi = np.zeros((NyB, NzB), np.float32)

        f1f2_dict = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        e1 = self._beam_e1; e2 = self._beam_e2

        def _tsc_w(d):
            w = np.zeros_like(d, dtype=np.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0]*d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        for cid in range(1, sample.chunk_total + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)  # Å
            if pos.size == 0:
                continue
            pos = pos @ stage.rotation
            pos += stage.translation

            nA = pos.shape[0]
            f1 = np.zeros(nA, np.float32)
            f2 = np.zeros(nA, np.float32)
            for el in pd.unique(spc):
                tbl = f1f2_dict.get(str(el))
                if tbl is None:
                    continue
                cplx = self.get_f1f2_from_params(self._energy, tbl)
                m = (spc == el)
                f1[m] = float(cplx.real)
                f2[m] = float(cplx.imag)

            au = pos[:, 0]*e1[0] + pos[:, 1]*e1[1] + pos[:, 2]*e1[2]
            av = pos[:, 0]*e2[0] + pos[:, 1]*e2[1] + pos[:, 2]*e2[2]
            iu = au / du + self._beam_uc
            iv = av / dv + self._beam_vc

            ic = np.floor(iu + 0.5).astype(np.int64)
            jc = np.floor(iv + 0.5).astype(np.int64)

            du_m1 = np.abs(iu - (ic - 1)); du_0 = np.abs(iu - ic); du_p1 = np.abs(iu - (ic + 1))
            dv_m1 = np.abs(iv - (jc - 1)); dv_0 = np.abs(iv - jc); dv_p1 = np.abs(iv - (jc + 1))

            wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
            wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

            w_phi_atom = (-scale * f1).astype(np.float32)
            w_tau_atom = ( scale * f2).astype(np.float32)

            idx_phi = []; w_phi = []
            idx_tau = []; w_tau = []

            def _push(ii, jj, fac, w_atom):
                inb = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                if not np.any(inb):
                    return np.empty((0,), np.int64), np.empty((0,), np.float32)
                rows = ii[inb]; cols = jj[inb]
                idx = (rows * NzB + cols).astype(np.int64)
                w = (w_atom[inb] * fac[inb]).astype(np.float32)
                return idx, w

            for dx, wx in [(-1, wu_m1), (0, wu_0), (1, wu_p1)]:
                ii = ic + dx
                for dy, wy in [(-1, wv_m1), (0, wv_0), (1, wv_p1)]:
                    jj = jc + dy
                    fac = wx * wy
                    idx, w = _push(ii, jj, fac, w_phi_atom); idx_phi.append(idx); w_phi.append(w)
                    idx, w = _push(ii, jj, fac, w_tau_atom); idx_tau.append(idx); w_tau.append(w)

            if idx_phi:
                idxp = np.concatenate(idx_phi); wp = np.concatenate(w_phi)
                idxt = np.concatenate(idx_tau); wt = np.concatenate(w_tau)
                # accumulate
                np.add.at(phi.ravel(), idxp, wp)
                np.add.at(tau.ravel(), idxt, wt)

        if kernel_radius > 0:
            rad = int(kernel_radius); sig = rad / 2.0
            y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
            k = np.exp(-(x*x + y*y) / (2.0*sig*sig)).astype(np.float32)
            k /= k.sum()
            # FFT conv (same size)
            Fk = np.fft.fft2(k, s=tau.shape)
            tau = np.fft.ifft2(np.fft.fft2(tau) * Fk).real.astype(np.float32)
            phi = np.fft.ifft2(np.fft.fft2(phi) * Fk).real.astype(np.float32)

        A_map = np.exp(-tau + 1j*phi).astype(np.complex64)
        return A_map

    def _compute_beam_column_A_map_gpu(self, sample, stage, kernel_radius=0):
        """
        Full-column transmission A(u,v)=exp(-tau + i*phi) on the beam grid (Ny_beam, Nz_beam).
        High-throughput GPU (multi-GPU aware via chunk loop). Returns np.complex64 (Ny,Nz).
        """
        if cp is None:
            return self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            return self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        # constants (Å)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10
        du, dv = self._beam_du, self._beam_dv
        NyB, NzB = self._beam_Ny, self._beam_Nz
        A_pix_A2 = float(du) * float(dv)
        scale = (r_e_A * lam_A) / A_pix_A2

        # pinned stage
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        f1f2_dict = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")

        partial = [None] * n_gpus
        chunks_per_gpu = sample.chunk_total // n_gpus
        remainder = sample.chunk_total % n_gpus

        # GPU worker
        def worker(dev_id, chunks, out_idx):
            cp.cuda.Device(dev_id).use()
            Rg = cp.asarray(R_pin); Tg = cp.asarray(T_pin)

            tau_acc = cp.zeros((NyB, NzB), dtype=cp.float32)
            phi_acc = cp.zeros((NyB, NzB), dtype=cp.float32)

            # TSC 1D kernel
            def _tsc_w(d):
                w = cp.zeros_like(d, dtype=cp.float32)
                m0 = d <= 0.5
                w[m0] = 0.75 - d[m0] * d[m0]
                m1 = (~m0) & (d <= 1.5)
                t = 1.5 - d[m1]
                w[m1] = 0.5 * t * t
                return w

            for cid in chunks:
                spc = sample.load_chunk_species(cid, use_gpu=False)
                pos = sample.load_chunk_positions(cid, use_gpu=False)  # Å
                nA = pos.shape[0]
                if nA == 0:
                    continue

                # per-atom f1,f2
                f1 = np.zeros(nA, np.float32)
                f2 = np.zeros(nA, np.float32)
                for el in pd.unique(spc):
                    tbl = f1f2_dict.get(str(el))
                    if tbl is None:
                        continue
                    cplx = self.get_f1f2_from_params(self._energy, tbl)
                    m = (spc == el)
                    f1[m] = float(cplx.real)
                    f2[m] = float(cplx.imag)

                f1g = cp.asarray(f1); f2g = cp.asarray(f2)
                posg = cp.asarray(pos, dtype=cp.float32)
                posg = posg @ Rg; posg += Tg

                # project onto (u,v)
                e1g = cp.asarray(self._beam_e1); e2g = cp.asarray(self._beam_e2)
                au = posg[:, 0]*e1g[0] + posg[:, 1]*e1g[1] + posg[:, 2]*e1g[2]
                av = posg[:, 0]*e2g[0] + posg[:, 1]*e2g[1] + posg[:, 2]*e2g[2]

                # continuous grid indices (center at uc/vc)
                iu = au / self._beam_du + self._beam_uc
                iv = av / self._beam_dv + self._beam_vc

                ic = cp.floor(iu + 0.5).astype(cp.int64)
                jc = cp.floor(iv + 0.5).astype(cp.int64)

                du_m1 = cp.abs(iu - (ic - 1)); du_0 = cp.abs(iu - ic); du_p1 = cp.abs(iu - (ic + 1))
                dv_m1 = cp.abs(iv - (jc - 1)); dv_0 = cp.abs(iv - jc); dv_p1 = cp.abs(iv - (jc + 1))

                wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
                wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

                w_phi_atom = (-scale * f1g).astype(cp.float32)
                w_tau_atom = ( scale * f2g).astype(cp.float32)

                idx_phi = []; w_phi = []
                idx_tau = []; w_tau = []

                # rows = u-index in [0..NyB-1], cols = v-index in [0..NzB-1]
                def _push(ii, jj, fac, w_atom):
                    inb = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                    if not bool(cp.any(inb)):
                        return cp.empty((0,), cp.int64), cp.empty((0,), cp.float32)
                    rows = ii[inb]; cols = jj[inb]
                    idx = (rows * NzB + cols).astype(cp.int64)
                    w = (w_atom[inb] * fac[inb]).astype(cp.float32)
                    return idx, w

                for dx, wx in [(-1, wu_m1), (0, wu_0), (1, wu_p1)]:
                    ii = ic + dx
                    for dy, wy in [(-1, wv_m1), (0, wv_0), (1, wv_p1)]:
                        jj = jc + dy
                        fac = wx * wy
                        idx, w = _push(ii, jj, fac, w_phi_atom); idx_phi.append(idx); w_phi.append(w)
                        idx, w = _push(ii, jj, fac, w_tau_atom); idx_tau.append(idx); w_tau.append(w)

                if idx_phi:
                    idxp = cp.concatenate(idx_phi); wp = cp.concatenate(w_phi)
                    idxt = cp.concatenate(idx_tau); wt = cp.concatenate(w_tau)
                    bins = int(NyB) * int(NzB)
                    phi_hist = self._safe_bincount_gpu(idxp, wp, bins, dtype=cp.float32)
                    tau_hist = self._safe_bincount_gpu(idxt, wt, bins, dtype=cp.float32)
                    phi_acc += phi_hist.reshape(NyB, NzB)
                    tau_acc += tau_hist.reshape(NyB, NzB)

                cp.get_default_memory_pool().free_all_blocks()

            # optional blur
            if kernel_radius > 0:
                rad = int(kernel_radius); sig = rad / 2.0
                yg = cp.arange(-rad, rad + 1, dtype=cp.float32)[:, None]
                xg = cp.arange(-rad, rad + 1, dtype=cp.float32)[None, :]
                kg = cp.exp(-(xg * xg + yg * yg) / (2.0 * sig * sig))
                kg /= cp.sum(kg)
                Fk = cp.fft.fft2(kg, tau_acc.shape)
                tau_acc = cp.fft.ifft2(cp.fft.fft2(tau_acc) * Fk).real
                phi_acc = cp.fft.ifft2(cp.fft.fft2(phi_acc) * Fk).real

            A_gpu = cp.exp(-tau_acc + 1j * phi_acc).astype(cp.complex64)
            partial[out_idx] = A_gpu.get()
            cp.get_default_memory_pool().free_all_blocks()

        # launch
        threads = []
        start = 1
        for gid in range(n_gpus):
            n_chunk = chunks_per_gpu + (1 if gid < remainder else 0)
            end = start + n_chunk
            t = threading.Thread(target=worker, args=(gid, range(start, end), gid))
            t.start(); threads.append(t)
            start = end
        for t in threads: t.join()

        # reduce
        A_total = np.ones((self._beam_Ny, self._beam_Nz), np.complex64)
        for p in partial:
            if p is not None:
                A_total *= p  # chunks are disjoint; summation is already combined via bincount
        return A_total
    
    def atomic_transmission(self, sample, detector, stage, use_gpu=True, kernel_radius=0):
        """
        Compute transmitted field on the beam grid, then resample to detector pixels.

        Steps:
        1) A(u,v) on beam grid via full-column TSC deposition (GPU/CPU).
        2) E_exit(u,v) = E0(u,v) * A(u,v)   on the beam grid.
        3) Resample E_exit(u,v) to detector pixel centers (bilinear), return (Ny_det,Nx_det).
        """
        # 1) full-column transmission A(u,v)
        if use_gpu and (cp is not None):
            A_beam = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius)
        else:
            A_beam = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        # 2) Exit field on beam grid
        E_beam = (self._beam_E0_map * A_beam).astype(np.complex64)  # (NyB,NzB)
        NyB, NzB = E_beam.shape
        du, dv = self._beam_du, self._beam_dv
        uc, vc = self._beam_uc, self._beam_vc

        # 3) Resample to detector pixels
        NyD, NxD = detector.shape
        pix = detector.pixel_coordinates
        if cp is not None and use_gpu:
            # move to GPU
            if not isinstance(pix, cp.ndarray):
                pix = cp.asarray(pix)
            e1g = cp.asarray(self._beam_e1); e2g = cp.asarray(self._beam_e2)
            u = pix[0]*e1g[0] + pix[1]*e1g[1] + pix[2]*e1g[2]
            v = pix[0]*e2g[0] + pix[1]*e2g[1] + pix[2]*e2g[2]
            iu = u / du + uc
            iv = v / dv + vc

            i0 = cp.floor(iu).astype(cp.int64); j0 = cp.floor(iv).astype(cp.int64)
            i1 = i0 + 1; j1 = j0 + 1
            fu = (iu - i0).astype(cp.float32); fv = (iv - j0).astype(cp.float32)

            i0 = cp.clip(i0, 0, NyB - 1); i1 = cp.clip(i1, 0, NyB - 1)
            j0 = cp.clip(j0, 0, NzB - 1); j1 = cp.clip(j1, 0, NzB - 1)

            E_gpu = cp.asarray(E_beam)

            idx00 = (i0 * NzB + j0).astype(cp.int64)
            idx01 = (i0 * NzB + j1).astype(cp.int64)
            idx10 = (i1 * NzB + j0).astype(cp.int64)
            idx11 = (i1 * NzB + j1).astype(cp.int64)

            E00 = E_gpu.ravel()[idx00]
            E01 = E_gpu.ravel()[idx01]
            E10 = E_gpu.ravel()[idx10]
            E11 = E_gpu.ravel()[idx11]

            one = cp.float32(1.0)
            E_flat = (E00 * (one - fu)*(one - fv) +
                    E01 * (one - fu)*fv +
                    E10 * (fu)*(one - fv) +
                    E11 * (fu)*fv).astype(cp.complex64)
            E_det = E_flat.reshape(NyD, NxD).get()
        else:
            pix = np.asarray(pix)
            e1 = self._beam_e1; e2 = self._beam_e2
            u = pix[0]*e1[0] + pix[1]*e1[1] + pix[2]*e1[2]
            v = pix[0]*e2[0] + pix[1]*e2[1] + pix[2]*e2[2]
            iu = u / du + uc
            iv = v / dv + vc

            i0 = np.floor(iu).astype(np.int64); j0 = np.floor(iv).astype(np.int64)
            i1 = i0 + 1; j1 = j0 + 1
            fu = (iu - i0).astype(np.float32); fv = (iv - j0).astype(np.float32)

            i0 = np.clip(i0, 0, NyB - 1); i1 = np.clip(i1, 0, NyB - 1)
            j0 = np.clip(j0, 0, NzB - 1); j1 = np.clip(j1, 0, NzB - 1)

            Eb = E_beam.ravel()
            idx00 = (i0 * NzB + j0).astype(np.int64)
            idx01 = (i0 * NzB + j1).astype(np.int64)
            idx10 = (i1 * NzB + j0).astype(np.int64)
            idx11 = (i1 * NzB + j1).astype(np.int64)

            E00 = Eb[idx00]; E01 = Eb[idx01]; E10 = Eb[idx10]; E11 = Eb[idx11]
            E_flat = (E00 * (1.0 - fu)*(1.0 - fv) +
                    E01 * (1.0 - fu)*fv +
                    E10 * fu*(1.0 - fv) +
                    E11 * fu*fv).astype(np.complex64)
            E_det = E_flat.reshape(NyD, NxD)

        # Return complex transmission map on detector (same convention as before)
        return E_det.astype(np.complex64)
    # -------------------------------------
    
    # -------------------------------------
    # Dynamical scattering
    def compute_intra_chunk_neighbors_gpu(
        self,
        sample,
        positions,           # cp.ndarray (N,3) in Å
        r_cut=5.0,
        max_neighbors_per_atom=32
    ):
        """
        Intra-chunk neighbor search with Å-consistent k and wavelength.
        """
        N = positions.shape[0]
        if N == 0:
            return [
                (np.array([], dtype=np.float32),
                np.zeros((0,3), dtype=np.float32),
                np.array([], dtype=np.int32))
                for _ in range(N)
            ]

        # 1) Cell list
        (sorted_positions,
        sorted_indices,
        cell_start,
        cell_end,
        box_min,
        cell_size,
        nx, ny, nz) = sample.build_cell_list_gpu(positions, r_cut)

        # 2) Buffers
        phase_gpu  = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        kx_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        ky_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        kz_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        idx_gpu    = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.int32)
        counts_gpu = cp.zeros((N,), dtype=cp.int32)

        # ---- Å UNITS ----
        wavelength_A = self._wavelength * 1e10         # meters -> Å
        k_val_A      = (2.0 * np.pi) / wavelength_A    # 1/Å

        # 3) Kernel launch
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
                cp.float32(k_val_A),
                cp.float32(wavelength_A),
                phase_gpu,
                kx_gpu,
                ky_gpu,
                kz_gpu,
                idx_gpu,
                counts_gpu,
                np.int32(N)
            )
        )

        # 4) To CPU ragged
        phase_arr  = phase_gpu.reshape(N, max_neighbors_per_atom).get()
        kx_arr     = kx_gpu.reshape(N, max_neighbors_per_atom).get()
        ky_arr     = ky_gpu.reshape(N, max_neighbors_per_atom).get()
        kz_arr     = kz_gpu.reshape(N, max_neighbors_per_atom).get()
        idx_arr    = idx_gpu.reshape(N, max_neighbors_per_atom).get()
        counts_arr = counts_gpu.get()
        sorted_idx_arr = sorted_indices.get()

        output = [None]*N
        for sorted_i in range(N):
            orig_i = sorted_idx_arr[sorted_i]
            used = min(counts_arr[sorted_i], max_neighbors_per_atom)
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

            kvec_sub = np.vstack([kx_sub, ky_sub, kz_sub]).T
            output[orig_i] = (ph_sub, kvec_sub, idx_sub)

        return output

    def compute_inter_chunk_neighbors_gpu(
        self,
        sample,
        pos_i,         # cp.ndarray (N_i,3) in Å
        pos_j,         # cp.ndarray (N_j,3) in Å
        r_cut,
        max_neighbors_per_atom=32
    ):
        """
        Cross-chunk neighbor search with Å-consistent k and wavelength.
        """
        N_i = pos_i.shape[0]
        N_j = pos_j.shape[0]
        if N_i == 0 and N_j == 0:
            return []
        if N_i == 0 or N_j == 0:
            return [(np.array([], dtype=np.float32),
                    np.zeros((0,3), dtype=np.float32),
                    np.array([], dtype=np.int32))
                    for _ in range(N_i + N_j)]

        pos_comb = cp.concatenate([pos_i, pos_j], axis=0)
        N_total  = N_i + N_j

        (sorted_positions,
        sorted_indices,
        cell_start,
        cell_end,
        box_min,
        cell_size,
        nx, ny, nz) = sample.build_cell_list_gpu(pos_comb, r_cut)

        phase_gpu  = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        kx_gpu     = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        ky_gpu     = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        kz_gpu     = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.float32)
        idx_gpu    = cp.zeros((N_total*max_neighbors_per_atom,), dtype=cp.int32)
        counts_gpu = cp.zeros((N_total,), dtype=cp.int32)

        # ---- Å UNITS ----
        wavelength_A = self._wavelength * 1e10
        k_val_A      = (2.0 * np.pi)/wavelength_A

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
                cp.float32(k_val_A),
                cp.float32(wavelength_A),
                phase_gpu,
                kx_gpu,
                ky_gpu,
                kz_gpu,
                idx_gpu,
                counts_gpu
            )
        )

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
            used = min(counts_arr[sorted_i], max_neighbors_per_atom)
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
                sample.write_chunk_nn_species([], cid)
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
            species_list  = []

            ### We build an augmented data structure (ph, kvec, idx, spc) ###
            results_intra_with_spc = [None] * n_atoms

            for i_atom, (ph, kvec_3, n_idx) in enumerate(results_intra):
                # The neighbor species for this atom come from chunk_species[n_idx].
                # n_idx is a 1D array of neighbor indices in the same chunk.
                n_spc = chunk_species[n_idx]  # CPU side indexing

                phase_list.append(ph.astype(np.float32))
                kvector_list.append(kvec_3.astype(np.float32))
                idx_list.append(n_idx.astype(np.int32))
                species_list.append(n_spc)  # store neighbor species array

                # Also store internally for pass B merges
                results_intra_with_spc[i_atom] = (ph, kvec_3, n_idx, n_spc)

            sample.write_chunk_nn_phase(phase_list, cid)
            sample.write_chunk_nn_scatter(kvector_list, cid)  # still using 'scatter' slot
            sample.write_chunk_nn_indices(idx_list, cid)
            sample.write_chunk_nn_species(species_list, cid)

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
        (unchanged docstring) — only the bounding-box quick reject now wraps CuPy
        reductions with bool(...).
        """
        # Build bounding boxes
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
            i_data = all_data_memory[i]
            pos_i  = i_bd["positions"]
            idx_i  = i_bd["indices"]
            spc_i  = i_bd["species"]
            if pos_i.size == 0:
                continue
            N_i = pos_i.shape[0]
            min_i, max_i = chunk_bounds[i]

            for j in range(i+1, sample.chunk_total+1):
                j_bd   = boundary_dict[j]
                j_data = all_data_memory[j]
                pos_j  = j_bd["positions"]
                idx_j  = j_bd["indices"]
                spc_j  = j_bd["species"]
                if pos_j.size == 0:
                    continue
                N_j = pos_j.shape[0]
                min_j, max_j = chunk_bounds[j]

                if (min_i is None) or (min_j is None):
                    continue

                # ---- bool(...) guards ----
                sep_ij = ((max_i + r_cut) < (min_j - r_cut)).any()
                sep_ji = ((max_j + r_cut) < (min_i - r_cut)).any()
                if bool(sep_ij) or bool(sep_ji):
                    continue

                # ... (rest of your original function body is unchanged) ...
                cross_list = self.compute_inter_chunk_neighbors_gpu(
                    sample, pos_i, pos_j, r_cut=r_cut,
                    max_neighbors_per_atom=max_neighbors_per_atom
                )

                idx_i_cpu = idx_i.get()
                idx_j_cpu = idx_j.get()

                for local_i in range(N_i):
                    (ph_new, kvec_new, idx_new) = cross_list[local_i]
                    if ph_new.size > 0:
                        global_i = idx_i_cpu[local_i]
                        (ph_old, kvec_old, idx_old, spc_old) = i_data[global_i]
                        spc_new = np.array([spc_j[n - N_i] if n >= N_i else spc_i[n]
                                            for n in idx_new], dtype=spc_old.dtype)
                        i_data[global_i] = (
                            np.concatenate([ph_old, ph_new]),
                            np.vstack([kvec_old, kvec_new]),
                            np.concatenate([idx_old, idx_new]),
                            np.concatenate([spc_old, spc_new])
                        )

                for local_j in range(N_j):
                    (ph_new, kvec_new, idx_new) = cross_list[N_i + local_j]
                    if ph_new.size > 0:
                        global_j = idx_j_cpu[local_j]
                        (ph_old, kvec_old, idx_old, spc_old) = j_data[global_j]
                        spc_new = np.array([spc_i[n] if n < N_i else spc_j[n - N_i]
                                            for n in idx_new], dtype=spc_old.dtype)
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
                species_list.append(spc_arr)  # keep user dtype or cast if needed

            sample.write_chunk_nn_phase(phase_list, cid)
            sample.write_chunk_nn_scatter(kvector_list, cid)  # re-use "scatter" slot for k vectors
            sample.write_chunk_nn_indices(idx_list, cid)
            sample.write_chunk_nn_species(species_list, cid)

        print(f"[beam] Completed nearest-neighbor calculation with cutoff={r_cut} "
            f"for {sample.chunk_total} chunks (GPU).")
        
    def atomic_scattering_dynamical(self, sample, detector, stage,
                                    n_bounces=0, offset=None, use_gpu=True,
                                    sub_chunk_size=100_000):
        """
        Multi-bounce GPU scattering. Now:
        • Passes f0_zero and remove_forward to interaction_kernal (fix).
        • Builds per-path f0_params / anomalous / f0(0) from stored neighbor species (fix).
        • Removes unused args from expand_paths_kernel launch (fix).
        """
        if (not use_gpu) or (cp is None):
            raise RuntimeError("GPU-based dynamical code requires CuPy and use_gpu=True.")

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            raise RuntimeError("No GPUs found for dynamical scattering.")

        chunk_total = sample.chunk_total
        if not chunk_total:
            final_result = np.zeros(detector.shape[::-1], dtype=np.complex64)
            if offset is not None:
                final_result -= offset
            return final_result

        print(f"[beam] Using GPU dynamical scattering with up to {n_bounces} bounce(s).")
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")

        # Scattering DBs
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero_map      = self._build_f0_zero_dict(db_dict_f0_all)

        Nx, Ny = detector.shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        # Detector coords (m) pinned
        mp = detector.pixel_coordinates
        px_pin = self.allocate_pinned_array(mp[0, :].astype(np.float32) / 1e10)
        py_pin = self.allocate_pinned_array(mp[1, :].astype(np.float32) / 1e10)
        pz_pin = self.allocate_pinned_array(mp[2, :].astype(np.float32) / 1e10)

        # Stage pinned
        R_stage_pin = self.allocate_pinned_array(stage.rotation)
        T_stage_pin = self.allocate_pinned_array(stage.translation)

        chunk_per_gpu = chunk_total // n_gpus
        remainder     = chunk_total % n_gpus
        partial_results = [None] * n_gpus

        interaction_kernel = self.build_interaction_kernel()
        expand_kernel      = self.build_expand_paths_kernel()

        # We don't expose a remove_forward toggle for dynamical; keep it off by default.
        remove_forward_flag = 0

        def gpu_worker(gpu_id, chunk_list, out_idx):
            cp.cuda.Device(gpu_id).use()

            # Stage & detector on this GPU
            Rg = cp.asarray(R_stage_pin, dtype=cp.float32)
            Tg = cp.asarray(T_stage_pin, dtype=cp.float32)
            pxg = cp.asarray(px_pin); pyg = cp.asarray(py_pin); pzg = cp.asarray(pz_pin)

            dfield_gpu = cp.zeros((Nx * Ny,), dtype=cp.complex64)

            block2d = (16, 16)
            grid2d  = ((Nx + block2d[0] - 1) // block2d[0],
                    (Ny + block2d[1] - 1) // block2d[1])

            block1d = 256

            # ----- helper: resolve species id -> element symbol -----
            def resolve_symbol(spc_id):
                # if it's already a string-like symbol
                if isinstance(spc_id, (str, np.str_)):
                    return str(spc_id)
                # try common mappings
                if hasattr(sample, "get_symbol_from_id"):
                    try: return str(sample.get_symbol_from_id(int(spc_id)))
                    except Exception: pass
                if hasattr(sample, "species_id_map"):
                    try: return str(sample.species_id_map[int(spc_id)])
                    except Exception: pass
                if hasattr(sample, "species_lookup"):
                    try: return str(sample.species_lookup[int(spc_id)])
                    except Exception: pass
                return None  # unknown => contributes zero

            # ----- helper: final scattering for a sub-chunk of expanded paths -----
            def process_subchunk(start_idx, end_idx,
                                out_x_gpu, out_y_gpu, out_z_gpu,
                                out_kx_gpu, out_ky_gpu, out_kz_gpu,
                                out_amp_gpu, out_spc_gpu, sub_sz):

                sub_x   = out_x_gpu[start_idx:end_idx]
                sub_y   = out_y_gpu[start_idx:end_idx]
                sub_z   = out_z_gpu[start_idx:end_idx]
                sub_kx  = out_kx_gpu[start_idx:end_idx]
                sub_ky  = out_ky_gpu[start_idx:end_idx]
                sub_kz  = out_kz_gpu[start_idx:end_idx]
                sub_amp = out_amp_gpu[start_idx:end_idx]
                sub_spc = out_spc_gpu[start_idx:end_idx].get()

                # Build per-path scattering arrays
                f0_local   = np.zeros((sub_sz, 11), dtype=np.float32)
                f0z_local  = np.zeros((sub_sz,),    dtype=np.float32)
                anom_local = np.zeros((sub_sz,),    dtype=np.complex64)

                # Fill by species
                for i in range(sub_sz):
                    sym = resolve_symbol(sub_spc[i])
                    if not sym:
                        continue
                    f0p = db_dict_f0_all.get(sym)
                    if f0p is not None:
                        f0_local[i, :] = f0p
                        f0z_local[i]   = f0_zero_map.get(sym, 0.0)
                    tbl = db_dict_f1f2_all.get(sym)
                    if tbl is not None:
                        anom_local[i] = self.get_f1f2_from_params(self._energy, tbl)

                f0_params_gpu = cp.asarray(f0_local)
                f0_zero_gpu   = cp.asarray(f0z_local)
                anom_gpu      = cp.asarray(anom_local)

                # Launch interaction kernel
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(sub_sz),
                        sub_kx, sub_ky, sub_kz,
                        sub_x,  sub_y,  sub_z,
                        sub_amp,
                        anom_gpu,
                        f0_params_gpu,
                        f0_zero_gpu,
                        pxg, pyg, pzg,
                        dfield_gpu,
                        np.int32(Nx),
                        np.int32(Ny),
                        np.int32(remove_forward_flag)
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()

            # ---- per-chunk processing ----
            for cidx in chunk_list:
                spc_host = sample.load_chunk_species(cidx, use_gpu=False)
                nA = int(spc_host.shape[0])
                if nA == 0:
                    continue

                pos = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                pos = pos @ Rg; pos += Tg

                px_at = pos[:, 0] / 1e10
                py_at = pos[:, 1] / 1e10
                pz_at = pos[:, 2] / 1e10

                # Build per-atom scattering for bounce=0
                f0_params_host = np.zeros((nA, 11), dtype=np.float32)
                anom_host      = np.zeros((nA,),    dtype=np.complex64)
                f0z_host       = np.zeros((nA,),    dtype=np.float32)

                unique_els = pd.unique(spc_host)
                for el in unique_els:
                    mask = (spc_host == el)
                    if el in db_dict_f0_all:
                        f0_params_host[mask] = db_dict_f0_all[el]
                        f0z_host[mask] = f0_zero_map.get(el, 0.0)
                    if el in db_dict_f1f2_all:
                        anom_host[mask] = self.get_f1f2_from_params(self._energy, db_dict_f1f2_all[el])

                f0_params_gpu = cp.asarray(f0_params_host)
                anom_gpu      = cp.asarray(anom_host)
                f0_zero_gpu   = cp.asarray(f0z_host)

                # initial wavevector = beam
                kx_atom_gpu = cp.full((nA,), self._kx_scalar, dtype=cp.float32)
                ky_atom_gpu = cp.full((nA,), self._ky_scalar, dtype=cp.float32)
                kz_atom_gpu = cp.full((nA,), self._kz_scalar, dtype=cp.float32)
                amp_atom_gpu= cp.ones((nA,), dtype=cp.complex64)

                # Direct scattering (bounce 0)
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(nA),
                        kx_atom_gpu, ky_atom_gpu, kz_atom_gpu,
                        px_at, py_at, pz_at,
                        amp_atom_gpu,
                        anom_gpu,
                        f0_params_gpu,
                        f0_zero_gpu,
                        pxg, pyg, pzg,
                        dfield_gpu,
                        np.int32(Nx),
                        np.int32(Ny),
                        np.int32(remove_forward_flag)
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()

                if n_bounces < 1:
                    cp.get_default_memory_pool().free_all_blocks()
                    continue

                # Load precomputed neighbor data (phase, kvec, indices, species)
                ph_flat,  offs_ph = sample.load_chunk_nn_phase(cidx)
                kx_flat, ky_flat, kz_flat, offs_kv = sample.load_chunk_nn_scatter(cidx)
                idx_flat, offs_ix = sample.load_chunk_nn_indices(cidx)
                spc_flat, offs_sp = sample.load_chunk_nn_species(cidx)

                neighborPhase_gpu = cp.asarray(ph_flat,  dtype=cp.float32)
                neighborKx_gpu    = cp.asarray(kx_flat,  dtype=cp.float32)
                neighborKy_gpu    = cp.asarray(ky_flat,  dtype=cp.float32)
                neighborKz_gpu    = cp.asarray(kz_flat,  dtype=cp.float32)
                neighborIdx_gpu   = cp.asarray(idx_flat, dtype=cp.int32)
                neighborSpc_gpu   = cp.asarray(spc_flat)  # dtype may be int or string codes -> kept as is

                neighborStart_gpu  = cp.asarray(offs_ph[:-1].astype(np.int32))
                neighborCount_gpu  = cp.asarray((offs_ph[1:] - offs_ph[:-1]).astype(np.int32))

                # Inputs for first expansion step
                cur_size   = nA
                in_x_gpu   = px_at.copy()
                in_y_gpu   = py_at.copy()
                in_z_gpu   = pz_at.copy()
                in_kx_gpu  = kx_atom_gpu.copy()
                in_ky_gpu  = ky_atom_gpu.copy()
                in_kz_gpu  = kz_atom_gpu.copy()
                in_amp_gpu = amp_atom_gpu.copy()
                in_idx_gpu = cp.arange(nA, dtype=cp.int32)

                expand_max = int(sub_chunk_size)

                for bounce_i in range(1, n_bounces + 1):
                    out_x_gpu   = cp.zeros((expand_max,), dtype=cp.float32)
                    out_y_gpu   = cp.zeros((expand_max,), dtype=cp.float32)
                    out_z_gpu   = cp.zeros((expand_max,), dtype=cp.float32)
                    out_kx_gpu  = cp.zeros((expand_max,), dtype=cp.float32)
                    out_ky_gpu  = cp.zeros((expand_max,), dtype=cp.float32)
                    out_kz_gpu  = cp.zeros((expand_max,), dtype=cp.float32)
                    out_amp_gpu = cp.zeros((expand_max,), dtype=cp.complex64)
                    out_idx_gpu = cp.zeros((expand_max + 1,), dtype=cp.int32)  # last slot is counter
                    out_spc_gpu = cp.zeros((expand_max,), dtype=neighborSpc_gpu.dtype)

                    out_idx_gpu[expand_max] = 0  # reset counter

                    nBlocks = (cur_size + block1d - 1) // block1d
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
                            neighborSpc_gpu,

                            np.int32(cur_size),

                            out_x_gpu, out_y_gpu, out_z_gpu,
                            out_kx_gpu, out_ky_gpu, out_kz_gpu,
                            out_amp_gpu, out_idx_gpu, out_spc_gpu,
                            np.int32(expand_max)
                        )
                    )
                    cp.cuda.stream.get_current_stream().synchronize()

                    expansions_written = int(out_idx_gpu[expand_max].get())
                    if expansions_written == 0:
                        break

                    batchSize = expand_max
                    nSubBatches = (expansions_written + batchSize - 1) // batchSize
                    for sb in range(nSubBatches):
                        sbStart = sb * batchSize
                        sbEnd   = min(sbStart + batchSize, expansions_written)
                        sub_sz  = sbEnd - sbStart
                        if sub_sz <= 0:
                            continue

                        process_subchunk(sbStart, sbEnd,
                                        out_x_gpu, out_y_gpu, out_z_gpu,
                                        out_kx_gpu, out_ky_gpu, out_kz_gpu,
                                        out_amp_gpu, out_spc_gpu, sub_sz)

                    # Prepare next bounce inputs
                    if bounce_i < n_bounces:
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

            # Save partial image
            partial_results[out_idx] = dfield_gpu.reshape((Ny, Nx)).get()

            # Cleanup
            del pxg, pyg, pzg, dfield_gpu
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Launch one worker per GPU
        threads = []
        start = 1
        for gid in range(n_gpus):
            n_chunk = chunk_per_gpu + (1 if gid < remainder else 0)
            end = start + n_chunk
            t = threading.Thread(target=gpu_worker, args=(gid, range(start, end), gid))
            t.start()
            threads.append(t)
            start = end
        for t in threads:
            t.join()

        # Reduce
        for part in partial_results:
            if part is not None:
                final_result += part

        if offset is not None:
            final_result -= offset

        return final_result
    # -------------------------------------
    
    # -------------------------------------
    # Atomic master
    def atomic_direct_interaction(self, sample, detector, stage,
                                scattering=True, scattering_params=[None],
                                transmission=True, transmission_params=[0.0],
                                use_gpu=True):
        """
        High-level wrapper; now allows `scattering_params` = [offset, remove_forward_bool].
        """
        Nx, Ny = detector.shape
        final_field = np.zeros((Ny, Nx), dtype=np.complex128)

        # Parse scattering params
        sc_offset = scattering_params[0] if (len(scattering_params) >= 1) else None
        use_depth_ein = scattering_params[1] if len(scattering_params) >= 2 else False

        if use_gpu and (cp is not None):
            if scattering:
                final_field += self.atomic_scattering_kinematic(
                    sample, detector, stage,
                    offset=sc_offset, use_gpu=True,
                    remove_forward=transmission,
                    use_depth_ein=use_depth_ein
                )
            if transmission:
                final_field += self.atomic_transmission(
                    sample, detector, stage, use_gpu=True,
                    kernel_radius=transmission_params[0]
                )
        else:
            if cp is None and use_gpu:
                print("[beam] Cupy not installed, running CPU mode.")
            if scattering:
                final_field += self.atomic_scattering_kinematic(
                    sample, detector, stage,
                    offset=sc_offset, use_gpu=False,
                    remove_forward=transmission
                )
            if transmission:
                final_field += self.atomic_transmission(
                    sample, detector, stage, use_gpu=False,
                    kernel_radius=transmission_params[0]
                )

        detector.input_pixel_values(final_field)
    # -------------------------------------
    
    # -------------------------------------
    # Wavefield propagation
    def _angular_spectrum_propagate_gpu(
            self, field, dx, dy, z, kernel,
            step_max=0.02, pad_factor=1.0):

        """
        Band‑limited angular spectrum propagation on GPU with *symmetric*
        zero‑padding sized from sampling and |z| so wrap‑around cannot occur.
        Long distances are automatically broken into |z|<=step_max segments
        (padding recomputed per sub‑step).
        """
        if cp is None:
            raise RuntimeError('CuPy required for GPU propagation')

        # break long distances into sub‑steps
        z = float(z)
        if abs(z) > step_max:
            n = int(np.ceil(abs(z) / step_max))
            dz = z / n
            out = cp.asarray(field) if isinstance(field, cp.ndarray) else cp.asarray(field, dtype=cp.complex64)
            for _ in range(n):
                out = self._angular_spectrum_propagate_gpu(out, dx, dy, dz, kernel,
                                                        step_max=step_max, pad_factor=pad_factor)
            return out

        # input sizes
        F0 = cp.asarray(field, dtype=cp.complex64)
        Ny, Nx = int(F0.shape[0]), int(F0.shape[1])

        # padding chosen from sampling and distance (plus optional min pad_factor)
        Nx2, Ny2 = self._choose_optimal_pad(
            Nx, Ny, float(dx), float(dy), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        y0 = (Ny2 - Ny) // 2
        x0 = (Nx2 - Nx) // 2

        # embed symmetrically
        Fp = cp.zeros((Ny2, Nx2), dtype=cp.complex64)
        Fp[y0:y0+Ny, x0:x0+Nx] = F0

        # k‑grids (rad/m), no shifts (fft2 uses non‑shifted ordering)
        k  = 2.0 * np.pi / float(self._wavelength)
        kx = (2.0 * np.pi) * cp.fft.fftfreq(Nx2, d=float(dx)).astype(cp.float32)
        ky = (2.0 * np.pi) * cp.fft.fftfreq(Ny2, d=float(dy)).astype(cp.float32)

        # forward FFT
        Fp = cp.fft.fft2(Fp)

        # multiply by propagator in‑place (GPU kernel)
        block = (16, 16)
        grid  = ((Nx2 + block[0] - 1)//block[0],
                (Ny2 + block[1] - 1)//block[1])
        kernel(grid, block,
            (kx, ky, cp.float32(k), cp.float32(z),
                np.int32(Nx2), np.int32(Ny2), Fp))

        # inverse FFT and center crop
        out = cp.fft.ifft2(Fp)
        return out[y0:y0+Ny, x0:x0+Nx]
    
    def _angular_spectrum_propagate_cpu(
            self, field, dx, dy, z, lib, ffi,
            step_max=0.02, pad_factor=1.0):

        """
        Band‑limited angular spectrum propagation on CPU with symmetric padding
        sized from sampling and |z|. Long distances are split into smaller steps.
        """
        z = float(z)
        if abs(z) > step_max:
            n = int(np.ceil(abs(z) / step_max))
            dz = z / n
            out = field
            for _ in range(n):
                out = self._angular_spectrum_propagate_cpu(
                        out, dx, dy, dz, lib, ffi,
                        step_max=step_max, pad_factor=pad_factor)
            return out

        # input (Ny, Nx)
        F0 = np.asarray(field, dtype=np.complex64, order='C')
        Ny, Nx = int(F0.shape[0]), int(F0.shape[1])

        Nx2, Ny2 = self._choose_optimal_pad(
            Nx, Ny, float(dx), float(dy), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        y0 = (Ny2 - Ny) // 2
        x0 = (Nx2 - Nx) // 2

        # embed symmetrically
        Fp = np.zeros((Ny2, Nx2), np.complex64)
        Fp[y0:y0+Ny, x0:x0+Nx] = F0

        # spectral axes (rad/m)
        k  = np.float32(2.0 * np.pi / float(self._wavelength))
        kx = (2.0*np.pi) * np.fft.fftfreq(Nx2, d=float(dx)).astype(np.float32)
        ky = (2.0*np.pi) * np.fft.fftfreq(Ny2, d=float(dy)).astype(np.float32)

        # forward FFT
        Fp = np.fft.fft2(Fp)

        # multiply by propagator (CPU)
        lib.prop_mul_cpu(
            np.int32(Nx2), np.int32(Ny2),
            ffi.cast('const float*', kx.ctypes.data),
            ffi.cast('const float*', ky.ctypes.data),
            k, np.float32(z),
            ffi.cast('float _Complex*', Fp.ctypes.data)
        )

        # inverse FFT and center crop
        out = np.fft.ifft2(Fp)
        return out[y0:y0+Ny, x0:x0+Nx]

    def _apply_thin_lens_box(self, field, dx, dy, lens_data, use_gpu=True):
        """
        Thin-lens (CRL) approximation. Multiplies by exp(-i*k/(2f)*r^2)
        and a uniform absorption factor if provided.
        """
        wavelength = self._wavelength
        k_val = 2.0 * np.pi / wavelength

        # mm -> m
        f  = lens_data['focal_length'] * 1e-3
        t  = lens_data['thickness'] * 1e-3
        nsigma = lens_data.get('absorption_sigma', np.inf)
        N_lenses = lens_data['number']

        Nx, Ny = field.shape[1], field.shape[0]
        x_arr = np.arange(Nx, dtype=np.float32)
        y_arr = np.arange(Ny, dtype=np.float32)
        cx = (Nx - 1) / 2.0
        cy = (Ny - 1) / 2.0

        if use_gpu and cp is not None:
            x_gpu = cp.asarray((x_arr - cx) * dx, dtype=cp.float32)
            y_gpu = cp.asarray((y_arr - cy) * dy, dtype=cp.float32)
            Xgpu = x_gpu[None, :].repeat(Ny, axis=0)
            Ygpu = y_gpu[:, None].repeat(Nx, axis=1)
            R2 = Xgpu * Xgpu + Ygpu * Ygpu

            phase_lens = -0.5 * (k_val / f) * R2
            cph = cp.cos(phase_lens)
            sph = cp.sin(phase_lens)

            F_gpu = cp.asarray(field, dtype=cp.complex64)
            real_part = F_gpu.real * cph - F_gpu.imag * sph
            imag_part = F_gpu.real * sph + F_gpu.imag * cph
            out = real_part + 1j * imag_part

            if not cp.isinf(nsigma):
                out *= cp.exp(- N_lenses * t / nsigma)

            return out.get()

        # ---- CPU path (numpy only) ----
        xx = (x_arr - cx) * dx
        yy = (y_arr - cy) * dy
        E_out = np.empty_like(field, dtype=np.complex64)

        for iy in range(Ny):
            r_y = yy[iy]
            for ix in range(Nx):
                r_x = xx[ix]
                r2 = r_x * r_x + r_y * r_y
                phase = -0.5 * (k_val / f) * r2
                cph = np.cos(phase)
                sph = np.sin(phase)
                val = field[iy, ix]
                re2 = val.real * cph - val.imag * sph
                im2 = val.real * sph + val.imag * cph
                E_out[iy, ix] = re2 + 1j * im2

        if not np.isinf(nsigma):
            E_out *= np.exp(- N_lenses * t / nsigma)

        return E_out

    def _apply_aperture(self, field, dx, dy, aperture_data, use_gpu=True):
        """
        Apply a real-space aperture (square or circular).
        Aperture size is in mm. We place the aperture center at field center.

        aperture_data dict fields:
          - 'shape': 'square' or 'circular'
          - 'width': float in mm
        """
        Nx, Ny = field.shape[1], field.shape[0]
        shape_type = aperture_data['shape'].lower()
        width_mm = aperture_data['width']
        width_m  = width_mm * 1e-3
        # Aperture extends from -w/2 .. +w/2 in x and y

        # Build coordinate arrays
        x_arr = np.arange(Nx, dtype=np.float32) - (Nx-1)/2.0
        y_arr = np.arange(Ny, dtype=np.float32) - (Ny-1)/2.0
        x_arr *= dx
        y_arr *= dy
        half = 0.5*width_m

        if use_gpu and cp is not None:
            x_gpu = cp.asarray(x_arr)
            y_gpu = cp.asarray(y_arr)
            Xgpu = x_gpu[None, :].repeat(Ny, axis=0)
            Ygpu = y_gpu[:, None].repeat(Nx, axis=1)
            if shape_type == 'square':
                mask = (cp.abs(Xgpu) <= half) & (cp.abs(Ygpu) <= half)
            elif shape_type == 'circular':
                R2 = Xgpu*Xgpu + Ygpu*Ygpu
                r0 = half
                mask = (R2 <= (r0*r0))
            else:
                # Fallback to square
                mask = (cp.abs(Xgpu) <= half) & (cp.abs(Ygpu) <= half)

            F_gpu = cp.asarray(field, dtype=cp.complex64)
            F_gpu[~mask] = 0.0 + 0.0j
            return F_gpu.get()

        else:
            E_out = np.copy(field)
            for iy in range(Ny):
                yy = y_arr[iy]
                for ix in range(Nx):
                    xx = x_arr[ix]
                    if shape_type == 'square':
                        if (abs(xx) > half) or (abs(yy) > half):
                            E_out[iy, ix] = 0.0
                    elif shape_type == 'circular':
                        if (xx*xx + yy*yy) > (half*half):
                            E_out[iy, ix] = 0.0
                    else:
                        # fallback
                        if (abs(xx) > half) or (abs(yy) > half):
                            E_out[iy, ix] = 0.0
            return E_out

    def wavefield_propagation(self, detector, optics_stack,
                            use_gpu=True, step_max=0.02, pad_factor=1.0):
        """
        Propagate detector.wavefield through an optics stack using a
        *band‑limited angular spectrum method*.

        Key details:
        - Symmetric zero‑padding sized from (dx,dy,λ,z) each step; avoids wrap‑around.
        - Evanescent components decay as exp(-|z|·α); no numerical growth for z<0.
        - Nyquist is respected automatically via the padding and sampling‑limited
            angular bound sinθ_max = min(1, λ/(2·d)).
        """

        # NOTE: detector.pixel_size is assumed (dy, dx) in Å; convert to meters.
        dy, dx = detector.pixel_size * 1e-10
        Ny, Nx = detector.shape
        E = detector.pixel_values  # complex64 (Ny, Nx)

        if use_gpu and cp is not None:
            kernel = self.build_propagation_multiplier_kernel()
            ffi, lib = None, None
        else:
            kernel = None
            ffi, lib = self.compile_propagation_multiplier_cffi()

        for elem in optics_stack.components:
            kind = elem['kind'].lower()

            if kind == 'free space':
                z = float(elem['length']) * 1e-3  # mm -> m
                if use_gpu and cp is not None:
                    E = self._angular_spectrum_propagate_gpu(
                            E, dx, dy, z, kernel,
                            step_max=step_max, pad_factor=pad_factor).get()
                else:
                    E = self._angular_spectrum_propagate_cpu(
                            E, dx, dy, z, lib, ffi,
                            step_max=step_max, pad_factor=pad_factor)

            elif kind == 'lens box':
                E = self._apply_thin_lens_box(E, dx, dy, elem, use_gpu and cp is not None)

            elif kind == 'aperture':
                E = self._apply_aperture(E, dx, dy, elem, use_gpu and cp is not None)

            else:
                raise ValueError(f'Unknown optics element "{kind}"')

        detector.input_pixel_values(E.astype(np.complex64))
    # -------------------------------------
    
