# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import json
import os
import gc
import threading
import warnings
from Logging import logging
try:
    import cupy as cp
    import cupyx
except ImportError:
    cp = None
    cupyx = None
from cffi import FFI
import databases.scattering
import importlib.resources as pkg_resources

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class beam(logging):
    
    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "atomic_scattering_kinematic",
        "atomic_scattering_dynamical",
        "atomic_transmission",
        "atomic_direct_interaction",
        "precompute_depth_ein_all_chunks",
        "wavefield_propagation",
        "create_beam",
        "read_beam_metadata",
        "write_beam_metadata"
    )

    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization    
    def __init__(self, directory=os.getcwd()):
        """
        Initialize a beam instance.

        Args:
            directory (str): Where beam metadata is read and written; created if
                missing.

        Also sets the SI constants h, c, q and caches h/q for eV-to-wavelength
        conversion.
        """
        super().__init__(log_name="beam")
        self.directory = directory
        self._direction = None
        self._energy = None
        self._wavelength = None
        self._pol_perp_rate = 0.5  # Default: unpolarized
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        # Constants (SI units)
        self._h = 6.62607015e-34
        self._c = 299792458
        self._q = 1.602176634e-19
        self._hq = self._h / self._q

    _DB_CACHE = {}          # parsed scattering tables, shared by all instances
    _MORTON_KERNEL = None   # compiled Morton-code kernel, built on first use

    def create_beam(self,
                    energy,
                    eV=True,
                    beam_shape="rectangular",
                    beam_size=(1000.0, 1000.0),
                    beam_samples=(256, 256),
                    beam_profile="uniform",
                    gaussian_waist=None,
                    pol_perp_rate=0.5):
        """
        Configure a forward-propagating (+x) beam and build its transverse grid.

        The direction is fixed to +x because the scattering kernels assume it.
        pol_perp_rate is the fraction of incident intensity polarized
        perpendicular to the scattering plane (rho_perp); 0.5 is unpolarized.

        Args:
            energy (float): Beam energy in eV, or in J when eV is False.
            eV (bool): Energy unit flag.
            beam_shape (str): "rectangular" or "circular" support.
            beam_size (tuple[float, float]): Support size (u, v) in angstrom.
            beam_samples (tuple[int, int]): Grid samples (Ny, Nz).
            beam_profile (str): "uniform" or "gaussian" amplitude profile.
            gaussian_waist (tuple[float, float] or None): 1/e^2 radii (wy, wz) in
                angstrom; None means half of beam_size per axis.
            pol_perp_rate (float): Perpendicular polarization fraction in [0, 1].
        """
        # Kernels assume a +x beam
        self._direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Energy (eV) -> wavelength
        if not eV:
            energy = energy / self._q
        self._energy = float(energy)
        self._wavelength = self._hq * self._c / self._energy

        # Only kx is nonzero for +x propagation
        k = 2.0 * np.pi / self._wavelength
        self._kx_scalar = np.float32(k)
        self._ky_scalar = np.float32(0.0)
        self._kz_scalar = np.float32(0.0)

        self._beam_shape   = str(beam_shape).lower()
        self._beam_size    = (float(beam_size[0]), float(beam_size[1]))
        self._beam_samples = (int(beam_samples[0]), int(beam_samples[1]))
        self._beam_profile = str(beam_profile).lower()
        self._gauss_waist  = gaussian_waist

        self._pol_perp_rate = float(np.clip(pol_perp_rate, 0.0, 1.0))

        # Transverse basis for +x propagation: e1 = +y, e2 = +z
        self._beam_e1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._beam_e2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._init_beam_grid()
        
    def _init_beam_grid(self):
        """
        Build the centered (u, v) transverse grid from beam_size / beam_samples
        and the zero-phase complex field map.

        Sets _beam_Ny/_beam_Nz (grid sizes), _beam_du/_beam_dv (spacings,
        angstrom), _beam_uc/_beam_vc (center indices), _beam_u_centers /
        _beam_v_centers (coordinates, angstrom) and _beam_E0_map (complex64).
        """
        Ny, Nz = self._beam_samples
        Sy, Sz = self._beam_size  # angstrom
        Ny = int(max(1, Ny)); Nz = int(max(1, Nz))
        self._beam_Ny, self._beam_Nz = Ny, Nz

        # angstrom per grid step
        self._beam_du = float(Sy) / Ny
        self._beam_dv = float(Sz) / Nz
        self._beam_uc = (Ny - 1) * 0.5  # center index along u
        self._beam_vc = (Nz - 1) * 0.5  # center index along v

        # Grid coordinates (angstrom)
        u_centers = (np.arange(Ny, dtype=np.float32) - self._beam_uc) * self._beam_du
        v_centers = (np.arange(Nz, dtype=np.float32) - self._beam_vc) * self._beam_dv
        U, V = np.meshgrid(u_centers, v_centers, indexing='ij')  # (Ny, Nz)

        # Support mask
        if self._beam_shape == "circular":
            ry = 0.5 * Sy
            rz = 0.5 * Sz
            mask = ((U / max(ry, 1e-9))**2 + (V / max(rz, 1e-9))**2) <= 1.0
        else:
            mask = np.ones_like(U, dtype=bool)  # full grid

        # Amplitude only; the phase is zero here
        if self._beam_profile == "gaussian":
            wy, wz = self._gauss_waist if (self._gauss_waist is not None) else (0.5 * Sy, 0.5 * Sz)
            wy = max(float(wy), 1e-6); wz = max(float(wz), 1e-6)
            A0 = np.exp(-((U / wy) ** 2 + (V / wz) ** 2)).astype(np.float32)
            A0 *= mask.astype(np.float32)
        else:
            A0 = mask.astype(np.float32)

        self._beam_u_centers = u_centers
        self._beam_v_centers = v_centers
        self._beam_E0_map = (A0.astype(np.float32) + 0.0j).astype(np.complex64)

    def set_wavefield(self, wavefield):
        """
        Replace the built-in entrance field with a user-supplied E0(u, v)
        sampled on the current transverse grid.

        Args:
            wavefield (np.ndarray): Shape (Ny, Nz) = beam_samples, all values
                finite. Real or complex; stored as C-contiguous complex64.

        Sampling: 'ij' indexing, axis 0 is u (+y), axis 1 is v (+z), beam
        along +x. Pitch du = beam_size[0]/Ny, dv = beam_size[1]/Nz (angstrom),
        so sample (i, j) sits at u = (i - (Ny-1)/2)*du, v = (j - (Nz-1)/2)*dv
        and the beam axis passes through fractional index ((Ny-1)/2, (Nz-1)/2).

        Amplitude: no renormalization. Built-in profiles peak at 1; the
        supplied field scales all downstream intensities directly.

        Call this LAST: create_beam() and read_beam_metadata() rebuild the
        field map and overwrite a custom wavefield. Sets _beam_profile to
        'custom-<sha1[:12]>' of the field bytes so on-disk Ein cache keys
        track the field contents.

        Raises:
            ValueError: Wrong shape or non-finite values.
        """
        import hashlib
        arr = np.asarray(wavefield)
        if arr.ndim != 2:
            raise ValueError(
                f"set_wavefield: expected a 2-D array of shape "
                f"({self._beam_Ny}, {self._beam_Nz}), got ndim={arr.ndim}"
            )
        expected_shape = (int(self._beam_Ny), int(self._beam_Nz))
        if tuple(arr.shape) != expected_shape:
            raise ValueError(
                f"set_wavefield: expected shape {expected_shape} "
                f"(beam_samples), got {tuple(arr.shape)}"
            )
        arr = np.ascontiguousarray(arr, dtype=np.complex64)
        if not np.all(np.isfinite(arr.real)) or not np.all(np.isfinite(arr.imag)):
            raise ValueError(
                "set_wavefield: wavefield contains non-finite values (NaN/Inf)"
            )
        self._beam_E0_map = arr
        self._beam_profile = "custom-" + hashlib.sha1(arr.tobytes()).hexdigest()[:12]

    def read_beam_metadata(self):
        """
        Restore the beam state from beam_metadata.json: normalized direction,
        k components, transverse basis (e1, e2) and the grid via
        _init_beam_grid(). Accepts older files without beam-grid fields.

        Raises:
            FileNotFoundError: No metadata file in self.directory.
        """
        metadata_filename = os.path.join(self.directory, "beam_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            beam_metadata = json.load(f)

        direction = beam_metadata.get("direction", None)
        if direction is None:
            direction = [1.0, 0.0, 0.0]
        self._direction = np.array(direction, dtype=np.float32)
        self._direction = self._direction / np.linalg.norm(self._direction)

        self._energy     = float(beam_metadata.get("energy", self._energy if self._energy is not None else 1.0))
        self._wavelength = float(beam_metadata.get("wavelength",
                                                (self._hq * self._c / self._energy)))

        k = 2.0 * np.pi / self._wavelength
        self._kx_scalar = float(self._direction[0] * k)
        self._ky_scalar = float(self._direction[1] * k)
        self._kz_scalar = float(self._direction[2] * k)

        self._beam_shape = str(beam_metadata.get("beam_shape", "rectangular")).lower()

        # Size (angstrom); defaults if missing or non-positive
        default_size = (1000.0, 1000.0)
        size_list = beam_metadata.get("beam_size", default_size)
        if size_list is None or len(size_list) != 2:
            size_list = default_size
        Sy = float(size_list[0]) if float(size_list[0]) > 0.0 else default_size[0]
        Sz = float(size_list[1]) if float(size_list[1]) > 0.0 else default_size[1]
        self._beam_size = (Sy, Sz)

        # Samples (Ny, Nz)
        samples = beam_metadata.get("beam_samples", None)
        if samples is None or (isinstance(samples, (list, tuple)) and len(samples) != 2):
            samples = (256, 256)
        Ny = int(samples[0]); Nz = int(samples[1])
        Ny = max(1, Ny); Nz = max(1, Nz)
        self._beam_samples = (Ny, Nz)

        # Profile and optional Gaussian waist
        self._beam_profile = str(beam_metadata.get("beam_profile", "uniform")).lower()
        gw = beam_metadata.get("gaussian_waist", None)
        if gw is None:
            # Missing waist: half of the size for a gaussian profile
            if self._beam_profile == "gaussian":
                self._gauss_waist = (0.5 * Sy, 0.5 * Sz)
            else:
                self._gauss_waist = None
        else:
            if isinstance(gw, (list, tuple)) and len(gw) == 2:
                self._gauss_waist = (float(gw[0]), float(gw[1]))
            else:
                # Malformed waist: same default
                self._gauss_waist = (0.5 * Sy, 0.5 * Sz) if self._beam_profile == "gaussian" else None

        # 0.5 = unpolarized
        self._pol_perp_rate = float(beam_metadata.get("pol_perp_rate", 0.5))
        self._pol_perp_rate = float(np.clip(self._pol_perp_rate, 0.0, 1.0))

        e1, e2 = self.make_orthonormal_basis(self._direction)
        self._beam_e1 = e1.astype(np.float32)
        self._beam_e2 = e2.astype(np.float32)

        if hasattr(self, "_init_beam_grid"):
            self._init_beam_grid()

    ## Data Handling Functions    
    def write_beam_metadata(self, override_directory=None):
        """
        Write the beam state, including the beam-grid definition, to
        beam_metadata.json.

        Grid fields: beam_samples [Ny, Nz], beam_profile, gaussian_waist
        [wy, wz] in angstrom (1/e^2 radii) or null, and metadata_version
        (>= 2 when the grid is present).

        Args:
            override_directory (str or None): Write here instead of
                self.directory.
        """
        # Tolerate missing attributes
        direction = self._direction.tolist() if getattr(self, "_direction", None) is not None else None
        energy    = getattr(self, "_energy", None)
        wavelength= getattr(self, "_wavelength", None)

        beam_shape   = getattr(self, "_beam_shape", "rectangular")
        beam_size    = list(getattr(self, "_beam_size", (1000.0, 1000.0)))     # angstrom
        beam_samples = getattr(self, "_beam_samples", None)
        if beam_samples is not None:
            beam_samples = [int(beam_samples[0]), int(beam_samples[1])]
        beam_profile = getattr(self, "_beam_profile", "uniform")
        gauss_waist  = getattr(self, "_gauss_waist", None)
        if gauss_waist is not None:
            gauss_waist = [float(gauss_waist[0]), float(gauss_waist[1])]
        pol_perp_rate = getattr(self, "_pol_perp_rate", 0.5)

        beam_metadata = {
            "metadata_version": 2,
            "direction"       : direction,
            "energy"          : energy,
            "wavelength"      : wavelength,
            "beam_shape"      : beam_shape,
            "beam_size"       : beam_size,       # [size_u_angstrom, size_v_angstrom]
            "beam_samples"    : beam_samples,    # [Ny, Nz]
            "beam_profile"    : beam_profile,    # "uniform" | "gaussian"
            "gaussian_waist"  : gauss_waist,     # [wy_angstrom, wz_angstrom] or null
            "pol_perp_rate"   : float(pol_perp_rate)  # Polarization rate [0.0-1.0]
        }

        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "beam_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "beam_metadata.json")

        with open(metadata_filename, "w") as f:
            json.dump(beam_metadata, f, indent=4)

    ## Static Functions
    # -------------------------------------
    # General
    @staticmethod
    def make_orthonormal_basis(direction):
        """
        Two unit vectors e1, e2 orthogonal to each other and to direction.

        Args:
            direction (np.ndarray): Shape (3,); need not be normalized.

        Returns:
            tuple[np.ndarray, np.ndarray]: e1, e2 of shape (3,), float32.
        """
        d = direction / np.linalg.norm(direction)

        # Pick an axis that is not almost parallel to d
        if abs(d[0]) < 0.9:
            temp = np.array([1, 0, 0], dtype=np.float32)
        else:
            temp = np.array([0, 1, 0], dtype=np.float32)

        e1 = np.cross(d, temp)
        e1 /= np.linalg.norm(e1)

        e2 = np.cross(d, e1)
        e2 /= np.linalg.norm(e2)

        return e1.astype(np.float32), e2.astype(np.float32)
    
    @staticmethod
    def allocate_pinned_array(np_array, dtype=np.float32):
        """
        Copy a NumPy array into page-locked (pinned) host memory for faster
        host-to-device transfers; the result can be passed to cp.array(...).

        Args:
            np_array (np.ndarray): Source array.
            dtype: dtype of the pinned copy; None keeps np_array.dtype.

        Returns:
            np.ndarray: Same shape as np_array, backed by pinned memory.

        Requires CuPy; does not check that cp is available.
        """
        if dtype is None:
            dtype = np_array.dtype
        shape = np_array.shape
        n_elems = np.prod(shape)

        memptr = cp.cuda.alloc_pinned_memory(
            n_elems * np.dtype(dtype).itemsize
        )
        pinned_arr = np.ndarray(shape=shape, dtype=dtype, buffer=memptr)
        pinned_arr[...] = np_array
        return pinned_arr
    
    @staticmethod
    def parse_f0_db_all(database_name='f0_WaasKirf.dat'):
        """
        Parse Waasmaier-Kirfel f0 parameters for all elements.

        Element sections start with "#S"; parameter lines carry 11 floats
        [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5].

        Args:
            database_name (str): Resource file under databases.scattering.

        Returns:
            dict[str, np.ndarray]: Element symbol -> float32 array of shape (11,).
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
        Parse Cromer-Liberman anomalous scattering tables (f1, f2) for all
        elements. Element sections start with "#S"; data rows are
        [Energy_eV, f1, f2].

        Args:
            database_name (str): Resource file under databases.scattering.

        Returns:
            dict[str, np.ndarray]: Element symbol -> float32 array of shape
            (N, 3) with columns [Energy_eV, f1, f2].
        """
        # Parsed once per process; re-parsing on every call was a real cost.
        # Callers never mutate the returned dict.
        cache_key = ("f1f2", str(database_name))
        cached = beam._DB_CACHE.get(cache_key)
        if cached is not None:
            return cached
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
            # Flush the last element
            if element is not None and len(param_list) > 0:
                f1f2_dict[element] = np.array(param_list, dtype=np.float32)
        beam._DB_CACHE[cache_key] = f1f2_dict
        return f1f2_dict

    @staticmethod
    def get_f1f2_from_params(energy, f1f2_table):
        """
        Piecewise-linear interpolation of f1 + 1j*f2 at the given energy;
        outside the table the end segment is extended linearly.

        Args:
            energy (float): Energy in eV.
            f1f2_table (np.ndarray): Shape (N, 3), columns [Energy_eV, f1, f2],
                ascending in energy.

        Returns:
            complex: f1 + 1j*f2.
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
        f1 = f10 + (f11 - f10) * w
        f2 = f20 + (f21 - f20) * w
        return f1 + 1j * f2
    
    @staticmethod
    def _build_f0_zero_dict(db_dict_f0_all):
        """
        f0(0) = c + sum(a1..a5) for each element.

        Args:
            db_dict_f0_all (dict): Element -> [a1..a5, c, b1..b5] parameters.

        Returns:
            dict: Element -> f0(0).
        """
        f0_0_dict = {}
        for el, params in db_dict_f0_all.items():
            a1, a2, a3, a4, a5, c = params[0], params[1], params[2], params[3], params[4], params[5]
            val = float(c + a1 + a2 + a3 + a4 + a5)
            f0_0_dict[el] = val
        return f0_0_dict
    # -------------------------------------
    
    # -------------------------------------
    # Direct Scattering
    @staticmethod
    def compile_compute_scattering_cffi():
        """
        Compile and return the CPU direct-scattering routine (CFFI):
            void compute_scattering_cffi(
                int atom_count,
                const float* positions,     // (atom_count, 3) in meters
                const float* f0_params,     // (atom_count, 11)
                const float* f0_zero,       // (atom_count,)
                int remove_forward,         // 0 or 1
                const float* s_anom_real,   // (atom_count,)
                const float* s_anom_imag,   // (atom_count,)
                const float* initial_amp_r, // (atom_count,)
                const float* initial_amp_i, // (atom_count,)
                int Ny, int Nz,
                const float* coords_x,      // (Ny*Nz) in meters
                const float* coords_y,
                const float* coords_z,
                float k_val,                // 2*pi/lambda in rad/m
                int apply_pol,              // 0 or 1
                float pol_perp_rate,        // rho_perp in [0, 1]
                int apply_spherical_decay,  // 0 or 1
                float* out_r, float* out_i  // (Ny*Nz)
            );

        Polarization: the amplitude is scaled by sqrt(P) with
        P = rho_perp + (1 - rho_perp) * cos(2*theta)^2, and cos(2*theta) ~ dx/r
        for a +x beam.

        Returns:
            tuple: (ffi_obj, C_mod) from cffi.verify; needs a working C compiler.
        """
        from cffi import FFI

        c_source = r'''
        #include <math.h>
        #include <stddef.h>
        #include <stdlib.h>

        static inline float get_f0_value(float Q_val, const float* params)
        {
            const float PI_F = 3.14159265358979323846f;
            const float K_SCALE_FACTOR = 0.25f * 1.0e-10f / PI_F;  // Q[m^-1] -> s[Angstrom^-1]
            const float s   = K_SCALE_FACTOR * Q_val;
            const float ss  = s*s;

            float f0_val = params[5]; // c
            for (int i=0;i<5;i++){
                const float ai = params[i];
                const float bi = params[6+i];
                f0_val += ai * expf(-bi * ss);
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
            const float *initial_amp_r,   // (atom_count,)
            const float *initial_amp_i,   // (atom_count,)
            int Ny, int Nz,
            const float *coords_x,        // (Ny*Nz) [m]
            const float *coords_y,
            const float *coords_z,
            float k_val,                  // 2*pi/lambda [rad/m]
            int   apply_pol,              // 0/1
            float pol_perp_rate,          // rho_perp in [0,1]
            int   apply_spherical_decay,  // 0/1
            float *out_r, float *out_i    // (Ny*Nz)
        )
        {
            const float PI_F = 3.14159265358979323846f;
            const float rE_F = 2.81794092e-15f;  // classical electron radius [m]
            const int pixel_count = Ny*Nz;

            // Per-pixel Q_cut: half a pixel's diagonal in Q-space
            int have_qcut = 1;
            float* Q_cut = (float*)malloc((size_t)pixel_count * sizeof(float));
            if (!Q_cut) have_qcut = 0;

            // Per-pixel R0: distance from origin to pixel
            int have_r0 = 1;
            float* R0_arr = (float*)malloc((size_t)pixel_count * sizeof(float));
            if (!R0_arr) have_r0 = 0;

            if (have_qcut || have_r0) {
                for (int p = 0; p < pixel_count; ++p) {
                    int ix = p % Ny;
                    int iy = p / Ny;

                    // Unit vector to this pixel
                    float tx = coords_x[p];
                    float ty = coords_y[p];
                    float tz = coords_z[p];
                    float R0 = sqrtf(tx*tx + ty*ty + tz*tz);
                    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
                    if (R0 > 0.0f) {
                        float invR0 = 1.0f / R0;
                        ux = tx * invR0; uy = ty * invR0; uz = tz * invR0;
                    }
                    if (have_r0) {
                        R0_arr[p] = R0;
                    }

                    if (have_qcut) {
                        // Right neighbor (left on the edge, self if single column)
                        int n_right = (ix + 1 < Ny) ? (p + 1) : ((ix > 0) ? (p - 1) : p);
                        float rx = coords_x[n_right];
                        float ry = coords_y[n_right];
                        float rz = coords_z[n_right];
                        float Rr = sqrtf(rx*rx + ry*ry + rz*rz);
                        float urx = 0.0f, ury = 0.0f, urz = 0.0f;
                        if (Rr > 0.0f) {
                            float invRr = 1.0f / Rr;
                            urx = rx*invRr; ury = ry*invRr; urz = rz*invRr;
                        }
                        float cos_dx = ux*urx + uy*ury + uz*urz;
                        if (cos_dx > 1.0f) cos_dx = 1.0f;
                        if (cos_dx < -1.0f) cos_dx = -1.0f;
                        float Qx = k_val * sqrtf(fmaxf(0.0f, 2.0f * (1.0f - cos_dx)));

                        // Up neighbor (down on the edge, self if single row)
                        int n_up = (iy + 1 < Nz) ? (p + Ny) : ((iy > 0) ? (p - Ny) : p);
                        float ux2 = coords_x[n_up];
                        float uy2 = coords_y[n_up];
                        float uz2 = coords_z[n_up];
                        float Ru = sqrtf(ux2*ux2 + uy2*uy2 + uz2*uz2);
                        float vux = 0.0f, vuy = 0.0f, vuz = 0.0f;
                        if (Ru > 0.0f) {
                            float invRu = 1.0f / Ru;
                            vux = ux2*invRu; vuy = uy2*invRu; vuz = uz2*invRu;
                        }
                        float cos_dy = ux*vux + uy*vuy + uz*vuz;
                        if (cos_dy > 1.0f) cos_dy = 1.0f;
                        if (cos_dy < -1.0f) cos_dy = -1.0f;
                        float Qy = k_val * sqrtf(fmaxf(0.0f, 2.0f * (1.0f - cos_dy)));

                        // Half-diagonal in Q-space
                        float Qhx = 0.5f * Qx;
                        float Qhy = 0.5f * Qy;
                        if (have_qcut) {
                            Q_cut[p] = sqrtf(Qhx*Qhx + Qhy*Qhy);
                        }
                    }
                }
            }

            const float wavelength_m = (2.0f * PI_F) / k_val;

            // Accumulate over atoms and pixels
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

                    // +x beam: cos(2*theta) = dx/r, Q = 2k sin(theta)
                    float dotv = (dx / r_det);
                    float tmp = 2.0f*(1.0f - dotv);
                    if (tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_val * sqrtf(tmp);

                    float f0_val = get_f0_value(Q_val, f0p);

                    // f = f0(Q) + f' + i f''
                    float s_re = (f0_val + sanr);
                    float s_im = (sani);

                    // Remove the forward amplitude inside Q < Q_cut
                    if (remove_forward) {
                        if (have_qcut) {
                            if (Q_val < Q_cut[p]) {
                                s_re -= (f00 + sanr);
                                s_im -= (sani);
                            }
                        } else {
                            // No Q_cut (malloc failed): subtract everywhere
                            s_re -= (f00 + sanr);
                            s_im -= (sani);
                        }
                    }

                    // Multiply by the complex entrance amplitude
                    float t_re = amp_r * s_re - amp_i * s_im;
                    float t_im = amp_r * s_im + amp_i * s_re;

                    // Phase k*(ax + r_det), reduced mod wavelength for float precision
                    float phase = k_val * (fmodf(ax, wavelength_m) + fmodf(r_det, wavelength_m));
                    float cph = cosf(phase);
                    float sph = sinf(phase);

                    float val_r = (t_re * cph - t_im * sph);
                    float val_i = (t_re * sph + t_im * cph);

                    // Relative spherical decay R0 / r_det
                    float scale_rel = 1.0f;
                    if (apply_spherical_decay && r_det > 0.0f) {
                        float R0_local;
                        if (have_r0) {
                            R0_local = R0_arr[p];
                        } else {
                            float tx = coords_x[p], ty = coords_y[p], tz = coords_z[p];
                            R0_local = sqrtf(tx*tx + ty*ty + tz*tz);
                        }
                        if (R0_local > 0.0f) {
                            scale_rel = R0_local / r_det;
                        }
                    }

                    // Polarization factor on amplitude
                    if (apply_pol) {
                        float P = pol_perp_rate + (1.0f - pol_perp_rate) * (dotv * dotv);
                        if (P < 0.0f) P = 0.0f;
                        if (P > 1.0f) P = 1.0f;
                        float scale = sqrtf(P);
                        val_r *= scale;
                        val_i *= scale;
                    }

                    // Scale by r_e and accumulate
                    val_r *= (rE_F * scale_rel);
                    val_i *= (rE_F * scale_rel);

                    out_r[p] += val_r;
                    out_i[p] += val_i;
                }
            }

            if (Q_cut) free(Q_cut);
            if (R0_arr) free(R0_arr);
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
                int Ny, int Nz,
                const float *coords_x,
                const float *coords_y,
                const float *coords_z,
                float k_val,
                int   apply_pol,
                float pol_perp_rate,
                int   apply_spherical_decay,
                float *out_r, float *out_i
            );
        """)
        C_mod = ffi_obj.verify(c_source, extra_compile_args=['-O3'])
        return ffi_obj, C_mod
    
    @staticmethod
    def _ein_bilinear_cpu(
        pos_np,   # (N,3) float32, Angstrom, on host
        tau,      # (NyB,NzB) float32
        phi,      # (NyB,NzB) float32
        E0,       # (NyB,NzB) complex64
        e1, e2, khat,  # (3,) float32
        du, dv, uc, vc,
        s_min, s_max
    ):
        """
        Per-atom entrance field Ein by bilinear sampling of the beam grid.

        Each atom is projected onto beam coordinates (u, v); tau, phi and E0
        are sampled bilinearly there and
            Ein = E0 * exp(-f * tau) * exp(i * f * phi)
        with f in [0, 1] the depth fraction along khat between s_min and s_max.
        Atoms outside the grid get Ein = 0 (no edge replication).

        Args:
            pos_np: (N, 3) float32 atom positions in Angstrom.
            tau, phi: (NyB, NzB) float32 attenuation and phase maps.
            E0: (NyB, NzB) complex64 incident field.
            e1, e2, khat: (3,) float32 transverse basis and beam direction.
            du, dv, uc, vc: grid spacings (Angstrom) and center indices.
            s_min, s_max: depth range along khat in Angstrom.

        Returns:
            np.ndarray: (N,) complex64.
        """
        N = int(pos_np.shape[0])
        out = np.zeros((N,), dtype=np.complex64)
        if N == 0:
            return out

        NyB, NzB = int(tau.shape[0]), int(tau.shape[1])

        # Fractional grid indices in the beam basis
        au =pos_np[:, 0]*e1[0] + pos_np[:, 1]*e1[1] + pos_np[:, 2]*e1[2]
        av = pos_np[:, 0]*e2[0] + pos_np[:, 1]*e2[1] + pos_np[:, 2]*e2[2]
        iu = au / float(du) + float(uc)
        iv = av / float(dv) + float(vc)

        # Atoms outside the grid get zero
        inb =(iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
        if not np.any(inb):
            return out

        # Bilinear weights and gather indices
        iu_in = iu[inb]; iv_in = iv[inb]
        i0 = np.floor(iu_in).astype(np.int64)
        j0 = np.floor(iv_in).astype(np.int64)
        i1 = np.clip(i0 + 1, 0, NyB - 1)
        j1 = np.clip(j0 + 1, 0, NzB - 1)

        fu = (iu_in - i0).astype(np.float32)
        fv = (iv_in - j0).astype(np.float32)
        w00 = (1.0 - fu)*(1.0 - fv)
        w01 = (1.0 - fu)*fv
        w10 = fu*(1.0 - fv)
        w11 = fu*fv

        r00 = i0 * NzB + j0
        r01 = i0 * NzB + j1
        r10 = i1 * NzB + j0
        r11 = i1 * NzB + j1

        tau_f = tau.ravel()
        phi_f = phi.ravel()
        E0_f  = E0.ravel()

        tau_s = tau_f[r00]*w00 + tau_f[r01]*w01 + tau_f[r10]*w10 + tau_f[r11]*w11
        phi_s = phi_f[r00]*w00 + phi_f[r01]*w01 + phi_f[r10]*w10 + phi_f[r11]*w11
        E0_s  = E0_f[r00]*w00 + E0_f[r01]*w01 + E0_f[r10]*w10 + E0_f[r11]*w11

        # Depth fraction f in [0, 1] along khat
        s_vals = pos_np[inb, 0]*khat[0] + pos_np[inb, 1]*khat[1] + pos_np[inb, 2]*khat[2]
        denom = float(s_max) - float(s_min)
        if not (denom > 0.0):
            denom = 1.0  # robust fallback
        f = np.clip((s_vals - float(s_min))/denom, 0.0, 1.0).astype(np.float32)

        amp = np.exp(-f * tau_s).astype(np.float32)
        phase = f * phi_s
        cph = np.cos(phase)
        sph = np.sin(phase)

        real = (E0_s.real * cph - E0_s.imag * sph) * amp
        imag = (E0_s.real * sph + E0_s.imag * cph) * amp
        out[inb] = (real + 1j*imag).astype(np.complex64)
        return out
    
    def set_phase_tolerance(self, phi_tol_rad: float):
        """
        Set the maximum phase error used to pick the series order N.
        Non-positive or invalid values fall back to 1e-3 rad.

        Args:
            phi_tol_rad (float): Maximum phase error per contribution, radians.
        """
        try:
            val = float(phi_tol_rad)
            if not (val > 0.0):
                val = 1e-3
        except Exception:
            val = 1e-3
        self._phase_tol_rad = val
        
    def _estimate_required_series_terms(self, a_max_m: float, R0_min_m: float, phi_tol_rad: float):
        """
        Smallest N for the series delta_r = R0 * (sqrt(1+t) - 1) such that the
        first omitted term keeps the phase error within tolerance:
        k * R0 * |C_{N+1}| * t_max^(N+1) <= phi_tol_rad.

        Args:
            a_max_m (float): Max atom distance from origin, meters.
            R0_min_m (float): Min detector pixel distance from origin, meters.
            phi_tol_rad (float): Max phase error, radians.

        Returns:
            dict: 'use_series' (bool), 'N' (int, terms to keep), 't_max'
            (float, worst-case |t|). use_series is False when t_max >= 1 or
            no N <= 32 meets the tolerance.
        """
        if getattr(self, "_wavelength", None) is None or self._wavelength <= 0.0:
            # No wavelength: fall back to EXACT
            return dict(use_series=False, N=0, t_max=float("inf"))

        k_val = 2.0 * np.pi / float(self._wavelength)  # rad/m

        if R0_min_m <= 0.0:
            return dict(use_series=False, N=0, t_max=float("inf"))

        # Worst-case |t| for sqrt(1+t) with |a| <= a_max: t = 2*rho + rho^2
        rho = float(a_max_m) / float(R0_min_m)
        t_max = 2.0 * rho + rho * rho

        # Convergence requires |t| < 1
        if not (t_max < 1.0):
            return dict(use_series=False, N=0, t_max=t_max)

        # Coefficients of sqrt(1+t) - 1: C1 = 1/2, Ck = C_{k-1} * (1/2 - (k-1)) / k
        Nmax = 32
        C = 0.5  # C1
        t = float(t_max)
        tk = t  # t^1

        # |Ck| * t^k for k = 1..Nmax
        coeff_pow = [(abs(C) * tk, 1)]  # (value, k)

        for k in range(2, Nmax + 1):
            num = 0.5 - (k - 1.0)
            C = C * (num / k)
            tk = tk * t
            coeff_pow.append((abs(C) * tk, k))

        use_series = False
        chosen_N = 0
        for N in range(1, Nmax):
            val, kpow = coeff_pow[N]  # term k = N+1, the first omitted
            err_r = R0_min_m * val
            err_phi = k_val * err_r
            if err_phi <= float(phi_tol_rad):
                use_series = True
                chosen_N = N  # keep exactly N terms
                break

        if not use_series:
            # Tolerance not met by Nmax terms: be conservative and use EXACT
            return dict(use_series=False, N=0, t_max=t_max)

        # Clamp
        if chosen_N < 1: chosen_N = 1
        if chosen_N > Nmax: chosen_N = Nmax

        return dict(use_series=True, N=chosen_N, t_max=t_max)
        
    def _require_forward_beam(self, k_vec=None):
        """Refuse a beam that is not along +x.

        Both interaction kernels take |k| from the x component alone
        (``fabsf(kx_atom[0])``) and the incident phase as k * x; ky and kz are
        accepted but never read.  ``create_beam`` forces +x, but ``load_beam``
        restores whatever the metadata holds, and a tilted beam would then be
        simulated silently as +x.

        Args:
            k_vec (tuple or None): (kx, ky, kz) to test; the beam's own
                scalars when None.

        Raises:
            ValueError: If the beam is not (to 1e-6 relative) along +x.
        """
        kx, ky, kz = ((self._kx_scalar, self._ky_scalar, self._kz_scalar)
                      if k_vec is None else k_vec)
        kx, ky, kz = float(kx), float(ky), float(kz)
        if kx <= 0.0 or abs(ky) > 1e-6 * kx or abs(kz) > 1e-6 * kx:
            raise ValueError(
                "The scattering kernels assume a beam along +x (they take |k| "
                f"from kx and the incident phase as k*x); got k = ({kx:.4e}, "
                f"{ky:.4e}, {kz:.4e}) 1/m.  Rotate the sample and detector "
                "with the stage instead of tilting the beam.")

    def _select_series_mode_once(self, sample, detector, safety_t_thresh=0.5, verbose=True):
        """
        Pick the global mode (SERIES vs EXACT) and series order N from the
        sample extent and the closest detector pixel.

        Args:
            sample: Has dimensions (Lx, Ly, Lz) in Angstrom, centered at 0.
            detector: Has pixel_coordinates of shape (3, Ny*Nz) in Angstrom.
            safety_t_thresh (float): Use EXACT unless worst-case |t| is below it.
            verbose (bool): Print the chosen mode and N.

        Sets self._global_use_series and self._series_terms. Uses
        self._phase_tol_rad (1e-3 rad if unset).
        """
        # Sample half-diagonal radius (meters)
        dims_A = np.asarray(sample.dimensions, dtype=float)
        half_A = 0.5 * dims_A
        a_max_A = float(np.sqrt(np.sum(half_A**2)))
        a_max_m = a_max_A * 1e-10

        # Closest detector pixel distance (meters)
        pix = detector.pixel_coordinates
        if (cp is not None) and isinstance(pix, cp.ndarray):
            pix_cpu = pix.get()
        else:
            pix_cpu = np.asarray(pix)
        r2_min_A2 = float(np.min(np.sum(pix_cpu * pix_cpu, axis=0)))
        R0_min_m = (r2_min_A2 ** 0.5) * 1e-10

        phi_tol = float(getattr(self, "_phase_tol_rad", 1e-3))

        est = self._estimate_required_series_terms(a_max_m, R0_min_m, phi_tol)
        use_series = bool(est["use_series"])
        N_auto = int(est["N"])
        t_max = float(est["t_max"])

        # Reject marginal series
        if not (R0_min_m > 0.0) or not (t_max < safety_t_thresh):
            use_series = False

        self._global_use_series = use_series
        self._series_terms = (N_auto if (use_series and N_auto >= 1) else 1)

        if verbose:
            mode_str = "SERIES" if use_series else "EXACT"

    @staticmethod
    def _morton_order_gpu(pos_g):
        """Z-order (Morton) permutation for atom positions on the device.

        Per-chunk local origins only help when each CHUNK_SIZE block is
        spatially compact; a randomly ordered list gives chunks that span the
        whole sample and no gain at all.

        Args:
            pos_g (cupy.ndarray): Shape (N, 3), float32, positions in metres.

        Returns:
            cupy.ndarray or None: int64 permutation, or None when the atoms
            are already in Morton order (sort and gather are then skipped).
        """
        n = int(pos_g.shape[0])
        if n < 2:
            return None
        lo = pos_g.min(axis=0)
        hi = pos_g.max(axis=0)
        span = cp.where(hi - lo > 0, hi - lo, cp.float32(1.0))
        # One fused kernel: quantise and interleave in registers, so the only
        # whole-array temporaries are the 8 B/atom code and argsort scratch.
        if beam._MORTON_KERNEL is None:
            beam._MORTON_KERNEL = cp.ElementwiseKernel(
                "float32 x, float32 y, float32 z, "
                "float32 lx, float32 ly, float32 lz, "
                "float32 sx, float32 sy, float32 sz",
                "uint64 code",
                r"""
                const float q = 2097151.0f;                 // (1 << 21) - 1
                unsigned long long a = (unsigned long long)((x - lx) / sx * q);
                unsigned long long b = (unsigned long long)((y - ly) / sy * q);
                unsigned long long c = (unsigned long long)((z - lz) / sz * q);
                #define SPREAD(v) \
                    v &= 0x1FFFFFull; \
                    v = (v | (v << 32)) & 0x1F00000000FFFFull; \
                    v = (v | (v << 16)) & 0x1F0000FF0000FFull; \
                    v = (v | (v << 8))  & 0x100F00F00F00F00Full; \
                    v = (v | (v << 4))  & 0x10C30C30C30C30C3ull; \
                    v = (v | (v << 2))  & 0x1249249249249249ull;
                SPREAD(a) SPREAD(b) SPREAD(c)
                #undef SPREAD
                code = a | (b << 1) | (c << 2);
                """,
                "mosaic_morton_code")
        code = beam._MORTON_KERNEL(
            pos_g[:, 0], pos_g[:, 1], pos_g[:, 2],
            lo[0], lo[1], lo[2], span[0], span[1], span[2])
        # Compare elementwise, NOT via cp.diff: these are uint64, so a
        # decreasing step wraps to a huge positive value, ``diff >= 0`` is
        # always true, and the sort (and with it the whole local-origin
        # scheme) is silently skipped.
        if bool((code[1:] >= code[:-1]).all()):
            return None
        return cp.argsort(code)

    @staticmethod
    def _chunk_origins_gpu(pos_m_g, chunk_size):
        """Per-chunk bounding-box centres and offsets relative to them.

        Origins are rounded to float32 before the offsets are formed, so the
        kernel sees exactly the value that was subtracted.

        Args:
            pos_m_g (cupy.ndarray): Shape (N, 3), float32, metres, already in
                kernel order.
            chunk_size (int): Kernel CHUNK_SIZE, i.e. atoms per origin.

        Returns:
            tuple: (off_g (N, 3) float32 offsets in metres,
                    org_g (nch * 3,) float32 origins in metres, raveled).
        """
        # Everything stays float32 and nothing of size N is duplicated.  The
        # origin is (min + max) / 2 of the float32 extrema, formed in float64
        # on the tiny (nch, 3) arrays and rounded once to float32; the offset
        # is a single correctly rounded float32 subtraction.  The partial last
        # chunk uses the bounding box of its real atoms.
        pos_m_g = cp.ascontiguousarray(pos_m_g, dtype=cp.float32)
        n = int(pos_m_g.shape[0])
        nch = (n + chunk_size - 1) // chunk_size
        nfull = n // chunk_size
        org = cp.empty((nch, 3), dtype=cp.float32)
        off = cp.empty_like(pos_m_g)
        if nfull:
            blk = pos_m_g[:nfull * chunk_size].reshape(nfull, chunk_size, 3)
            lo = blk.min(axis=1).astype(cp.float64)
            hi = blk.max(axis=1).astype(cp.float64)
            org[:nfull] = ((lo + hi) * 0.5).astype(cp.float32)
            cp.subtract(blk, org[:nfull, None, :],
                        out=off[:nfull * chunk_size].reshape(nfull, chunk_size, 3))
        if nfull < nch:
            rest = pos_m_g[nfull * chunk_size:]
            lo = rest.min(axis=0).astype(cp.float64)
            hi = rest.max(axis=0).astype(cp.float64)
            org[nfull] = ((lo + hi) * 0.5).astype(cp.float32)
            cp.subtract(rest, org[nfull][None, :], out=off[nfull * chunk_size:])
        return off, cp.ascontiguousarray(org.ravel())


    @staticmethod
    def _species_tables(spc, db_f0, f0_zero_lookup, anom_lookup):
        """Collapse per-atom species labels to codes plus per-species tables.

        The fast kernel takes the form-factor coefficients, anomalous term and
        f0(0) as per-species launch constants, so it needs one table row per
        species and an index per atom.

        Args:
            spc (numpy.ndarray): (N,) species labels, in kernel atom order.
            db_f0 (dict): element -> 11 Waasmaier-Kirfel coefficients.
            f0_zero_lookup (dict): element -> f0 at Q = 0.
            anom_lookup (dict): element -> complex f' + i f''.

        Returns:
            tuple: (code (N,) int32, wk (S, 11), f0z (S,), anom (S,) complex).
        """
        code, names = beam._species_codes(spc)
        wk, f0z, anom = beam._species_rows(names, db_f0, f0_zero_lookup,
                                           anom_lookup)
        return code.astype(np.int32), wk, f0z, anom

    @staticmethod
    def _species_codes(spc):
        """(code (N,) int32, names (S,)) with code[i] indexing names.

        Labels that fit one 64-bit word (the usual <U2 symbols) are factorized
        as integers, far faster than a sort-based unique over 1e8+ labels;
        anything wider falls back to np.unique.
        """
        spc = np.asarray(spc)
        if spc.dtype.kind != "U":
            spc = spc.astype(str)
        spc = np.ascontiguousarray(spc)
        if spc.dtype.itemsize == 8:
            code, keys = pd.factorize(spc.view(np.uint64))
            names = np.ascontiguousarray(np.asarray(keys, dtype=np.uint64)).view(spc.dtype)
        else:
            names, code = np.unique(spc, return_inverse=True)
        return code.astype(np.int32), names

    @staticmethod
    def _species_rows(names, db_f0, f0_zero_lookup, anom_lookup):
        """Per-species form-factor rows in the order of ``names``."""
        wk = np.zeros((names.size, 11), dtype=np.float64)
        f0z = np.zeros(names.size, dtype=np.float64)
        anom = np.zeros(names.size, dtype=np.complex128)
        for i, el in enumerate(names):
            el_s = str(el)
            p = db_f0.get(el_s)
            if p is not None:
                wk[i] = np.asarray(p, dtype=np.float64)
                f0z[i] = float(f0_zero_lookup.get(el_s, 0.0))
            anom[i] = complex(anom_lookup.get(el_s, 0.0 + 0.0j))
        return wk, f0z, anom

    def _fast_scatter(self, pos_m_g, amps_host, spc, db_f0, f0_zero_lookup,
                      anom_lookup, xg, yg, zg, dfield_gpu, Ny, Nz, k_global,
                      m_beams, analyser_kind, remove_forward, polarization,
                      decay, pol_rate, scatter_chunk, det_extent=None,
                      species_key=None):
        """Try the compile-time specialised fast kernel for one chunk.

        Every optional term is a compile-time constant there, and the series
        order, form-factor fit, register blocking and stream count are chosen
        from this geometry.  It declines anything it cannot represent exactly;
        nothing is written then and the caller runs the general kernel.

        Returns:
            bool: True if the chunk was scattered here.
        """
        if not getattr(self, "_use_fast_kernel", True):
            return False
        try:
            import fast_kernel as _fk
        except Exception as exc:                       # pragma: no cover
            self._log("verbose", f"[beam] fast path unavailable: {exc}")
            return False
        ok, why = _fk.applicable(m_beams, analyser_kind,
                                 getattr(self, "_global_use_series", True))
        if not ok:
            self._log("verbose", f"[beam] fast path declined: {why}")
            return False

        # Species codes depend only on the chunk's label array, which a scan
        # re-reads unchanged every step, so they are cached by file identity.
        # The per-species rows depend on the energy and are cheap: rebuilt
        # every call.
        cache = self.__dict__.setdefault("_species_code_cache", {})
        hit = cache.get(species_key) if species_key is not None else None
        if hit is not None and hit[0].shape[0] == int(pos_m_g.shape[0]):
            code, names = hit
        else:
            code, names = self._species_codes(spc)
            if species_key is not None:
                if len(cache) >= 8:
                    cache.pop(next(iter(cache)))
                cache[species_key] = (code, names)
        wk, f0z, anom = self._species_rows(names, db_f0, f0_zero_lookup,
                                           anom_lookup)
        chunk = int(getattr(self, "_kernel_chunk_size", 128))
        # Use the tighter of the fast path's default tolerance and what
        # set_phase_tolerance() asked for, as the general kernel does.
        tol_user = float(getattr(self, "_phase_tol_rad", 1e-3))
        tol = min(float(getattr(self, "_kernel_phase_tol_rad", 1e-6)), tol_user)
        tol_f0 = float(getattr(self, "_kernel_f0_tol", 1e-7))
        phasor_fp64 = bool(getattr(self, "_kernel_phasor_fp64", True))
        n = int(pos_m_g.shape[0])
        if n == 0:
            return True

        # Decide the form-factor fit for EVERY species up front: a species
        # that first appears in a later sub-chunk must not decline after
        # earlier sub-chunks have already been accumulated.
        if det_extent is None:
            det_extent = _fk.detector_extent(xg, yg, zg)
        R0_min, dv_lo, dv_hi = det_extent
        radius = float(cp.linalg.norm(pos_m_g, axis=1).max())
        ok, worst, npan = _fk.check_f0_fit(
            wk, *_fk.fit_interval(dv_lo, dv_hi, radius, R0_min),
            float(k_global), tol_f0)
        if not ok:
            self._log("verbose",
                      f"[beam] fast path declined: f0 fit reaches {worst:.1e} "
                      f"> {tol_f0:.0e} over this detector ({npan} panels)")
            return False
        # Series truncation: with local origins (R0 >= 1 mm) the argument is
        # set by the ~nm chunk radius and cannot miss; without them it is set
        # by the sample radius, so run dispatch's test once here rather than
        # letting it decline mid-way through the sub-chunks.
        if not _fk.use_local_origins(radius, R0_min):
            nser, ser_err = _fk.series_order(radius, R0_min, tol, float(k_global))
            if ser_err > max(tol, tol_user):
                self._log("verbose",
                          f"[beam] fast path declined: series error {ser_err:.1e} "
                          f"> {max(tol, tol_user):.0e} at order {nser} "
                          f"(R0_min={R0_min:.2e} m)")
                return False

        amps = amps_host[:, 0] if getattr(amps_host, "ndim", 1) > 1 else amps_host
        for c0 in range(0, n, scatter_chunk):
            c1 = min(c0 + scatter_chunk, n)
            sl = amps[c0:c1]
            if isinstance(sl, cp.ndarray):
                amp_g = cp.ascontiguousarray(sl.astype(cp.complex64))
            else:
                amp_g = cp.asarray(
                    np.ascontiguousarray(sl).astype(np.complex64))
            # No host-side hold on amp_g and no synchronize: CuPy's pool is
            # stream-ordered, and dispatch joins its side streams back to
            # this one before returning.
            handled = _fk.dispatch(
                pos_m_g[c0:c1], amp_g, code[c0:c1], wk, f0z, anom,
                xg, yg, zg, dfield_gpu, Ny, Nz, float(k_global),
                polarization=bool(polarization), pol_rate=float(pol_rate),
                decay=bool(decay), remove_forward=bool(remove_forward),
                m_beams=m_beams, analyser_kind=analyser_kind,
                use_series=getattr(self, "_global_use_series", True),
                chunk=chunk, tol_rad=tol, det_extent=det_extent,
                tol_f0=tol_f0, phasor_fp64=phasor_fp64, tol_decline=tol_user)
            if not handled:            # cannot happen mid-way: checked above
                return False
        return True

    def _scatter_subchunk_size(self, M=1, resident_bytes=0):
        """Atoms per scatter sub-chunk, so the per-atom staging fits in memory.

        Budget ~150 transient device bytes per atom (sorted positions, Morton
        scratch, offsets, form-factor tables, k vectors) plus 8 B x M for the
        amplitude slice.  Counting only the resident arrays under-estimates
        this ~3x and over-commits the card.

        Args:
            M (int): beam channels (amplitude columns).
            resident_bytes (int): what the caller already holds for the whole
                chunk while sub-chunks stream, so it is not counted twice.

        Returns:
            int: atoms per sub-chunk, a multiple of the kernel CHUNK_SIZE.
            The kernel indexes chunk origins launch-locally, so a sub-chunk
            starting off a CHUNK_SIZE boundary would read a neighbour's
            origin and produce a plausible-looking but wrong pattern.
        """
        chunk = int(getattr(self, "_kernel_chunk_size", 128))
        if getattr(self, "_scatter_chunk_override", None):
            n = int(self._scatter_chunk_override)
        else:
            try:
                free_gpu_b, _ = cp.cuda.runtime.memGetInfo()
                budget = 0.8 * free_gpu_b - float(resident_bytes)
                n = int(min(50_000_000,
                            max(500_000, budget // (150 + 8 * int(M)))))
            except Exception:
                n = 500_000
        return max(chunk, (n // chunk) * chunk)

    def _stage_general_subchunk(self, pos_sub, spc_sub, amp_rows, use_origins,
                                chunk_size, db_f0, f0_zero_lookup, anom_lookup,
                                k_in):
        """Stage one scatter sub-chunk for the general kernel.

        Everything per-atom -- the Z-order sort, local origins, form-factor
        tables, k vectors and amplitude slice -- is built here, sized to the
        sub-chunk, so the device-memory bound actually holds.

        Args:
            pos_sub (cupy.ndarray): (n, 3) float32 absolute positions, metres.
            spc_sub (numpy.ndarray): (n,) species labels, same order.
            amp_rows (numpy.ndarray): (n, M) incident amplitudes, same order.
            use_origins (bool): Sort and use per-CHUNK_SIZE local origins.
            chunk_size (int): Kernel CHUNK_SIZE.
            k_in (tuple): (kx, ky, kz) in 1/m.

        Returns:
            dict: device arrays keyed n, px, py, pz, kx, ky, kz, f0p, f0z,
            anom, amp, org.  ``org`` starts at this sub-chunk's first chunk,
            which the kernel's launch-local ``base / CHUNK`` index expects.
        """
        order_g = self._morton_order_gpu(pos_sub) if use_origins else None
        if order_g is not None:
            pos_sub = pos_sub[order_g]
            order_np = cp.asnumpy(order_g)
            del order_g
            spc_sub = spc_sub[order_np]
            amp_rows = amp_rows[order_np]
        if use_origins:
            off_g, org_d = self._chunk_origins_gpu(pos_sub, chunk_size)
        else:
            off_g, org_d = pos_sub, cp.zeros((3,), dtype=cp.float32)
        n = int(off_g.shape[0])
        px = cp.ascontiguousarray(off_g[:, 0])
        py = cp.ascontiguousarray(off_g[:, 1])
        pz = cp.ascontiguousarray(off_g[:, 2])
        del off_g, pos_sub

        f0p_host = np.zeros((n, 11), dtype=np.float32)
        f0z_host = np.zeros(n, dtype=np.float32)
        anom_host = np.zeros(n, dtype=np.complex64)
        for el in np.unique(spc_sub):
            el_s = str(el)
            mask_np = (spc_sub == el_s)
            f0p = db_f0.get(el_s)
            if f0p is not None:
                f0p_host[mask_np] = f0p
                f0z_host[mask_np] = f0_zero_lookup.get(el_s, 0.0)
            if el_s in anom_lookup:
                anom_host[mask_np] = anom_lookup[el_s]
        kx, ky, kz = (cp.full((n,), kc, dtype=cp.float32) for kc in k_in)
        if isinstance(amp_rows, cp.ndarray):          # already on the device
            amp_d = cp.ascontiguousarray(amp_rows.reshape(-1))
        else:
            amp_d = cp.asarray(np.ascontiguousarray(amp_rows).reshape(-1))
        return dict(n=n, px=px, py=py, pz=pz, kx=kx, ky=ky, kz=kz,
                    f0p=cp.asarray(f0p_host), f0z=cp.asarray(f0z_host),
                    anom=cp.asarray(anom_host), amp=amp_d, org=org_d)

    def _origin_policy(self, pos_m_g, R0_min):
        """Decide whether per-chunk local origins should be used.

        The shift helps in the far field and the Fresnel zone but hurts in
        the extreme near field, and the failure tracks the ABSOLUTE camera
        length, not just the geometric ratio: the double-float prologue that
        resolves |r0| is itself evaluated against a small R0.  Both
        conditions therefore have to hold.

        Args:
            pos_m_g (cupy.ndarray): Shape (N, 3), float32, metres.
            R0_min (float): Smallest sample-to-pixel distance in metres.

        Returns:
            tuple: (enabled (bool), eps_geom (float) = L_sample / R0_min).
        """
        forced = getattr(self, "_use_chunk_origins", None)
        L_s = float(cp.linalg.norm(pos_m_g, axis=1).max())   # policy input,
        # not a phase: float32 is plenty
        eps_geom = L_s / R0_min if R0_min > 0 else float("inf")
        if forced is not None:
            return bool(forced), eps_geom
        return (eps_geom <= 1e-2 and R0_min >= 1e-3), eps_geom

    def build_interaction_kernel(self, series_terms: int | None = None,
                                 force_mode: str | None = None,
                                 m_beams: int = 1):
        """
        Build and cache the FP32-only kinematic interaction CUDA kernel.

        Args:
            series_terms (int or None): Terms in the sqrt(1+t)-1 expansion,
                clamped to [1, 32]. None uses self._series_terms.
            force_mode (str or None): "series" or "exact"; None uses
                self._global_use_series.
            m_beams (int): Beam channels per atom compiled into the kernel.

        Returns:
            cupy.RawKernel: Compiled interaction kernel.

        Note:
            Analyser parameters are launch arguments: apply_analyser (0/1),
            analyser_kind (0=off, 1=top_hat, 2=darwin), centre_dir (unit
            vector, origin -> detector centre), accept_angle_rad and
            darwin_halfwidth_rad.  Each event's amplitude is scaled by
                - top_hat: 1 within accept_angle_rad of centre_dir, else 0
                - darwin: 1 / (1 + (delta / darwin_halfwidth_rad)^2)
        """
        if cp is None:
            raise RuntimeError("CuPy is required for GPU scattering kernels.")

        if series_terms is None:
            N = int(getattr(self, "_series_terms", 2))
        else:
            N = int(series_terms)
        if N < 1: N = 1
        if N > 32: N = 32

        if force_mode is not None:
            use_series = (str(force_mode).lower() == "series")
        else:
            use_series = bool(getattr(self, "_global_use_series", True))

        global_use_series = 1 if use_series else 0

        M_compile = int(max(1, m_beams))

        # Launch-tuning knobs: scheduling only.  None change the arithmetic
        # or its order, so results are bit-identical across settings.
        n_pro      = int(getattr(self, "_kernel_n_pro", 8))
        pro_mode   = int(getattr(self, "_kernel_pro_mode", 1))
        stage_c    = int(bool(getattr(self, "_kernel_stage_c", False)))
        chunk_size = int(getattr(self, "_kernel_chunk_size", 128))
        unroll     = int(getattr(self, "_kernel_unroll", 4))
        maxreg     = getattr(self, "_kernel_maxreg", None)

        # Cache by (N, mode, M, tuning)
        if not hasattr(self, "_interaction_kernel_cache"):
            self._interaction_kernel_cache = {}
        key = ("v3_dynamical", N, global_use_series, M_compile,
               chunk_size, unroll, maxreg, n_pro, pro_mode, stage_c)
        if key in self._interaction_kernel_cache:
            return self._interaction_kernel_cache[key]

        _cuda_source = r'''
        #include <math.h>

        // Compile-time settings
        #ifndef N_SERIES
        #define N_SERIES 2
        #endif
        #if N_SERIES < 1
        #undef N_SERIES
        #define N_SERIES 1
        #endif
        #if N_SERIES > 32
        #undef N_SERIES
        #define N_SERIES 32
        #endif

        #ifndef GLOBAL_USE_SERIES
        #define GLOBAL_USE_SERIES 1
        #endif

        // Binomial terms used by the per-chunk origin prologue (k>=2 only;
        // the k=1 term is carried exactly in double-float).
        #ifndef N_PRO
        #define N_PRO 8
        #endif

        // 1 = exact algebraic dR (default), 2 = binomial series.
        #ifndef PROLOGUE_MODE
        #define PROLOGUE_MODE 1
        #endif

        // Stage C: factor the amplitude terms out of the atom loop.  OFF by
        // default; 0 is bit-identical to the unfactored form.
        #ifndef STAGE_C
        #define STAGE_C 0
        #endif

        // M_BEAMS_COMPILE is the per-atom amplitude vector length.  M=1
        // reproduces the kinematic kernel bit-identically (the lattice-phase
        // factor is 1 for the forward beam).
        #ifndef M_BEAMS_COMPILE
        #define M_BEAMS_COMPILE 1
        #endif
        #if M_BEAMS_COMPILE < 1
        #undef M_BEAMS_COMPILE
        #define M_BEAMS_COMPILE 1
        #endif

        #ifndef CHUNK_SIZE
        #define CHUNK_SIZE 128
        #endif
        #ifndef UNROLL_ATOMS
        #define UNROLL_ATOMS 4
        #endif

        // Split constants for robust FP32 argument reduction
        #define TWOPI_H 6.2831854820251465f
        #define TWOPI_L -1.748455531469517e-07f
        #define INV_TWOPI_H 0.15915493667125702f
        #define PI_H 3.1415927410125732f

        extern "C" {

        __device__ __forceinline__ void two_prod_fma(float a, float b, float& p, float& e)
        {
            p = a * b;
            e = fmaf(a, b, -p);
        }

        // FP32-robust sincos(2*pi * g . r_a).  g.r_a can reach ~1e7 cycles
        // for sample-scale crystals, so each product is split exactly with
        // two_prod_fma, the high parts are two-summed, the total is reduced
        // modulo one cycle, and the fraction is scaled by the TWOPI_H/TWOPI_L
        // split before __sincosf (|arg| <= pi).  For g = 0 this returns
        // exactly (sn=0, cs=1), keeping M=1 bit-identical to kinematic.
        __device__ __forceinline__ void sincos_2pi_dot_g_r(
            float gx, float gy, float gz,
            float ax, float ay, float az,
            float& sn, float& cs)
        {
            // exact-split component products
            float hx, lx; two_prod_fma(gx, ax, hx, lx);
            float hy, ly; two_prod_fma(gy, ay, hy, ly);
            float hz, lz; two_prod_fma(gz, az, hz, lz);

            // two-sum the high parts
            float s1 = hx + hy;
            float bb = s1 - hx;
            float e1 = (hx - (s1 - bb)) + (hy - bb);

            float s2 = s1 + hz;
            bb = s2 - s1;
            float e2 = (s1 - (s2 - bb)) + (hz - bb);

            // total = s2 (high) + low (small)
            float low = e1 + e2 + lx + ly + lz;

            // reduce modulo 1 cycle
            float q = nearbyintf(s2 + low);
            float frac = (s2 - q) + low;          // |frac| <= 0.5 cycles

            // scale by the 2*pi double-split; |arg| <= pi is accurate for __sincosf
            float arg = fmaf(frac, TWOPI_H, frac * TWOPI_L);
            __sincosf(arg, &sn, &cs);
        }

        // Robust sincos of (k*s) with FP32-only modulo-2pi reduction
        __device__ __forceinline__ void sincos_k_times_reduced(float k, float s, float& sn, float& cs)
        {
            float xh, xl;
            two_prod_fma(k, s, xh, xl);

            float q = nearbyintf(fmaf(xh, INV_TWOPI_H, xl * INV_TWOPI_H));

            float r = fmaf(-q, TWOPI_H, xh);
            r = fmaf(-q, TWOPI_L, r);
            r = r + xl;

            if (r > PI_H)       r = fmaf(-1.0f, TWOPI_H, r);
            else if (r < -PI_H) r = fmaf( 1.0f, TWOPI_H, r);

            __sincosf(r, &sn, &cs);
        }

        // ---- double-float (unevaluated fp32 pair) helpers -----------------
        // FP32 throughout; none of this emits an FP64 instruction.
        __device__ __forceinline__ void two_sum(float a, float b, float& s, float& e)
        {
            s = a + b;
            float bb = s - a;
            e = (a - (s - bb)) + (b - bb);
        }

        // Exact dot product of two fp32 vectors as an unevaluated hi+lo pair.
        __device__ __forceinline__ void df_dot3(
            float ax, float ay, float az, float bx, float by, float bz,
            float& hi, float& lo)
        {
            float hx, lx; two_prod_fma(ax, bx, hx, lx);
            float hy, ly; two_prod_fma(ay, by, hy, ly);
            float hz, lz; two_prod_fma(az, bz, hz, lz);
            float s1, e1; two_sum(hx, hy, s1, e1);
            float s2, e2; two_sum(s1, hz, s2, e2);
            hi = s2;
            lo = ((e1 + e2) + (lx + ly)) + lz;
        }

        // As sincos_k_times_reduced with a double-float argument.  REQUIRES a
        // normalised pair (|s_lo| << |s_hi|*eps): a large s_lo pushes the
        // reduced argument outside __sincosf's accurate range.
        __device__ __forceinline__ void sincos_k_times_reduced_dd(
            float k, float s_hi, float s_lo, float& sn, float& cs)
        {
            float xh, xl;
            two_prod_fma(k, s_hi, xh, xl);
            xl = fmaf(k, s_lo, xl);

            float q = nearbyintf(fmaf(xh, INV_TWOPI_H, xl * INV_TWOPI_H));
            float r = fmaf(-q, TWOPI_H, xh);
            r = fmaf(-q, TWOPI_L, r);
            r = r + xl;

            if (r > PI_H)       r = fmaf(-1.0f, TWOPI_H, r);
            else if (r < -PI_H) r = fmaf( 1.0f, TWOPI_H, r);
            __sincosf(r, &sn, &cs);
        }

        // Waasmaier-Kirfel f0(Q)
        __device__ __forceinline__ float get_f0_from_params(float Q_val, const float* params)
        {
            const float PI_F   = 3.14159265358979323846f;
            const float K_SCALE= 0.25f * 1.0e-10f / PI_F;  // Q[m^-1] -> s[Angstrom^-1]
            float s  = K_SCALE * Q_val;
            float ss = s * s;

            float f0 = params[5];
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                float ai = params[i];
                float bi = params[6 + i];
                f0 += ai * __expf(-bi * ss);
            }
            return f0;
        }

        // Series for sqrt(1+t) - 1 to N terms.  The repeated-power form is
        // deliberate: C1 = 1/2, so 0.5f*t is exact and the remaining terms
        // enter as fma corrections.  A Horner form rounds an intermediate
        // near 0.5 and again on the final *t, which is ~10x worse at |t|~1e-7.
        __device__ __forceinline__ float sqrt1pm1_series(float t)
        {
            float coeff = 0.5f;  // C1
            float tk    = t;     // t^1
            float poly  = coeff * tk;

            #pragma unroll
            for (int k = 2; k <= N_SERIES; ++k) {
                float kf  = (float)k;
                float num = 0.5f - (kf - 1.0f);
                coeff = coeff * (num / kf);
                tk = tk * t;
                poly = fmaf(coeff, tk, poly);
            }
            return poly;
        }

        // analyser_kind: 0=off, 1=top_hat, 2=darwin
        // g_vecs_in: (M_BEAMS_COMPILE * 3) floats in 1/m, the reciprocal-
        // lattice vectors for the per-atom M-channel coherent sum (Eq. 13 of
        // the dynamical-method plan).  M=1 with g=0 is bit-identical to the
        // kinematic case.  initial_amp has length nAtoms * M_BEAMS_COMPILE,
        // indexed as initial_amp[a * M + m].
        __global__ void interaction_kernal(
            const int   nAtoms,
            const float* __restrict__ kx_atom,
            const float* __restrict__ ky_atom,
            const float* __restrict__ kz_atom,
            const float* __restrict__ px,   // atom positions in meters
            const float* __restrict__ py,
            const float* __restrict__ pz,
            const float2* __restrict__ initial_amp,  // (nAtoms * M) interleaved
            const float2* __restrict__ scattering_anom,
            const float*  __restrict__ f0_params,
            const float*  __restrict__ f0_zero,
            const float* __restrict__ x_coords,  // detector coords in meters
            const float* __restrict__ y_coords,
            const float* __restrict__ z_coords,
            float2*      __restrict__ detector_field,
            const int    Ny,
            const int    Nz,
            const int    remove_forward,
            const int    apply_polarization,
            const int    apply_spherical_decay,
            const float  pol_perp_rate,
            const int    apply_analyser,
            const int    analyser_kind,
            const float  centre_x, const float centre_y, const float centre_z,
            const float  accept_angle_rad,
            const float  darwin_halfwidth_rad,
            const float* __restrict__ g_vecs_in,  // (M * 3) floats in 1/m
            const int    apply_origin,            // 0 = legacy single origin
            const float* __restrict__ chunk_org)  // (nChunks * 3) floats, m
        {
            const float rE_F = 2.81794092e-15f;

            int ix = blockIdx.x * blockDim.x + threadIdx.x;
            int iy = blockIdx.y * blockDim.y + threadIdx.y;
            const bool valid_pixel = (ix < Ny && iy < Nz);
            const int pidx = valid_pixel ? (iy * Ny + ix) : 0;

            // Per-pixel state (initialised only for valid threads)
            float tx = 0.f, ty = 0.f, tz = 0.f;
            float R0 = 0.f, invR0 = 0.f;
            float ux = 0.f, uy = 0.f, uz = 0.f;
            float k_global = 0.f;
            float sb = 0.f, cb = 1.f;
            float Q_cut = 0.f;
            float cx = 0.f, cy = 0.f, cz = 0.f;
            float cos_accept = 0.f;
            float2 sum_rel = make_float2(0.0f, 0.0f);

            // Grid-uniform: with M=1 and g=0 the lattice-phase factor is
            // exactly (sn=0, cs=1), so sincos_2pi_dot_g_r collapses to the
            // stored amplitude.  g_vecs_in is a runtime pointer, so nvcc
            // cannot fold this itself.
            bool g_all_zero = true;
            #pragma unroll
            for (int m = 0; m < M_BEAMS_COMPILE; ++m) {
                g_all_zero = g_all_zero
                    && (g_vecs_in[m*3+0] == 0.0f)
                    && (g_vecs_in[m*3+1] == 0.0f)
                    && (g_vecs_in[m*3+2] == 0.0f);
            }

            if (valid_pixel) {
                tx = x_coords[pidx];
                ty = y_coords[pidx];
                tz = z_coords[pidx];

                // Pixel sightline and unit vector from origin
                R0 = sqrtf(tx*tx + ty*ty + tz*tz);
                if (R0 > 0.0f) {
                    invR0 = 1.0f / R0;
                    ux = tx * invR0;
                    uy = ty * invR0;
                    uz = tz * invR0;
                }

                k_global = fabsf(kx_atom[0]);

                // Base phasor exp(i*k*R0)
                sincos_k_times_reduced(k_global, R0, sb, cb);

                // Q_cut from local pixel size
                {
                    int n_right = (ix + 1 < Ny) ? (pidx + 1) : ((ix > 0) ? (pidx - 1) : pidx);
                    int n_up    = (iy + 1 < Nz) ? (pidx + Ny) : ((iy > 0) ? (pidx - Ny) : pidx);

                    // neighbor +x/-x
                    float rx = x_coords[n_right];
                    float ry = y_coords[n_right];
                    float rz = z_coords[n_right];
                    float Rr = sqrtf(rx*rx + ry*ry + rz*rz);
                    float urx = 0.0f, ury = 0.0f, urz = 0.0f;
                    if (Rr > 0.0f) { float invRr = 1.0f / Rr; urx = rx*invRr; ury = ry*invRr; urz = rz*invRr; }
                    float cos_dx = ux*urx + uy*ury + uz*urz;
                    cos_dx = fminf(1.0f, fmaxf(-1.0f, cos_dx));
                    float Qx = k_global * __fsqrt_rn(fmaxf(0.0f, 2.0f * (1.0f - cos_dx)));

                    // neighbor +y/-y
                    float ux2 = x_coords[n_up];
                    float uy2 = y_coords[n_up];
                    float uz2 = z_coords[n_up];
                    float Ru = sqrtf(ux2*ux2 + uy2*uy2 + uz2*uz2);
                    float vux = 0.0f, vuy = 0.0f, vuz = 0.0f;
                    if (Ru > 0.0f) { float invRu = 1.0f / Ru; vux = ux2*invRu; vuy = uy2*invRu; vuz = uz2*invRu; }
                    float cos_dy = ux*vux + uy*vuy + uz*vuz;
                    cos_dy = fminf(1.0f, fmaxf(-1.0f, cos_dy));
                    float Qy = k_global * __fsqrt_rn(fmaxf(0.0f, 2.0f * (1.0f - cos_dy)));

                    // diagonal half-width in Q (approximate pixel "radius" in Q-space)
                    float Qhx = 0.5f * Qx;
                    float Qhy = 0.5f * Qy;
                    Q_cut = __fsqrt_rn(Qhx*Qhx + Qhy*Qhy);
                }

                // Normalized analyser centre axis
                float cen_norm = sqrtf(centre_x*centre_x + centre_y*centre_y + centre_z*centre_z);
                if (cen_norm > 0.0f) { float invC = 1.0f / cen_norm; cx = centre_x*invC; cy = centre_y*invC; cz = centre_z*invC; }
                cos_accept = cosf(accept_angle_rad);
            }

            // Shared memory: ALL threads, valid pixel or not, must take part
            // in the loads and reach every __syncthreads below.
            __shared__ float  s_px[CHUNK_SIZE];
            __shared__ float  s_py[CHUNK_SIZE];
            __shared__ float  s_pz[CHUNK_SIZE];
            __shared__ float2 s_amp_M[CHUNK_SIZE * M_BEAMS_COMPILE];
            __shared__ float2 s_anm[CHUNK_SIZE];
            __shared__ float  s_params[CHUNK_SIZE * 11];
            __shared__ float  s_f0z[CHUNK_SIZE];

            const int threads_in_block = blockDim.x * blockDim.y;
            const int t_id = threadIdx.y * blockDim.x + threadIdx.x;

            for (int base = 0; base < nAtoms; base += CHUNK_SIZE) {
                for (int t = t_id; t < CHUNK_SIZE; t += threads_in_block) {
                    int a = base + t;
                    if (a < nAtoms) {
                        s_px[t] = px[a]; s_py[t] = py[a]; s_pz[t] = pz[a];
                        #pragma unroll
                        for (int m = 0; m < M_BEAMS_COMPILE; ++m) {
                            s_amp_M[t * M_BEAMS_COMPILE + m] =
                                initial_amp[a * M_BEAMS_COMPILE + m];
                        }
                        s_anm[t]= scattering_anom[a];
                        s_f0z[t]= f0_zero[a];
                        #pragma unroll
                        for (int j=0;j<11;++j)
                            s_params[t*11 + j] = f0_params[a*11 + j];
                    }
                }
                __syncthreads();

                // ---- per-(chunk, pixel) local origin ------------------------
                // Atom positions arrive as offsets d = r_a - r0 for this chunk.
                // Everything of magnitude |r0| is resolved here, once per chunk
                // per pixel, in double-float, so the atom loop only rounds
                // quantities of magnitude |d| and the FP32 phase floor becomes
                // k*|d|*eps instead of k*|r_a|*eps.
                const int cidx = base / CHUNK_SIZE;
                float s0 = 0.0f, c0 = 1.0f;
                float R0p = R0, invR0p = invR0;
                float upx = ux, upy = uy, upz = uz;
                float qx = tx, qy = ty, qz = tz;

                if (apply_origin && valid_pixel) {
                    float ox = chunk_org[cidx*3 + 0];
                    float oy = chunk_org[cidx*3 + 1];
                    float oz = chunk_org[cidx*3 + 2];

                    // R0 = |t| as a double-float
                    float tt_h, tt_l; df_dot3(tx,ty,tz, tx,ty,tz, tt_h, tt_l);
                    float R0h    = sqrtf(tt_h);
                    float invR0h = 1.0f / R0h;
                    float R0l    = (fmaf(-R0h, R0h, tt_h) + tt_l) * (0.5f * invR0h);

                    float td_h, td_l; df_dot3(tx,ty,tz, ox,oy,oz, td_h, td_l);
                    float oo_h, oo_l; df_dot3(ox,oy,oz, ox,oy,oz, oo_h, oo_l);

                    // s = (t . r0)/R0 as a double-float.  This split sets the
                    // floor: with sp0h alone the fp32 rounding of an |r0|-sized
                    // quantity dominates however many series terms are summed.
                    float sp0h = td_h * invR0h;
                    float sp0l = (fmaf(-sp0h, R0h, td_h)
                                  + (td_l - sp0h * R0l)) * invR0h;

                    float dR_h, dR_l;
#if PROLOGUE_MODE == 2
                    // Binomial series: dR = R0*(sqrt(1+t)-1) with
                    //   t = -2 s/R0 + |r0|^2/R0^2; the k=1 term is -s + |r0|^2/(2 R0).
                    // LIMITATION: |r0|^2/(2 R0) is formed in plain fp32, so
                    // this floors at k*|r0|^2*eps/(2 R0) regardless of N_PRO.
                    // Fine for sub-micron samples, not general.
                    float tv = fmaf(oo_h, invR0h * invR0h, -2.0f * sp0h * invR0h);
                    float coeff = 0.5f, tk = tv, Qs = 0.0f;
                    #pragma unroll
                    for (int kk = 2; kk <= N_PRO; ++kk) {
                        float kf = (float)kk;
                        coeff = coeff * ((0.5f - (kf - 1.0f)) / kf);
                        tk = tk * tv;
                        Qs = fmaf(coeff, tk, Qs);
                    }
                    dR_h = -sp0h;
                    dR_l = -sp0l + fmaf(oo_h, 0.5f * invR0h, R0h * Qs);
#else
                    // Exact algebraic form (default):
                    //   dR = (|r0|^2 - 2 t.r0) / (|t - r0| + |t|)
                    // No series, no truncation.  The numerator cancels when
                    // dR ~ 0, but it is carried as a double-float so dR stays
                    // good to |r0| * eps^2.
                    float s_, e_;
                    two_sum(tt_h, -2.0f * td_h, s_, e_);
                    float a_h = s_, a_l = e_ + (tt_l - 2.0f * td_l);
                    two_sum(a_h, oo_h, s_, e_);
                    float w2h = s_, w2l = (e_ + a_l) + oo_l;

                    float Rph   = sqrtf(w2h);
                    float invRph = 1.0f / Rph;
                    float Rpl   = (fmaf(-Rph, Rph, w2h) + w2l) * (0.5f * invRph);

                    two_sum(oo_h, -2.0f * td_h, s_, e_);
                    float num_h = s_, num_l = e_ + (oo_l - 2.0f * td_l);
                    two_sum(Rph, R0h, s_, e_);
                    float den_h = s_, den_l = (e_ + Rpl) + R0l;
                    float invden = 1.0f / den_h;
                    dR_h = num_h * invden;
                    dR_l = (fmaf(-dR_h, den_h, num_h)
                            + (num_l - dR_h * den_l)) * invden;
#endif

                    // Chunk phasor exp(i k (r0x + dR)).  The two_sum
                    // renormalises the pair before the reduction; required.
                    float ph_h, ph_e; two_sum(ox, dR_h, ph_h, ph_e);
                    sincos_k_times_reduced_dd(k_global, ph_h, ph_e + dR_l, s0, c0);

                    // Rebase the pixel geometry on the shifted origin.  fp32 is
                    // sufficient from here because |d| is small.
                    R0p    = R0h + (dR_h + dR_l);
                    invR0p = 1.0f / R0p;
                    qx = tx - ox; qy = ty - oy; qz = tz - oz;
                    upx = qx * invR0p; upy = qy * invR0p; upz = qz * invR0p;
                }

                float2 csum = make_float2(0.0f, 0.0f);

#if STAGE_C
                // ---- Stage C: per-(chunk, pixel) scattering prefactor -------
                // f0(Q), anomalous terms, polarization, forward removal and
                // r_e are constant over a chunk to within 2.4 * |d| / R0, so
                // they are applied once when the chunk sum is folded in.  The
                // error scales with the chunk radius (~1 nm), not the sample.
                // REQUIRES a single species per launch: parameters are read
                // at atom index 0.
                float2 stot_c = make_float2(1.0f, 0.0f);
                if (valid_pixel) {
                    float dotv_c = upx;                       // +x incidence
                    float tmp_c  = 2.0f * (1.0f - dotv_c);
                    if (tmp_c < 0.0f) tmp_c = 0.0f;
                    float Q_c = (tmp_c > 0.0f)
                              ? (k_global * (tmp_c * rsqrtf(tmp_c))) : 0.0f;
                    float f0c = get_f0_from_params(Q_c, f0_params);
                    stot_c.x = f0c + scattering_anom[0].x;
                    stot_c.y = scattering_anom[0].y;
                    if (remove_forward && (Q_c < Q_cut)) {
                        stot_c.x -= (f0_zero[0] + scattering_anom[0].x);
                        stot_c.y -= scattering_anom[0].y;
                    }
                    if (apply_polarization) {
                        float P = pol_perp_rate
                                + (1.0f - pol_perp_rate) * (dotv_c * dotv_c);
                        P = fminf(1.0f, fmaxf(0.0f, P));
                        float sc = (P > 0.0f) ? (P * rsqrtf(P)) : 0.0f;
                        stot_c.x *= sc; stot_c.y *= sc;
                    }
                    stot_c.x *= rE_F; stot_c.y *= rE_F;
                }
#endif

                // Only valid pixels process atoms
                if (valid_pixel) {
                    #pragma unroll UNROLL_ATOMS
                    for (int j = 0; j < CHUNK_SIZE; ++j) {
                        int a = base + j;
                        if (a >= nAtoms) break;

                        float ax = s_px[j];
                        float ay = s_py[j];
                        float az = s_pz[j];

#if STAGE_C
                        {   // geometry and phase only; amplitude terms are in
                            // stot_c.  R0p/r_det = 1/(1+poly) ~ 1 - poly, so
                            // the spherical decay is one add.
                            float sproj = fmaf(upz, az, fmaf(upy, ay, upx*ax));
                            float a2    = fmaf(az, az, fmaf(ay, ay, ax*ax));
                            float tval  = fmaf(a2, invR0p * invR0p,
                                               -2.0f * sproj * invR0p);
                            float poly  = sqrt1pm1_series(tval);
                            float s_rel, c_rel;
                            sincos_k_times_reduced(k_global,
                                                   fmaf(R0p, poly, ax),
                                                   s_rel, c_rel);
                            float w  = apply_spherical_decay ? (1.0f - poly) : 1.0f;
                            float wr = w * c_rel, wi = w * s_rel;
                            float2 Eg = s_amp_M[j * M_BEAMS_COMPILE + 0];
                            csum.x += Eg.x * wr - Eg.y * wi;
                            csum.y += Eg.x * wi + Eg.y * wr;
                            continue;
                        }
#endif

                        float dx = qx - ax;
                        float dy = qy - ay;
                        float dz = qz - az;
                        float d2 = dx*dx + dy*dy + dz*dz;
                        if (!(d2 > 0.0f)) continue;

                        #if GLOBAL_USE_SERIES
                        // r_det never enters the phase in the series path, so
                        // the 2-ulp rsqrtf is ample for dotv, Q and the 1/r decay.
                        float inv_r = rsqrtf(d2);
                        float r_det = d2 * inv_r;
                        #else
                        // Exact path: delta_r = r_det - R0, so r_det IS the
                        // phase and must stay correctly rounded.
                        float r_det = sqrtf(d2);
                        float inv_r = 1.0f / r_det;
                        #endif

                        // +x incidence approximation
                        float dotv = dx * inv_r;

                        float tmp = 2.0f * (1.0f - dotv);
                        if (tmp < 0.0f) tmp = 0.0f;
                        float Q_val = (tmp > 0.0f)
                                    ? (k_global * (tmp * rsqrtf(tmp)))
                                    : 0.0f;

                        const float* param_ptr = &s_params[j*11];
                        float f0v = get_f0_from_params(Q_val, param_ptr);

                        // Scattering factor, optionally minus the forward amplitude
                        float2 s_tot;
                        s_tot.x = f0v + s_anm[j].x;
                        s_tot.y = s_anm[j].y;

                        if (remove_forward && (Q_val < Q_cut)) {
                            s_tot.x -= (s_f0z[j] + s_anm[j].x);
                            s_tot.y -= (s_anm[j].y);
                        }

                        // M-channel coherent sum with lattice-phase factors
                        // (Eq. 13 of the dynamical-method plan).  M=1 with
                        // g=0 is bit-identical to the single-amplitude path.
                        float2 amp;
                        if (g_all_zero) {
                            // every g is zero: each channel contributes Eg exactly
                            amp = s_amp_M[j * M_BEAMS_COMPILE + 0];
                            #pragma unroll
                            for (int m = 1; m < M_BEAMS_COMPILE; ++m) {
                                float2 Eg = s_amp_M[j * M_BEAMS_COMPILE + m];
                                amp.x += Eg.x;
                                amp.y += Eg.y;
                            }
                        } else {
                        amp = make_float2(0.0f, 0.0f);
                        #pragma unroll
                        for (int m = 0; m < M_BEAMS_COMPILE; ++m) {
                            float gx = g_vecs_in[m*3 + 0];
                            float gy = g_vecs_in[m*3 + 1];
                            float gz = g_vecs_in[m*3 + 2];
                            float sg, cg;
                            sincos_2pi_dot_g_r(gx, gy, gz, ax, ay, az, sg, cg);
                            float2 Eg = s_amp_M[j * M_BEAMS_COMPILE + m];
                            amp.x += Eg.x * cg - Eg.y * sg;
                            amp.y += Eg.x * sg + Eg.y * cg;
                        }
                        }
                        float real_part = amp.x * s_tot.x - amp.y * s_tot.y;
                        float imag_part = amp.x * s_tot.y + amp.y * s_tot.x;

                        float delta_r;
                        #if GLOBAL_USE_SERIES
                            if (R0p > 0.0f) {
                                float sproj = fmaf(upz, az, fmaf(upy, ay, upx*ax));
                                float a2    = fmaf(az, az, fmaf(ay, ay, ax*ax));
                                float tval  = -2.0f * sproj * invR0p + a2 * (invR0p * invR0p);
                                delta_r = R0p * sqrt1pm1_series(tval);
                            } else {
                                delta_r = r_det;
                            }
                        #else
                            delta_r = r_det - R0;
                        #endif

                        float s_rel, c_rel;
                        sincos_k_times_reduced(k_global, ax + delta_r, s_rel, c_rel);

                        float2 val;
                        val.x = real_part * c_rel - imag_part * s_rel;
                        val.y = real_part * s_rel + imag_part * c_rel;

                        if (apply_polarization) {
                            float P = pol_perp_rate + (1.0f - pol_perp_rate) * (dotv * dotv);
                            P = fminf(1.0f, fmaxf(0.0f, P));
                            float sc = (P > 0.0f) ? (P * rsqrtf(P)) : 0.0f;
                            val.x *= sc; val.y *= sc;
                        }

                        // Optional analyser
                        if (apply_analyser) {
                            // Unit direction atom -> pixel.  The acceptance
                            // test is a hard threshold in cosang, so a 2-ulp
                            // rsqrtf error flips atoms across the boundary;
                            // keep the correctly rounded reciprocal.
                            float inv_rd = 1.0f / sqrtf(d2);
                            float rux = dx * inv_rd;
                            float ruy = dy * inv_rd;
                            float ruz = dz * inv_rd;
                            float cosang = rux*cx + ruy*cy + ruz*cz;
                            cosang = fminf(1.0f, fmaxf(-1.0f, cosang));

                            float scaleA = 1.0f;
                            if (analyser_kind == 1) {
                                // top-hat acceptance
                                if (cosang < cos_accept) scaleA = 0.0f;
                            } else if (analyser_kind == 2) {
                                float delta = acosf(cosang);
                                float hw = darwin_halfwidth_rad;
                                if (hw > 0.0f) {
                                    float r = delta / hw;
                                    scaleA = 1.0f / (1.0f + r*r);
                                } else {
                                    scaleA = 1.0f;
                                }
                            }
                            val.x *= scaleA; val.y *= scaleA;
                        }

                        // Optional relative spherical decay
                        float amp_rel = 1.0f;
                        if (apply_spherical_decay) {
                            amp_rel = (R0p > 0.0f) ? (R0p * inv_r) : 1.0f;
                        }

                        csum.x += val.x * rE_F * amp_rel;
                        csum.y += val.y * rE_F * amp_rel;
                    }
                }
                if (valid_pixel) {
#if STAGE_C
                    float rr = csum.x * c0 - csum.y * s0;
                    float ii = csum.x * s0 + csum.y * c0;
                    sum_rel.x += rr * stot_c.x - ii * stot_c.y;
                    sum_rel.y += rr * stot_c.y + ii * stot_c.x;
#else
                    sum_rel.x += csum.x * c0 - csum.y * s0;
                    sum_rel.y += csum.x * s0 + csum.y * c0;
#endif
                }
                __syncthreads();
            }

            if (valid_pixel) {
                float2 sum_rot;
                sum_rot.x = sum_rel.x * cb - sum_rel.y * sb;
                sum_rot.y = sum_rel.x * sb + sum_rel.y * cb;

                detector_field[pidx].x += sum_rot.x;
                detector_field[pidx].y += sum_rot.y;
            }
        } // kernel

        } // extern "C"
        ''';

        kernel_module = cp.RawModule(
            code=_cuda_source,
            backend='nvcc',
            options=(
                '--gpu-architecture=native',
                '-O3', '--ftz=true', '--fmad=true',
                f'-DN_SERIES={N}',
                f'-DGLOBAL_USE_SERIES={global_use_series}',
                f'-DM_BEAMS_COMPILE={M_compile}',
                f'-DCHUNK_SIZE={chunk_size}',
                f'-DUNROLL_ATOMS={unroll}',
                f'-DN_PRO={n_pro}',
                f'-DPROLOGUE_MODE={pro_mode}',
                f'-DSTAGE_C={stage_c}',
            ) + ((f'-maxrregcount={int(maxreg)}',) if maxreg else ())
        )
        kern = kernel_module.get_function('interaction_kernal')
        self._interaction_kernel_cache[key] = kern
        return kern
    
    @staticmethod
    def build_ein_sampler_kernel():
        """
        Build the kernel that bilinearly samples E0, tau and phi on the beam grid.

        For each position the kernel projects to beam-basis (u, v), samples
        tau, phi and E0 bilinearly, and forms
        Ein = E0 * exp(-f * tau) * exp(i * f * phi), with f the depth fraction
        along the beam direction.

        Returns:
            cupy.RawKernel: Kernel handle named "ein_bilinear_kernel".

        Notes:
            - Out-of-bounds samples are set to zero (no edge clamping).
            - Positions and grid spacings are in Angstrom.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for build_ein_sampler_kernel")

        src = r'''
        #include <math.h>

        extern "C" __global__
        void ein_bilinear_kernel(
            const float* __restrict__ pos,   // (N,3) in Angstrom
            const int N,

            const float* __restrict__ tau,   // (NyB*NzB)
            const float* __restrict__ phi,   // (NyB*NzB)
            const float2* __restrict__ E0,   // (NyB*NzB)

            const int NyB,
            const int NzB,

            const float inv_du,
            const float inv_dv,
            const float uc,
            const float vc,

            const float* __restrict__ e1,    // len=3
            const float* __restrict__ e2,    // len=3
            const float* __restrict__ khat,  // len=3

            const float s_min,
            const float s_max,

            float2* __restrict__ out_amp
        )
        {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            float denom = s_max - s_min;
            if (!(denom > 0.0f)) denom = 1.0f;

            for (int i = tid; i < N; i += stride)
            {
                float x = pos[3*i + 0];
                float y = pos[3*i + 1];
                float z = pos[3*i + 2];

                // Project to beam basis
                float au = e1[0]*x + e1[1]*y + e1[2]*z;
                float av = e2[0]*x + e2[1]*y + e2[2]*z;

                // Continuous -> grid indices
                float iu = au * inv_du + uc;
                float iv = av * inv_dv + vc;

                // Hard in-bounds check (no clamping when out)
                if (iu < 0.0f || iu > (float)(NyB - 1) ||
                    iv < 0.0f || iv > (float)(NzB - 1))
                {
                    out_amp[i] = make_float2(0.0f, 0.0f);
                    continue;
                }

                int i0 = (int)floorf(iu);
                int j0 = (int)floorf(iv);
                int i1 = i0 + 1;
                int j1 = j0 + 1;

                // Clamp neighbours to the grid
                i0 = max(0, min(NyB - 1, i0));
                i1 = max(0, min(NyB - 1, i1));
                j0 = max(0, min(NzB - 1, j0));
                j1 = max(0, min(NzB - 1, j1));

                // Bilinear weights
                float fu = fminf(1.0f, fmaxf(0.0f, iu - (float)i0));
                float fv = fminf(1.0f, fmaxf(0.0f, iv - (float)j0));
                float w00 = (1.0f - fu)*(1.0f - fv);
                float w01 = (1.0f - fu)*fv;
                float w10 = fu*(1.0f - fv);
                float w11 = fu*fv;

                int r00 = i0 * NzB + j0;
                int r01 = i0 * NzB + j1;
                int r10 = i1 * NzB + j0;
                int r11 = i1 * NzB + j1;

                // Sample tau, phi
                float tau_s = w00 * tau[r00] + w01 * tau[r01] + w10 * tau[r10] + w11 * tau[r11];
                float phi_s = w00 * phi[r00] + w01 * phi[r01] + w10 * phi[r10] + w11 * phi[r11];

                // Sample E0 (complex)
                float2 e00 = E0[r00];
                float2 e01 = E0[r01];
                float2 e10 = E0[r10];
                float2 e11 = E0[r11];
                float2 e0s;
                e0s.x = w00*e00.x + w01*e01.x + w10*e10.x + w11*e11.x;
                e0s.y = w00*e00.y + w01*e01.y + w10*e10.y + w11*e11.y;

                // Depth fraction f in [0,1]
                float s_val = khat[0]*x + khat[1]*y + khat[2]*z;
                float f = (s_val - s_min) / denom;
                f = fminf(1.0f, fmaxf(0.0f, f));

                // Ein = E0 * exp(-f*tau) * exp(i*f*phi)
                float amp = __expf(-f * tau_s);
                float phase = f * phi_s;
                float sn, cs;
                __sincosf(phase, &sn, &cs);

                float2 scale; scale.x = amp * cs; scale.y = amp * sn;

                float2 outv;
                outv.x = e0s.x * scale.x - e0s.y * scale.y;
                outv.y = e0s.x * scale.y + e0s.y * scale.x;

                out_amp[i] = outv;
            }
        }
        ''';

        mod = cp.RawModule(
            code=src,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return mod.get_function('ein_bilinear_kernel')
    # -------------------------------------

    def _auto_select_beams(self, crystal, stage, M_max=2):
        """
        Pick up to M_max reflections closest to the Ewald sphere.

        Enumerates (h, k, l) of the conventional cell, rotates them by the
        stage, and ranks by |s_g|.  The forward beam (g = 0) is always index 0.

        Args:
            crystal: Provides lattice_matrix_conventional (3x3, rows = a, b, c
                in Angstrom) and lattice_volume_conventional.
            stage: Provides rotation (3x3).
            M_max: Maximum number of beams including the forward beam.

        Returns:
            list[dict]: One descriptor per beam with keys
                "hkl": (h, k, l)
                "G": (3,) reciprocal vector, crystallographic convention
                     (1/Angstrom, no 2pi); phase is exp(-2*pi*i * G . r)
                "k_vec": (3,) k0 + 2*pi*G, physics convention (1/Angstrom)
                "excitation_error": |s_g| in 1/Angstrom
                "cos_theta": obliquity factor for the propagator
        """
        two_pi = 2.0 * np.pi
        lam_A = float(self._wavelength) * 1e10  # meters -> Angstrom
        k_mag = two_pi / lam_A  # 1/Angstrom

        # Incident wavevector in lab frame (along +x)
        k_hat = np.asarray(self._direction, dtype=np.float64)
        k_hat = k_hat / np.linalg.norm(k_hat)
        k0 = k_mag * k_hat  # (3,)

        # lattice_matrix_conventional has a, b, c as ROWS; transpose so they
        # are columns, then left-multiply by the stage rotation.
        R_stage = np.asarray(stage.rotation, dtype=np.float64)
        lat_conv = np.asarray(crystal.lattice_matrix_conventional, dtype=np.float64)
        # Lattice vectors in lab frame, columns = (a, b, c)_lab.
        lat_lab = R_stage @ lat_conv.T

        V_cell = float(crystal.lattice_volume_conventional)
        # Reciprocal vectors (1/A, no 2pi): b_i* = cross(a_j, a_k) / V, built
        # from the COLUMNS of lat_lab.
        recip = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            recip[i] = np.cross(
                lat_lab[:, (i + 1) % 3], lat_lab[:, (i + 2) % 3]
            ) / V_cell

        # Candidate (h,k,l) range, capped at +-6
        h_max = max(1, int(np.ceil(2.0 * k_mag / max(two_pi * np.linalg.norm(recip[0]), 1e-12))))
        k_max = max(1, int(np.ceil(2.0 * k_mag / max(two_pi * np.linalg.norm(recip[1]), 1e-12))))
        l_max = max(1, int(np.ceil(2.0 * k_mag / max(two_pi * np.linalg.norm(recip[2]), 1e-12))))
        h_max = min(h_max, 6)
        k_max = min(k_max, 6)
        l_max = min(l_max, 6)

        candidates = []
        for h in range(-h_max, h_max + 1):
            for k in range(-k_max, k_max + 1):
                for l in range(-l_max, l_max + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    G_cryst = h * recip[0] + k * recip[1] + l * recip[2]
                    G_phys = two_pi * G_cryst
                    if np.linalg.norm(G_phys) > 2.0 * k_mag:
                        continue
                    k_g = k0 + G_phys
                    s_g = (np.dot(k_g, k_g) - k_mag ** 2) / (2.0 * k_mag)
                    candidates.append({
                        "hkl": (h, k, l),
                        "G": G_cryst.astype(np.float32),
                        "k_vec": k_g.astype(np.float32),
                        "excitation_error": float(abs(s_g)),
                    })

        candidates.sort(key=lambda c: c["excitation_error"])

        beams = [{
            "hkl": (0, 0, 0),
            "G": np.zeros(3, dtype=np.float32),
            "k_vec": k0.astype(np.float32),
            "excitation_error": 0.0,
            "cos_theta": 1.0,
        }]
        for c in candidates:
            if len(beams) >= M_max:
                break
            beams.append(c)
        # Obliquity factor cos(theta_g) for the angular-spectrum propagator (Eq. 11).
        for bd in beams:
            kv = np.asarray(bd["k_vec"], dtype=np.float64)
            kn = float(np.linalg.norm(kv))
            if kn > 0.0:
                bd["cos_theta"] = float(np.clip(np.dot(kv / kn, k_hat), -1.0, 1.0))
            else:
                bd["cos_theta"] = 1.0
        return beams

    def _auto_detect_beams(self, sample, stage, M_max=2,
                           n_subsample=10000, fft_N=128):
        """
        Find Bragg reflections from atom positions alone (no crystal info).

        Bins the atoms into a density grid, takes a 3D FFT, and keeps the
        peaks closest to the Ewald sphere.  Returns only the forward beam
        when no clear peaks exist (e.g. amorphous material).

        Args:
            sample: Chunked sample object.
            stage: Provides rotation (3x3) and translation (3,).
            M_max: Maximum number of beams including the forward beam.
            n_subsample: Atoms used for the FFT (random subsample above this).
            fft_N: Density-grid bins per axis.

        Returns:
            list[dict]: Same format as _auto_select_beams.
        """
        two_pi = 2.0 * np.pi
        lam_A = float(self._wavelength) * 1e10
        k_mag = two_pi / lam_A

        k_hat = np.asarray(self._direction, dtype=np.float64)
        k_hat = k_hat / np.linalg.norm(k_hat)
        k0 = k_mag * k_hat

        forward_beam = {
            "hkl": (0, 0, 0),
            "G": np.zeros(3, dtype=np.float32),
            "k_vec": k0.astype(np.float32),
            "excitation_error": 0.0,
            "cos_theta": 1.0,
        }

        # Atom positions in the lab frame
        positions = []
        R = stage.rotation.astype(np.float64)
        T = stage.translation.astype(np.float64)
        for cid in range(1, int(sample.chunk_total or 0) + 1):
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float64)
            if pos.size == 0:
                continue
            positions.append(pos @ R.T + T)

        if not positions:
            return [forward_beam]

        all_pos = np.concatenate(positions, axis=0)

        if len(all_pos) > n_subsample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(all_pos), n_subsample, replace=False)
            pos_sub = all_pos[idx]
        else:
            pos_sub = all_pos

        # Bin atoms into a 3D density grid
        pos_min = pos_sub.min(axis=0)
        pos_max = pos_sub.max(axis=0)
        extent = pos_max - pos_min
        extent = np.maximum(extent, 1.0)

        pad = 0.05 * extent
        grid_min = pos_min - pad
        grid_extent = extent + 2.0 * pad
        N = int(fft_N)

        frac = (pos_sub - grid_min) / grid_extent  # 0..1
        ijk = np.clip((frac * N).astype(np.int64), 0, N - 1)
        density = np.zeros((N, N, N), dtype=np.float64)
        np.add.at(density, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)

        F = np.fft.fftn(density)
        F_mag = np.abs(F)
        F_mag[0, 0, 0] = 0.0  # remove DC

        # Reciprocal-space coordinates (1/Angstrom, crystallographic convention)
        dx = grid_extent / N  # real-space pixel size per axis
        freq = [np.fft.fftfreq(N, d=dx[i]) for i in range(3)]

        threshold = 0.3 * F_mag.max()
        if threshold < 1e-12:
            return [forward_beam]

        # Keep peaks inside the Ewald-accessible range |G_phys| < 2k
        GX, GY, GZ = np.meshgrid(freq[0], freq[1], freq[2], indexing='ij')
        G_phys_mag_sq = (two_pi ** 2) * (GX * GX + GY * GY + GZ * GZ)
        four_k_sq = (2.0 * k_mag) ** 2

        peak_mask = (F_mag > threshold) & (G_phys_mag_sq > 0.01) & (G_phys_mag_sq < four_k_sq)
        peak_idx = np.argwhere(peak_mask)

        if len(peak_idx) == 0:
            return [forward_beam]

        candidates = []
        for ix, iy, iz in peak_idx:
            g_cryst = np.array([freq[0][ix], freq[1][iy], freq[2][iz]],
                               dtype=np.float64)
            g_phys = two_pi * g_cryst
            k_g = k0 + g_phys
            s_g = (np.dot(k_g, k_g) - k_mag ** 2) / (2.0 * k_mag)
            candidates.append({
                "hkl": (int(ix), int(iy), int(iz)),
                "G": g_cryst.astype(np.float32),
                "k_vec": k_g.astype(np.float32),
                "excitation_error": float(abs(s_g)),
            })

        candidates.sort(key=lambda c: c["excitation_error"])

        beams = [forward_beam]
        for c in candidates:
            if len(beams) >= M_max:
                break
            beams.append(c)
        for bd in beams:
            kv = np.asarray(bd["k_vec"], dtype=np.float64)
            kn = float(np.linalg.norm(kv))
            if kn > 0.0:
                bd["cos_theta"] = float(np.clip(np.dot(kv / kn, k_hat), -1.0, 1.0))
            else:
                bd["cos_theta"] = 1.0
        return beams

    @staticmethod
    def _beams_from_g_vectors(g_vectors, k0, k_mag):
        """
        Beam descriptors from user-supplied G vectors (forward beam first).

        Args:
            g_vectors: Iterable of (3,) reciprocal vectors, crystallographic
                convention (1/Angstrom, no 2pi).
            k0: (3,) incident wavevector (2pi/lambda, 1/Angstrom).
            k_mag: |k0|.
        """
        two_pi = 2.0 * np.pi
        # Unit forward direction for cos(theta).
        k0_arr = np.asarray(k0, dtype=np.float64)
        k0_norm = float(np.linalg.norm(k0_arr))
        k_hat = k0_arr / k0_norm if k0_norm > 0.0 else np.array([1.0, 0.0, 0.0])
        beams = [{
            "hkl": (0, 0, 0),
            "G": np.zeros(3, dtype=np.float32),
            "k_vec": k0_arr.astype(np.float32),
            "excitation_error": 0.0,
            "cos_theta": 1.0,
        }]
        for i, g in enumerate(g_vectors):
            g = np.asarray(g, dtype=np.float64)
            g_phys = two_pi * g
            k_g = k0_arr + g_phys
            s_g = (np.dot(k_g, k_g) - k_mag ** 2) / (2.0 * k_mag)
            kgn = float(np.linalg.norm(k_g))
            ct = float(np.clip(np.dot(k_g / kgn, k_hat), -1.0, 1.0)) if kgn > 0.0 else 1.0
            beams.append({
                "hkl": (i + 1, 0, 0),
                "G": g.astype(np.float32),
                "k_vec": k_g.astype(np.float32),
                "excitation_error": float(abs(s_g)),
                "cos_theta": ct,
            })
        return beams

    def _build_structure_factor_maps_gpu(self, sample, stage, slice_edges_A,
                                         beam_info, kernel_radius=0,
                                         born_convention=False):
        """
        Per-slice complex structure-factor maps for every delta-g the
        coupling matrix needs.

        For M beams, element A_{ab} needs chi_{g_a - g_b}.  Each atom is
        deposited onto the beam grid with TSC weights, phase
        exp(-2*pi*i * delta_g . r) and amplitude f(|delta_g|).

        Normalization:
        - born_convention=False: prefactor r_e*lambda^2 / (2*pi*du*dv), a
          column integral.  The transmission step must then use
          exp(i*k*chi) with no dz factor.
        - born_convention=True: per-slice prefactor
          r_e*lambda^2 / (pi*du*dv*dz_k), a true voxel-density chi (Eq. 7).
          The transmission step must then use exp(i*k*chi*dz).

        Args:
            sample: Chunked sample object.
            stage: Provides rotation and translation.
            slice_edges_A: (n_slices+1,) depth edges in Angstrom.
            beam_info: Beam descriptors from _auto_select_beams.
            kernel_radius: Gaussian blur radius in pixels (0 = off).
            born_convention: Use the per-slice voxel-density normalization.

        Returns:
            dict: (a, b) -> list of n_slices complex64 arrays, each (NyB, NzB).
        """
        M = len(beam_info)
        nS = int(len(slice_edges_A) - 1)

        r_e_A = 2.81794092e-5
        lam_A = float(self._wavelength) * 1e10
        two_pi = 2.0 * np.pi

        du, dv = float(self._beam_du), float(self._beam_dv)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        A_pix_A2 = du * dv
        # Legacy column-integral prefactor (per-slice constant).
        C = (r_e_A * lam_A * lam_A) / (two_pi * A_pix_A2)
        # Per-slice Born/Authier voxel-density prefactor.
        edges_arr = np.asarray(slice_edges_A, dtype=np.float64)
        if nS > 0:
            dz_per_slice = np.diff(edges_arr).astype(np.float64)
            dz_per_slice = np.where(dz_per_slice > 0.0, dz_per_slice, 1.0)
        else:
            dz_per_slice = np.ones(0, dtype=np.float64)
        C_born = (r_e_A * lam_A * lam_A) / (
            np.pi * A_pix_A2 * dz_per_slice
        )  # (nS,)

        # delta_g per (a, b) pair
        unique_dg = {}
        for a in range(M):
            for b in range(M):
                dg = beam_info[a]["G"] - beam_info[b]["G"]
                unique_dg[(a, b)] = dg.astype(np.float64)

        f1f2_dict = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict = self._build_f0_zero_dict(f0_params_dict)

        e1 = self._beam_e1.astype(np.float32)
        e2 = self._beam_e2.astype(np.float32)
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)

        dg_magnitudes = {}
        for key, dg in unique_dg.items():
            dg_magnitudes[key] = float(np.linalg.norm(dg))

        def _tsc_w(d):
            w = np.zeros_like(d, dtype=np.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0] * d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        accum = {}
        for key in unique_dg:
            accum[key] = [np.zeros((NyB, NzB), dtype=np.complex64) for _ in range(nS)]

        for cid in range(1, int(sample.chunk_total or 0) + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
            if pos.size == 0:
                continue

            pos = pos @ stage.rotation.astype(np.float32).T
            pos += stage.translation.astype(np.float32)
            nA = pos.shape[0]

            # Per-element scattering factors
            f1_arr = np.zeros(nA, np.float32)
            f2_arr = np.zeros(nA, np.float32)
            f0z_arr = np.zeros(nA, np.float32)
            for el in np.unique(spc):
                el_s = str(el)
                m = (spc == el_s)
                f0z_arr[m] = float(f0_zero_dict.get(el_s, 0.0))
                tbl = f1f2_dict.get(el_s)
                if tbl is not None:
                    cplx = self.get_f1f2_from_params(self._energy, tbl)
                    f1_arr[m] = float(cplx.real)
                    f2_arr[m] = float(cplx.imag)

            # Beam-basis coords and slice index
            au = pos[:, 0] * e1[0] + pos[:, 1] * e1[1] + pos[:, 2] * e1[2]
            av = pos[:, 0] * e2[0] + pos[:, 1] * e2[1] + pos[:, 2] * e2[2]
            iu = au / du + float(self._beam_uc)
            iv = av / dv + float(self._beam_vc)
            s_vals = pos[:, 0] * k_hat[0] + pos[:, 1] * k_hat[1] + pos[:, 2] * k_hat[2]
            k_idx = np.clip(np.searchsorted(slice_edges_A, s_vals, side="right") - 1, 0, nS - 1)

            inb = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
            if not np.any(inb):
                continue

            iu_s, iv_s = iu[inb], iv[inb]
            ki_s = k_idx[inb]
            pos_s = pos[inb]
            f1_s, f2_s, f0z_s = f1_arr[inb], f2_arr[inb], f0z_arr[inb]
            spc_s = spc[inb]

            # TSC grid centers and weights
            ic = np.floor(iu_s + 0.5).astype(np.int64)
            jc = np.floor(iv_s + 0.5).astype(np.int64)
            wu_list, wv_list = [], []
            for dx in [-1, 0, 1]:
                wu_list.append(_tsc_w(np.abs(iu_s - (ic + dx))))
                wv_list.append(_tsc_w(np.abs(iv_s - (jc + dx))))

            for key, dg in unique_dg.items():
                dg_mag = dg_magnitudes[key]

                if dg_mag < 1e-10:
                    f_real = (f0z_s + f1_s).astype(np.float32)
                    f_imag = -f2_s.astype(np.float32)
                    phase = np.zeros(len(pos_s), dtype=np.float64)
                else:
                    s_val = dg_mag / (4.0 * np.pi)
                    ss = s_val * s_val
                    f_real = np.zeros(len(pos_s), dtype=np.float32)
                    f_imag = -f2_s.copy()
                    for el in np.unique(spc_s):
                        el_s = str(el)
                        m = (spc_s == el_s)
                        params = f0_params_dict.get(el_s)
                        if params is not None:
                            f0_val = float(params[5])
                            for ii in range(5):
                                f0_val += float(params[ii]) * np.exp(-float(params[6 + ii]) * ss)
                            f_real[m] = f0_val + f1_s[m]
                        else:
                            f_real[m] = f1_s[m]
                    phase = -two_pi * (dg[0] * pos_s[:, 0].astype(np.float64)
                                       + dg[1] * pos_s[:, 1].astype(np.float64)
                                       + dg[2] * pos_s[:, 2].astype(np.float64))

                cos_ph = np.cos(phase).astype(np.float32)
                sin_ph = np.sin(phase).astype(np.float32)
                w_real = f_real * cos_ph - f_imag * sin_ph
                w_imag = f_real * sin_ph + f_imag * cos_ph

                for di, dx in enumerate([-1, 0, 1]):
                    ii = ic + dx
                    for dj, dy in enumerate([-1, 0, 1]):
                        jj = jc + dy
                        fac = wu_list[di] * wv_list[dj]
                        mask = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                        if not np.any(mask):
                            continue
                        pidx = (ii[mask] * NzB + jj[mask]).astype(np.int64)
                        wsel = fac[mask]
                        kis = ki_s[mask]
                        vals = (w_real[mask] * wsel) + 1j * (w_imag[mask] * wsel)
                        for s in np.unique(kis):
                            ms = (kis == s)
                            np.add.at(accum[key][s].ravel(), pidx[ms], vals[ms].astype(np.complex64))

        # Prefactor (per-slice if born_convention) and optional blur.
        result = {}
        for key in unique_dg:
            maps = accum[key]
            for i in range(nS):
                if born_convention:
                    maps[i] = (np.complex64(C_born[i]) * maps[i]).astype(np.complex64)
                else:
                    maps[i] = (C * maps[i]).astype(np.complex64)
            result[key] = maps

        if int(kernel_radius) > 0:
            rad = int(kernel_radius)
            sig = max(1e-6, rad / 2.0)
            y, x = np.ogrid[-rad:rad + 1, -rad:rad + 1]
            kern = np.exp(-(x * x + y * y) / (2.0 * sig * sig)).astype(np.float32)
            kern /= max(kern.sum(), 1e-20)
            Fk = np.fft.fft2(kern, s=(NyB, NzB))
            for key in result:
                for i in range(nS):
                    m = result[key][i]
                    m = np.fft.ifft2(np.fft.fft2(m) * Fk).astype(np.complex64)
                    result[key][i] = m

        return result

    @staticmethod
    def build_beam_coupling_kernel():
        """
        CUDA kernel applying the exact 2x2 matrix exponential of
        M = i*k*dz * [[chi0, chi_mh], [chi_h, chi0]] to (E0, E1) per pixel.

        Returns:
            cupy.RawKernel: "beam_couple_2x2_kernel".
        """
        if cp is None:
            raise RuntimeError("CuPy is required for beam coupling kernel.")

        src = r'''
        #include <math.h>

        __device__ __forceinline__ float2 cmul(float2 a, float2 b) {
            return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
        }
        __device__ __forceinline__ float2 cadd(float2 a, float2 b) {
            return make_float2(a.x + b.x, a.y + b.y);
        }
        __device__ __forceinline__ float2 cexp(float2 z) {
            float er = expf(z.x);
            float s, c;
            __sincosf(z.y, &s, &c);
            return make_float2(er * c, er * s);
        }
        __device__ __forceinline__ float2 csqrt(float2 z) {
            float r = sqrtf(z.x*z.x + z.y*z.y);
            float mag = sqrtf(r);
            if (mag < 1e-30f) return make_float2(0.0f, 0.0f);
            float angle = atan2f(z.y, z.x) * 0.5f;
            float s, c;
            __sincosf(angle, &s, &c);
            return make_float2(mag * c, mag * s);
        }
        __device__ __forceinline__ float2 ccosh(float2 z) {
            float2 ez = cexp(z);
            float2 emz = cexp(make_float2(-z.x, -z.y));
            return make_float2(0.5f*(ez.x + emz.x), 0.5f*(ez.y + emz.y));
        }
        __device__ __forceinline__ float2 csinh(float2 z) {
            float2 ez = cexp(z);
            float2 emz = cexp(make_float2(-z.x, -z.y));
            return make_float2(0.5f*(ez.x - emz.x), 0.5f*(ez.y - emz.y));
        }
        __device__ __forceinline__ float2 cdiv(float2 a, float2 b) {
            float denom = b.x*b.x + b.y*b.y;
            if (denom < 1e-30f) return make_float2(0.0f, 0.0f);
            float inv = 1.0f / denom;
            return make_float2((a.x*b.x + a.y*b.y) * inv,
                               (a.y*b.x - a.x*b.y) * inv);
        }

        extern "C" __global__
        void beam_couple_2x2_kernel(
            float2* __restrict__ E0,
            float2* __restrict__ E1,
            const float2* __restrict__ chi0,
            const float2* __restrict__ chi_h,
            const float2* __restrict__ chi_mh,
            const float  k_dz,
            const int    N
        )
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N) return;

            float2 e0 = E0[idx];
            float2 e1 = E1[idx];
            float2 c0  = chi0[idx];
            float2 ch  = chi_h[idx];
            float2 cmh = chi_mh[idx];

            // M = i * k_dz * [[c0, cmh], [ch, c0]]
            float2 m00 = make_float2(-c0.y  * k_dz, c0.x  * k_dz);
            float2 m01 = make_float2(-cmh.y * k_dz, cmh.x * k_dz);
            float2 m10 = make_float2(-ch.y  * k_dz, ch.x  * k_dz);
            float2 m11 = m00;

            float2 half_tr = m00;
            float2 det = make_float2(
                m00.x*m11.x - m00.y*m11.y - (m01.x*m10.x - m01.y*m10.y),
                m00.x*m11.y + m00.y*m11.x - (m01.x*m10.y + m01.y*m10.x));

            float2 half_tr_sq = cmul(half_tr, half_tr);
            float2 w2 = make_float2(half_tr_sq.x - det.x, half_tr_sq.y - det.y);
            float2 w = csqrt(w2);
            float2 exp_ht = cexp(half_tr);

            float w_mag = sqrtf(w.x*w.x + w.y*w.y);
            float2 cosh_w, sinhw_over_w;
            if (w_mag < 1e-6f) {
                cosh_w = make_float2(1.0f, 0.0f);
                sinhw_over_w = make_float2(1.0f + (w2.x / 6.0f), w2.y / 6.0f);
            } else {
                cosh_w = ccosh(w);
                float2 sinh_w = csinh(w);
                sinhw_over_w = cdiv(sinh_w, w);
            }

            float2 R00 = cmul(exp_ht, cosh_w);
            float2 R01 = cmul(exp_ht, cmul(sinhw_over_w, m01));
            float2 R10 = cmul(exp_ht, cmul(sinhw_over_w, m10));
            float2 R11 = R00;

            E0[idx] = cadd(cmul(R00, e0), cmul(R01, e1));
            E1[idx] = cadd(cmul(R10, e0), cmul(R11, e1));
        }
        ''';

        mod = cp.RawModule(
            code=src,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return mod.get_function('beam_couple_2x2_kernel')

    def _beam_transmission_step_gpu(self, E_beams, chi_maps_slice, k_A):
        """
        Per-pixel matrix-exponential transmission step (Eq. 10) on GPU.

        M=1: E0 *= exp(i*k_A*chi0).  M=2: closed-form 2x2 exponential of
        i*k_A*[[chi0, chi_-h], [chi_h, chi0]].  M>2: not implemented (Pade
        path deferred).

        Units: with voxel-density chi maps (born_convention=True) k_A must be
        k*dz; with legacy column-integral maps k_A is just k, since dz is
        already inside chi.

        Args:
            E_beams: list of M CuPy (NyB, NzB) complex64 arrays.
            chi_maps_slice: dict (a, b) -> complex64 GPU array (NyB, NzB).
            k_A: transmission phase prefactor.

        Returns:
            E_beams, updated in place.
        """
        M = len(E_beams)

        if M == 1:
            chi0 = chi_maps_slice[(0, 0)]
            if not isinstance(chi0, cp.ndarray):
                chi0 = cp.asarray(chi0, dtype=cp.complex64)
            arg = (1j * k_A * chi0).astype(cp.complex64)
            E_beams[0] = (E_beams[0] * cp.exp(arg)).astype(cp.complex64)
            return E_beams

        if M == 2:
            if not hasattr(self, '_beam_couple_kernel_cache'):
                self._beam_couple_kernel_cache = self.build_beam_coupling_kernel()
            kernel = self._beam_couple_kernel_cache

            NyB, NzB = E_beams[0].shape
            N = NyB * NzB

            chi0 = chi_maps_slice[(0, 0)]
            chi_h = chi_maps_slice[(1, 0)]
            chi_mh = chi_maps_slice[(0, 1)]

            if not isinstance(chi0, cp.ndarray):
                chi0 = cp.asarray(chi0, dtype=cp.complex64)
            if not isinstance(chi_h, cp.ndarray):
                chi_h = cp.asarray(chi_h, dtype=cp.complex64)
            if not isinstance(chi_mh, cp.ndarray):
                chi_mh = cp.asarray(chi_mh, dtype=cp.complex64)

            E0_flat = E_beams[0].ravel()
            E1_flat = E_beams[1].ravel()

            block = 256
            grid = (N + block - 1) // block
            kernel(
                (grid,), (block,),
                (E0_flat, E1_flat,
                 chi0.ravel(), chi_h.ravel(), chi_mh.ravel(),
                 np.float32(k_A), np.int32(N))
            )
            E_beams[0] = E0_flat.reshape(NyB, NzB)
            E_beams[1] = E1_flat.reshape(NyB, NzB)
            return E_beams

        raise NotImplementedError(
            f"Beam transmission for M={M} > 2 not yet implemented.")

    def _beam_propagation_step_gpu(self, E_beams, dz_A, beam_info):
        """
        Angular-spectrum propagation of each beam by dz_A Angstrom, with the
        beam's own carrier wave subtracted exactly (Eq. 11).

        Subtracting beam_info[m]['k_vec'] keeps the Bragg-beam envelope in
        step with the exp(-i 2 pi G_h . r) carrier of the chi_h maps, which
        is what lets E_g build coherently over many slices.

        Args:
            E_beams: list of M CuPy (NyB, NzB) complex64 arrays.
            dz_A: slice thickness in Angstrom.
            beam_info: M descriptors with 'k_vec' (1/Angstrom, with 2 pi);
                'cos_theta' is the fallback if k_vec is missing.

        Returns:
            list of M propagated CuPy arrays.
        """
        if not hasattr(self, '_prop_kernel_cache') or self._prop_kernel_cache is None:
            self._prop_kernel_cache = self.build_propagation_multiplier_kernel()
        kernel = self._prop_kernel_cache

        # Axis convention (critical).  Envelopes are shaped (_beam_Ny, _beam_Nz)
        # = (axis0 = e1/u, axis1 = e2/v).  _angular_spectrum_propagate_gpu
        # builds ky over axis1 from `dy` and kz over axis0 from `dz`, and the
        # kernel pairs k_g_perp_y with ky and k_g_perp_z with kz.  So e2/v
        # quantities go in the `dy`/`k_g_perp_y` slots and e1/u quantities in
        # `dz`/`k_g_perp_z`.  Swapping them subtracts the carrier on the wrong
        # axis, E_g de-phases every slice, and no Pendellosung appears.
        dy_m = float(self._beam_dv) * 1e-10      # e2/v pitch -> ky / axis1
        dz_pix_m = float(self._beam_du) * 1e-10  # e1/u pitch -> kz / axis0
        dz_m = float(dz_A) * 1e-10

        # Beam-grid axes in the lab frame.
        k_hat = np.asarray(self._direction, dtype=np.float64)
        k_hat = k_hat / np.linalg.norm(k_hat)
        e1 = np.asarray(self._beam_e1, dtype=np.float64)
        e2 = np.asarray(self._beam_e2, dtype=np.float64)
        for m in range(len(E_beams)):
            kvec_iA = np.asarray(beam_info[m].get('k_vec', None), dtype=np.float64)
            if kvec_iA is None or kvec_iA.size != 3:
                # Legacy cos_theta fallback
                cos_theta_m = float(beam_info[m].get("cos_theta", 1.0))
                E_beams[m] = self._angular_spectrum_propagate_gpu(
                    E_beams[m], dy_m, dz_pix_m, dz_m, kernel,
                    step_max=0.02, pad_factor=1.0, padding_mode="edge",
                    cos_theta=cos_theta_m,
                )
                continue
            # Project k_vec (1/A with 2 pi) onto the beam-grid axes:
            # e2 -> k_g_perp_y, e1 -> k_g_perp_z (see the axis note above).
            k_g_axis_iA = float(np.dot(kvec_iA, k_hat))
            k_g_perp_y_iA = float(np.dot(kvec_iA, e2))  # ky <-> e2/axis1
            k_g_perp_z_iA = float(np.dot(kvec_iA, e1))  # kz <-> e1/axis0
            # 1/A (with 2 pi) -> rad/m, matching dy_m, dz_pix_m, dz_m.
            k_g_axis_m = k_g_axis_iA * 1e10
            k_g_perp_y_m = k_g_perp_y_iA * 1e10
            k_g_perp_z_m = k_g_perp_z_iA * 1e10
            E_beams[m] = self._angular_spectrum_propagate_gpu(
                E_beams[m], dy_m, dz_pix_m, dz_m, kernel,
                step_max=0.02, pad_factor=1.0, padding_mode="edge",
                k_g_axis=k_g_axis_m,
                k_g_perp_y=k_g_perp_y_m,
                k_g_perp_z=k_g_perp_z_m,
            )
        return E_beams

    def _beam_coupling_step_gpu(self, E_beams, chi_maps_slice, k_A,
                                dz_A=None, beam_info=None):
        """
        Compatibility wrapper: transmission step, plus the per-beam ASP
        propagation step (Lie-Trotter split) when both dz_A and beam_info
        are given.

        Args:
            E_beams: list of M CuPy (NyB, NzB) complex64 arrays.
            chi_maps_slice: dict (a, b) -> complex64 GPU array (NyB, NzB).
            k_A: transmission phase prefactor.
            dz_A: slice thickness in Angstrom, for propagation.
            beam_info: M beam descriptors, for propagation.

        Returns:
            list of M CuPy arrays.
        """
        E_beams = self._beam_transmission_step_gpu(E_beams, chi_maps_slice, k_A)
        if dz_A is not None and beam_info is not None:
            E_beams = self._beam_propagation_step_gpu(E_beams, dz_A, beam_info)
        return E_beams

    # -------------------------------------------------------------------------
    # FP32-robust atom-table builder for the dynamical multislice.  The
    # integer/fractional voxel-coordinate split uses _two_prod_fp32 and
    # _two_sum_fp32 so atoms at sample-scale (~mm) positions still get
    # sub-voxel weights at FP32 precision.
    # -------------------------------------------------------------------------
    def _build_atom_table_for_multislice(self, sample, stage, edges_A, n_final):
        """
        Per-chunk FP32-robust atom tables for the multislice + LS pipeline.

        Tables are kept per chunk rather than concatenated, so the multislice
        loop and the LS step can work one chunk at a time and peak host
        memory stays bounded by the largest chunk (billion-atom samples would
        otherwise need >20 GB).  iu_int / iv_int are int32; voxel grids never
        approach 2^31 cells per axis.

        Args:
            sample: Chunked sample object.
            stage: Rotation (3x3) and translation (3,) in Angstrom.
            edges_A: (n_final + 1,) slice depth edges in Angstrom.
            n_final: Number of slices.

        Returns:
            dict cid (1-indexed chunk id) -> table with keys

                all_pos      (N_chunk, 3) float32 lab-frame positions (A)
                all_spc      (N_chunk,) species labels
                iu_int, iv_int    (N_chunk,) int32 voxel indices, sorted by slice
                iu_frac, iv_frac  (N_chunk,) float32 weights in [0, 1)
                slice_starts (n_final + 1,) int64 slice boundaries into THIS
                             chunk's sorted arrays
                N_total      atoms in this chunk

            plus "_meta" with N_total_all (atoms over all chunks) and
            chunk_ids (ordered populated chunk ids).  None if no atoms.
        """
        NyB = int(self._beam_Ny)
        NzB = int(self._beam_Nz)
        du_A = float(self._beam_du)
        dv_A = float(self._beam_dv)
        inv_du = np.float32(1.0 / du_A)
        inv_dv = np.float32(1.0 / dv_A)
        e1 = self._beam_e1.astype(np.float32)
        e2 = self._beam_e2.astype(np.float32)
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        uc = np.float32(self._beam_uc)
        vc = np.float32(self._beam_vc)

        per_chunk = {}
        chunk_ids = []
        N_total_all = 0

        chunk_total = int(sample.chunk_total or 0)
        for cid in range(1, chunk_total + 1):
            spc_host = sample.load_chunk_species(cid, use_gpu=False)
            pos_host = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
            if pos_host.size == 0:
                continue
            pos_lab = pos_host @ stage.rotation.astype(np.float32).T
            pos_lab = pos_lab + stage.translation.astype(np.float32)

            # ---- FP32-robust dot products pos_lab . e1 and pos_lab . e2 ----
            hx, lx = self._two_prod_fp32(pos_lab[:, 0], e1[0])
            hy, ly = self._two_prod_fp32(pos_lab[:, 1], e1[1])
            hz, lz = self._two_prod_fp32(pos_lab[:, 2], e1[2])
            s1, e_s1 = self._two_sum_fp32(hx, hy)
            au_high, e_s2 = self._two_sum_fp32(s1, hz)
            au_low = (e_s1 + e_s2 + lx + ly + lz).astype(np.float32)

            hx2, lx2 = self._two_prod_fp32(pos_lab[:, 0], e2[0])
            hy2, ly2 = self._two_prod_fp32(pos_lab[:, 1], e2[1])
            hz2, lz2 = self._two_prod_fp32(pos_lab[:, 2], e2[2])
            s1v, e_s1v = self._two_sum_fp32(hx2, hy2)
            av_high, e_s2v = self._two_sum_fp32(s1v, hz2)
            av_low = (e_s1v + e_s2v + lx2 + ly2 + lz2).astype(np.float32)

            # ---- FP32-robust division au / du_A and av / dv_A ----
            iu_h_high, iu_h_low = self._two_prod_fp32(au_high, inv_du)
            iu_total_low = (iu_h_low + au_low * inv_du).astype(np.float32)
            iu_h_high = (iu_h_high + uc).astype(np.float32)

            iv_h_high, iv_h_low = self._two_prod_fp32(av_high, inv_dv)
            iv_total_low = (iv_h_low + av_low * inv_dv).astype(np.float32)
            iv_h_high = (iv_h_high + vc).astype(np.float32)

            # Split into int32 index + float32 fraction (FP32-robust).
            iu_int_h = np.floor(iu_h_high).astype(np.int32)
            frac_h_u = (iu_h_high - iu_int_h.astype(np.float32)).astype(np.float32)
            iu_frac = (frac_h_u + iu_total_low).astype(np.float32)
            wrap_u = np.floor(iu_frac).astype(np.int32)
            iu_int = (iu_int_h + wrap_u).astype(np.int32)
            iu_frac = (iu_frac - wrap_u.astype(np.float32)).astype(np.float32)

            iv_int_h = np.floor(iv_h_high).astype(np.int32)
            frac_h_v = (iv_h_high - iv_int_h.astype(np.float32)).astype(np.float32)
            iv_frac = (frac_h_v + iv_total_low).astype(np.float32)
            wrap_v = np.floor(iv_frac).astype(np.int32)
            iv_int = (iv_int_h + wrap_v).astype(np.int32)
            iv_frac = (iv_frac - wrap_v.astype(np.float32)).astype(np.float32)

            # Slice index along the beam direction
            s_vals = (pos_lab[:, 0] * k_hat[0]
                      + pos_lab[:, 1] * k_hat[1]
                      + pos_lab[:, 2] * k_hat[2])
            k_idx = np.clip(
                np.searchsorted(edges_A, s_vals, side="right") - 1,
                0, n_final - 1,
            ).astype(np.int32)

            # Sort by slice so each slice's atoms form a contiguous range.
            sort_idx = np.argsort(k_idx, kind='stable')
            pos_lab = pos_lab[sort_idx]
            spc_host = np.asarray(spc_host)[sort_idx]
            iu_int = iu_int[sort_idx]
            iv_int = iv_int[sort_idx]
            iu_frac = iu_frac[sort_idx]
            iv_frac = iv_frac[sort_idx]
            k_idx = k_idx[sort_idx]

            # slice_starts index into THIS chunk's sorted arrays.
            chunk_slice_starts = np.zeros(n_final + 1, dtype=np.int64)
            counts = np.bincount(k_idx, minlength=n_final)
            np.cumsum(counts, out=chunk_slice_starts[1:])

            N_chunk = int(pos_lab.shape[0])
            per_chunk[cid] = {
                "all_pos": pos_lab,
                "all_spc": spc_host,
                "iu_int": iu_int,
                "iv_int": iv_int,
                "iu_frac": iu_frac,
                "iv_frac": iv_frac,
                "slice_starts": chunk_slice_starts,
                "N_total": N_chunk,
            }
            chunk_ids.append(cid)
            N_total_all += N_chunk

        if not chunk_ids:
            return None
        per_chunk["_meta"] = {
            "N_total_all": int(N_total_all),
            "chunk_ids": chunk_ids,
        }
        return per_chunk

    def _sample_envelopes_at_atoms_inplace(
        self, E_beams_gpu, iu_int, iv_int, iu_frac, iv_frac,
        out_M_amps_slice, NyB, NzB,
    ):
        """
        Bilinear-interpolate the M envelopes at atom positions, writing into
        out_M_amps_slice.  Atoms outside the grid get 0.

        Args:
            E_beams_gpu: list of M CuPy complex64 arrays (NyB, NzB).
            iu_int, iv_int: (N,) voxel indices from _build_atom_table_for_multislice.
            iu_frac, iv_frac: (N,) float32 weights in [0, 1).
            out_M_amps_slice: (N, M) numpy complex64, written in place.
            NyB, NzB: grid dimensions.
        """
        M = len(E_beams_gpu)
        in_grid = ((iu_int >= 0) & (iu_int < NyB - 1) &
                   (iv_int >= 0) & (iv_int < NzB - 1))
        iy0 = np.clip(iu_int, 0, NyB - 2)
        iz0 = np.clip(iv_int, 0, NzB - 2)
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        fy = iu_frac
        fz = iv_frac

        for m_idx in range(M):
            E_m = cp.asnumpy(E_beams_gpu[m_idx])
            v00 = E_m[iy0, iz0]
            v10 = E_m[iy1, iz0]
            v01 = E_m[iy0, iz1]
            v11 = E_m[iy1, iz1]
            interp = (v00 * (1.0 - fy) * (1.0 - fz)
                      + v10 * fy * (1.0 - fz)
                      + v01 * (1.0 - fy) * fz
                      + v11 * fy * fz)
            interp = interp.astype(np.complex64, copy=False)
            interp[~in_grid] = 0.0
            out_M_amps_slice[:, m_idx] = interp

    def _run_multislice_with_sampling(
        self, chi_maps, beam_info, transmission_k_A, dz_A, n_final,
        per_chunk_tables, atom_M_amps_per_chunk,
        NyB, NzB, M, thickness_A,
        apply_propagation=True,
        slice_iter_reverse=False,
        initial_envelope=None,
    ):
        """
        Multislice driver (Eq. 9-11) that samples the envelopes at atom
        positions as each slice is passed.

        Forward (slice_iter_reverse=False): k = 0 .. n_final-1; per slice
        T(+dz, chi) then P(+dz), then sample slice k.  Initial state is
        E_0 = self._beam_E0_map, E_m = 0 for m > 0, unless initial_envelope
        is given.

        Reverse (slice_iter_reverse=True), used by the reciprocal multislice
        for the output-side correction (Lorentz reciprocity): k = n_final-1
        .. 0; per slice P(-dz) then T(+dz, chi), then sample slice k.  The
        chi maps are reused UNCONJUGATED: reciprocity in an absorbing medium
        relies on the symmetric (non-Hermitian) Helmholtz operator, and
        chi -> chi* would give the gain-medium inverse instead.  Validated
        to 0.16% relative error on a forward + backward round trip.
        initial_envelope is required (the entrance wave at the EXIT face,
        travelling along -k_out).

        Args:
            chi_maps: dict (a, b) -> list of n_final (NyB, NzB) complex64
                arrays from _build_structure_factor_maps_gpu(born_convention=True).
            beam_info: list of M beam descriptors.
            transmission_k_A: transmission phase prefactor
                (= pi / lambda * dz_A for born-convention maps).
            dz_A: slice thickness in Angstrom, always positive; the sign for
                the reverse pass is handled here.
            n_final: number of slices.
            per_chunk_tables: from _build_atom_table_for_multislice.
            atom_M_amps_per_chunk: dict cid -> (N_chunk, M) complex64,
                written in place.
            NyB, NzB, M: grid dimensions and beam count.
            thickness_A: total crystal thickness in Angstrom.
            apply_propagation: apply the ASP step (off for thickness-
                degenerate cases).
            slice_iter_reverse: run the reverse variant above.
            initial_envelope: list of M CuPy (NyB, NzB) complex64 arrays, or
                None for (self._beam_E0_map, 0, ...).

        Returns:
            None; atom_M_amps_per_chunk is written in place.
        """
        chunk_ids = per_chunk_tables["_meta"]["chunk_ids"]

        # Initial envelope state on GPU
        if initial_envelope is None:
            E_beams_gpu = [cp.asarray(self._beam_E0_map.astype(np.complex64))]
            for _m in range(1, M):
                E_beams_gpu.append(cp.zeros((NyB, NzB), dtype=cp.complex64))
        else:
            if len(initial_envelope) != M:
                raise ValueError(
                    f"initial_envelope must have {M} entries (got "
                    f"{len(initial_envelope)})")
            E_beams_gpu = []
            for em in initial_envelope:
                if isinstance(em, cp.ndarray):
                    E_beams_gpu.append(em.astype(cp.complex64, copy=True))
                else:
                    E_beams_gpu.append(cp.asarray(em, dtype=cp.complex64))

        # Per-step propagation distance (signed for reverse pass)
        propagate_ok = bool(apply_propagation) and (thickness_A > 0.0) and (dz_A > 0.0)
        prop_dz_A = -dz_A if slice_iter_reverse else dz_A

        slice_order = range(n_final - 1, -1, -1) if slice_iter_reverse else range(n_final)

        for k in slice_order:
            chi_slice = {key: cp.asarray(chi_maps[key][k], dtype=cp.complex64)
                         for key in chi_maps}

            if slice_iter_reverse:
                # Reverse: P(-dz) then T(+dz, chi) -- the physical backward
                # wave, not the inverse of the forward step.
                if propagate_ok:
                    try:
                        self._beam_propagation_step_gpu(E_beams_gpu, prop_dz_A, beam_info)
                    except Exception:
                        pass
                self._beam_transmission_step_gpu(
                    E_beams_gpu, chi_slice, transmission_k_A)
            else:
                # Forward: T(+dz, chi) then P(+dz)
                self._beam_transmission_step_gpu(
                    E_beams_gpu, chi_slice, transmission_k_A)
                if propagate_ok:
                    try:
                        self._beam_propagation_step_gpu(E_beams_gpu, prop_dz_A, beam_info)
                    except Exception:
                        pass

            del chi_slice
            cp.get_default_memory_pool().free_all_blocks()

            # Sample slice k's atoms in every chunk, after the slice's operators.
            for cid in chunk_ids:
                tbl = per_chunk_tables[cid]
                a_start = int(tbl["slice_starts"][k])
                a_end = int(tbl["slice_starts"][k + 1])
                if a_end <= a_start:
                    continue
                self._sample_envelopes_at_atoms_inplace(
                    E_beams_gpu,
                    tbl["iu_int"][a_start:a_end],
                    tbl["iv_int"][a_start:a_end],
                    tbl["iu_frac"][a_start:a_end],
                    tbl["iv_frac"][a_start:a_end],
                    atom_M_amps_per_chunk[cid][a_start:a_end, :],
                    NyB, NzB,
                )

        # Free envelope grids
        for _i in range(len(E_beams_gpu)):
            del E_beams_gpu[0]
        cp.get_default_memory_pool().free_all_blocks()

    def atomic_scattering_dynamical(self, sample, detector, stage,
                                    crystal=None,
                                    M=1,
                                    g_vectors=None,
                                    offset=None,
                                    use_gpu=True,
                                    n_slices=None,
                                    target_phase_step=0.1,
                                    kernel_radius=0,
                                    pad_factor=2,
                                    padding_mode="edge",
                                    absorption_multiplier=1.0,
                                    apply_polarization=False,
                                    remove_forward=True,
                                    spherical_decay=False,
                                    apply_propagation=True,
                                    analyser_mode="off",
                                    analyser_acceptance_angle_rad=25e-4,
                                    analyser_darwin_halfwidth_rad=0.0,
                                    NN_dist_A=None,
                                    convergent_regime_check="warn",
                                    commensurate_supercell=False,
                                    force_unconverged=False,
                                    dynamical_mode="forward_only",
                                    multi_gpu=False,
                                    n_gpus=None):
        """
        Atomistic dynamical X-ray scattering.  A voxelized multi-beam multislice
        gives the wavefield inside the sample; the envelopes E_g are sampled
        trilinearly at every atom and each atom rescatters to the detector via a
        Lippmann-Schwinger pass (Eq. 6 of the dynamical-method plan) with phase
        2*pi*g_m.r_a + k_0|r_det - r_a| - k_in.r_a.  Exact limits: vacuum
        kinematic (M=1, chi=0), refraction and Beer-Lambert absorption (M=1),
        two-beam Pendellosung and Borrmann transmission (M=2), and
        Takagi-Taupin defect imaging (M=2).

        Beams for M > 1 come from, in priority order: ``g_vectors``, the
        ``crystal`` (reflections nearest the Ewald sphere), or a 3D FFT of the
        atomic density (falls back to M=1 for amorphous samples).

        Args:
            sample: Chunked sample object.
            detector: Detector with ``pixel_coordinates`` (3, Ny*Nz) in Angstrom.
            stage: Stage providing ``rotation`` (3x3) and ``translation`` (3,).
            crystal: Crystal used for fast beam selection.
            M (int): Number of coupled beams (1 = kinematic + transmission).
            g_vectors: List of (3,) crystallographic G vectors (1/Angstrom, no 2pi).
            offset: Complex field subtracted from the result.
            use_gpu (bool): Must be True; CuPy is required.
            n_slices (int or None): Number of depth slices; auto if None.
            target_phase_step (float): Per-slice phase step for the auto-slicer.
            kernel_radius (int): Gaussian blur radius for chi maps.
            pad_factor (float): FFT padding factor for ASP propagation.
            padding_mode (str): "edge" or "constant".
            absorption_multiplier (float): Scales absorption (1.0 = physical).
            apply_polarization (bool): Apply the per-pixel polarization factor.
            remove_forward (bool): Remove forward scattering in the kernel.
            spherical_decay (bool): Apply 1/R decay.
            apply_propagation (bool): Apply the per-beam ASP step in the
                split-step (Eq. 11).  False gives a transmission-only
                multislice (column approximation).
            NN_dist_A (float or None): Nearest-neighbour distance in Angstrom
                for the convergent-regime check; estimated from the sample if
                None (check skipped with a warning if that fails).
            convergent_regime_check (str): "off", "warn", or "error".  For
                M > 1 the beam grid pitch must lie in [NN_dist/3, 1/(2*|g_max|)].
                Too coarse: chi_g aliases and the result is silently kinematic
                only.  Too fine: per-atom amplitudes are biased by 30-40%.
                Use "error" in validation pipelines.
            commensurate_supercell (bool): Allow voxel pitch down to NN_dist/2
                for perfect commensurate supercells.
            force_unconverged (bool): Suppress the RuntimeError raised when the
                convergent window collapses (|g_max|*NN_dist >= 1.5).
                Diagnostic use only.
            dynamical_mode (str): "forward_only" runs one forward multislice
                with kinematic output-side propagation.  "full" adds a
                reciprocal multislice (Lorentz reciprocity) and multiplies the
                two per-atom envelopes; it differs only off-Bragg and in
                asymmetric geometries.
            multi_gpu (bool): Split the final LS kernel pass across CUDA
                devices; stages 1-3 are replicated per device.  Matches
                single-GPU up to FP32 reduction order.
            n_gpus (int or None): Cap on GPUs used when multi_gpu=True.

        Returns:
            np.ndarray: Complex64 array of shape (Nz, Ny).
        """
        M = int(max(1, M))
        use_gpu = bool(use_gpu and (cp is not None))

        if not use_gpu:
            raise RuntimeError(
                "GPU is required for dynamical scattering (use_gpu=True, CuPy needed).")

        Ny, Nz = detector.shape
        final_result = np.zeros((Nz, Ny), dtype=np.complex64)

        chunk_total = int(sample.chunk_total or 0)
        if chunk_total == 0:
            if offset is not None:
                final_result -= offset
            return final_result

        # Constants (Angstrom units)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        two_pi = 2.0 * np.pi
        lam_A = float(self._wavelength) * 1e10
        kA = two_pi / lam_A   # 1/Angstrom
        abs_m = float(absorption_multiplier)

        # ---------------- 1. Depth bounds and slicing ----------------------
        s_min_A, s_max_A = self._compute_global_depth_bounds(sample, stage)
        thickness_A = float(max(0.0, s_max_A - s_min_A))

        delta_list_cached = None
        beta_list_cached = None
        if thickness_A <= 0.0:
            n_final = 1
            edges_A = np.array([0.0, 1.0], dtype=np.float32)
        else:
            if n_slices is None:
                (n_final, edges_A,
                 delta_list_cached, beta_list_cached, _) = \
                    self._auto_slice_count_linear_regime(
                        sample=sample, stage=stage,
                        kernel_radius=kernel_radius,
                        target_step=float(target_phase_step),
                        use_gpu=use_gpu, max_slices=2048,
                        n_init=None, absorption_multiplier=abs_m,
                    )
            else:
                n_final = int(max(1, n_slices))
                edges_A = np.linspace(s_min_A, s_max_A, n_final + 1, dtype=np.float32)

        dz_A = float(thickness_A) / max(n_final, 1)

        # ---------------- 2. Beam selection (with cos_theta) ----------------
        k_hat = np.asarray(self._direction, dtype=np.float64)
        k_hat = k_hat / np.linalg.norm(k_hat)
        k0_vec = (kA * k_hat).astype(np.float64)

        if M > 1:
            if g_vectors is not None:
                beam_info = self._beams_from_g_vectors(g_vectors, k0_vec, kA)
            elif crystal is not None:
                beam_info = self._auto_select_beams(crystal, stage, M_max=M)
            else:
                beam_info = self._auto_detect_beams(sample, stage, M_max=M)
            M = len(beam_info)
        else:
            beam_info = [{
                "hkl": (0, 0, 0),
                "G": np.zeros(3, dtype=np.float32),
                "k_vec": k0_vec.astype(np.float32),
                "excitation_error": 0.0,
                "cos_theta": 1.0,
            }]

        # ---------------- 2.5 Convergent-regime check (master_plan I.6) -----
        # For M > 1 the pitch (du, dv) set by _init_beam_grid must lie in
        # [NN_dist/3, 1/(2*|g_max|)].  Too coarse: chi_g aliases and the result
        # is silently kinematic/refractive only.  Too fine: bimodal chi field,
        # per-atom amplitudes biased 30-40%.  Warn or raise only; never resize
        # the grid here, since callers may depend on it downstream.
        if M > 1 and convergent_regime_check != "off":
            check_mode = str(convergent_regime_check).lower().strip()
            if check_mode not in ("off", "warn", "error"):
                check_mode = "warn"

            NN_resolved = NN_dist_A
            if NN_resolved is None:
                NN_resolved = self._estimate_nearest_neighbor_distance(
                    sample, n_samples=2048)

            if NN_resolved is not None and NN_resolved > 0:
                g_list = [np.asarray(b["G"], dtype=np.float64) for b in beam_info]
                Lx_A_check = float(NyB) * float(self._beam_du)
                Ly_A_check = float(NzB) * float(self._beam_dv)
                Lz_A_check = float(max(thickness_A, 1.0))
                try:
                    Nx_auto, Ny_auto, _Nz_auto, dx_target, dy_target, _dz_target, diag = \
                        beam._auto_voxel_grid(
                            g_list, (Lx_A_check, Ly_A_check, Lz_A_check),
                            float(NN_resolved),
                            float(dz_A) if dz_A > 0 else 1.0,
                            strict_nyquist=(not commensurate_supercell),
                            force_unconverged=force_unconverged,
                        )
                except RuntimeError as _e:
                    if check_mode == "error":
                        raise
                    warnings.warn(
                        f"[atomic_scattering_dynamical] convergent-regime "
                        f"window collapsed: {_e}.  Continuing in unconverged "
                        f"regime; expect 30-40% bias on Bragg-channel "
                        f"amplitudes.")
                else:
                    du_now = float(self._beam_du)
                    dv_now = float(self._beam_dv)
                    in_regime = (
                        diag["dx_tsc_lower"] <= du_now <= diag["dx_nyquist_upper"]
                        and diag["dx_tsc_lower"] <= dv_now <= diag["dx_nyquist_upper"]
                    )
                    if not in_regime:
                        msg = (
                            f"Beam grid (du={du_now:.4f} A, dv={dv_now:.4f} A) "
                            f"is outside the convergent window for M={M}: "
                            f"[{diag['dx_tsc_lower']:.4f}, "
                            f"{diag['dx_nyquist_upper']:.4f}] A.  "
                            f"_auto_voxel_grid recommends ~{dx_target:.4f} A.  "
                        )
                        if du_now > diag["dx_nyquist_upper"]:
                            msg += ("Current grid is TOO COARSE: chi_g is "
                                    "aliased; simulation will silently "
                                    "produce KINEMATIC-only / refractive "
                                    "results (no Pendellosung, no Borrmann). "
                                    " ")
                        elif du_now < diag["dx_tsc_lower"]:
                            msg += ("Current grid is TOO FINE: chi field is "
                                    "bimodal; per-atom amplitudes biased by "
                                    "30-40%.  ")
                        msg += (f"Suggested action: increase beam_samples so "
                                f"that du = dv ~ {dx_target:.4f} A (e.g. "
                                f"beam_samples=({Nx_auto},{Ny_auto}) for the "
                                f"current sample cross-section).")
                        if check_mode == "error":
                            raise RuntimeError(
                                "[atomic_scattering_dynamical] " + msg)
                        warnings.warn(
                            "[atomic_scattering_dynamical] " + msg)
            else:
                warnings.warn(
                    "[atomic_scattering_dynamical] convergent-regime check "
                    "skipped: NN_dist_A could not be resolved (pass NN_dist_A "
                    "explicitly to enable the check).")

        # ---------------- 3. Build chi maps --------------------------------
        # M=1: legacy column-integral chi_0 from _compute_beam_slice_integrals_*;
        #      the transmission step uses k_A = 2*pi/lambda with dz already
        #      folded into the column integral (same convention as
        #      atomic_transmission).
        # M>1: voxel-density Born-convention chi_g maps.  The transmission
        #      propagator is exp(-i*pi*dz/lambda * X) (Eq. 10), so the phase
        #      prefactor is pi/lambda*dz, NOT 2*pi/lambda*dz: the 1/pi in the
        #      chi prefactor (Eq. 7) pairs with this pi.  Using 2*pi doubles
        #      the Pendellosung frequency.
        if M == 1:
            transmission_k_A = float(kA)
            if thickness_A > 0.0:
                if delta_list_cached is None or beta_list_cached is None:
                    delta_list, beta_list = self._compute_beam_slice_integrals_gpu(
                        sample, stage, edges_A, kernel_radius)
                else:
                    delta_list = delta_list_cached
                    beta_list = beta_list_cached
                chi_maps = {(0, 0): []}
                for k in range(n_final):
                    dk = delta_list[k] if isinstance(delta_list[k], np.ndarray) \
                        else delta_list[k].get()
                    bk = beta_list[k] if isinstance(beta_list[k], np.ndarray) \
                        else beta_list[k].get()
                    chi_k = np.empty((NyB, NzB), dtype=np.complex64)
                    chi_k.real = -dk.astype(np.float32)
                    chi_k.imag = np.maximum((abs_m * bk).astype(np.float32), 0.0)
                    chi_maps[(0, 0)].append(chi_k)
            else:
                chi_maps = {(0, 0): [np.zeros((NyB, NzB), dtype=np.complex64)
                                     for _ in range(n_final)]}
        else:
            # Eq. 10: kernel prefactor is (pi/lambda)*dz, not (2*pi/lambda)*dz.
            transmission_k_A = float(np.pi / lam_A) * float(dz_A)
            if thickness_A > 0.0:
                sf_maps = self._build_structure_factor_maps_gpu(
                    sample, stage, edges_A, beam_info, kernel_radius,
                    born_convention=True)
                chi_maps = {}
                for key in sf_maps:
                    chi_maps[key] = []
                    for k in range(n_final):
                        chi_k = -sf_maps[key][k].astype(np.complex64)
                        chi_k.imag *= abs_m
                        chi_maps[key].append(chi_k)
            else:
                chi_maps = {}
                for a in range(M):
                    for b in range(M):
                        chi_maps[(a, b)] = [np.zeros((NyB, NzB), dtype=np.complex64)
                                            for _ in range(n_final)]

        # ---------------- 4. (Initial envelopes are now set inside the
        #                       _run_multislice_with_sampling helper.) -------

        # ---------------- 5. FP32-robust per-chunk atom tables -------------
        per_chunk_tables = self._build_atom_table_for_multislice(
            sample, stage, edges_A, n_final)
        if per_chunk_tables is None:
            if offset is not None:
                final_result -= offset
            return final_result

        chunk_ids = per_chunk_tables["_meta"]["chunk_ids"]
        N_total = int(per_chunk_tables["_meta"]["N_total_all"])

        # Per-chunk M-vector amplitudes, host-resident (~M * N_chunk * 8 bytes
        # each), so the multislice and LS steps can work one chunk at a time.
        atom_M_amps_per_chunk = {
            cid: np.zeros((per_chunk_tables[cid]["N_total"], M),
                          dtype=np.complex64)
            for cid in chunk_ids
        }

        # ---------------- 6. Multislice loop (Lie-Trotter split-step) ------
        # Forward pass: samples the input-side illumination A_g_F at every
        # atom into atom_M_amps_per_chunk.
        self._run_multislice_with_sampling(
            chi_maps, beam_info, transmission_k_A, dz_A, n_final,
            per_chunk_tables, atom_M_amps_per_chunk,
            NyB, NzB, M, thickness_A,
            apply_propagation=apply_propagation,
            slice_iter_reverse=False,
            initial_envelope=None,
        )

        # ---------------- 6b. Reciprocal multislice (dynamical_mode="full") -
        # Output-side dynamical correction via Lorentz reciprocity: a second
        # multislice runs backward through the same medium, entering at the
        # exit face.  Its per-atom amplitudes A_g_R multiply A_g_F elementwise
        # so the LS pass sees the combined illumination.  At symmetric Laue
        # exact Bragg the product is constant in z_a; the correction shows up
        # off-Bragg, in asymmetric Laue, and near the Borrmann condition.
        dyn_mode_str = str(dynamical_mode or "forward_only").lower().strip()
        if dyn_mode_str not in ("forward_only", "full"):
            warnings.warn(
                f"[atomic_scattering_dynamical] unknown dynamical_mode="
                f"{dynamical_mode!r}; defaulting to 'forward_only'.")
            dyn_mode_str = "forward_only"

        if dyn_mode_str == "full" and M > 1 and thickness_A > 0.0 and dz_A > 0.0:
            # Virtual source direction for the reciprocal pass: the detector's
            # centre pixel, which minimises sampling error for compact ROIs.
            # Wide ROIs can be run as separate crops and combined.
            mp_h = np.asarray(detector.pixel_coordinates)
            cidx_pix = (Nz // 2) * Ny + (Ny // 2)
            cx_A = float(mp_h[0, cidx_pix])
            cy_A = float(mp_h[1, cidx_pix])
            cz_A = float(mp_h[2, cidx_pix])
            cnorm_A = float(np.sqrt(cx_A * cx_A + cy_A * cy_A + cz_A * cz_A))
            if cnorm_A > 0:
                k_out_central = (kA / cnorm_A) * np.array(
                    [cx_A, cy_A, cz_A], dtype=np.float64)
            else:
                k_out_central = beam_info[0]["k_vec"].astype(np.float64) + \
                                np.asarray(beam_info[1]["G"], dtype=np.float64)

            # Reciprocal channels are centred on -k_out_central + g_m.  The chi
            # maps are reused unchanged (reciprocity, NOT chi -> chi*).
            beam_info_recip = []
            for m_idx in range(M):
                G_m = np.asarray(beam_info[m_idx]["G"], dtype=np.float64)
                k_recip_m = -k_out_central + G_m
                cos_theta_recip = float(abs(k_recip_m[2]) / kA) if kA > 0 else 1.0
                beam_info_recip.append({
                    "hkl": beam_info[m_idx].get("hkl", (0, 0, 0)),
                    "G": G_m.astype(np.float32),
                    "k_vec": k_recip_m.astype(np.float32),
                    "excitation_error": 0.0,
                    "cos_theta": cos_theta_recip,
                })

            # Entrance condition at the exit face: unit plane wave in the m=0
            # channel, zero elsewhere; the pass marches from z = thickness to 0.
            initial_env_recip = [cp.ones((NyB, NzB), dtype=cp.complex64)]
            for _m in range(1, M):
                initial_env_recip.append(cp.zeros((NyB, NzB), dtype=cp.complex64))

            # Separate per-chunk buffer for A_g_R, combined with A_g_F below.
            atom_M_amps_recip = {
                cid: np.zeros((per_chunk_tables[cid]["N_total"], M),
                              dtype=np.complex64)
                for cid in chunk_ids
            }
            self._run_multislice_with_sampling(
                chi_maps, beam_info_recip, transmission_k_A, dz_A, n_final,
                per_chunk_tables, atom_M_amps_recip,
                NyB, NzB, M, thickness_A,
                apply_propagation=apply_propagation,
                slice_iter_reverse=True,
                initial_envelope=initial_env_recip,
            )

            # A_g[a, m] <- A_g_F[a, m] * A_g_R[a, m]; the LS kernel uses it as is.
            for cid in chunk_ids:
                atom_M_amps_per_chunk[cid] *= atom_M_amps_recip[cid]
            del atom_M_amps_recip
            del initial_env_recip
            cp.get_default_memory_pool().free_all_blocks()


        # ---------------- 7. Per-chunk LS far-field kernel pass ------------
        # One chunk at a time, so per-atom GPU buffers scale with chunk size,
        # not N_total.  This is what keeps billion-atom samples from OOM-ing.

        db_f0 = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        # Per-element f0(0) and anomalous (f1, f2), keyed by element symbol.
        f0_zero_lookup = {}
        anom_lookup = {}
        for el_sym, f0p in db_f0.items():
            f0_zero_lookup[el_sym] = float(
                f0p[5] + f0p[0] + f0p[1] + f0p[2] + f0p[3] + f0p[4]
            )
        for el_sym, tbl in db_f1f2.items():
            anom_lookup[el_sym] = self.get_f1f2_from_params(self._energy, tbl)

        # Kernel for this M (cached by build_interaction_kernel).
        if not hasattr(self, "_phase_tol_rad"):
            self._phase_tol_rad = 1e-3
        if not hasattr(self, "_series_terms"):
            self._series_terms = 2
        if not hasattr(self, "_global_use_series"):
            self._global_use_series = True

        interaction_kernel = self.build_interaction_kernel(
            series_terms=self._series_terms,
            force_mode=("series" if self._global_use_series else "exact"),
            m_beams=M,
        )

        # g vectors, 1/Angstrom -> 1/m; constant across atoms, uploaded once.
        g_vecs_host = np.zeros((M, 3), dtype=np.float32)
        for m_idx in range(M):
            g_vecs_host[m_idx, :] = (
                np.asarray(beam_info[m_idx]["G"], dtype=np.float32) * 1e10
            )
        g_vecs_d = cp.asarray(g_vecs_host.ravel())

        # Forward wavevector (1/m) is the kernel's global phase reference; the
        # m > 0 lattice phases are added inside the kernel (Eq. 13).
        k_vec_A0 = beam_info[0]["k_vec"]
        k_in_x = np.float32(float(k_vec_A0[0]) * 1e10)
        k_in_y = np.float32(float(k_vec_A0[1]) * 1e10)
        k_in_z = np.float32(float(k_vec_A0[2]) * 1e10)
        self._require_forward_beam((k_in_x, k_in_y, k_in_z))

        # Detector coordinates (metres), constant across chunks.
        mp = detector.pixel_coordinates
        xg = cp.asarray((mp[0, :].astype(np.float32) / 1e10))
        yg = cp.asarray((mp[1, :].astype(np.float32) / 1e10))
        zg = cp.asarray((mp[2, :].astype(np.float32) / 1e10))

        dfield_gpu = cp.zeros((Ny * Nz,), dtype=cp.complex64)
        block2d = (16, 16)
        grid2d = ((Ny + block2d[0] - 1) // block2d[0],
                  (Nz + block2d[1] - 1) // block2d[1])

        remove_forward_flag = np.int32(1 if remove_forward else 0)
        apply_polarization_flag = np.int32(1 if apply_polarization else 0)
        apply_decay_flag = np.int32(1 if spherical_decay else 0)
        pol_rate = np.float32(getattr(self, "_pol_perp_rate", 0.5))

        # Analyser arguments follow atomic_direct_interaction: "off",
        # "top-hat"/"tophat"/"top_hat", or "darwin"/"rolloff".
        if isinstance(analyser_mode, str):
            _am = analyser_mode.strip().lower()
            if _am in ("off", "none", "disabled"):
                analyser_kind_int = 0
            elif _am in ("top_hat", "tophat", "top-hat", "top"):
                analyser_kind_int = 1
            elif _am in ("darwin", "rolloff", "roll-off"):
                analyser_kind_int = 2
            else:
                analyser_kind_int = 0
        else:
            analyser_kind_int = int(analyser_mode)
            if analyser_kind_int not in (0, 1, 2):
                analyser_kind_int = 0
        apply_analyser_flag = np.int32(1 if analyser_kind_int != 0 else 0)
        analyser_kind_flag = np.int32(analyser_kind_int)

        # Analyser centre = unit vector from origin to the detector's centre pixel.
        if apply_analyser_flag:
            mp_h = np.asarray(detector.pixel_coordinates)
            cidx = (Nz // 2) * Ny + (Ny // 2)
            cx_m = float(mp_h[0, cidx]) / 1e10
            cy_m = float(mp_h[1, cidx]) / 1e10
            cz_m = float(mp_h[2, cidx]) / 1e10
            cnorm = float(np.sqrt(cx_m * cx_m + cy_m * cy_m + cz_m * cz_m))
            if cnorm > 0:
                cx_m /= cnorm
                cy_m /= cnorm
                cz_m /= cnorm
        else:
            cx_m = cy_m = cz_m = 0.0
        centre_x_f = np.float32(cx_m)
        centre_y_f = np.float32(cy_m)
        centre_z_f = np.float32(cz_m)
        accept_angle_f = np.float32(float(analyser_acceptance_angle_rad))
        darwin_hw_f = np.float32(float(analyser_darwin_halfwidth_rad))

        # Size the scatter sub-chunk from free GPU memory.  The largest
        # resident per-chunk array here is the float32 position array
        # (12 B/atom); the M amplitudes stay on the host.
        _n_max = max(int(per_chunk_tables[c]["N_total"]) for c in chunk_ids)
        SCATTER_CHUNK = self._scatter_subchunk_size(M, resident_bytes=12 * _n_max)
        _kernel_chunk = int(getattr(self, "_kernel_chunk_size", 128))

        # Smallest sample-to-pixel distance, used by the origin policy.
        _R0_min = float(cp.sqrt(xg * xg + yg * yg + zg * zg).min())
        # (R0_min, dotv_lo, dotv_hi), constant across chunks; the fast path
        # fits its form-factor polynomial over that dotv interval.
        try:
            import fast_kernel as _fk_mod
            _det_extent = _fk_mod.detector_extent(xg, yg, zg)
        except Exception:
            _det_extent = None

        # ---------------- 7b. Multi-GPU dispatch (optional) ----------------
        # Partition chunk_ids across GPUs and run only the LS kernel pass on
        # each device in a thread; stages 1-3 are already done above.  The LS
        # sum is additive over atom partitions, so summing the per-GPU
        # detector arrays matches single-GPU up to FP32 reduction order.
        try:
            n_gpus_avail = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            n_gpus_avail = 1
        if multi_gpu and n_gpus is not None:
            n_gpus_use = int(min(max(1, n_gpus), n_gpus_avail))
        elif multi_gpu:
            n_gpus_use = max(1, n_gpus_avail)
        else:
            n_gpus_use = 1

        if n_gpus_use > 1 and len(chunk_ids) > 1:
            shards = [[] for _ in range(n_gpus_use)]
            for i, cid in enumerate(chunk_ids):
                shards[i % n_gpus_use].append(cid)

            partial_results = [None] * n_gpus_use

            def _ls_worker(dev_id, my_chunks, result_index):
                if not my_chunks:
                    partial_results[result_index] = np.zeros((Nz, Ny),
                                                              dtype=np.complex64)
                    return
                cp.cuda.Device(dev_id).use()

                # Per-device copies of detector and beam-set state; the kernel
                # handle is cached per device by build_interaction_kernel.
                _interaction_kernel = self.build_interaction_kernel(
                    series_terms=self._series_terms,
                    force_mode=("series" if self._global_use_series else "exact"),
                    m_beams=M,
                )
                _g_vecs_d = cp.asarray(g_vecs_host.ravel())
                _xg = cp.asarray((mp[0, :].astype(np.float32) / 1e10))
                _yg = cp.asarray((mp[1, :].astype(np.float32) / 1e10))
                _zg = cp.asarray((mp[2, :].astype(np.float32) / 1e10))
                _dfield_gpu = cp.zeros((Ny * Nz,), dtype=cp.complex64)

                for cid in my_chunks:
                    tbl = per_chunk_tables[cid]
                    N_chunk = int(tbl["N_total"])
                    if N_chunk == 0:
                        continue
                    # Per-chunk local origins (see the single-GPU path).
                    pos_chunk = tbl["all_pos"]
                    _pos_m = cp.asarray((pos_chunk / 1e10).astype(np.float32))

                    # Fast path first, exactly as in the single-GPU branch.
                    if self._fast_scatter(
                            _pos_m, atom_M_amps_per_chunk[cid],
                            tbl["all_spc"], db_f0, f0_zero_lookup,
                            anom_lookup, _xg, _yg, _zg, _dfield_gpu, Ny, Nz,
                            float(abs(k_in_x)), M,
                            analyser_kind_int if int(apply_analyser_flag) else 0,
                            remove_forward, apply_polarization,
                            spherical_decay, float(pol_rate), SCATTER_CHUNK):
                        del _pos_m
                        cp.get_default_memory_pool().free_all_blocks()
                        continue

                    _use_org, _eps_geom = self._origin_policy(_pos_m, _R0_min)
                    _apply_origin_flag = np.int32(1 if _use_org else 0)
                    spc_all = tbl["all_spc"]
                    amps_chunk = atom_M_amps_per_chunk[cid]

                    # Sub-chunk staging as in the single-GPU branch, so
                    # SCATTER_CHUNK bounds device memory here too.
                    for c_start in range(0, N_chunk, SCATTER_CHUNK):
                        c_end = min(c_start + SCATTER_CHUNK, N_chunk)
                        sub = self._stage_general_subchunk(
                            _pos_m[c_start:c_end], spc_all[c_start:c_end],
                            amps_chunk[c_start:c_end, :], _use_org,
                            _kernel_chunk, db_f0, f0_zero_lookup, anom_lookup,
                            (k_in_x, k_in_y, k_in_z))
                        _interaction_kernel(
                            grid2d, block2d,
                            (
                                np.int32(sub["n"]),
                                sub["kx"], sub["ky"], sub["kz"],
                                sub["px"], sub["py"], sub["pz"],
                                sub["amp"],
                                sub["anom"], sub["f0p"], sub["f0z"],
                                _xg, _yg, _zg,
                                _dfield_gpu,
                                np.int32(Ny), np.int32(Nz),
                                remove_forward_flag,
                                apply_polarization_flag,
                                apply_decay_flag,
                                pol_rate,
                                apply_analyser_flag, analyser_kind_flag,
                                centre_x_f, centre_y_f, centre_z_f,
                                accept_angle_f, darwin_hw_f,
                                _g_vecs_d,
                                _apply_origin_flag,
                                sub["org"],
                            )
                        )
                        cp.cuda.stream.get_current_stream().synchronize()
                        del sub
                    del _pos_m
                    cp.get_default_memory_pool().free_all_blocks()

                partial_results[result_index] = (
                    _dfield_gpu.reshape((Nz, Ny)).get()
                )
                del _dfield_gpu, _xg, _yg, _zg, _g_vecs_d
                cp.get_default_memory_pool().free_all_blocks()

            threads = []
            for gid in range(n_gpus_use):
                t = threading.Thread(target=_ls_worker,
                                     args=(gid, shards[gid], gid))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

            final_result = np.zeros((Nz, Ny), dtype=np.complex64)
            for pr in partial_results:
                if pr is not None:
                    final_result += pr

            # Free the single-GPU prep buffers.
            del dfield_gpu, xg, yg, zg, g_vecs_d
            cp.get_default_memory_pool().free_all_blocks()

            if offset is not None:
                final_result -= offset
            return final_result

        # ---------------- 7c. Single-GPU Stage-4 LS pass (default) ---------
        for cid in chunk_ids:
            tbl = per_chunk_tables[cid]
            N_chunk = int(tbl["N_total"])
            if N_chunk == 0:
                continue
            # ---- per-chunk local origins ------------------------------
            # Positions are Morton-sorted so each CHUNK_SIZE block is compact,
            # then passed as offsets from the block's own origin; the FP32
            # phase floor scales with k*|offset|*eps instead of k*|r_atom|*eps.
            pos_chunk = tbl["all_pos"]
            pos_m_g = cp.asarray((pos_chunk / 1e10).astype(np.float32))

            # Fast path first.  It does its own sorting, origins, padding and
            # tuning from absolute positions; if it declines, nothing has been
            # written and the general path below runs.
            if self._fast_scatter(
                    pos_m_g, atom_M_amps_per_chunk[cid], tbl["all_spc"],
                    db_f0, f0_zero_lookup, anom_lookup, xg, yg, zg,
                    dfield_gpu, Ny, Nz, float(abs(k_in_x)), M,
                    analyser_kind_int if int(apply_analyser_flag) else 0,
                    remove_forward, apply_polarization, spherical_decay,
                    float(pol_rate), SCATTER_CHUNK, _det_extent):
                del pos_m_g
                cp.get_default_memory_pool().free_all_blocks()
                continue

            use_origins, eps_geom = self._origin_policy(pos_m_g, _R0_min)
            apply_origin_flag = np.int32(1 if use_origins else 0)
            self._log(
                "verbose",
                f"[beam] LS chunk {cid}: local origins "
                f"{'on' if use_origins else 'off'} "
                f"(eps_geom={eps_geom:.3e}, CHUNK={_kernel_chunk}, "
                f"sub-chunks={-(-N_chunk // SCATTER_CHUNK)})"
            )
            spc_all = tbl["all_spc"]
            atom_M_amps_chunk = atom_M_amps_per_chunk[cid]

            # All per-atom staging is sized to the sub-chunk, so SCATTER_CHUNK
            # genuinely bounds device memory.
            for c_start in range(0, N_chunk, SCATTER_CHUNK):
                c_end = min(c_start + SCATTER_CHUNK, N_chunk)
                sub = self._stage_general_subchunk(
                    pos_m_g[c_start:c_end], spc_all[c_start:c_end],
                    atom_M_amps_chunk[c_start:c_end, :], use_origins,
                    _kernel_chunk, db_f0, f0_zero_lookup, anom_lookup,
                    (k_in_x, k_in_y, k_in_z))
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(sub["n"]),
                        sub["kx"], sub["ky"], sub["kz"],
                        sub["px"], sub["py"], sub["pz"],
                        sub["amp"],
                        sub["anom"], sub["f0p"], sub["f0z"],
                        xg, yg, zg,
                        dfield_gpu,
                        np.int32(Ny), np.int32(Nz),
                        remove_forward_flag,
                        apply_polarization_flag,
                        apply_decay_flag,
                        pol_rate,
                        apply_analyser_flag, analyser_kind_flag,
                        centre_x_f, centre_y_f, centre_z_f,
                        accept_angle_f, darwin_hw_f,
                        g_vecs_d,
                        apply_origin_flag,
                        sub["org"],
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()
                del sub

            # Free this chunk's GPU arrays.  Pool teardown stays out of the
            # sub-chunk loop; per launch it forced a full free/realloc cycle.
            del pos_m_g
            cp.get_default_memory_pool().free_all_blocks()

        final_result = dfield_gpu.reshape((Nz, Ny)).get()

        # Cleanup
        del dfield_gpu, xg, yg, zg
        del g_vecs_d
        cp.get_default_memory_pool().free_all_blocks()

        if offset is not None:
            final_result -= offset
        return final_result
    # -------------------------------------

    @staticmethod
    def _next_pow_two(n) -> int:
        """
        Smallest power of two that is >= n (1 for n < 1).
        """
        n_int = int(np.ceil(n))
        if n_int < 1:
            return 1
        return 1 << (n_int - 1).bit_length()

    @staticmethod
    def _choose_optimal_pad(
        Ny: int, Nz: int, dy: float, dz: float,
        wavelength: float, z: float,
        safety: float = 1.1,
        enforce_pow2: bool = True,
        min_pad_factor: float = 1.0,
    ):
        """
        Symmetric padding sizes for angular-spectrum propagation so nothing
        wraps around after distance |z|.

        Sampling supports sin(theta_max) = min(1, lambda / (2*d)) per axis; the
        half-padding is |z| * tan(theta_max), converted to pixels, scaled by
        ``safety`` and optionally rounded up to a power of two.

        Args:
            Ny, Nz (int): Original sizes along Y (horizontal) and Z (vertical).
            dy, dz (float): Pixel sizes in metres.
            wavelength (float): Wavelength in metres.
            z (float): Propagation distance in metres.
            safety (float): Multiplicative safety factor on the padding.
            enforce_pow2 (bool): Round padded sizes up to a power of two.
            min_pad_factor (float): Minimum growth factor applied to Ny and Nz
                regardless of geometry.

        Returns:
            tuple[int, int]: (Ny_pad, Nz_pad).
        """
        zabs = abs(float(z))
        if zabs == 0.0:
            Ny2 = max(int(np.ceil(Ny * min_pad_factor)), Ny)
            Nz2 = max(int(np.ceil(Nz * min_pad_factor)), Nz)
            if enforce_pow2:
                Ny2 = beam._next_pow_two(Ny2)
                Nz2 = beam._next_pow_two(Nz2)
            return int(Ny2), int(Nz2)

        # Sampling-limited maximum angles
        sry = min(1.0, float(wavelength) / (2.0 * float(dy)))
        srz = min(1.0, float(wavelength) / (2.0 * float(dz)))
        # Avoid tan(pi/2) by clamping
        sry = min(sry, 0.999999)
        srz = min(srz, 0.999999)

        # Cap tangent to prevent runaway padding when pixel_size < lambda/2
        tany = min(sry / np.sqrt(max(1e-18, 1.0 - sry * sry)), 2.0)
        tanz = min(srz / np.sqrt(max(1e-18, 1.0 - srz * srz)), 2.0)

        pad_y_m = safety * zabs * tany
        pad_z_m = safety * zabs * tanz

        pad_y_px = int(np.ceil(pad_y_m / float(dy)))
        pad_z_px = int(np.ceil(pad_z_m / float(dz)))

        Ny2 = Ny + 2 * pad_y_px
        Nz2 = Nz + 2 * pad_z_px

        Ny2 = max(Ny2, int(np.ceil(Ny * min_pad_factor)))
        Nz2 = max(Nz2, int(np.ceil(Nz * min_pad_factor)))

        # Cap padded dimensions to prevent GPU OOM from pow2 rounding
        max_padded_dim = 8192
        Ny2 = min(Ny2, max_padded_dim)
        Nz2 = min(Nz2, max_padded_dim)

        if enforce_pow2:
            Ny2 = beam._next_pow_two(Ny2)
            Nz2 = beam._next_pow_two(Nz2)

        return int(Ny2), int(Nz2)
    
    @staticmethod
    def build_propagation_multiplier_kernel():
        """
        Build a CUDA kernel that multiplies a spectrum in place by the exact
        free-space propagator with the beam's carrier subtracted:
            H = exp(+i*z*(sqrt(k^2 - kyt^2 - kzt^2) - k_g_axis))    [propagating]
            H = exp(-|z|*sqrt(kyt^2 + kzt^2 - k^2) - i*z*k_g_axis)  [evanescent]
        with (kyt, kzt) = (ky + k_g_perp_y, kz + k_g_perp_z).

        Forward beam: k_g_axis=k, k_g_perp=0, giving phase z*(sqrt(k^2-kt^2)-k).
        Bragg beam: k_g_axis=k_g.axis, k_g_perp=(k_g.e_y, k_g.e_z), which keeps
        the chi_h map's exp(-i*2*pi*G_h.r) carrier consistent across slices so
        E_g builds up coherently.

        Returns:
            cupy.RawKernel: Kernel named "prop_mul_kernel".
        """
        src = r'''
        #include <math.h>

        extern "C" __global__
        void prop_mul_kernel(
            const float* __restrict__ ky,   // length Ny   [rad/m]
            const float* __restrict__ kz,   // length Nz   [rad/m]
            const float  k,                 // 2*pi/lambda [rad/m]
            const float  z,                 // propagation [m]
            const int    Ny,
            const int    Nz,
            float2* __restrict__ F,         // spectrum (Nz*Ny), row-major
            const float  k_g_axis,          // beam k along propagation axis [rad/m]
            const float  k_g_perp_y,        // beam k along y of beam grid [rad/m]
            const float  k_g_perp_z)        // beam k along z of beam grid [rad/m]
        {
            int ix = blockIdx.x * blockDim.x + threadIdx.x;
            int iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= Ny || iy >= Nz) return;

            const int idx = iy * Ny + ix;

            const float kyv = ky[ix];
            const float kzv = kz[iy];

            // Exact propagator with the beam's own carrier subtracted: (ky, kz)
            // is the envelope's deviation from carrier k_g, so the total
            // transverse k is (ky + k_g_perp_y, kz + k_g_perp_z) and the axial
            // component is sqrt(k^2 - kt^2); subtracting k_g_axis leaves the
            // envelope phase.  Forward beam: reduces to (sqrt(k^2 - kt^2) - k) z.
            // Bragg beam: phase ~ (k - k_g_axis) z = -2*pi*G_h_axis*z, which
            // cancels chi_h's per-slice DC phase advance so E_g builds coherently.
            const float kyt = kyv + k_g_perp_y;
            const float kzt = kzv + k_g_perp_z;
            const float kt2 = kyt * kyt + kzt * kzt;
            const float kx2 = k * k - kt2;

            float phase, amp;
            if (kx2 >= 0.0f) {
                phase = z * (sqrtf(kx2) - k_g_axis);
                amp   = 1.0f;
            } else {
                amp   = expf(-fabsf(z) * sqrtf(-kx2));
                phase = -z * k_g_axis;
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

        # Strict compilers want ASCII-only source.
        src = src.encode('ascii', 'backslashreplace').decode('ascii')

        mod = cp.RawModule(
            code    = src,
            backend = 'nvcc',
            options = ('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true')
        )
        return mod.get_function('prop_mul_kernel')

    @staticmethod
    def compile_propagation_multiplier_cffi():
        """
        CPU (CFFI) propagation multiplier for angular-spectrum steps.  Multiplies
        a complex spectrum F (row-major Nz x Ny) in place by
            propagating:  H = exp(+i*z*sqrt(k^2 - kt^2))
            evanescent:   H = exp(-|z|*sqrt(kt^2 - k^2))  (real decay)

        Returns:
            tuple: (ffi, lib); call lib.prop_mul_cpu(Ny, Nz, ky, kz, k, z, F).
        """
        source = r'''
        #include <math.h>
        #include <complex.h>

        void prop_mul_cpu(
            const int      Ny,
            const int      Nz,
            const float*   ky,    /* rad/m, length Ny */
            const float*   kz,    /* rad/m, length Nz */
            const float    k,     /* 2*pi/lambda */
            const float    z,     /* meters */
            float _Complex* F)    /* spectrum (Nz*Ny), row-major */
        {
            const float az = fabsf(z);
            for (int iy = 0; iy < Nz; ++iy) {
                const float kzv = kz[iy];
                for (int ix = 0; ix < Ny; ++ix) {
                    const float kyv = ky[ix];
                    const float kt2 = kyv*kyv + kzv*kzv;
                    const float kx2 = k*k - kt2;

                    float phase, amp;
                    if (kx2 >= 0.0f) {
                        phase = z * sqrtf(kx2);
                        amp   = 1.0f;
                    } else {
                        phase = 0.0f;
                        amp   = expf(-az * sqrtf(-kx2));
                    }

                    const float cph = cosf(phase);
                    const float sph = sinf(phase);
                    const float _Complex H = amp * (cph + I*sph);

                    const int idx = iy * Ny + ix;
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
        CuPy bincount that tolerates empty inputs and drops non-finite or
        out-of-range indices.

        Args:
            idxs: Index array; None or empty returns zeros.
            weights: Weights summed per bin; None counts occurrences.
            size (int): Length of the output histogram.
            dtype: Output dtype; defaults to the weights dtype, or float32.

        Returns:
            cupy.ndarray: Histogram of length ``size``.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for _safe_bincount_gpu")

        if size <= 0:
            return cp.zeros((0,), dtype=cp.float32 if dtype is None else dtype)

        if idxs is None or int(getattr(idxs, "size", 0)) == 0:
            return cp.zeros((size,), dtype=cp.float32 if dtype is None else dtype)

        idxs = cp.asarray(idxs)

        # Drop non-finite indices
        m = cp.isfinite(idxs)
        if not bool(m.all()):
            idxs = idxs[m]
            if weights is not None:
                weights = cp.asarray(weights)[m]
        idxs = idxs.astype(cp.int64, copy=False)

        if idxs.size == 0:
            return cp.zeros((size,), dtype=cp.float32 if dtype is None else dtype)

        # Keep only indices in range [0, size)
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
    def precompute_depth_ein_all_chunks(
        self,
        sample,
        stage,
        use_gpu=True,
        ein_cache_dir=None,
        recompute_cache=False,
        kernel_radius=0,
        chunk_ids=None
    ):
        """
        Precompute per-atom entrance amplitudes Ein for each chunk and cache them.

        Each chunk is stage-transformed, sampled against the beam grid, and
        written to ``ein_chunk_{cid}_{hash}.npz`` (array "ein", shape (N_atoms,)).
        The hash covers beam, stage, grid, and the global depth window
        [s_min, s_max] along the beam, so an entry is only reused for an
        identical setup.

        Args:
            sample: Exposes chunk_total and load_chunk_positions(cid, use_gpu)
                -> (Ni, 3) Angstrom.
            stage: rotation (3x3) and translation (3,).
            use_gpu: Use the GPU path when CuPy is available.
            ein_cache_dir: Cache directory; defaults to <self.directory>/ein_cache.
            recompute_cache: Overwrite existing cache entries.
            kernel_radius: Gaussian blur radius in pixels for phi and tau when
                building A(u,v); 0 disables.
            chunk_ids: Chunk IDs to process; default is all chunks 1..chunk_total.

        Returns:
            tuple[str, str]: (cache_dir, cache_key_hash).

        Notes:
            The GPU path uses pinned memory and a ring of CUDA streams per GPU
            to overlap H2D, compute, D2H, and disk writes; the CPU path uses a
            thread pool for the NPZ writes.
        """
        import hashlib, threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        use_gpu = bool(use_gpu and (cp is not None))

        if sample.chunk_total is None or int(sample.chunk_total) == 0:
            raise ValueError("No chunks to precompute Ein for.")

        if chunk_ids is None:
            chunk_ids = list(range(1, int(sample.chunk_total) + 1))
        else:
            chunk_ids = list(chunk_ids)

        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # A(u,v) is computed once and shared by every chunk.
        if use_gpu:
            A_beam_np = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius=kernel_radius)
        else:
            A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=kernel_radius)

        # Cache key covers beam, stage, grid, and depth window.
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
            s_min=float(s_min),
            s_max=float(s_max)
        )
        key_hash = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode("utf-8")).hexdigest()
        cache_dir = ein_cache_dir or os.path.join(self.directory, "ein_cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Skip chunks already cached unless asked to recompute.
        to_do = []
        if recompute_cache:
            to_do = chunk_ids
        else:
            for cid in chunk_ids:
                p = os.path.join(cache_dir, f"ein_chunk_{cid}_{key_hash}.npz")
                if not os.path.isfile(p):
                    to_do.append(cid)

        if not to_do:
            return cache_dir, key_hash

        # CPU path: numpy Ein plus a small saver pool for the NPZ writes.
        if not use_gpu:
            tau_np = (-np.log(np.abs(A_beam_np) + np.float32(1e-20))).astype(np.float32)
            phi_np = np.angle(A_beam_np).astype(np.float32)
            E0_np  = self._beam_E0_map.astype(np.complex64)
            e1     = self._beam_e1.astype(np.float32)
            e2     = self._beam_e2.astype(np.float32)
            khat   = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
            du     = float(self._beam_du)
            dv     = float(self._beam_dv)
            uc     = float(self._beam_uc)
            vc     = float(self._beam_vc)
            R_np   = np.asarray(stage.rotation, dtype=np.float32)
            T_np   = np.asarray(stage.translation, dtype=np.float32)

            save_threads = int(os.getenv("BEAM_EIN_SAVE_THREADS", "2"))
            with ThreadPoolExecutor(max_workers=max(1, save_threads)) as saver:
                futures = []
                for cid in to_do:
                    cache_path = os.path.join(cache_dir, f"ein_chunk_{cid}_{key_hash}.npz")
                    pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
                    if pos.size == 0:
                        futures.append(saver.submit(np.savez_compressed, cache_path, ein=np.zeros((0,), np.complex64)))
                        continue
                    pos = pos @ R_np.T
                    pos = pos + T_np
                    ein_np = self._ein_bilinear_cpu(
                        pos_np=pos, tau=tau_np, phi=phi_np, E0=E0_np,
                        e1=e1, e2=e2, khat=khat, du=du, dv=dv, uc=uc, vc=vc,
                        s_min=s_min, s_max=s_max
                    ).astype(np.complex64)
                    futures.append(saver.submit(np.savez_compressed, cache_path, ein=ein_np))
                for f in as_completed(futures):
                    _ = f.result()
            return cache_dir, key_hash

        # ---------------- GPU streaming path ----------------

        # Host copies of the static maps; each device gets its own copy.
        tau_host = (-np.log(np.abs(A_beam_np) + np.float32(1e-20))).astype(np.float32)
        phi_host = np.angle(A_beam_np).astype(np.float32)
        E0_host  = self._beam_E0_map.astype(np.complex64)
        e1_host  = self._beam_e1.astype(np.float32)
        e2_host  = self._beam_e2.astype(np.float32)
        khat_host= (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        R_host   = np.asarray(stage.rotation, dtype=np.float32)
        T_host   = np.asarray(stage.translation, dtype=np.float32)

        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
        except Exception:
            n_gpus = 1
        n_gpus = max(1, n_gpus)
        streams_per_gpu = max(1, int(os.getenv("BEAM_EIN_STREAMS_PER_GPU", "4")))
        save_threads = max(1, int(os.getenv("BEAM_EIN_SAVE_THREADS", "6")))

        # Round-robin chunks across GPUs.
        shards = [[] for _ in range(n_gpus)]
        for i, cid in enumerate(to_do):
            shards[i % n_gpus].append(cid)

        # The saver holds the pinned buffer alive until the write has finished.
        def _save_npz_keepalive(path, arr_view, pinned_mem):
            try:
                np.savez_compressed(path, ein=np.asarray(arr_view, dtype=np.complex64))
            except Exception:
                np.savez(path, ein=np.asarray(arr_view, dtype=np.complex64))

        def gpu_worker(dev_id, my_chunks):
            if not my_chunks:
                return
            cp.cuda.Device(dev_id).use()

            tau_g = cp.asarray(tau_host)
            phi_g = cp.asarray(phi_host)
            E0_g  = cp.asarray(E0_host)
            e1g   = cp.asarray(e1_host)
            e2g   = cp.asarray(e2_host)
            khatg = cp.asarray(khat_host)
            Rg    = cp.asarray(R_host)
            Tg    = cp.asarray(T_host)

            if getattr(self, "_ein_kernel", None) is None:
                self._ein_kernel = self.build_ein_sampler_kernel()

            # Stream ring; each slot holds one in-flight chunk.
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(streams_per_gpu)]
            slot_event = [None] * streams_per_gpu
            slot_chunk = [None] * streams_per_gpu
            slot_devout= [None] * streams_per_gpu
            slot_host_mem = [None] * streams_per_gpu
            slot_host_view= [None] * streams_per_gpu

            from concurrent.futures import ThreadPoolExecutor, as_completed
            saver = ThreadPoolExecutor(max_workers=save_threads)
            save_futs = []

            # Wait for the slot's copy-back, then hand the buffer to the saver.
            def flush_slot(idx, cache_dir_local):
                ev = slot_event[idx]
                if ev is None:
                    return
                ev.synchronize()
                cid = slot_chunk[idx]
                path = os.path.join(cache_dir_local, f"ein_chunk_{cid}_{key_hash}.npz")
                hv = slot_host_view[idx]
                pm = slot_host_mem[idx]
                save_futs.append(saver.submit(_save_npz_keepalive, path, hv, pm))
                slot_event[idx] = None
                slot_chunk[idx] = None
                slot_devout[idx]= None
                slot_host_mem[idx] = None
                slot_host_view[idx]= None

            for n, cid in enumerate(my_chunks):
                s_id = n % streams_per_gpu
                st = streams[s_id]

                # Slot still in flight from an earlier chunk: drain it first.
                if slot_event[s_id] is not None:
                    flush_slot(s_id, cache_dir)

                pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
                if pos.size == 0:
                    empty_path = os.path.join(cache_dir, f"ein_chunk_{cid}_{key_hash}.npz")
                    try:
                        np.savez_compressed(empty_path, ein=np.zeros((0,), np.complex64))
                    except Exception:
                        np.savez(empty_path, ein=np.zeros((0,), np.complex64))
                    continue

                with st:
                    pos_g = cp.asarray(pos)
                    pos_g = pos_g @ Rg.T
                    pos_g += Tg

                    ein_g = self._ein_for_positions_gpu_fast(
                        pos_g=pos_g,
                        tau_g=tau_g, phi_g=phi_g, E0_g=E0_g,
                        e1g=e1g, e2g=e2g, khat_g=khatg,
                        s_min=np.float32(s_min), s_max=np.float32(s_max),
                        stream=st
                    )

                    # Async D2H into pinned memory; the event marks completion.
                    nbytes = int(ein_g.size) * 8  # complex64
                    pmem = cp.cuda.alloc_pinned_memory(nbytes)
                    h_view = np.frombuffer(pmem, dtype=np.complex64, count=ein_g.size)
                    cp.cuda.runtime.memcpyAsync(
                        int(pmem.ptr),
                        int(ein_g.data.ptr),
                        nbytes,
                        cp.cuda.runtime.memcpyDeviceToHost,
                        st.ptr
                    )
                    ev = cp.cuda.Event()
                    ev.record(st)

                # Slot state for flush_slot.
                slot_event[s_id] = ev
                slot_chunk[s_id] = cid
                slot_devout[s_id]= ein_g
                slot_host_mem[s_id] = pmem
                slot_host_view[s_id]= h_view

                del pos, pos_g

            # Drain the remaining slots and wait for every save.
            for s_id in range(streams_per_gpu):
                if slot_event[s_id] is not None:
                    flush_slot(s_id, cache_dir)
            for f in as_completed(save_futs):
                _ = f.result()
            saver.shutdown(wait=True)

            del tau_g, phi_g, E0_g, e1g, e2g, khatg, Rg, Tg
            for st in streams:
                st.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # One worker thread per GPU.
        threads = []
        for dev_id in range(n_gpus):
            t = threading.Thread(target=gpu_worker, args=(dev_id, shards[dev_id]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        cp.get_default_memory_pool().free_all_blocks()

        return cache_dir, key_hash
    
    # -------------------------------------------------------------------------
    # FP32 extended-precision helpers: NumPy mirrors of the kernel-side
    # ``two_prod_fma`` (Veltkamp-Dekker split).  They keep atom-table coordinate
    # conversion accurate in FP32 when positions reach sample scale (~10^7 A).
    # -------------------------------------------------------------------------
    @staticmethod
    def _two_prod_fp32(a, b):
        """
        Veltkamp-Dekker split of the FP32 product: returns ``(p, e)`` with
        ``a*b == p + e`` at FP32 precision. Vectorized; both outputs float32.
        """
        SPLIT = np.float32(4097.0)  # 2**12 + 1; valid for FP32 (23-bit mantissa)
        a32 = np.asarray(a, dtype=np.float32)
        b32 = np.asarray(b, dtype=np.float32)
        p = (a32 * b32).astype(np.float32)
        ca = (SPLIT * a32).astype(np.float32)
        a_hi = (ca - (ca - a32)).astype(np.float32)
        a_lo = (a32 - a_hi).astype(np.float32)
        cb = (SPLIT * b32).astype(np.float32)
        b_hi = (cb - (cb - b32)).astype(np.float32)
        b_lo = (b32 - b_hi).astype(np.float32)
        e = ((((a_hi * b_hi - p).astype(np.float32) + a_hi * b_lo).astype(np.float32)
              + a_lo * b_hi).astype(np.float32) + a_lo * b_lo).astype(np.float32)
        return p, e

    @staticmethod
    def _two_sum_fp32(a, b):
        """
        Knuth two-sum: returns ``(s, e)`` with ``a + b == s + e`` exactly in
        FP32. Vectorized; both outputs float32.
        """
        a32 = np.asarray(a, dtype=np.float32)
        b32 = np.asarray(b, dtype=np.float32)
        s = (a32 + b32).astype(np.float32)
        bb = (s - a32).astype(np.float32)
        e = ((a32 - (s - bb)).astype(np.float32)
             + (b32 - bb).astype(np.float32)).astype(np.float32)
        return s, e

    @staticmethod
    def _auto_voxel_grid(
        g_vectors_invA,
        supercell_extent_A,
        NN_dist_A,
        dz_slice_A,
        *,
        safety=0.9,
        strict_nyquist=True,
        force_unconverged=False,
    ):
        """
        Choose the coarsest voxel grid that still sits in the convergent regime.

        Bounds on dx (master plan I.6):
            dx > NN_dist / 3          (TSC stencil overlap)
            dx < 1 / (2 * |g_max|)    (chi_g Nyquist, general crystals)
        A window exists only if |g_max| * NN_dist < 3/2.

        Args:
            g_vectors_invA: (3,) vectors in 1/Angstrom (no 2pi), including g=0.
            supercell_extent_A: (Lx, Ly, Lz) in Angstrom.
            NN_dist_A: Nearest-neighbour distance in Angstrom.
            dz_slice_A: Requested z-slice thickness in Angstrom.
            safety: Factor below the Nyquist upper bound in strict mode.
            strict_nyquist: True for general/defected crystals; False allows
                the aggressive spacing for commensurate perfect supercells.
            force_unconverged: Warn instead of raising when the window has
                collapsed, and pick dx between the inverted bounds.

        Returns:
            tuple: ``(Nx, Ny, Nz_slices, dx_A, dy_A, dz_A, diagnostics_dict)``.

        Raises:
            RuntimeError: Window collapsed in strict mode without
                ``force_unconverged``.
        """
        import math
        import warnings

        # |g_max| over (g_m - g_n) pairs
        gv = [np.asarray(g, dtype=np.float64) for g in g_vectors_invA]
        pair_mags = []
        for i, gi in enumerate(gv):
            for j, gj in enumerate(gv):
                if i != j:
                    pair_mags.append(float(np.linalg.norm(gi - gj)))
        g_max = max(pair_mags) if pair_mags else 1e-9

        dx_nyquist_upper = 1.0 / (2.0 * g_max)
        dx_tsc_lower = float(NN_dist_A) / 3.0

        window_collapsed = (dx_tsc_lower > dx_nyquist_upper)
        if window_collapsed and strict_nyquist and not force_unconverged:
            raise RuntimeError(
                f"_auto_voxel_grid: convergent window has collapsed for the "
                f"requested beam set + material. |g_max| = {g_max:.4f} 1/A, "
                f"NN_dist = {NN_dist_A:.4f} A, product = "
                f"{g_max * NN_dist_A:.3f} (must be < 1.5).\n"
                f"  Lower bound (TSC overlap):    dx > {dx_tsc_lower:.4f} A\n"
                f"  Upper bound (chi_g Nyquist):  dx < {dx_nyquist_upper:.4f} A\n"
                f"Suggested actions: (a) restrict to a lower-order reflection "
                f"such that |g| < {1.5/NN_dist_A:.4f} 1/A; (b) use a denser "
                f"material with NN_dist < {1.5/g_max:.4f} A; or (c) pass "
                f"force_unconverged=True to proceed at reduced accuracy "
                f"(rel_err typically 30-40% for h-coupling reflections)."
            )

        if strict_nyquist and not window_collapsed:
            dx_target = dx_nyquist_upper * float(safety)
            if dx_target < dx_tsc_lower:
                dx_target = dx_tsc_lower * 1.05
        elif window_collapsed:
            dx_target = 0.5 * (dx_tsc_lower + dx_nyquist_upper)
            warnings.warn(
                f"_auto_voxel_grid: proceeding with collapsed window at user "
                f"request. dx={dx_target:.4f} A is outside both bounds "
                f"[{dx_tsc_lower:.4f}, {dx_nyquist_upper:.4f}]. "
                f"Result will not be quantitatively converged.")
        else:
            # Aggressive mode for commensurate perfect supercells.
            dx_target = float(NN_dist_A) * float(safety) * 0.5

        Lx, Ly, Lz = (float(v) for v in supercell_extent_A)
        Nx = max(1, int(math.ceil(Lx / dx_target)))
        Ny = max(1, int(math.ceil(Ly / dx_target)))
        Nz_slices = max(1, int(math.ceil(Lz / float(dz_slice_A))))
        dx = Lx / Nx
        dy = Ly / Ny

        diagnostics = {
            "g_max_invA": float(g_max),
            "NN_dist_A": float(NN_dist_A),
            "dx_nyquist_upper": float(dx_nyquist_upper),
            "dx_tsc_lower": float(dx_tsc_lower),
            "dx_target": float(dx_target),
            "dx_achieved": float(dx),
            "Nx": int(Nx), "Ny": int(Ny), "Nz_slices": int(Nz_slices),
            "in_convergent_regime": bool(
                (1.0 / (2.0 * dx)) > g_max and 3.0 * dx >= float(NN_dist_A)),
            "window_collapsed": bool(window_collapsed),
            "mode": "strict" if strict_nyquist else "aggressive",
        }
        return Nx, Ny, Nz_slices, dx, dy, float(dz_slice_A), diagnostics

    def _estimate_nearest_neighbor_distance(self, sample, n_samples=2048):
        """
        Estimate the nearest-neighbour distance of a chunked sample.

        Pick a couple of anchor atoms, gather every atom inside a small
        axis-aligned box around each, and brute-force the pairwise minimum on
        that local set. The box guarantees the anchor's true neighbours are
        present no matter how atoms are ordered in the chunk.

        Uniform random subsampling does not work here: for N ~ 10^8 atoms a
        2048-atom subset almost never contains a true NN pair, so its minimum
        is an accidental near-miss and can be 3-10x too large. The box approach
        is one O(N) mask plus an O(M^2) brute force over ~10^3 atoms.

        Args:
            sample: Chunked sample with chunk_total and load_chunk_positions(cid).
            n_samples: Unused; kept for backward compatibility.

        Returns:
            float NN distance in Angstrom, or None if estimation fails.
        """
        try:
            chunk_total = int(sample.chunk_total or 0)
            if chunk_total == 0:
                return None
            try:
                first_chunk = sample.load_chunk_positions(1, use_gpu=False)
            except Exception:
                return None
            positions = np.asarray(first_chunk, dtype=np.float32)
            N = int(positions.shape[0])
            if N < 2:
                return None

            # A 15 A half-edge holds ~10^3 atoms in any crystal (NN spans
            # ~0.7-5 A), so the local brute force has constant cost.
            R_box_A = 15.0
            # A second anchor covers the case where the first lands on a
            # surface atom or defect site.
            n_anchors = 2
            best_d2 = np.inf
            rng = np.random.default_rng(seed=0xBEEF)
            anchor_idx = rng.choice(N, size=min(n_anchors, N), replace=False)
            for ai in anchor_idx:
                anchor = positions[int(ai)]
                mask = (
                    (np.abs(positions[:, 0] - anchor[0]) < R_box_A) &
                    (np.abs(positions[:, 1] - anchor[1]) < R_box_A) &
                    (np.abs(positions[:, 2] - anchor[2]) < R_box_A)
                )
                local = positions[mask]
                M = int(local.shape[0])
                if M < 2:
                    # Isolated anchor; try the next one.
                    continue

                # Sparse box (surface atom, unusual crystal): widen it.
                if M < 8:
                    mask2 = (
                        (np.abs(positions[:, 0] - anchor[0]) < 4 * R_box_A) &
                        (np.abs(positions[:, 1] - anchor[1]) < 4 * R_box_A) &
                        (np.abs(positions[:, 2] - anchor[2]) < 4 * R_box_A)
                    )
                    local = positions[mask2]
                    M = int(local.shape[0])
                    if M < 2:
                        continue

                # Brute-force pairwise minimum; M ~ 10^3, so O(M^2) is cheap.
                if M <= 4096:
                    d2_mat = np.sum(
                        (local[:, None, :] - local[None, :, :]) ** 2, axis=-1)
                    np.fill_diagonal(d2_mat, np.inf)
                    d2_min = float(np.min(d2_mat))
                else:
                    # Unexpectedly large local set: KD-tree if scipy is
                    # present, else blocked pairwise.
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(local)
                        d, _ = tree.query(local, k=2)
                        d_nn = d[:, 1]
                        d_nn = d_nn[np.isfinite(d_nn) & (d_nn > 0.0)]
                        d2_min = float(d_nn.min() ** 2) if d_nn.size else np.inf
                    except ImportError:
                        d2_min = np.inf
                        block = 256
                        for i in range(0, M, block):
                            Si = local[i:i + block]
                            d2 = np.sum(
                                (Si[:, None, :] - local[None, :, :]) ** 2, axis=-1)
                            upper = min(block, M - i)
                            for r in range(upper):
                                d2[r, i + r] = np.inf
                            db = float(np.min(d2))
                            if db < d2_min:
                                d2_min = db

                if d2_min < best_d2 and d2_min > 0.0:
                    best_d2 = d2_min

            if not np.isfinite(best_d2) or best_d2 <= 0.0:
                return None
            return float(np.sqrt(best_d2))
        except Exception:
            return None

    def _compute_global_depth_bounds(self, sample, stage):
        """
        Front-to-back extent of the sample along the beam direction, in Angstrom.

        Each chunk is stage-transformed and projected onto the unit beam
        direction; the global min and max of that projection are returned.

        Args:
            sample: Exposes chunk_total and load_chunk_positions(cid, use_gpu=False).
            stage: rotation (3x3) and translation (3,).

        Returns:
            tuple[float, float]: (s_min, s_max) in Angstrom, or (0.0, 1.0) when
            the bounds are degenerate.
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
            pos = pos @ R.T
            pos += T
            s_vals = pos @ k_hat
            cur_min = np.min(s_vals)
            cur_max = np.max(s_vals)
            if cur_min < s_min: s_min = cur_min
            if cur_max > s_max: s_max = cur_max

        if not np.isfinite(s_min) or (s_max <= s_min):
            return 0.0, 1.0
        return float(s_min), float(s_max)

    def cpu_scatter_chunk_cffi(self, complied_code, ffi_obj, chunk_id, sample,
                            Ny, Nz, coords_x_m, coords_y_m, coords_z_m,
                            db_dict_f0_all, db_dict_f1f2_all, k_val,
                            stage, detector=None, remove_forward_component=False,
                            initial_amp_complex=None,
                            apply_polarization=False,
                            apply_spherical_decay=True):
        """
        Kinematic scattering for one chunk on the CPU via the CFFI kernel.

        Builds the per-atom tables (f0 parameters, f0(0), anomalous f'+if'',
        entrance amplitudes), applies the stage transform, converts positions to
        metres, and calls the compiled C routine to accumulate the complex field.

        Args:
            complied_code: CFFI module exposing compute_scattering_cffi.
            ffi_obj: CFFI FFI object for pointer casts.
            chunk_id (int): Chunk to process.
            sample: Provides load_chunk_species / load_chunk_positions.
            Ny, Nz (int): Detector width and height in pixels.
            coords_x_m, coords_y_m, coords_z_m: Flattened detector pixel
                coordinates in metres, length Ny*Nz.
            db_dict_f0_all (dict): element -> f0 parameters (11,).
            db_dict_f1f2_all (dict): element -> table of [E, f1, f2].
            k_val (float): Wave number 2*pi/lambda in rad/m.
            stage: rotation (3x3) and translation (3,).
            detector: Unused; kept for API parity.
            remove_forward_component (bool): Subtract f0(0) from f0(Q).
            initial_amp_complex: Per-atom entrance amplitudes (complex64), or
                None for ones.
            apply_polarization (bool): Apply the polarization factor in the
                kernel using self._pol_perp_rate.
            apply_spherical_decay (bool): Apply relative 1/R scaling.

        Returns:
            np.ndarray: complex64 field of shape (Nz, Ny).
        """
        species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
        atom_count = int(species_chunk_np.shape[0])
        if atom_count == 0:
            return np.zeros((Nz, Ny), dtype=np.complex64)

        scattering_anom_np_real = np.zeros(atom_count, dtype=np.float32)
        scattering_anom_np_imag = np.zeros(atom_count, dtype=np.float32)
        f0_params_np            = np.zeros((atom_count, 11), dtype=np.float32)
        f0_zero_np              = np.zeros((atom_count,), dtype=np.float32)

        # Fill the per-atom tables element by element.
        f0_zero_dict = self._build_f0_zero_dict(db_dict_f0_all)
        unique_elements = pd.unique(species_chunk_np)
        for el in unique_elements:
            el = str(el)
            if el not in db_dict_f0_all:
                continue
            mask = (species_chunk_np == el)
            # Anomalous term at this beam energy.
            table = db_dict_f1f2_all.get(el, None)
            if table is not None:
                cplx = self.get_f1f2_from_params(self._energy, table)
                scattering_anom_np_real[mask] = float(cplx.real)
                scattering_anom_np_imag[mask] = float(cplx.imag)
            f0_params_np[mask] = db_dict_f0_all[el]
            f0_zero_np[mask]   = float(f0_zero_dict.get(el, 0.0))

        # Stage transform, then metres for the C kernel.
        positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
        positions_chunk = positions_chunk @ stage.rotation.T
        positions_chunk += stage.translation
        positions_chunk_m = positions_chunk / 1e10

        if initial_amp_complex is None:
            amp_r = np.ones((atom_count,), dtype=np.float32)
            amp_i = np.zeros((atom_count,), dtype=np.float32)
        else:
            amp_r = np.asarray(np.real(initial_amp_complex), dtype=np.float32, order='C')
            amp_i = np.asarray(np.imag(initial_amp_complex), dtype=np.float32, order='C')
            if amp_r.shape[0] != atom_count:
                raise ValueError(f"initial_amp_complex size mismatch for chunk {chunk_id}")

        out_r = np.zeros(Ny*Nz, dtype=np.float32)
        out_i = np.zeros(Ny*Nz, dtype=np.float32)

        # C needs contiguous buffers.
        positions_chunk_m = np.ascontiguousarray(positions_chunk_m)
        f0_params_np      = np.ascontiguousarray(f0_params_np)
        f0_zero_np        = np.ascontiguousarray(f0_zero_np)
        s_anom_r          = np.ascontiguousarray(scattering_anom_np_real)
        s_anom_i          = np.ascontiguousarray(scattering_anom_np_imag)
        amp_r             = np.ascontiguousarray(amp_r)
        amp_i             = np.ascontiguousarray(amp_i)

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
            Ny, Nz,
            coords_x_ptr, coords_y_ptr, coords_z_ptr,
            k_val,
            int(1 if apply_polarization else 0),
            float(self._pol_perp_rate),
            int(1 if apply_spherical_decay else 0),
            out_r_ptr, out_i_ptr
        )

        return (out_r + 1j*out_i).reshape((Nz, Ny)).astype(np.complex64)
    
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
        recompute_cache=False,
        apply_polarization=False,
        apply_spherical_decay=True
    ):
        """
        Kinematic scattering on the CPU; returns the complex detector field.

        Chunks run in a thread pool. Each chunk gets per-atom scattering tables
        and entrance amplitudes (cached depth-dependent Ein, or E0 sampled on
        the beam grid), then the CFFI kernel accumulates its field.

        Args:
            sample: Exposes chunk_total, load_chunk_species(cid, use_gpu=False)
                -> (Ni,), and load_chunk_positions(cid, use_gpu=False)
                -> (Ni, 3) Angstrom.
            measurement_positions: (3, Ny*Nz) detector pixel positions in
                Angstrom (NumPy or CuPy).
            measurement_shape (tuple[int, int]): (Ny, Nz).
            stage: rotation (3x3) and translation (3,).
            detector: Unused; kept for parity with the GPU path.
            remove_forward_component (bool): Subtract f0(0) in the kernel. Keep
                this consistent with the transmission step so the forward term
                is not counted twice.
            use_depth_ein (bool): Use cached per-atom Ein; missing entries are
                precomputed first.
            ein_cache_dir: Directory for Ein cache files.
            recompute_cache (bool): Recompute Ein even if cached.
            apply_polarization (bool): Apply the polarization factor in the kernel.
            apply_spherical_decay (bool): Apply relative 1/R scaling in the kernel.

        Returns:
            np.ndarray: complex64 field of shape (Nz, Ny).
        """
        import hashlib
        Ny, Nz = measurement_shape

        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        # Wave number in rad/m.
        k_val = np.float32(2.0 * np.pi / self._wavelength)

        # Detector coordinates on the host, in metres.
        if cp is not None and isinstance(measurement_positions, cp.ndarray):
            measurement_positions = measurement_positions.get()
        coords_x_m = np.ascontiguousarray(measurement_positions[0, :].astype(np.float32) / 1e10)
        coords_y_m = np.ascontiguousarray(measurement_positions[1, :].astype(np.float32) / 1e10)
        coords_z_m = np.ascontiguousarray(measurement_positions[2, :].astype(np.float32) / 1e10)

        chunk_total = int(sample.chunk_total or 0)
        if chunk_total == 0:
            return np.zeros((Nz, Ny), dtype=np.complex64)

        # Depth window along the beam for Ein/E0 sampling.
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Fill in any missing Ein cache entries.
        key_hash = None
        cache_dir = None
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
                s_min=float(s_min),
                s_max=float(s_max)
            )
            key_hash = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode('utf-8')).hexdigest()
            cache_dir = ein_cache_dir or os.path.join(self.directory, "ein_cache")
            os.makedirs(cache_dir, exist_ok=True)

            missing = []
            for cid in range(1, chunk_total + 1):
                p = os.path.join(cache_dir, f"ein_chunk_{cid}_{key_hash}.npz")
                if recompute_cache or (not os.path.isfile(p)):
                    missing.append(cid)
            if missing:
                self.precompute_depth_ein_all_chunks(
                    sample, stage,
                    use_gpu=(cp is not None),
                    ein_cache_dir=cache_dir,
                    recompute_cache=recompute_cache,
                    kernel_radius=0,
                    chunk_ids=missing
                )

        ffi_obj, complied_code = self.compile_compute_scattering_cffi()

        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_threads = min(chunk_total, multiprocessing.cpu_count())

        # Zero tau/phi maps give plain E0 sampling when depth Ein is off.
        tau_zero = np.zeros((self._beam_Ny, self._beam_Nz), dtype=np.float32)
        phi_zero = np.zeros_like(tau_zero)
        E0_np    = self._beam_E0_map.astype(np.complex64)
        e1       = self._beam_e1.astype(np.float32)
        e2       = self._beam_e2.astype(np.float32)
        khat     = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        du       = float(self._beam_du)
        dv       = float(self._beam_dv)
        uc       = float(self._beam_uc)
        vc       = float(self._beam_vc)

        def worker(chunk_id):
            species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
            atom_count = int(species_chunk_np.shape[0])
            if atom_count == 0:
                return np.zeros((Nz, Ny), dtype=np.complex64)

            # Per-atom f0 parameters, f0(0), and anomalous terms.
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
                table = db_dict_f1f2_all.get(el, None)
                if table is not None:
                    cplx = self.get_f1f2_from_params(self._energy, table)
                    scattering_anom_np_real[mask] = float(cplx.real)
                    scattering_anom_np_imag[mask] = float(cplx.imag)
                f0_params_np[mask] = db_dict_f0_all[el]
                f0_zero_np[mask]   = float(f0_zero_dict.get(el, 0.0))

            # Stage transform, then metres for the C kernel.
            positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
            positions_chunk = positions_chunk @ stage.rotation.T
            positions_chunk += stage.translation
            positions_chunk_m = positions_chunk / 1e10

            # Entrance amplitudes: cached Ein or E0-only sampling.
            if use_depth_ein:
                cache_path = os.path.join(cache_dir, f"ein_chunk_{chunk_id}_{key_hash}.npz")
                with np.load(cache_path) as npz:
                    init_amp = npz["ein"].astype(np.complex64)
            else:
                init_amp = self._ein_bilinear_cpu(
                    pos_np=positions_chunk,
                    tau=tau_zero,
                    phi=phi_zero,
                    E0=E0_np,
                    e1=e1, e2=e2, khat=khat,
                    du=du, dv=dv, uc=uc, vc=vc,
                    s_min=s_min, s_max=s_max
                ).astype(np.complex64)

            out = self.cpu_scatter_chunk_cffi(
                complied_code, ffi_obj, chunk_id, sample, Ny, Nz,
                coords_x_m, coords_y_m, coords_z_m,
                db_dict_f0_all, db_dict_f1f2_all, k_val, stage,
                detector=None, remove_forward_component=remove_forward_component,
                initial_amp_complex=init_amp,
                apply_polarization=apply_polarization,
                apply_spherical_decay=apply_spherical_decay
            )
            return out

        final_result = np.zeros((Nz, Ny), dtype=np.complex64)
        with ThreadPoolExecutor(max_workers=n_threads) as exe:
            futures = {exe.submit(worker, cid): cid for cid in range(1, chunk_total + 1)}
            for fut in as_completed(futures):
                final_result += fut.result()
        return final_result

    def _ein_for_positions_gpu_fast(
        self,
        pos_g,            # (N,3) float32, Angstrom, on device
        tau_g,            # (NyB,NzB) float32, on device
        phi_g,            # (NyB,NzB) float32, on device
        E0_g,             # (NyB,NzB) complex64, on device
        e1g, e2g, khat_g, # (3,) float32, on device
        s_min, s_max,     # float32, Angstrom
        stream=None
    ):
        """
        Ein for a set of atom positions on the GPU via the fused bilinear kernel.

        Each atom is projected onto beam coordinates (u, v); tau, phi and E0 are
        bilinearly sampled on the beam grid; then
            Ein = E0 * exp(-f * tau) * exp(i * f * phi)
        with f the depth fraction along khat between s_min and s_max.

        Every array must already live on the current device; nothing is copied
        here. Uses self._ein_kernel, compiling it on first use.

        Returns:
            cupy.ndarray: (N,) complex64 Ein per atom, on device.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for _ein_for_positions_gpu_fast")

        kernel = getattr(self, "_ein_kernel", None)
        if kernel is None:
            kernel = self.build_ein_sampler_kernel()
            self._ein_kernel = kernel
        N = int(pos_g.shape[0])
        out = cp.zeros((N,), dtype=cp.complex64)

        threads = 256
        blocks = (N + threads - 1) // threads

        # Reciprocals so the kernel avoids divisions.
        inv_du = cp.float32(1.0 / float(self._beam_du))
        inv_dv = cp.float32(1.0 / float(self._beam_dv))

        NyB = int(self._beam_Ny)
        NzB = int(self._beam_Nz)
        uc = cp.float32(self._beam_uc)
        vc = cp.float32(self._beam_vc)

        # Order must match the kernel signature.
        args = (
            pos_g.astype(cp.float32, copy=False).ravel(),
            np.int32(N),
            tau_g.astype(cp.float32, copy=False).ravel(),
            phi_g.astype(cp.float32, copy=False).ravel(),
            E0_g.view(cp.float32).ravel(),  # pass as float2 underlying storage
            np.int32(NyB),
            np.int32(NzB),
            inv_du,
            inv_dv,
            uc,
            vc,
            e1g.astype(cp.float32, copy=False),
            e2g.astype(cp.float32, copy=False),
            khat_g.astype(cp.float32, copy=False),
            cp.float32(s_min),
            cp.float32(s_max),
            out.ravel()
        )

        kernel((blocks,), (threads,), args, stream=stream)
        return out
    
    def interact_beam_gpu(
        self,
        sample,
        measurement_positions,
        measurement_shape,
        stage,
        remove_forward: bool = False,
        use_depth_ein: bool = False,
        ein_cache_dir: str | None = None,
        recompute_cache: bool = False,
        apply_polarization: bool = False,
        spherical_decay: bool = True,
        analyser_mode: str | int = "off",
        analyser_acceptance_angle_rad: float = 0.0,
        analyser_darwin_halfwidth_rad: float = 0.0
    ):
        """
        Kinematic scattering on the GPU(s) with an optional analyser. Chunks are
        split across devices and streamed through the scatter kernel; falls back
        to interact_beam_cpu when CuPy or a GPU is missing. analyser_mode is
        "off" | "top_hat" | "darwin" or 0 | 1 | 2 (GPU only).
        """
        _analyser_off = (
            analyser_mode is None
            or (isinstance(analyser_mode, str) and analyser_mode.strip().lower() in ("off", "none", "disabled"))
            or (not isinstance(analyser_mode, str) and analyser_mode == 0)
        )
        if cp is None:
            self._log("normal", "[beam] CuPy not installed, falling back to CPU.")
            if not _analyser_off:
                warnings.warn("analyser acceptance is GPU-only; ignored on CPU path")
            return self.interact_beam_cpu(
                sample,
                measurement_positions,
                measurement_shape,
                stage,
                detector=None,
                remove_forward_component=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache,
                apply_polarization=apply_polarization,
                apply_spherical_decay=spherical_decay
            )

        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
        except Exception:
            n_gpus = 0
        if n_gpus < 1:
            self._log("normal", "[beam] No GPUs found, falling back to CPU.")
            if not _analyser_off:
                warnings.warn("analyser acceptance is GPU-only; ignored on CPU path")
            return self.interact_beam_cpu(
                sample,
                measurement_positions,
                measurement_shape,
                stage,
                detector=None,
                remove_forward_component=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache,
                apply_polarization=apply_polarization,
                apply_spherical_decay=spherical_decay
            )

        import hashlib
        self._log("normal", f"[beam] Found {n_gpus} GPU(s).")

        class _TmpDet:
            def __init__(self, pix): self.pixel_coordinates = pix
        _det_for_mode = _TmpDet(measurement_positions)

        if not hasattr(self, "_phase_tol_rad"):
            self._phase_tol_rad = 1e-3
        self._select_series_mode_once(sample, _det_for_mode, safety_t_thresh=0.5, verbose=True)

        interaction_kernel = self.build_interaction_kernel(
            series_terms=self._series_terms,
            force_mode=("series" if self._global_use_series else "exact")
        )

        db_f0   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero = self._build_f0_zero_dict(db_f0)
        self._require_forward_beam()

        Ny, Nz = measurement_shape
        final_result = np.zeros((Nz, Ny), dtype=np.complex64)

        # Detector pixel coordinates (metres) in pinned memory.
        x_coords = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        y_coords = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        z_coords = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        # Detector centre (metres) for the analyser.
        ciy = int(Ny // 2)
        ciz = int(Nz // 2)
        cidx = ciz * Ny + ciy
        centre_x = float(x_coords[cidx]); centre_y = float(y_coords[cidx]); centre_z = float(z_coords[cidx])

        # Analyser mode -> kernel kind code.
        if isinstance(analyser_mode, str):
            m = analyser_mode.strip().lower()
            if m in ("off", "none", "disabled"): analyser_kind = 0
            elif m in ("top_hat", "tophat", "top-hat", "top"): analyser_kind = 1
            elif m in ("darwin", "rolloff", "roll-off"): analyser_kind = 2
            else: analyser_kind = 0
        else:
            analyser_kind = int(analyser_mode)
            if analyser_kind not in (0,1,2): analyser_kind = 0
        apply_analyser = 1 if analyser_kind != 0 else 0

        chunk_total = int(sample.chunk_total or 0)
        self._log("normal", f"[beam] Total of {chunk_total} chunk(s) to process.")
        if chunk_total == 0:
            return np.zeros((Nz, Ny), dtype=np.complex64)

        # Depth window along the beam for Ein/E0 sampling.
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Fill in any missing Ein cache entries.
        key_hash = None
        cache_dir = None
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
                s_min=float(s_min),
                s_max=float(s_max)
            )
            key_hash = hashlib.sha1(json.dumps(key_obj, sort_keys=True).encode('utf-8')).hexdigest()
            cache_dir = ein_cache_dir or os.path.join(self.directory, "ein_cache")
            os.makedirs(cache_dir, exist_ok=True)

            missing = []
            for cid in range(1, chunk_total + 1):
                p = os.path.join(cache_dir, f"ein_chunk_{cid}_{key_hash}.npz")
                if recompute_cache or (not os.path.isfile(p)):
                    missing.append(cid)
            if missing:
                self._log("verbose", f"[beam] Precomputing Ein for {len(missing)} chunk(s).")
                self.precompute_depth_ein_all_chunks(
                    sample, stage,
                    use_gpu=True,
                    ein_cache_dir=cache_dir,
                    recompute_cache=recompute_cache,
                    kernel_radius=0,
                    chunk_ids=missing
                )

        # Split chunks across devices.
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus
        partial_results = [None] * n_gpus

        try:
            _streams_per_gpu = max(1, int(os.getenv("BEAM_STREAMS_PER_GPU", "3")))
        except Exception:
            _streams_per_gpu = 3

        # The Ein kernel also does E0-only sampling.
        if getattr(self, "_ein_kernel", None) is None:
            self._ein_kernel = self.build_ein_sampler_kernel()

        def gpu_worker(gpu_id, chunk_indices, result_index):
            cp.cuda.Device(gpu_id).use()

            Rg = cp.asarray(R_pin, dtype=cp.float32)
            Tg = cp.asarray(T_pin, dtype=cp.float32)
            xg = cp.asarray(x_coords); yg = cp.asarray(y_coords); zg = cp.asarray(z_coords)
            # Single forward channel with g = 0: the kernel's M-channel loop
            # collapses to sincos(0) = (0, 1), bit-identical to the
            # single-amplitude path.
            g_vecs_d = cp.zeros(3, dtype=cp.float32)
            _org_dummy_d = cp.zeros(3, dtype=cp.float32)

            # Zero tau/phi maps give plain E0 sampling on this device.
            E0_g  = cp.asarray(self._beam_E0_map.astype(np.complex64))
            tau_g = cp.zeros(E0_g.shape, dtype=cp.float32)
            phi_g = cp.zeros_like(tau_g)
            e1g   = cp.asarray(self._beam_e1.astype(np.float32))
            e2g   = cp.asarray(self._beam_e2.astype(np.float32))
            khatg = cp.asarray((self._direction / np.linalg.norm(self._direction)).astype(np.float32))

            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(_streams_per_gpu)]
            dfields = [cp.zeros((Ny * Nz,), dtype=cp.complex64) for _ in streams]

            block = (32, 16)
            grid  = ((Ny + block[0] - 1) // block[0],
                    (Nz + block[1] - 1) // block[1])

            for i, cidx in enumerate(chunk_indices):
                s_id = i % len(streams)
                streams[s_id].synchronize()

                spc = sample.load_chunk_species(cidx, use_gpu=False)
                nA = int(spc.shape[0])
                if nA == 0:
                    continue
                # Species-file identity lets the fast path reuse its species
                # codes across the steps of a scan.
                _spc_file = os.path.join(str(getattr(sample, "directory", "")),
                                         f"atomic_species_{cidx}.npy")
                try:
                    _st = os.stat(_spc_file)
                    species_key = (_spc_file, int(_st.st_mtime_ns), int(_st.st_size), nA)
                except OSError:
                    species_key = None

                _anom_lu = {}                     # element -> f' + i f''
                for el in pd.unique(spc):
                    tbl = db_f1f2.get(el) if el in db_f0 else None
                    if tbl is not None:
                        _anom_lu[str(el)] = complex(
                            self.get_f1f2_from_params(self._energy, tbl))

                with streams[s_id]:
                    # Positions to device, then the stage transform.
                    pos = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                    pos = pos @ Rg.T
                    pos += Tg

                    if use_depth_ein:
                        cache_path = os.path.join(cache_dir, f"ein_chunk_{cidx}_{key_hash}.npz")
                        with np.load(cache_path) as npz:
                            arr = npz["ein"]
                        initial_amp = cp.asarray(arr.astype(np.complex64))
                    else:
                        # E0-only sampling (tau=0, phi=0), zero outside beam grid
                        initial_amp = self._ein_for_positions_gpu_fast(
                            pos_g=pos,
                            tau_g=tau_g, phi_g=phi_g, E0_g=E0_g,
                            e1g=e1g, e2g=e2g, khat_g=khatg,
                            s_min=np.float32(s_min), s_max=np.float32(s_max),
                            stream=streams[s_id]
                        )

                    # Try the fast path before any per-atom staging: it carries
                    # its species tables as launch constants, so the host
                    # form-factor fill and per-atom uploads below would be
                    # wasted work.  It also uses per-chunk local origins.
                    pos_m = (pos / 1e10).astype(cp.float32)
                    del pos
                    # Both paths stream the chunk in sub-chunks sized to free
                    # memory: a sample chunk is bounded only by chunk_volume
                    # and can be far too large to hand over whole.  Resident
                    # meanwhile: pos_m plus the incident field, 20 B/atom.
                    sub_n = self._scatter_subchunk_size(1, resident_bytes=20 * nA)
                    if self._fast_scatter(
                            pos_m, initial_amp, spc,
                            db_f0, f0_zero, _anom_lu, xg, yg, zg,
                            dfields[s_id], Ny, Nz,
                            float(abs(self._kx_scalar)), 1,
                            int(analyser_kind) if apply_analyser else 0,
                            remove_forward, apply_polarization,
                            spherical_decay, float(self._pol_perp_rate), sub_n,
                            species_key=species_key):
                        del pos_m, initial_amp
                        continue

                    # General kernel, staged per sub-chunk, with a single
                    # global origin (use_origins=False).
                    spc_np = np.asarray(spc)
                    for c0 in range(0, nA, sub_n):
                        c1 = min(c0 + sub_n, nA)
                        sub = self._stage_general_subchunk(
                            pos_m[c0:c1], spc_np[c0:c1], initial_amp[c0:c1],
                            False, 128, db_f0, f0_zero, _anom_lu,
                            (self._kx_scalar, self._ky_scalar, self._kz_scalar))
                        interaction_kernel(
                            grid, block,
                            (
                                np.int32(sub["n"]),
                                sub["kx"], sub["ky"], sub["kz"],
                                sub["px"], sub["py"], sub["pz"],
                                sub["amp"],
                                sub["anom"], sub["f0p"], sub["f0z"],
                                xg, yg, zg,
                                dfields[s_id],
                                np.int32(Ny),
                                np.int32(Nz),
                                np.int32(1 if remove_forward else 0),
                                np.int32(1 if apply_polarization else 0),
                                np.int32(1 if spherical_decay else 0),
                                np.float32(self._pol_perp_rate),
                                np.int32(1 if apply_analyser else 0),
                                np.int32(int(analyser_kind)),
                                np.float32(centre_x), np.float32(centre_y), np.float32(centre_z),
                                np.float32(analyser_acceptance_angle_rad),
                                np.float32(analyser_darwin_halfwidth_rad),
                                g_vecs_d,
                                np.int32(0),
                                _org_dummy_d,
                            ),
                            stream=streams[s_id]
                        )
                        del sub
                    del pos_m, initial_amp

            for st in streams:
                st.synchronize()

            dfield_total = dfields[0]
            for j in range(1, len(dfields)):
                dfield_total += dfields[j]

            partial_results[result_index] = dfield_total.reshape((Nz, Ny)).get()

            del xg, yg, zg, E0_g, tau_g, phi_g, e1g, e2g, khatg
            gc.collect()

        threads = []
        start_chunk = 1
        for gid in range(n_gpus):
            my_count = chunks_per_gpu + (1 if gid < remainder else 0)
            end_chunk = start_chunk + my_count
            cinds = list(range(start_chunk, end_chunk))
            start_chunk = end_chunk
            t = threading.Thread(target=gpu_worker, args=(gid, cinds, gid))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        for pr in partial_results:
            if pr is not None:
                final_result += pr
        cp.get_default_memory_pool().free_all_blocks()

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
        recompute_cache: bool = False,
        apply_polarization: bool = False,
        spherical_decay: bool = False,
        analyser_mode: str | int = "off",
        analyser_acceptance_angle_rad: float = 0.0,
        analyser_darwin_halfwidth_rad: float = 0.0
    ):
        """
        Single-bounce kinematic scattering on the detector, on GPU or CPU.

        Args:
            sample: Chunked atom positions and species.
            detector: pixel_coordinates (3, Ny*Nz) in Angstrom and shape
                (Ny, Nz), Ny = width, Nz = height.
            stage: rotation (3x3) and translation (3,).
            offset: Subtracted from the final field if given.
            use_gpu: Use the GPU path when CuPy is available.
            remove_forward: Subtract f0(0) from f0(Q) to drop the forward term.
            use_depth_ein: Use cached per-atom depth-dependent Ein.
            ein_cache_dir: Directory for Ein cache files.
            recompute_cache: Recompute the Ein cache even if present.
            apply_polarization: Apply the polarization factor.
            spherical_decay: Apply 1/R spherical decay.
            analyser_mode: "off"/0, "top_hat"/1, or "darwin"/2. GPU only;
                ignored with a warning on the CPU path.
            analyser_acceptance_angle_rad: Top-hat acceptance angle in radians.
            analyser_darwin_halfwidth_rad: Darwin half-width in radians.

        Returns:
            Complex field of shape (Nz, Ny), i.e. (shape[1], shape[0]), with
            offset subtracted if given.
        """
        measurement_positions = detector.pixel_coordinates  # (3, Ny*Nz) in Angstrom
        Ny, Nz = detector.shape

        if use_gpu and (cp is not None):
            final_field = self.interact_beam_gpu(
                sample,
                measurement_positions,
                (Ny, Nz),
                stage,
                remove_forward=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache,
                apply_polarization=apply_polarization,
                spherical_decay=spherical_decay,
                analyser_mode=analyser_mode,
                analyser_acceptance_angle_rad=analyser_acceptance_angle_rad,
                analyser_darwin_halfwidth_rad=analyser_darwin_halfwidth_rad
            )
        else:
            if cp is None and use_gpu:
                self._log("normal", "[beam] Cupy not installed, running CPU mode.")
            # The analyser exists only in the GPU kernels; warn if one was asked for.
            analyser_off = (
                analyser_mode is None
                or (isinstance(analyser_mode, str)
                    and analyser_mode.strip().lower() in ("off", "none", "disabled"))
                or (not isinstance(analyser_mode, str) and analyser_mode == 0)
            )
            if not analyser_off:
                warnings.warn("analyser acceptance is GPU-only; ignored on CPU path")
            final_field = self.interact_beam_cpu(
                sample,
                measurement_positions,
                (Ny, Nz),
                stage,
                detector=None,
                remove_forward_component=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache,
                apply_polarization=apply_polarization,
                apply_spherical_decay=spherical_decay
            )

        return (final_field - offset) if (offset is not None) else final_field
    # -------------------------------------
        
    # -------------------------------------
    # Direct transmission
    def _compute_beam_slice_integrals_cpu(self, sample, stage, slice_edges_A, kernel_radius=0):
        """
        Per-slice column integrals of the forward scattering factors on the beam
        grid (CPU, Angstrom units).

        Atoms are stage-transformed, binned by depth s along the beam direction
        into slices [edges[k], edges[k+1]), and deposited on the (NyB, NzB) grid
        with a separable 3x3 TSC kernel:
            delta_int[k] = C * sum_slice (f0(0) + f1) * W_TSC
            beta_int[k]  = C * sum_slice  f2         * W_TSC
        with C = r_e * lambda^2 / (2*pi * A_pix). f0(0) comes from Waasmaier-Kirfel,
        (f1, f2) from Cromer-Liberman interpolated at the beam energy.

        Args:
            sample: Chunked sample (chunk_total, load_chunk_positions,
                load_chunk_species), positions in Angstrom.
            stage: Rigid transform (rotation 3x3, translation 3) in Angstrom.
            slice_edges_A: Shape (n_slices+1,), monotonic depth edges in Angstrom.
            kernel_radius (int): Gaussian blur radius in pixels per slice; 0 = off.

        Returns:
            (delta_int, beta_int): lists of n_slices float32 (NyB, NzB) arrays.
            beta_int is clamped >= 0 after blurring so no slice shows gain.
        """
        # Constants in Angstrom
        r_e_A = 2.81794092e-5
        lam_A = float(self._wavelength) * 1e10
        two_pi = 2.0 * np.pi
        kA = two_pi / lam_A

        du, dv = float(self._beam_du), float(self._beam_dv)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        A_pix_A2 = du * dv

        # Per-slice accumulators of forward sums (real: f0(0)+f1, imag: f2)
        nS = int(len(slice_edges_A) - 1)
        sum_real = [np.zeros((NyB, NzB), np.float32) for _ in range(nS)]
        sum_imag = [np.zeros((NyB, NzB), np.float32) for _ in range(nS)]

        f1f2_dict      = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict   = self._build_f0_zero_dict(f0_params_dict)

        e1 = self._beam_e1.astype(np.float32)
        e2 = self._beam_e2.astype(np.float32)
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)

        def _tsc_w(d):
            w = np.zeros_like(d, dtype=np.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0]*d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        for cid in range(1, int(sample.chunk_total or 0) + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
            if pos.size == 0:
                continue

            # Stage transform in Angstrom
            pos = pos @ stage.rotation.astype(np.float32).T
            pos += stage.translation.astype(np.float32)

            nA = pos.shape[0]
            f1 = np.zeros(nA, np.float32)
            f2 = np.zeros(nA, np.float32)
            f0z = np.zeros(nA, np.float32)
            for el in np.unique(spc):
                el_s = str(el)
                m = (spc == el_s)
                f0z[m] = float(f0_zero_dict.get(el_s, 0.0))
                tbl = f1f2_dict.get(el_s)
                if tbl is not None:
                    cplx = self.get_f1f2_from_params(self._energy, tbl)
                    f1[m] = float(cplx.real)
                    f2[m] = float(cplx.imag)

            # Beam-basis grid coordinates and depth-slice index
            au = pos[:, 0]*e1[0] + pos[:, 1]*e1[1] + pos[:, 2]*e1[2]
            av = pos[:, 0]*e2[0] + pos[:, 1]*e2[1] + pos[:, 2]*e2[2]
            iu = au/du + float(self._beam_uc)
            iv = av/dv + float(self._beam_vc)

            s_vals = pos[:, 0]*k_hat[0] + pos[:, 1]*k_hat[1] + pos[:, 2]*k_hat[2]
            k_idx = np.clip(np.searchsorted(slice_edges_A, s_vals, side="right") - 1, 0, nS - 1)

            inb = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
            if not np.any(inb):
                continue

            iu = iu[inb]; iv = iv[inb]
            ki = k_idx[inb]
            fr = (f0z[inb] + f1[inb]).astype(np.float32)
            fi = (f2[inb]).astype(np.float32)

            ic = np.floor(iu + 0.5).astype(np.int64)
            jc = np.floor(iv + 0.5).astype(np.int64)

            du_m1 = np.abs(iu - (ic - 1)); du_0 = np.abs(iu - ic); du_p1 = np.abs(iu - (ic + 1))
            dv_m1 = np.abs(iv - (jc - 1)); dv_0 = np.abs(iv - jc); dv_p1 = np.abs(iv - (jc + 1))

            wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
            wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

            # 3x3 TSC with per-slice accumulation
            for dx, wx in [(-1, wu_m1), (0, wu_0), (1, wu_p1)]:
                ii = ic + dx
                for dy, wy in [(-1, wv_m1), (0, wv_0), (1, wv_p1)]:
                    jj = jc + dy
                    fac = (wx * wy)
                    mask = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                    if not np.any(mask):
                        continue
                    rows = ii[mask]; cols = jj[mask]
                    pidx = (rows * NzB + cols).astype(np.int64)
                    wsel = fac[mask]
                    frs  = fr[mask]
                    fis  = fi[mask]
                    kis  = ki[mask]

                    # One add.at per slice index
                    for s in np.unique(kis):
                        ms = (kis == s)
                        if not np.any(ms):
                            continue
                        np.add.at(sum_real[s].ravel(), pidx[ms], (frs[ms] * wsel[ms]).astype(np.float32))
                        np.add.at(sum_imag[s].ravel(), pidx[ms], (fis[ms] * wsel[ms]).astype(np.float32))

        C = (r_e_A * (lam_A * lam_A)) / (2.0 * np.pi * A_pix_A2)
        delta_int = [C * sr for sr in sum_real]
        beta_int  = [C * si for si in sum_imag]

        # Optional per-slice blur
        if int(kernel_radius) > 0:
            rad = int(kernel_radius); sig = max(1e-6, rad / 2.0)
            y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
            k = np.exp(-(x*x + y*y) / (2.0 * sig * sig)).astype(np.float32)
            k /= max(k.sum(), 1e-20)
            Fk = np.fft.fft2(k, s=delta_int[0].shape)
            for i in range(len(delta_int)):
                d = np.fft.ifft2(np.fft.fft2(delta_int[i]) * Fk).real.astype(np.float32)
                b = np.fft.ifft2(np.fft.fft2(beta_int[i])  * Fk).real.astype(np.float32)
                beta_int[i]  = np.maximum(0.0, b)
                delta_int[i] = d

        return delta_int, beta_int


    def _compute_beam_slice_integrals_gpu(self, sample, stage, slice_edges_A, kernel_radius=0):
        """
        GPU version of _compute_beam_slice_integrals_cpu; same arguments and
        physics. Falls back to CPU when CuPy or a GPU is missing.

        Slices are accumulated in windows sized to fit GPU memory. Chunks are
        cached on host up to a RAM budget; the rest are streamed from disk with
        a two-deep prefetch on a dedicated H2D stream.

        Returns:
            (delta_int, beta_int): lists of n_slices float32 (NyB, NzB) NumPy
            arrays (results are staged back to host). beta_int clamped >= 0.
        """
        if (cp is None) or (cp.cuda.runtime.getDeviceCount() < 1):
            return self._compute_beam_slice_integrals_cpu(sample, stage, slice_edges_A, kernel_radius)

        nS = int(len(slice_edges_A) - 1)

        # Constants in Angstrom
        r_e_A = 2.81794092e-5
        lam_A = float(self._wavelength) * 1e10
        A_pix_A2 = float(self._beam_du) * float(self._beam_dv)
        C = (r_e_A * (lam_A * lam_A)) / (2.0 * np.pi * A_pix_A2)

        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        bins = NyB * NzB
        Rg = cp.asarray(stage.rotation, dtype=cp.float32)
        Tg = cp.asarray(stage.translation, dtype=cp.float32)

        e1g = cp.asarray(self._beam_e1, dtype=cp.float32)
        e2g = cp.asarray(self._beam_e2, dtype=cp.float32)
        khatg = cp.asarray((self._direction / np.linalg.norm(self._direction)).astype(np.float32))

        f1f2_dict = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero   = self._build_f0_zero_dict(f0_params)

        edges_g = cp.asarray(slice_edges_A, dtype=cp.float32)

        def _tsc_w(d):
            w = cp.zeros_like(d, dtype=cp.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0]*d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        # Slice window: two float32 accumulators of window_size*bins must fit
        # in half of free GPU memory.
        accum_bytes_full = int(nS) * bins * 4 * 2
        try:
            free_b, _ = cp.cuda.runtime.memGetInfo()
            budget = int(0.5 * free_b)  # rest for atom data + temporaries
        except Exception:
            budget = 2 * 1024**3  # 2 GB fallback
        if accum_bytes_full <= budget:
            window_size = nS  # fits in one shot
        else:
            window_size = max(1, budget // (bins * 4 * 2))

        # Host-side accumulator for all slices
        sum_real_host = np.zeros((nS, NyB, NzB), dtype=np.float32)
        sum_imag_host = np.zeros((nS, NyB, NzB), dtype=np.float32)

        # Atom batch cap from free GPU memory
        def _atom_batch_cap():
            try:
                free_b, _ = cp.cuda.runtime.memGetInfo()
                # Per-atom footprint: pos 12 + fr,fi 8 + projections ~80 + TSC ~120 bytes
                bytes_per_atom = 220
                cap = int(0.4 * free_b / max(bytes_per_atom, 1))
                return max(32768, cap)
            except Exception:
                return 2_000_000

        # Host cache of chunk positions + forward factors, bounded by free RAM;
        # chunks that do not fit are reloaded per slice window.
        def _species_factors(spc_arr):
            nA_local = int(spc_arr.shape[0])
            fr_l = np.zeros(nA_local, np.float32)
            fi_l = np.zeros(nA_local, np.float32)
            for el in np.unique(spc_arr):
                el_s = str(el)
                m = (spc_arr == el_s)
                fr_l[m] = float(f0_zero.get(el_s, 0.0))
                tbl = f1f2_dict.get(el_s)
                if tbl is not None:
                    cplx = self.get_f1f2_from_params(self._energy, tbl)
                    fr_l[m] += float(cplx.real)
                    fi_l[m]  = float(cplx.imag)
            return fr_l, fi_l

        try:
            import psutil
            avail_host_b = int(psutil.virtual_memory().available)
        except Exception:
            avail_host_b = 4 * 1024**3  # conservative 4 GB fallback
        cache_budget = max(int(0.9 * avail_host_b), 256 * 1024**2)

        total_chunks = int(sample.chunk_total or 0)
        chunk_cache = {}      # cid -> (pos, fr_h, fi_h)
        empty_chunks = set()
        cached_bytes = 0
        for cid in range(1, total_chunks + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32, copy=False)
            nA = pos.shape[0]
            if nA == 0:
                empty_chunks.add(cid)
                continue
            bytes_here = int(pos.nbytes) + nA * 8  # pos + fr_h(4) + fi_h(4)
            if cached_bytes + bytes_here > cache_budget:
                # Budget reached; remaining chunks are streamed on demand.
                del pos, spc
                break
            fr_h, fi_h = _species_factors(spc)
            chunk_cache[cid] = (pos, fr_h, fi_h)
            cached_bytes += bytes_here
        n_cached = len(chunk_cache)
        n_uncached = total_chunks - n_cached - len(empty_chunks)
        self._log("normal",
                  f"[beam] chunk cache: {n_cached}/{total_chunks} chunks cached "
                  f"({cached_bytes/1024**3:.2f} GB), "
                  f"{n_uncached} streamed from disk per slice window.")

        from concurrent.futures import ThreadPoolExecutor
        prefetch_window = 2
        prefetch_pool = ThreadPoolExecutor(max_workers=prefetch_window)
        stream_h2d = cp.cuda.Stream(non_blocking=True)

        def _prefetch_chunk(cid):
            """
            Worker-thread task: return (cid, (pos_g, fr_g, fi_g)) or (cid, None).

            Uncached chunks are loaded from disk, pinned, and copied on the
            dedicated H2D stream. The worker syncs that stream itself, so the
            main thread never blocks on the copy.
            """
            if cid in empty_chunks:
                return cid, None
            if cid in chunk_cache:
                pos_h, fr_h, fi_h = chunk_cache[cid]
                pin_required = False
            else:
                spc_l = sample.load_chunk_species(cid, use_gpu=False)
                pos_l = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32, copy=False)
                if pos_l.shape[0] == 0:
                    empty_chunks.add(cid)
                    return cid, None
                fr_l, fi_l = _species_factors(spc_l)
                # Pinned buffers make the H2D copy truly asynchronous.
                try:
                    pos_h = self.allocate_pinned_array(pos_l, dtype=np.float32)
                    fr_h  = self.allocate_pinned_array(fr_l,  dtype=np.float32)
                    fi_h  = self.allocate_pinned_array(fi_l,  dtype=np.float32)
                except Exception:
                    # Pinned memory exhausted: pageable fallback, copy is
                    # synchronous inside cp.asarray.
                    pos_h, fr_h, fi_h = pos_l, fr_l, fi_l
                pin_required = True
            with stream_h2d:
                pos_g = cp.asarray(pos_h)
                fr_g  = cp.asarray(fr_h)
                fi_g  = cp.asarray(fi_h)
            # Sync on the worker, not on main, so compute kernels keep running.
            stream_h2d.synchronize()
            # Release pinned buffers; cached chunks keep their copy in chunk_cache.
            if pin_required:
                del pos_h, fr_h, fi_h
            return cid, (pos_g, fr_g, fi_g)

        try:
            for w_start in range(0, nS, window_size):
                w_end = min(w_start + window_size, nS)
                w_len = w_end - w_start
                w_bins = w_len * bins

                sum_real_flat = cp.zeros(w_bins, dtype=cp.float32)
                sum_imag_flat = cp.zeros(w_bins, dtype=cp.float32)

                # Prime the pipeline with prefetch_window chunks ahead.
                inflight = {}
                for cid_init in range(1, min(prefetch_window, total_chunks) + 1):
                    inflight[cid_init] = prefetch_pool.submit(_prefetch_chunk, cid_init)
                next_cid = prefetch_window + 1

                for cid in range(1, total_chunks + 1):
                    # Block until this chunk is on the GPU.
                    fut = inflight.pop(cid, None)
                    if fut is None:
                        # Not prefetched; fetch now.
                        fut = prefetch_pool.submit(_prefetch_chunk, cid)
                    _, gpu_data = fut.result()

                    # Keep the pool working ahead of compute.
                    if next_cid <= total_chunks:
                        inflight[next_cid] = prefetch_pool.submit(_prefetch_chunk, next_cid)
                        next_cid += 1

                    if gpu_data is None:
                        continue
                    pos_g, fr_g, fi_g = gpu_data
                    nA = int(pos_g.shape[0])

                    batch_cap = _atom_batch_cap()

                    for b_start in range(0, nA, batch_cap):
                        b_end = min(b_start + batch_cap, nA)

                        # GPU views (no copy) into the prefetched arrays.
                        pos_b = pos_g[b_start:b_end]
                        fr = fr_g[b_start:b_end]
                        fi = fi_g[b_start:b_end]

                        posg = pos_b @ Rg.T
                        posg = posg + Tg

                        au = posg[:, 0]*e1g[0] + posg[:, 1]*e1g[1] + posg[:, 2]*e1g[2]
                        av = posg[:, 0]*e2g[0] + posg[:, 1]*e2g[1] + posg[:, 2]*e2g[2]
                        iu = au/float(self._beam_du) + float(self._beam_uc)
                        iv = av/float(self._beam_dv) + float(self._beam_vc)

                        s_vals = posg[:, 0]*khatg[0] + posg[:, 1]*khatg[1] + posg[:, 2]*khatg[2]
                        ki = cp.clip(cp.searchsorted(edges_g, s_vals, side="right") - 1, 0, nS - 1)

                        # Atoms inside the grid and this slice window. No
                        # bool(inb.any()) check: it would force a host sync per
                        # batch, and empty scatter_adds are cheap.
                        inb = ((iu >= 0.0) & (iu <= (NyB - 1)) &
                               (iv >= 0.0) & (iv <= (NzB - 1)) &
                               (ki >= w_start) & (ki < w_end))

                        iu = iu[inb]; iv = iv[inb]
                        fr_b = fr[inb]; fi_b = fi[inb]
                        ki_b = (ki[inb] - w_start).astype(cp.int64)

                        ic = cp.floor(iu + 0.5).astype(cp.int64)
                        jc = cp.floor(iv + 0.5).astype(cp.int64)

                        du_m1 = cp.abs(iu - (ic - 1)); du_0 = cp.abs(iu - ic); du_p1 = cp.abs(iu - (ic + 1))
                        dv_m1 = cp.abs(iv - (jc - 1)); dv_0 = cp.abs(iv - jc); dv_p1 = cp.abs(iv - (jc + 1))

                        wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
                        wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

                        # fused scatter_add over (slice, pixel)
                        for dx, wx in [(-1, wu_m1), (0, wu_0), (1, wu_p1)]:
                            ii = ic + dx
                            for dy, wy in [(-1, wv_m1), (0, wv_0), (1, wv_p1)]:
                                jj = jc + dy
                                fac = wx * wy
                                mask = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                                if not bool(mask.any()):
                                    continue
                                pidx = (ii[mask] * NzB + jj[mask]).astype(cp.int64)
                                flat_idx = (ki_b[mask] * bins + pidx)
                                wsel = fac[mask]
                                cupyx.scatter_add(sum_real_flat, flat_idx, (fr_b[mask] * wsel).astype(cp.float32))
                                cupyx.scatter_add(sum_imag_flat, flat_idx, (fi_b[mask] * wsel).astype(cp.float32))

                    del pos_g, fr_g, fi_g, gpu_data

                # Copy window results to host
                sum_real_host[w_start:w_end] = sum_real_flat.reshape(w_len, NyB, NzB).get()
                sum_imag_host[w_start:w_end] = sum_imag_flat.reshape(w_len, NyB, NzB).get()
                del sum_real_flat, sum_imag_flat
        finally:
            prefetch_pool.shutdown(wait=True)

        # Per-slice integrals, kept on host
        delta_int = [np.float32(C) * sum_real_host[s] for s in range(nS)]
        beta_int  = [np.float32(C) * sum_imag_host[s] for s in range(nS)]

        # Optional blur, one slice at a time on GPU
        if int(kernel_radius) > 0 and len(delta_int) > 0:
            rad = int(kernel_radius); sig = max(1e-6, rad / 2.0)
            yg = cp.arange(-rad, rad + 1, dtype=cp.float32)[:, None]
            xg = cp.arange(-rad, rad + 1, dtype=cp.float32)[None, :]
            kg = cp.exp(-(xg * xg + yg * yg) / (2.0 * sig * sig))
            kg /= cp.sum(kg)
            Fk = cp.fft.fft2(kg, delta_int[0].shape)
            for i in range(len(delta_int)):
                d_g = cp.asarray(delta_int[i])
                b_g = cp.asarray(beta_int[i])
                d_g = cp.fft.ifft2(cp.fft.fft2(d_g) * Fk).real.astype(cp.float32)
                b_g = cp.fft.ifft2(cp.fft.fft2(b_g) * Fk).real.astype(cp.float32)
                b_g = cp.maximum(cp.float32(0.0), b_g)
                delta_int[i] = d_g.get()
                beta_int[i]  = b_g.get()
            del d_g, b_g
            cp.get_default_memory_pool().free_all_blocks()

        return delta_int, beta_int
    
    def _auto_slice_count_linear_regime(
        self,
        sample,
        stage,
        kernel_radius=0,
        target_step=0.1,
        use_gpu=True,
        max_slices=2048,
        n_init=None,
        absorption_multiplier=1.0,
    ):
        """
        Choose the number of projection slices so every thin-slice update
        A_k = exp(-tau_k + i*phi_k) satisfies max(|phi_k|, tau_k) <= target_step
        on every pixel.

        Slice integrals are computed once at a power-of-two fine resolution
        (>= the lower bound from the full-thickness A map, capped at max_slices).
        Coarser trial counts are checked by merging fine slices, doubling n
        until the target is met. Slices are equal-depth bins.

        Args:
            sample, stage: As in _compute_beam_slice_integrals_gpu.
            kernel_radius (int): Blur radius forwarded to the slice integrals.
            target_step (float): Max per-slice phase (rad) or attenuation.
            use_gpu (bool): Use the GPU helpers when CuPy is available.
            max_slices (int): Hard cap on the slice count.
            n_init (int or None): Starting trial count instead of the computed n0.
            absorption_multiplier (float): Scales tau when checking the bound.

        Returns:
            (n_final, edges_A, delta_list, beta_list, info): slice count,
            (n_final+1,) depth edges in Angstrom, per-slice integrals for that
            count, and a dict with 'phi_max', 'tau_max', 'n0'. Zero thickness
            returns n_final=1 with empty lists.
        """
        use_gpu = bool(use_gpu and (cp is not None))

        # Depth window and trivial thickness guard
        s_min_A, s_max_A = self._compute_global_depth_bounds(sample, stage)
        thickness_A = float(max(0.0, s_max_A - s_min_A))
        if thickness_A <= 0.0:
            return 1, np.array([s_min_A, s_max_A], dtype=np.float32), [], [], {"phi_max": 0.0, "tau_max": 0.0, "n0": 1}

        two_pi = 2.0 * np.pi
        lam_A = float(self._wavelength) * 1e10
        kA = two_pi / lam_A
        ts = float(max(1e-6, target_step))

        # Quick lower bound from the full-thickness A(u,v)
        if use_gpu:
            A_full = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius)
            Ab = cp.asarray(A_full)
            phi_tot_max = float(cp.max(cp.abs(cp.angle(Ab))).get())
            tau_tot_max = float(cp.max(cp.maximum(-cp.log(cp.abs(Ab) + cp.float32(1e-20)), cp.float32(0.0))).get())
        else:
            A_full = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)
            Ab = np.asarray(A_full)
            phi_tot_max = float(np.max(np.abs(np.angle(Ab))))
            tau_tot_max = float(np.max(np.maximum(-np.log(np.abs(Ab) + np.float32(1e-20)), 0.0)))

        abs_m = float(absorption_multiplier)
        n0 = int(max(1, np.ceil(max(phi_tot_max, abs_m * tau_tot_max) / ts)))
        n_start = int(max(1, n_init if (n_init is not None) else n0))
        n_start = min(n_start, int(max_slices))

        # Round up to the next power of 2 so all coarser trial values divide evenly
        n_fine = n_start
        p = 1
        while p < n_fine:
            p *= 2
        n_fine = min(p, int(max_slices))

        # Slice integrals once, at the finest resolution
        edges_fine = np.linspace(s_min_A, s_max_A, n_fine + 1, dtype=np.float32)
        if use_gpu:
            delta_fine, beta_fine = self._compute_beam_slice_integrals_gpu(
                sample, stage, edges_fine, kernel_radius)
        else:
            delta_fine, beta_fine = self._compute_beam_slice_integrals_cpu(
                sample, stage, edges_fine, kernel_radius)

        # Max per-slice phase/attenuation for n_trial, merging groups of
        # n_fine // n_trial consecutive fine slices (host numpy arrays).
        def _check_n(n_trial):
            if n_trial == n_fine:
                d_list, b_list = delta_fine, beta_fine
            else:
                group = n_fine // n_trial
                d_list, b_list = [], []
                for i in range(n_trial):
                    s = i * group
                    d_merged = delta_fine[s].copy()
                    b_merged = beta_fine[s].copy()
                    for j in range(1, group):
                        d_merged += delta_fine[s + j]
                        b_merged += beta_fine[s + j]
                    d_list.append(d_merged)
                    b_list.append(b_merged)

            phi_max = 0.0
            tau_max = 0.0
            for d, b in zip(d_list, b_list):
                pm = float(np.max(np.abs((-kA) * d)))
                tm = float(np.max(np.maximum(kA * abs_m * b, 0.0)))
                if pm > phi_max: phi_max = pm
                if tm > tau_max: tau_max = tm

            return phi_max, tau_max, d_list, b_list

        # Search from coarsest (n_start) upward by doubling until target is met
        n = n_start
        while True:
            # Only trial values that evenly divide n_fine
            n_trial = min(n, n_fine)
            while n_fine % n_trial != 0 and n_trial < n_fine:
                n_trial += 1
            n_trial = min(n_trial, n_fine)

            phi_max, tau_max, d_list, b_list = _check_n(n_trial)

            if max(phi_max, tau_max) <= ts or n_trial >= int(max_slices):
                edges_out = np.linspace(s_min_A, s_max_A, n_trial + 1, dtype=np.float32)
                info = {"phi_max": float(phi_max), "tau_max": float(tau_max), "n0": int(n0)}
                return int(max(1, n_trial)), edges_out, d_list, b_list, info

            if n_trial >= n_fine:
                # Finest resolution reached
                edges_out = np.linspace(s_min_A, s_max_A, n_fine + 1, dtype=np.float32)
                info = {"phi_max": float(phi_max), "tau_max": float(tau_max), "n0": int(n0)}
                return int(max(1, n_fine)), edges_out, delta_fine, beta_fine, info

            n = min(n * 2, int(max_slices))

    def _compute_beam_column_A_map_cpu(self, sample, stage, kernel_radius=0):
        """
        Transmission column map A(u,v) = exp(-tau + i*phi) on the beam grid (CPU).

        Each atom is stage-transformed, projected onto the beam basis and
        deposited with a 3x3 TSC kernel. The column sums are scaled to
        delta/beta integrals, converted to phi = -k*delta, tau = k*beta, and
        exponentiated. Only atoms projecting inside the grid contribute.

        Args:
            sample: Chunked sample (chunk_total, load_chunk_positions,
                load_chunk_species), positions in Angstrom.
            stage: Rigid transform (rotation 3x3, translation 3) in Angstrom.
            kernel_radius (int): Gaussian blur radius in pixels; 0 = off.

        Returns:
            complex64 array of shape (NyB, NzB). tau is clamped >= 0 after the
            blur so the map never shows gain.
        """
        # Constants (angstrom)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10

        # Beam-grid geometry
        du, dv = float(self._beam_du), float(self._beam_dv)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        A_pix_A2 = du * dv

        sum_real = np.zeros((NyB, NzB), np.float32)  # sum of f0(0)+f1
        sum_imag = np.zeros((NyB, NzB), np.float32)  # sum of f2

        f1f2_dict      = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict   = self._build_f0_zero_dict(f0_params_dict)

        e1 = self._beam_e1
        e2 = self._beam_e2

        def _tsc_w(d):
            # 1D TSC weights for distances in pixel units
            w = np.zeros_like(d, dtype=np.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0]*d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        for cid in range(1, sample.chunk_total + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)  # Angstrom
            if pos.size == 0:
                continue

            # Stage transform (Angstrom)
            pos = pos @ stage.rotation.T
            pos += stage.translation

            nA = pos.shape[0]
            f1  = np.zeros(nA, np.float32)
            f2  = np.zeros(nA, np.float32)
            f0z = np.zeros(nA, np.float32)
            for el in np.unique(spc):
                el_s = str(el)
                m = (spc == el_s)
                f0z[m] = float(f0_zero_dict.get(el_s, 0.0))
                tbl = f1f2_dict.get(el_s)
                if tbl is not None:
                    cplx = self.get_f1f2_from_params(self._energy, tbl)
                    f1[m] = float(cplx.real)
                    f2[m] = float(cplx.imag)

            # Project to beam basis and grid indices
            au = pos[:, 0]*e1[0] + pos[:, 1]*e1[1] + pos[:, 2]*e1[2]
            av = pos[:, 0]*e2[0] + pos[:, 1]*e2[1] + pos[:, 2]*e2[2]
            iu = au/du + self._beam_uc
            iv = av/dv + self._beam_vc

            inb = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
            if not np.any(inb):
                continue

            iu = iu[inb]; iv = iv[inb]
            fr = (f0z[inb] + f1[inb]).astype(np.float32)  # real forward factor
            fi = (f2[inb]).astype(np.float32)             # imag forward factor

            ic = np.floor(iu + 0.5).astype(np.int64)
            jc = np.floor(iv + 0.5).astype(np.int64)

            du_m1 = np.abs(iu - (ic - 1)); du_0 = np.abs(iu - ic); du_p1 = np.abs(iu - (ic + 1))
            dv_m1 = np.abs(iv - (jc - 1)); dv_0 = np.abs(iv - jc); dv_p1 = np.abs(iv - (jc + 1))

            wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
            wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

            idx_list_R = []; w_list_R = []
            idx_list_I = []; w_list_I = []

            def _push(ii, jj, fac, val):
                mask = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                if not np.any(mask):
                    return
                rows = ii[mask]; cols = jj[mask]
                idx  = (rows * NzB + cols).astype(np.int64)
                w    = (val[mask] * fac[mask]).astype(np.float32)
                return idx, w

            # 3x3 TSC deposition
            for dx, wx in [(-1, wu_m1), (0, wu_0), (1, wu_p1)]:
                ii = ic + dx
                for dy, wy in [( -1, wv_m1), (0, wv_0), (1, wv_p1)]:
                    jj   = jc + dy
                    fac  = wx * wy

                    r = _push(ii, jj, fac, fr)
                    if r is not None:
                        idx_list_R.append(r[0]); w_list_R.append(r[1])
                    r = _push(ii, jj, fac, fi)
                    if r is not None:
                        idx_list_I.append(r[0]); w_list_I.append(r[1])

            if idx_list_R:
                idxR = np.concatenate(idx_list_R); wR = np.concatenate(w_list_R)
                idxI = np.concatenate(idx_list_I); wI = np.concatenate(w_list_I)
                np.add.at(sum_real.ravel(), idxR, wR)
                np.add.at(sum_imag.ravel(), idxI, wI)

        # Column integrals delta/beta = C * sums, then phi = -k*delta, tau = k*beta
        two_pi = 2.0 * np.pi
        C = (r_e_A * (lam_A * lam_A)) / (two_pi * A_pix_A2)  # dimensionless
        delta_int = C * sum_real.astype(np.float32)
        beta_int  = C * sum_imag.astype(np.float32)

        kA = two_pi / lam_A
        phi = (-kA * delta_int).astype(np.float32)
        tau = ( kA * beta_int ).astype(np.float32)

        # Never allow gain
        tau = np.maximum(tau, np.float32(0.0))

        # Optional blur
        if kernel_radius > 0:
            rad = int(kernel_radius); sig = rad / 2.0
            y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
            k = np.exp(-(x*x + y*y) / (2.0*sig*sig)).astype(np.float32)
            k /= k.sum()
            Fk = np.fft.fft2(k, s=phi.shape)
            phi = np.fft.ifft2(np.fft.fft2(phi) * Fk).real.astype(np.float32)
            tau = np.fft.ifft2(np.fft.fft2(tau) * Fk).real.astype(np.float32)
            tau = np.maximum(tau, np.float32(0.0))  # keep no-gain after blur

        A_map = np.exp(-tau + 1j * phi).astype(np.complex64)
        return A_map

    def _compute_beam_column_A_map_gpu(self, sample, stage, kernel_radius=0):
        """
        GPU version of _compute_beam_column_A_map_cpu. Chunks are split across
        all visible GPUs; each builds a partial A map that is multiplied into
        the total on the host. Falls back to CPU without a GPU.
        """
        if cp is None:
            return self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            return self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        # Constants (angstrom)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10

        du, dv = float(self._beam_du), float(self._beam_dv)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        A_pix_A2 = du * dv

        # Pin stage for faster H2D
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        f1f2_dict      = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict   = self._build_f0_zero_dict(f0_params_dict)

        partial = [None] * n_gpus
        chunks_per_gpu = sample.chunk_total // n_gpus
        remainder      = sample.chunk_total % n_gpus

        def worker(dev_id, chunks, out_idx):
            cp.cuda.Device(dev_id).use()
            Rg = cp.asarray(R_pin); Tg = cp.asarray(T_pin)

            sum_real_g = cp.zeros((NyB, NzB), dtype=cp.float32)
            sum_imag_g = cp.zeros((NyB, NzB), dtype=cp.float32)

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
                pos = sample.load_chunk_positions(cid, use_gpu=False)  # Angstrom
                nA  = pos.shape[0]
                if nA == 0:
                    continue

                # Build per-atom forward factors on host
                f1  = np.zeros(nA, np.float32)
                f2  = np.zeros(nA, np.float32)
                f0z = np.zeros(nA, np.float32)
                for el in pd.unique(spc):
                    el_s = str(el)
                    m = (spc == el_s)
                    f0z[m] = float(f0_zero_dict.get(el_s, 0.0))
                    tbl = f1f2_dict.get(el_s)
                    if tbl is not None:
                        cplx = self.get_f1f2_from_params(self._energy, tbl)
                        f1[m] = float(cplx.real)
                        f2[m] = float(cplx.imag)

                fr_g = cp.asarray(f0z + f1, dtype=cp.float32)  # real forward part
                fi_g = cp.asarray(f2,         dtype=cp.float32)  # imag forward part

                posg = cp.asarray(pos, dtype=cp.float32)
                posg = posg @ Rg.T; posg += Tg

                e1g = cp.asarray(self._beam_e1)
                e2g = cp.asarray(self._beam_e2)
                au = posg[:, 0]*e1g[0] + posg[:, 1]*e1g[1] + posg[:, 2]*e1g[2]
                av = posg[:, 0]*e2g[0] + posg[:, 1]*e2g[1] + posg[:, 2]*e2g[2]

                iu = au/du + self._beam_uc
                iv = av/dv + self._beam_vc

                inb = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
                if not bool(cp.any(inb)):
                    cp.get_default_memory_pool().free_all_blocks()
                    continue

                iu = iu[inb]; iv = iv[inb]
                fr = fr_g[inb]; fi = fi_g[inb]

                ic = cp.floor(iu + 0.5).astype(cp.int64)
                jc = cp.floor(iv + 0.5).astype(cp.int64)

                du_m1 = cp.abs(iu - (ic - 1)); du_0 = cp.abs(iu - ic); du_p1 = cp.abs(iu - (ic + 1))
                dv_m1 = cp.abs(iv - (jc - 1)); dv_0 = cp.abs(iv - jc); dv_p1 = cp.abs(iv - (jc + 1))

                wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
                wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

                idxR = []; wR = []
                idxI = []; wI = []

                def _push(ii, jj, fac, val):
                    mask = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                    if not bool(cp.any(mask)):
                        return None
                    rows = ii[mask]; cols = jj[mask]
                    idx  = (rows * NzB + cols).astype(cp.int64)
                    w    = (val[mask] * fac[mask]).astype(cp.float32)
                    return idx, w

                for dx, wx in [(-1, wu_m1), (0, wu_0), (1, wu_p1)]:
                    ii = ic + dx
                    for dy, wy in [(-1, wv_m1), (0, wv_0), (1, wv_p1)]:
                        jj = jc + dy
                        fac = wx * wy
                        r = _push(ii, jj, fac, fr)
                        if r is not None:
                            idxR.append(r[0]); wR.append(r[1])
                        r = _push(ii, jj, fac, fi)
                        if r is not None:
                            idxI.append(r[0]); wI.append(r[1])

                if idxR:
                    idxR = cp.concatenate(idxR); wR = cp.concatenate(wR)
                    idxI = cp.concatenate(idxI); wI = cp.concatenate(wI)
                    bins = NyB * NzB
                    sum_real_g += self._safe_bincount_gpu(idxR, wR, bins, dtype=cp.float32).reshape(NyB, NzB)
                    sum_imag_g += self._safe_bincount_gpu(idxI, wI, bins, dtype=cp.float32).reshape(NyB, NzB)

                cp.get_default_memory_pool().free_all_blocks()

            # Convert forward sums -> phi, tau
            two_pi = 2.0 * np.pi
            C = (r_e_A * (lam_A * lam_A)) / (two_pi * A_pix_A2)
            delta_int = C * sum_real_g
            beta_int  = C * sum_imag_g

            kA = two_pi / lam_A
            phi_g = (-kA * delta_int).astype(cp.float32)
            tau_g = ( kA * beta_int ).astype(cp.float32)
            tau_g = cp.maximum(tau_g, cp.float32(0.0))  # no gain

            if kernel_radius > 0:
                rad = int(kernel_radius); sig = rad / 2.0
                yg = cp.arange(-rad, rad + 1, dtype=cp.float32)[:, None]
                xg = cp.arange(-rad, rad + 1, dtype=cp.float32)[None, :]
                kg = cp.exp(-(xg * xg + yg * yg) / (2.0 * sig * sig))
                kg /= cp.sum(kg)
                Fk = cp.fft.fft2(kg, phi_g.shape)
                phi_g = cp.fft.ifft2(cp.fft.fft2(phi_g) * Fk).real.astype(cp.float32)
                tau_g = cp.fft.ifft2(cp.fft.fft2(tau_g) * Fk).real.astype(cp.float32)
                tau_g = cp.maximum(tau_g, cp.float32(0.0))

            A_gpu = cp.exp(-tau_g + 1j * phi_g).astype(cp.complex64)
            partial[out_idx] = A_gpu.get()
            cp.get_default_memory_pool().free_all_blocks()

        # One thread per GPU
        import threading
        threads = []
        start = 1
        for gid in range(n_gpus):
            n_chunk = chunks_per_gpu + (1 if gid < remainder else 0)
            end = start + n_chunk
            t = threading.Thread(target=worker, args=(gid, range(start, end), gid))
            t.start(); threads.append(t)
            start = end
        for t in threads:
            t.join()

        # Chunk contributions multiply into the total
        A_total = np.ones((NyB, NzB), np.complex64)
        for p in partial:
            if p is not None:
                A_total *= p
        return A_total
    
    def atomic_transmission(self, sample, detector, stage,
                            use_gpu=True, kernel_radius=0,
                            padding_mode="edge", pad_constant=0.0,
                            n_slices=None, target_phase_step=0.1,
                            pad_factor=2, absorption_multiplier=1.0):
        """
        Projection-only multislice transmission of the beam through the sample.

        The beam field is multiplied slice by slice, E <- E * A_k(u,v) with
        A_k = exp(-tau_k + i*phi_k), with no free-space propagation between
        slices. The exit field is bilinearly resampled onto the detector pixels
        (out-of-bounds pixels get 0) and, if the detector plane is offset from
        the exit plane, propagated by the signed mean offset. Zero thickness
        passes the beam through unchanged.

        Args:
            sample: Chunked atoms with species.
            detector: Provides shape=(Ny, Nz) and pixel_coordinates (3, Ny*Nz)
                in Angstrom; Ny = width, Nz = height. pixel_size (dy, dz) in
                Angstrom is used for the offset hop when present, otherwise the
                spacing is estimated from the projected (u, v) coordinates.
            stage: Rigid transform (rotation 3x3, translation 3) in Angstrom.
            use_gpu (bool): Use CuPy paths when available.
            kernel_radius (int): Gaussian blur radius (pixels) for per-slice maps.
            padding_mode (str): "edge" or "constant" padding for the offset hop.
            pad_constant (float): Pad value when padding_mode="constant".
            n_slices (int or None): Slice count; None auto-selects via
                _auto_slice_count_linear_regime with target_phase_step.
            target_phase_step (float): Per-slice linear-regime target (rad).
            pad_factor (float): Minimum FFT padding factor (>= 1).
            absorption_multiplier (float): Scales tau (1 = physical, 0 = none).

        Returns:
            complex64 array of shape (Nz, Ny): exit field on the detector.

        Geometry is in Angstrom internally; the propagation step uses metres.
        """
        use_gpu = bool(use_gpu and (cp is not None))

        # Beam grid info (angstrom units)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        du_A = float(self._beam_du)
        dv_A = float(self._beam_dv)

        # Depth bounds in angstrom along the beam direction
        s_min_A, s_max_A = self._compute_global_depth_bounds(sample, stage)
        thickness_A = float(max(0.0, s_max_A - s_min_A))

        two_pi = 2.0 * np.pi
        lam_A = float(self._wavelength) * 1e10
        kA = two_pi / lam_A

        # -------- Auto-slice selection for linear per-slice increments --------
        delta_list = None
        beta_list = None
        edges_A = None

        abs_m = float(absorption_multiplier)

        if thickness_A <= 0.0:
            n_final = 1
        else:
            if n_slices is None:
                n_final, edges_A, delta_list, beta_list, _ = self._auto_slice_count_linear_regime(
                    sample=sample,
                    stage=stage,
                    kernel_radius=kernel_radius,
                    target_step=float(target_phase_step),
                    use_gpu=use_gpu,
                    max_slices=2048,
                    n_init=None,
                    absorption_multiplier=abs_m
                )
            else:
                n_final = int(max(1, n_slices))
        
        print(f"{n_final} steps used")
        
        # -------- Build exit wave on the beam grid (projection-only multislice) --------
        if thickness_A <= 0.0:
            # Zero thickness: beam passes through unchanged
            if use_gpu:
                E_exit = cp.asarray(self._beam_E0_map.astype(np.complex64))
            else:
                E_exit = self._beam_E0_map.astype(np.complex64)
        else:
            # Product of per-slice transmissions only (no intra-slice propagation)
            if edges_A is None:
                edges_A = np.linspace(s_min_A, s_max_A, n_final + 1, dtype=np.float32)

            if use_gpu:
                if delta_list is None or beta_list is None:
                    delta_list, beta_list = self._compute_beam_slice_integrals_gpu(
                        sample, stage, edges_A, kernel_radius
                    )
                # One stacked H2D copy instead of 2*n_final small ones.
                delta_g = cp.asarray(np.stack(delta_list, axis=0))   # (nS, NyB, NzB)
                beta_g  = cp.asarray(np.stack(beta_list,  axis=0))
                E = cp.asarray(self._beam_E0_map.astype(np.complex64))
                for k in range(n_final):
                    phi_k = (-kA * delta_g[k]).astype(cp.float32)
                    tau_k = cp.maximum((kA * abs_m * beta_g[k]).astype(cp.float32),
                                       cp.float32(0.0))
                    arg_k = cp.empty_like(phi_k, dtype=cp.complex64)
                    arg_k.real = -tau_k
                    arg_k.imag = phi_k
                    E = (E * cp.exp(arg_k)).astype(cp.complex64)
                del delta_g, beta_g
                E_exit = E
            else:
                if delta_list is None or beta_list is None:
                    delta_list, beta_list = self._compute_beam_slice_integrals_cpu(
                        sample, stage, edges_A, kernel_radius
                    )
                E = self._beam_E0_map.astype(np.complex64)
                for k in range(n_final):
                    phi_k = (-kA * delta_list[k]).astype(np.float32)
                    tau_k = (kA * abs_m * beta_list[k]).astype(np.float32)
                    tau_k = np.maximum(tau_k, np.float32(0.0))
                    arg_k = np.empty_like(phi_k, dtype=np.complex64)
                    arg_k.real = -tau_k
                    arg_k.imag = phi_k
                    A_k = np.exp(arg_k)
                    E = (E * A_k).astype(np.complex64)
                E_exit = E

        # -------- Resample E_exit (beam grid) to detector pixels by bilinear interpolation --------
        Ny, Nz = detector.shape
        pix = detector.pixel_coordinates  # (3, Ny*Nz) in angstrom

        if use_gpu:
            pix_g = pix if isinstance(pix, cp.ndarray) else cp.asarray(pix)
            e1g = cp.asarray(self._beam_e1)
            e2g = cp.asarray(self._beam_e2)

            # transverse coordinates of each detector pixel
            u = pix_g[0] * e1g[0] + pix_g[1] * e1g[1] + pix_g[2] * e1g[2]
            v = pix_g[0] * e2g[0] + pix_g[1] * e2g[1] + pix_g[2] * e2g[2]

            # beam-grid fractional indices
            iu = u / cp.float32(du_A) + cp.float32(self._beam_uc)
            iv = v / cp.float32(dv_A) + cp.float32(self._beam_vc)

            mask = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))

            # bilinear neighbours and weights
            i0 = cp.floor(iu).astype(cp.int64); j0 = cp.floor(iv).astype(cp.int64)
            i1 = cp.clip(i0 + 1, 0, NyB - 1); j1 = cp.clip(j0 + 1, 0, NzB - 1)
            fu = (iu - i0).astype(cp.float32); fv = (iv - j0).astype(cp.float32)

            Eb = E_exit if isinstance(E_exit, cp.ndarray) else cp.asarray(E_exit)
            idx00 = (i0 * NzB + j0).astype(cp.int64)
            idx01 = (i0 * NzB + j1).astype(cp.int64)
            idx10 = (i1 * NzB + j0).astype(cp.int64)
            idx11 = (i1 * NzB + j1).astype(cp.int64)

            E00 = Eb.ravel()[idx00]; E01 = Eb.ravel()[idx01]
            E10 = Eb.ravel()[idx10]; E11 = Eb.ravel()[idx11]

            one = cp.float32(1.0)
            E_det_exit = (E00 * (one - fu) * (one - fv) +
                        E01 * (one - fu) * fv +
                        E10 * fu * (one - fv) +
                        E11 * fu * fv).astype(cp.complex64)

            # zero out-of-bounds
            E_det_exit = cp.where(mask, E_det_exit, cp.complex64(0.0 + 0.0j))
            E_det_exit = E_det_exit.reshape(Nz, Ny)
        else:
            pix_cpu = pix.get() if (cp is not None and isinstance(pix, cp.ndarray)) else np.asarray(pix)
            e1 = self._beam_e1; e2 = self._beam_e2

            u = pix_cpu[0] * e1[0] + pix_cpu[1] * e1[1] + pix_cpu[2] * e1[2]
            v = pix_cpu[0] * e2[0] + pix_cpu[1] * e2[1] + pix_cpu[2] * e2[2]

            iu = u / du_A + self._beam_uc
            iv = v / dv_A + self._beam_vc

            mask = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))

            i0 = np.floor(iu).astype(np.int64); j0 = np.floor(iv).astype(np.int64)
            i1 = np.clip(i0 + 1, 0, NyB - 1); j1 = np.clip(j0 + 1, 0, NzB - 1)
            fu = (iu - i0).astype(np.float32); fv = (iv - j0).astype(np.float32)

            Eb = np.asarray(E_exit).ravel()
            idx00 = (i0 * NzB + j0).astype(np.int64)
            idx01 = (i0 * NzB + j1).astype(np.int64)
            idx10 = (i1 * NzB + j0).astype(np.int64)
            idx11 = (i1 * NzB + j1).astype(np.int64)

            E00 = Eb[idx00]; E01 = Eb[idx01]; E10 = Eb[idx10]; E11 = Eb[idx11]
            E_det_exit = (E00 * (1.0 - fu) * (1.0 - fv) +
                        E01 * (1.0 - fu) * fv +
                        E10 * fu * (1.0 - fv) +
                        E11 * fu * fv).astype(np.complex64)
            E_det_exit[~mask] = 0.0 + 0.0j
            E_det_exit = E_det_exit.reshape(Nz, Ny)

        # -------- Optional free-space hop from exit plane to detector plane --------
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)

        if use_gpu:
            pix_cpu_for_s = pix if isinstance(pix, np.ndarray) else pix.get()
        else:
            pix_cpu_for_s = np.asarray(pix)

        s_det = (pix_cpu_for_s[0, :] * k_hat[0] +
                pix_cpu_for_s[1, :] * k_hat[1] +
                pix_cpu_for_s[2, :] * k_hat[2]).astype(np.float64)

        s_det_min = float(np.min(s_det))
        s_det_max = float(np.max(s_det))
        s_det_mean = float(np.mean(s_det))
        plane_span_A = s_det_max - s_det_min
        tol_plane_A = max(1e-3, 1e-6 * abs(s_det_mean))
        tol_off_A = 1e-3

        dz_A = s_det_mean - float(s_max_A)
        need_propagation = False
        if plane_span_A <= tol_plane_A:
            need_propagation = (abs(dz_A) > tol_off_A)
        else:
            # Non-planar detector: propagate by mean offset
            need_propagation = (abs(dz_A) > tol_off_A)

        if not need_propagation:
            return (E_det_exit.get() if (use_gpu and isinstance(E_det_exit, cp.ndarray)) else E_det_exit).astype(np.complex64)

        # Propagate full detector field by dz using the detector sampling
        dz_m = float(dz_A) * 1e-10

        # Detector spacing: pixel_size if available, else estimated from (u, v)
        def _estimate_dy_dz_from_uv(u_flat, v_flat, nz, ny):
            u_img = u_flat.reshape(nz, ny)
            v_img = v_flat.reshape(nz, ny)
            # dy = u step across columns, dz = v step across rows
            du_cols = np.abs(u_img[:, 1:] - u_img[:, :-1]).ravel()
            dv_rows = np.abs(v_img[1:, :] - v_img[:-1, :]).ravel()
            dy_A_est = float(np.median(du_cols)) if du_cols.size else 0.0
            dz_A_est = float(np.median(dv_rows)) if dv_rows.size else 0.0
            # Fall back to the beam-grid spacing
            if not np.isfinite(dy_A_est) or dy_A_est <= 0.0:
                dy_A_est = du_A
            if not np.isfinite(dz_A_est) or dz_A_est <= 0.0:
                dz_A_est = dv_A
            return dy_A_est * 1e-10, dz_A_est * 1e-10

        # dy, dz in metres
        have_psize = hasattr(detector, "pixel_size")
        dy_m = dz_m2 = None
        if have_psize:
            try:
                ps_y, ps_z = detector.pixel_size   # (dy, dz) lab frame in Angstroms
                dy_m = float(ps_y) * 1e-10         # col spacing = lab Y spacing
                dz_m2 = float(ps_z) * 1e-10        # row spacing = lab Z spacing
            except Exception:
                dy_m = dz_m2 = None

        if use_gpu:
            e1l = self._beam_e1; e2l = self._beam_e2
            u = pix_cpu_for_s[0] * e1l[0] + pix_cpu_for_s[1] * e1l[1] + pix_cpu_for_s[2] * e1l[2]
            v = pix_cpu_for_s[0] * e2l[0] + pix_cpu_for_s[1] * e2l[1] + pix_cpu_for_s[2] * e2l[2]
            if (dy_m is None) or (dz_m2 is None) or (dy_m <= 0.0) or (dz_m2 <= 0.0):
                dy_m, dz_m2 = _estimate_dy_dz_from_uv(u, v, Nz, Ny)
        else:
            # u, v already computed above for CPU branch
            if (dy_m is None) or (dz_m2 is None) or (dy_m <= 0.0) or (dz_m2 <= 0.0):
                dy_m, dz_m2 = _estimate_dy_dz_from_uv(u, v, Nz, Ny)

        if use_gpu:
            prop_kernel = self.build_propagation_multiplier_kernel()
            E_gpu = E_det_exit if isinstance(E_det_exit, cp.ndarray) else cp.asarray(E_det_exit)
            E_gpu = self._angular_spectrum_propagate_gpu(
                field=E_gpu, dy=dy_m, dz=dz_m2, z=dz_m, kernel=prop_kernel,
                step_max=0.02, pad_factor=float(pad_factor),
                padding_mode=str(padding_mode), pad_constant=float(pad_constant)
            )
            return E_gpu.get().astype(np.complex64)
        else:
            ffi, lib = self.compile_propagation_multiplier_cffi()
            E_out = self._angular_spectrum_propagate_cpu(
                field=E_det_exit, dy=dy_m, dz=dz_m2, z=dz_m, lib=lib, ffi=ffi,
                step_max=0.02, pad_factor=float(pad_factor),
                padding_mode=str(padding_mode), pad_constant=float(pad_constant)
            )
            return E_out.astype(np.complex64)
    # -------------------------------------    
    
    # -------------------------------------
    # Atomic master
    def atomic_direct_interaction(self, sample, detector, stage,
                                scattering=True, sc_kwargs=None,
                                transmission=False, tr_kwargs=None,
                                use_gpu=True):
        """
        Compute the scattering and/or transmission field and write the sum to
        the detector.

        Args:
            sample: Sample with the chunk accessors the backends need.
            detector: Provides shape (Ny, Nz), pixel_coordinates and
                input_pixel_values(array).
            stage: Rigid transform (rotation 3x3, translation 3).
            scattering (bool): Include the kinematic scattering term.
            sc_kwargs (dict or None): Keyword arguments forwarded to
                atomic_scattering_kinematic (e.g. offset, use_depth_ein,
                apply_polarization, spherical_decay, remove_forward, use_gpu).
            transmission (bool): Include the transmission term.
            tr_kwargs (dict or None): Keyword arguments forwarded to
                atomic_transmission (e.g. kernel_radius, padding_mode,
                pad_constant, n_slices, absorption_multiplier, use_gpu).
            use_gpu (bool): Not used here; set use_gpu inside sc_kwargs or
                tr_kwargs to control each backend.

        Returns:
            None. The combined complex64 field of shape (Nz, Ny) is written via
            detector.input_pixel_values.

        When both terms are enabled, pass remove_forward=True in sc_kwargs so
        the forward f0(0) term is not counted twice.
        """
        Ny, Nz = detector.shape
        final_field = np.zeros((Nz, Ny), dtype=np.complex64)

        # -------- Compute and combine terms --------
        if scattering:
            sc_field = self.atomic_scattering_kinematic(
                sample=sample,
                detector=detector,
                stage=stage,
                **(sc_kwargs or {})
            )
            final_field += np.asarray(sc_field, dtype=np.complex64)

        if transmission:
            tx_field = self.atomic_transmission(
                sample=sample,
                detector=detector,
                stage=stage,
                **(tr_kwargs or {})
            )
            final_field += np.asarray(tx_field, dtype=np.complex64)

        detector.input_pixel_values(final_field)
    # -------------------------------------
    
    # -------------------------------------
    # Wavefield propagation
    def _angular_spectrum_propagate_gpu(
            self, field, dy, dz, z, kernel,
            step_max=0.02, pad_factor=1.0,
            padding_mode: str = "edge",
            pad_constant: float = 0.0,
            cos_theta: float = 1.0,
            k_g_axis: float = None,
            k_g_perp_y: float = 0.0,
            k_g_perp_z: float = 0.0,
        ):
        """
        Band-limited angular-spectrum propagation on the GPU.

        Distances longer than step_max are split into equal sub-steps to bound
        phase error and wrap-around. Each step pads the field symmetrically to
        the size from _choose_optimal_pad (power of two, at least pad_factor
        times larger), applies FFT -> transfer function (CUDA kernel) -> IFFT,
        and centre-crops back to (Nz, Ny).

        Args:
            field: Complex (Nz, Ny) array, NumPy or CuPy.
            dy, dz: Pixel size along Y (width) and Z (height) in metres.
            z (float): Propagation distance in metres, may be negative.
            kernel: "prop_mul_kernel" from build_propagation_multiplier_kernel.
            step_max (float): Maximum sub-step distance in metres.
            pad_factor (float): Minimum padding factor (>= 1).
            padding_mode (str): "edge" replicates edges, "constant" uses pad_constant.
            cos_theta, k_g_axis, k_g_perp_y, k_g_perp_z: Carrier wavevector of
                the field in rad/m. If k_g_axis is None it is taken as
                k*cos_theta with zero transverse components.

        Returns:
            cupy.ndarray: complex64 field after propagation, shape (Nz, Ny).

        Raises RuntimeError without CuPy.
        """
        if cp is None:
            raise RuntimeError('CuPy required for GPU propagation')

        # Split long distances into sub-steps
        z = float(z)
        if abs(z) > step_max:
            n = int(np.ceil(abs(z) / step_max))
            dz_step = z / n
            out = cp.asarray(field) if isinstance(field, cp.ndarray) else cp.asarray(field, dtype=cp.complex64)
            for _ in range(n):
                out = self._angular_spectrum_propagate_gpu(
                    out, dy, dz, dz_step, kernel,
                    step_max=step_max, pad_factor=pad_factor,
                    padding_mode=padding_mode, pad_constant=pad_constant,
                    cos_theta=cos_theta,
                    k_g_axis=k_g_axis, k_g_perp_y=k_g_perp_y, k_g_perp_z=k_g_perp_z,
                )
            return out

        F0 = cp.asarray(field, dtype=cp.complex64)
        Nz, Ny = int(F0.shape[0]), int(F0.shape[1])

        # Symmetric padding from sampling, distance and pad_factor
        Ny2, Nz2 = self._choose_optimal_pad(
            Ny, Nz, float(dy), float(dz), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        z0 = (Nz2 - Nz) // 2
        y0 = (Ny2 - Ny) // 2

        pmode = (padding_mode or "edge").lower()
        if pmode == "constant":
            Fp = cp.full((Nz2, Ny2), complex(pad_constant), dtype=cp.complex64)
            Fp[z0:z0+Nz, y0:y0+Ny] = F0
        else:
            pad_spec = ((z0, Nz2 - Nz - z0), (y0, Ny2 - Ny - y0))
            Fp = cp.pad(F0, pad_spec, mode='edge')

        # k-grids in rad/m, unshifted to match fft2 ordering
        k  = 2.0 * np.pi / float(self._wavelength)
        ky = (2.0 * np.pi) * cp.fft.fftfreq(Ny2, d=float(dy)).astype(cp.float32)
        kz = (2.0 * np.pi) * cp.fft.fftfreq(Nz2, d=float(dz)).astype(cp.float32)

        # Without k_g_axis, assume a tilted beam with no transverse carrier:
        # k_g_axis = k*cos_theta, k_g_perp = 0.
        if k_g_axis is None:
            k_g_axis_v = float(k) * float(cos_theta)
        else:
            k_g_axis_v = float(k_g_axis)

        Fp = cp.fft.fft2(Fp)

        # Propagator multiply in place
        block = (16, 16)
        grid  = ((Ny2 + block[0] - 1)//block[0],
                (Nz2 + block[1] - 1)//block[1])
        kernel(grid, block,
            (ky, kz, cp.float32(k), cp.float32(z),
                np.int32(Ny2), np.int32(Nz2), Fp,
                cp.float32(k_g_axis_v),
                cp.float32(k_g_perp_y),
                cp.float32(k_g_perp_z)))

        # Inverse FFT and centre crop
        out = cp.fft.ifft2(Fp)
        return out[z0:z0+Nz, y0:y0+Ny]
    
    def _angular_spectrum_propagate_cpu(
            self, field, dy, dz, z, lib, ffi,
            step_max=0.02, pad_factor=1.0,
            padding_mode: str = "edge",
            pad_constant: float = 0.0
        ):
        """
        CPU counterpart of _angular_spectrum_propagate_gpu, using
        lib.prop_mul_cpu from compile_propagation_multiplier_cffi. Same
        sub-stepping, padding and cropping; no carrier-tilt arguments.

        Args:
            field: Complex (Nz, Ny) array.
            dy, dz: Pixel size along Y (width) and Z (height) in metres.
            z (float): Propagation distance in metres, may be negative.
            lib, ffi: CFFI library and FFI object.
            step_max, pad_factor, padding_mode, pad_constant: As in the GPU version.

        Returns:
            np.ndarray: complex64 field after propagation, shape (Nz, Ny).
        """
        z = float(z)
        if abs(z) > step_max:
            n = int(np.ceil(abs(z) / step_max))
            dz_step = z / n
            out = field
            for _ in range(n):
                out = self._angular_spectrum_propagate_cpu(
                    out, dy, dz, dz_step, lib, ffi,
                    step_max=step_max, pad_factor=pad_factor,
                    padding_mode=padding_mode, pad_constant=pad_constant
                )
            return out

        F0 = np.asarray(field, dtype=np.complex64, order='C')
        Nz, Ny = int(F0.shape[0]), int(F0.shape[1])

        # Symmetric padding
        Ny2, Nz2 = self._choose_optimal_pad(
            Ny, Nz, float(dy), float(dz), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        z0 = (Nz2 - Nz) // 2
        y0 = (Ny2 - Ny) // 2

        pmode = (padding_mode or "edge").lower()
        if pmode == "constant":
            Fp = np.full((Nz2, Ny2), pad_constant + 0j, dtype=np.complex64)
            Fp[z0:z0+Nz, y0:y0+Ny] = F0
        else:
            pad_spec = ((z0, Nz2 - Nz - z0), (y0, Ny2 - Ny - y0))
            Fp = np.pad(F0, pad_spec, mode='edge')

        # Spectral axes (rad/m)
        k  = np.float32(2.0 * np.pi / float(self._wavelength))
        ky = (2.0*np.pi) * np.fft.fftfreq(Ny2, d=float(dy)).astype(np.float32)
        kz = (2.0*np.pi) * np.fft.fftfreq(Nz2, d=float(dz)).astype(np.float32)

        Fp = np.fft.fft2(Fp)

        # Propagator multiply in place
        lib.prop_mul_cpu(
            np.int32(Ny2), np.int32(Nz2),
            ffi.cast('const float*', ky.ctypes.data),
            ffi.cast('const float*', kz.ctypes.data),
            k, np.float32(z),
            ffi.cast('float _Complex*', Fp.ctypes.data)
        )

        # Inverse FFT and center crop
        out = np.fft.ifft2(Fp)
        return out[z0:z0+Nz, y0:y0+Ny]

    def wavefield_propagation(self, detector, optics,
                              use_gpu=True, step_max=0.02, pad_factor=1.0,
                              padding_mode: str = "edge",
                              pad_constant: float = 0.0, save_field=True):
        """
        Propagate the detector wavefield through an optics stack.

        Args:
            detector: Provides pixel_size (dy, dz) in Angstrom, shape (Ny, Nz),
                pixel_values (complex64, (Nz, Ny)) and input_pixel_values(array).
            optics: Provides apply_stack(field, dx, dy, wavelength,
                propagate_free_space, use_gpu).
            use_gpu, step_max, pad_factor, padding_mode, pad_constant:
                Free-space propagation controls (see _angular_spectrum_propagate_gpu).
            save_field (bool): Write the result back to the detector; otherwise
                return it.

        Returns:
            None, or the complex64 (Nz, Ny) field when save_field is False.

        Raises ValueError if the detector field or the beam wavelength is unset.
        """
        # Pixel sizes: Angstrom -> metres
        pixel_size = np.asarray(detector.pixel_size, dtype=np.float64)
        dy, dz = pixel_size * 1e-10
        E = detector.pixel_values  # complex64 (Nz, Ny)

        if E is None:
            raise ValueError(
                "detector.pixel_values is None. Run atomic_direct_interaction() first "
                "to populate the detector field before calling wavefield_propagation()."
            )

        if self._wavelength is None:
            raise ValueError(
                "Beam wavelength is not set. Call beam.define_beam() first to set energy/wavelength."
            )

        # Build free-space propagator as a closure so optics need not import beam.
        if use_gpu and cp is not None:
            kernel = self.build_propagation_multiplier_kernel()
            def _propagate_free_space(F, dym, dzm, z):
                return self._angular_spectrum_propagate_gpu(
                    F, dym, dzm, z, kernel,
                    step_max=step_max, pad_factor=pad_factor,
                    padding_mode=padding_mode, pad_constant=pad_constant
                ).get()
        else:
            ffi, lib = self.compile_propagation_multiplier_cffi()
            def _propagate_free_space(F, dym, dzm, z):
                return self._angular_spectrum_propagate_cpu(
                    F, dym, dzm, z, lib, ffi,
                    step_max=step_max, pad_factor=pad_factor,
                    padding_mode=padding_mode, pad_constant=pad_constant
                )

        # Optics uses (dx, dy) = (column, row) spacing.
        E = optics.apply_stack(
            field=E,
            dx=dy,   # column spacing = lab Y pixel size
            dy=dz,   # row spacing = lab Z pixel size
            wavelength=self._wavelength,
            propagate_free_space=_propagate_free_space,
            use_gpu=(use_gpu and (cp is not None))
        )

        if save_field is True:
            detector.input_pixel_values(E.astype(np.complex64))
        else:
            return E.astype(np.complex64)
    # -------------------------------------