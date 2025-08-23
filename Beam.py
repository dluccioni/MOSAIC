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
        Initialize a beam instance.

        Args:
            directory (str, optional): Directory used to store and read beam-related
                metadata. Defaults to the current working directory.

        Notes:
            - Creates the directory if it does not exist.
            - Initializes physical constants in SI units (Planck constant h, speed of
            light c, elementary charge q) and caches h/q for fast eV-to-wavelength
            conversion.
        """
        self.directory = directory
        self._direction = None
        self._energy = None
        self._wavelength = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        # Constants (SI units)
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
                    gaussian_waist=None,
                    pol_perp_rate=0.5):
        """
        Configure a forward-propagating (+x) beam and build its transverse grid.

        The beam direction is fixed to +x for performance. Polarization is modeled
        by pol_perp_rate, the fraction of incident intensity polarized perpendicular
        to the scattering plane (rho_perp). A value of 0.5 corresponds to an
        unpolarized beam.

        Args:
            energy (float): Beam energy. If eV is True, interpreted as electron-volts;
                otherwise as joules.
            eV (bool, optional): If True, the energy is in eV. If False, energy is in J.
            beam_shape (str, optional): "rectangular" or "circular" support on the
                transverse grid.
            beam_size (tuple[float, float], optional): Physical size (u, v) of the
                beam support in angstrom.
            beam_samples (tuple[int, int], optional): Number of samples (Ny, Nz) on
                the transverse grid.
            beam_profile (str, optional): "uniform" or "gaussian" amplitude profile.
            gaussian_waist (tuple[float, float] or None, optional): Gaussian 1/e^2
                radii (wy, wz) in angstrom. If None and beam_profile is "gaussian",
                defaults to half of beam_size per axis.
            pol_perp_rate (float, optional): Fraction in [0, 1] for polarization
                perpendicular to the scattering plane. 0.5 is unpolarized.

        Returns:
            None
        """
        # Force direction to +x for performance and consistency
        self._direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Energy -> wavelength bookkeeping
        if not eV:
            energy = energy / self._q
        self._energy = float(energy)
        self._wavelength = self._hq * self._c / self._energy

        # k-vector scalars; only kx is nonzero for +x propagation
        k = 2.0 * np.pi / self._wavelength
        self._kx_scalar = np.float32(k)
        self._ky_scalar = np.float32(0.0)
        self._kz_scalar = np.float32(0.0)

        # Store shape, size, sampling, and profile
        self._beam_shape   = str(beam_shape).lower()
        self._beam_size    = (float(beam_size[0]), float(beam_size[1]))
        self._beam_samples = (int(beam_samples[0]), int(beam_samples[1]))
        self._beam_profile = str(beam_profile).lower()
        self._gauss_waist  = gaussian_waist

        # Polarization bookkeeping
        self._pol_perp_rate = float(np.clip(pol_perp_rate, 0.0, 1.0))

        # Transverse basis vectors are exactly unit y and unit z for +x propagation
        self._beam_e1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self._beam_e2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Build the transverse grid and the initial field map E0(u, v)
        self._init_beam_grid()
        
    def _init_beam_grid(self):
        """
        Build the (u, v) transverse grid centered at 0 using current beam_size and
        beam_samples, and create the complex field map without phase.

        Sets:
            - _beam_Ny, _beam_Nz: grid sizes along u and v
            - _beam_du, _beam_dv: grid spacings in angstrom
            - _beam_uc, _beam_vc: center indices
            - _beam_u_centers, _beam_v_centers: coordinate arrays (angstrom)
            - _beam_E0_map: complex64 amplitude map with zero phase
        """
        Ny, Nz = self._beam_samples
        Sy, Sz = self._beam_size  # angstrom
        Ny = int(max(1, Ny)); Nz = int(max(1, Nz))
        self._beam_Ny, self._beam_Nz = Ny, Nz

        # Grid spacings (angstrom per grid step)
        self._beam_du = float(Sy) / Ny
        self._beam_dv = float(Sz) / Nz
        self._beam_uc = (Ny - 1) * 0.5  # center index along u
        self._beam_vc = (Nz - 1) * 0.5  # center index along v

        # Grid center coordinates (Ny, Nz) in angstrom
        u_centers = (np.arange(Ny, dtype=np.float32) - self._beam_uc) * self._beam_du
        v_centers = (np.arange(Nz, dtype=np.float32) - self._beam_vc) * self._beam_dv
        U, V = np.meshgrid(u_centers, v_centers, indexing='ij')  # (Ny, Nz)

        # Support mask: circular or full rectangular support
        if self._beam_shape == "circular":
            ry = 0.5 * Sy
            rz = 0.5 * Sz
            mask = ((U / max(ry, 1e-9))**2 + (V / max(rz, 1e-9))**2) <= 1.0
        else:
            mask = np.ones_like(U, dtype=bool)  # rectangular support equals full grid

        # Amplitude profile (no phase here)
        if self._beam_profile == "gaussian":
            wy, wz = self._gauss_waist if (self._gauss_waist is not None) else (0.5 * Sy, 0.5 * Sz)
            wy = max(float(wy), 1e-6); wz = max(float(wz), 1e-6)
            A0 = np.exp(-((U / wy) ** 2 + (V / wz) ** 2)).astype(np.float32)
            A0 *= mask.astype(np.float32)
        else:
            A0 = mask.astype(np.float32)

        self._beam_u_centers = u_centers
        self._beam_v_centers = v_centers
        # Complex field with zero phase
        self._beam_E0_map = (A0.astype(np.float32) + 0.0j).astype(np.complex64)
        
    def read_beam_metadata(self):
        """
        Read beam metadata from JSON and restore the beam state, including the
        transverse grid.

        Rebuilds:
            - Normalized direction vector
            - k-vector components (_kx_scalar, _ky_scalar, _kz_scalar)
            - Orthonormal transverse basis (self._beam_e1, self._beam_e2)
            - Beam grid (centers, spacings, E0 profile) via _init_beam_grid()

        Compatibility:
            Backward compatible with metadata files that predate beam-grid fields.

        Raises:
            FileNotFoundError: If the metadata JSON file cannot be found in the
                configured directory.
        """
        metadata_filename = os.path.join(self.directory, "beam_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            beam_metadata = json.load(f)

        # Core scalars and direction
        direction = beam_metadata.get("direction", None)
        if direction is None:
            direction = [1.0, 0.0, 0.0]
        self._direction = np.array(direction, dtype=np.float32)
        self._direction = self._direction / np.linalg.norm(self._direction)

        self._energy     = float(beam_metadata.get("energy", self._energy if self._energy is not None else 1.0))
        self._wavelength = float(beam_metadata.get("wavelength",
                                                (self._hq * self._c / self._energy)))

        # Wavevector components (derived from direction and wavelength)
        k = 2.0 * np.pi / self._wavelength
        self._kx_scalar = float(self._direction[0] * k)
        self._ky_scalar = float(self._direction[1] * k)
        self._kz_scalar = float(self._direction[2] * k)

        # Beam-grid primitives
        self._beam_shape = str(beam_metadata.get("beam_shape", "rectangular")).lower()

        # Ensure non-degenerate default sizes (angstrom)
        default_size = (1000.0, 1000.0)
        size_list = beam_metadata.get("beam_size", default_size)
        if size_list is None or len(size_list) != 2:
            size_list = default_size
        Sy = float(size_list[0]) if float(size_list[0]) > 0.0 else default_size[0]
        Sz = float(size_list[1]) if float(size_list[1]) > 0.0 else default_size[1]
        self._beam_size = (Sy, Sz)

        # Samples (Ny, Nz). Fall back to a sensible grid if missing.
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
            # If profile is gaussian but waist is missing, default to half of size
            if self._beam_profile == "gaussian":
                self._gauss_waist = (0.5 * Sy, 0.5 * Sz)
            else:
                self._gauss_waist = None
        else:
            # Accept list/tuple/float-like
            if isinstance(gw, (list, tuple)) and len(gw) == 2:
                self._gauss_waist = (float(gw[0]), float(gw[1]))
            else:
                # Malformed input -> safe default if gaussian, else None
                self._gauss_waist = (0.5 * Sy, 0.5 * Sz) if self._beam_profile == "gaussian" else None

        # Transverse basis and grid build
        e1, e2 = self.make_orthonormal_basis(self._direction)
        self._beam_e1 = e1.astype(np.float32)
        self._beam_e2 = e2.astype(np.float32)

        # Build the beam grid and E0(u, v) based on loaded settings
        if hasattr(self, "_init_beam_grid"):
            self._init_beam_grid()

        print(f"Beam metadata loaded from {metadata_filename}.")

    ## Data Handling Functions    
    def write_beam_metadata(self, override_directory=None):
        """
        Serialize the beam state, including the beam-grid definition, to a JSON file.

        Newly saved fields:
            - beam_samples: [Ny, Nz] on the transverse (u, v) grid
            - beam_profile: "uniform" or "gaussian"
            - gaussian_waist: [wy, wz] in angstrom for Gaussian profile (1/e^2 radii)
            or null
            - metadata_version: integer schema tag (>= 2 when beam grid is present)

        Args:
            override_directory (str or None, optional): If provided, write the
                metadata file to this directory instead of self.directory.

        Returns:
            None
        """
        # Graceful fallbacks if older attributes are not present
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

        beam_metadata = {
            "metadata_version": 2,
            "direction"       : direction,
            "energy"          : energy,
            "wavelength"      : wavelength,
            "beam_shape"      : beam_shape,
            "beam_size"       : beam_size,       # [size_u_angstrom, size_v_angstrom]
            "beam_samples"    : beam_samples,    # [Ny, Nz]
            "beam_profile"    : beam_profile,    # "uniform" | "gaussian"
            "gaussian_waist"  : gauss_waist      # [wy_angstrom, wz_angstrom] or null
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
        Construct two unit vectors orthonormal to a given direction.

        Given a beam direction vector d, this returns (e1, e2) such that:
        - e1 dot d = 0
        - e2 dot d = 0
        - e1 dot e2 = 0
        - ||e1|| = ||e2|| = 1

        Args:
            direction (np.ndarray): Array-like of shape (3,) representing the
                reference direction.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays of shape (3,) corresponding
                to e1 and e2.

        Notes:
            The implementation picks a temporary axis not nearly colinear with d,
            then uses cross products to build an orthonormal basis.
        """
        d = direction / np.linalg.norm(direction)

        # Pick an axis that is not almost parallel to d
        if abs(d[0]) < 0.9:
            temp = np.array([1, 0, 0], dtype=np.float32)
        else:
            temp = np.array([0, 1, 0], dtype=np.float32)

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
        Allocate page-locked (pinned) host memory and copy data from a NumPy array.

        This is useful for faster host-to-device transfers with CuPy. The returned
        array is backed by pinned memory and can be passed to cp.array(...).

        Args:
            np_array (np.ndarray): Source array to copy from.
            dtype (np.dtype or type, optional): Desired dtype of the pinned array.
                If None, uses np_array.dtype. Defaults to np.float32.

        Returns:
            np.ndarray: A NumPy array with the same shape as np_array, backed by
            pinned host memory.

        Notes:
            Requires CuPy to be available in the runtime. This function does not
            validate that cp is not None.
        """
        if dtype is None:
            dtype = np_array.dtype
        shape = np_array.shape
        n_elems = np.prod(shape)

        # Allocate a pinned memory block via CuPy
        memptr = cp.cuda.alloc_pinned_memory(
            n_elems * np.dtype(dtype).itemsize
        )
        # Wrap the pinned block in a NumPy array view
        pinned_arr = np.ndarray(shape=shape, dtype=dtype, buffer=memptr)
        # Copy input data into pinned array
        pinned_arr[...] = np_array
        return pinned_arr
    
    @staticmethod
    def parse_f0_db_all(database_name='f0_WaasKirf.dat'):
        """
        Parse Waasmaier-Kirfel f0 parameters for all elements from a resource file.

        The file is expected to be packaged under databases.scattering and contain
        element sections introduced by lines starting with "#S". Parameter lines
        contain exactly 11 float values:
            [a1, a2, a3, a4, a5, c, b1, b2, b3, b4, b5]

        Args:
            database_name (str, optional): Resource filename within databases.scattering.
                Defaults to "f0_WaasKirf.dat".

        Returns:
            dict[str, np.ndarray]: Mapping from element symbol to a float32 array
            of shape (11,) with the parameters in the order listed above.
        """
        db_dict = {}
        with pkg_resources.open_text(databases.scattering, database_name) as db_file:
            element = None
            for line in db_file:
                # Section header sets current element
                if line.startswith('#S'):
                    element = line.split()[2].strip()
                # Parameter lines carry 11 numbers; skip comments and empty lines
                elif (not line.startswith('#')) and element is not None:
                    params = np.fromiter((float(x) for x in line.split()), dtype=np.float32)
                    if params.size == 11:
                        db_dict[element] = params
        return db_dict

    @staticmethod
    def parse_f1f2_db_all(database_name='f1f2_CromerLiberman.dat'):
        """
        Parse Cromer-Liberman anomalous scattering tables (f1, f2) for all elements.

        The file is expected to be packaged under databases.scattering and contain
        element sections introduced by lines starting with "#S". Data rows contain
        three floats per line:
            [Energy_eV, f1, f2]

        Args:
            database_name (str, optional): Resource filename within databases.scattering.
                Defaults to "f1f2_CromerLiberman.dat".

        Returns:
            dict[str, np.ndarray]: Mapping from element symbol to an array of shape
            (N, 3) with columns [Energy_eV, f1, f2]. dtype=float32.
        """
        f1f2_dict = {}
        with pkg_resources.open_text(databases.scattering, database_name) as db_file:
            element = None
            param_list = []
            for line in db_file:
                # Start of a new element section
                if line.startswith('#S'):
                    if element is not None and len(param_list) > 0:
                        f1f2_dict[element] = np.array(param_list, dtype=np.float32)
                    element = line.split()[2].strip()
                    param_list = []
                # Data line with three values
                elif not line.startswith('#') and element is not None:
                    row_vals = [float(val) for val in line.split()]
                    if len(row_vals) == 3:
                        param_list.append(row_vals)
            # Commit the last element, if any
            if element is not None and len(param_list) > 0:
                f1f2_dict[element] = np.array(param_list, dtype=np.float32)
        return f1f2_dict

    @staticmethod
    def get_f1f2_from_params(energy, f1f2_table):
        """
        Linearly interpolate the complex anomalous scattering factor at a given energy.

        This performs a piecewise-linear interpolation over rows [E, f1, f2] and
        returns f1 + 1j*f2 at the requested energy. If the energy falls outside the
        table bounds, the nearest endpoint value is used.

        Args:
            energy (float): Energy in eV at which to evaluate f1 and f2.
            f1f2_table (np.ndarray): Array of shape (N, 3) with columns
                [Energy_eV, f1, f2], sorted by energy ascending.

        Returns:
            complex: Interpolated value f1 + 1j*f2 at the requested energy.
        """
        E = energy
        energies = f1f2_table[:, 0]
        idx = np.searchsorted(energies, E)
        if idx >= len(energies):
            idx = len(energies) - 1
        if idx == 0:
            idx = 1

        # Linear interpolation between bracketing samples
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
        Build and return a CPU scattering routine (CFFI) with optional polarization.
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
                int Nx, int Ny,
                const float* coords_x,      // (Nx*Ny) in meters
                const float* coords_y,
                const float* coords_z,
                float k_val,                // 2*pi/lambda in rad/m
                int apply_pol,              // 0 or 1
                float pol_perp_rate,        // rho_perp in [0, 1]
                float* out_r, float* out_i  // (Nx*Ny)
            );

        Polarization model:
            The scattered complex amplitude is scaled by sqrt(P) where
            P = rho_perp + (1 - rho_perp) * (cos(2*theta))^2.
            For a +x incident beam, cos(2*theta) is approximated by dx/r.

        Returns:
            tuple: (ffi_obj, C_mod) from cffi.verify for calling the compiled function.

        Notes:
            - Requires a working C compiler through cffi.
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
            int Nx, int Ny,
            const float *coords_x,        // (Nx*Ny) [m]
            const float *coords_y,
            const float *coords_z,
            float k_val,                  // 2*pi/lambda [rad/m]
            int   apply_pol,              // 0/1
            float pol_perp_rate,          // rho_perp in [0,1]
            float *out_r, float *out_i    // (Nx*Ny)
        )
        {
            const float PI_F = 3.14159265358979323846f;
            const float rE_F = 2.81794092e-15f;  // classical electron radius [m]
            const int pixel_count = Nx*Ny;

            // Precompute per-pixel Q_cut (one-pixel "radius" in Q-space)
            int have_qcut = 1;
            float* Q_cut = (float*)malloc((size_t)pixel_count * sizeof(float));
            if (!Q_cut) have_qcut = 0;

            // We will also precompute per-pixel R0 (distance from origin to pixel).
            int have_r0 = 1;
            float* R0_arr = (float*)malloc((size_t)pixel_count * sizeof(float));
            if (!R0_arr) have_r0 = 0;

            if (have_qcut || have_r0) {
                for (int p = 0; p < pixel_count; ++p) {
                    // 2D index
                    int ix = p % Nx;
                    int iy = p / Nx;

                    // Center pixel vector and unit direction
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
                        // Right neighbor (or left if on right edge; else self if single column)
                        int n_right = (ix + 1 < Nx) ? (p + 1) : ((ix > 0) ? (p - 1) : p);
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

                        // Up neighbor (or down if on top edge; else self if single row)
                        int n_up = (iy + 1 < Ny) ? (p + Nx) : ((iy > 0) ? (p - Nx) : p);
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

                        // Diagonal half-width in Q (approximate pixel radius in Q-space)
                        float Qhx = 0.5f * Qx;
                        float Qhy = 0.5f * Qy;
                        if (have_qcut) {
                            Q_cut[p] = sqrtf(Qhx*Qhx + Qhy*Qhy);
                        }
                    }
                }
            }

            // Precompute wavelength from k
            const float wavelength_m = (2.0f * PI_F) / k_val;

            // Main accumulation
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

                    // +x incidence approximation for scattering angle
                    float dotv = (dx / r_det);
                    float tmp = 2.0f*(1.0f - dotv);
                    if (tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_val * sqrtf(tmp);

                    // f0(Q)
                    float f0_val = get_f0_value(Q_val, f0p);

                    // Build scattering factor including anomalous
                    float s_re = (f0_val + sanr);
                    float s_im = (sani);

                    // (B) Remove full forward amplitude inside Q < Q_cut
                    if (remove_forward) {
                        if (have_qcut) {
                            if (Q_val < Q_cut[p]) {
                                // subtract (f0(0) + anomalous)
                                s_re -= (f00 + sanr);
                                s_im -= (sani);
                            }
                        } else {
                            // Fallback when Q_cut not available
                            s_re -= (f00 + sanr);
                            s_im -= (sani);
                        }
                    }

                    // multiply by complex entrance amplitude
                    float t_re = amp_r * s_re - amp_i * s_im;
                    float t_im = amp_r * s_im + amp_i * s_re;

                    // Phase: ax + r_det (modulo wavelength reduction for stability)
                    float phase = k_val * (fmodf(ax, wavelength_m) + fmodf(r_det, wavelength_m));
                    float cph = cosf(phase);
                    float sph = sinf(phase);

                    float val_r = (t_re * cph - t_im * sph);
                    float val_i = (t_re * sph + t_im * cph);

                    // (A) Relative spherical-decay factor: R0 / r_det
                    float scale_rel = 1.0f;
                    if (r_det > 0.0f) {
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

                    // polarization factor applied on amplitude
                    if (apply_pol) {
                        float P = pol_perp_rate + (1.0f - pol_perp_rate) * (dotv * dotv);
                        if (P < 0.0f) P = 0.0f;
                        if (P > 1.0f) P = 1.0f;
                        float scale = sqrtf(P);
                        val_r *= scale;
                        val_i *= scale;
                    }

                    // Final scaling and accumulate
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
        # Declare the C function interface for cffi
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
                int   apply_pol,
                float pol_perp_rate,
                float *out_r, float *out_i
            );
        """)
        # Compile with optimization
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
        CPU evaluator: Ein = E0 * exp(-f*tau) * exp(i*f*phi) with bilinear sampling.
        This version returns Ein=0 for atoms that project outside the beam grid.
        Returns complex64 array of length N.
        """
        N = int(pos_np.shape[0])
        out = np.zeros((N,), dtype=np.complex64)
        if N == 0:
            return out

        NyB, NzB = int(tau.shape[0]), int(tau.shape[1])

        # Project to beam basis and grid index space
        au = pos_np[:, 0]*e1[0] + pos_np[:, 1]*e1[1] + pos_np[:, 2]*e1[2]
        av = pos_np[:, 0]*e2[0] + pos_np[:, 1]*e2[1] + pos_np[:, 2]*e2[2]
        iu = au / float(du) + float(uc)
        iv = av / float(dv) + float(vc)

        # In-bounds mask (no edge replication when out)
        inb = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
        if not np.any(inb):
            return out

        # Work only on in-bounds subset
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

        # Depth fraction f in [0,1]
        s_vals = pos_np[inb, 0]*khat[0] + pos_np[inb, 1]*khat[1] + pos_np[inb, 2]*khat[2]
        denom = float(s_max) - float(s_min)
        if not (denom > 0.0):
            denom = 1.0
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
        Set the target maximum phase error (radians) for choosing the series order N.
        Default (if never set) is 1e-6 rad.

        Args:
            phi_tol_rad (float): desired maximum phase error per contribution in radians.
        """
        try:
            val = float(phi_tol_rad)
            if not (val > 0.0):
                val = 1e-6
        except Exception:
            val = 1e-6
        self._phase_tol_rad = val
        
    def _estimate_required_series_terms(self, a_max_m: float, R0_min_m: float, phi_tol_rad: float):
        """
        Estimate the minimum N for the series delta_r = R0 * (sqrt(1+t) - 1),
        such that k * |err_r| <= phi_tol_rad, using worst-case |t| and the
        next-omitted-term bound: err_r ≈ R0 * |C_{N+1}| * |t|^{N+1}.

        Returns:
            dict with keys:
                'use_series' (bool), 'N' (int), 't_max' (float)
        """
        import math
        # Guard wavelength and k
        if getattr(self, "_wavelength", None) is None or self._wavelength <= 0.0:
            # Cannot determine, fall back to EXACT
            return dict(use_series=False, N=0, t_max=float("inf"))

        k_val = 2.0 * math.pi / float(self._wavelength)  # rad/m

        if R0_min_m <= 0.0:
            return dict(use_series=False, N=0, t_max=float("inf"))

        # Worst-case dimensionless t bound for sqrt(1+t), with u·a and |a| <= a_max
        rho = float(a_max_m) / float(R0_min_m)
        t_max = 2.0 * rho + rho * rho

        # Convergence requires |t| < 1
        if not (t_max < 1.0):
            return dict(use_series=False, N=0, t_max=t_max)

        # Iterate coefficients for sqrt(1+t) - 1:
        # C1 = +1/2, Ck = C_{k-1} * ((1/2 - (k-1)) / k)
        # We want smallest N with: k * R0 * |C_{N+1}| * t^{N+1} <= phi_tol_rad
        Nmax = 32
        C = 0.5  # C1
        t = float(t_max)
        # We will maintain t^k and Ck. For N candidate, check NEXT term (N+1).
        # Precompute powers incrementally
        tk = t  # t^1

        # Prepare list of |Ck|*t^k for k starting at 1
        coeff_pow = [(abs(C) * tk, 1)]  # (value, k)

        # Build up to Nmax+1 so we can test the next omitted term
        for k in range(2, Nmax + 1):
            num = 0.5 - (k - 1.0)
            C = C * (num / k)
            tk = tk * t
            coeff_pow.append((abs(C) * tk, k))

        # Now choose N
        use_series = False
        chosen_N = 0
        for N in range(1, Nmax):
            # next omitted is k = N+1
            val, kpow = coeff_pow[N]  # 0-based index, so N -> N+1 term
            err_r = R0_min_m * val
            err_phi = k_val * err_r
            if err_phi <= float(phi_tol_rad):
                use_series = True
                chosen_N = N  # keep exactly N terms
                break

        if not use_series:
            # Not meeting tolerance up to Nmax; you may still choose SERIES with Nmax,
            # but we return EXACT to be conservative.
            return dict(use_series=False, N=0, t_max=t_max)

        # Clamp
        if chosen_N < 1: chosen_N = 1
        if chosen_N > Nmax: chosen_N = Nmax

        return dict(use_series=True, N=chosen_N, t_max=t_max)
        
    def _select_series_mode_once(self, sample, detector, safety_t_thresh=0.5, verbose=True):
        """
        Decide global mode (SERIES vs EXACT) and the series order N automatically.
        Prints the chosen mode and N.

        Uses:
        - sample.dimensions: (Lx, Ly, Lz) in Angstrom, centered at 0
        - detector.pixel_coordinates: (3, Nx*Ny) in Angstrom
        - wavelength from self._wavelength
        - phase tolerance from self._phase_tol_rad (default 1e-2 rad if unset)
        - safety_t_thresh for convergence margin (default 0.5)

        Sets:
        self._global_use_series (bool)
        self._series_terms (int)
        """
        import numpy as _np

        # Sample half-diagonal radius (meters)
        dims_A = _np.asarray(sample.dimensions, dtype=float)
        half_A = 0.5 * dims_A
        a_max_A = float(_np.sqrt(_np.sum(half_A**2)))
        a_max_m = a_max_A * 1e-10

        # Closest detector pixel distance (meters)
        pix = detector.pixel_coordinates
        if (cp is not None) and isinstance(pix, cp.ndarray):
            pix_cpu = pix.get()
        else:
            pix_cpu = _np.asarray(pix)
        r2_min_A2 = float(_np.min(_np.sum(pix_cpu * pix_cpu, axis=0)))
        R0_min_m = (r2_min_A2 ** 0.5) * 1e-10

        # Phase tolerance
        phi_tol = float(getattr(self, "_phase_tol_rad", 1e-2))

        # Estimate N and check convergence bound
        est = self._estimate_required_series_terms(a_max_m, R0_min_m, phi_tol)
        use_series = bool(est["use_series"])
        N_auto = int(est["N"])
        t_max = float(est["t_max"])

        # Apply safety threshold on |t| to avoid marginal series
        if not (R0_min_m > 0.0) or not (t_max < safety_t_thresh):
            use_series = False

        # Persist selection
        self._global_use_series = use_series
        self._series_terms = (N_auto if (use_series and N_auto >= 1) else 1)

        if verbose:
            mode_str = "SERIES" if use_series else "EXACT"
            print("[beam] Geometric mode: {0} (N={1}, phi_tol={2:.3g} rad, a_max={3:.3e} m, R0_min={4:.3e} m, |t|max={5:.3g})"
                .format(mode_str, self._series_terms, phi_tol, a_max_m, R0_min_m, t_max))

    def build_interaction_kernel(self, series_terms: int | None = None, force_mode: str | None = None):
        """
        Build (and cache) the FP32-only kinematic kernel with a global mode and N
        baked in at compile time.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for GPU scattering kernels.")

        # Resolve N and global mode from self unless overridden
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

        # Cache by (N, mode)
        if not hasattr(self, "_interaction_kernel_cache"):
            self._interaction_kernel_cache = {}
        key = (N, global_use_series)
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

        #define CHUNK_SIZE 128

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

        // f0(Q) as before
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

        // Series for sqrt(1+t) - 1 up to N terms
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

        // Main kernel (global mode baked in)
        __global__ void interaction_kernal(
            const int   nAtoms,
            const float* __restrict__ kx_atom,
            const float* __restrict__ ky_atom,
            const float* __restrict__ kz_atom,
            const float* __restrict__ px,   // atom positions in meters
            const float* __restrict__ py,
            const float* __restrict__ pz,
            const float2* __restrict__ initial_amp,
            const float2* __restrict__ scattering_anom,
            const float*  __restrict__ f0_params,
            const float*  __restrict__ f0_zero,
            const float* __restrict__ x_coords,  // detector coords in meters
            const float* __restrict__ y_coords,
            const float* __restrict__ z_coords,
            float2*      __restrict__ detector_field,
            const int    Nx,
            const int    Ny,
            const int    remove_forward,
            const int    apply_polarization,
            const float  pol_perp_rate)
        {
            const float rE_F = 2.81794092e-15f;

            int ix = blockIdx.x * blockDim.x + threadIdx.x;
            int iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= Nx || iy >= Ny) return;
            const int pidx = iy * Nx + ix;

            float tx = x_coords[pidx];
            float ty = y_coords[pidx];
            float tz = z_coords[pidx];

            // Pixel sightline (from origin) and unit vector
            float R0 = sqrtf(tx*tx + ty*ty + tz*tz);
            float invR0 = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
            if (R0 > 0.0f) {
                invR0 = 1.0f / R0;
                ux = tx * invR0;
                uy = ty * invR0;
                uz = tz * invR0;
            }

            if (nAtoms <= 0) return;
            float k_global = fabsf(kx_atom[0]);

            // Base phasor exp(i*k*R0)
            float sb, cb;
            sincos_k_times_reduced(k_global, R0, sb, cb);

            // Compute Q_cut from the local pixel size:
            // take neighbor in +x (or -x at edge) and +y (or -y), build unit directions,
            // then convert angular deltas to Q-steps; use diagonal half-width.
            float Q_cut = 0.0f;
            {
                int n_right = (ix + 1 < Nx) ? (pidx + 1) : ((ix > 0) ? (pidx - 1) : pidx);
                int n_up    = (iy + 1 < Ny) ? (pidx + Nx) : ((iy > 0) ? (pidx - Nx) : pidx);

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

            // Shared tiles
            __shared__ float  s_px[CHUNK_SIZE];
            __shared__ float  s_py[CHUNK_SIZE];
            __shared__ float  s_pz[CHUNK_SIZE];
            __shared__ float2 s_amp[CHUNK_SIZE];
            __shared__ float2 s_anm[CHUNK_SIZE];
            __shared__ float  s_params[CHUNK_SIZE * 11];
            __shared__ float  s_f0z[CHUNK_SIZE];

            const int threads_in_block = blockDim.x * blockDim.y;
            const int t_id = threadIdx.y * blockDim.x + threadIdx.x;

            float2 sum_rel = make_float2(0.0f, 0.0f);

            for (int base = 0; base < nAtoms; base += CHUNK_SIZE) {
                for (int t = t_id; t < CHUNK_SIZE; t += threads_in_block) {
                    int a = base + t;
                    if (a < nAtoms) {
                        s_px[t] = px[a]; s_py[t] = py[a]; s_pz[t] = pz[a];
                        s_amp[t]= initial_amp[a];
                        s_anm[t]= scattering_anom[a];
                        s_f0z[t]= f0_zero[a];
                        #pragma unroll
                        for (int j=0;j<11;++j)
                            s_params[t*11 + j] = f0_params[a*11 + j];
                    }
                }
                __syncthreads();

                #pragma unroll 4
                for (int j = 0; j < CHUNK_SIZE; ++j) {
                    int a = base + j;
                    if (a >= nAtoms) break;

                    float ax = s_px[j];
                    float ay = s_py[j];
                    float az = s_pz[j];

                    float dx = tx - ax;
                    float dy = ty - ay;
                    float dz = tz - az;
                    float r_det = sqrtf(dx*dx + dy*dy + dz*dz);
                    if (!(r_det > 0.0f)) continue;

                    // +x incidence approximation
                    float dotv = dx / r_det;

                    float tmp = 2.0f * (1.0f - dotv);
                    if (tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_global * __fsqrt_rn(tmp);

                    const float* param_ptr = &s_params[j*11];
                    float f0v = get_f0_from_params(Q_val, param_ptr);

                    // (B) Build scattering factor and optionally remove full forward amplitude
                    float2 s_tot;
                    s_tot.x = f0v + s_anm[j].x;
                    s_tot.y = s_anm[j].y;

                    if (remove_forward && (Q_val < Q_cut)) {
                        // subtract (f0(0) + anomalous) -> leaves (f0(Q) - f0(0)), imag -> 0
                        s_tot.x -= (s_f0z[j] + s_anm[j].x);
                        s_tot.y -= (s_anm[j].y);
                    }

                    float2 amp = s_amp[j];
                    float real_part = amp.x * s_tot.x - amp.y * s_tot.y;
                    float imag_part = amp.x * s_tot.y + amp.y * s_tot.x;

                    float delta_r;

                    #if GLOBAL_USE_SERIES
                        if (R0 > 0.0f) {
                            float sproj = fmaf(uz, az, fmaf(uy, ay, ux*ax));
                            float a2    = fmaf(az, az, fmaf(ay, ay, ax*ax));
                            float tval  = -2.0f * sproj * invR0 + a2 * (invR0 * invR0);
                            delta_r = R0 * sqrt1pm1_series(tval);
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
                        float sc = __fsqrt_rn(P);
                        val.x *= sc; val.y *= sc;
                    }

                    // (A) Relative spherical-decay factor
                    float amp_rel = (R0 > 0.0f) ? (R0 / r_det) : 1.0f;

                    sum_rel.x += val.x * rE_F * amp_rel;
                    sum_rel.y += val.y * rE_F * amp_rel;
                }
                __syncthreads();
            }

            float2 sum_rot;
            sum_rot.x = sum_rel.x * cb - sum_rel.y * sb;
            sum_rot.y = sum_rel.x * sb + sum_rel.y * cb;

            detector_field[pidx].x += sum_rot.x;
            detector_field[pidx].y += sum_rot.y;
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
            )
        )
        kern = kernel_module.get_function('interaction_kernal')
        self._interaction_kernel_cache[key] = kern
        return kern
    
    @staticmethod
    def build_ein_sampler_kernel():
        """
        CUDA kernel: bilinearly sample tau, phi, and E0 on the beam grid; then
        write Ein = E0 * exp(-f*tau) * exp(i*f*phi) for a list of positions.

        This version zeros Ein for atoms that project outside the beam grid
        (no edge clamping when out-of-bounds).
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

                // Clamp neighbors to valid range (safe inside-grid interpolation)
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
    
    # -------------------------------------
    # Dynamical
    @staticmethod
    def build_intra_neighbor_search_kernel():
        """
        Build a CuPy RawKernel for intra-chunk neighbor search.

        This kernel finds neighbors of each atom within the same chunk and records
        for each neighbor:
            - phase = k_val * mod(distance, wavelength)
            - wave vector components (kx, ky, kz)

        Returns:
            cupy.RawKernel: Compiled kernel handle named "intra_neighbor_search_kernel".
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
            // 27 neighbor cell offsets for a 3x3x3 stencil
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

            // Cell index for atom i
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

                            // phase = k_val * mod(dist, wavelength)
                            float mod_val = fmodf(dist, wavelength);
                            float phase_val = k_val * mod_val;

                            // wave vector from i->j with magnitude k_val
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
        Build a CuPy RawKernel for inter-chunk neighbor search.

        The kernel finds neighbors across two boundary sets (chunk i and chunk j),
        records phase and (kx, ky, kz), and excludes i->i or j->j pairs.

        Returns:
            cupy.RawKernel: Compiled kernel handle named "inter_neighbor_search_kernel".
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
            // Neighbor offsets for 3x3x3 stencil
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

                            // phase and wave vector for i->j
                            float mod_val = fmodf(dist, wavelength);
                            float phase_val = k_val*mod_val;

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
        Build a CUDA kernel that expands scattering paths using neighbor data.

        The kernel performs, for each incoming path and each of its neighbors:
        - Multiply the amplitude by exp(i*phase) using the neighbor phase.
        - Multiply by a per-bounce complex factor s0[j] ~= f0(0) + f1 + i*f2 when
            the neighbor atom j is local to the chunk; otherwise use 1.
        - Write the neighbor atom position (meters) if local; write NaN and index
            -1 if not local, so the host can filter non-expandable paths.
        - Pass through the neighbor wave-vector components (neighborKx/Ky/Kz).
        - Write the neighbor species code for later lookups.

        Returns:
            cupy.RawKernel: Compiled kernel handle named "expand_paths_kernel".
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

            // Neighbor info (per-atom, flattened)
            const int*    neighborStart,
            const int*    neighborCount,
            const float*  neighborPhase,
            const float*  neighborKx,
            const float*  neighborKy,
            const float*  neighborKz,
            const int*    neighborIdxAtom,  // neighbor index j (may be out of chunk)
            const int*    neighborSpc,      // int32 species code per neighbor

            // Global size
            const int     numIncomingPaths,

            // Local-chunk lookups (positions in meters, s0 ~= f0(0)+anom)
            const float*  atom_x_m,
            const float*  atom_y_m,
            const float*  atom_z_m,
            const float2* s0_per_atom,      // length = nAtomsLocal
            const int     nAtomsLocal,

            // Outputs (capacity = maxPaths)
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

            float2 AmpIn = in_amp[idx];
            int    src   = in_atomIndex[idx];

            int startN = neighborStart[src];
            int countN = neighborCount[src];

            for (int n = 0; n < countN; ++n) {
                int gN = startN + n;

                // Phase and per-bounce scatter
                float  ph  = neighborPhase[gN];
                float2 eip = cplx_expf(ph);

                // Multiply by exp(i*phase)
                float2 A1;
                A1.x = AmpIn.x * eip.x - AmpIn.y * eip.y;
                A1.y = AmpIn.x * eip.y + AmpIn.y * eip.x;

                // Neighbor atom index
                int j = neighborIdxAtom[gN];

                // Multiply by s0[j] if local; otherwise s0 = 1
                float2 s0 = make_float2(1.f, 0.f);
                if (j >= 0 && j < nAtomsLocal) {
                    s0 = s0_per_atom[j];
                }

                float2 A2;
                A2.x = A1.x * s0.x - A1.y * s0.y;
                A2.y = A1.x * s0.y + A1.y * s0.x;

                // Append to output buffer
                int outPos = atomicAdd((unsigned int*)&out_atomIndex[maxPaths], 1);
                if (outPos < maxPaths) {
                    // Write neighbor position (meters) if local, else mark invalid
                    if (j >= 0 && j < nAtomsLocal) {
                        out_x[outPos] = atom_x_m[j];
                        out_y[outPos] = atom_y_m[j];
                        out_z[outPos] = atom_z_m[j];
                        out_atomIndex[outPos] = j;
                    } else {
                        float nanv = __int_as_float(0x7fffffff); // qNaN bit pattern
                        out_x[outPos] = nanv;
                        out_y[outPos] = nanv;
                        out_z[outPos] = nanv;
                        out_atomIndex[outPos] = -1;
                    }

                    // Carry neighbor direction (units as provided)
                    out_kx[outPos] = neighborKx[gN];
                    out_ky[outPos] = neighborKy[gN];
                    out_kz[outPos] = neighborKz[gN];

                    out_amp[outPos] = A2;
                    out_spc[outPos] = neighborSpc[gN];
                }
            }
        }
        } // extern "C"
        '''

        mod = cp.RawModule(
            code=code,
            options=('-O3', '--ftz=true', '--fmad=true'),
            backend='nvcc'
        )
        return mod.get_function('expand_paths_kernel')
    
    @staticmethod
    def _next_pow_two(n) -> int:
        """
        Return the next power of two greater than or equal to n.

        Accepts int, float, or NumPy scalar inputs. Works for n up to 2**63 - 1.

        Args:
            n (int or float): Input value.

        Returns:
            int: The next power of two >= n. For n < 1, returns 1.
        """
        # Ensure a Python int
        n_int = int(np.ceil(n))
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
        Compute symmetric padding sizes for angular spectrum propagation so that
        wrap-around is avoided after distance |z|.

        Model:
            The largest propagating angles supported by sampling are
                sin(theta_x_max) = min(1, lambda / (2*dx))
                sin(theta_y_max) = min(1, lambda / (2*dy))
            Then the required half-padding (meters) is
                pad_x = |z| * tan(theta_x_max)
                pad_y = |z| * tan(theta_y_max).
            Convert to pixels, apply a safety factor, and optionally round up to
            powers of two.

        Args:
            Nx (int): Original x size.
            Ny (int): Original y size.
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            wavelength (float): Wavelength in meters.
            z (float): Propagation distance in meters.
            safety (float, optional): Multiplicative safety factor for padding.
            enforce_pow2 (bool, optional): If True, round padded sizes to next
                power of two.
            min_pad_factor (float, optional): Minimum multiplicative growth factor
                applied to Nx and Ny regardless of geometric padding.

        Returns:
            tuple[int, int]: Padded sizes (Nx_pad, Ny_pad).
        """
        zabs = abs(float(z))
        if zabs == 0.0:
            Nx2 = max(int(np.ceil(Nx * min_pad_factor)), Nx)
            Ny2 = max(int(np.ceil(Ny * min_pad_factor)), Ny)
            if enforce_pow2:
                Nx2 = beam._next_pow_two(Nx2)
                Ny2 = beam._next_pow_two(Ny2)
            return int(Nx2), int(Ny2)

        # Sampling-limited maximum angles
        srx = min(1.0, float(wavelength) / (2.0 * float(dx)))
        sry = min(1.0, float(wavelength) / (2.0 * float(dy)))
        # Avoid tan(pi/2) by clamping
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

        # Enforce a minimum multiplicative padding if requested
        Nx2 = max(Nx2, int(np.ceil(Nx * min_pad_factor)))
        Ny2 = max(Ny2, int(np.ceil(Ny * min_pad_factor)))

        if enforce_pow2:
            Nx2 = beam._next_pow_two(Nx2)
            Ny2 = beam._next_pow_two(Ny2)

        return int(Nx2), int(Ny2)
    
    @staticmethod
    def build_propagation_multiplier_kernel():
        """
        Build a CUDA kernel that multiplies a spectrum by the free-space propagator.

        For each spatial frequency (kx, ky), the kernel applies
            H = exp(+i*z*sqrt(k^2 - kt^2)) for propagating components (kt^2 <= k^2),
            H = exp(-|z|*sqrt(kt^2 - k^2)) for evanescent components (pure decay).

        The spectrum F is updated in place.

        Returns:
            cupy.RawKernel: Compiled kernel handle named "prop_mul_kernel".
        """
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

            // Build multiplier H(kt):
            // - exp(+i z sqrt(k^2 - kt^2)) for propagating components
            // - exp(-|z| sqrt(kt^2 - k^2)) for evanescent components (decay only)
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

        # Ensure ASCII-only source for compilation environments that are strict
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
        Build a CPU propagation multiplier via CFFI for angular spectrum steps.

        The compiled function multiplies a complex spectrum F (row-major Ny x Nx)
        by H(kx, ky, z) using the same definition as the CUDA version:
        - propagating:  H = exp(+i*z*sqrt(k^2 - kt^2))
        - evanescent:   H = exp(-|z|*sqrt(kt^2 - k^2))  (real decay)

        Returns:
            tuple: (ffi, lib) where lib.prop_mul_cpu(...) performs the in-place
            multiplication on a provided complex array.

        Notes:
            The C signature is:
                void prop_mul_cpu(
                    const int Nx,
                    const int Ny,
                    const float* kx,
                    const float* ky,
                    const float k,
                    const float z,
                    float _Complex* F);
        """
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
        Robust CuPy bincount with guards for empty inputs, NaNs, and out-of-range indices.

        This function mirrors numpy.bincount semantics but adds:
        - Early returns with zeros for empty or invalid inputs.
        - Filtering of non-finite indices.
        - Clipping to the valid index range [0, size).
        - Optional dtype control for the output array.

        Args:
            idxs (array-like or None): Index array. If None or empty, returns zeros.
            weights (array-like or None): Optional weights array. If None, counts
                occurrences; otherwise sums weights.
            size (int): Length of the output histogram.
            dtype (cupy.dtype or numpy.dtype or type, optional): Desired output dtype.
                If None and weights is None, defaults to float32 after bincount.

        Returns:
            cupy.ndarray: Histogram of length `size`.

        Raises:
            RuntimeError: If CuPy is not available.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for _safe_bincount_gpu")

        if size <= 0:
            return cp.zeros((0,), dtype=cp.float32 if dtype is None else dtype)

        # Handle None or empty index arrays
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

        # Choose dtype
        if dtype is None:
            dtype = (weights.dtype if weights is not None else cp.float32)

        # Compute histogram
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
        Precompute Ein for requested chunks using a streaming pipeline that keeps
        the GPU busy. Supports multi-GPU by sharding chunks across devices and
        using multiple CUDA streams per GPU for overlap.

        Env knobs (optional):
            BEAM_EIN_STREAMS_PER_GPU   : int, default 4   (concurrent streams per GPU)
            BEAM_EIN_SAVE_THREADS      : int, default 2   (threads for NPZ writes)
        """
        import hashlib, json, os, gc, threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Decide backend
        use_gpu = bool(use_gpu and (cp is not None))

        # Sanity
        if sample.chunk_total is None or int(sample.chunk_total) == 0:
            raise ValueError("No chunks to precompute Ein for.")

        # Determine which chunks to process
        if chunk_ids is None:
            chunk_ids = list(range(1, int(sample.chunk_total) + 1))
        else:
            chunk_ids = list(chunk_ids)

        # Depth bounds and beam maps
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Compute A(u,v) once (GPU if requested, else CPU)
        if use_gpu:
            A_beam_np = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius=kernel_radius)
        else:
            A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=kernel_radius)

        # Cache key and dir
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

        # Filter already-cached chunks unless forced to recompute
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

        # CPU fallback path (parallel NPZ saves)
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
                    pos = pos @ R_np
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

        # GPU streaming path
        # Prepare static maps on host once; each GPU will copy its own device copies.
        tau_host = (-np.log(np.abs(A_beam_np) + np.float32(1e-20))).astype(np.float32)
        phi_host = np.angle(A_beam_np).astype(np.float32)
        E0_host  = self._beam_E0_map.astype(np.complex64)
        e1_host  = self._beam_e1.astype(np.float32)
        e2_host  = self._beam_e2.astype(np.float32)
        khat_host= (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        R_host   = np.asarray(stage.rotation, dtype=np.float32)
        T_host   = np.asarray(stage.translation, dtype=np.float32)

        # Config
        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
        except Exception:
            n_gpus = 1
        n_gpus = max(1, n_gpus)
        streams_per_gpu = max(1, int(os.getenv("BEAM_EIN_STREAMS_PER_GPU", "4")))
        save_threads = max(1, int(os.getenv("BEAM_EIN_SAVE_THREADS", "6")))

        # Shard chunks across GPUs
        shards = [[] for _ in range(n_gpus)]
        for i, cid in enumerate(to_do):
            shards[i % n_gpus].append(cid)

        # Utility: async NPZ save (keep pinned mem alive until write completes)
        def _save_npz_keepalive(path, arr_view, pinned_mem):
            try:
                # Wrap in try so a failed compressed save falls back to uncompressed.
                np.savez_compressed(path, ein=np.asarray(arr_view, dtype=np.complex64))
            except Exception:
                np.savez(path, ein=np.asarray(arr_view, dtype=np.complex64))
            # When this function returns, references to arr_view and pinned_mem drop.

        def gpu_worker(dev_id, my_chunks):
            if not my_chunks:
                return
            cp.cuda.Device(dev_id).use()

            # Device copies of static maps
            tau_g = cp.asarray(tau_host)
            phi_g = cp.asarray(phi_host)
            E0_g  = cp.asarray(E0_host)
            e1g   = cp.asarray(e1_host)
            e2g   = cp.asarray(e2_host)
            khatg = cp.asarray(khat_host)
            Rg    = cp.asarray(R_host)
            Tg    = cp.asarray(T_host)

            # Build kernel once (cached on self)
            if getattr(self, "_ein_kernel", None) is None:
                self._ein_kernel = self.build_ein_sampler_kernel()

            # Streams and ring slots
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(streams_per_gpu)]
            # Per-slot state
            slot_event = [None] * streams_per_gpu
            slot_chunk = [None] * streams_per_gpu
            slot_devout= [None] * streams_per_gpu
            slot_host_mem = [None] * streams_per_gpu
            slot_host_view= [None] * streams_per_gpu

            # Thread pool for saving
            saver = ThreadPoolExecutor(max_workers=save_threads)
            save_futs = []

            # Helper: flush finished slot (wait, then schedule save)
            def flush_slot(idx, cache_dir_local):
                ev = slot_event[idx]
                if ev is None:
                    return
                # Wait for D2H to complete
                ev.synchronize()

                # Kick off NPZ save while GPU continues
                cid = slot_chunk[idx]
                path = os.path.join(cache_dir_local, f"ein_chunk_{cid}_{key_hash}.npz")
                hv = slot_host_view[idx]
                pm = slot_host_mem[idx]
                save_futs.append(saver.submit(_save_npz_keepalive, path, hv, pm))

                # Clear slot
                slot_event[idx] = None
                slot_chunk[idx] = None
                slot_devout[idx]= None
                slot_host_mem[idx] = None
                slot_host_view[idx]= None

            # Main loop
            for n, cid in enumerate(my_chunks):
                s_id = n % streams_per_gpu
                st = streams[s_id]

                # If this slot is busy, flush it now (waits only for its own event)
                if slot_event[s_id] is not None:
                    flush_slot(s_id, cache_dir)

                # Load positions
                pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)
                if pos.size == 0:
                    # No atoms: write empty cache immediately (no GPU work)
                    empty_path = os.path.join(cache_dir, f"ein_chunk_{cid}_{key_hash}.npz")
                    try:
                        np.savez_compressed(empty_path, ein=np.zeros((0,), np.complex64))
                    except Exception:
                        np.savez(empty_path, ein=np.zeros((0,), np.complex64))
                    continue

                # H2D copy and transform on this stream
                with st:
                    pos_g = cp.asarray(pos)
                    pos_g = pos_g @ Rg
                    pos_g += Tg

                    # Compute Ein on device
                    ein_g = self._ein_for_positions_gpu_fast(
                        pos_g=pos_g,
                        tau_g=tau_g, phi_g=phi_g, E0_g=E0_g,
                        e1g=e1g, e2g=e2g, khat_g=khatg,
                        s_min=np.float32(s_min), s_max=np.float32(s_max),
                        stream=st
                    )

                    # Async D2H into pinned host buffer
                    nbytes = int(ein_g.size) * 8  # complex64
                    pmem = cp.cuda.alloc_pinned_memory(nbytes)
                    # Create a numpy view into pinned memory (kept alive in slot)
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

                # Stash slot state
                slot_event[s_id] = ev
                slot_chunk[s_id] = cid
                slot_devout[s_id]= ein_g
                slot_host_mem[s_id] = pmem
                slot_host_view[s_id]= h_view

                # Release local refs quickly
                del pos, pos_g

            # Flush any remaining in-flight slots
            for s_id in range(streams_per_gpu):
                if slot_event[s_id] is not None:
                    flush_slot(s_id, cache_dir)

            # Wait for all saves to complete
            for f in as_completed(save_futs):
                _ = f.result()
            saver.shutdown(wait=True)

            # Clean up device allocations on this worker
            del tau_g, phi_g, E0_g, e1g, e2g, khatg, Rg, Tg
            for st in streams:
                st.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Launch one thread per GPU
        threads = []
        for dev_id in range(n_gpus):
            t = threading.Thread(target=gpu_worker, args=(dev_id, shards[dev_id]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        cp.get_default_memory_pool().free_all_blocks()
        
        return cache_dir, key_hash
    
    def _compute_global_depth_bounds(self, sample, stage):
        """
        Compute global front-to-back bounds along the beam direction in angstrom.

        For each chunk, atom positions are rotated and translated by the stage,
        then projected onto the unit beam direction. The global minimum and
        maximum along that axis are returned.

        Args:
            sample: Object providing chunk_total and load_chunk_positions(cid, use_gpu=False).
            stage: Object providing rotation (3x3) and translation (3,) arrays.

        Returns:
            tuple[float, float]: (s_min, s_max) in angstrom. If no valid data is
            found, returns (0.0, 1.0) as a safe fallback.
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
            # Transform positions by the stage
            pos = pos @ R
            pos += T
            # Project onto beam direction
            s_vals = pos @ k_hat
            cur_min = np.min(s_vals)
            cur_max = np.max(s_vals)
            if cur_min < s_min: s_min = cur_min
            if cur_max > s_max: s_max = cur_max

        if not np.isfinite(s_min) or (s_max <= s_min):
            # Fallback to a harmless span if bounds are degenerate
            return 0.0, 1.0
        return float(s_min), float(s_max)

    def cpu_scatter_chunk_cffi(self, complied_code, ffi_obj, chunk_id, sample,
                            Nx, Ny, coords_x_m, coords_y_m, coords_z_m,
                            db_dict_f0_all, db_dict_f1f2_all, k_val,
                            stage, detector=None, remove_forward_component=False,
                            initial_amp_complex=None,
                            apply_polarization=False,
                            apply_spherical_decay=True):
        """
        Compute kinematic scattering on CPU for a single chunk using the CFFI kernel.

        This function prepares per-atom parameters (f0 params, anomalous f1/f2,
        initial amplitudes), applies the stage transform, converts units to meters,
        and invokes the compiled C routine to accumulate the complex field.

        Args:
            complied_code: CFFI-verified module exposing compute_scattering_cffi.
            ffi_obj: CFFI FFI object used to cast pointers.
            chunk_id (int): Chunk identifier to process.
            sample: Object with load_chunk_species and load_chunk_positions helpers.
            Nx (int): Detector width in pixels.
            Ny (int): Detector height in pixels.
            coords_x_m (np.ndarray): Flattened detector x coordinates (meters), length Nx*Ny.
            coords_y_m (np.ndarray): Flattened detector y coordinates (meters), length Nx*Ny.
            coords_z_m (np.ndarray): Flattened detector z coordinates (meters), length Nx*Ny.
            db_dict_f0_all (dict): Map element -> f0 parameters (11,).
            db_dict_f1f2_all (dict): Map element -> table of [E, f1, f2].
            k_val (float): Wave number 2*pi/lambda in rad/m.
            stage: Object with rotation (3x3) and translation (3,) arrays.
            detector: Unused placeholder for API parity.
            remove_forward_component (bool): If True, subtract f0(0) from f0(Q).
            initial_amp_complex (np.ndarray or None): Optional per-atom entrance
                amplitudes as complex64. If None, uses ones.
            apply_polarization (bool): If True, apply polarization scaling inside
                the kernel using self._pol_perp_rate.

        Returns:
            np.ndarray: Complex64 array of shape (Ny, Nx) for this chunk.
        """
        species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
        atom_count = int(species_chunk_np.shape[0])
        if atom_count == 0:
            # Early return for empty chunks
            return np.zeros((Ny, Nx), dtype=np.complex64)

        # Allocate per-atom arrays
        scattering_anom_np_real = np.zeros(atom_count, dtype=np.float32)
        scattering_anom_np_imag = np.zeros(atom_count, dtype=np.float32)
        f0_params_np            = np.zeros((atom_count, 11), dtype=np.float32)
        f0_zero_np              = np.zeros((atom_count,), dtype=np.float32)

        # Precompute f0(0) by element
        f0_zero_dict = self._build_f0_zero_dict(db_dict_f0_all)
        unique_elements = pd.unique(species_chunk_np)
        for el in unique_elements:
            el = str(el)
            if el not in db_dict_f0_all:
                continue
            mask = (species_chunk_np == el)
            # Anomalous term at this beam energy
            table = db_dict_f1f2_all.get(el, None)
            if table is not None:
                cplx = self.get_f1f2_from_params(self._energy, table)
                scattering_anom_np_real[mask] = float(cplx.real)
                scattering_anom_np_imag[mask] = float(cplx.imag)
            # f0 parameters and f0(0)
            f0_params_np[mask] = db_dict_f0_all[el]
            f0_zero_np[mask]   = float(f0_zero_dict.get(el, 0.0))

        # Stage transform, convert to meters for the C kernel
        positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
        positions_chunk = positions_chunk @ stage.rotation
        positions_chunk += stage.translation
        positions_chunk_m = positions_chunk / 1e10

        # Initial complex amplitudes per atom
        if initial_amp_complex is None:
            amp_r = np.ones((atom_count,), dtype=np.float32)
            amp_i = np.zeros((atom_count,), dtype=np.float32)
        else:
            amp_r = np.asarray(np.real(initial_amp_complex), dtype=np.float32, order='C')
            amp_i = np.asarray(np.imag(initial_amp_complex), dtype=np.float32, order='C')
            if amp_r.shape[0] != atom_count:
                raise ValueError(f"initial_amp_complex size mismatch for chunk {chunk_id}")

        # Output accumulators (real and imaginary parts)
        out_r = np.zeros(Nx*Ny, dtype=np.float32)
        out_i = np.zeros(Nx*Ny, dtype=np.float32)

        # Ensure contiguous arrays before passing to C
        positions_chunk_m = np.ascontiguousarray(positions_chunk_m)
        f0_params_np      = np.ascontiguousarray(f0_params_np)
        f0_zero_np        = np.ascontiguousarray(f0_zero_np)
        s_anom_r          = np.ascontiguousarray(scattering_anom_np_real)
        s_anom_i          = np.ascontiguousarray(scattering_anom_np_imag)
        amp_r             = np.ascontiguousarray(amp_r)
        amp_i             = np.ascontiguousarray(amp_i)

        # Cast pointers for the C call
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

        # Invoke the compiled C kernel
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
            int(1 if apply_polarization else 0),
            int(1 if apply_spherical_decay else 0),
            float(self._pol_perp_rate),
            out_r_ptr, out_i_ptr
        )

        # Return complex field reshaped to (Ny, Nx)
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
        recompute_cache=False,
        apply_polarization=False,
        apply_spherical_decay=True
    ):
        """
        CPU path for kinematic scattering. Adds apply_spherical_decay to toggle 1/R.
        """
        import hashlib, json, os
        Nx, Ny = measurement_shape

        # Load scattering databases
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        # Wave number
        k_val = np.float32(2.0 * np.pi / self._wavelength)

        # Ensure detector coordinates are on CPU and in meters
        if cp is not None and isinstance(measurement_positions, cp.ndarray):
            measurement_positions = measurement_positions.get()
        coords_x_m = np.ascontiguousarray(measurement_positions[0, :].astype(np.float32) / 1e10)
        coords_y_m = np.ascontiguousarray(measurement_positions[1, :].astype(np.float32) / 1e10)
        coords_z_m = np.ascontiguousarray(measurement_positions[2, :].astype(np.float32) / 1e10)

        # Handle empty samples
        chunk_total = int(sample.chunk_total or 0)
        if chunk_total == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)

        # Compute global depth bounds once (used for Ein and E0 sampling)
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Ensure Ein caches exist if requested
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
                # Use GPU for precompute if available; otherwise CPU
                self.precompute_depth_ein_all_chunks(
                    sample, stage,
                    use_gpu=(cp is not None),
                    ein_cache_dir=cache_dir,
                    recompute_cache=recompute_cache,
                    kernel_radius=0,
                    chunk_ids=missing
                )

        # Compile the CPU CFFI kernel
        ffi_obj, complied_code = self.compile_compute_scattering_cffi()

        # Threaded loop over chunks
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_threads = min(chunk_total, multiprocessing.cpu_count())

        # Static beam maps and basis for E0 sampling when depth_ein is False
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
                return np.zeros((Ny, Nx), dtype=np.complex64)

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

            positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
            positions_chunk = positions_chunk @ stage.rotation
            positions_chunk += stage.translation
            positions_chunk_m = positions_chunk / 1e10

            if use_depth_ein:
                cache_path = os.path.join(cache_dir, f"ein_chunk_{chunk_id}_{key_hash}.npz")
                with np.load(cache_path) as npz:
                    init_amp = npz["ein"].astype(np.complex64)
            else:
                # Sample the incident beam E0(u,v); zero outside beam grid
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
                complied_code, ffi_obj, chunk_id, sample, Nx, Ny,
                coords_x_m, coords_y_m, coords_z_m,
                db_dict_f0_all, db_dict_f1f2_all, k_val, stage,
                detector=None, remove_forward_component=remove_forward_component,
                initial_amp_complex=init_amp,
                apply_polarization=apply_polarization,
                apply_spherical_decay=apply_spherical_decay
            )
            return out

        final_result = np.zeros((Ny, Nx), dtype=np.complex64)
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
        GPU evaluator for Ein using the fused kernel.
        Returns: cupy.ndarray, shape (N,), dtype=complex64, on device.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for _ein_for_positions_gpu_fast")

        N = int(pos_g.shape[0])
        if N == 0:
            return cp.zeros((0,), dtype=cp.complex64)

        NyB, NzB = int(tau_g.shape[0]), int(tau_g.shape[1])

        kernel = getattr(self, "_ein_kernel", None)
        if kernel is None:
            kernel = self.build_ein_sampler_kernel()
            self._ein_kernel = kernel

        inv_du = cp.float32(1.0 / float(self._beam_du))
        inv_dv = cp.float32(1.0 / float(self._beam_dv))
        uc = cp.float32(self._beam_uc)
        vc = cp.float32(self._beam_vc)

        out = cp.empty((N,), dtype=cp.complex64)

        threads = 256
        blocks = (N + threads - 1) // threads
        blocks = min(max(blocks, 1), 65535)

        args = (
            pos_g.astype(cp.float32, copy=False).ravel(),
            np.int32(N),
            tau_g.ravel(),
            phi_g.ravel(),
            E0_g.ravel(),
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
        spherical_decay: bool = True
    ):
        """
        If use_depth_ein is False, initial amplitudes are sampled from the
        incident E0(u,v) on the GPU with Ein=0 outside the beam grid.
        """
        if cp is None:
            print("[beam] CuPy not installed, falling back to CPU.")
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

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            print("[beam] No GPUs found, falling back to CPU.")
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

        import hashlib, json, os
        print(f"[beam] Found {n_gpus} GPU(s).")

        class _TmpDet:
            def __init__(self, pix): self.pixel_coordinates = pix
        _det_for_mode = _TmpDet(measurement_positions)

        if not hasattr(self, "_phase_tol_rad"):
            self._phase_tol_rad = 1e-2
        self._select_series_mode_once(sample, _det_for_mode, safety_t_thresh=0.5, verbose=True)

        interaction_kernel = self.build_interaction_kernel(
            series_terms=self._series_terms,
            force_mode=("series" if self._global_use_series else "exact")
        )

        db_f0   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero = self._build_f0_zero_dict(db_f0)

        Nx, Ny = measurement_shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        x_coords = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        y_coords = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        z_coords = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        chunk_total = int(sample.chunk_total or 0)
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")
        if chunk_total == 0:
            return np.zeros((Ny, Nx), dtype=np.complex64)

        # Global depth bounds for Ein/E0 sampling
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Ensure Ein caches exist if requested
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
                print(f"[beam] Precomputing Ein for {len(missing)} chunk(s).")
                self.precompute_depth_ein_all_chunks(
                    sample, stage,
                    use_gpu=True,
                    ein_cache_dir=cache_dir,
                    recompute_cache=recompute_cache,
                    kernel_radius=0,
                    chunk_ids=missing
                )

        # Distribute chunks across devices
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus
        partial_results = [None] * n_gpus

        try:
            _streams_per_gpu = max(1, int(os.getenv("BEAM_STREAMS_PER_GPU", "3")))
        except Exception:
            _streams_per_gpu = 3

        # Ensure Ein kernel (used also for E0-only sampling)
        if getattr(self, "_ein_kernel", None) is None:
            self._ein_kernel = self.build_ein_sampler_kernel()

        def gpu_worker(gpu_id, chunk_indices, result_index):
            cp.cuda.Device(gpu_id).use()

            Rg = cp.asarray(R_pin, dtype=cp.float32)
            Tg = cp.asarray(T_pin, dtype=cp.float32)
            xg = cp.asarray(x_coords)
            yg = cp.asarray(y_coords)
            zg = cp.asarray(z_coords)

            # Static beam maps for E0/EIN sampling on this device
            E0_g  = cp.asarray(self._beam_E0_map.astype(np.complex64))
            tau_g = cp.zeros(E0_g.shape, dtype=cp.float32)
            phi_g = cp.zeros_like(tau_g)
            e1g   = cp.asarray(self._beam_e1.astype(np.float32))
            e2g   = cp.asarray(self._beam_e2.astype(np.float32))
            khatg = cp.asarray((self._direction / np.linalg.norm(self._direction)).astype(np.float32))

            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(_streams_per_gpu)]
            dfields = [cp.zeros((Nx * Ny,), dtype=cp.complex64) for _ in streams]

            block = (32, 16)
            grid  = ((Nx + block[0] - 1) // block[0],
                    (Ny + block[1] - 1) // block[1])

            for i, cidx in enumerate(chunk_indices):
                s_id = i % len(streams)
                streams[s_id].synchronize()

                spc = sample.load_chunk_species(cidx, use_gpu=False)
                nA = int(spc.shape[0])
                if nA == 0:
                    continue

                s_anom_host = np.zeros(nA, np.complex64)
                f0p_host    = np.zeros((nA, 11), np.float32)
                f0z_host    = np.zeros(nA, np.float32)
                for el in pd.unique(spc):
                    if el not in db_f0:
                        continue
                    m = (spc == el)
                    f0p_host[m] = db_f0[el]
                    f0z_host[m] = f0_zero.get(el, 0.0)
                    tbl = db_f1f2.get(el)
                    if tbl is not None:
                        s_anom_host[m] = self.get_f1f2_from_params(self._energy, tbl)

                with streams[s_id]:
                    # Positions to device, transform, then meters
                    pos = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                    pos = pos @ Rg
                    pos += Tg

                    if use_depth_ein:
                        # Load precomputed Ein
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

                    px = (pos[:, 0] / 1e10).astype(cp.float32)
                    py = (pos[:, 1] / 1e10).astype(cp.float32)
                    pz = (pos[:, 2] / 1e10).astype(cp.float32)

                    kx_cp = cp.full(nA, self._kx_scalar, dtype=cp.float32)
                    ky_cp = cp.full(nA, self._ky_scalar, dtype=cp.float32)
                    kz_cp = cp.full(nA, self._kz_scalar, dtype=cp.float32)

                    s_anom_cp     = cp.asarray(s_anom_host)
                    f0_params_cp  = cp.asarray(f0p_host)
                    f0_zero_cp    = cp.asarray(f0z_host)

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
                            dfields[s_id],
                            np.int32(Nx),
                            np.int32(Ny),
                            np.int32(1 if remove_forward else 0),
                            np.int32(1 if apply_polarization else 0),
                            np.int32(1 if spherical_decay else 0),
                            np.float32(self._pol_perp_rate)
                        ),
                        stream=streams[s_id]
                    )

            for st in streams:
                st.synchronize()

            dfield_total = dfields[0]
            for j in range(1, len(dfields)):
                dfield_total += dfields[j]

            partial_results[result_index] = dfield_total.reshape((Ny, Nx)).get()

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
        spherical_decay: bool = False
    ):
        """
        Compute kinematic atomic scattering (single bounce) and return the field.

        This is a high-level wrapper that routes to the GPU or CPU path. When
        `use_depth_ein` is True, a depth-dependent entrance amplitude is computed
        on the beam grid and interpolated at atom positions.

        Args:
            sample: Sample object that exposes chunk_total and per-chunk loaders.
            detector: Detector object with `shape` and `pixel_coordinates`.
            stage: Stage object with `rotation` (3x3) and `translation` (3,) arrays.
            offset (np.ndarray or None): Optional complex field to subtract from the
                final result. If provided, it must broadcast to (Ny, Nx).
            use_gpu (bool): If True and CuPy is available, use the GPU path.
            remove_forward (bool): If True, subtract f0(0) in the scattering kernel.
            use_depth_ein (bool): If True, compute depth-dependent entrance amplitude.
            ein_cache_dir (str or None): Directory for entrance-amplitude cache files.
            recompute_cache (bool): If True, recompute entrance-amplitude cache.
            apply_polarization (bool): If True, apply polarization scaling in kernel.

        Returns:
            np.ndarray: Complex64 array of shape (Ny, Nx) with the scattered field.
        """
        # Pull detector coordinates once (angstrom)
        measurement_positions = detector.pixel_coordinates
        Nx, Ny = detector.shape

        if use_gpu and (cp is not None):
            # GPU path
            final_field = self.interact_beam_gpu(
                sample,
                measurement_positions,
                (Nx, Ny),
                stage,
                remove_forward=remove_forward,
                use_depth_ein=use_depth_ein,
                ein_cache_dir=ein_cache_dir,
                recompute_cache=recompute_cache,
                apply_polarization=apply_polarization,
                spherical_decay=spherical_decay
            )
        else:
            # CPU path; optionally warn if GPU was requested but not available
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
                recompute_cache=recompute_cache,
                apply_polarization=apply_polarization,
                apply_spherical_decay=spherical_decay
            )

        # Optional offset subtraction
        return (final_field - offset) if (offset is not None) else final_field
    # -------------------------------------
        
    # -------------------------------------
    # Direct transmission
    def _compute_beam_column_A_map_cpu(self, sample, stage, kernel_radius=0):
        """
        Compute A(u,v) = exp(-tau + i*phi) on the beam grid (CPU).

        Only atoms whose (iu,iv) are inside [0..NyB-1]x[0..NzB-1] contribute.
        This prevents out-of-beam atoms from contributing to transmission.
        """
        import numpy as _np

        # Constants (angstrom)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10

        # Beam-grid geometry
        du, dv = float(self._beam_du), float(self._beam_dv)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        A_pix_A2 = du * dv

        # Accumulate column sums of forward factors (real and imag parts)
        sum_real = _np.zeros((NyB, NzB), _np.float32)  # sum of f0(0)+f1
        sum_imag = _np.zeros((NyB, NzB), _np.float32)  # sum of f2

        # Databases
        f1f2_dict      = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict   = self._build_f0_zero_dict(f0_params_dict)

        e1 = self._beam_e1
        e2 = self._beam_e2

        def _tsc_w(d):
            # 1D TSC weights for distances in pixel units (centered on integer indices)
            w = _np.zeros_like(d, dtype=_np.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0]*d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        for cid in range(1, sample.chunk_total + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(_np.float32)  # Angstrom
            if pos.size == 0:
                continue

            # Stage transform in real space (angstrom)
            pos = pos @ stage.rotation
            pos += stage.translation

            nA = pos.shape[0]
            f1  = _np.zeros(nA, _np.float32)
            f2  = _np.zeros(nA, _np.float32)
            f0z = _np.zeros(nA, _np.float32)
            for el in _np.unique(spc):
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
            if not _np.any(inb):
                continue

            iu = iu[inb]; iv = iv[inb]
            fr = (f0z[inb] + f1[inb]).astype(_np.float32)  # real forward factor
            fi = (f2[inb]).astype(_np.float32)             # imag forward factor

            ic = _np.floor(iu + 0.5).astype(_np.int64)
            jc = _np.floor(iv + 0.5).astype(_np.int64)

            du_m1 = _np.abs(iu - (ic - 1)); du_0 = _np.abs(iu - ic); du_p1 = _np.abs(iu - (ic + 1))
            dv_m1 = _np.abs(iv - (jc - 1)); dv_0 = _np.abs(iv - jc); dv_p1 = _np.abs(iv - (jc + 1))

            wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
            wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

            idx_list_R = []; w_list_R = []
            idx_list_I = []; w_list_I = []

            def _push(ii, jj, fac, val):
                mask = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                if not _np.any(mask):
                    return
                rows = ii[mask]; cols = jj[mask]
                idx  = (rows * NzB + cols).astype(_np.int64)
                w    = (val[mask] * fac[mask]).astype(_np.float32)
                return idx, w

            # 3x3 TSC deposition for both real and imaginary forward sums
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
                idxR = _np.concatenate(idx_list_R); wR = _np.concatenate(w_list_R)
                idxI = _np.concatenate(idx_list_I); wI = _np.concatenate(w_list_I)
                _np.add.at(sum_real.ravel(), idxR, wR)
                _np.add.at(sum_imag.ravel(), idxI, wI)

        # Convert forward sums -> total phase/attenuation (column integrals)
        # phi = -k*delta_int, tau = k*beta_int, with:
        # delta_int = (r_e * lambda^2 / (2*pi) / A_pix) * sum_real
        # beta_int  = (r_e * lambda^2 / (2*pi) / A_pix) * sum_imag
        two_pi = 2.0 * _np.pi
        C = (r_e_A * (lam_A * lam_A)) / (two_pi * A_pix_A2)  # dimensionless
        delta_int = C * sum_real.astype(_np.float32)
        beta_int  = C * sum_imag.astype(_np.float32)

        kA = two_pi / lam_A
        phi = (-kA * delta_int).astype(_np.float32)
        tau = ( kA * beta_int ).astype(_np.float32)

        # Numerical safety: never allow gain
        tau = _np.maximum(tau, _np.float32(0.0))

        # Optional blur (same as before)
        if kernel_radius > 0:
            rad = int(kernel_radius); sig = rad / 2.0
            y, x = _np.ogrid[-rad:rad+1, -rad:rad+1]
            k = _np.exp(-(x*x + y*y) / (2.0*sig*sig)).astype(_np.float32)
            k /= k.sum()
            Fk = _np.fft.fft2(k, s=phi.shape)
            phi = _np.fft.ifft2(_np.fft.fft2(phi) * Fk).real.astype(_np.float32)
            tau = _np.fft.ifft2(_np.fft.fft2(tau) * Fk).real.astype(_np.float32)
            tau = _np.maximum(tau, _np.float32(0.0))  # keep no-gain after blur

        A_map = _np.exp(-tau + 1j * phi).astype(_np.complex64)
        return A_map

    def _compute_beam_column_A_map_gpu(self, sample, stage, kernel_radius=0):
        """
        Compute A(u,v) = exp(-tau + i*phi) on the beam grid (GPU).

        Only atoms with (iu,iv) inside the beam grid contribute.
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

        # Databases
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
                posg = posg @ Rg; posg += Tg

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

        # Launch workers
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

        # Combine multiplicatively (independent chunk products)
        A_total = np.ones((NyB, NzB), np.complex64)
        for p in partial:
            if p is not None:
                A_total *= p
        return A_total
    
    def atomic_transmission(self, sample, detector, stage,
                            use_gpu=True, kernel_radius=0,
                            padding_mode: str = "edge",
                            pad_constant: float = 0.0):
        """
        Compute the transmitted field with propagation performed on the FULL
        DETECTOR GRID (NyD, NxD), not on the beam grid.

        Steps:
        1) Build A(u,v) on the beam grid and the exit-plane field E_exit(u,v).
        2) Bilinearly resample E_exit onto EVERY detector pixel; zero OOB.
        3) Decide the propagation distance from the sample exit plane to the
            detector plane using beam direction.
        4) If needed, propagate the full detector field using the detector's
            sampling (pixel_size if available; otherwise estimated from geometry).

        Returns:
        np.ndarray complex64 of shape (NyD, NxD)
        """
        # 1) A(u,v) on beam grid, possibly blurred
        if use_gpu and (cp is not None):
            A_beam = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius)
        else:
            A_beam = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        # Exit-plane field on the beam grid (NyB, NzB)
        E_exit = (self._beam_E0_map * A_beam).astype(np.complex64)
        NyB, NzB = E_exit.shape
        du_A = float(self._beam_du)  # Angstrom (row axis corresponds to u)
        dv_A = float(self._beam_dv)  # Angstrom (col axis corresponds to v)

        # 2) Resample E_exit to the detector grid (before propagation)
        NyD, NxD = detector.shape
        pix = detector.pixel_coordinates

        # Build u,v for each detector pixel
        if use_gpu and (cp is not None):
            pix_g = pix if isinstance(pix, cp.ndarray) else cp.asarray(pix)
            e1g = cp.asarray(self._beam_e1)
            e2g = cp.asarray(self._beam_e2)

            # transverse coordinates of each detector pixel
            u = pix_g[0] * e1g[0] + pix_g[1] * e1g[1] + pix_g[2] * e1g[2]
            v = pix_g[0] * e2g[0] + pix_g[1] * e2g[1] + pix_g[2] * e2g[2]

            # beam-grid fractional indices
            iu = u / cp.float32(du_A) + cp.float32(self._beam_uc)
            iv = v / cp.float32(dv_A) + cp.float32(self._beam_vc)

            # in-bounds mask
            mask = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))

            # neighbors and weights
            i0 = cp.floor(iu).astype(cp.int64); j0 = cp.floor(iv).astype(cp.int64)
            i1 = i0 + 1; j1 = j0 + 1
            i0 = cp.clip(i0, 0, NyB - 1); i1 = cp.clip(i1, 0, NyB - 1)
            j0 = cp.clip(j0, 0, NzB - 1); j1 = cp.clip(j1, 0, NzB - 1)
            fu = (iu - i0).astype(cp.float32); fv = (iv - j0).astype(cp.float32)

            Eb = E_exit if isinstance(E_exit, cp.ndarray) else cp.asarray(E_exit)
            idx00 = (i0 * NzB + j0).astype(cp.int64)
            idx01 = (i0 * NzB + j1).astype(cp.int64)
            idx10 = (i1 * NzB + j0).astype(cp.int64)
            idx11 = (i1 * NzB + j1).astype(cp.int64)

            E00 = Eb.ravel()[idx00]
            E01 = Eb.ravel()[idx01]
            E10 = Eb.ravel()[idx10]
            E11 = Eb.ravel()[idx11]

            one = cp.float32(1.0)
            E_det_exit = (E00 * (one - fu) * (one - fv) +
                        E01 * (one - fu) * fv +
                        E10 * fu * (one - fv) +
                        E11 * fu * fv).astype(cp.complex64)

            # zero out-of-bounds
            E_det_exit = cp.where(mask, E_det_exit, cp.complex64(0.0 + 0.0j))
            E_det_exit = E_det_exit.reshape(NyD, NxD)

        else:
            pix_cpu = pix.get() if (cp is not None and isinstance(pix, cp.ndarray)) else np.asarray(pix)
            e1 = self._beam_e1; e2 = self._beam_e2

            u = pix_cpu[0] * e1[0] + pix_cpu[1] * e1[1] + pix_cpu[2] * e1[2]
            v = pix_cpu[0] * e2[0] + pix_cpu[1] * e2[1] + pix_cpu[2] * e2[2]

            iu = u / du_A + self._beam_uc
            iv = v / dv_A + self._beam_vc

            mask = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))

            i0 = np.floor(iu).astype(np.int64); j0 = np.floor(iv).astype(np.int64)
            i1 = i0 + 1; j1 = j0 + 1
            i0 = np.clip(i0, 0, NyB - 1); i1 = np.clip(i1, 0, NyB - 1)
            j0 = np.clip(j0, 0, NzB - 1); j1 = np.clip(j1, 0, NzB - 1)
            fu = (iu - i0).astype(np.float32); fv = (iv - j0).astype(np.float32)

            Eb = (E_exit if isinstance(E_exit, np.ndarray) else np.asarray(E_exit)).ravel()
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
            E_det_exit = E_det_exit.reshape(NyD, NxD)

        # 3) Decide whether free-space propagation is needed, using beam direction
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        _, s_max = self._compute_global_depth_bounds(sample, stage)  # exit plane depth (Angstrom)

        if use_gpu and (cp is not None):
            pix_cpu = pix if isinstance(pix, np.ndarray) else pix.get()
        else:
            pix_cpu = pix_cpu if "pix_cpu" in locals() else np.asarray(pix)

        s_det = (pix_cpu[0, :] * k_hat[0] +
                pix_cpu[1, :] * k_hat[1] +
                pix_cpu[2, :] * k_hat[2]).astype(np.float64)

        s_det_min = float(np.min(s_det))
        s_det_max = float(np.max(s_det))
        s_det_mean = float(np.mean(s_det))
        plane_span_A = s_det_max - s_det_min
        tol_plane_A = max(1e-3, 1e-6 * abs(s_det_mean))
        tol_off_A = 1e-3

        need_propagation = False
        dz_A = s_det_mean - float(s_max)
        if plane_span_A <= tol_plane_A:
            need_propagation = (abs(dz_A) > tol_off_A)
        else:
            # Non-planar detector: propagate by mean offset
            need_propagation = (abs(dz_A) > tol_off_A)
            if need_propagation:
                print("[beam] atomic_transmission: detector appears non-planar "
                    "(Delta s range={:.3g} A). Propagating by mean Delta z={:.3g} A."
                    .format(plane_span_A, dz_A))

        if not need_propagation:
            # Return the exit field mapped to the detector without additional propagation
            return (E_det_exit.get() if (use_gpu and cp is not None and isinstance(E_det_exit, cp.ndarray))
                    else E_det_exit).astype(np.complex64)

        # 4) Propagate the FULL detector field by dz using detector sampling
        dz_m = float(dz_A) * 1e-10

        # Prefer detector.pixel_size if available; fallback to estimating from geometry
        def _estimate_dx_dy_from_uv(u_flat, v_flat, ny, nx):
            u_img = u_flat.reshape(ny, nx)
            v_img = v_flat.reshape(ny, nx)
            # dy corresponds to change of u across rows; dx corresponds to change of v across cols
            du_rows = np.abs(u_img[1:, :] - u_img[:-1, :]).ravel()
            dv_cols = np.abs(v_img[:, 1:] - v_img[:, :-1]).ravel()
            dy_A_est = float(np.median(du_rows)) if du_rows.size else 0.0
            dx_A_est = float(np.median(dv_cols)) if dv_cols.size else 0.0
            # Safety fallback
            if not np.isfinite(dy_A_est) or dy_A_est <= 0.0:
                dy_A_est = du_A
            if not np.isfinite(dx_A_est) or dx_A_est <= 0.0:
                dx_A_est = dv_A
            return dx_A_est * 1e-10, dy_A_est * 1e-10  # meters

        # Determine dx, dy (meters) for the detector grid
        have_psize = hasattr(detector, "pixel_size")
        dx_m = dy_m = None
        if have_psize:
            try:
                dy_A_ps, dx_A_ps = detector.pixel_size  # documented as (dy, dx) in Angstrom
                dy_m = float(dy_A_ps) * 1e-10
                dx_m = float(dx_A_ps) * 1e-10
            except Exception:
                dx_m = dy_m = None

        if (dx_m is None) or (dy_m is None) or (dx_m <= 0.0) or (dy_m <= 0.0):
            # Estimate from u,v geometry
            if use_gpu and (cp is not None):
                u_cpu = u.get() if isinstance(u, cp.ndarray) else np.asarray(u)
                v_cpu = v.get() if isinstance(v, cp.ndarray) else np.asarray(v)
                dx_m, dy_m = _estimate_dx_dy_from_uv(u_cpu, v_cpu, NyD, NxD)
            else:
                dx_m, dy_m = _estimate_dx_dy_from_uv(u, v, NyD, NxD)

        # Propagate on GPU or CPU over the FULL detector field
        if use_gpu and (cp is not None):
            kernel = self.build_propagation_multiplier_kernel()
            E_gpu = E_det_exit if isinstance(E_det_exit, cp.ndarray) else cp.asarray(E_det_exit)
            E_gpu = self._angular_spectrum_propagate_gpu(
                field=E_gpu, dx=dx_m, dy=dy_m, z=dz_m, kernel=kernel,
                step_max=0.02, pad_factor=1.0,
                padding_mode=padding_mode, pad_constant=pad_constant
            )
            return E_gpu.get().astype(np.complex64)
        else:
            ffi, lib = self.compile_propagation_multiplier_cffi()
            E_out = self._angular_spectrum_propagate_cpu(
                field=E_det_exit, dx=dx_m, dy=dy_m, z=dz_m, lib=lib, ffi=ffi,
                step_max=0.02, pad_factor=1.0,
                padding_mode=padding_mode, pad_constant=pad_constant
            )
            return E_out.astype(np.complex64)
    # -------------------------------------
    
    # -------------------------------------
    # Dynamical scattering
    def compute_intra_chunk_neighbors_gpu(
        self,
        sample,
        positions,           # cp.ndarray (N,3) in angstrom
        r_cut=5.0,
        max_neighbors_per_atom=32
    ):
        """
        Find intra-chunk nearest neighbors on GPU and return per-atom neighbor data.

        Positions are in angstrom. The function builds a cell list, runs a CUDA
        kernel to find neighbors within `r_cut`, and records for each atom:
        - phase values,
        - local wave-vector components (kx, ky, kz) in 1/angstrom,
        - neighbor indices.

        Args:
            sample: Sample object providing `build_cell_list_gpu`.
            positions (cupy.ndarray): Array of shape (N, 3) in angstrom.
            r_cut (float): Cutoff radius in angstrom.
            max_neighbors_per_atom (int): Maximum stored neighbors per atom.

        Returns:
            list: Length-N list. Each entry is a tuple
                (phase_array, kvec_3_array, neighbor_idx_array) where
                phase_array has dtype float32 and shape (M,),
                kvec_3_array has dtype float32 and shape (M, 3),
                neighbor_idx_array has dtype int32 and shape (M,).

        Notes:
            Wavelength and wavenumber are used in angstrom units for consistency.
        """
        N = positions.shape[0]
        if N == 0:
            # Return empty structures for empty input
            return [
                (np.array([], dtype=np.float32),
                np.zeros((0,3), dtype=np.float32),
                np.array([], dtype=np.int32))
                for _ in range(N)
            ]

        # 1) Build a cell list for neighbor search
        (sorted_positions,
        sorted_indices,
        cell_start,
        cell_end,
        box_min,
        cell_size,
        nx, ny, nz) = sample.build_cell_list_gpu(positions, r_cut)

        # 2) Allocate output buffers on GPU
        phase_gpu  = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        kx_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        ky_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        kz_gpu     = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.float32)
        idx_gpu    = cp.zeros((N*max_neighbors_per_atom,), dtype=cp.int32)
        counts_gpu = cp.zeros((N,), dtype=cp.int32)

        # ---- Angstrom units ----
        wavelength_A = self._wavelength * 1e10         # meters -> angstrom
        k_val_A      = (2.0 * np.pi) / wavelength_A    # 1/angstrom

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

        # 4) Convert to CPU ragged list-of-arrays
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
        pos_i,         # cp.ndarray (N_i,3) in angstrom
        pos_j,         # cp.ndarray (N_j,3) in angstrom
        r_cut,
        max_neighbors_per_atom=32
    ):
        """
        Find cross-chunk nearest neighbors on GPU and return per-atom neighbor data.

        This computes neighbors between two disjoint boundary sets (chunk i and
        chunk j). For each atom, it records phase, local wave-vector components,
        and neighbor indices for neighbors within the cutoff.

        Args:
            sample: Sample object providing `build_cell_list_gpu`.
            pos_i (cupy.ndarray): Positions for chunk i, shape (N_i, 3), in angstrom.
            pos_j (cupy.ndarray): Positions for chunk j, shape (N_j, 3), in angstrom.
            r_cut (float): Cutoff radius in angstrom.
            max_neighbors_per_atom (int): Maximum stored neighbors per atom.

        Returns:
            list: Length N_i + N_j. For each atom (in concatenated ordering),
                a tuple (phase_array, kvec_3_array, neighbor_idx_array) where:
                - phase_array: float32, shape (M,)
                - kvec_3_array: float32, shape (M, 3), units 1/angstrom
                - neighbor_idx_array: int32, shape (M,)

        Notes:
            Wavelength and wavenumber are computed in angstrom units to match inputs.
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

        # ---- Angstrom units ----
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
        Pass A: compute intra-chunk neighbors and encode species as int32 codes.

        For each chunk:
        * Encode species strings to contiguous int32 codes (GPU safe).
        * Run intra-chunk neighbor search to obtain (phase, kvec, idx) per atom.
        * Identify boundary atoms within `r_cut` of any face and collect their
            positions, original indices, and species codes for Pass B.

        Args:
            sample: Sample object with chunk accessors and GPU utilities.
            r_cut (float): Cutoff radius in angstrom.
            max_neighbors_per_atom (int): Maximum neighbors stored per atom.

        Returns:
            tuple: (boundary_dict, all_data_memory)
                boundary_dict: dict mapping chunk_id to:
                    "positions": cupy.ndarray (Nb, 3) in angstrom,
                    "indices": cupy.ndarray (Nb,) int32, original atom indices,
                    "species": numpy.ndarray (Nb,) int32 species codes.
                all_data_memory: dict mapping chunk_id to a list of length n_atoms
                    with entries (phase_arr, kvec_3, idx_arr, spc_codes).

        Notes:
            Requires CuPy. Species code maps are stored on `self` for reuse.
        """
        # Lazy species codec
        if not hasattr(self, "_species_code_map"):
            self._species_code_map = {}   # sym -> code (int)
            self._species_decode   = []   # code -> sym (list)

        def _sym_of(x):
            # Normalize species to a string
            if isinstance(x, (str, np.str_)):
                return str(x)
            if hasattr(sample, "get_symbol_from_id"):
                try:
                    return str(sample.get_symbol_from_id(int(x)))
                except Exception:
                    return str(x)
            return str(x)

        def _encode_species(arr):
            # Map species strings to contiguous int32 codes
            out = np.empty(arr.shape, dtype=np.int32)
            for i, v in enumerate(arr):
                sym = _sym_of(v)
                code = self._species_code_map.get(sym)
                if code is None:
                    code = len(self._species_decode)
                    self._species_code_map[sym] = code
                    self._species_decode.append(sym)
                out[i] = code
            return out

        boundary_dict   = {}
        all_data_memory = {}

        for cid in range(1, sample.chunk_total+1):
            chunk_positions = sample.load_chunk_positions(cid, use_gpu=True)
            chunk_species   = sample.load_chunk_species(cid, use_gpu=False)
            n_atoms = chunk_positions.shape[0]

            if n_atoms == 0:
                # Store empties and a boundary placeholder
                sample.write_chunk_nn_phase([], cid)
                sample.write_chunk_nn_scatter([], cid)
                sample.write_chunk_nn_indices([], cid)
                sample.write_chunk_nn_species([], cid)
                boundary_dict[cid] = {
                    "positions": cp.zeros((0,3), dtype=cp.float32),
                    "indices":   cp.zeros((0,),  dtype=cp.int32),
                    "species":   np.array([], dtype=np.int32)
                }
                all_data_memory[cid] = []
                continue

            # Encode species -> int32 codes
            chunk_codes = _encode_species(chunk_species)

            results_intra = self.compute_intra_chunk_neighbors_gpu(
                sample, chunk_positions, r_cut=r_cut,
                max_neighbors_per_atom=max_neighbors_per_atom
            )

            # Boundary set (angstrom)
            min_val = cp.min(chunk_positions, axis=0)
            max_val = cp.max(chunk_positions, axis=0)
            margin  = r_cut
            cond_min = cp.any((chunk_positions - min_val) < margin, axis=1)
            cond_max = cp.any((max_val - chunk_positions) < margin, axis=1)
            boundary_mask = (cond_min | cond_max)
            boundary_positions = chunk_positions[boundary_mask]
            boundary_indices   = cp.arange(n_atoms, dtype=cp.int32)[boundary_mask]
            boundary_mask_cpu  = boundary_mask.get()
            boundary_species   = chunk_codes[boundary_mask_cpu]  # codes

            phase_list    = []
            kvector_list  = []
            idx_list      = []
            species_list  = []
            results_intra_with_spc = [None] * n_atoms

            for i_atom, (ph, kvec_3, n_idx) in enumerate(results_intra):
                n_spc_codes = chunk_codes[n_idx]  # codes for neighbors

                phase_list.append(ph.astype(np.float32))
                kvector_list.append(kvec_3.astype(np.float32))
                idx_list.append(n_idx.astype(np.int32))
                species_list.append(n_spc_codes.astype(np.int32))

                results_intra_with_spc[i_atom] = (ph, kvec_3, n_idx, n_spc_codes)

            # Persist per-atom arrays for Pass A
            sample.write_chunk_nn_phase(phase_list, cid)
            sample.write_chunk_nn_scatter(kvector_list, cid)
            sample.write_chunk_nn_indices(idx_list, cid)
            sample.write_chunk_nn_species(species_list, cid)

            all_data_memory[cid] = results_intra_with_spc
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
        Pass B: add cross-chunk neighbors to per-atom lists using boundary sets.

        Each pair of chunks that may interact within `r_cut` is tested using a
        fast bounding-box check. If interaction is possible, a GPU cross-chunk
        neighbor search is performed and merged into the per-atom neighbor data.

        Args:
            sample: Sample object.
            boundary_dict (dict): Output of Pass A; per-chunk boundary sets.
            all_data_memory (dict): Output of Pass A; per-atom neighbor data to update.
            r_cut (float): Cutoff radius in angstrom.
            max_neighbors_per_atom (int): Maximum neighbors per atom used in kernels.

        Returns:
            dict: Updated `all_data_memory` with cross-chunk neighbors merged in.
        """
        # Build bounding boxes for quick rejection
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
            spc_i  = i_bd["species"]  # codes
            if pos_i.size == 0:
                continue
            N_i = pos_i.shape[0]
            min_i, max_i = chunk_bounds[i]

            for j in range(i+1, sample.chunk_total+1):
                j_bd   = boundary_dict[j]
                j_data = all_data_memory[j]
                pos_j  = j_bd["positions"]
                idx_j  = j_bd["indices"]
                spc_j  = j_bd["species"]  # codes
                if pos_j.size == 0:
                    continue
                N_j = pos_j.shape[0]
                min_j, max_j = chunk_bounds[j]

                if (min_i is None) or (min_j is None):
                    continue

                # Quick reject using expanded AABB test; wrap in bool(...) for CuPy
                bool_sep_ij = bool(((max_i + r_cut) < (min_j - r_cut)).any())
                bool_sep_ji = bool(((max_j + r_cut) < (min_i - r_cut)).any())
                if bool_sep_ij or bool_sep_ji:
                    continue

                cross_list = self.compute_inter_chunk_neighbors_gpu(
                    sample, pos_i, pos_j, r_cut=r_cut,
                    max_neighbors_per_atom=max_neighbors_per_atom
                )

                idx_i_cpu = idx_i.get()
                idx_j_cpu = idx_j.get()

                # Attach neighbors into i_data (codes preserved)
                for local_i in range(N_i):
                    (ph_new, kvec_new, idx_new) = cross_list[local_i]
                    if ph_new.size > 0:
                        global_i = idx_i_cpu[local_i]
                        (ph_old, kvec_old, idx_old, spc_old) = i_data[global_i]
                        spc_new = np.array([spc_j[n - N_i] if n >= N_i else spc_i[n]
                                            for n in idx_new], dtype=np.int32)
                        i_data[global_i] = (
                            np.concatenate([ph_old, ph_new]),
                            np.vstack([kvec_old, kvec_new]),
                            np.concatenate([idx_old, idx_new]),
                            np.concatenate([spc_old, spc_new])
                        )

                # Attach neighbors into j_data (codes preserved)
                for local_j in range(N_j):
                    (ph_new, kvec_new, idx_new) = cross_list[N_i + local_j]
                    if ph_new.size > 0:
                        global_j = idx_j_cpu[local_j]
                        (ph_old, kvec_old, idx_old, spc_old) = j_data[global_j]
                        spc_new = np.array([spc_i[n] if n < N_i else spc_j[n - N_i]
                                            for n in idx_new], dtype=np.int32)
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
        Compute nearest neighbors for all atoms using GPU (Passes A, B, C).

        Pass A: Intra-chunk neighbors and boundary sets.
        Pass B: Inter-chunk neighbors merged via boundary sets.
        Pass C: Persist final arrays (phase, kvec, idx, species code) to sample.

        Args:
            sample: Sample object with chunk accessors and writers.
            r_cut (float): Cutoff radius in angstrom.
            use_gpu (bool): Must be True. Raises if CuPy is not available.
            max_neighbors_per_atom (int): Maximum neighbors stored per atom.

        Returns:
            None
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
                                    sub_chunk_size=100_000,
                                    apply_polarization: bool = False):
        """
        Compute multi-bounce (dynamical) scattering on GPU and return the field.

        Paths are expanded up to `n_bounces` using precomputed neighbor lists, with
        per-bounce scattering factors applied. Contributions are accumulated on the
        detector grid. Polarization scaling can be enabled.

        Args:
            sample: Sample object with chunks and neighbor data on disk.
            detector: Detector object with `shape` and `pixel_coordinates`.
            stage: Stage object with `rotation` (3x3) and `translation` (3,) arrays.
            n_bounces (int): Number of secondary bounces to simulate (>= 0).
            offset (np.ndarray or None): Optional complex field to subtract at the end.
            use_gpu (bool): Must be True; raises if CuPy is not available.
            sub_chunk_size (int): Max number of expanded paths processed per batch.
            apply_polarization (bool): If True, apply polarization factor in kernels.

        Returns:
            np.ndarray: Complex64 array of shape (Ny, Nx) with the scattered field.
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

        db_f0   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        # Initialize species code maps if needed
        if not hasattr(self, "_species_code_map"):
            self._species_code_map = {}
            self._species_decode   = []

        def _ensure_codes_from(arr):
            # Ensure species code mapping for all labels in `arr`
            out = np.empty(arr.shape, dtype=np.int32)
            for i, v in enumerate(arr):
                if isinstance(v, (str, np.str_)):
                    sym = str(v)
                elif hasattr(sample, "get_symbol_from_id"):
                    try: sym = str(sample.get_symbol_from_id(int(v)))
                    except Exception: sym = str(v)
                else:
                    sym = str(v)
                code = self._species_code_map.get(sym)
                if code is None:
                    code = len(self._species_decode)
                    self._species_code_map[sym] = code
                    self._species_decode.append(sym)
                out[i] = code
            return out

        # Build LUTs for codes -> scattering parameters
        n_codes = len(self._species_decode)
        code_to_f0_params = np.zeros((max(1, n_codes), 11), np.float32)
        code_to_f0_zero   = np.zeros((max(1, n_codes),),   np.float32)
        code_to_anom      = np.zeros((max(1, n_codes),),   np.complex64)

        for code, sym in enumerate(self._species_decode):
            f0p = db_f0.get(sym)
            if f0p is not None:
                code_to_f0_params[code, :] = f0p
                code_to_f0_zero[code] = float(f0p[5] + f0p[0] + f0p[1] + f0p[2] + f0p[3] + f0p[4])
            tbl = db_f1f2.get(sym)
            if tbl is not None:
                code_to_anom[code] = self.get_f1f2_from_params(self._energy, tbl)

        Nx, Ny = detector.shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        mp = detector.pixel_coordinates
        px_pin = self.allocate_pinned_array(mp[0, :].astype(np.float32) / 1e10)
        py_pin = self.allocate_pinned_array(mp[1, :].astype(np.float32) / 1e10)
        pz_pin = self.allocate_pinned_array(mp[2, :].astype(np.float32) / 1e10)

        R_stage_pin = self.allocate_pinned_array(stage.rotation)
        T_stage_pin = self.allocate_pinned_array(stage.translation)

        chunk_per_gpu = chunk_total // n_gpus
        remainder     = chunk_total % n_gpus
        partial_results = [None] * n_gpus

        interaction_kernel = self.build_interaction_kernel()
        expand_kernel      = self.build_expand_paths_kernel()
        remove_forward_flag = 0  # keep anomalous term

        def gpu_worker(gpu_id, chunk_list, out_idx):
            cp.cuda.Device(gpu_id).use()

            Rg = cp.asarray(R_stage_pin, dtype=cp.float32)
            Tg = cp.asarray(T_stage_pin, dtype=cp.float32)
            pxg = cp.asarray(px_pin); pyg = cp.asarray(py_pin); pzg = cp.asarray(pz_pin)

            lut_f0p = cp.asarray(code_to_f0_params)
            lut_f0z = cp.asarray(code_to_f0_zero)
            lut_anm = cp.asarray(code_to_anom)

            dfield_gpu = cp.zeros((Nx * Ny,), dtype=cp.complex64)
            block2d = (16, 16)
            grid2d  = ((Nx + block2d[0] - 1) // block2d[0],
                    (Ny + block2d[1] - 1) // block2d[1])
            block1d = 256

            for cidx in chunk_list:
                spc_host = sample.load_chunk_species(cidx, use_gpu=False)
                nA = int(spc_host.shape[0])
                if nA == 0:
                    continue

                codes_host = _ensure_codes_from(spc_host)
                codes_gpu  = cp.asarray(codes_host, dtype=cp.int32)

                pos = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                pos = pos @ Rg; pos += Tg

                px_at = (pos[:, 0] / 1e10).astype(cp.float32)
                py_at = (pos[:, 1] / 1e10).astype(cp.float32)
                pz_at = (pos[:, 2] / 1e10).astype(cp.float32)

                f0z_gpu = lut_f0z[codes_gpu]
                anm_gpu = lut_anm[codes_gpu]
                s0_gpu  = (f0z_gpu + anm_gpu).astype(cp.complex64)

                # Bounce 0
                f0_params_gpu = lut_f0p[codes_gpu]
                anom_gpu0     = anm_gpu
                f0_zero_gpu   = f0z_gpu

                kx_atom_gpu = cp.full((nA,), self._kx_scalar, dtype=cp.float32)
                ky_atom_gpu = cp.full((nA,), self._ky_scalar, dtype=cp.float32)
                kz_atom_gpu = cp.full((nA,), self._kz_scalar, dtype=cp.float32)
                amp_atom_gpu= cp.ones((nA,), dtype=cp.complex64)

                # Updated signature: pass apply_spherical_decay=0
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(nA),
                        kx_atom_gpu, ky_atom_gpu, kz_atom_gpu,
                        px_at, py_at, pz_at,
                        amp_atom_gpu,
                        anom_gpu0,
                        f0_params_gpu,
                        f0_zero_gpu,
                        pxg, pyg, pzg,
                        dfield_gpu,
                        np.int32(Nx),
                        np.int32(Ny),
                        np.int32(remove_forward_flag),
                        np.int32(1 if apply_polarization else 0),
                        np.int32(0),
                        np.float32(self._pol_perp_rate)
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()

                if n_bounces < 1:
                    cp.get_default_memory_pool().free_all_blocks()
                    continue

                ph_flat,  offs_ph = sample.load_chunk_nn_phase(cidx)
                kx_flat, ky_flat, kz_flat, offs_kv = sample.load_chunk_nn_scatter(cidx)
                idx_flat, offs_ix = sample.load_chunk_nn_indices(cidx)
                spc_flat, offs_sp = sample.load_chunk_nn_species(cidx)

                neighborPhase_gpu = cp.asarray(ph_flat,  dtype=cp.float32)
                neighborKx_gpu    = cp.asarray(kx_flat,  dtype=cp.float32)
                neighborKy_gpu    = cp.asarray(ky_flat,  dtype=cp.float32)
                neighborKz_gpu    = cp.asarray(kz_flat,  dtype=cp.float32)
                neighborIdx_gpu   = cp.asarray(idx_flat, dtype=cp.int32)
                neighborSpc_gpu   = cp.asarray(spc_flat, dtype=cp.int32)

                neighborStart_gpu = cp.asarray(offs_ph[:-1].astype(np.int32))
                neighborCount_gpu = cp.asarray((offs_ph[1:] - offs_ph[:-1]).astype(np.int32))

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

                def process_subchunk(sbStart, sbEnd,
                                    out_x_gpu, out_y_gpu, out_z_gpu,
                                    out_kx_gpu, out_ky_gpu, out_kz_gpu,
                                    out_amp_gpu, out_idx_gpu, out_spc_gpu):
                    # Process a subset of newly expanded paths and accumulate on detector
                    sub_x  = out_x_gpu[sbStart:sbEnd]
                    sub_y  = out_y_gpu[sbStart:sbEnd]
                    sub_z  = out_z_gpu[sbStart:sbEnd]
                    sub_kx = out_kx_gpu[sbStart:sbEnd]
                    sub_ky = out_ky_gpu[sbStart:sbEnd]
                    sub_kz = out_kz_gpu[sbStart:sbEnd]
                    sub_amp= out_amp_gpu[sbStart:sbEnd]
                    sub_idx= out_idx_gpu[sbStart:sbEnd]
                    sub_spc= out_spc_gpu[sbStart:sbEnd]

                    valid = (sub_idx >= 0) & (sub_idx < nA)
                    if not bool(valid.any()):
                        return 0

                    sub_x   = sub_x[valid];   sub_y   = sub_y[valid];   sub_z   = sub_z[valid]
                    sub_kx  = sub_kx[valid];  sub_ky  = sub_ky[valid];  sub_kz  = sub_kz[valid]
                    sub_amp = sub_amp[valid]; sub_spc = sub_spc[valid]

                    f0p_paths = lut_f0p[sub_spc]
                    f0z_paths = lut_f0z[sub_spc]
                    anm_paths = lut_anm[sub_spc]

                    # Updated signature: pass apply_spherical_decay=0
                    interaction_kernel(
                        grid2d, block2d,
                        (
                            np.int32(int(sub_x.size)),
                            sub_kx, sub_ky, sub_kz,
                            sub_x,  sub_y,  sub_z,
                            sub_amp,
                            anm_paths,
                            f0p_paths,
                            f0z_paths,
                            pxg, pyg, pzg,
                            dfield_gpu,
                            np.int32(Nx),
                            np.int32(Ny),
                            np.int32(remove_forward_flag),
                            np.int32(1 if apply_polarization else 0),
                            np.int32(0),
                            np.float32(self._pol_perp_rate)
                        )
                    )
                    cp.cuda.stream.get_current_stream().synchronize()
                    return int(sub_x.size)

                for bounce_i in range(1, n_bounces + 1):
                    # Buffers for expanded paths (capacity limited by `expand_max`)
                    out_x_gpu   = cp.empty((expand_max,), dtype=cp.float32)
                    out_y_gpu   = cp.empty((expand_max,), dtype=cp.float32)
                    out_z_gpu   = cp.empty((expand_max,), dtype=cp.float32)
                    out_kx_gpu  = cp.empty((expand_max,), dtype=cp.float32)
                    out_ky_gpu  = cp.empty((expand_max,), dtype=cp.float32)
                    out_kz_gpu  = cp.empty((expand_max,), dtype=cp.float32)
                    out_amp_gpu = cp.empty((expand_max,), dtype=cp.complex64)
                    out_idx_gpu = cp.empty((expand_max + 1,), dtype=cp.int32)
                    out_spc_gpu = cp.empty((expand_max,), dtype=cp.int32)

                    # Last element holds the number of expansions written
                    out_idx_gpu[expand_max] = 0

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

                            px_at, py_at, pz_at,
                            s0_gpu, np.int32(nA),

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

                    # Accumulate in sub-batches
                    batchSize = expand_max
                    nSubBatches = (expansions_written + batchSize - 1) // batchSize
                    for sb in range(nSubBatches):
                        sbStart = sb * batchSize
                        sbEnd   = min(sbStart + batchSize, expansions_written)
                        _ = process_subchunk(sbStart, sbEnd,
                                            out_x_gpu, out_y_gpu, out_z_gpu,
                                            out_kx_gpu, out_ky_gpu, out_kz_gpu,
                                            out_amp_gpu, out_idx_gpu, out_spc_gpu)

                    # Prepare inputs for next bounce
                    if bounce_i < n_bounces:
                        valid_next = (out_idx_gpu[:expansions_written] >= 0) & \
                                    (out_idx_gpu[:expansions_written] < nA)
                        if not bool(valid_next.any()):
                            break
                        sel = valid_next.nonzero()[0]

                        in_x_gpu   = out_x_gpu[sel]
                        in_y_gpu   = out_y_gpu[sel]
                        in_z_gpu   = out_z_gpu[sel]
                        in_kx_gpu  = out_kx_gpu[sel]
                        in_ky_gpu  = out_ky_gpu[sel]
                        in_kz_gpu  = out_kz_gpu[sel]
                        in_amp_gpu = out_amp_gpu[sel]
                        in_idx_gpu = out_idx_gpu[sel]
                        cur_size   = int(sel.size)

                    cp.get_default_memory_pool().free_all_blocks()

                cp.get_default_memory_pool().free_all_blocks()

            partial_results[out_idx] = dfield_gpu.reshape((Ny, Nx)).get()
            del pxg, pyg, pzg, dfield_gpu
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

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
        High-level wrapper that combines scattering and transmission contributions.
        ### TODO: NEED TO BE UPDATED FOR NEW OPTIONS ###

        Behavior:
        * If `scattering` is True, calls `atomic_scattering_kinematic`.
        * If `transmission` is True, calls `atomic_transmission`.
        * Forward component removal in scattering is auto-toggled from the
            `transmission` flag to avoid double-counting the forward term.

        Args:
            sample: Sample object with chunk accessors.
            detector: Detector object with `shape`, `pixel_coordinates`,
                and `input_pixel_values(field)`.
            stage: Stage object with `rotation` (3x3) and `translation` (3,) arrays.
            scattering (bool): If True, include kinematic scattering term.
            scattering_params (list): Optional parameters for scattering:
                [offset, use_depth_ein]. Index 0 is a complex field offset (or None).
                Index 1 is a bool that enables depth-dependent entrance amplitude.
            transmission (bool): If True, include transmission term.
            transmission_params (list): Parameters for transmission:
                [kernel_radius]. Index 0 is Gaussian blur radius (pixels) for A(u,v).
            use_gpu (bool): If True and CuPy is available, use GPU code paths.

        Returns:
            None. The combined complex field is written back into `detector`.
        """
        Nx, Ny = detector.shape
        final_field = np.zeros((Ny, Nx), dtype=np.complex64)

        # Parse scattering params
        sc_offset = scattering_params[0] if (len(scattering_params) >= 1) else None
        use_depth_ein = scattering_params[1] if len(scattering_params) >= 2 else False

        if scattering:
            sc = self.atomic_scattering_kinematic(
                sample, detector, stage,
                offset=sc_offset, use_gpu=bool(use_gpu and (cp is not None)),
                remove_forward=bool(transmission),      # remove forward if also transmitting
                use_depth_ein=bool(use_depth_ein)
            )
            final_field += np.asarray(sc, dtype=np.complex64)

        if transmission:
            tx = self.atomic_transmission(
                sample, detector, stage,
                use_gpu=bool(use_gpu and (cp is not None)),
                kernel_radius=transmission_params[0]
            )
            final_field += np.asarray(tx, dtype=np.complex64)

        detector.input_pixel_values(final_field)
    # -------------------------------------
    
    # -------------------------------------
    # Wavefield propagation
    def _angular_spectrum_propagate_gpu(
            self, field, dx, dy, z, kernel,
            step_max=0.02, pad_factor=1.0,
            padding_mode: str = "edge",
            pad_constant: float = 0.0
        ):
        """
        Band-limited angular spectrum propagation on GPU with symmetric padding.

        The distance `z` is split into sub-steps if `abs(z) > step_max`, each
        applied in sequence. Padding is chosen based on sampling and `|z|` to
        limit wrap-around, then the spectrum is multiplied by a propagation
        transfer function on the GPU.

        Args:
            field (array-like): Complex field, shape (Ny, Nx). CuPy or NumPy.
            dx (float): Pixel size along x (meters).
            dy (float): Pixel size along y (meters).
            z (float): Propagation distance in meters (can be negative).
            kernel: Compiled CUDA kernel returned by `build_propagation_multiplier_kernel`.
            step_max (float): Maximum step size in meters. Longer distances are
                split into ceil(abs(z)/step_max) steps.
            pad_factor (float): Minimum multiplicative padding factor. Must be >= 1.0.
            padding_mode (str): "edge" to replicate edges, or "constant" to pad
                with a constant value.
            pad_constant (float): Value used when `padding_mode == "constant"`.

        Returns:
            cupy.ndarray: Complex64 field after propagation, cropped back to (Ny, Nx).

        Raises:
            RuntimeError: If CuPy is not available.

        Notes:
            Padding sizes are also rounded up to the next power of two for FFT speed.
        """
        if cp is None:
            raise RuntimeError('CuPy required for GPU propagation')

        # Split long distances into sub-steps
        z = float(z)
        if abs(z) > step_max:
            n = int(np.ceil(abs(z) / step_max))
            dz = z / n
            out = cp.asarray(field) if isinstance(field, cp.ndarray) else cp.asarray(field, dtype=cp.complex64)
            for _ in range(n):
                out = self._angular_spectrum_propagate_gpu(
                    out, dx, dy, dz, kernel,
                    step_max=step_max, pad_factor=pad_factor,
                    padding_mode=padding_mode, pad_constant=pad_constant
                )
            return out

        # Input sizes
        F0 = cp.asarray(field, dtype=cp.complex64)
        Ny, Nx = int(F0.shape[0]), int(F0.shape[1])

        # Choose symmetric padding from sampling and distance (also apply pad_factor)
        Nx2, Ny2 = self._choose_optimal_pad(
            Nx, Ny, float(dx), float(dy), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        y0 = (Ny2 - Ny) // 2
        x0 = (Nx2 - Nx) // 2

        # Configurable padding
        pmode = (padding_mode or "edge").lower()
        if pmode == "constant":
            Fp = cp.full((Ny2, Nx2), complex(pad_constant), dtype=cp.complex64)
            Fp[y0:y0+Ny, x0:x0+Nx] = F0
        else:
            # Default to "edge" replication
            pad_spec = ((y0, Ny2 - Ny - y0), (x0, Nx2 - Nx - x0))
            Fp = cp.pad(F0, pad_spec, mode='edge')

        # k-grids (rad/m), no shifts (fft2 uses non-shifted ordering)
        k  = 2.0 * np.pi / float(self._wavelength)
        kx = (2.0 * np.pi) * cp.fft.fftfreq(Nx2, d=float(dx)).astype(cp.float32)
        ky = (2.0 * np.pi) * cp.fft.fftfreq(Ny2, d=float(dy)).astype(cp.float32)

        # Forward FFT
        Fp = cp.fft.fft2(Fp)

        # Multiply by propagator in place via CUDA kernel
        block = (16, 16)
        grid  = ((Nx2 + block[0] - 1)//block[0],
                (Ny2 + block[1] - 1)//block[1])
        kernel(grid, block,
            (kx, ky, cp.float32(k), cp.float32(z),
                np.int32(Nx2), np.int32(Ny2), Fp))

        # Inverse FFT and center crop back to original size
        out = cp.fft.ifft2(Fp)
        return out[y0:y0+Ny, x0:x0+Nx]
    
    def _angular_spectrum_propagate_cpu(
            self, field, dx, dy, z, lib, ffi,
            step_max=0.02, pad_factor=1.0,
            padding_mode: str = "edge",
            pad_constant: float = 0.0
        ):
        """
        Band-limited angular spectrum propagation on CPU with symmetric padding.

        The distance `z` is split into sub-steps if `abs(z) > step_max`. Padding is
        chosen from sampling and `|z|`, rounded to a power of two for FFTs.

        Args:
            field (array-like): Complex field, shape (Ny, Nx). NumPy array preferred.
            dx (float): Pixel size along x (meters).
            dy (float): Pixel size along y (meters).
            z (float): Propagation distance in meters (can be negative).
            lib: CFFI-verified library with `prop_mul_cpu`.
            ffi: CFFI interface object.
            step_max (float): Maximum step size in meters for auto-splitting.
            pad_factor (float): Minimum multiplicative padding factor. Must be >= 1.0.
            padding_mode (str): "edge" or "constant".
            pad_constant (float): Value used when `padding_mode == "constant"`.

        Returns:
            np.ndarray: Complex64 field after propagation, cropped to (Ny, Nx).
        """
        z = float(z)
        if abs(z) > step_max:
            n = int(np.ceil(abs(z) / step_max))
            dz = z / n
            out = field
            for _ in range(n):
                out = self._angular_spectrum_propagate_cpu(
                    out, dx, dy, dz, lib, ffi,
                    step_max=step_max, pad_factor=pad_factor,
                    padding_mode=padding_mode, pad_constant=pad_constant
                )
            return out

        # Input (Ny, Nx)
        F0 = np.asarray(field, dtype=np.complex64, order='C')
        Ny, Nx = int(F0.shape[0]), int(F0.shape[1])

        # Choose symmetric padding and centers
        Nx2, Ny2 = self._choose_optimal_pad(
            Nx, Ny, float(dx), float(dy), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        y0 = (Ny2 - Ny) // 2
        x0 = (Nx2 - Nx) // 2

        # Configurable padding
        pmode = (padding_mode or "edge").lower()
        if pmode == "constant":
            Fp = np.full((Ny2, Nx2), pad_constant + 0j, dtype=np.complex64)
            Fp[y0:y0+Ny, x0:x0+Nx] = F0
        else:
            pad_spec = ((y0, Ny2 - Ny - y0), (x0, Nx2 - Nx - x0))
            Fp = np.pad(F0, pad_spec, mode='edge')

        # Spectral axes (rad/m)
        k  = np.float32(2.0 * np.pi / float(self._wavelength))
        kx = (2.0*np.pi) * np.fft.fftfreq(Nx2, d=float(dx)).astype(np.float32)
        ky = (2.0*np.pi) * np.fft.fftfreq(Ny2, d=float(dy)).astype(np.float32)

        # Forward FFT
        Fp = np.fft.fft2(Fp)

        # Multiply by propagator (CPU implementation)
        lib.prop_mul_cpu(
            np.int32(Nx2), np.int32(Ny2),
            ffi.cast('const float*', kx.ctypes.data),
            ffi.cast('const float*', ky.ctypes.data),
            k, np.float32(z),
            ffi.cast('float _Complex*', Fp.ctypes.data)
        )

        # Inverse FFT and center crop
        out = np.fft.ifft2(Fp)
        return out[y0:y0+Ny, x0:x0+Nx]

    def _apply_thin_lens_box(self, field, dx, dy, lens_data, use_gpu=True):
        """
        Apply a thin-lens phase and optional uniform absorption.

        The lens multiplies the field by exp(-i * k/(2f) * r^2). If
        `absorption_sigma` is provided, a uniform attenuation factor is applied.

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            lens_data (dict): Lens parameters:
                - 'focal_length' (float, mm)
                - 'thickness' (float, mm)
                - 'number' (int): number of identical lens elements
                - 'absorption_sigma' (float, meters) optional
            use_gpu (bool): If True and CuPy is available, use a GPU path.

        Returns:
            np.ndarray: Complex64 array with the lens applied (CPU path returns NumPy;
            GPU path returns NumPy after copying back).

        Notes:
            The lens is centered on the field center. Units: input focal length and
            thickness are in mm and converted to meters inside.
        """
        wavelength = self._wavelength
        k_val = 2.0 * np.pi / wavelength

        # Convert mm -> m
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
            # Build coordinate grids on GPU (meters from pixel indices)
            x_gpu = cp.asarray((x_arr - cx) * dx, dtype=cp.float32)
            y_gpu = cp.asarray((y_arr - cy) * dy, dtype=cp.float32)
            Xgpu = x_gpu[None, :].repeat(Ny, axis=0)
            Ygpu = y_gpu[:, None].repeat(Nx, axis=1)
            R2 = Xgpu * Xgpu + Ygpu * Ygpu

            # Thin lens phase
            phase_lens = -0.5 * (k_val / f) * R2
            cph = cp.cos(phase_lens)
            sph = cp.sin(phase_lens)

            F_gpu = cp.asarray(field, dtype=cp.complex64)
            real_part = F_gpu.real * cph - F_gpu.imag * sph
            imag_part = F_gpu.real * sph + F_gpu.imag * cph
            out = real_part + 1j * imag_part

            # Optional uniform absorption (N_lenses elements)
            if not cp.isinf(nsigma):
                out *= cp.exp(- N_lenses * t / nsigma)

            return out.get()

        # ---- CPU path (numpy only) ----
        # Precompute coordinate arrays in meters
        xx = (x_arr - cx) * dx
        yy = (y_arr - cy) * dy
        E_out = np.empty_like(field, dtype=np.complex64)

        # Apply lens phase per pixel
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

        # Optional uniform absorption
        if not np.isinf(nsigma):
            E_out *= np.exp(- N_lenses * t / nsigma)

        return E_out

    def _apply_aperture(self, field, dx, dy, aperture_data, use_gpu=True):
        """
        Apply a real-space aperture (square or circular) centered on the field.

        The aperture passes pixels within the specified width and zeros out the rest.
        Aperture width is specified in millimeters and converted to meters.

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            aperture_data (dict): Aperture specification:
                - 'shape': 'square' or 'circular'
                - 'width': float in millimeters
            use_gpu (bool): If True and CuPy is available, use GPU path.

        Returns:
            np.ndarray: Complex64 field with the aperture applied (NumPy array).
        """
        Nx, Ny = field.shape[1], field.shape[0]
        shape_type = aperture_data['shape'].lower()
        width_mm = aperture_data['width']
        width_m  = width_mm * 1e-3
        # Aperture extends from -w/2 .. +w/2 in x and y

        # Build coordinate arrays centered at field center
        x_arr = np.arange(Nx, dtype=np.float32) - (Nx-1)/2.0
        y_arr = np.arange(Ny, dtype=np.float32) - (Ny-1)/2.0
        x_arr *= dx
        y_arr *= dy
        half = 0.5*width_m

        if use_gpu and cp is not None:
            # Vectorized mask on GPU
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
                # Fallback to square if unknown shape
                mask = (cp.abs(Xgpu) <= half) & (cp.abs(Ygpu) <= half)

            F_gpu = cp.asarray(field, dtype=cp.complex64)
            F_gpu[~mask] = 0.0 + 0.0j
            return F_gpu.get()

        else:
            # CPU loop with simple mask checks
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
                        # Fallback to square
                        if (abs(xx) > half) or (abs(yy) > half):
                            E_out[iy, ix] = 0.0
            return E_out

    def wavefield_propagation(self, detector, optics,
                            use_gpu=True, step_max=0.02, pad_factor=1.0,
                            padding_mode: str = "edge",
                            pad_constant: float = 0.0):
        """
        Propagate the detector wavefield through an optics stack.

        Uses a band-limited angular spectrum method for all "free space" elements,
        and pointwise modifiers for "lens box" and "aperture" elements.

        Args:
            detector: Detector object with:
                - pixel_size: tuple (dy, dx) in Angstrom.
                - shape: tuple (Ny, Nx).
                - pixel_values: complex64 array of shape (Ny, Nx).
                - input_pixel_values(array): method to write the updated field.
            optics: Object with attribute `components`, a list of dicts.
                Supported element kinds and required keys:
                - "free space": {"kind": "free space", "length": mm}
                - "lens box": {"kind": "lens box", "focal_length": mm,
                                "thickness": mm, "number": int,
                                "absorption_sigma": meters (optional)}
                - "aperture": {"kind": "aperture", "shape": "square" or "circular",
                                "width": mm}
            use_gpu (bool): If True and CuPy is available, use GPU propagation.
            step_max (float): Maximum single propagation step in meters. Longer
                distances are split into sub-steps of size <= step_max.
            pad_factor (float): Minimum multiplicative padding used by the propagation
                helpers when choosing FFT sizes. Must be >= 1.0.
            padding_mode (str): Padding policy for propagation; "edge" replicates
                edge values, "constant" pads with a constant.
            pad_constant (float): Constant pad value when padding_mode == "constant".

        Returns:
            None. The updated complex field is written back into `detector` via
            `detector.input_pixel_values(...)`.

        Raises:
            ValueError: If an element with an unknown "kind" is encountered.

        Notes:
            detector.pixel_size is interpreted as (dy, dx) in Angstrom and converted
            to meters for propagation.
        """
        # Convert detector pixel sizes from Angstrom to meters.
        dy, dx = detector.pixel_size * 1e-10
        Ny, Nx = detector.shape
        E = detector.pixel_values  # complex64 (Ny, Nx)

        # Choose propagation backend: GPU kernel or CPU CFFI helper.
        if use_gpu and cp is not None:
            kernel = self.build_propagation_multiplier_kernel()
            ffi, lib = None, None
        else:
            kernel = None
            ffi, lib = self.compile_propagation_multiplier_cffi()

        # Walk the optics stack and apply each element in order.
        for elem in optics.components:
            kind = elem['kind'].lower()

            if kind == 'free space':
                # Convert millimeters to meters for propagation distance.
                z = float(elem['length']) * 1e-3
                if use_gpu and cp is not None:
                    E = self._angular_spectrum_propagate_gpu(
                            E, dx, dy, z, kernel,
                            step_max=step_max, pad_factor=pad_factor,
                            padding_mode=padding_mode, pad_constant=pad_constant
                        ).get()
                else:
                    E = self._angular_spectrum_propagate_cpu(
                            E, dx, dy, z, lib, ffi,
                            step_max=step_max, pad_factor=pad_factor,
                            padding_mode=padding_mode, pad_constant=pad_constant
                        )

            elif kind == 'lens box':
                # Apply thin-lens phase and optional uniform absorption via optics_stack
                E = optics._apply_thin_lens_box(
                        E, dx, dy, elem, self._wavelength, use_gpu and cp is not None
                    )
                
            elif kind == 'bragg magnifier 2b':
                # Anisotropic resample with amplitude/phase from the component dict.
                E = optics._apply_bragg_magnifier_2b(
                        E, dx, dy, elem, use_gpu and cp is not None
                    )
                
            elif kind == 'aperture':
                # Apply aperture via optics_stack (accepts 'shape' or 'type')
                E = optics._apply_aperture(
                        E, dx, dy, elem, use_gpu and cp is not None
                    )

            else:
                # Unknown element type: fail fast with the element kind in the message.
                raise ValueError(f'Unknown optics element "{kind}"')

        # Write back the final complex field to the detector.
        detector.input_pixel_values(E.astype(np.complex64))
    # -------------------------------------