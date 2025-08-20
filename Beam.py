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

        The compiled C function signature is:

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
            - Only comments and docstrings were edited. Computation is unchanged.
            - Requires a working C compiler through cffi.
        """
        from cffi import FFI

        c_source = r'''
        #include <math.h>
        #include <stddef.h>

        static inline float get_f0_value(float Q_val, const float* params)
        {
            const float PI_F = 3.14159265358979323846f;
            const float K_SCALE_FACTOR = 0.25f * 1.0e-10f / PI_F;  // 0.25 * Angstrom / pi
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

                    float dotv = (dx / r_det);  // cos(2*theta) approx for +x incidence
                    float tmp = 2.0f*(1.0f - dotv);
                    if (tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_val * sqrtf(tmp);

                    float f0_val = get_f0_value(Q_val, f0p);
                    if (remove_forward != 0) { f0_val -= f00; }

                    float s_re = (f0_val + sanr);
                    float s_im = (sani);

                    // multiply by complex entrance amplitude
                    float t_re = amp_r * s_re - amp_i * s_im;
                    float t_im = amp_r * s_im + amp_i * s_re;

                    float wavelength_m = (2.0f * PI_F) / k_val;  // derived from k
                    float phase = k_val * (fmodf(ax, wavelength_m) + fmodf(r_det, wavelength_m));
                    float cph = cosf(phase);
                    float sph = sinf(phase);

                    float val_r = (t_re * cph - t_im * sph) * rE_F;
                    float val_i = (t_re * sph + t_im * cph) * rE_F;

                    // polarization factor applied on amplitude
                    if (apply_pol) {
                        float P = pol_perp_rate + (1.0f - pol_perp_rate) * (dotv * dotv);
                        if (P < 0.0f) P = 0.0f;
                        if (P > 1.0f) P = 1.0f;
                        float scale = sqrtf(P);
                        val_r *= scale;
                        val_i *= scale;
                    }

                    out_r[p] += val_r;
                    out_i[p] += val_i;
                }
            }
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
    def build_interaction_kernel():
        """
        Drop-in replacement: FP32-only, numerically stable kinematic kernel.

        Key changes vs original:
        - Robust phase using delta-r expansion:
                delta_r = -dot(u,a) + 0.5*(|a|^2 - (dot(u,a))^2)/R0,
            where u = pixel / |pixel|, R0 = |pixel|.
        - Per-pixel base phasor exp(i*k*R0) applied once at the end.
        - No FP64 anywhere; relies on CUDA __sincosf for reduction.
        - Signature, name, and types identical to original for drop-in.

        Returns:
            cupy.RawKernel: compiled kernel handle named 'interaction_kernal'.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for GPU scattering kernels.")

        _cuda_source = r'''
        #include <math.h>

        // Compile-time tuning
        #define CHUNK_SIZE 128

        extern "C" {

        // Evaluate f0(Q) in FP32 (Waasmaier-Kirfel), identical math to your original
        __device__ __forceinline__ float get_f0_from_params(float Q_val, const float* params)
        {
            const float PI_F   = 3.14159265358979323846f;
            const float K_SCALE= 0.25f * 1.0e-10f / PI_F;  // Q[m^-1] -> s[Angstrom^-1]
            float s  = K_SCALE * Q_val;
            float ss = s * s;

            float f0 = params[5]; // c
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                float ai = params[i];
                float bi = params[6 + i];
                // use FMA-friendly exp
                f0 += ai * __expf(-bi * ss);
            }
            return f0;
        }

        // New kernel: same name and signature as before
        __global__ void interaction_kernal(
            const int   nAtoms,
            const float* __restrict__ kx_atom,
            const float* __restrict__ ky_atom,
            const float* __restrict__ kz_atom,
            const float* __restrict__ px,
            const float* __restrict__ py,
            const float* __restrict__ pz,
            const float2* __restrict__ initial_amp,
            const float2* __restrict__ scattering_anom,
            const float*  __restrict__ f0_params,
            const float*  __restrict__ f0_zero,
            const float* __restrict__ x_coords,
            const float* __restrict__ y_coords,
            const float* __restrict__ z_coords,
            float2*      __restrict__ detector_field,
            const int    Nx,
            const int    Ny,
            const int    remove_forward,
            const int    apply_polarization,
            const float  pol_perp_rate)
        {
            const float PI_F = 3.14159265358979323846f;
            const float rE_F = 2.81794092e-15f;  // classical electron radius [m]

            // Pixel index
            int ix = blockIdx.x * blockDim.x + threadIdx.x;
            int iy = blockIdx.y * blockDim.y + threadIdx.y;
            if (ix >= Nx || iy >= Ny) return;
            const int pidx = iy * Nx + ix;

            // Per-pixel detector coords (meters)
            float tx = x_coords[pidx];
            float ty = y_coords[pidx];
            float tz = z_coords[pidx];

            // Per-pixel reference distance and unit vector u = pixel / |pixel|
            // Add tiny eps to avoid div by zero (rare degenerate case)
            float R0  = sqrtf(tx*tx + ty*ty + tz*tz);
            if (!(R0 > 0.0f)) {
                // Degenerate pixel at origin: nothing to do
                return;
            }
            float invR0 = 1.0f / R0;
            float ux = tx * invR0;
            float uy = ty * invR0;
            float uz = tz * invR0;

            // Beam wavenumber k: use first entry (constant for a given beam)
            float k_global = 0.0f;
            if (nAtoms > 0) {
                k_global = fabsf(kx_atom[0]);   // your code stores 2*pi/lambda on +x
            } else {
                // nothing to accumulate
                return;
            }

            // Precompute base phasor exp(i*k*R0) once per pixel
            float s0, c0;
            __sincosf(k_global * R0, &s0, &c0);

            // Shared memory tile for per-atom data (float32)
            __shared__ float  s_px[CHUNK_SIZE];
            __shared__ float  s_py[CHUNK_SIZE];
            __shared__ float  s_pz[CHUNK_SIZE];
            __shared__ float2 s_amp[CHUNK_SIZE];
            __shared__ float2 s_anm[CHUNK_SIZE];
            __shared__ float  s_params[CHUNK_SIZE * 11];
            __shared__ float  s_f0z[CHUNK_SIZE];

            const int threads_in_block = blockDim.x * blockDim.y;
            const int t_id = threadIdx.y * blockDim.x + threadIdx.x;

            // Accumulator in the "relative" phase frame (no base phasor yet)
            float2 sum_rel = make_float2(0.0f, 0.0f);

            // Loop over atoms in tiles
            for (int base = 0; base < nAtoms; base += CHUNK_SIZE) {

                // Stage a tile into shared memory
                for (int t = t_id; t < CHUNK_SIZE; t += threads_in_block) {
                    int a = base + t;
                    if (a < nAtoms) {
                        s_px[t] = px[a];
                        s_py[t] = py[a];
                        s_pz[t] = pz[a];
                        s_amp[t]= initial_amp[a];
                        s_anm[t]= scattering_anom[a];
                        s_f0z[t]= f0_zero[a];
                        #pragma unroll
                        for (int j = 0; j < 11; ++j)
                            s_params[t*11 + j] = f0_params[a*11 + j];
                    }
                }
                __syncthreads();

                // Process the tile
                #pragma unroll 4
                for (int j = 0; j < CHUNK_SIZE; ++j) {
                    int a = base + j;
                    if (a >= nAtoms) break;

                    // Atom position in meters
                    float ax = s_px[j];
                    float ay = s_py[j];
                    float az = s_pz[j];

                    // Vector pixel->atom differences (for dotv and fallback metrics)
                    float dx = tx - ax;
                    float dy = ty - ay;
                    float dz = tz - az;

                    // Unit-direction projection s = dot(u, a) and |a|^2
                    // Use FMA to minimize roundoff
                    float sproj = fmaf(uz, az, fmaf(uy, ay, ux*ax));           // dot(u,a)
                    float a2    = fmaf(az, az, fmaf(ay, ay, ax*ax));           // |a|^2

                    // delta_r = -s + 0.5*(|a|^2 - s^2)/R0  (meters)
                    float s2     = sproj * sproj;
                    float corr   = 0.5f * (a2 - s2) * invR0;
                    float delta_r= -sproj + corr;

                    // Approx r using stable decomposition: r = R0 + delta_r
                    float r_det = R0 + delta_r;
                    if (!(r_det > 0.0f)) continue;

                    // dotv for polarization and Q-estimate (same as your original)
                    float dotv = dx / r_det;           // +x incidence approximation
                    float tmp = 2.0f * (1.0f - dotv);
                    if (tmp < 0.0f) tmp = 0.0f;
                    float Q_val = k_global * __fsqrt_rn(tmp);

                    // f0(Q) +/- forward term
                    const float* param_ptr = &s_params[j*11];
                    float f0v = get_f0_from_params(Q_val, param_ptr);
                    if (remove_forward) f0v -= s_f0z[j];

                    // Scattering factor including anomalous
                    float2 s_tot;
                    s_tot.x = f0v + s_anm[j].x;
                    s_tot.y = s_anm[j].y;

                    // Multiply by entrance amplitude (complex)
                    float2 amp = s_amp[j];
                    float real_part = amp.x * s_tot.x - amp.y * s_tot.y;
                    float imag_part = amp.x * s_tot.y + amp.y * s_tot.x;

                    // Relative phase only: k * (ax + delta_r)
                    float small_path = ax + delta_r;                        // meters, small
                    float phase_rel  = fmaf(k_global, small_path, 0.0f);    // radians

                    float s_rel, c_rel;
                    __sincosf(phase_rel, &s_rel, &c_rel);

                    // Rotate by relative phase
                    float2 val;
                    val.x = real_part * c_rel - imag_part * s_rel;
                    val.y = real_part * s_rel + imag_part * c_rel;

                    // Polarization factor on amplitude (unchanged)
                    if (apply_polarization) {
                        float P = pol_perp_rate + (1.0f - pol_perp_rate) * (dotv * dotv);
                        if (P < 0.0f) P = 0.0f;
                        if (P > 1.0f) P = 1.0f;
                        float scale = __fsqrt_rn(P);
                        val.x *= scale;
                        val.y *= scale;
                    }

                    // Thomson scaling and accumulate (in relative frame)
                    sum_rel.x += val.x * rE_F;
                    sum_rel.y += val.y * rE_F;
                }
                __syncthreads();
            }

            // Apply the per-pixel base phasor: exp(i*k*R0)
            float2 sum_rot;
            sum_rot.x = sum_rel.x * c0 - sum_rel.y * s0;
            sum_rot.y = sum_rel.x * s0 + sum_rel.y * c0;

            // Write out
            int out_idx = pidx;
            detector_field[out_idx].x += sum_rot.x;
            detector_field[out_idx].y += sum_rot.y;
        } // kernel

        } // extern "C"
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
                            apply_polarization=False):
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
        apply_polarization=False
    ):
        """
        Orchestrate CPU kinematic scattering over all chunks.

        Steps:
            1) Load scattering databases (f0 and f1/f2).
            2) Prepare detector coordinates in meters and beam wave number.
            3) Optionally compute or load depth-dependent entrance amplitude Ein
            on the beam grid and interpolate it at atom positions.
            4) Compile the CPU scattering kernel and process all chunks in parallel.
            5) Sum complex fields from all chunks.

        Args:
            sample: Provides chunk_total and loaders for species and positions.
            measurement_positions (np.ndarray or cupy.ndarray): Array of shape
                (3, Nx*Ny) with pixel coordinates in angstrom.
            measurement_shape (tuple[int, int]): (Nx, Ny) detector shape.
            stage: Object providing rotation (3x3) and translation (3,) arrays.
            detector: Unused placeholder for API parity.
            remove_forward_component (bool): If True, subtract f0(0) inside the kernel.
            use_depth_ein (bool): If True, compute Ein using the current beam grid
                and use it as per-atom entrance amplitude.
            ein_cache_dir (str or None): Directory for Ein cache files. If None,
                uses a directory under self.directory.
            recompute_cache (bool): If True, force recomputation of Ein.
            apply_polarization (bool): If True, apply polarization scaling in kernel.

        Returns:
            np.ndarray: Complex64 array of shape (Ny, Nx) with the final field.
        """
        import hashlib, json
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

        # Optional depth-dependent entrance amplitude
        A_beam_np = None
        s_min = s_max = None
        if use_depth_ein:
            A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=0)
            s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Prepare cache key for Ein if enabled
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

        # Pre-bind some beam-grid arrays for Ein interpolation
        if use_depth_ein:
            NyB, NzB = self._beam_Ny, self._beam_Nz
            du, dv = self._beam_du, self._beam_dv
            uc, vc = self._beam_uc, self._beam_vc
            e1 = self._beam_e1; e2 = self._beam_e2
            khat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
            E0_map = self._beam_E0_map.astype(np.complex64)

            def _ein_for_positions_cpu(pos_np):
                # Project atom positions onto beam transverse basis (u, v)
                au = pos_np[:, 0]*e1[0] + pos_np[:, 1]*e1[1] + pos_np[:, 2]*e1[2]
                av = pos_np[:, 0]*e2[0] + pos_np[:, 1]*e2[1] + pos_np[:, 2]*e2[2]
                iu = au / du + uc
                iv = av / dv + vc

                # Bilinear weights and indices
                i0 = np.floor(iu).astype(np.int64); j0 = np.floor(iv).astype(np.int64)
                i1 = np.clip(i0 + 1, 0, NyB-1);     j1 = np.clip(j0 + 1, 0, NzB-1)
                i0 = np.clip(i0,       0, NyB-1);   j0 = np.clip(j0,       0, NzB-1)

                fu = (iu - i0).astype(np.float32); fv = (iv - j0).astype(np.float32)
                r00 = (i0 * NzB + j0); r01 = (i0 * NzB + j1)
                r10 = (i1 * NzB + j0); r11 = (i1 * NzB + j1)

                # Interpolate A_beam and E0 on the grid
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

                # Depth fraction along the beam
                s = pos_np @ khat
                f = np.clip((s - s_min) / (s_max - s_min + 1e-12), 0.0, 1.0).astype(np.float32)

                # Ein = E0 * A^f; handle tiny amplitudes robustly
                tiny = 1e-12
                Ein = np.exp(np.log(A_s + 0j) * f) * E0_s
                mask00 = (np.abs(A_s) < tiny) & (f < tiny)
                Ein[mask00] = E0_s[mask00]
                return Ein.astype(np.complex64)

        # Compile the CPU CFFI kernel
        ffi_obj, complied_code = self.compile_compute_scattering_cffi()

        # Threaded loop over chunks
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_threads = min(chunk_total, multiprocessing.cpu_count())

        def worker(chunk_id):
            # Load and transform positions for this chunk
            pos_A = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
            if pos_A.size == 0:
                return np.zeros((Ny, Nx), dtype=np.complex64)

            pos_A = pos_A @ stage.rotation
            pos_A += stage.translation

            init_amp = None
            if use_depth_ein:
                cache_dir_local = ein_cache_dir or os.path.join(self.directory, "ein_cache")
                cache_path = os.path.join(cache_dir_local, f"ein_chunk_{chunk_id}_{key_hash}.npz")
                # Try cache first
                if (not recompute_cache) and os.path.isfile(cache_path):
                    try:
                        with np.load(cache_path) as npz:
                            arr = npz["ein"]
                        if arr.shape[0] == pos_A.shape[0]:
                            init_amp = arr.astype(np.complex64, copy=False)
                    except Exception:
                        init_amp = None
                # Compute and cache if needed
                if init_amp is None:
                    init_amp = _ein_for_positions_cpu(pos_A)
                    try:
                        np.savez_compressed(cache_path, ein=init_amp)
                    except Exception:
                        pass

            # Scatter this chunk on CPU
            return self.cpu_scatter_chunk_cffi(
                complied_code, ffi_obj, chunk_id, sample, Nx, Ny,
                coords_x_m, coords_y_m, coords_z_m,
                db_dict_f0_all, db_dict_f1f2_all, k_val, stage,
                detector, remove_forward_component,
                initial_amp_complex=init_amp,
                apply_polarization=apply_polarization
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
        recompute_cache: bool = False,
        apply_polarization: bool = False
    ):
        """
        Orchestrate GPU kinematic scattering across chunks and GPUs.

        Behavior:
            - Falls back to CPU if CuPy is not available or if no GPU is found.
            - Distributes chunks across available GPUs and accumulates results.
            - Optionally computes depth-dependent entrance amplitude Ein on the beam
            grid and interpolates it at atom positions, with on-disk caching.
            - Applies optional removal of the forward component and optional
            polarization scaling inside the CUDA kernel.

        Args:
            sample: Provides chunk_total and per-chunk loaders.
            measurement_positions (np.ndarray or cupy.ndarray): Pixel coordinates
                shaped (3, Nx*Ny) in angstrom.
            measurement_shape (tuple[int, int]): (Nx, Ny).
            stage: Object with rotation (3x3) and translation (3,) arrays.
            remove_forward (bool): If True, subtract f0(0) in the kernel.
            use_depth_ein (bool): If True, compute Ein and use as entrance amplitude.
            ein_cache_dir (str or None): Directory for Ein cache files.
            recompute_cache (bool): If True, force recomputation of Ein cache.
            apply_polarization (bool): If True, apply polarization scaling.

        Returns:
            np.ndarray: Complex64 array of shape (Ny, Nx) with the final field.
        """
        if cp is None:
            # Fallback to CPU when CuPy is not installed
            print("[beam] CuPy not installed, falling back to CPU.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage,
                                        remove_forward_component=remove_forward,
                                        use_depth_ein=use_depth_ein,
                                        ein_cache_dir=ein_cache_dir,
                                        recompute_cache=recompute_cache,
                                        apply_polarization=apply_polarization)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            # Fallback to CPU when no CUDA device is available
            print("[beam] No GPUs found, falling back to CPU.")
            return self.interact_beam_cpu(sample, measurement_positions, measurement_shape, stage,
                                        remove_forward_component=remove_forward,
                                        use_depth_ein=use_depth_ein,
                                        ein_cache_dir=ein_cache_dir,
                                        recompute_cache=recompute_cache,
                                        apply_polarization=apply_polarization)

        import hashlib, json

        print(f"[beam] Found {n_gpus} GPU(s).")

        # Load databases once on host
        db_f0   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')
        f0_zero = self._build_f0_zero_dict(db_f0)

        Nx, Ny = measurement_shape
        final_result = np.zeros((Ny, Nx), dtype=np.complex64)

        # Pinned host buffers for detector coordinates and stage transform
        x_coords = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        y_coords = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        z_coords = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)

        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        chunk_total = sample.chunk_total
        print(f"[beam] Total of {chunk_total} chunk(s) to process.")

        # Optional Ein precomputation (CPU or GPU path reused internally)
        A_beam_np = None
        s_min = s_max = None
        if use_depth_ein:
            if cp is not None:
                A_beam_np = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius=0)
            else:
                A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=0)
            s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Prepare cache key for Ein if enabled
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

        # Partition chunks across GPUs
        chunks_per_gpu = chunk_total // n_gpus
        remainder = chunk_total % n_gpus
        partial_results = [None] * n_gpus

        # Compile the interaction kernel once
        interaction_kernel = self.build_interaction_kernel()

        def gpu_worker(gpu_id, x_coords, y_coords, z_coords, chunk_indices, result_index):
            # Select device for this worker
            cp.cuda.Device(gpu_id).use()

            # Upload stage and detector arrays
            Rg = cp.asarray(R_pin, dtype=cp.float32)
            Tg = cp.asarray(T_pin, dtype=cp.float32)
            xg = cp.asarray(x_coords); yg = cp.asarray(y_coords); zg = cp.asarray(z_coords)

            dfield = cp.zeros((Nx * Ny,), dtype=cp.complex64)

            block = (16, 16)
            grid  = ((Nx + block[0] - 1) // block[0],
                    (Ny + block[1] - 1) // block[1])

            # Bind beam-grid data for Ein interpolation if enabled
            if use_depth_ein:
                A_gpu = cp.asarray(A_beam_np)
                E0_gpu = cp.asarray(self._beam_E0_map)
                NyB, NzB = A_gpu.shape
                du, dv = self._beam_du, self._beam_dv
                uc, vc = self._beam_uc, self._beam_vc
                du_g, dv_g = cp.float32(du), cp.float32(dv)
                uc_g, vc_g = cp.float32(uc), cp.float32(vc)
                e1g = cp.asarray(self._beam_e1); e2g = cp.asarray(self._beam_e2)
                khat = cp.asarray(self._direction / np.linalg.norm(self._direction), dtype=cp.float32)
                smin = cp.float32(s_min); smax = cp.float32(s_max)

            def _ein_for_positions(pos_g):
                # Project onto beam transverse basis
                au = pos_g[:, 0]*e1g[0] + pos_g[:, 1]*e1g[1] + pos_g[:, 2]*e1g[2]
                av = pos_g[:, 0]*e2g[0] + pos_g[:, 1]*e2g[1] + pos_g[:, 2]*e2g[2]
                iu = au / du_g + uc_g
                iv = av / dv_g + vc_g

                # Bilinear interpolation indices and weights
                i0 = cp.floor(iu).astype(cp.int64); j0 = cp.floor(iv).astype(cp.int64)
                i1 = cp.clip(i0 + 1, 0, NyB-1); j1 = cp.clip(j0 + 1, 0, NzB-1)
                i0 = cp.clip(i0, 0, NyB - 1);     j0 = cp.clip(j0, 0, NzB - 1)

                fu = (iu - i0).astype(cp.float32); fv = (iv - j0).astype(cp.float32)
                one = cp.float32(1.0)

                r00 = (i0 * NzB + j0).astype(cp.int64); r01 = (i0 * NzB + j1).astype(cp.int64)
                r10 = (i1 * NzB + j0).astype(cp.int64); r11 = (i1 * NzB + j1).astype(cp.int64)

                # Interpolate A_beam and E0 on the grid
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

                # Depth fraction along the beam
                s = pos_g[:, 0]*khat[0] + pos_g[:, 1]*khat[1] + pos_g[:, 2]*khat[2]
                f = cp.clip((s - smin) / (smax - smin + cp.float32(1e-12)), 0.0, 1.0).astype(cp.float32)

                # Ein = E0 * A^f; handle tiny amplitudes robustly
                tiny = cp.float32(1e-12)
                absA = cp.abs(A_s)
                Ein = cp.exp(cp.log(A_s + 0j) * f) * E0_s
                mask00 = (absA < tiny) & (f < tiny)
                Ein = cp.where(mask00, E0_s, Ein)
                return Ein

            # Process assigned chunks on this GPU
            for cidx in chunk_indices:
                spc = sample.load_chunk_species(cidx, use_gpu=False)
                nA = spc.shape[0]
                if nA == 0:
                    continue

                pos = cp.array(sample.load_chunk_positions(cidx, use_gpu=True), dtype=cp.float32)
                pos = pos @ Rg; pos += Tg

                # Build per-atom tables on host and upload compactly
                s_anom = np.zeros(nA, np.complex64)
                f0p    = np.zeros((nA, 11), np.float32)
                f0z    = np.zeros(nA, np.float32)
                for el in pd.unique(spc):
                    if el not in db_f0:
                        continue
                    m = (spc == el)
                    f0p[m] = db_f0[el]
                    f0z[m] = f0_zero.get(el, 0.0)
                    tbl = db_f1f2.get(el)
                    if tbl is not None:
                        s_anom[m] = self.get_f1f2_from_params(self._energy, tbl)

                if use_depth_ein:
                    cache_path = os.path.join(cache_dir, f"ein_chunk_{cidx}_{key_hash}.npz")
                    initial_amp = None
                    # Attempt to load Ein from cache
                    if (not recompute_cache) and os.path.isfile(cache_path):
                        try:
                            with np.load(cache_path) as npz:
                                arr = npz["ein"]
                            if arr.shape[0] == nA:
                                initial_amp = cp.asarray(arr.astype(np.complex64))
                        except Exception:
                            initial_amp = None
                    # Compute and cache Ein if needed
                    if initial_amp is None:
                        initial_amp = _ein_for_positions(pos)
                        try:
                            np.savez_compressed(cache_path, ein=initial_amp.get())
                        except Exception:
                            pass
                else:
                    initial_amp = cp.ones(nA, dtype=cp.complex64)

                # Convert positions to meters
                px = pos[:, 0] / 1e10; py = pos[:, 1] / 1e10; pz = pos[:, 2] / 1e10

                # k components for +x propagation
                kx_cp = cp.full(nA, self._kx_scalar, dtype=cp.float32)
                ky_cp = cp.full(nA, self._ky_scalar, dtype=cp.float32)
                kz_cp = cp.full(nA, self._kz_scalar, dtype=cp.float32)

                # Upload per-atom parameters
                s_anom_cp = cp.asarray(s_anom)
                f0_params_cp = cp.asarray(f0p)
                f0_zero_cp   = cp.asarray(f0z)

                # Launch interaction kernel
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
                        np.int32(1 if remove_forward else 0),
                        np.int32(1 if apply_polarization else 0),
                        np.float32(self._pol_perp_rate)
                    )
                )
                # Release any cached blocks for this iteration
                cp.get_default_memory_pool().free_all_blocks()

            # Copy back this GPU's partial result
            partial_results[result_index] = dfield.reshape((Ny, Nx)).get()
            del xg, yg, zg
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Start one worker thread per GPU
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

        # Join threads and sum results
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
        recompute_cache: bool = False,
        apply_polarization: bool = False
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
                apply_polarization=apply_polarization
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
                apply_polarization=apply_polarization
            )

        # Optional offset subtraction
        return (final_field - offset) if (offset is not None) else final_field
    # -------------------------------------
        
    # -------------------------------------
    # Direct transmission
    def _compute_beam_column_A_map_cpu(self, sample, stage, kernel_radius=0):
        """
        Compute the transmission map A(u,v) = exp(-tau + i*phi) on the beam grid (CPU).

        For each chunk, atoms are rotated and translated by the stage, projected onto
        the beam transverse basis, and deposited onto the beam grid using a Triangular
        Shaped Cloud (TSC) kernel. The result is the product of per-atom contributions,
        expressed via accumulated absorption (tau) and phase (phi).

        Args:
            sample: Sample object with chunk accessors.
            stage: Stage object with `rotation` (3x3) and `translation` (3,) arrays.
            kernel_radius (int): Optional Gaussian blur radius in pixels applied to
                tau and phi after accumulation. Zero disables the blur.

        Returns:
            np.ndarray: Complex64 array of shape (Ny_beam, Nz_beam) with A(u,v).
        """
        # Constants (angstrom)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10
        du, dv = self._beam_du, self._beam_dv
        NyB, NzB = self._beam_Ny, self._beam_Nz
        A_pix_A2 = float(du) * float(dv)
        # Scale linking per-atom factors to per-pixel tau/phi
        scale = (r_e_A * lam_A) / A_pix_A2

        # Accumulators for absorption (tau) and phase (phi)
        tau = np.zeros((NyB, NzB), np.float32)
        phi = np.zeros((NyB, NzB), np.float32)

        # Databases for f1, f2, and f0(0)
        f1f2_dict = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict = self._build_f0_zero_dict(f0_params_dict)

        e1 = self._beam_e1; e2 = self._beam_e2

        def _tsc_w(d):
            # 1D TSC weights on distances in pixel units
            w = np.zeros_like(d, dtype=np.float32)
            m0 = d <= 0.5
            w[m0] = 0.75 - d[m0]*d[m0]
            m1 = (~m0) & (d <= 1.5)
            t = 1.5 - d[m1]
            w[m1] = 0.5 * t * t
            return w

        for cid in range(1, sample.chunk_total + 1):
            spc = sample.load_chunk_species(cid, use_gpu=False)
            pos = sample.load_chunk_positions(cid, use_gpu=False).astype(np.float32)  # angstrom
            if pos.size == 0:
                continue
            # Apply stage transform in real space (angstrom)
            pos = pos @ stage.rotation
            pos += stage.translation

            nA = pos.shape[0]
            f1  = np.zeros(nA, np.float32)
            f2  = np.zeros(nA, np.float32)
            f0z = np.zeros(nA, np.float32)

            # Fill per-atom f0(0), f1, f2 by species
            for el in pd.unique(spc):
                el_s = str(el)
                m = (spc == el)
                f0z[m] = float(f0_zero_dict.get(el_s, 0.0))
                tbl = f1f2_dict.get(el_s)
                if tbl is not None:
                    cplx = self.get_f1f2_from_params(self._energy, tbl)
                    f1[m] = float(cplx.real)
                    f2[m] = float(cplx.imag)

            # Project to beam basis (u, v), then to continuous grid indices
            au = pos[:, 0]*e1[0] + pos[:, 1]*e1[1] + pos[:, 2]*e1[2]
            av = pos[:, 0]*e2[0] + pos[:, 1]*e2[1] + pos[:, 2]*e2[2]
            iu = au / du + self._beam_uc
            iv = av / dv + self._beam_vc

            # TSC central pixel and neighbor distances
            ic = np.floor(iu + 0.5).astype(np.int64)
            jc = np.floor(iv + 0.5).astype(np.int64)

            du_m1 = np.abs(iu - (ic - 1)); du_0 = np.abs(iu - ic); du_p1 = np.abs(iu - (ic + 1))
            dv_m1 = np.abs(iv - (jc - 1)); dv_0 = np.abs(iv - jc); dv_p1 = np.abs(iv - (jc + 1))

            wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
            wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)
            
            # Per-atom weights for tau and phi
            w_phi_atom = (-scale * (f0z + f1)).astype(np.float32)
            w_tau_atom = ( scale *  f2).astype(np.float32)

            idx_phi = []; w_phi = []
            idx_tau = []; w_tau = []

            def _push(ii, jj, fac, w_atom):
                # Append valid (linear_index, weight) pairs for accumulation
                inb = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                if not np.any(inb):
                    return np.empty((0,), np.int64), np.empty((0,), np.float32)
                rows = ii[inb]; cols = jj[inb]
                idx = (rows * NzB + cols).astype(np.int64)
                w = (w_atom[inb] * fac[inb]).astype(np.float32)
                return idx, w

            # 3x3 TSC stencil accumulation
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
                # Scatter-add into flat arrays
                np.add.at(phi.ravel(), idxp, wp)
                np.add.at(tau.ravel(), idxt, wt)

        # Optional Gaussian blur on tau and phi (same-size FFT convolution)
        if kernel_radius > 0:
            rad = int(kernel_radius); sig = rad / 2.0
            y, x = np.ogrid[-rad:rad+1, -rad:rad+1]
            k = np.exp(-(x*x + y*y) / (2.0*sig*sig)).astype(np.float32)
            k /= k.sum()
            Fk = np.fft.fft2(k, s=tau.shape)
            tau = np.fft.ifft2(np.fft.fft2(tau) * Fk).real.astype(np.float32)
            phi = np.fft.ifft2(np.fft.fft2(phi) * Fk).real.astype(np.float32)

        # Compose final complex transmission
        A_map = np.exp(-tau + 1j*phi).astype(np.complex64)
        return A_map

    def _compute_beam_column_A_map_gpu(self, sample, stage, kernel_radius=0):
        """
        Compute the transmission map A(u,v) = exp(-tau + i*phi) on the beam grid (GPU).

        If no GPU is present, this falls back to the CPU implementation. Accumulation
        uses a TSC deposition onto the beam grid, with optional Gaussian blur on
        tau and phi.

        Args:
            sample: Sample object with chunk accessors.
            stage: Stage object with `rotation` (3x3) and `translation` (3,) arrays.
            kernel_radius (int): Optional Gaussian blur radius in pixels applied to
                tau and phi after accumulation. Zero disables the blur.

        Returns:
            np.ndarray: Complex64 array of shape (Ny_beam, Nz_beam) with A(u,v).
        """
        if cp is None:
            return self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        n_gpus = cp.cuda.runtime.getDeviceCount()
        if n_gpus < 1:
            return self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        # Constants (angstrom)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10
        du, dv = self._beam_du, self._beam_dv
        NyB, NzB = self._beam_Ny, self._beam_Nz
        A_pix_A2 = float(du) * float(dv)
        scale = (r_e_A * lam_A) / A_pix_A2

        # Pin stage arrays for faster host->device copies
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        # Databases for anomalous and f0(0)
        f1f2_dict = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict = self._build_f0_zero_dict(f0_params_dict)

        partial = [None] * n_gpus
        chunks_per_gpu = sample.chunk_total // n_gpus
        remainder = sample.chunk_total % n_gpus

        # Worker processes a subset of chunks on one device
        def worker(dev_id, chunks, out_idx):
            cp.cuda.Device(dev_id).use()
            Rg = cp.asarray(R_pin); Tg = cp.asarray(T_pin)

            tau_acc = cp.zeros((NyB, NzB), dtype=cp.float32)
            phi_acc = cp.zeros((NyB, NzB), dtype=cp.float32)

            def _tsc_w(d):
                # 1D TSC weights on distances in pixel units
                w = cp.zeros_like(d, dtype=cp.float32)
                m0 = d <= 0.5
                w[m0] = 0.75 - d[m0] * d[m0]
                m1 = (~m0) & (d <= 1.5)
                t = 1.5 - d[m1]
                w[m1] = 0.5 * t * t
                return w

            for cid in chunks:
                spc = sample.load_chunk_species(cid, use_gpu=False)
                pos = sample.load_chunk_positions(cid, use_gpu=False)  # angstrom
                nA = pos.shape[0]
                if nA == 0:
                    continue

                # Build per-atom f0(0), f1, f2 on host
                f1  = np.zeros(nA, np.float32)
                f2  = np.zeros(nA, np.float32)
                f0z = np.zeros(nA, np.float32)
                for el in pd.unique(spc):
                    el_s = str(el)
                    m = (spc == el)
                    f0z[m] = float(f0_zero_dict.get(el_s, 0.0))
                    tbl = f1f2_dict.get(el_s)
                    if tbl is not None:
                        cplx = self.get_f1f2_from_params(self._energy, tbl)
                        f1[m] = float(cplx.real)
                        f2[m] = float(cplx.imag)

                f1g  = cp.asarray(f1);   f2g  = cp.asarray(f2)
                f0zg = cp.asarray(f0z)

                posg = cp.asarray(pos, dtype=cp.float32)
                # Apply stage transform in real space (angstrom)
                posg = posg @ Rg; posg += Tg

                # Project onto (u,v) in the beam basis
                e1g = cp.asarray(self._beam_e1); e2g = cp.asarray(self._beam_e2)
                au = posg[:, 0]*e1g[0] + posg[:, 1]*e1g[1] + posg[:, 2]*e1g[2]
                av = posg[:, 0]*e2g[0] + posg[:, 1]*e2g[1] + posg[:, 2]*e2g[2]

                # Continuous grid indices (center at uc/vc)
                iu = au / self._beam_du + self._beam_uc
                iv = av / self._beam_dv + self._beam_vc

                ic = cp.floor(iu + 0.5).astype(cp.int64)
                jc = cp.floor(iv + 0.5).astype(cp.int64)

                du_m1 = cp.abs(iu - (ic - 1)); du_0 = cp.abs(iu - ic); du_p1 = cp.abs(iu - (ic + 1))
                dv_m1 = cp.abs(iv - (jc - 1)); dv_0 = cp.abs(iv - jc); dv_p1 = cp.abs(iv - (jc + 1))

                wu_m1, wu_0, wu_p1 = _tsc_w(du_m1), _tsc_w(du_0), _tsc_w(du_p1)
                wv_m1, wv_0, wv_p1 = _tsc_w(dv_m1), _tsc_w(dv_0), _tsc_w(dv_p1)

                # Per-atom weights for tau and phi
                w_phi_atom = (-scale * (f0zg + f1g)).astype(cp.float32)
                w_tau_atom = ( scale *  f2g).astype(cp.float32)

                idx_phi = []; w_phi = []
                idx_tau = []; w_tau = []

                def _push(ii, jj, fac, w_atom):
                    # Append valid (linear_index, weight) pairs for accumulation
                    inb = (ii >= 0) & (ii < NyB) & (jj >= 0) & (jj < NzB) & (fac > 0.0)
                    if not bool(cp.any(inb)):
                        return cp.empty((0,), cp.int64), cp.empty((0,), cp.float32)
                    rows = ii[inb]; cols = jj[inb]
                    idx = (rows * NzB + cols).astype(cp.int64)
                    w = (w_atom[inb] * fac[inb]).astype(cp.float32)
                    return idx, w

                # 3x3 TSC stencil accumulation
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
                    # Accumulate into flat arrays via safe GPU bincount
                    phi_hist = self._safe_bincount_gpu(idxp, wp, bins, dtype=cp.float32)
                    tau_hist = self._safe_bincount_gpu(idxt, wt, bins, dtype=cp.float32)
                    phi_acc += phi_hist.reshape(NyB, NzB)
                    tau_acc += tau_hist.reshape(NyB, NzB)

                # Release unused blocks between chunks
                cp.get_default_memory_pool().free_all_blocks()

            # Optional Gaussian blur
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

        # Launch workers per GPU
        threads = []
        start = 1
        for gid in range(n_gpus):
            n_chunk = chunks_per_gpu + (1 if gid < remainder else 0)
            end = start + n_chunk
            t = threading.Thread(target=worker, args=(gid, range(start, end), gid))
            t.start(); threads.append(t)
            start = end
        for t in threads: t.join()

        # Combine independent partial products
        A_total = np.ones((self._beam_Ny, self._beam_Nz), np.complex64)
        for p in partial:
            if p is not None:
                A_total *= p
        return A_total
    
    def atomic_transmission(self, sample, detector, stage,
                            use_gpu=True, kernel_radius=0,
                            padding_mode: str = "edge",
                            pad_constant: float = 0.0):
        """
        Compute the transmitted field at the sample exit plane and map it to the detector.

        Steps:
        1) Build A(u,v) on the beam grid (CPU or GPU).
        2) Form the exit-plane field E_plane = E0(u,v) * A(u,v).
        3) If the detector plane is not coincident with the exit plane, propagate
            E_plane by free-space angular spectrum.
        4) Bilinearly resample the field onto detector pixels.

        Args:
            sample: Sample object with chunk accessors.
            detector: Detector object with `shape`, `pixel_coordinates`, and `pixel_size`.
            stage: Stage object with `rotation` (3x3) and `translation` (3,) arrays.
            use_gpu (bool): If True and CuPy is available, use GPU for steps that support it.
            kernel_radius (int): Optional Gaussian blur radius in pixels used when
                computing A(u,v). Zero disables the blur.
            padding_mode (str): Padding strategy for propagation. One of {"edge", "constant"}.
            pad_constant (float): Fill value when `padding_mode == "constant"`.

        Returns:
            np.ndarray: Complex64 array of shape (Ny_detector, Nx_detector) on the detector plane.
        """
        # 1) A(u,v) on the beam grid
        if use_gpu and (cp is not None):
            A_beam = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius)
        else:
            A_beam = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius)

        # 2) Exit field on the sample exit plane
        E_plane = (self._beam_E0_map * A_beam).astype(np.complex64)
        NyB, NzB = E_plane.shape
        du_A = float(self._beam_du)  # angstrom
        dv_A = float(self._beam_dv)  # angstrom

        # Geometry to detect detector-plane offset relative to exit plane
        k_hat = (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        _, s_max = self._compute_global_depth_bounds(sample, stage)  # angstrom (exit plane)
        pix = detector.pixel_coordinates
        pix_cpu = pix.get() if (cp is not None and isinstance(pix, cp.ndarray)) else np.asarray(pix)
        s_det = (pix_cpu[0, :] * k_hat[0] + pix_cpu[1, :] * k_hat[1] + pix_cpu[2, :] * k_hat[2]).astype(np.float64)
        s_det_min, s_det_max = float(np.min(s_det)), float(np.max(s_det))
        s_det_mean = float(np.mean(s_det))
        plane_span_A = s_det_max - s_det_min
        tol_plane_A = max(1e-3, 1e-6 * abs(s_det_mean))
        tol_off_A   = 1e-3

        need_propagation = False
        dz_A = 0.0
        if plane_span_A <= tol_plane_A:
            dz_A = s_det_mean - float(s_max)
            need_propagation = (abs(dz_A) > tol_off_A)
        else:
            dz_A = s_det_mean - float(s_max)
            if abs(dz_A) > tol_off_A:
                need_propagation = True
                print(f"[beam] atomic_transmission: detector appears non-planar (Delta s range={plane_span_A:.3g} A). "
                    f"Propagating by mean Delta z={dz_A:.3g} A.")

        # 3) Propagate if needed
        if need_propagation:
            dz_m = dz_A * 1e-10  # angstrom -> meters
            dx_m = dv_A * 1e-10  # columns (v)
            dy_m = du_A * 1e-10  # rows    (u)

            if use_gpu and (cp is not None):
                kernel = self.build_propagation_multiplier_kernel()
                E_gpu = cp.asarray(E_plane)
                E_gpu = self._angular_spectrum_propagate_gpu(
                    field=E_gpu, dx=dx_m, dy=dy_m, z=dz_m, kernel=kernel,
                    step_max=0.02, pad_factor=1.0,
                    padding_mode=padding_mode, pad_constant=pad_constant
                )
                E_plane = E_gpu
            else:
                ffi, lib = self.compile_propagation_multiplier_cffi()
                E_plane = self._angular_spectrum_propagate_cpu(
                    field=E_plane, dx=dx_m, dy=dy_m, z=dz_m, lib=lib, ffi=ffi,
                    step_max=0.02, pad_factor=1.0,
                    padding_mode=padding_mode, pad_constant=pad_constant
                ).astype(np.complex64)

        # 4) Bilinear resampling to detector pixels
        NyD, NxD = detector.shape
        if use_gpu and (cp is not None):
            pix_g = pix if isinstance(pix, cp.ndarray) else cp.asarray(pix)
            e1g = cp.asarray(self._beam_e1); e2g = cp.asarray(self._beam_e2)
            u = pix_g[0]*e1g[0] + pix_g[1]*e1g[1] + pix_g[2]*e1g[2]
            v = pix_g[0]*e2g[0] + pix_g[1]*e2g[1] + pix_g[2]*e2g[2]
            iu = u / cp.float32(du_A) + cp.float32(self._beam_uc)
            iv = v / cp.float32(dv_A) + cp.float32(self._beam_vc)

            i0 = cp.floor(iu).astype(cp.int64); j0 = cp.floor(iv).astype(cp.int64)
            i1 = i0 + 1; j1 = j0 + 1
            fu = (iu - i0).astype(cp.float32); fv = (iv - j0).astype(cp.float32)

            i0 = cp.clip(i0, 0, NyB - 1); i1 = cp.clip(i1, 0, NyB - 1)
            j0 = cp.clip(j0, 0, NzB - 1); j1 = cp.clip(j1, 0, NzB - 1)

            E_src = E_plane if isinstance(E_plane, cp.ndarray) else cp.asarray(E_plane)
            idx00 = (i0 * NzB + j0).astype(cp.int64)
            idx01 = (i0 * NzB + j1).astype(cp.int64)
            idx10 = (i1 * NzB + j0).astype(cp.int64)
            idx11 = (i1 * NzB + j1).astype(cp.int64)

            E00 = E_src.ravel()[idx00]
            E01 = E_src.ravel()[idx01]
            E10 = E_src.ravel()[idx10]
            E11 = E_src.ravel()[idx11]

            one = cp.float32(1.0)
            E_flat = (E00 * (one - fu)*(one - fv) +
                    E01 * (one - fu)*fv +
                    E10 * (fu)*(one - fv) +
                    E11 * (fu)*fv).astype(cp.complex64)
            E_det = E_flat.reshape(NyD, NxD).get()
        else:
            e1 = self._beam_e1; e2 = self._beam_e2
            u = pix_cpu[0]*e1[0] + pix_cpu[1]*e1[1] + pix_cpu[2]*e1[2]
            v = pix_cpu[0]*e2[0] + pix_cpu[1]*e2[1] + pix_cpu[2]*e2[2]
            iu = u / du_A + self._beam_uc
            iv = v / dv_A + self._beam_vc

            i0 = np.floor(iu).astype(np.int64); j0 = np.floor(iv).astype(np.int64)
            i1 = i0 + 1; j1 = j0 + 1
            fu = (iu - i0).astype(np.float32); fv = (iv - j0).astype(np.float32)

            i0 = np.clip(i0, 0, NyB - 1); i1 = np.clip(i1, 0, NyB - 1)
            j0 = np.clip(j0, 0, NzB - 1); j1 = np.clip(j1, 0, NzB - 1)

            Eb = (E_plane if isinstance(E_plane, np.ndarray) else np.asarray(E_plane)).ravel()
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

        return E_det.astype(np.complex64)
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
        final_field = np.zeros((Ny, Nx), dtype=np.complex128)

        # Parse scattering params
        sc_offset = scattering_params[0] if (len(scattering_params) >= 1) else None
        use_depth_ein = scattering_params[1] if len(scattering_params) >= 2 else False

        if use_gpu and (cp is not None):
            if scattering:
                final_field += self.atomic_scattering_kinematic(
                    sample, detector, stage,
                    offset=sc_offset, use_gpu=True,
                    remove_forward=transmission,      # remove forward if also transmitting
                    use_depth_ein=use_depth_ein
                )
            if transmission:
                final_field += self.atomic_transmission(
                    sample, detector, stage, use_gpu=True,
                    kernel_radius=transmission_params[0]
                )
        else:
            # CPU path; warn if GPU requested but not available
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

    def wavefield_propagation(self, detector, optics_stack,
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
            optics_stack: Object with attribute `components`, a list of dicts.
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
        for elem in optics_stack.components:
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
                # Apply thin-lens phase and optional uniform absorption.
                E = self._apply_thin_lens_box(E, dx, dy, elem, use_gpu and cp is not None)

            elif kind == 'aperture':
                # Apply real-space aperture mask centered at the field center.
                E = self._apply_aperture(E, dx, dy, elem, use_gpu and cp is not None)

            else:
                # Unknown element type: fail fast with the element kind in the message.
                raise ValueError(f'Unknown optics element "{kind}"')

        # Write back the final complex field to the detector.
        detector.input_pixel_values(E.astype(np.complex64))
    # -------------------------------------