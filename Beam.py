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
            directory (str, optional): Directory used to store and read beam-related
                metadata. Defaults to the current working directory.

        Notes:
            - Creates the directory if it does not exist.
            - Initializes physical constants in SI units (Planck constant h, speed of
            light c, elementary charge q) and caches h/q for fast eV-to-wavelength
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

    def set_wavefield(self, wavefield):
        """
        Set a user-defined complex entrance wavefield on the transverse beam grid.

        Replaces the internally generated amplitude map (uniform, gaussian, ...)
        with an arbitrary complex field E0(u, v) sampled on the current grid.

        Args:
            wavefield (np.ndarray): 2-D array of shape
                (beam_samples[0], beam_samples[1]) = (Ny, Nz) with all values
                finite. Real or complex; stored as C-contiguous complex64.

        Sampling contract:
            - 'ij' indexing with [u, v] = (y, z): axis 0 is u (parallel to +y),
            axis 1 is v (parallel to +z); the beam propagates along +x.
            - Grid pitch is du = beam_size[0]/Ny and dv = beam_size[1]/Nz in
            angstrom, so sample (i, j) sits at u = (i - (Ny-1)/2)*du,
            v = (j - (Nz-1)/2)*dv and the beam axis passes through the
            fractional index center ((Ny-1)/2, (Nz-1)/2).

        Amplitude convention:
            No renormalization is applied. Built-in profiles peak at unity
            amplitude; the supplied field is used as-is, so its amplitude
            directly scales all downstream intensities.

        Notes:
            - create_beam() and read_beam_metadata() rebuild the field map from
            the stored beam parameters, overwriting a custom wavefield. Call
            set_wavefield() LAST, and call it again after any beam parameter
            change or metadata reload.
            - Sets _beam_profile to 'custom-<sha1[:12]>' of the field bytes so
            on-disk Ein cache keys change with the field contents.

        Raises:
            ValueError: If the array is not 2-D of shape (Ny, Nz), or contains
                non-finite values (NaN/Inf).
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

        # Polarization rate (default to 0.5 = unpolarized if not in metadata)
        self._pol_perp_rate = float(beam_metadata.get("pol_perp_rate", 0.5))
        self._pol_perp_rate = float(np.clip(self._pol_perp_rate, 0.0, 1.0))

        # Transverse basis and grid build
        e1, e2 = self.make_orthonormal_basis(self._direction)
        self._beam_e1 = e1.astype(np.float32)
        self._beam_e2 = e2.astype(np.float32)

        # Build the beam grid and E0(u, v) based on loaded settings
        if hasattr(self, "_init_beam_grid"):
            self._init_beam_grid()

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
                int Ny, int Nz,
                const float* coords_x,      // (Ny*Nz) in meters
                const float* coords_y,
                const float* coords_z,
                float k_val,                // 2*pi/lambda in rad/m
                int apply_pol,              // 0 or 1
                float pol_perp_rate,        // rho_perp in [0, 1]
                float* out_r, float* out_i  // (Ny*Nz)
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
                    int ix = p % Ny;
                    int iy = p / Ny;

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

                        // Up neighbor (or down if on top edge; else self if single row)
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
        Compute per-atom entrance field Ein by bilinearly sampling the beam grid.

        For each atom position in Angstrom, this projects onto the beam-basis
        coordinates (u,v), performs bilinear sampling of tau, phi, and E0 on the
        beam grid, and evaluates:
            Ein = E0 * exp(-f * tau) * exp(i * f * phi)
        where f in [0, 1] is the normalized depth fraction along khat between
        s_min and s_max. Atoms projecting outside the beam grid receive Ein = 0.

        Args:
            pos_np (np.ndarray): Shape (N, 3), float32. Atom positions in Angstrom.
            tau (np.ndarray): Shape (NyB, NzB), float32. Attenuation map on beam grid.
            phi (np.ndarray): Shape (NyB, NzB), float32. Phase map on beam grid.
            E0 (np.ndarray): Shape (NyB, NzB), complex64. Incident field on beam grid.
            e1 (np.ndarray): Shape (3,), float32. First transverse unit vector.
            e2 (np.ndarray): Shape (3,), float32. Second transverse unit vector.
            khat (np.ndarray): Shape (3,), float32. Unit beam direction.
            du (float): Beam-grid spacing along u in Angstrom.
            dv (float): Beam-grid spacing along v in Angstrom.
            uc (float): Beam-grid center index along u.
            vc (float): Beam-grid center index along v.
            s_min (float): Minimum depth along khat in Angstrom.
            s_max (float): Maximum depth along khat in Angstrom.

        Returns:
            np.ndarray: Shape (N,), complex64. Entrance field for each atom; zero
                for atoms outside the beam grid.

        Notes:
            - No edge replication: out-of-bounds indices are treated as zero.
            - A small guard on the depth denominator prevents division by zero.
        """
        N = int(pos_np.shape[0])
        out = np.zeros((N,), dtype=np.complex64)
        if N == 0:
            return out

        NyB, NzB = int(tau.shape[0]), int(tau.shape[1])

        # Project positions to beam basis and then to fractional grid indices
        au = pos_np[:, 0]*e1[0] + pos_np[:, 1]*e1[1] + pos_np[:, 2]*e1[2]
        av = pos_np[:, 0]*e2[0] + pos_np[:, 1]*e2[1] + pos_np[:, 2]*e2[2]
        iu = au / float(du) + float(uc)
        iv = av / float(dv) + float(vc)

        # Hard in-bounds mask; atoms outside the grid get zero
        inb = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))
        if not np.any(inb):
            return out

        # Bilinear weights and gather indices (restricted to in-bounds atoms)
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

        # Normalize depth along khat into f in [0,1]
        s_vals = pos_np[inb, 0]*khat[0] + pos_np[inb, 1]*khat[1] + pos_np[inb, 2]*khat[2]
        denom = float(s_max) - float(s_min)
        if not (denom > 0.0):
            denom = 1.0  # robust fallback
        f = np.clip((s_vals - float(s_min))/denom, 0.0, 1.0).astype(np.float32)

        # Combine attenuation and phase
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
        Set the target maximum phase error for choosing the series order N.

        Args:
            phi_tol_rad (float): Desired maximum phase error per contribution in radians.
                Default (if never set) is 1e-3 rad.
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
        Estimate the minimum N for the series delta_r = R0 * (sqrt(1+t) - 1).

        Uses worst-case |t| and the next-omitted-term bound to ensure
        k * |err_r| <= phi_tol_rad, where err_r ≈ R0 * |C_{N+1}| * |t|^{N+1}.

        Args:
            a_max_m (float): Maximum atom displacement from origin in meters.
            R0_min_m (float): Minimum detector pixel distance from origin in meters.
            phi_tol_rad (float): Target maximum phase error in radians.

        Returns:
            dict: Dictionary with keys:
                - 'use_series' (bool): Whether series expansion is appropriate.
                - 'N' (int): Number of series terms to use.
                - 't_max' (float): Worst-case dimensionless parameter |t|.
        """
        # Guard wavelength and k
        if getattr(self, "_wavelength", None) is None or self._wavelength <= 0.0:
            # Cannot determine, fall back to EXACT
            return dict(use_series=False, N=0, t_max=float("inf"))

        k_val = 2.0 * np.pi / float(self._wavelength)  # rad/m

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

        Analyzes the sample geometry and detector configuration to determine whether
        series expansion is appropriate and, if so, how many terms to use.

        Args:
            sample: Object providing dimensions (Lx, Ly, Lz) in Angstrom, centered at 0.
            detector: Object providing pixel_coordinates of shape (3, Ny*Nz) in Angstrom.
            safety_t_thresh (float, optional): Convergence margin threshold for |t|.
                Defaults to 0.5.
            verbose (bool, optional): If True, prints the chosen mode and N.
                Defaults to True.

        Note:
            Sets the following instance attributes:
                - self._global_use_series (bool): Whether to use series expansion.
                - self._series_terms (int): Number of series terms to use.

            Uses self._wavelength and self._phase_tol_rad (default 1e-3 rad if unset).
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

        # Phase tolerance
        phi_tol = float(getattr(self, "_phase_tol_rad", 1e-3))

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

    def build_interaction_kernel(self, series_terms: int | None = None,
                                 force_mode: str | None = None,
                                 m_beams: int = 1):
        """
        Build and cache the FP32-only kinematic interaction CUDA kernel.

        Compiles a CUDA kernel for kinematic scattering with optional analyser
        and spherical decay support baked as runtime arguments.

        Args:
            series_terms (int or None, optional): Number of series terms for the
                sqrt(1+t)-1 expansion. If None, uses self._series_terms.
                Clamped to [1, 32].
            force_mode (str or None, optional): If "series", forces series mode;
                if "exact", forces exact mode. If None, uses self._global_use_series.

        Returns:
            cupy.RawKernel: Compiled CUDA kernel handle for the interaction kernel.

        Raises:
            RuntimeError: If CuPy is not available.

        Note:
            Analyser parameters are passed at kernel launch time:
                - apply_analyser: int (0/1)
                - analyser_kind: int (0=off, 1=top_hat, 2=darwin)
                - centre_dir: float3, unit vector (origin -> detector centre)
                - accept_angle_rad: float
                - darwin_halfwidth_rad: float

            The kernel scales each event's complex amplitude by:
                - top_hat: 1 if angle(out_dir, centre_dir) <= accept_angle_rad else 0
                - darwin: 1 / (1 + (delta / darwin_halfwidth_rad)^2)
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

        M_compile = int(max(1, m_beams))

        # Cache by (N, mode, M)
        if not hasattr(self, "_interaction_kernel_cache"):
            self._interaction_kernel_cache = {}
        key = ("v3_dynamical", N, global_use_series, M_compile)
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

        // M_BEAMS_COMPILE controls the per-atom amplitude vector length.
        // M=1 reproduces the kinematic kernel bit-identically (lattice-phase
        // factor is exp(2*pi*i*0*r)=1 for the forward beam).
        #ifndef M_BEAMS_COMPILE
        #define M_BEAMS_COMPILE 1
        #endif
        #if M_BEAMS_COMPILE < 1
        #undef M_BEAMS_COMPILE
        #define M_BEAMS_COMPILE 1
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

        // FP32-robust sincos(2*pi * g . r_a) using exact-split arithmetic.
        // Required for sample-scale crystals at hard-X-ray reflections, where
        // g.r_a may reach ~10^7 cycles. Algorithm:
        //   (1) two_prod_fma each component product (g_i * a_i) into hi+lo
        //   (2) Two-sum accumulation of three high parts (captures cancellation)
        //   (3) Total low = sum of (h_err1, h_err2, low_x, low_y, low_z)
        //   (4) Reduce modulo 1 cycle: q = round(s2 + low), frac = (s2 - q) + low
        //   (5) Multiply frac by 2*pi via TWOPI_H + TWOPI_L double-FP32 split,
        //       then __sincosf on the result (|arg| <= pi).
        // For g = (0, 0, 0) returns exactly (sn=0, cs=1), preserving M=1
        // bit-identical kinematic behavior.
        __device__ __forceinline__ void sincos_2pi_dot_g_r(
            float gx, float gy, float gz,
            float ax, float ay, float az,
            float& sn, float& cs)
        {
            // Step 1: exact-split component products
            float hx, lx; two_prod_fma(gx, ax, hx, lx);
            float hy, ly; two_prod_fma(gy, ay, hy, ly);
            float hz, lz; two_prod_fma(gz, az, hz, lz);

            // Step 2: two-sum accumulation of the high parts
            float s1 = hx + hy;
            float bb = s1 - hx;
            float e1 = (hx - (s1 - bb)) + (hy - bb);

            float s2 = s1 + hz;
            bb = s2 - s1;
            float e2 = (s1 - (s2 - bb)) + (hz - bb);

            // Step 3: total = s2 (high) + low (small)
            float low = e1 + e2 + lx + ly + lz;

            // Step 4: reduce modulo 1 cycle
            float q = nearbyintf(s2 + low);
            float frac = (s2 - q) + low;          // |frac| <= 0.5 cycles

            // Step 5: multiply by 2*pi using TWOPI_H + TWOPI_L double-split,
            // then sincos.  |arg| <= pi, where __sincosf is FP32-accurate.
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

        // analyser_kind: 0=off, 1=top_hat, 2=darwin
        // g_vecs_in: (M_BEAMS_COMPILE * 3) floats in 1/m. Provides the
        // reciprocal-lattice vectors for the per-atom M-channel coherent
        // sum (Eq. 13 of the dynamical-method plan). For M=1 with all
        // zeros the kernel reduces bit-identically to the kinematic case.
        // initial_amp must have length nAtoms * M_BEAMS_COMPILE, indexed
        // as initial_amp[a * M + m].
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
            const float* __restrict__ g_vecs_in)  // (M * 3) floats in 1/m
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

            // Shared memory and thread indexing — ALL threads must participate
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

                // Only valid pixels process atoms
                if (valid_pixel) {
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

                        // Build scattering factor and optionally remove full forward amplitude
                        float2 s_tot;
                        s_tot.x = f0v + s_anm[j].x;
                        s_tot.y = s_anm[j].y;

                        if (remove_forward && (Q_val < Q_cut)) {
                            s_tot.x -= (s_f0z[j] + s_anm[j].x);
                            s_tot.y -= (s_anm[j].y);
                        }

                        // M-channel coherent sum with lattice-phase factors
                        // (Eq. 13 first term of the dynamical-method plan).
                        // For M=1 with g=(0,0,0), sincos_2pi_dot_g_r returns
                        // (sn=0, cs=1) and amp reduces to s_amp_M[j*1+0],
                        // bit-identical to the legacy single-amplitude path.
                        float2 amp = make_float2(0.0f, 0.0f);
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

                        // Optional analyser
                        if (apply_analyser) {
                            // unit direction from atom -> pixel
                            float inv_rd = 1.0f / r_det;
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
                            amp_rel = (R0 > 0.0f) ? (R0 / r_det) : 1.0f;
                        }

                        sum_rel.x += val.x * rE_F * amp_rel;
                        sum_rel.y += val.y * rE_F * amp_rel;
                    }
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
            )
        )
        kern = kernel_module.get_function('interaction_kernal')
        self._interaction_kernel_cache[key] = kern
        return kern
    
    @staticmethod
    def build_ein_sampler_kernel():
        """
        Build the CUDA kernel that bilinearly samples E0, tau, and phi on the beam grid.

        For each input position, the kernel:
        1) Projects to beam-basis coordinates (u, v).
        2) Performs bilinear sampling of tau, phi, and E0.
        3) Computes Ein = E0 * exp(-f * tau) * exp(i * f * phi) with
            f derived from the depth fraction along the beam direction.

        Args:
            None

        Returns:
            cupy.RawKernel: Compiled kernel handle named "ein_bilinear_kernel".

        Raises:
            RuntimeError: If CuPy is not available.

        Notes:
            - Out-of-bounds samples are set to zero (no edge clamping).
            - The kernel expects Angstrom units for positions and grid spacings.
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

    def _auto_select_beams(self, crystal, stage, M_max=2):
        """
        Select up to M_max Bragg reflections closest to the Ewald sphere.

        Enumerates reciprocal lattice vectors of the crystal (using the
        conventional cell), transforms them by sample orientation and stage
        rotation, and ranks them by excitation error.  The forward beam
        (g = 0) is always included as beam index 0.

        Args:
            crystal: Crystal object providing lattice_matrix_conventional (3x3,
                rows = a, b, c in Angstrom) and lattice_volume_conventional.
            stage: Stage object providing rotation (3x3).
            M_max (int): Maximum number of beams including the forward beam.

        Returns:
            list[dict]: Length-M list of beam descriptors, each containing:
                - "hkl": tuple (h, k, l)
                - "G": ndarray (3,), reciprocal lattice vector in crystallographic
                       convention (1/Angstrom, no 2pi).  Used for structure factor
                       phase: exp(-2*pi*i * G . r).
                - "k_vec": ndarray (3,), beam wavevector k0 + 2*pi*G in physics
                       convention (2*pi/lambda, 1/Angstrom).
                - "excitation_error": float |s_g| in 1/Angstrom
        """
        two_pi = 2.0 * np.pi
        lam_A = float(self._wavelength) * 1e10  # meters -> Angstrom
        k_mag = two_pi / lam_A  # 1/Angstrom

        # Incident wavevector in lab frame (along +x)
        k_hat = np.asarray(self._direction, dtype=np.float64)
        k_hat = k_hat / np.linalg.norm(k_hat)
        k0 = k_mag * k_hat  # (3,)

        # Lattice vectors in sample frame -> lab frame.
        # ``lattice_matrix_conventional`` is stored with COLUMNS as the
        # cartesian lattice vectors (a, b, c) -- this matches Crystal's
        # ``rotate_crystal`` which updates the matrix as ``R @ M`` (rotates
        # each column by R).  ``stage.rotation`` is then applied as a
        # left-multiply on each column too.
        R_stage = np.asarray(stage.rotation, dtype=np.float64)
        lat_conv = np.asarray(crystal.lattice_matrix_conventional, dtype=np.float64)
        # Lattice vectors in lab frame, columns = (a, b, c)_lab.
        lat_lab = R_stage @ lat_conv

        V_cell = float(crystal.lattice_volume_conventional)
        # Reciprocal lattice vectors (1/A, no 2pi): b_i* = cross(a_j, a_k) / V.
        # Use COLUMNS of lat_lab (lat_lab[:, k]) as the cartesian a/b/c vectors.
        recip = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            recip[i] = np.cross(
                lat_lab[:, (i + 1) % 3], lat_lab[:, (i + 2) % 3]
            ) / V_cell

        # Enumerate candidate (h,k,l) within accessible range
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
        # Annotate per-beam obliquity factor cos(theta_g) used by the
        # angular-spectrum propagator (Eq. 11 of the dynamical-method plan).
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
        Auto-detect Bragg reflections from atom positions without crystal info.

        Computes a 3D FFT of the binned atomic density to find peaks in
        reciprocal space, then selects those with the smallest excitation
        error (closest to the Ewald sphere).  Falls back to M=1 if no
        significant peaks are found (e.g. amorphous material).

        Args:
            sample: Chunked sample object.
            stage: Stage object with rotation (3x3) and translation (3,).
            M_max (int): Maximum number of beams including the forward beam.
            n_subsample (int): Maximum atoms to use for the FFT (random
                subsample if the total exceeds this).
            fft_N (int): Number of bins per dimension for the density grid.

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

        # Collect atom positions
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

        # Subsample
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

        # Fractional grid coordinates -> bin indices
        frac = (pos_sub - grid_min) / grid_extent  # 0..1
        ijk = np.clip((frac * N).astype(np.int64), 0, N - 1)
        density = np.zeros((N, N, N), dtype=np.float64)
        np.add.at(density, (ijk[:, 0], ijk[:, 1], ijk[:, 2]), 1.0)

        # 3D FFT
        F = np.fft.fftn(density)
        F_mag = np.abs(F)
        F_mag[0, 0, 0] = 0.0  # remove DC

        # Reciprocal-space coordinates (1/Angstrom, crystallographic convention)
        dx = grid_extent / N  # real-space pixel size per axis
        freq = [np.fft.fftfreq(N, d=dx[i]) for i in range(3)]

        # Find peaks above threshold
        threshold = 0.3 * F_mag.max()
        if threshold < 1e-12:
            return [forward_beam]

        # Pre-compute squared-magnitude grid for accessibility filter
        GX, GY, GZ = np.meshgrid(freq[0], freq[1], freq[2], indexing='ij')
        G_phys_mag_sq = (two_pi ** 2) * (GX * GX + GY * GY + GZ * GZ)
        four_k_sq = (2.0 * k_mag) ** 2

        peak_mask = (F_mag > threshold) & (G_phys_mag_sq > 0.01) & (G_phys_mag_sq < four_k_sq)
        peak_idx = np.argwhere(peak_mask)

        if len(peak_idx) == 0:
            return [forward_beam]

        # Build candidates with excitation error
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
        Build beam descriptors from user-supplied G vectors.

        Args:
            g_vectors: Iterable of (3,) arrays, each a reciprocal-lattice
                vector in crystallographic convention (1/Angstrom, no 2pi).
            k0: ndarray (3,), incident wavevector (2pi/lambda, 1/Angstrom).
            k_mag: float, |k0|.

        Returns:
            list[dict]: Beam descriptors (forward beam + one per g_vector).
        """
        two_pi = 2.0 * np.pi
        # Reference forward direction (unit) for the cos(theta) calculation.
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
        Build per-slice complex structure factor maps for each unique delta-g
        vector needed by the beam coupling matrix.

        For M beams with reciprocal lattice vectors g_0, ..., g_{M-1}, the
        coupling matrix element A_{ab} requires the susceptibility chi_{g_a - g_b}.
        This method deposits atoms with complex phase factors
        exp(-2*pi*i * delta_g . r_atom) weighted by f(|delta_g|) onto the beam
        grid using TSC interpolation, for each unique delta_g.

        Two normalization conventions are supported:

        - ``born_convention=False`` (default, legacy column-integral form):
          maps use prefactor ``r_e * lambda^2 / (2 * pi * du * dv)``.  This is
          the form historically consumed by ``_beam_coupling_step_gpu`` where
          the transmission propagator is applied as ``exp(i * k * chi)``
          without an explicit dz factor.

        - ``born_convention=True`` (Born/Authier voxel-density form):
          per-slice maps use prefactor ``r_e * lambda^2 / (pi * du * dv * dz_k)``
          where ``dz_k = slice_edges_A[k+1] - slice_edges_A[k]``.  Maps are
          true voxel-density susceptibilities (Eq. 7 of the dynamical-method
          plan).  The transmission propagator must then be applied with an
          explicit dz factor: ``exp(i * k * chi * dz)``.

        Args:
            sample: Chunked sample object.
            stage: Stage object with rotation and translation.
            slice_edges_A: (n_slices+1,) array of depth edges in Angstrom.
            beam_info: List of beam descriptors from _auto_select_beams.
            kernel_radius: Gaussian blur radius in pixels (0 = off).
            born_convention: Use the corrected per-slice voxel-density
                normalization.

        Returns:
            dict: Maps (a, b) tuples to lists of n_slices complex64 NumPy
                arrays, each of shape (NyB, NzB).
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

        # Unique delta_g vectors
        unique_dg = {}
        for a in range(M):
            for b in range(M):
                dg = beam_info[a]["G"] - beam_info[b]["G"]
                unique_dg[(a, b)] = dg.astype(np.float64)

        # Databases
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

        # Initialize accumulators
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

        # Apply prefactor (per-slice for born_convention, scalar otherwise)
        # and optional blur.
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
        Build a CUDA kernel for 2x2 beam coupling via matrix exponential.

        The kernel applies the exact closed-form 2x2 matrix exponential
        of the coupling matrix M = i*k*dz * [[chi0, chi_mh], [chi_h, chi0]]
        to update the two beam wavefields E0, E1 at each grid point.

        Returns:
            cupy.RawKernel: Compiled kernel "beam_couple_2x2_kernel".
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
        Apply per-pixel matrix-exponential transmission step (Eq. 10 of the
        dynamical-method plan) on GPU.

        For M=1: applies the scalar update E0 *= exp(i*k_A*chi0).
        For M=2: applies the closed-form 2x2 matrix exponential of
                 i*k_A*[[chi0, chi_-h], [chi_h, chi0]].
        For M>2: not yet implemented (PadÃ© path deferred).

        Note on units: ``k_A`` should equal ``k * dz`` for true voxel-density
        chi maps (born_convention=True in _build_structure_factor_maps_gpu).
        For legacy column-integral chi maps, ``k_A`` should equal ``k`` only
        (with no dz factor); the dz is already absorbed into the chi values.

        Args:
            E_beams: list of M CuPy arrays, each (NyB, NzB) complex64.
            chi_maps_slice: dict mapping (a,b) -> complex64 GPU array (NyB, NzB).
            k_A: float, transmission-step phase prefactor.

        Returns:
            list of M CuPy arrays (modified in-place, also returned).
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
        Per-beam angular-spectrum propagation by ``dz_A`` Angstrom with
        per-beam carrier-wave subtraction (Eq. 11 of the dynamical-method
        plan, exact form).

        Each beam m has its own carrier wave with k-vector beam_info[m]['k_vec'].
        The propagator subtracts this carrier exactly, so the envelope phase
        evolution for the Bragg beam (g != 0) preserves the chi_h voxel maps'
        exp(-i 2 pi G_h . r) carrier across slices.  This is the key to coherent
        E_g buildup over many slices.

        Args:
            E_beams: list of M CuPy arrays (NyB, NzB) complex64.
            dz_A: slice thickness in Angstrom.
            beam_info: list of M beam descriptors providing 'k_vec' (1/Angstrom,
                with 2 pi) and 'cos_theta' (legacy fallback if k_vec missing).

        Returns:
            list of M propagated CuPy arrays.
        """
        if not hasattr(self, '_prop_kernel_cache') or self._prop_kernel_cache is None:
            self._prop_kernel_cache = self.build_propagation_multiplier_kernel()
        kernel = self._prop_kernel_cache

        # Axis convention (critical -- see below).  The envelope arrays are
        # shaped (_beam_Ny, _beam_Nz) = (axis0 = e1/u, axis1 = e2/v).
        # _angular_spectrum_propagate_gpu does `Nz, Ny = F.shape[0],
        # F.shape[1]`, then builds ky over Ny (= axis1 = e2/v) with the
        # `dy` arg and kz over Nz (= axis0 = e1/u) with the `dz` arg; the
        # multiplier kernel pairs k_g_perp_y with ky and k_g_perp_z with kz.
        # So to stay consistent the e2/v quantities must be passed as the
        # `dy`/`k_g_perp_y` arguments and the e1/u quantities as `dz`/
        # `k_g_perp_z`.  (Passing them the other way round subtracts the
        # Bragg carrier on the wrong grid axis -> E_g de-phases per slice
        # and never builds coherently -> no Pendellosung.)
        dy_m = float(self._beam_dv) * 1e-10      # e2/v pitch -> ky / axis1
        dz_pix_m = float(self._beam_du) * 1e-10  # e1/u pitch -> kz / axis0
        dz_m = float(dz_A) * 1e-10

        # Beam-grid axes (in lab frame).  Convert k from 1/A to rad/m.
        k_hat = np.asarray(self._direction, dtype=np.float64)
        k_hat = k_hat / np.linalg.norm(k_hat)
        e1 = np.asarray(self._beam_e1, dtype=np.float64)
        e2 = np.asarray(self._beam_e2, dtype=np.float64)
        # Convert 1/A (with 2 pi) to rad/m: multiply by 1e10.
        for m in range(len(E_beams)):
            kvec_iA = np.asarray(beam_info[m].get('k_vec', None), dtype=np.float64)
            if kvec_iA is None or kvec_iA.size != 3:
                # Fallback to cos_theta legacy path
                cos_theta_m = float(beam_info[m].get("cos_theta", 1.0))
                E_beams[m] = self._angular_spectrum_propagate_gpu(
                    E_beams[m], dy_m, dz_pix_m, dz_m, kernel,
                    step_max=0.02, pad_factor=1.0, padding_mode="edge",
                    cos_theta=cos_theta_m,
                )
                continue
            # Project k_vec onto beam-grid axes (1/A with 2 pi).  The
            # carrier-subtraction kernel pairs k_g_perp_y with ky (= field
            # axis1 = e2/v) and k_g_perp_z with kz (= field axis0 = e1/u),
            # so the e2 component must go to k_g_perp_y and the e1 component
            # to k_g_perp_z (see the axis-convention note above).
            k_g_axis_iA = float(np.dot(kvec_iA, k_hat))
            k_g_perp_y_iA = float(np.dot(kvec_iA, e2))  # ky <-> e2/axis1
            k_g_perp_z_iA = float(np.dot(kvec_iA, e1))  # kz <-> e1/axis0
            # Convert to rad/m for the kernel (which expects SI units consistent
            # with dy_m, dz_pix_m, dz_m).  k [1/A with 2 pi] * 1e10 = k [rad/m].
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
        Backward-compatibility wrapper.  Calls transmission-only by default
        (legacy semantics).  When ``dz_A`` and ``beam_info`` are both
        provided, additionally applies the per-beam ASP propagation step
        (Lie-Trotter split-step).

        Args:
            E_beams: list of M CuPy arrays (NyB, NzB) complex64.
            chi_maps_slice: dict (a,b) -> complex64 GPU array (NyB, NzB).
            k_A: transmission-step phase prefactor.
            dz_A: optional slice thickness in Angstrom for ASP propagation.
            beam_info: optional list of M beam descriptors (need cos_theta).

        Returns:
            list of M CuPy arrays.
        """
        E_beams = self._beam_transmission_step_gpu(E_beams, chi_maps_slice, k_A)
        if dz_A is not None and beam_info is not None:
            E_beams = self._beam_propagation_step_gpu(E_beams, dz_A, beam_info)
        return E_beams

    # -------------------------------------------------------------------------
    # FP32-robust atom-table builder for the dynamical multislice (Phase 4.2
    # of the dynamical-method plan).  The integer/fractional voxel-coordinate
    # split uses _two_prod_fp32 and _two_sum_fp32 so atoms at sample-scale
    # positions (~mm) still resolve sub-voxel weights to FP32 precision.
    # -------------------------------------------------------------------------
    def _build_atom_table_for_multislice(self, sample, stage, edges_A, n_final):
        """
        Build per-chunk FP32-robust atom tables for the multislice + LS pipeline.

        The previous implementation concatenated every per-chunk array into one
        global aggregated buffer (all_pos, all_iu_int, ...).  For
        billion-atom samples this peaks at >20 GB host RAM (positions +
        int64 indices + fractions + species + atom_M_amps), which OOMs the
        process.  This version keeps each chunk's atom table independent so
        the multislice loop and the LS step can iterate per-chunk and keep
        peak memory bounded by the largest single chunk (~30 MB for 1M
        atoms).

        ``iu_int`` / ``iv_int`` are stored as int32 instead of int64 to
        halve their memory footprint -- voxel grids never exceed 2^31 - 1
        cells per axis in practice.

        Args:
            sample: Chunked sample object.
            stage: Stage with rotation (3x3) and translation (3,) in Angstrom.
            edges_A: (n_final + 1,) array of slice depth edges in Angstrom.
            n_final: Number of slices.

        Returns:
            dict mapping ``cid`` (int chunk id, 1-indexed) to a per-chunk
            table dict with keys:

                all_pos    (N_chunk, 3) float32 lab-frame positions (A)
                all_spc    (N_chunk,) species labels
                iu_int     (N_chunk,) int32 voxel integer indices (sorted by slice)
                iv_int     (N_chunk,) int32 voxel integer indices (sorted by slice)
                iu_frac    (N_chunk,) float32 fractional weights in [0, 1)
                iv_frac    (N_chunk,) float32 fractional weights in [0, 1)
                slice_starts (n_final + 1,) int64 slice-boundary indices
                              into THIS chunk's sorted arrays
                N_total    int  -- atoms in this chunk

            Plus a special key ``"_meta"`` with:
                N_total_all int  -- total atoms across all chunks
                chunk_ids   list[int] -- ordered list of populated chunk ids

            Returns None if no atoms.
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

            # Extract integer + fractional parts (FP32-robust).  We keep the
            # integer indices in int32 to halve memory vs the previous int64
            # implementation -- voxel-grid sizes never need more.
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

            # Sort THIS chunk's atoms by slice in place; keeps slice atoms
            # contiguous so the multislice loop can do range slices directly.
            sort_idx = np.argsort(k_idx, kind='stable')
            pos_lab = pos_lab[sort_idx]
            spc_host = np.asarray(spc_host)[sort_idx]
            iu_int = iu_int[sort_idx]
            iv_int = iv_int[sort_idx]
            iu_frac = iu_frac[sort_idx]
            iv_frac = iv_frac[sort_idx]
            k_idx = k_idx[sort_idx]

            # Per-chunk slice_starts indexing into THIS chunk's arrays.
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
        Bilinear-interpolate the M envelope wavefields at atom positions.

        Consumes the FP32-robust pre-split (int, frac) coordinate schema from
        ``_build_atom_table_for_multislice``.  Atoms outside the grid are
        zeroed.  Modifies ``out_M_amps_slice`` in place.

        Args:
            E_beams_gpu: list of M CuPy complex64 arrays (NyB, NzB).
            iu_int, iv_int: (N_atoms_in_slice,) int64 voxel indices.
            iu_frac, iv_frac: (N_atoms_in_slice,) float32 weights in [0, 1).
            out_M_amps_slice: (N_atoms_in_slice, M) numpy complex64 (modified).
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
        Stage-2 multislice driver with inline Stage-3 envelope sampling.

        Implements Eq. (9-11) of the dynamical-method plan. Iterates the
        Lie-Trotter split-step transmission + ASP propagation through the
        crystal, sampling envelopes at atom positions inside each slice as
        they are encountered.

        Forward semantics (slice_iter_reverse=False):
            Loop k = 0, 1, ..., n_final - 1.
            Per slice: T(+dz, chi) then P(+dz), then sample atoms in slice k.
            Initial condition: E_0 = self._beam_E0_map (forward beam = unity),
            E_m = 0 for m > 0, unless initial_envelope is provided.

        Reverse semantics (slice_iter_reverse=True), used by the reciprocal
        multislice for output-side correction (master_plan II.7, Lorentz
        reciprocity):
            Loop k = n_final - 1, n_final - 2, ..., 0.
            Per slice: P(-dz) then T(+dz, chi), then sample atoms in slice k.
            chi maps are the SAME as forward (no conjugation -- Lorentz
            reciprocity in absorbing media uses the symmetric, non-Hermitian
            structure of the Helmholtz operator; chi -> chi* is the wrong
            time-reversal duality which yields the gain-medium inverse).
            Empirically validated in validation_simulation.py to 0.16% rel err
            on the forward + physical-backward round-trip symmetric mode.
            initial_envelope must be supplied (the entrance wave at the EXIT
            face of the crystal, in the -k_out direction).

        Args:
            chi_maps: dict (a, b) -> list of n_final complex64 NumPy arrays
                each shaped (NyB, NzB).  Voxel-density susceptibility maps
                from _build_structure_factor_maps_gpu(born_convention=True).
            beam_info: list of M beam descriptors (provides cos_theta for
                the propagation step).
            transmission_k_A: float, transmission-step phase prefactor
                (= pi / lambda * dz_A for born-convention chi maps).
            dz_A: float, slice thickness in Angstrom (always positive; the
                helper handles sign internally based on slice_iter_reverse).
            n_final: int, number of slices.
            per_chunk_tables: dict cid -> table with iu_int, iv_int, iu_frac,
                iv_frac, slice_starts (built by _build_atom_table_for_multislice).
            atom_M_amps_per_chunk: dict cid -> (N_chunk, M) complex64 array
                (modified in-place by Stage 3).
            NyB, NzB: int, beam-grid dimensions.
            M: int, number of beams.
            thickness_A: float, total crystal thickness in Angstrom.
            apply_propagation: if True, apply ASP step (default).  Disabled
                automatically for thickness-degenerate cases.
            slice_iter_reverse: if True, run the physical-backward variant
                described above.
            initial_envelope: optional list of M CuPy complex64 arrays of
                shape (NyB, NzB), used as the initial condition.  If None,
                defaults to (self._beam_E0_map, 0, 0, ...).

        Returns:
            None.  atom_M_amps_per_chunk modified in-place.
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

        # Slice iteration order
        slice_order = range(n_final - 1, -1, -1) if slice_iter_reverse else range(n_final)

        for k in slice_order:
            chi_slice = {key: cp.asarray(chi_maps[key][k], dtype=cp.complex64)
                         for key in chi_maps}

            if slice_iter_reverse:
                # Reverse: P(-dz) then T(+dz, chi).  This is the *physical*
                # backward wave through the same absorbing medium, NOT the
                # mathematical inverse of forward.
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

            # Sample envelopes at atoms in slice k of every chunk that has
            # atoms in this slice.  Same call for both forward and reverse:
            # we sample AFTER the slice's operations have been applied.
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
        Atomistic dynamical X-ray scattering via voxelized multi-beam
        multislice with per-atom Lippmann-Schwinger rescattering.

        Implements the master equation (Eq. 6 of the dynamical-method plan):
            psi(r_det) = psi_0(r_det) - (r_e * lambda^2 * k_0^2 / pi)
                         * sum_a F_a(Q) * G_0(r_det, r_a) * psi_dyn(r_a)
        where psi_dyn is recovered by trilinear sampling of the multislice
        envelope wavefields E_g_m at every atom position and recombining via
        the per-(atom, beam, pixel) phase
            Phi_{a,m,p} = 2*pi * g_m . r_a + k_0 |r_det - r_a| - k_in . r_a.

        Recovers vacuum kinematic (M=1, chi=0), refractive forward (M=1,
        amorphous), Beer-Lambert absorption (M=1, absorbing), two-beam
        Pendelloesung (M=2, perfect crystal at exact Bragg), Borrmann
        anomalous transmission (M=2, absorbing centrosymmetric crystal),
        and Takagi-Taupin defect imaging (M=2, defected crystal) as exact
        limits.

        Beam selection (M > 1) priority:
            1. Explicit ``g_vectors``: user-supplied reciprocal-lattice list.
            2. ``crystal`` object: nearest reflections to the Ewald sphere.
            3. Auto-detect: 3D FFT of atomic density (amorphous fallback to M=1).

        Args:
            sample: Chunked sample object.
            detector: Detector with ``pixel_coordinates`` (3, Ny*Nz) in Angstrom.
            stage: Stage providing ``rotation`` (3x3) and ``translation`` (3,).
            crystal: Optional crystal for fast beam selection.
            M (int): Number of coupled beams (1 = kinematic + transmission).
            g_vectors: Optional list of (3,) crystallographic-convention G
                vectors (1/Angstrom, no 2pi).
            offset: Optional complex field to subtract from the result.
            use_gpu (bool): Use GPU acceleration.
            n_slices (int or None): Number of depth slices; auto if None.
            target_phase_step (float): Per-slice phase step for auto-slicer.
            kernel_radius (int): Gaussian blur radius for chi maps.
            pad_factor (float): FFT padding factor for ASP propagation.
            padding_mode (str): "edge" or "constant".
            absorption_multiplier (float): Scale absorption (1.0 = physical).
            apply_polarization (bool): Apply per-pixel polarization factor.
            remove_forward (bool): Remove forward scattering at the kernel.
            spherical_decay (bool): Apply 1/R decay.
            apply_propagation (bool): If True, apply per-beam ASP propagation
                step in the Lie-Trotter split-step (Eq. 11).  If False, the
                multislice is transmission-only, useful for column-approximation
                regimes (uniform crystal) and for backward compatibility.
            NN_dist_A (float or None): Nearest-neighbor distance in Angstrom,
                used by the convergent-regime check (master_plan I.6).
                If None and M > 1, estimated from a random subset of sample
                atoms; if estimation fails, the check is skipped with a
                warning.
            convergent_regime_check (str): One of {"off", "warn", "error"}.
                Controls behaviour when the existing beam grid pitch falls
                outside the convergent window [NN_dist/3, 1/(2*|g_max|)] for
                M > 1.  Default "warn".  Use "error" for validation pipelines
                that must not silently produce kinematic-only or biased
                output.
            commensurate_supercell (bool): If True, allow aggressive-mode
                voxel pitch (down to NN_dist/2) for perfect commensurate
                supercells.
            force_unconverged (bool): If True, suppress the RuntimeError
                that would otherwise be raised when |g_max|*NN_dist >= 1.5
                (collapsed window).  For diagnostic use only.
            dynamical_mode (str): One of {"forward_only", "full"}.
                "forward_only" (default) runs a single forward multislice
                and uses kinematic free-space output-side propagation
                (column approximation; correct for thin samples and weak
                reflections).  "full" additionally runs a reciprocal
                multislice via Lorentz reciprocity (master_plan II.7) to
                capture output-side dynamical corrections; the per-atom
                amplitude becomes the elementwise product of forward and
                reciprocal envelopes.  At symmetric Laue exact Bragg the
                two modes give the same answer at the Bragg peak (the
                output-side correction "absorbs" into the per-atom factor
                already present from the forward multislice exit value);
                differences are visible at off-Bragg detector pixels and
                in asymmetric geometries.
            multi_gpu (bool): If True, parallelise the Stage-4 LS kernel
                pass across all available CUDA devices.  Stages 1-3
                (chi_g build, multislice, atom-position envelope sampling)
                are replicated per the master_plan II.6a "default
                replicated" recommendation; only the LS pass is partitioned.
                Bit-identical to single-GPU within FP32 reduction-order noise.
                Default False.
            n_gpus (int or None): If multi_gpu=True, cap the number of GPUs
                used.  Default None = use all available.

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
        # The existing beam grid (du, dv) was set up by _init_beam_grid; for
        # M > 1 we validate that its in-plane pitch falls inside the convergent
        # window [NN_dist/3, 1/(2*|g_max|)].  Outside this window the
        # simulation runs but produces silently wrong results:
        #   - too coarse (du > 1/(2|g_max|)):  chi_g aliased -> refractive
        #     only; multislice misses Bragg-channel coupling, recovers
        #     kinematic Born series.
        #   - too fine   (du < NN_dist/3):     bimodal chi field -> per-atom
        #     amplitudes biased by 30-40%.
        # This check surfaces the issue via a UserWarning (or RuntimeError if
        # convergent_regime_check="error"); it does NOT auto-resize the beam
        # grid, since that would silently invalidate caller-managed grid
        # setup downstream of this method.
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
        # M=1: legacy column-integral chi_0 from _compute_beam_slice_integrals_*.
        #      Transmission step uses k_A = kA = 2*pi/lambda (no dz; dz is
        #      absorbed into the column-integral chi).  This is the legacy
        #      convention, kept for backward compatibility with the M=1 code
        #      path of the simulator (atomic_transmission, etc.).
        # M>1: voxel-density Born-convention chi_g maps.  Per Eq. 10 of the
        #      dynamical-method plan, the transmission propagator is
        #      exp(-i * pi * dz / lambda * X), so the transmission-step
        #      phase prefactor is k_A = pi / lambda * dz_A (NOT 2*pi/lambda *
        #      dz_A -- the (1/pi) in the chi prefactor (Eq. 7) is paired
        #      with the (pi) in the transmission propagator; using 2*pi
        #      doubles the Pendellosung frequency).
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
            # Eq. 10:  E(z+dz) = exp(-i * pi * dz / lambda * X) E(z)
            # so the kernel prefactor (kernel multiplies M = i * k_dz * X)
            # must equal (pi / lambda) * dz, not (2*pi / lambda) * dz.
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

        # Per-chunk M-vector amplitudes (host-resident).  Each chunk's
        # buffer is at most ~M * (typical chunk atom count) * 8 bytes
        # = ~16 MB for M=2 and 1M atoms, so total host footprint is bounded
        # by the same ~M * N_total * 8 bytes as the previous concatenated
        # buffer -- the gain is that the multislice and LS steps below can
        # process one chunk at a time and never need to touch the full
        # array on the GPU.
        atom_M_amps_per_chunk = {
            cid: np.zeros((per_chunk_tables[cid]["N_total"], M),
                          dtype=np.complex64)
            for cid in chunk_ids
        }

        # ---------------- 6. Multislice loop (Lie-Trotter split-step) ------
        # Forward pass: builds A_g_F (input-side dynamical illumination at
        # every atom) into atom_M_amps_per_chunk via inline Stage-3 sampling.
        self._run_multislice_with_sampling(
            chi_maps, beam_info, transmission_k_A, dz_A, n_final,
            per_chunk_tables, atom_M_amps_per_chunk,
            NyB, NzB, M, thickness_A,
            apply_propagation=apply_propagation,
            slice_iter_reverse=False,
            initial_envelope=None,
        )

        # ---------------- 6b. Reciprocal multislice (dynamical_mode="full") -
        # Output-side dynamical correction via Lorentz reciprocity
        # (master_plan II.7).  Runs a SECOND multislice physically backward
        # through the same absorbing medium with the entrance plane at the
        # exit face of the crystal.  The per-atom amplitude sampled from the
        # reciprocal envelopes (A_g_R) is multiplied elementwise into A_g_F
        # so that Stage 4 (the LS far-field kernel pass) sees the combined
        # illumination factor for each atom.
        #
        # At symmetric Laue exact Bragg the per-atom product
        #     A_g_F(z_a) * A_g_R(z_a)
        # is constant in z_a (= env_in); see master_plan II.7 z-cancellation
        # discussion.  The output-side correction is visible at off-Bragg
        # detector pixels, in asymmetric Laue, and at thick absorbing
        # crystals near the Borrmann condition.
        dyn_mode_str = str(dynamical_mode or "forward_only").lower().strip()
        if dyn_mode_str not in ("forward_only", "full"):
            warnings.warn(
                f"[atomic_scattering_dynamical] unknown dynamical_mode="
                f"{dynamical_mode!r}; defaulting to 'forward_only'.")
            dyn_mode_str = "forward_only"

        if dyn_mode_str == "full" and M > 1 and thickness_A > 0.0 and dz_A > 0.0:
            # Build the central detector direction k_out_central (used as the
            # virtual source direction for the reciprocal multislice).  The
            # detector pixel at the geometric centre of the array is the
            # natural choice: it minimises the per-output-direction sampling
            # error for compact detector ROIs.  For wide ROIs the user can
            # call the routine multiple times with different detector
            # crops and combine; multi-direction interpolation is left as a
            # follow-up optimisation (see implementation_plan 4b.3).
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

            # Reciprocal beam set: each reciprocal beam channel is centred on
            # k_in_recip + g_m where k_in_recip = -k_out_central.  The chi
            # maps are reused unchanged (Lorentz reciprocity, NOT chi -> chi*).
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

            # Initial envelope for the reciprocal pass: unit plane wave in
            # the forward (m=0) reciprocal channel, zero in the others.  The
            # pass marches backward from z = thickness_A toward z = 0, so the
            # "initial" envelope is the entrance condition at the exit face.
            initial_env_recip = [cp.ones((NyB, NzB), dtype=cp.complex64)]
            for _m in range(1, M):
                initial_env_recip.append(cp.zeros((NyB, NzB), dtype=cp.complex64))

            # Allocate a separate per-chunk buffer for the reciprocal-pass
            # amplitudes A_g_R; combine them with A_g_F afterwards.
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

            # Combine: A_g[a, m] <- A_g_F[a, m] * A_g_R[a, m] elementwise.
            # The Stage-4 kernel ingests the combined buffer unchanged.
            for cid in chunk_ids:
                atom_M_amps_per_chunk[cid] *= atom_M_amps_recip[cid]
            del atom_M_amps_recip
            del initial_env_recip
            cp.get_default_memory_pool().free_all_blocks()


        # ---------------- 7. Per-chunk LS far-field kernel pass ------------
        # We process the LS step ONE CHUNK AT A TIME so the per-atom GPU
        # buffers (positions, M-vector amplitudes, form factors, k-vectors)
        # are bounded by chunk size, not by N_total.  This is what allows
        # billion-atom samples to be processed without OOM-ing the GPU.

        db_f0 = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_f1f2 = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        # Precompute per-element f0(0) and anomalous (f1, f2) once -- these
        # are tiny dictionaries indexed by element symbol, not by atom.
        f0_zero_lookup = {}
        anom_lookup = {}
        for el_sym, f0p in db_f0.items():
            f0_zero_lookup[el_sym] = float(
                f0p[5] + f0p[0] + f0p[1] + f0p[2] + f0p[3] + f0p[4]
            )
        for el_sym, tbl in db_f1f2.items():
            anom_lookup[el_sym] = self.get_f1f2_from_params(self._energy, tbl)

        # Build kernel with this specific M (cached by build_interaction_kernel)
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

        # Pack g vectors (1/Angstrom) -> 1/m for the kernel.  Tiny GPU array,
        # constant across all atoms, allocated once for the whole pass.
        g_vecs_host = np.zeros((M, 3), dtype=np.float32)
        for m_idx in range(M):
            g_vecs_host[m_idx, :] = (
                np.asarray(beam_info[m_idx]["G"], dtype=np.float32) * 1e10
            )
        g_vecs_d = cp.asarray(g_vecs_host.ravel())

        # Forward beam wavevector (1/m) used by the kernel for the global
        # phase reference (sincos_k_times_reduced).  The lattice-vector
        # phase factors for m > 0 are added INSIDE the kernel via the
        # M-channel sum (Eq. 13).
        k_vec_A0 = beam_info[0]["k_vec"]
        k_in_x = np.float32(float(k_vec_A0[0]) * 1e10)
        k_in_y = np.float32(float(k_vec_A0[1]) * 1e10)
        k_in_z = np.float32(float(k_vec_A0[2]) * 1e10)

        # Detector coordinates (meters) -- constant across chunks.
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

        # Resolve analyser arguments (mirror atomic_direct_interaction's
        # convention).  ``analyser_mode``: "off" (default), "top-hat" /
        # "tophat" / "top_hat", or "darwin" / "rolloff".  When enabled, the
        # analyser centre direction is the unit vector from sample origin
        # to the geometric centre of the detector pixel array.
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

        # Centre direction for the analyser = unit vector from origin to the
        # geometric centre pixel of the detector.
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

        # Auto-size the scatter sub-chunk to use up to 90 % of the free
        # GPU memory at the time of the call.  Each scatter sub-chunk
        # allocates roughly:
        #   amp slice: nA * M * 8 bytes (complex64)
        #   pos / k / form-factor slices: ~50 bytes / atom (already on GPU)
        # so the GPU peak per call ~ nA * (50 + 8*M) bytes.  Dividing
        # 90 % of the free GPU memory by that gives the largest nA we
        # can afford; we cap at a generous 50M to bound kernel runtime.
        try:
            free_gpu_b, _ = cp.cuda.runtime.memGetInfo()
            bytes_per_atom = 50 + 8 * M
            SCATTER_CHUNK = int(min(
                50_000_000,
                max(500_000, (0.9 * free_gpu_b) // max(bytes_per_atom, 1))
            ))
        except Exception:
            SCATTER_CHUNK = 500_000

        # ---------------- 7b. Multi-GPU dispatch (optional) ----------------
        # If multi_gpu=True, partition chunk_ids across N_GPUs and run the
        # Stage-4 LS kernel pass on each GPU in a thread.  Stages 1-3 are
        # already complete (chi_g maps + multislice envelopes are sampled
        # into atom_M_amps_per_chunk above), so multi-GPU only parallelises
        # the LS kernel pass.  This matches master_plan II.6a's "default
        # replicated" mode: each GPU has full chi_g, full multislice work
        # is replicated (no parallelism gain there since multislice is
        # serial along z), and the LS kernel pass is partitioned.
        #
        # Bit-identical to single-GPU within FP32 noise: the LS sum is
        # mathematically additive over atom partitions, so summing the
        # partial detector arrays from each GPU recovers the single-GPU
        # answer exactly modulo the order of FP32 reductions.
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
            # Partition chunk_ids round-robin across GPUs
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

                # Per-device copies of the static detector and beam-set
                # state.  Each device builds its own kernel handle (cached
                # per-device by build_interaction_kernel via CuPy's per-
                # device kernel cache).
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
                    spc_chunk = tbl["all_spc"]
                    _f0p_host = np.zeros((N_chunk, 11), dtype=np.float32)
                    _f0z_host = np.zeros(N_chunk, dtype=np.float32)
                    _anom_host = np.zeros(N_chunk, dtype=np.complex64)
                    for el in np.unique(spc_chunk):
                        el_s = str(el)
                        mask_np = (spc_chunk == el_s)
                        f0p = db_f0.get(el_s)
                        if f0p is not None:
                            _f0p_host[mask_np] = f0p
                            _f0z_host[mask_np] = f0_zero_lookup.get(el_s, 0.0)
                        if el_s in anom_lookup:
                            _anom_host[mask_np] = anom_lookup[el_s]
                    _f0p_d = cp.asarray(_f0p_host)
                    _f0z_d = cp.asarray(_f0z_host)
                    _anom_d = cp.asarray(_anom_host)

                    pos_chunk = tbl["all_pos"]
                    _px = cp.asarray((pos_chunk[:, 0] / 1e10).astype(np.float32))
                    _py = cp.asarray((pos_chunk[:, 1] / 1e10).astype(np.float32))
                    _pz = cp.asarray((pos_chunk[:, 2] / 1e10).astype(np.float32))

                    _kx = cp.full((N_chunk,), k_in_x, dtype=cp.float32)
                    _ky = cp.full((N_chunk,), k_in_y, dtype=cp.float32)
                    _kz = cp.full((N_chunk,), k_in_z, dtype=cp.float32)

                    amps_chunk = atom_M_amps_per_chunk[cid]

                    for c_start in range(0, N_chunk, SCATTER_CHUNK):
                        c_end = min(c_start + SCATTER_CHUNK, N_chunk)
                        nA_sub = c_end - c_start
                        amp_M_slice = cp.asarray(
                            amps_chunk[c_start:c_end, :].reshape(-1)
                        )
                        _interaction_kernel(
                            grid2d, block2d,
                            (
                                np.int32(nA_sub),
                                _kx[c_start:c_end],
                                _ky[c_start:c_end],
                                _kz[c_start:c_end],
                                _px[c_start:c_end],
                                _py[c_start:c_end],
                                _pz[c_start:c_end],
                                amp_M_slice,
                                _anom_d[c_start:c_end],
                                _f0p_d[c_start:c_end],
                                _f0z_d[c_start:c_end],
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
                            )
                        )
                        cp.cuda.stream.get_current_stream().synchronize()
                        del amp_M_slice
                    del _f0p_d, _f0z_d, _anom_d, _px, _py, _pz, _kx, _ky, _kz
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

            # Sum partial detector arrays from each GPU
            final_result = np.zeros((Nz, Ny), dtype=np.complex64)
            for pr in partial_results:
                if pr is not None:
                    final_result += pr

            # Cleanup placeholder buffers from the single-GPU prep above
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
            # Build per-chunk per-atom form factors from the species labels.
            spc_chunk = tbl["all_spc"]
            f0_params_host = np.zeros((N_chunk, 11), dtype=np.float32)
            f0_zero_host = np.zeros(N_chunk, dtype=np.float32)
            anom_host = np.zeros(N_chunk, dtype=np.complex64)
            unique_elements = np.unique(spc_chunk)
            for el in unique_elements:
                el_s = str(el)
                mask_np = (spc_chunk == el_s)
                f0p = db_f0.get(el_s)
                if f0p is not None:
                    f0_params_host[mask_np] = f0p
                    f0_zero_host[mask_np] = f0_zero_lookup.get(el_s, 0.0)
                if el_s in anom_lookup:
                    anom_host[mask_np] = anom_lookup[el_s]

            # Per-chunk GPU arrays.  Allocated once per chunk; freed at
            # the end of the chunk loop.
            f0_params_chunk = cp.asarray(f0_params_host)
            f0_zero_chunk = cp.asarray(f0_zero_host)
            anom_chunk = cp.asarray(anom_host)
            del f0_params_host, f0_zero_host, anom_host

            pos_chunk = tbl["all_pos"]
            px_chunk = cp.asarray((pos_chunk[:, 0] / 1e10).astype(np.float32))
            py_chunk = cp.asarray((pos_chunk[:, 1] / 1e10).astype(np.float32))
            pz_chunk = cp.asarray((pos_chunk[:, 2] / 1e10).astype(np.float32))

            kx_chunk = cp.full((N_chunk,), k_in_x, dtype=cp.float32)
            ky_chunk = cp.full((N_chunk,), k_in_y, dtype=cp.float32)
            kz_chunk = cp.full((N_chunk,), k_in_z, dtype=cp.float32)

            atom_M_amps_chunk = atom_M_amps_per_chunk[cid]

            # Stream within this chunk in scatter-sub-chunks so the
            # GPU-side amp slice stays small even for jumbo chunks.
            for c_start in range(0, N_chunk, SCATTER_CHUNK):
                c_end = min(c_start + SCATTER_CHUNK, N_chunk)
                nA_sub = c_end - c_start
                amp_M_slice = cp.asarray(
                    atom_M_amps_chunk[c_start:c_end, :].reshape(-1)
                )
                interaction_kernel(
                    grid2d, block2d,
                    (
                        np.int32(nA_sub),
                        kx_chunk[c_start:c_end],
                        ky_chunk[c_start:c_end],
                        kz_chunk[c_start:c_end],
                        px_chunk[c_start:c_end],
                        py_chunk[c_start:c_end],
                        pz_chunk[c_start:c_end],
                        amp_M_slice,
                        anom_chunk[c_start:c_end],
                        f0_params_chunk[c_start:c_end],
                        f0_zero_chunk[c_start:c_end],
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
                    )
                )
                cp.cuda.stream.get_current_stream().synchronize()
                del amp_M_slice
                cp.get_default_memory_pool().free_all_blocks()

            # Free this chunk's GPU arrays before moving on.
            del f0_params_chunk, f0_zero_chunk, anom_chunk
            del px_chunk, py_chunk, pz_chunk
            del kx_chunk, ky_chunk, kz_chunk
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
        Ny: int, Nz: int, dy: float, dz: float,
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
                sin(theta_y_max) = min(1, lambda / (2*dy))
                sin(theta_z_max) = min(1, lambda / (2*dz))
            Then the required half-padding (meters) is
                pad_y = |z| * tan(theta_y_max)
                pad_z = |z| * tan(theta_z_max).
            Convert to pixels, apply a safety factor, and optionally round up to
            powers of two.

        Args:
            Ny (int): Original width size (Y, horizontal).
            Nz (int): Original height size (Z, vertical).
            dy (float): Pixel size along Y in meters.
            dz (float): Pixel size along Z in meters.
            wavelength (float): Wavelength in meters.
            z (float): Propagation distance in meters.
            safety (float, optional): Multiplicative safety factor for padding.
            enforce_pow2 (bool, optional): If True, round padded sizes to next
                power of two.
            min_pad_factor (float, optional): Minimum multiplicative growth factor
                applied to Ny and Nz regardless of geometric padding.

        Returns:
            tuple[int, int]: Padded sizes (Ny_pad, Nz_pad).
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

        # Enforce a minimum multiplicative padding if requested
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
        Build a CUDA kernel that multiplies a spectrum by the free-space propagator.

        For each spatial frequency (ky, kz), the kernel applies the exact
        carrier-subtracted propagator:
            H = exp(+i*z*(sqrt(k^2 - (kyt)^2 - (kzt)^2) - k_g_axis))   [propagating]
            H = exp(-|z|*sqrt(...) - i*z*k_g_axis)                      [evanescent]
        where (kyt, kzt) = (ky + k_g_perp_y, kz + k_g_perp_z) is the
        carrier-shifted transverse k.

        For the forward beam, pass k_g_axis=k, k_g_perp_y=k_g_perp_z=0,
        which recovers the on-axis form phase = z*(sqrt(k^2 - kt^2) - k).
        For the Bragg beam, pass k_g_axis=k_g . axis, k_g_perp_y=k_g . e_y,
        k_g_perp_z=k_g . e_z, which makes the propagator preserve the
        chi_h(r) voxel map's exp(-i*2*pi*G_h.r) carrier across slices,
        enabling coherent buildup of E_g in the multislice forward pass.

        The spectrum F is updated in place.

        Returns:
            cupy.RawKernel: Compiled kernel handle named "prop_mul_kernel".
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

            // Exact propagator with the beam's own carrier wave subtracted.
            // For an envelope E_g whose carrier is k_g, the FFT spatial frequency
            // qy, qz represents the deviation from carrier in the lab beam grid.
            // The total wave's transverse k is (qy + k_g_perp_y, qz + k_g_perp_z).
            // The total kx along propagation axis is sqrt(k^2 - (qy+k_gy)^2 - (qz+k_gz)^2).
            // Subtracting the carrier's axis component k_g_axis gives the envelope phase.
            //
            // For the forward beam (k_g = k_in along axis): k_g_axis=k, k_g_perp_y=k_g_perp_z=0.
            // This reduces to phase = (sqrt(k^2 - kt^2) - k) z, the original
            // carrier-subtracted exact form.
            //
            // For the Bragg beam (k_g = k_in + 2*pi*G_h): k_g_axis=k_g . axis, k_g_perp =
            // k_g . (e_y, e_z). The exact formula correctly handles the chi_h voxel
            // maps' exp(-i 2*pi G_h . r) carrier factor: it gives propagator phase
            // approx +(k - k_g_axis)*z at the chi_h spatial frequency, which equals
            // -2*pi*G_h_axis*z, exactly compensating chi_h's per-slice DC phase
            // advance and allowing coherent buildup of E_g across slices.
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

        The compiled function multiplies a complex spectrum F (row-major Nz x Ny)
        by H(ky, kz, z) using the same definition as the CUDA version:
        - propagating:  H = exp(+i*z*sqrt(k^2 - kt^2))
        - evanescent:   H = exp(-|z|*sqrt(kt^2 - k^2))  (real decay)

        Returns:
            tuple: (ffi, lib) where lib.prop_mul_cpu(...) performs the in-place
            multiplication on a provided complex array.

        Notes:
            The C signature is:
                void prop_mul_cpu(
                    const int Ny,
                    const int Nz,
                    const float* ky,
                    const float* kz,
                    const float k,
                    const float z,
                    float _Complex* F);
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
        Precompute and cache depth-dependent entrance amplitudes (Ein) per chunk.

        This streams chunks through CPU or GPU to produce Ein arrays consistent with
        the current beam grid and stage transform, then writes:
            ein_chunk_{cid}_{hash}.npz  -> array "ein" of shape (N_atoms_chunk,)
        where the cache key encodes beam, stage, and depth-window parameters.

        Args:
            sample: Object exposing:
                - chunk_total (int)
                - load_chunk_positions(cid, use_gpu=False) -> (Ni,3) Angstrom
            stage: Object with rotation (3x3) and translation (3,) arrays.
            use_gpu (bool): If True and CuPy is available, use the GPU path.
            ein_cache_dir (str or None): Directory to store NPZ files. Defaults to
                "<self.directory>/ein_cache".
            recompute_cache (bool): If True, overwrite existing cache entries.
            kernel_radius (int): Optional Gaussian blur radius (pixels) applied to
                phi and tau when building A(u,v). Set 0 to disable.
            chunk_ids (iterable[int] or None): If provided, only process these
                chunk IDs. Defaults to all chunks 1..chunk_total.

        Returns:
            tuple[str, str]: (cache_dir, cache_key_hash) used for the generated files.

        Raises:
            ValueError: If there are no chunks to precompute.

        Notes:
            - Uses pinned memory and multiple CUDA streams per GPU to overlap H2D,
            compute, D2H, and disk writes.
            - On CPU, a thread pool is used to parallelize NPZ writes.
            - The Ein definition relies on the beam grid, entrance E0, and the
            global depth window [s_min, s_max] along the beam direction.
        """
        import hashlib, threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Backend selection
        use_gpu = bool(use_gpu and (cp is not None))

        # Sanity checks
        if sample.chunk_total is None or int(sample.chunk_total) == 0:
            raise ValueError("No chunks to precompute Ein for.")

        # Determine which chunks to process
        if chunk_ids is None:
            chunk_ids = list(range(1, int(sample.chunk_total) + 1))
        else:
            chunk_ids = list(chunk_ids)

        # Depth bounds and beam maps
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Compute A(u,v) once; reuse for all chunks (GPU or CPU)
        if use_gpu:
            A_beam_np = self._compute_beam_column_A_map_gpu(sample, stage, kernel_radius=kernel_radius)
        else:
            A_beam_np = self._compute_beam_column_A_map_cpu(sample, stage, kernel_radius=kernel_radius)

        # Build cache key (captures beam, stage, grid, and depth window)
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

        # Skip chunks already cached unless recompute is requested
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

        # CPU fallback path: compute Ein with numpy and write NPZs via a saver pool
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
                    # Apply stage transform before sampling
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

        # ---------------- GPU streaming path below (unchanged except comments) ----------------

        # Host copies of static maps and vectors (copied to each device)
        tau_host = (-np.log(np.abs(A_beam_np) + np.float32(1e-20))).astype(np.float32)
        phi_host = np.angle(A_beam_np).astype(np.float32)
        E0_host  = self._beam_E0_map.astype(np.complex64)
        e1_host  = self._beam_e1.astype(np.float32)
        e2_host  = self._beam_e2.astype(np.float32)
        khat_host= (self._direction / np.linalg.norm(self._direction)).astype(np.float32)
        R_host   = np.asarray(stage.rotation, dtype=np.float32)
        T_host   = np.asarray(stage.translation, dtype=np.float32)

        # Discover devices and streaming configuration
        try:
            n_gpus = cp.cuda.runtime.getDeviceCount()
        except Exception:
            n_gpus = 1
        n_gpus = max(1, n_gpus)
        streams_per_gpu = max(1, int(os.getenv("BEAM_EIN_STREAMS_PER_GPU", "4")))
        save_threads = max(1, int(os.getenv("BEAM_EIN_SAVE_THREADS", "6")))

        # Round-robin shard the chunk list across GPUs
        shards = [[] for _ in range(n_gpus)]
        for i, cid in enumerate(to_do):
            shards[i % n_gpus].append(cid)

        # Async saver keeps pinned host memory alive until write completes
        def _save_npz_keepalive(path, arr_view, pinned_mem):
            try:
                np.savez_compressed(path, ein=np.asarray(arr_view, dtype=np.complex64))
            except Exception:
                np.savez(path, ein=np.asarray(arr_view, dtype=np.complex64))

        def gpu_worker(dev_id, my_chunks):
            if not my_chunks:
                return
            cp.cuda.Device(dev_id).use()

            # Device copies of static inputs
            tau_g = cp.asarray(tau_host)
            phi_g = cp.asarray(phi_host)
            E0_g  = cp.asarray(E0_host)
            e1g   = cp.asarray(e1_host)
            e2g   = cp.asarray(e2_host)
            khatg = cp.asarray(khat_host)
            Rg    = cp.asarray(R_host)
            Tg    = cp.asarray(T_host)

            # Build kernel once per process
            if getattr(self, "_ein_kernel", None) is None:
                self._ein_kernel = self.build_ein_sampler_kernel()

            # Allocate one stream ring per GPU
            streams = [cp.cuda.Stream(non_blocking=True) for _ in range(streams_per_gpu)]
            slot_event = [None] * streams_per_gpu
            slot_chunk = [None] * streams_per_gpu
            slot_devout= [None] * streams_per_gpu
            slot_host_mem = [None] * streams_per_gpu
            slot_host_view= [None] * streams_per_gpu

            # NPZ saver
            from concurrent.futures import ThreadPoolExecutor, as_completed
            saver = ThreadPoolExecutor(max_workers=save_threads)
            save_futs = []

            # Helper that waits for copy-back and schedules the disk write
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

            # Main loop over chunks
            for n, cid in enumerate(my_chunks):
                s_id = n % streams_per_gpu
                st = streams[s_id]

                # If this slot is still in flight, finalize it first
                if slot_event[s_id] is not None:
                    flush_slot(s_id, cache_dir)

                # Load positions and transform to stage frame on device
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

                    # Compute Ein on device
                    ein_g = self._ein_for_positions_gpu_fast(
                        pos_g=pos_g,
                        tau_g=tau_g, phi_g=phi_g, E0_g=E0_g,
                        e1g=e1g, e2g=e2g, khat_g=khatg,
                        s_min=np.float32(s_min), s_max=np.float32(s_max),
                        stream=st
                    )

                    # Async device-to-host into pinned memory
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

                # Stash slot state for later flush
                slot_event[s_id] = ev
                slot_chunk[s_id] = cid
                slot_devout[s_id]= ein_g
                slot_host_mem[s_id] = pmem
                slot_host_view[s_id]= h_view

                del pos, pos_g

            # Flush remaining slots and wait for all saves
            for s_id in range(streams_per_gpu):
                if slot_event[s_id] is not None:
                    flush_slot(s_id, cache_dir)
            for f in as_completed(save_futs):
                _ = f.result()
            saver.shutdown(wait=True)

            # Cleanup device allocations for this worker
            del tau_g, phi_g, E0_g, e1g, e2g, khatg, Rg, Tg
            for st in streams:
                st.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

        # Launch one worker per GPU
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
    # FP32 extended-precision helpers (Phase 1.4 of dynamical-method plan).
    # Mirror the kernel-side ``two_prod_fma`` (Veltkamp-Dekker exact split) but
    # in NumPy.  Used for FP32-robust atom-table coordinate conversion when
    # atomic positions are at sample-scale (~10^7 A).
    # -------------------------------------------------------------------------
    @staticmethod
    def _two_prod_fp32(a, b):
        """
        Veltkamp-Dekker exact split of FP32 product ``a * b`` into ``(p, e)``.

        ``a*b == p + e`` at FP32 precision (so adding ``e`` to ``p`` in FP32
        recovers the un-rounded product).  Vectorized over arrays.

        Args:
            a, b: Array-like, FP32-castable.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(p, e)`` both float32.
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
        Knuth's two-sum: split ``a + b`` into ``(s, e)`` such that ``a + b == s + e``
        exactly in FP32.  Vectorized.

        Args:
            a, b: Array-like, FP32-castable.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(s, e)`` both float32.
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
        Determine the minimum-Nx voxel grid for the convergent regime.

        Bounds (master plan I.6):
            Lower:  dx > NN_dist / 3            (TSC stencil overlap)
            Upper:  dx < 1 / (2 * |g_max|)      (chi_g Nyquist for general crystals)

        Window-existence condition:  |g_max| * NN_dist < 3/2.

        Args:
            g_vectors_invA: Iterable of (3,) ndarrays in 1/Angstrom (no 2pi),
                including the forward beam (g=0).
            supercell_extent_A: (Lx, Ly, Lz) tuple in Angstrom.
            NN_dist_A: Nearest-neighbor distance in Angstrom.
            dz_slice_A: Requested z-slice thickness.
            safety: Multiplicative safety factor below the upper bound for
                strict mode (default 0.9).
            strict_nyquist: True for general/defected crystals; False for
                commensurate perfect supercells (aggressive mode).
            force_unconverged: If True, do not raise when the convergent window
                has collapsed; choose dx in the inverted range and emit a
                warning.

        Returns:
            tuple: ``(Nx, Ny, Nz_slices, dx_A, dy_A, dz_A, diagnostics_dict)``.

        Raises:
            RuntimeError: If the convergent window is empty AND
                ``force_unconverged`` is False AND ``strict_nyquist`` is True.
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
            # strict_nyquist=False: aggressive mode (commensurate perfect supercells)
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
        Estimate the nearest-neighbor distance in a chunked sample.

        Strategy: pick a few "anchor" atoms; for each, extract atoms within a
        small axis-aligned bounding box centred on the anchor, then run a
        brute-force pairwise minimum on that small local set.  The bounding
        box guarantees spatial locality, so the local set always contains the
        anchor's true nearest neighbours regardless of how atoms are stored
        in the chunk.

        Why naive uniform random subsampling fails on large samples: for a
        sample of N atoms with coordination number c, the expected number of
        true NN pairs in a random subset of size n is ~ 2*n*c/N.  For N=10^8,
        n=2048, c=4 this is ~10^-4 -- the random subset almost never contains
        any true NN pair, so the reported "minimum" is dominated by accidental
        near-misses and can be 3-10x too large (we observed 8.0 A for a Ge
        sample whose true NN is 2.45 A).

        The bounding-box approach is O(N) for the mask op and O(M^2) for the
        brute-force where M is the local set size (~10^3 for typical crystals
        with a 30 A box).  Total cost: ~1-3 seconds even for 10^8-atom samples.

        Args:
            sample: chunked sample object with chunk_total chunks and
                load_chunk_positions(cid).
            n_samples: kept for backward compatibility; unused.

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

            # Box half-edge for the local-neighbourhood selection.  15 A
            # contains ~10^3 atoms in any conceivable crystal (NN spans
            # ~0.7-5 A across all materials), so the local brute force is
            # bounded by a constant cost regardless of sample size.
            R_box_A = 15.0
            # Two anchors gives belt-and-suspenders against the rare case of
            # an anchor landing on a surface atom or defect site.  For bulk
            # crystals one anchor is exact; multiple anchors guarantee the
            # answer for any sample including defected/surface ones.
            n_anchors = 2
            best_d2 = np.inf
            rng = np.random.default_rng(seed=0xBEEF)
            anchor_idx = rng.choice(N, size=min(n_anchors, N), replace=False)
            for ai in anchor_idx:
                anchor = positions[int(ai)]
                # Axis-aligned bounding-box mask (vectorized; one pass over N).
                mask = (
                    (np.abs(positions[:, 0] - anchor[0]) < R_box_A) &
                    (np.abs(positions[:, 1] - anchor[1]) < R_box_A) &
                    (np.abs(positions[:, 2] - anchor[2]) < R_box_A)
                )
                local = positions[mask]
                M = int(local.shape[0])
                if M < 2:
                    # Anchor is isolated (very sparse sample or surface atom);
                    # try the next anchor.
                    continue

                # Adaptive: if the local box is too sparse, expand it.  This
                # handles edge cases like surface atoms or unusual crystals.
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

                # Brute-force pairwise minimum on the local set.  M is small
                # (~10^3 for typical crystals), so the O(M^2) compute is fast.
                if M <= 4096:
                    d2_mat = np.sum(
                        (local[:, None, :] - local[None, :, :]) ** 2, axis=-1)
                    np.fill_diagonal(d2_mat, np.inf)
                    d2_min = float(np.min(d2_mat))
                else:
                    # Local set unexpectedly large; use scipy KD-tree if
                    # available, else block-wise pairwise.
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
            pos = pos @ R.T
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
                            Ny, Nz, coords_x_m, coords_y_m, coords_z_m,
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
            Ny (int): Detector width in pixels.
            Nz (int): Detector height in pixels.
            coords_x_m (np.ndarray): Flattened detector x coordinates (meters), length Ny*Nz.
            coords_y_m (np.ndarray): Flattened detector y coordinates (meters), length Ny*Nz.
            coords_z_m (np.ndarray): Flattened detector z coordinates (meters), length Ny*Nz.
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
            np.ndarray: Complex64 array of shape (Nz, Ny) for this chunk.
        """
        species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
        atom_count = int(species_chunk_np.shape[0])
        if atom_count == 0:
            # Early return for empty chunks
            return np.zeros((Nz, Ny), dtype=np.complex64)

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
        positions_chunk = positions_chunk @ stage.rotation.T
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
        out_r = np.zeros(Ny*Nz, dtype=np.float32)
        out_i = np.zeros(Ny*Nz, dtype=np.float32)

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
            Ny, Nz,
            coords_x_ptr, coords_y_ptr, coords_z_ptr,
            k_val,
            int(1 if apply_polarization else 0),
            float(self._pol_perp_rate),
            int(1 if apply_spherical_decay else 0),
            out_r_ptr, out_i_ptr
        )

        # Return complex field reshaped to (Nz, Ny)
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
        Compute kinematic scattering on CPU and return the detector field.

        This CPU path prepares per-chunk per-atom scattering parameters, optionally
        samples a depth-dependent entrance field Ein, and accumulates the complex
        field on the detector grid. A 1/R spherical-decay toggle is exposed.

        Args:
            sample: Object exposing per-chunk loaders:
                - load_chunk_species(cid, use_gpu=False) -> (Ni,)
                - load_chunk_positions(cid, use_gpu=False) -> (Ni,3) Angstrom
                - chunk_total (int)
            measurement_positions (np.ndarray or cupy.ndarray): Shape (3, Ny*Nz)
                detector pixel positions in Angstrom.
            measurement_shape (tuple[int, int]): (Ny, Nz) detector shape.
            stage: Object with rotation (3x3) and translation (3,) arrays.
            detector: Unused placeholder for API parity with the GPU path.
            remove_forward_component (bool): If True, subtract f0(0) inside the
                CPU kernel to avoid double counting the forward term.
            use_depth_ein (bool): If True, use cached per-atom Ein values; if any
                are missing, they are precomputed and cached.
            ein_cache_dir (str or None): Directory for Ein cache files.
            recompute_cache (bool): If True, recompute Ein even if already cached.
            apply_polarization (bool): If True, apply polarization factor inside
                the CPU CFFI kernel.
            apply_spherical_decay (bool): If True, apply relative 1/R scaling in
                the CPU kernel.

        Returns:
            np.ndarray: Complex64 array of shape (Nz, Ny) with the accumulated field.

        Notes:
            - Uses a threaded loop across chunks for CPU parallelism.
            - The forward component toggle should be consistent with transmission
            to avoid double-counting the forward-scattered term.
        """
        import hashlib
        Ny, Nz = measurement_shape

        # Load scattering databases once
        db_dict_f0_all   = self.parse_f0_db_all('f0_WaasKirf.dat')
        db_dict_f1f2_all = self.parse_f1f2_db_all('f1f2_CromerLiberman.dat')

        # Wave number (meters)
        k_val = np.float32(2.0 * np.pi / self._wavelength)

        # Ensure detector coordinates are on CPU and in meters
        if cp is not None and isinstance(measurement_positions, cp.ndarray):
            measurement_positions = measurement_positions.get()
        coords_x_m = np.ascontiguousarray(measurement_positions[0, :].astype(np.float32) / 1e10)
        coords_y_m = np.ascontiguousarray(measurement_positions[1, :].astype(np.float32) / 1e10)
        coords_z_m = np.ascontiguousarray(measurement_positions[2, :].astype(np.float32) / 1e10)

        # Empty-sample fast path
        chunk_total = int(sample.chunk_total or 0)
        if chunk_total == 0:
            return np.zeros((Nz, Ny), dtype=np.complex64)

        # Global depth window along the beam for Ein/E0 sampling
        s_min, s_max = self._compute_global_depth_bounds(sample, stage)

        # Ensure Ein cache is present if requested
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
                # Precompute Ein as needed (GPU if available, else CPU)
                self.precompute_depth_ein_all_chunks(
                    sample, stage,
                    use_gpu=(cp is not None),
                    ein_cache_dir=cache_dir,
                    recompute_cache=recompute_cache,
                    kernel_radius=0,
                    chunk_ids=missing
                )

        # Compile the CPU CFFI kernel once
        ffi_obj, complied_code = self.compile_compute_scattering_cffi()

        # Threaded per-chunk loop
        import multiprocessing
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_threads = min(chunk_total, multiprocessing.cpu_count())

        # Static beam maps and basis for E0 sampling when Ein is disabled
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
            # Skip empty chunks early
            species_chunk_np = sample.load_chunk_species(chunk_id, use_gpu=False)
            atom_count = int(species_chunk_np.shape[0])
            if atom_count == 0:
                return np.zeros((Nz, Ny), dtype=np.complex64)

            # Build per-atom scattering parameters (f0 params, f0(0), anomalous)
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

            # Stage transform to sample frame and convert to meters for the C kernel
            positions_chunk = sample.load_chunk_positions(chunk_id, use_gpu=False).astype(np.float32)
            positions_chunk = positions_chunk @ stage.rotation.T
            positions_chunk += stage.translation
            positions_chunk_m = positions_chunk / 1e10

            # Choose initial amplitudes: depth-dependent Ein or E0-only sampling
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

            # Invoke the CFFI scattering kernel for this chunk
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

        # Accumulate results across chunks
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
        Compute Ein for a set of atom positions on the GPU using the fused bilinear-sampler kernel.

        For each atom at pos_g[n], this projects onto the beam-basis coordinates (u, v), bilinearly
        samples tau, phi and E0 on the beam grid, then evaluates
            Ein = E0 * exp(-f * tau) * exp(i * f * phi)
        where f is the normalized depth fraction along khat between s_min and s_max.

        Args:
            pos_g (cupy.ndarray): Shape (N, 3), float32. Atom positions in Angstrom, on device.
            tau_g (cupy.ndarray): Shape (NyB, NzB), float32. Attenuation map on device.
            phi_g (cupy.ndarray): Shape (NyB, NzB), float32. Phase map on device.
            E0_g (cupy.ndarray): Shape (NyB, NzB), complex64. Incident field on device.
            e1g (cupy.ndarray): Shape (3,), float32. First transverse unit vector (device).
            e2g (cupy.ndarray): Shape (3,), float32. Second transverse unit vector (device).
            khat_g (cupy.ndarray): Shape (3,), float32. Unit beam direction (device).
            s_min (float): Minimum depth along khat in Angstrom.
            s_max (float): Maximum depth along khat in Angstrom.
            stream (cupy.cuda.Stream | None): Optional CUDA stream for the launch.

        Returns:
            cupy.ndarray: Shape (N,), complex64. Ein per atom on device.

        Raises:
            RuntimeError: If CuPy is not available.

        Notes:
            - Expects build_ein_sampler_kernel() to have been called previously so that the
            kernel is compiled and cached on self._ein_kernel.
            - All arrays must already live on the same device; no host-device transfers occur here.
        """
        if cp is None:
            raise RuntimeError("CuPy is required for _ein_for_positions_gpu_fast")

        # Kernel handle and output allocation
        kernel = getattr(self, "_ein_kernel", None)
        if kernel is None:
            kernel = self.build_ein_sampler_kernel()
            self._ein_kernel = kernel
        N = int(pos_g.shape[0])
        out = cp.zeros((N,), dtype=cp.complex64)

        # Compute grid/block sizes for a simple 1D launch (tuned elsewhere)
        threads = 256
        blocks = (N + threads - 1) // threads

        # Precompute reciprocals to avoid divisions in the kernel
        inv_du = cp.float32(1.0 / float(self._beam_du))
        inv_dv = cp.float32(1.0 / float(self._beam_dv))

        NyB = int(self._beam_Ny)
        NzB = int(self._beam_Nz)
        uc = cp.float32(self._beam_uc)
        vc = cp.float32(self._beam_vc)

        # Assemble kernel arguments (keep order in sync with kernel signature)
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

        # Launch
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
        GPU kinematic scattering with optional analyser.

        analyser_mode: "off" | "top_hat" | "darwin" or 0|1|2
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

        Ny, Nz = measurement_shape
        final_result = np.zeros((Nz, Ny), dtype=np.complex64)

        # Detector pixel coordinates (meters) in pinned memory
        x_coords = self.allocate_pinned_array(measurement_positions[0, :].astype(np.float32) / 1e10)
        y_coords = self.allocate_pinned_array(measurement_positions[1, :].astype(np.float32) / 1e10)
        z_coords = self.allocate_pinned_array(measurement_positions[2, :].astype(np.float32) / 1e10)
        R_pin = self.allocate_pinned_array(stage.rotation)
        T_pin = self.allocate_pinned_array(stage.translation)

        # Compute detector centre vector (meters) once on host
        ciy = int(Ny // 2)
        ciz = int(Nz // 2)
        cidx = ciz * Ny + ciy
        centre_x = float(x_coords[cidx]); centre_y = float(y_coords[cidx]); centre_z = float(z_coords[cidx])

        # Map analyser mode -> kind
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

        # Global depth window along the beam for Ein/E0 sampling
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
                self._log("verbose", f"[beam] Precomputing Ein for {len(missing)} chunk(s).")
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
            xg = cp.asarray(x_coords); yg = cp.asarray(y_coords); zg = cp.asarray(z_coords)
            # M=1 forward beam: g = (0,0,0) per beam channel.  The kernel's
            # M-channel loop reduces to a single channel with sincos returning
            # (0, 1), bit-identical to the legacy single-amplitude path.
            g_vecs_d = cp.zeros(3, dtype=cp.float32)

            # Static beam maps for E0/EIN sampling on this device
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
                    pos = pos @ Rg.T
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
                        ),
                        stream=streams[s_id]
                    )

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
        Top-level convenience wrapper for single-bounce kinematic scattering.

        Computes the scattered field on the detector using either GPU or CPU path,
        with optional depth-dependent entrance amplitudes and analyser filtering.

        Args:
            sample: Sample object providing chunked atom positions and species.
            detector: Detector object with pixel_coordinates (3, Ny*Nz) in Angstrom
                and shape (Ny, Nz) — Ny = width, Nz = height.
            stage: Stage object with rotation (3x3) and translation (3,) arrays.
            offset (np.ndarray or None, optional): If provided, subtracted from the
                final field. Defaults to None.
            use_gpu (bool, optional): If True and CuPy is available, use GPU path.
                Defaults to True.
            remove_forward (bool, optional): If True, subtract f0(0) from f0(Q) to
                remove the forward component. Defaults to False.
            use_depth_ein (bool, optional): If True, use cached per-atom Ein values.
                Defaults to False.
            ein_cache_dir (str or None, optional): Directory for Ein cache files.
                Defaults to None.
            recompute_cache (bool, optional): If True, recompute Ein cache even if
                present. Defaults to False.
            apply_polarization (bool, optional): If True, apply polarization factor.
                Defaults to False.
            spherical_decay (bool, optional): If True, apply 1/R spherical decay.
                Defaults to False.
            analyser_mode (str or int, optional): Analyser filtering mode.
                "off"/0, "top_hat"/1, or "darwin"/2. Defaults to "off".
                GPU path only; ignored with a warning on the CPU path.
            analyser_acceptance_angle_rad (float, optional): Acceptance angle for
                top_hat mode in radians. Defaults to 0.0.
            analyser_darwin_halfwidth_rad (float, optional): Half-width for darwin
                mode in radians. Defaults to 0.0.

        Returns:
            np.ndarray or cupy.ndarray: Complex field on detector of shape (Nz, Ny),
                i.e. (shape[1], shape[0]), optionally with offset subtracted.
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
            # CPU path
            if cp is None and use_gpu:
                self._log("normal", "[beam] Cupy not installed, running CPU mode.")
            # Analyser acceptance is implemented only in the GPU kernels;
            # interact_beam_cpu does not accept the analyser kwargs, so they
            # are not forwarded here. Warn if an analyser was requested.
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

        # Offset subtraction
        return (final_field - offset) if (offset is not None) else final_field
    # -------------------------------------
        
    # -------------------------------------
    # Direct transmission
    def _compute_beam_slice_integrals_cpu(self, sample, stage, slice_edges_A, kernel_radius=0):
        """
        Compute per-slice forward integrals on the beam grid using CPU (Angstrom units).

        For depth slices bounded by slice_edges_A along the beam direction,
        accumulates two slice-wise column integrals on the (NyB, NzB) beam grid:
            - delta_int[k](u,v) = C * sum_atoms_in_slice (f0(0) + f1) * W_TSC
            - beta_int[k](u,v)  = C * sum_atoms_in_slice (f2) * W_TSC

        where C = r_e * lambda^2 / (2*pi * A_pix), W_TSC is the TSC deposition kernel.

        Args:
            sample: Chunked sample object providing chunk_total,
                load_chunk_positions(cid, use_gpu=False) -> (Ni, 3) in Angstrom,
                and load_chunk_species(cid, use_gpu=False) -> (Ni,).
            stage: Rigid transform object with rotation (3x3) and translation (3,)
                arrays applied in Angstrom before deposition.
            slice_edges_A (array_like): Shape (n_slices+1,). Monotonic depth edges
                [s0, ..., sN] in Angstrom along the unit beam direction. Atom with
                depth s goes to slice k where s in [edges[k], edges[k+1]).
            kernel_radius (int, optional): Gaussian blur radius (pixels) applied
                per slice to delta_int/beta_int. Defaults to 0 (disabled).

        Returns:
            tuple: (delta_int_list, beta_int_list) where each is a list of n_slices
                float32 arrays of shape (NyB, NzB) in Angstrom units.

        Note:
            - f0(0) computed from Waasmaier-Kirfel parameters.
            - (f1, f2)(E) from Cromer-Liberman, linearly interpolated at beam energy.
            - Slicing performed along beam direction after stage transform.
            - Uses separable 1D TSC weights in u and v (3x3 stencil).
            - tau_k is clamped >= 0 after optional blurring to prevent unphysical gain.
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

        # Databases and forward factors
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

            # Beam-basis and slice index
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

                    # group by slice to avoid mixing
                    for s in np.unique(kis):
                        ms = (kis == s)
                        if not np.any(ms):
                            continue
                        np.add.at(sum_real[s].ravel(), pidx[ms], (frs[ms] * wsel[ms]).astype(np.float32))
                        np.add.at(sum_imag[s].ravel(), pidx[ms], (fis[ms] * wsel[ms]).astype(np.float32))

        # sum_real/imag -> per-slice integrals
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
        Compute per-slice forward integrals on the beam grid using GPU (Angstrom units).

        Args:
            sample: Chunked sample object (same as CPU version).
            stage: Rigid transform object with rotation and translation (Angstrom).
            slice_edges_A (array_like): Shape (n_slices+1,). Monotonic depth edges
                in Angstrom (same as CPU version).
            kernel_radius (int, optional): Gaussian blur radius in pixels. Defaults to 0.

        Returns:
            tuple: (delta_int_list, beta_int_list) where each is a list of n_slices
                float32 arrays of shape (NyB, NzB). Elements are CuPy arrays on GPU,
                or NumPy arrays if falling back to CPU.

        Note:
            - Falls back to CPU if CuPy is unavailable or no GPU is detected.
            - Atom-wise forward factors assembled on host, then transferred once.
            - Depth binning uses cp.searchsorted on edges (Angstrom).
            - TSC deposition vectorized via cupyx.scatter_add.
            - Optional per-slice Gaussian blur uses cuFFT; tau_k clamped >= 0.
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

        # decide window size for accumulator
        # Each window holds W slices; 2 flat arrays of W*bins float32
        accum_bytes_full = int(nS) * bins * 4 * 2
        try:
            free_b, _ = cp.cuda.runtime.memGetInfo()
            budget = int(0.5 * free_b)  # reserve 50 % for atom data + temporaries
        except Exception:
            budget = 2 * 1024**3  # 2 GB fallback
        if accum_bytes_full <= budget:
            window_size = nS  # fits in one shot
        else:
            window_size = max(1, budget // (bins * 4 * 2))

        # Host-side accumulator for all slices
        sum_real_host = np.zeros((nS, NyB, NzB), dtype=np.float32)
        sum_imag_host = np.zeros((nS, NyB, NzB), dtype=np.float32)

        # adaptive atom batch cap
        def _atom_batch_cap():
            try:
                free_b, _ = cp.cuda.runtime.memGetInfo()
                # Per-atom footprint: pos(12) + fr,fi(8) + projections ~80 bytes + TSC ~120 bytes
                bytes_per_atom = 220
                cap = int(0.4 * free_b / max(bytes_per_atom, 1))
                return max(32768, cap)
            except Exception:
                return 2_000_000

        # Per-chunk host-side scattering factors. The full atom-positions +
        # forward-factor cache for a multi-tens-of-billions-of-atoms sample
        # can easily exceed available host RAM, so the cache is bounded to
        # a fraction of free memory; chunks that don't fit are reloaded on
        # demand inside the slice-window loop below. Total I/O is unchanged
        # whenever the slice loop runs in a single window (the common case).
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
                # Cache budget reached; release this chunk and stop caching.
                # Remaining chunks are loaded on demand below.
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
            Worker-thread task: returns (cid, gpu_data_or_None).

            For cached chunks this just transfers the already-host data H2D.
            For streamed (uncached) chunks it loads from disk, computes
            forward factors, pins the host arrays, and copies them to GPU
            on the dedicated H2D stream. The worker thread synchronizes on
            that stream before returning so callers receive ready-to-use
            GPU arrays.
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
                # Pin streamed chunks so the H2D copy on stream_h2d is truly
                # asynchronous with the main thread's compute kernels.
                try:
                    pos_h = self.allocate_pinned_array(pos_l, dtype=np.float32)
                    fr_h  = self.allocate_pinned_array(fr_l,  dtype=np.float32)
                    fi_h  = self.allocate_pinned_array(fi_l,  dtype=np.float32)
                except Exception:
                    # Pinning can fail if pinned memory is exhausted; fall
                    # back to pageable arrays. H2D will still happen, just
                    # synchronously inside cp.asarray.
                    pos_h, fr_h, fi_h = pos_l, fr_l, fi_l
                pin_required = True
            with stream_h2d:
                pos_g = cp.asarray(pos_h)
                fr_g  = cp.asarray(fr_h)
                fi_g  = cp.asarray(fi_h)
            # Synchronize on the worker thread, NOT main, so the main thread
            # can keep running compute kernels on its stream while we wait
            # for this chunk's transfers to land on the device.
            stream_h2d.synchronize()
            # Drop pinned host buffers now that data is on GPU; for cached
            # chunks the canonical reference lives in chunk_cache.
            if pin_required:
                del pos_h, fr_h, fi_h
            return cid, (pos_g, fr_g, fi_g)

        try:
            # Process slices in windows
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
                        # Late submission (in case prefetch_window > total)
                        fut = prefetch_pool.submit(_prefetch_chunk, cid)
                    _, gpu_data = fut.result()

                    # Submit the next prefetch immediately so the pool is
                    # always working on chunks ahead of compute.
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

                        # Filter to atoms within beam grid AND current slice
                        # window. Empty-mask scatter_adds are cheap GPU
                        # no-ops, so we deliberately avoid bool(inb.any())
                        # here -- that would force a host sync per batch.
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

        # Convert sums to per-slice integrals (keep on host as numpy)
        delta_int = [np.float32(C) * sum_real_host[s] for s in range(nS)]
        beta_int  = [np.float32(C) * sum_imag_host[s] for s in range(nS)]

        # Optional blur per slice (transiently on GPU, one at a time)
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
        Choose the number of projection slices so each slice stays in the linear regime.

        Ensures every thin-slice update A_k(u,v) = exp(-tau_k + i*phi_k) satisfies
        max(|phi_k(u,v)|, tau_k(u,v)) <= target_step across all slices and pixels.

        Args:
            sample: Chunked sample object. Stage transform is applied (Angstrom).
            stage: Rigid transform object with rotation and translation arrays.
            kernel_radius (int, optional): Gaussian blur radius (pixels) forwarded
                to the per-slice integrals. Defaults to 0.
            target_step (float, optional): Maximum allowed per-slice change
                (radians or unitless attenuation). Defaults to 0.1.
            use_gpu (bool, optional): If True and CuPy is available, use GPU path
                for A(u,v) and per-slice integrals. Defaults to True.
            max_slices (int, optional): Hard cap on slice count to avoid runaway
                refinement. Defaults to 2048.
            n_init (int or None, optional): If provided, start refinement from this
                value instead of the computed lower bound n0.
            absorption_multiplier (float, optional): Multiplicative factor for
                absorption coefficient. Applied when checking tau bounds.
                Defaults to 1.0.

        Returns:
            tuple: (n_final, edges_A, delta_list, beta_list, info) where:
                - n_final (int): Chosen number of slices (>= 1).
                - edges_A (np.ndarray): Shape (n_final+1,), depth edges in Angstrom.
                - delta_list (list): Per-slice delta integrals (CuPy or NumPy arrays).
                - beta_list (list): Per-slice beta integrals (CuPy or NumPy arrays).
                - info (dict): Diagnostics with keys 'phi_max', 'tau_max', 'n0'.

        Note:
            - Thickness is measured along the unit beam direction after stage transform.
            - If thickness <= 0, returns a single empty slice with zeros.
            - Refinement uses equal-depth bins (robust and monotone in n).
            - Does not modify global state; caller decides how to use returned data.
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

        # Compute slice integrals ONCE at the finest resolution
        edges_fine = np.linspace(s_min_A, s_max_A, n_fine + 1, dtype=np.float32)
        if use_gpu:
            delta_fine, beta_fine = self._compute_beam_slice_integrals_gpu(
                sample, stage, edges_fine, kernel_radius)
        else:
            delta_fine, beta_fine = self._compute_beam_slice_integrals_cpu(
                sample, stage, edges_fine, kernel_radius)

        # Helper: check max per-slice phase/attenuation for a given n
        # by merging groups of (n_fine // n) consecutive fine slices.
        # Slice integrals are numpy arrays (host-side) so use numpy ops.
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
                # Already at finest resolution computed; return it
                edges_out = np.linspace(s_min_A, s_max_A, n_fine + 1, dtype=np.float32)
                info = {"phi_max": float(phi_max), "tau_max": float(tau_max), "n0": int(n0)}
                return int(max(1, n_fine)), edges_out, delta_fine, beta_fine, info

            n = min(n * 2, int(max_slices))

    def _compute_beam_column_A_map_cpu(self, sample, stage, kernel_radius=0):
        """
        Build the transmission column map A(u,v) = exp(-tau + i*phi) on the beam grid using CPU.

        For each atom in each chunk, the algorithm projects its position into beam-basis
        grid coordinates (u,v), accumulates real (tau) and imaginary (phi) parts on nearby
        pixels using a compact kernel (nearest grid point or higher-order, as implemented),
        and finally exponentiates to form A.

        Args:
            sample: Chunked sample object; must provide:
                - chunk_total (int)
                - load_chunk_positions(cid, use_gpu=False) -> (Ni, 3) Angstrom
                - load_chunk_species(cid, use_gpu=False) -> (Ni,)
            stage: Object providing rotation (3x3) and translation (3,) for sample-to-beam frame.
            kernel_radius (int): Optional Gaussian blur radius (pixels) applied to tau and phi
                after accumulation; set 0 to disable.

        Returns:
            numpy.ndarray: Complex64 array of shape (NyB, NzB) with A(u,v) on the beam grid.

        Notes:
            - Only atoms whose projected indices fall inside the beam grid contribute.
            - tau is lower-bounded at 0 after blur to avoid unphysical gain.
            - Units: positions and spacings are in Angstrom.
        """
        # Constants (angstrom)
        r_e_A = 2.81794092e-5
        lam_A = self._wavelength * 1e10

        # Beam-grid geometry
        du, dv = float(self._beam_du), float(self._beam_dv)
        NyB, NzB = int(self._beam_Ny), int(self._beam_Nz)
        A_pix_A2 = du * dv

        # Accumulate column sums of forward factors (real and imag parts)
        sum_real = np.zeros((NyB, NzB), np.float32)  # sum of f0(0)+f1
        sum_imag = np.zeros((NyB, NzB), np.float32)  # sum of f2

        # Databases
        f1f2_dict      = self.parse_f1f2_db_all("f1f2_CromerLiberman.dat")
        f0_params_dict = self.parse_f0_db_all('f0_WaasKirf.dat')
        f0_zero_dict   = self._build_f0_zero_dict(f0_params_dict)

        e1 = self._beam_e1
        e2 = self._beam_e2

        def _tsc_w(d):
            # 1D TSC weights for distances in pixel units (centered on integer indices)
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

            # Stage transform in real space (angstrom)
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
                idxR = np.concatenate(idx_list_R); wR = np.concatenate(w_list_R)
                idxI = np.concatenate(idx_list_I); wI = np.concatenate(w_list_I)
                np.add.at(sum_real.ravel(), idxR, wR)
                np.add.at(sum_imag.ravel(), idxI, wI)

        # Convert forward sums -> total phase/attenuation (column integrals)
        # phi = -k*delta_int, tau = k*beta_int, with:
        # delta_int = (r_e * lambda^2 / (2*pi) / A_pix) * sum_real
        # beta_int  = (r_e * lambda^2 / (2*pi) / A_pix) * sum_imag
        two_pi = 2.0 * np.pi
        C = (r_e_A * (lam_A * lam_A)) / (two_pi * A_pix_A2)  # dimensionless
        delta_int = C * sum_real.astype(np.float32)
        beta_int  = C * sum_imag.astype(np.float32)

        kA = two_pi / lam_A
        phi = (-kA * delta_int).astype(np.float32)
        tau = ( kA * beta_int ).astype(np.float32)

        # Numerical safety: never allow gain
        tau = np.maximum(tau, np.float32(0.0))

        # Optional blur (same as before)
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
                            padding_mode="edge", pad_constant=0.0,
                            n_slices=None, target_phase_step=0.1,
                            pad_factor=2, absorption_multiplier=1.0):
        """
        Compute projection-only multislice transmission.

        Updates the beam field on the beam grid using a pure projection model:
        E <- E * A_k(u,v) for k = 0..n_slices-1, where A_k = exp(-tau_k + i*phi_k).
        No angular-spectrum propagation between slices. After exit surface, E_exit(u,v)
        is bilinearly resampled to detector pixels, with an optional free-space hop
        if the detector plane is offset from the exit plane.

        Args:
            sample: Chunked atoms with species; used by the slice-integral helpers.
            detector: Object providing shape=(Ny, Nz) and pixel_coordinates (3, Ny*Nz)
                in Angstrom — Ny = width, Nz = height.
            stage: Rigid transform with rotation (3x3) and translation (3,) in Angstrom.
            use_gpu (bool, optional): Use GPU for slice integrals and propagation
                when CuPy is available. Defaults to True.
            kernel_radius (int, optional): Gaussian blur radius (pixels) applied to
                per-slice maps. Defaults to 0.
            padding_mode (str, optional): Padding policy for exit-to-detector propagation.
                One of "edge" or "constant". Defaults to "edge".
            pad_constant (float, optional): Constant pad value when padding_mode="constant".
                Defaults to 0.0.
            n_slices (int or None, optional): Number of slices. If None, auto-selected
                via _auto_slice_count_linear_regime with target_phase_step.
            target_phase_step (float, optional): Per-slice linear-regime target
                (radians / unitless) for auto-slicer. Defaults to 0.1.
            pad_factor (float, optional): Minimum multiplicative padding for
                angular-spectrum FFT sizes (>= 1). Defaults to 2.
            absorption_multiplier (float, optional): Multiplicative factor applied to
                the absorption coefficient (beta / imaginary part of refractive index).
                1.0 = physical absorption, 0.0 = no absorption, >1.0 = enhanced.
                Defaults to 1.0.

        Returns:
            np.ndarray: Complex64 array of shape (Nz, Ny) with the exit field sampled
                on the detector (after optional free-space hop).

        Note:
            - Zero/degenerate thickness: falls back to single-slice A(u,v).
            - Detector pixels projected to (u,v) via beam basis; out-of-bounds get 0.
            - Detector offset: propagates by signed offset distance using
              detector.pixel_size if available, else estimates from (u,v).
            - All geometry in Angstrom internally; propagation converts to meters.
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

        # -------- New: robust auto-slice selection for linear per-slice increments --------
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
            # Zero thickness — no interaction, pass beam through unchanged
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
                # Stack the per-slice delta / beta lists once and copy to
                # GPU in a single H2D transfer. The previous slice-by-slice
                # path issued 2 * n_final separate H2D copies plus a forced
                # `free_all_blocks` per slice, both of which serialize the
                # default stream.
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

            # in-bounds mask
            mask = (iu >= 0.0) & (iu <= (NyB - 1)) & (iv >= 0.0) & (iv <= (NzB - 1))

            # neighbors and weights
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

        # Prefer detector.pixel_size if available; fallback to estimating from geometry
        def _estimate_dy_dz_from_uv(u_flat, v_flat, nz, ny):
            u_img = u_flat.reshape(nz, ny)
            v_img = v_flat.reshape(nz, ny)
            # dy corresponds to change of u across cols; dz corresponds to change of v across rows
            du_cols = np.abs(u_img[:, 1:] - u_img[:, :-1]).ravel()
            dv_rows = np.abs(v_img[1:, :] - v_img[:-1, :]).ravel()
            dy_A_est = float(np.median(du_cols)) if du_cols.size else 0.0
            dz_A_est = float(np.median(dv_rows)) if dv_rows.size else 0.0
            # Safety fallback
            if not np.isfinite(dy_A_est) or dy_A_est <= 0.0:
                dy_A_est = du_A
            if not np.isfinite(dz_A_est) or dz_A_est <= 0.0:
                dz_A_est = dv_A
            return dy_A_est * 1e-10, dz_A_est * 1e-10

        # Determine dy, dz (meters) for the detector grid
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
        Combine scattering and transmission contributions and write the result to the detector.

        Args:
            sample: Sample object with chunk accessors required by the backends.
            detector: Detector object exposing:
                - shape -> (Ny, Nz)
                - pixel_coordinates
                - input_pixel_values(array)
            stage: Stage object with:
                - rotation (3x3)
                - translation (3,)
            scattering (bool): If True, include the kinematic scattering term.
            scattering_params (list | tuple | dict | None): Configuration for the
                scattering call. The following forms are accepted.

                List/tuple form (backward compatible):
                    [offset=None,
                    use_depth_ein=False,
                    ein_cache_dir=None,
                    recompute_cache=False,
                    apply_polarization=False,
                    spherical_decay=False,
                    remove_forward=None,
                    use_gpu=None]

                    Notes:
                    - remove_forward=None means: default to transmission being True.
                    - use_gpu (if provided) overrides the function-level use_gpu for the
                    scattering call.

                Dict form (preferred):
                    {
                    "offset": np.ndarray | None,
                    "use_depth_ein": bool,
                    "ein_cache_dir": str | None,
                    "recompute_cache": bool,
                    "apply_polarization": bool,
                    "spherical_decay": bool,
                    "remove_forward": bool | None,
                    "use_gpu": bool | None
                    }

            transmission (bool): If True, include the transmission term.
            transmission_params (list | tuple | dict | None): Configuration for the
                transmission call. Accepted forms:

                List/tuple form (backward compatible):
                    [kernel_radius=0.0,
                    padding_mode="edge",
                    pad_constant=0.0,
                    use_gpu=None]

                Dict form:
                    {
                    "kernel_radius": float,
                    "padding_mode": str,      # "edge" or "constant"
                    "pad_constant": float,
                    "use_gpu": bool | None
                    }

            use_gpu (bool): Global GPU preference. Each sub-call may override this via
                its own "...params" value. If CuPy is unavailable, CPU paths are used.

        Returns:
            None: The combined complex field is written to
            detector.input_pixel_values(field).

        Notes:
            - If both scattering and transmission are enabled and remove_forward is
            not explicitly provided, the forward f0(0) term is removed from the
            scattering call to avoid double counting.
            - All GPU toggles honor CuPy availability; when CuPy is not present,
            CPU implementations are used regardless of requested GPU usage.
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
        Band-limited angular spectrum propagation on GPU with symmetric padding.

        The total distance z is split into ceil(abs(z)/step_max) sub-steps to bound
        phase error and wrap-around. For each step:
        1) Pad the input to sizes chosen by _choose_optimal_pad (also enforces a
            minimum multiplicative pad_factor and rounds to power-of-two for FFTs).
        2) FFT -> multiply by the free-space transfer function using the provided
            CUDA kernel (built by build_propagation_multiplier_kernel) -> IFFT.
        3) Center-crop back to (Nz, Ny).

        Args:
            field (array-like): Complex field, shape (Nz, Ny). NumPy or CuPy.
            dy (float): Pixel size along Y (width) in meters.
            dz (float): Pixel size along Z (height) in meters.
            z (float): Propagation distance in meters (can be negative).
            kernel (cupy.RawKernel): "prop_mul_kernel" compiled by
                build_propagation_multiplier_kernel.
            step_max (float): Maximum per-step distance in meters.
            pad_factor (float): Minimum multiplicative padding factor (>=1.0).
            padding_mode (str): "edge" to replicate edges or "constant" to pad with
                pad_constant.
            pad_constant (float): Value used when padding_mode == "constant".

        Returns:
            cupy.ndarray: Complex64 field after propagation, cropped to (Nz, Ny).

        Raises:
            RuntimeError: If CuPy is not available.
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

        # Input sizes
        F0 = cp.asarray(field, dtype=cp.complex64)
        Nz, Ny = int(F0.shape[0]), int(F0.shape[1])

        # Choose symmetric padding from sampling and distance (also apply pad_factor)
        Ny2, Nz2 = self._choose_optimal_pad(
            Ny, Nz, float(dy), float(dz), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        z0 = (Nz2 - Nz) // 2
        y0 = (Ny2 - Ny) // 2

        # Configurable padding
        pmode = (padding_mode or "edge").lower()
        if pmode == "constant":
            Fp = cp.full((Nz2, Ny2), complex(pad_constant), dtype=cp.complex64)
            Fp[z0:z0+Nz, y0:y0+Ny] = F0
        else:
            # Default to "edge" replication
            pad_spec = ((z0, Nz2 - Nz - z0), (y0, Ny2 - Ny - y0))
            Fp = cp.pad(F0, pad_spec, mode='edge')

        # k-grids (rad/m), no shifts (fft2 uses non-shifted ordering)
        k  = 2.0 * np.pi / float(self._wavelength)
        ky = (2.0 * np.pi) * cp.fft.fftfreq(Ny2, d=float(dy)).astype(cp.float32)
        kz = (2.0 * np.pi) * cp.fft.fftfreq(Nz2, d=float(dz)).astype(cp.float32)

        # Backward compatibility: if k_g_axis not provided, derive from cos_theta.
        # For a beam tilted at angle theta from the propagation axis with NO
        # transverse carrier shift, k_g_axis = k*cos_theta and k_g_perp = 0.
        # This recovers the old behavior for callers that haven't been updated
        # to pass full k_g vectors.
        if k_g_axis is None:
            k_g_axis_v = float(k) * float(cos_theta)
        else:
            k_g_axis_v = float(k_g_axis)

        # Forward FFT
        Fp = cp.fft.fft2(Fp)

        # Multiply by propagator in place via CUDA kernel
        block = (16, 16)
        grid  = ((Ny2 + block[0] - 1)//block[0],
                (Nz2 + block[1] - 1)//block[1])
        kernel(grid, block,
            (ky, kz, cp.float32(k), cp.float32(z),
                np.int32(Ny2), np.int32(Nz2), Fp,
                cp.float32(k_g_axis_v),
                cp.float32(k_g_perp_y),
                cp.float32(k_g_perp_z)))

        # Inverse FFT and center crop back to original size
        out = cp.fft.ifft2(Fp)
        return out[z0:z0+Nz, y0:y0+Ny]
    
    def _angular_spectrum_propagate_cpu(
            self, field, dy, dz, z, lib, ffi,
            step_max=0.02, pad_factor=1.0,
            padding_mode: str = "edge",
            pad_constant: float = 0.0
        ):
        """
        Band-limited angular spectrum propagation on CPU with symmetric padding.

        Splits distance z into sub-steps if abs(z) > step_max to improve numerical
        stability, pads out to sizes chosen by _choose_optimal_pad, multiplies the
        spectrum by the free-space transfer function using lib.prop_mul_cpu, and
        crops back to the original size.

        Args:
            field (array-like): Complex field, shape (Nz, Ny). NumPy preferred.
            dy (float): Pixel size along Y (width) in meters.
            dz (float): Pixel size along Z (height) in meters.
            z (float): Propagation distance in meters (can be negative).
            lib: CFFI-verified library with function prop_mul_cpu(...).
            ffi: CFFI FFI object.
            step_max (float): Maximum per-step distance in meters.
            pad_factor (float): Minimum multiplicative padding factor (>=1.0).
            padding_mode (str): "edge" to replicate edges or "constant" to pad
                with pad_constant.
            pad_constant (float): Value used when padding_mode == "constant".

        Returns:
            np.ndarray: Complex64 field after propagation, cropped to (Nz, Ny).
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

        # Input (Nz, Ny)
        F0 = np.asarray(field, dtype=np.complex64, order='C')
        Nz, Ny = int(F0.shape[0]), int(F0.shape[1])

        # Choose symmetric padding and centers
        Ny2, Nz2 = self._choose_optimal_pad(
            Ny, Nz, float(dy), float(dz), float(self._wavelength), float(z),
            safety=1.1, enforce_pow2=True, min_pad_factor=max(1.0, float(pad_factor))
        )
        z0 = (Nz2 - Nz) // 2
        y0 = (Ny2 - Ny) // 2

        # Configurable padding
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

        # Forward FFT
        Fp = np.fft.fft2(Fp)

        # Multiply by propagator (CPU implementation)
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
            detector: Provides:
                - pixel_size: tuple (dy, dz) in Angstrom.
                - shape: tuple (Ny, Nz).
                - pixel_values: complex64 array of shape (Nz, Ny).
                - input_pixel_values(array): method to write the updated field.
            optics: An optics object exposing:
                - components (list of dicts)
                - apply_stack(field, dy, dz, wavelength, propagate_free_space, use_gpu)
            use_gpu, step_max, pad_factor, padding_mode, pad_constant: Propagation controls.
            save_field (bool): If True, writes back to detector; else returns the field.

        Returns:
            None or np.ndarray: Writes to detector by default; returns the field if save_field is False.
        """
        # Convert detector pixel sizes from Angstrom to meters.
        pixel_size = np.asarray(detector.pixel_size, dtype=np.float64)
        dy, dz = pixel_size * 1e-10
        E = detector.pixel_values  # complex64 (Nz, Ny)

        # Check that pixel_values has been populated
        if E is None:
            raise ValueError(
                "detector.pixel_values is None. Run atomic_direct_interaction() first "
                "to populate the detector field before calling wavefield_propagation()."
            )

        # Check that beam wavelength is set
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

        # Single call into optics now handles the whole stack.
        # Optics uses (dx, dy) for (column_spacing, row_spacing) internally.
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