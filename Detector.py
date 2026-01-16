# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
try:
    import cupy as cp
except ImportError:
    cp = None
import json
import os
from Logging import logging

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class detector(logging):

    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "create_detector",
        "read_detector_metadata",
        "write_detector_metadata",
        "write_Efield_values",
        "read_Efield_values",
        "position_detector_relative",
        "position_detector_absolute",
        "plot_detector",
        "plot_detector_angles",
        "plot_detector_position",
        "get_detector_axis",
        "coordinate_conversion",
    )
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=os.getcwd()):
        """
        Initialize a detector object.

        Args:
          directory (str, optional): Folder used for I/O operations. Defaults
            to the current working directory. The directory is created if it
            does not exist.
        """
        super().__init__(log_name="detector")
        self.directory = directory
        self._shape = None
        self._pixel_size = None
        self._center = None
        self._direction = None
        self._two_theta = None
        self._eta = None
        self._distance = None
        self._pixel_coordinates = None
        self._pixel_values = None
        self._pixel_phase = None
        self._pixel_amplitude = None
        self._pixel_intensity = None
        self._geometry = None
        self._construction_mode = None
        self._input_mode = None
        self._angular_range = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        
    def create_detector(self, shape, pixel_size, geometry='rectangular',
                        construction_mode='plane', input_mode='spatial'):
        """
        Create the pixel layout in plane or shell construction mode.

        Creates a detector pixel grid in either rectangular or ring layout, on
        either a flat plane or spherical shell.

        Construction Modes:
            - "plane": Pixels on flat detector plane (standard CCD/area detector).
              The initial plane lies on x = 0, with detector normal along +x.
            - "shell": Pixels on spherical shell centered at origin (equal solid
              angle coverage). Built at unit radius and scaled during positioning.

        Geometry Types:
            - "rectangular": Regular grid of pixels
            - "ring": Concentric rings (plane) or latitude circles (shell)

        Input Modes:
            - "spatial" (default): shape is pixel counts, pixel_size is spacing
            - "angular": shape is angular range (degrees), pixel_size is resolution (degrees)

        Args:
          shape: Depends on geometry and input_mode:
            - rectangular + spatial: (Ny, Nz) pixel counts
            - rectangular + angular: (theta_y_min, theta_y_max, theta_z_min, theta_z_max) in degrees
            - ring + spatial: (N_two_theta, N_eta) bin counts
            - ring + angular: (two_theta_inner, two_theta_outer) in degrees
          pixel_size (tuple[float, float]): Depends on geometry, construction_mode, and input_mode:
            - rectangular + plane + spatial: (dy, dz) in Angstroms
            - rectangular + shell + spatial: (d_theta_y, d_theta_z) in degrees
            - rectangular + angular: (d_theta_y, d_theta_z) in degrees
            - ring + plane + spatial: (d_two_theta, d_eta) in Angstroms
            - ring + shell + spatial: (d_two_theta, d_eta) in degrees
            - ring + angular: (d_two_theta, d_eta) in degrees
          geometry (str): "rectangular" (default) or "ring".
          construction_mode (str): "plane" (default) or "shell".
          input_mode (str): "spatial" (default) or "angular".
            - spatial: shape is pixel counts, pixel_size is spacing
            - angular: shape is angular range, pixel_size is angular resolution
              Actual pixel counts computed and stored in self.shape.

        Raises:
          ValueError: If an unsupported geometry, construction_mode, or input_mode is provided.

        Notes:
          - Pixel coordinates are stored in a stacked array with shape (3, Npixels)
            in the order (x, y, z), raveled in C order.
          - For angular mode, actual pixel counts are computed from range/resolution.
          - For ring + angular, d_eta is adjusted to ensure 360° wraps exactly.
          - For angular mode, the original angular range is stored in self.angular_range.
          - For shell mode: pixels distributed on unit sphere, scaled to distance
            during positioning via position_detector_absolute().
        """
        self._shape = shape
        self._pixel_size = pixel_size
        self._center = np.array([0.0, 0.0, 0.0])
        self._direction = np.array([1.0, 0.0, 0.0])
        self._two_theta = 0
        self._eta = 0
        self._distance = 0
        self._geometry = geometry.lower()

        # Validate and store construction mode
        construction_mode_lower = construction_mode.lower()
        if construction_mode_lower not in ('plane', 'shell'):
            raise ValueError(f"construction_mode must be 'plane' or 'shell', got '{construction_mode}'")
        self._construction_mode = construction_mode_lower

        # Validate and store input mode
        input_mode_lower = input_mode.lower()
        if input_mode_lower not in ('spatial', 'angular'):
            raise ValueError(f"input_mode must be 'spatial' or 'angular', got '{input_mode}'")
        self._input_mode = input_mode_lower
        self._angular_range = None  # Reset, will be set if angular mode

        # Build the initial pixel array in the y-z plane (x = 0) ----------------
        if geometry.lower() == 'rectangular':
            if self._input_mode == 'angular':
                # Angular mode: shape is (theta_y_min, theta_y_max, theta_z_min, theta_z_max)
                if len(shape) != 4:
                    raise ValueError("For angular mode rectangular, shape must be "
                                   "(theta_y_min, theta_y_max, theta_z_min, theta_z_max)")

                theta_y_min, theta_y_max = float(shape[0]), float(shape[1])
                theta_z_min, theta_z_max = float(shape[2]), float(shape[3])
                d_theta_y, d_theta_z = float(pixel_size[0]), float(pixel_size[1])

                # Validate
                if theta_y_min >= theta_y_max or theta_z_min >= theta_z_max:
                    raise ValueError("Angular range min must be less than max")

                # Calculate pixel counts
                Ny = int(np.ceil((theta_y_max - theta_y_min) / d_theta_y))
                Nz = int(np.ceil((theta_z_max - theta_z_min) / d_theta_z))

                # Store actual shape and recalculate actual resolution
                self._shape = (Ny, Nz)
                d_theta_y_actual = (theta_y_max - theta_y_min) / Ny
                d_theta_z_actual = (theta_z_max - theta_z_min) / Nz
                self._pixel_size = np.array([d_theta_y_actual, d_theta_z_actual], dtype=np.float32)

                # Store angular range for reference
                self._angular_range = (theta_y_min, theta_y_max, theta_z_min, theta_z_max)

                # Convert to radians and create grid
                theta_y_min_rad = np.deg2rad(theta_y_min)
                theta_y_max_rad = np.deg2rad(theta_y_max)
                theta_z_min_rad = np.deg2rad(theta_z_min)
                theta_z_max_rad = np.deg2rad(theta_z_max)

                # Create angular arrays (bin centers)
                theta_y_lin = np.linspace(theta_y_min_rad, theta_y_max_rad, Ny,
                                         endpoint=False, dtype=np.float32)
                theta_y_lin += (theta_y_max_rad - theta_y_min_rad) / (2 * Ny)
                theta_z_lin = np.linspace(theta_z_min_rad, theta_z_max_rad, Nz,
                                         endpoint=False, dtype=np.float32)
                theta_z_lin += (theta_z_max_rad - theta_z_min_rad) / (2 * Nz)

                # Build at unit distance using gnomonic projection
                THETA_Y, THETA_Z = np.meshgrid(theta_y_lin, theta_z_lin)
                rho = np.sqrt(THETA_Y**2 + THETA_Z**2)
                c = np.arctan(rho)
                mask = rho > 1e-10
                X = np.cos(c)
                Y = np.where(mask, np.sin(c) * (THETA_Y / rho), 0.0).astype(np.float32)
                Z = np.where(mask, np.sin(c) * (THETA_Z / rho), 0.0).astype(np.float32)
                self._pixel_coordinates = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))

            else:  # spatial mode (existing behavior)
                # Create 1D axes centered on 0 for y and z. The endpoints are chosen
                # so that there are exactly shape[0] and shape[1] samples.
                y_lin = np.linspace(-(self.shape[0]/2)*self.pixel_size[0],
                                    +(self.shape[0]/2)*self.pixel_size[0],
                                    self.shape[0], dtype=np.float32)
                z_lin = np.linspace(-(self.shape[1]/2)*self.pixel_size[1],
                                    +(self.shape[1]/2)*self.pixel_size[1],
                                    self.shape[1], dtype=np.float32)
                # Mesh in y-z; X is identically 0 in this initial plane.
                Y, Z = np.meshgrid(y_lin, z_lin)
                X = np.full_like(Y, 0)
                self._pixel_coordinates = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
        elif geometry.lower() == 'ring':
            if self._input_mode == 'angular':
                # Angular mode: shape is (two_theta_inner, two_theta_outer)
                if len(shape) != 2:
                    raise ValueError("For angular mode ring, shape must be "
                                   "(two_theta_inner, two_theta_outer)")

                two_theta_inner_deg = float(shape[0])
                two_theta_outer_deg = float(shape[1])
                d_two_theta_deg = float(pixel_size[0])
                d_eta_deg = float(pixel_size[1])

                # Validate
                if two_theta_inner_deg >= two_theta_outer_deg:
                    raise ValueError("two_theta_inner must be less than two_theta_outer")
                if two_theta_inner_deg < 0:
                    raise ValueError("two_theta_inner must be >= 0")
                max_two_theta = 90.0 if self._construction_mode == 'plane' else 180.0
                if two_theta_outer_deg > max_two_theta:
                    raise ValueError(f"two_theta_outer must be <= {max_two_theta}° for "
                                   f"{self._construction_mode} mode")

                # Calculate pixel counts
                N_two_theta = int(np.ceil((two_theta_outer_deg - two_theta_inner_deg) / d_two_theta_deg))
                N_eta = int(np.floor(360.0 / d_eta_deg))
                if N_eta < 1:
                    N_eta = 1

                # Store actual shape and resolution
                self._shape = (N_two_theta, N_eta)
                d_two_theta_actual = (two_theta_outer_deg - two_theta_inner_deg) / N_two_theta
                d_eta_actual = 360.0 / N_eta
                self._pixel_size = np.array([d_two_theta_actual, d_eta_actual], dtype=np.float32)

                # Store angular range
                self._angular_range = (two_theta_inner_deg, two_theta_outer_deg)

                # Convert to radians
                two_theta_inner = np.deg2rad(two_theta_inner_deg)
                two_theta_outer = np.deg2rad(two_theta_outer_deg)

                # Create angular arrays (bin centers)
                d_two_theta_rad = (two_theta_outer - two_theta_inner) / N_two_theta
                two_theta_lin = two_theta_inner + (np.arange(N_two_theta, dtype=np.float32) + 0.5) * d_two_theta_rad
                eta_lin = 2.0 * np.pi * (np.arange(N_eta, dtype=np.float32) + 0.5) / N_eta

                ETA, TWO_THETA = np.meshgrid(eta_lin, two_theta_lin, indexing='xy')

                if self._construction_mode == 'plane':
                    # Build at unit distance (r = tan(2theta))
                    R = np.tan(TWO_THETA)
                    Y = R * np.cos(ETA)
                    Z = R * np.sin(ETA)
                    X = np.zeros_like(Y, dtype=np.float32)
                else:  # shell
                    # Direct spherical coordinates on unit sphere
                    X = np.cos(TWO_THETA).astype(np.float32)
                    Y = (np.sin(TWO_THETA) * np.sin(ETA)).astype(np.float32)
                    Z = (np.sin(TWO_THETA) * np.cos(ETA)).astype(np.float32)

                self._pixel_coordinates = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))

            else:  # spatial mode (existing behavior)
                # Compute inner radius so that its circumference equals N_eta * d_eta.
                r_in = (self.shape[1] * self.pixel_size[1]) / (2 * np.pi)
                # Outer radius adds N_two_theta radial bins of size d_two_theta.
                r_out = r_in + self.shape[0] * self.pixel_size[0]
                # Note: r_out is currently not used directly; radii are sampled with 0.5-bin offset.
                i_lin = np.arange(self.shape[0], dtype=np.float32)
                j_lin = np.arange(self.shape[1], dtype=np.float32)
                r_lin = r_in + (i_lin + 0.5) * self.pixel_size[0]
                phi_lin = 2.0 * np.pi * (j_lin + 0.5) / self.shape[1]
                PHI, R = np.meshgrid(phi_lin, r_lin, indexing='xy')
                Y = R * np.cos(PHI)
                Z = R * np.sin(PHI)
                X = np.zeros_like(Y)
                self._pixel_coordinates = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
        else:
            raise ValueError(f"Unknown geometry '{geometry}'. Choose 'rectangular' or 'ring'.")

        # Apply shell projection if construction_mode is 'shell' ----------------
        # Note: For angular mode, shell projection is already applied during geometry creation
        if self._construction_mode == 'shell' and self._input_mode == 'spatial':
            # Get the current plane coordinates
            X_plane, Y_plane, Z_plane = self._pixel_coordinates

            if geometry.lower() == 'rectangular':
                # For rectangular geometry on shell, interpret pixel_size as angular spacing (degrees)
                # Convert to radians
                theta_y = np.deg2rad(self.pixel_size[0])
                theta_z = np.deg2rad(self.pixel_size[1])

                # Create angular grid centered on 0 (in radians)
                theta_y_lin = np.linspace(-(self.shape[0]/2) * theta_y,
                                          +(self.shape[0]/2) * theta_y,
                                          self.shape[0], dtype=np.float32)
                theta_z_lin = np.linspace(-(self.shape[1]/2) * theta_z,
                                          +(self.shape[1]/2) * theta_z,
                                          self.shape[1], dtype=np.float32)

                # Create meshgrid of angular coordinates
                THETA_Y, THETA_Z = np.meshgrid(theta_y_lin, theta_z_lin)

                # Use gnomonic (tangent plane) projection to map to unit sphere
                # This creates a rectangular patch on the sphere centered at (1,0,0)
                rho = np.sqrt(THETA_Y**2 + THETA_Z**2)
                c = np.arctan(rho)

                # Handle the center point (rho=0) to avoid division by zero
                mask = rho > 1e-10
                X_shell = np.cos(c)
                Y_shell = np.zeros_like(THETA_Y)
                Z_shell = np.zeros_like(THETA_Z)

                # For non-zero rho, compute spherical projection
                Y_shell = np.where(mask, np.sin(c) * (THETA_Y / rho), 0.0)
                Z_shell = np.where(mask, np.sin(c) * (THETA_Z / rho), 0.0)

                # Update pixel coordinates with shell projection
                self._pixel_coordinates = np.vstack((X_shell.ravel(), Y_shell.ravel(), Z_shell.ravel()))

            elif geometry.lower() == 'ring':
                # For ring geometry on shell, interpret pixel_size as angular spacing (degrees)
                # Convert to radians
                d_two_theta_rad = np.deg2rad(self.pixel_size[0])
                d_eta_rad = np.deg2rad(self.pixel_size[1])

                # Create angular arrays
                # two_theta: starts from d_two_theta_rad/2 (0.5-bin offset)
                two_theta_lin = (np.arange(self.shape[0], dtype=np.float32) + 0.5) * d_two_theta_rad

                # eta: full rotation around the shell, with 0.5-bin offset
                eta_lin = 2.0 * np.pi * (np.arange(self.shape[1], dtype=np.float32) + 0.5) / self.shape[1]

                # Create meshgrid
                ETA, TWO_THETA = np.meshgrid(eta_lin, two_theta_lin, indexing='xy')

                # Convert to Cartesian on unit sphere
                # Standard spherical coordinates: (theta=two_theta from +x axis, phi=eta azimuthal)
                X_shell = np.cos(TWO_THETA)
                Y_shell = np.sin(TWO_THETA) * np.sin(ETA)
                Z_shell = np.sin(TWO_THETA) * np.cos(ETA)

                # Update pixel coordinates with shell projection
                self._pixel_coordinates = np.vstack((X_shell.ravel(), Y_shell.ravel(), Z_shell.ravel()))

    def read_detector_metadata(self, override_directory=None):
        """
        Restore detector metadata from disk.

        Reads a JSON file named "detector_metadata.json" and restores fields
        that describe the detector state. Pixel arrays (coordinates or values)
        are not restored by this method.

        Args:
            override_directory (str, optional): If provided, the JSON is read
              from this directory instead of self.directory.

        Raises:
            FileNotFoundError: If the metadata file does not exist at the
              resolved path.

        Side Effects:
            Updates internal attributes such as shape, pixel_size, center,
            direction, two_theta, eta, distance, and geometry.
        """
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "detector_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "detector_metadata.json")

        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            detector_metadata = json.load(f)

        # Convert lists back to NumPy arrays (where appropriate)
        if detector_metadata["shape"] is not None:
            self._shape = tuple(detector_metadata["shape"])
        if detector_metadata["pixel_size"] is not None:
            self._pixel_size = np.array(detector_metadata["pixel_size"], dtype=np.float32)
        if detector_metadata["center"] is not None:
            self._center = np.array(detector_metadata["center"], dtype=np.float32)
        if detector_metadata["direction"] is not None:
            self._direction = np.array(detector_metadata["direction"], dtype=np.float32)
        if detector_metadata["two_theta"] is not None:
            self._two_theta = float(detector_metadata["two_theta"])
        if detector_metadata["eta"] is not None:
            self._eta = float(detector_metadata["eta"])
        if detector_metadata["distance"] is not None:
            self._distance = float(detector_metadata["distance"])
        self._geometry = detector_metadata["geometry"]

        # Load construction_mode with backwards compatibility (default to 'plane')
        self._construction_mode = detector_metadata.get("construction_mode", "plane")

        # Load input_mode with backwards compatibility (default to 'spatial')
        self._input_mode = detector_metadata.get("input_mode", "spatial")

        # Load angular_range (None for spatial mode)
        angular_range = detector_metadata.get("angular_range")
        self._angular_range = tuple(angular_range) if angular_range is not None else None

        # Regenerate pixel coordinates from loaded metadata
        if self._shape is not None and self._pixel_size is not None and self._geometry is not None:
            # Use create_detector to rebuild geometry (includes shell projection if needed)
            # Note: This resets center, direction, distance, two_theta, eta to zero
            # We'll restore the loaded values below
            saved_distance = self._distance
            saved_two_theta = self._two_theta
            saved_eta = self._eta
            saved_center = self._center.copy() if self._center is not None else None
            saved_direction = self._direction.copy() if self._direction is not None else None

            # For angular mode, use the angular_range as shape
            if self._input_mode == 'angular' and self._angular_range is not None:
                shape_to_use = self._angular_range
            else:
                shape_to_use = self._shape

            self.create_detector(
                shape=shape_to_use,
                pixel_size=self._pixel_size,
                geometry=self._geometry,
                construction_mode=self._construction_mode,
                input_mode=self._input_mode
            )

            # Restore loaded position values
            self._distance = saved_distance
            self._two_theta = saved_two_theta
            self._eta = saved_eta

            # Apply stored position (distance, two_theta, eta) to pixel coordinates
            if self._distance is not None and self._distance > 0:
                two_theta = self._two_theta if self._two_theta is not None else 0
                eta = self._eta if self._eta is not None else 0

                # Scale to distance if needed:
                # - Shell mode: always scale (built at unit radius)
                # - Angular mode + plane: scale (built at unit distance)
                if self._construction_mode == 'shell':
                    self._pixel_coordinates *= self._distance
                elif self._input_mode == 'angular' and self._construction_mode == 'plane':
                    self._pixel_coordinates *= self._distance

                # Compute rotation matrix
                R = self.get_rotation_detector(two_theta, eta)

                # Update detector normal direction
                base_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self._direction = R @ base_normal

                # Compute center position on sphere
                self._center = self._distance * np.array([
                    np.cos(two_theta),
                    np.sin(two_theta) * np.sin(eta),
                    np.sin(two_theta) * np.cos(eta)
                ], dtype=np.float32)

                # Apply rotation based on construction mode
                if self._construction_mode == 'plane':
                    # Plane mode: rotate then translate to center
                    self._pixel_coordinates = R @ self._pixel_coordinates + self._center[:, None]
                else:  # shell mode
                    # Shell mode: only rotate (already at distance from origin)
                    self._pixel_coordinates = R @ self._pixel_coordinates

    ## Data Handling Functions
    def write_detector_metadata(self, override_directory=None):
        """
        Write detector metadata to a JSON file.

        Serializes key fields (shape, pixel_size, center, direction, two_theta,
        eta, distance, geometry) to "detector_metadata.json". Pixel arrays are
        not stored.

        Args:
          override_directory (str, optional): If provided, write into this
            directory instead of self.directory.

        Returns:
          None
        """
        # Convert NumPy arrays/types to native Python types so JSON can handle them
        detector_metadata = {
            "shape": [int(x) for x in self._shape] if self._shape is not None else None,
            "pixel_size": [float(x) for x in self._pixel_size] if self._pixel_size is not None else None,
            "center": [float(x) for x in self._center] if self._center is not None else None,
            "direction": [float(x) for x in self._direction] if self._direction is not None else None,
            "two_theta": float(self._two_theta) if self._two_theta is not None else None,
            "eta": float(self._eta) if self._eta is not None else None,
            "distance": float(self._distance) if self._distance is not None else None,
            "geometry": str(self._geometry) if self._geometry is not None else None,
            "construction_mode": str(self._construction_mode) if hasattr(self, '_construction_mode') and self._construction_mode is not None else "plane",
            "input_mode": str(self._input_mode) if hasattr(self, '_input_mode') and self._input_mode is not None else "spatial",
            "angular_range": [float(x) for x in self._angular_range] if hasattr(self, '_angular_range') and self._angular_range is not None else None
        }

        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "detector_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "detector_metadata.json")

        # Write as nicely formatted JSON
        with open(metadata_filename, "w") as f:
            json.dump(detector_metadata, f, indent=4)
        print(f"Detector metadata written to {metadata_filename} in JSON format.")
        
    def write_Efield_values(self, field=None, filename="Efield_values.npy", override_directory=None):
        """
        Save complex per-pixel field values to a .npy file.

        By default the method saves the current internal pixel values. If
        a NumPy or CuPy array is provided through the 'field' argument, that
        array is saved instead.

        Args:
          field (np.ndarray or cupy.ndarray, optional): Complex array to save.
            If None, saves self._pixel_values.
          filename (str, optional): Output filename. Defaults to
            "Efield_values.npy".
          override_directory (str, optional): If provided, write into this
            directory instead of self.directory.

        Raises:
          ValueError: If no internal pixel values are available and 'field'
            is None.

        Returns:
          None
        """
        if override_directory is not None:
            outfile = os.path.join(override_directory, filename)
        else:
            outfile = os.path.join(self.directory, filename)
        if field == None:
            if self._pixel_values is None:
                raise ValueError("Pixel values are not initialized; cannot save them.")
            pxvals = self._pixel_values
            print(f"Detector pixel values saved to {outfile}.")
        else:
            pxvals = field
            print(f"Field values saved to {outfile}.")
        # Convert CuPy arrays back to host memory before saving ---------------
        if cp is not None and isinstance(pxvals, cp.ndarray):
            pxvals = pxvals.get()
        np.save(outfile, pxvals) 
        
    def read_Efield_values(self, internal=True, filename="Efield_values.npy", override_directory=None):
        """
        Load complex per-pixel field values from a .npy file.

        If 'internal' is True, the loaded values are assigned to the detector
        and derived quantities (phase, amplitude, intensity) are updated.
        Otherwise the loaded array is returned.

        Args:
          internal (bool, optional): If True, assign to detector state and
            print a message. If False, return the loaded array. Default True.
          filename (str, optional): Filename to read. Defaults to
            "Efield_values.npy".
          override_directory (str, optional): If provided, read from this
            directory instead of self.directory.

        Raises:
          FileNotFoundError: If the specified file does not exist.

        Returns:
          None or np.ndarray: Returns the loaded array only when
          internal=False.
        """
        if override_directory is not None:
            infile = os.path.join(override_directory, filename)
        else:
            infile = os.path.join(self.directory, filename)
        if not os.path.isfile(infile):
            raise FileNotFoundError(f"No .npy file found at {infile}")
        loaded_values = np.load(infile)
        if internal == True:
            self.input_pixel_values(loaded_values)
            print(f"Detector pixel values loaded from {infile}.")
        else:
            print(f"Field values loaded from {infile}.")
            return loaded_values
            
    
    ## Static Functions
    @staticmethod
    def get_rotation(axis,angle):
        """
        Compute a 3x3 right-handed rotation matrix.

        Uses Rodrigues' formula to rotate by 'angle' radians about 'axis'. The
        input axis is normalized internally.

        Args:
          axis (array-like): Rotation axis, shape (3,).
          angle (float): Rotation angle in radians.

        Returns:
          np.ndarray: Rotation matrix of shape (3, 3).
        """
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        d = 1.0 - c
        x, y, z = axis
        return np.array([[c + d*x*x,     d*x*y - z*s,   d*x*z + y*s],
                         [d*y*x + z*s,   c + d*y*y,     d*y*z - x*s],
                         [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]])
        
    ## Main Functions     
    def position_detector_relative(self, distance, two_theta, eta, degrees=True):
        """
        Move the detector by relative increments in distance, 2theta, and eta.

        Internally, this computes the new absolute pose and reuses
        position_detector_absolute to apply it robustly.

        Args:
            distance (float): Relative change to distance (same units as stored).
            two_theta (float): Relative change to 2theta (deg if degrees=True else rad).
            eta (float): Relative change to eta (deg if degrees=True else rad).
            degrees (bool): Interpret the angular increments in degrees if True.

        Returns:
            None
        """
        # Current absolute angles are stored in radians
        dtt = np.deg2rad(two_theta) if degrees else float(two_theta)
        det = np.deg2rad(eta) if degrees else float(eta)

        new_distance = float(self._distance) + float(distance)
        new_two_theta = float(self._two_theta) + dtt
        new_eta = float(self._eta) + det

        # Reuse absolute setter to avoid compounding transforms
        self.position_detector_absolute(new_distance, new_two_theta, new_eta, degrees=False)
        
    def position_detector_absolute(self, distance, two_theta, eta, degrees=True):
        """
        Set the detector to an absolute pose given (distance, two_theta, eta).

        This rebuilds the base pixel grid using the detector's current geometry
        (rectangular or ring) and construction mode (plane or shell), then applies
        the absolute rotation and translation.

        For plane mode: Pixels are rotated and translated to the specified position.
        For shell mode: Pixels are scaled to the distance (built at unit radius),
                       then rotated. All pixels remain equidistant from origin.

        Angles are interpreted in degrees if degrees=True.

        Args:
            distance (float): Absolute sample-to-detector-center distance (Angstroms).
                             For shell mode, this is the radius of the spherical shell.
            two_theta (float): Absolute 2theta angle (rotation about +Y axis).
            eta (float): Absolute eta angle (rotation about +X axis).
            degrees (bool): Interpret angles as degrees if True, radians if False.

        Returns:
            None

        Notes:
            - Shell mode pixels maintain constant distance from origin (spherical shell)
            - Plane mode pixels maintain constant distance from detector center (flat plane)
            - The method preserves the construction_mode set during create_detector()
        """
        if self._shape is None or self._pixel_size is None:
            raise ValueError(
                "Detector shape/pixel_size are not initialized. "
                "Call create_detector(...) before positioning."
            )

        # Convert to radians if needed
        if degrees:
            two_theta = np.deg2rad(two_theta)
            eta = np.deg2rad(eta)

        # Rebuild the base grid in its canonical plane, preserving geometry and construction mode
        geometry = self._geometry if self._geometry is not None else "rectangular"
        construction_mode = self._construction_mode if hasattr(self, '_construction_mode') and self._construction_mode is not None else "plane"
        input_mode = self._input_mode if hasattr(self, '_input_mode') and self._input_mode is not None else "spatial"

        # For angular mode, use the saved angular_range as shape
        if input_mode == 'angular' and hasattr(self, '_angular_range') and self._angular_range is not None:
            shape_to_use = self._angular_range
        else:
            shape_to_use = self.shape

        self.create_detector(shape_to_use, self.pixel_size, geometry=geometry,
                           construction_mode=construction_mode, input_mode=input_mode)

        # Record absolute state
        self._two_theta = float(two_theta)
        self._eta = float(eta)
        self._distance = float(distance)

        # Scale to distance if needed:
        # - Shell mode: always scale (built at unit radius)
        # - Angular mode + plane: scale (built at unit distance)
        if self._construction_mode == 'shell':
            self._pixel_coordinates *= self._distance
        elif self._input_mode == 'angular' and self._construction_mode == 'plane':
            # Angular mode plane detectors are built at unit distance, scale by actual distance
            self._pixel_coordinates *= self._distance

        # Absolute rotation for this pose
        R = self.get_rotation_detector(self._two_theta, self._eta)

        # Update the detector normal (start from +x after create_detector)
        base_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self._direction = R @ base_normal

        # Center on the 2theta/eta sphere at the requested distance
        self._center = self._distance * np.array([
            np.cos(self._two_theta),
            np.sin(self._two_theta) * np.sin(self._eta),
            np.sin(self._two_theta) * np.cos(self._eta)
        ], dtype=np.float32)

        # Apply rotation based on construction mode
        if self._construction_mode == 'plane':
            # Plane mode: rotate then translate to center
            self._pixel_coordinates = R @ self._pixel_coordinates + self._center[:, None]
        else:  # shell mode
            # Shell mode: only rotate (already at distance from origin)
            self._pixel_coordinates = R @ self._pixel_coordinates
        self.display_detector_values(degrees=True)
        
    def get_rotation_detector(self, two_theta, eta):
        """
        Composite rotation used by this detector.

        Returns a matrix that applies +two_theta about +y and then +eta about
        +x. The order matches the convention used throughout this class.

        Args:
          two_theta (float): Rotation about +y in radians.
          eta (float): Rotation about +x in radians.

        Returns:
          np.ndarray: Rotation matrix of shape (3, 3).
        """
        eta_matrix = self.get_rotation(np.array([-1.0, 0.0, 0.0]), eta)
        two_theta_matrix = self.get_rotation(eta_matrix@np.array([0.0, -1.0, 0.0]), two_theta)
        return eta_matrix @ two_theta_matrix
        
    def get_detector_position_cartesian(self):
        """
        Return the detector center and normal vector.

        Returns:
          tuple[np.ndarray, np.ndarray]: (center, direction), where each has
          shape (3,).
        """
        return self._center,self._direction
    
    def input_pixel_values(self,pixel_values):
        """
        Assign complex pixel values and update derived fields.

        Args:
          pixel_values (np.ndarray): Complex array matching the detector pixel
            layout. The array is stored directly; no copy is forced.

        Returns:
          None
        """
        self._pixel_values = pixel_values
        self._pixel_phase = np.angle(self.pixel_values)
        self._pixel_amplitude = np.abs(self.pixel_values)
        self._pixel_intensity = self.pixel_amplitude**2

    def coordinate_conversion(self,data,input_system="cartesian",output_system="angular",units="deg"):
        """
        Convert between Cartesian and angular coordinate systems.

        The angular representation uses:
          - component 0: eta
          - component 1: two_theta
          - component 2: distance (radius)

        The Cartesian representation uses:
          - component 0: x
          - component 1: y
          - component 2: z

        Args:
          data (np.ndarray): Stacked coordinates of shape (3, N).
          input_system (str): "cartesian" or "angular". Default "cartesian".
          output_system (str): "cartesian" or "angular". Default "angular".
          units (str): "deg" or "rad". Applies when converting to or from
            angular coordinates. Default "deg".

        Returns:
          np.ndarray: Converted coordinates with the same shape as input.

        Raises:
          ValueError: If an unsupported output_system is requested.

        Notes:
          - When converting to Cartesian and units == "deg", angles are first
            converted to radians.
          - When converting to angular and units == "deg", outputs are returned
            in degrees.
        """
        if input_system == output_system:
            return data
        elif output_system == "cartesian":
            if units == "deg":
                data[0] = np.deg2rad(data[0])
                data[1] = np.deg2rad(data[1])
            eta_pixels = data[0]
            two_theta_pixels = data[1]
            distance = data[2]
            x = distance * np.cos(two_theta_pixels)
            y = distance * np.sin(two_theta_pixels) * np.sin(eta_pixels)
            z = distance * np.sin(two_theta_pixels) * np.cos(eta_pixels)
            return np.stack((x, y, z), axis=0)
        elif output_system == "angular":
            two_theta_pixels = np.arctan2(np.sqrt(data[1]**2 + data[2]**2), data[0])
            eta_pixels = np.arctan2(data[1], data[2])
            distance = np.sqrt(data[0]**2 + data[1]**2 + data[2]**2)
            if units == "deg":
                two_theta_pixels = np.rad2deg(two_theta_pixels)
                eta_pixels = np.rad2deg(eta_pixels)
            return np.stack((eta_pixels, two_theta_pixels, distance), axis=0)
        
    def get_detector_axis(
        self, 
        system="angular", 
        units="deg", 
        axis=None
    ):
        """
        Return detector coordinates arranged on the pixel grid.

        Retrieves pixel coordinates, optionally converts to angular space, and
        reshapes to a 3D array suitable for plotting or analysis.

        Args:
          system (str): "cartesian" or "angular". Default "angular".
          units (str): "deg" or "rad" for angular output. Ignored for
            "cartesian". Default "deg".
          axis (int or None): If None, return full array of shape
            (3, shape[1], shape[0]). If 0, 1, or 2, return only that component
            with shape (shape[1], shape[0]). The ordering (shape[1], shape[0])
            matches how meshgrid is constructed in this class.

        Returns:
          np.ndarray: Coordinates with shape (3, shape[1], shape[0]) or a
          single component with shape (shape[1], shape[0]).

        Raises:
          ValueError: If an unknown 'system' is requested.
        """
        # 1) Get raw pixel coordinates (move from GPU to host if needed) -------
        if isinstance(self.pixel_coordinates, cp.ndarray):
            coords = self.pixel_coordinates.get()
        else:
            coords = self.pixel_coordinates
        # 2) Convert if needed --------------------------------------------------
        if system == "angular":
            coords = self.coordinate_conversion(coords, output_system="angular", units=units)
        elif system != "cartesian":
            raise ValueError(f"Unknown system '{system}'. Use 'cartesian' or 'angular'.")
        # 3) Reshape from (3, Ny*Nz) -> (3, Nz, Ny); image-like orientation ----
        coords_reshaped = coords.reshape(3, self.shape[1], self.shape[0])
        # 4) Optional axis selection -------------------------------------------
        if axis is not None:
            return coords_reshaped[axis]
        else:
            return coords_reshaped
        
    def display_detector_values(self, degrees=True):
        """
        Print a table of key detector values (distance, angles, center, normal).

        Args:
            degrees (bool): If True, angles are shown in degrees; otherwise radians.

        Returns:
            None
        """
        tt = float(self._two_theta) if self._two_theta is not None else 0.0
        et = float(self._eta) if self._eta is not None else 0.0
        if degrees:
            tt_out = np.rad2deg(tt)
            et_out = np.rad2deg(et)
            tt_label = "two_theta_deg"
            et_label = "eta_deg"
        else:
            tt_out = tt
            et_out = et
            tt_label = "two_theta_rad"
            et_label = "eta_rad"

        print("Detector State:")
        cols = [
            "distance", tt_label, et_label
        ]
        vals = [[
            float(self._distance if self._distance is not None else 0.0), float(tt_out), float(et_out)
        ]]
        if pd is not None:
            df = pd.DataFrame(vals, columns=cols)
            df.index = ["Values"]
            print(df)
        else:
            # Fallback plain-text table if pandas is unavailable
            row = vals[0]
            for c, v in zip(cols, row):
                print(f"  {c:>12s}: {v:.6g}")
        
    def plot_detector(
        self,
        type="Intensity",
        title=None,
        scaling="linear",
        vmin=None,vmax=None,
        xlim=None,ylim=None,
        figsize=(8, 6), 
        cmap="gist_gray"
    ):
        """
        Display a 2D image of pixel values on the detector.

        For plane mode: Uses imshow with the detector extent computed from shape * pixel_size.
        For shell mode: Uses orthographic projection with scipy.interpolate.griddata to map
                       the spherical shell onto a 2D plane. Falls back to scatter plot if
                       scipy is unavailable.

        The method supports plotting Intensity, Phase, or Amplitude.

        Args:
          type (str): "Intensity", "Phase", or "Amplitude". Default "Intensity".
          title (str or None): Matplotlib title. If None, uses 'type' with mode indicator.
          scaling (str): "linear" or "log". For "log", applies log10 to data.
          vmin (float or None): Lower color limit. Default None.
          vmax (float or None): Upper color limit. Default None.
          xlim (tuple[float, float] or None): X-axis limits. Default None.
          ylim (tuple[float, float] or None): Y-axis limits. Default None.
          figsize (tuple[float, float]): Figure size in inches. Default (8, 6).
          cmap (str or Colormap): Colormap for the image. Default "gist_gray".

        Returns:
          tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: (fig, ax).

        Notes:
          - For "Phase", a black-white-black segmented colormap is used.
          - When scaling="log", non-positive values are not handled specially
            here; ensure your data are positive or preprocessed.
          - Shell mode automatically adds "(shell mode)" to the title.
          - Shell mode visualization is an orthographic projection onto the Y-Z plane.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as pltcolor
        if type == "Intensity":
            plot_val = self.pixel_intensity
            cmap = cmap
        elif type == "Phase":
            plot_val = self.pixel_phase
            colors = ["white", "black", "white"]
            cmap=pltcolor.LinearSegmentedColormap.from_list("", colors)
        elif type == "Amplitude":
            plot_val = self.pixel_amplitude
            cmap = cmap
        if scaling == "log":
            plot_val = np.log10(plot_val)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1)

        # Check if shell mode requires orthographic projection
        if hasattr(self, '_construction_mode') and self._construction_mode == 'shell':
            # Use orthographic projection onto YZ plane for shell geometry
            coords = self.pixel_coordinates
            Y, Z = coords[1], coords[2]

            # Create regular grid for interpolation
            y_min, y_max = Y.min(), Y.max()
            z_min, z_max = Z.min(), Z.max()

            # Add small padding to avoid edge artifacts
            y_padding = (y_max - y_min) * 0.05
            z_padding = (z_max - z_min) * 0.05
            y_min -= y_padding
            y_max += y_padding
            z_min -= z_padding
            z_max += z_padding

            grid_y, grid_z = np.mgrid[y_min:y_max:256j, z_min:z_max:256j]

            # Interpolate pixel values onto regular grid
            try:
                from scipy.interpolate import griddata
                plot_val_flat = plot_val.ravel()
                points = np.column_stack((Y, Z))
                grid_values = griddata(points, plot_val_flat, (grid_y, grid_z), method='linear')
            except ImportError:
                print("Warning: scipy not available, using scatter plot instead of interpolation")
                sc = ax1.scatter(Y, Z, c=plot_val.ravel(), s=2, cmap=cmap, vmin=vmin, vmax=vmax)
                ax1.set_xlabel("Y (Å)")
                ax1.set_ylabel("Z (Å)")
                if title is None:
                    ax1.set_title(f"{type} (shell mode)")
                else:
                    ax1.set_title(title)
                ax1.axis('scaled')
                cbar = fig.colorbar(sc, ax=ax1)
                cbar.set_label(type)
                if xlim is not None:
                    ax1.set_xlim(xlim)
                if ylim is not None:
                    ax1.set_ylim(ylim)
                return fig, ax1

            # Plot interpolated grid
            im = ax1.imshow(
                grid_values.T,  # Transpose for correct orientation
                extent=[y_min, y_max, z_min, z_max],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='equal'
            )
            ax1.set_xlabel("Y (Å)")
            ax1.set_ylabel("Z (Å)")
            if title is None:
                ax1.set_title(f"{type} (shell mode)")
            else:
                ax1.set_title(title)
        else:
            # Plane mode - use existing imshow implementation
            # Compute extent from half-size in each dimension
            detector_extent = self._shape*self._pixel_size/2
            im = ax1.imshow(
                plot_val,
                extent=[-detector_extent[0], detector_extent[0], -detector_extent[1], detector_extent[1]],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax
            )
            ax1.set_xlabel("Y (Å)")
            ax1.set_ylabel("Z (Å)")
            if title is None:
                ax1.set_title(type)
            else:
                ax1.set_title(title)
            ax1.axis('scaled')

        # Common formatting for both modes
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label(type)
        if xlim is not None:
            ax1.set_xlim(xlim)
        if ylim is not None:
            ax1.set_ylim(ylim)
        return fig, ax1
    
    def plot_detector_angles(
        self,
        type="Intensity",
        title=None,
        scaling="linear",
        degrees=True,
        figsize=(8, 6),
        cmap="gist_gray",
        vmin=None,vmax=None,
        xlim=None,ylim=None,
        marker_size=2,
        grid_resolution=256
    ):
        """
        Visualize pixel values in diffraction-angle space (η vs 2θ).

        For detectors that form a regular grid in angular space (ring + shell geometry,
        or angular input mode), uses direct imshow. For all other configurations
        (rectangular plane, ring plane, etc.), interpolates onto a regular grid
        for smooth visualization using scipy.griddata.

        Args:
          type (str): "Intensity", "Phase", or "Amplitude". Default "Intensity".
          title (str or None): Matplotlib title. If None, uses 'type'.
          scaling (str): "linear" or "log". For "log", applies log10 after
            replacing non-positive values with a small positive number.
          degrees (bool): If True, axes are shown in degrees; otherwise radians.
          figsize (tuple[float, float]): Figure size in inches. Default (8, 6).
          cmap (str or Colormap): Colormap used for the plot.
          vmin (float or None): Lower color limit. Default None.
          vmax (float or None): Upper color limit. Default None.
          xlim (tuple[float, float] or None): X-axis limits. Default None.
          ylim (tuple[float, float] or None): Y-axis limits. Default None.
          marker_size (float): Marker size for scatter plot fallback. Default 2.
          grid_resolution (int): Resolution of interpolation grid for detectors
            not forming a regular angular grid (default 256). Ignored for
            ring + shell geometry or angular input mode detectors.

        Returns:
          tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: (fig, ax).

        Raises:
          ValueError: If an unknown 'type' is requested or if required arrays
            are not initialized.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as pltcolor

        # ----- choose data and colormap ---------------------------------------
        if type == "Intensity":
            plot_val = self.pixel_intensity
        elif type == "Phase":
            plot_val = self.pixel_phase
            cmap = pltcolor.LinearSegmentedColormap.from_list("", ["white", "black", "white"])
        elif type == "Amplitude":
            plot_val = self.pixel_amplitude
        else:
            raise ValueError("Unknown plot type {!r}. Choose from 'Intensity', 'Phase', or 'Amplitude'.".format(type))

        if plot_val is None:
            raise ValueError(f"Detector pixel values for '{type}' have not been initialized.")

        # ----- optional log scaling -------------------------------------------
        if scaling == "log":
            plot_val = np.where(plot_val <= 0, 1e-20, plot_val)  # avoid log(0)
            plot_val = np.log10(plot_val)

        # ----- convert detector coordinates to angles -------------------------
        coords = self.pixel_coordinates
        if coords is None:
            raise ValueError("Detector pixel coordinates are not initialized.")

        x, y, z = coords
        two_theta_pixels = np.arctan2(np.sqrt(y**2 + z**2), x)
        eta_pixels       = np.arctan2(y, z)

        if degrees:
            two_theta_pixels = np.degrees(two_theta_pixels)
            eta_pixels       = np.degrees(eta_pixels)

        # ----- plotting --------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        # Determine if detector is rectangular in angular space
        geometry = getattr(self, '_geometry', 'rectangular')
        construction_mode = getattr(self, '_construction_mode', 'plane')
        input_mode = getattr(self, '_input_mode', 'spatial')

        # Detectors that form a regular grid in angular space:
        # - Ring + Shell: pixels at regular (2θ, η) intervals
        # - Angular input mode: creates regular angular grids by definition
        is_regular_angular_grid = (
            (geometry == 'ring' and construction_mode == 'shell') or
            input_mode == 'angular'
        )

        if is_regular_angular_grid:
            # Direct imshow - pixels form a regular grid in angular space
            # Reshape to 2D array matching detector shape
            plot_val_2d = plot_val.reshape(self._shape)

            # Compute extent from angular coordinates
            eta_min, eta_max = eta_pixels.min(), eta_pixels.max()
            two_theta_min, two_theta_max = two_theta_pixels.min(), two_theta_pixels.max()

            im = ax.imshow(
                plot_val_2d,  # Transpose: shape is (Ny, Nz), imshow expects (rows, cols)
                extent=[eta_min, eta_max, two_theta_min, two_theta_max],
                origin='lower',
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect='auto'
            )
            cbar = fig.colorbar(im, ax=ax)

        else:
            # Non-rectangular: use interpolation
            try:
                from scipy.interpolate import griddata

                # Determine grid bounds with padding
                eta_min, eta_max = eta_pixels.min(), eta_pixels.max()
                two_theta_min, two_theta_max = two_theta_pixels.min(), two_theta_pixels.max()

                eta_padding = (eta_max - eta_min) * 0.02
                two_theta_padding = (two_theta_max - two_theta_min) * 0.02

                eta_min -= eta_padding
                eta_max += eta_padding
                two_theta_min -= two_theta_padding
                two_theta_max += two_theta_padding

                # Create regular grid for interpolation
                grid_eta, grid_two_theta = np.mgrid[
                    eta_min:eta_max:complex(0, grid_resolution),
                    two_theta_min:two_theta_max:complex(0, grid_resolution)
                ]

                # Interpolate pixel values onto regular grid
                points = np.column_stack((eta_pixels, two_theta_pixels))
                grid_values = griddata(points, plot_val.ravel(), (grid_eta, grid_two_theta), method='linear')

                # Plot interpolated grid
                im = ax.imshow(
                    grid_values,
                    extent=[eta_min, eta_max, two_theta_min, two_theta_max],
                    origin='lower',
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect='auto'
                )
                cbar = fig.colorbar(im, ax=ax)

            except ImportError:
                print("Warning: scipy not available, falling back to scatter plot")
                # Fallback to scatter
                sc = ax.scatter(
                    eta_pixels,
                    two_theta_pixels,
                    c=plot_val.ravel(),
                    s=marker_size,
                    cmap=cmap,
                    marker='.',
                    vmin=vmin,
                    vmax=vmax
                )
                cbar = fig.colorbar(sc, ax=ax)

        cbar.set_label(type)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(r"$\eta$" + (" (deg)" if degrees else " (rad)"))
        ax.set_ylabel(r"$2\theta$" + (" (deg)" if degrees else " (rad)"))
        if title is None:
            # Add construction mode indicator to title
            mode_str = 'shell' if (hasattr(self, '_construction_mode') and self._construction_mode == 'shell') else 'plane'
            ax.set_title(f"{type} ({mode_str} mode)")
        else:
            ax.set_title(title)
        return fig, ax

    def plot_detector_position(self,elev=0,azim=90,figsize=(8, 8),title="Detector Position"):
        """
        Visualize detector center and pixels in 3D.

        Produces a simple 3D scatter plot with the origin in red and pixels
        in blue.

        Args:
          elev (float): Elevation for 3D view_init in degrees. Default 0.
          azim (float): Azimuth for 3D view_init in degrees. Default 90.
          figsize (tuple[float, float]): Figure size in inches. Default (8, 8).
          title (str): Axes title. Default "Detector Position".

        Returns:
          tuple[matplotlib.figure.Figure, matplotlib.axes._subplots.Axes3DSubplot]:
          (fig, ax).
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        ax1.scatter(
                0,
                0,
                0,
                c='r', marker='o'
            )
        ax1.scatter(
                self.pixel_coordinates[0, :],
                self.pixel_coordinates[1, :],
                self.pixel_coordinates[2, :],
                c='b', marker='.'
            )
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_proj_type('ortho')
        ax1.axis('scaled')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title(title)
        return fig, ax1
    
    # Properties
    @property
    def center(self):
        """np.ndarray: Detector center position (shape (3,))."""
        if self._center is None:
            print("self._center has not been initialized yet")
        return self._center
    
    @property
    def shape(self):
        """tuple[int, int]: Detector shape in pixels as (Ny, Nz)."""
        if self._shape is None:
            print("self._shape has not been initialized yet")
        return self._shape
    
    @property
    def pixel_size(self):
        """np.ndarray: Pixel size along (y, z)."""
        if self._pixel_size is None:
            print("self._pixel_size has not been initialized yet")
        return self._pixel_size
    
    @property
    def pixel_coordinates(self):
        """np.ndarray: Stacked pixel coordinates with shape (3, Npixels)."""
        if self._pixel_coordinates is None:
            print("self._pixel_coordinates has not been initialized yet")
        return self._pixel_coordinates
    
    @property
    def pixel_values(self):
        """np.ndarray: Complex per-pixel field values."""
        if self._pixel_values is None:
            print("self._pixel_values has not been initialized yet")
        return self._pixel_values
    
    @property
    def pixel_phase(self):
        """np.ndarray: Phase of complex pixel values."""
        if self._pixel_phase is None:
            self._pixel_phase = np.angle(self.pixel_values)
        return self._pixel_phase
    
    @property
    def pixel_amplitude(self):
        """np.ndarray: Magnitude of complex pixel values."""
        if self._pixel_amplitude is None:
            self._pixel_amplitude = np.abs(self.pixel_values)
        return self._pixel_amplitude
    
    @property
    def pixel_intensity(self):
        """np.ndarray: Intensity equals magnitude squared of pixel values."""
        if self._pixel_intensity is None:
            self._pixel_intensity = np.abs(self.pixel_values)**2
        return self._pixel_intensity
    
    @property
    def two_theta(self):
        """float: Accumulated two_theta angle in radians."""
        if self._two_theta is None:
            print("self._two_theta has not been initialized yet")
        return self._two_theta
    
    @property
    def eta(self):
        """float: Accumulated eta angle in radians."""
        if self._eta is None:
            print("self._eta has not been initialized yet")
        return self._eta

    @property
    def distance(self):
        """float: Accumulated distance along detector normal."""
        if self._distance is None:
            print("self._distance has not been initialized yet")
        return self._distance

    @property
    def input_mode(self):
        """str: Input mode used to create this detector ('spatial' or 'angular')."""
        return self._input_mode

    @property
    def angular_range(self):
        """tuple or None: Angular range for detectors created in angular mode.

        Returns:
            For rectangular: (theta_y_min, theta_y_max, theta_z_min, theta_z_max) in degrees
            For ring: (two_theta_inner, two_theta_outer) in degrees
            For spatial mode: None
        """
        return self._angular_range
