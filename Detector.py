# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import json
import os

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class detector:

    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=os.getcwd()):
        """Initialize a detector object.

        Args:
          directory (str, optional): Folder used for I/O operations. Defaults
            to the current working directory. The directory is created if it
            does not exist.
        """
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
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        
    def create_detector(self, shape, pixel_size, geometry='rectangular'):
        """Create the pixel layout on the y-z plane.

        Creates a detector pixel grid in either a rectangular or ring layout.
        The initial plane lies on x = 0, so the initial detector normal points
        along +x. Pixel coordinates are stored in a stacked array with shape
        (3, Npixels) in the order (x, y, z), raveled in C order.

        Args:
          shape (tuple[int, int]): For "rectangular", (Ny, Nz). For "ring",
            (N_two_theta, N_eta), where N_two_theta is the number of radial
            bins and N_eta is the number of azimuth bins.
          pixel_size (tuple[float, float]): For "rectangular", (dy, dz). For
            "ring", (d_two_theta, d_eta) where d_eta is an arc length per
            angular bin at the inner radius.
          geometry (str): "rectangular" (default) or "ring".

        Raises:
          ValueError: If an unsupported geometry string is provided.

        Notes:
          - For the rectangular layout, the grid is centered at (0, 0) in the
            y-z plane with uniform spacing.
          - For the ring layout, pixels are placed at radii r_in + (i + 0.5)*d_two_theta
            and angles 2*pi*(j + 0.5)/N_eta.
        """
        self._shape = shape
        self._pixel_size = pixel_size
        self._center = np.array([0.0, 0.0, 0.0])
        self._direction = np.array([1.0, 0.0, 0.0])
        self._two_theta = 0
        self._eta = 0
        self._distance = 0
        self._geometry = geometry.lower()

        # Build the initial pixel array in the y-z plane (x = 0) ----------------
        if geometry.lower() == 'rectangular':
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
        
    def read_detector_metadata(self, override_directory=None):
        """Restore detector metadata from disk.

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
        
    ## Data Handling Functions
    def write_detector_metadata(self, override_directory=None):
        """Write detector metadata to a JSON file.

        Serializes key fields (shape, pixel_size, center, direction, two_theta,
        eta, distance, geometry) to "detector_metadata.json". Pixel arrays are
        not stored.

        Args:
          override_directory (str, optional): If provided, write into this
            directory instead of self.directory.

        Returns:
          None
        """
        # Convert NumPy arrays to Python lists so JSON can handle them
        detector_metadata = {
            "shape": list(self._shape) if self._shape is not None else None,
            "pixel_size": list(self._pixel_size) if self._pixel_size is not None else None,
            "center": self._center.tolist() if self._center is not None else None,
            "direction": self._direction.tolist() if self._direction is not None else None,
            "two_theta": float(self._two_theta) if self._two_theta is not None else None,
            "eta": float(self._eta) if self._eta is not None else None,
            "distance": float(self._distance) if self._distance is not None else None,
            "geometry": self._geometry
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
        """Save complex per-pixel field values to a .npy file.

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
        if override_directory is not None:
            outfile = os.path.join(override_directory, filename)
        else:
            outfile = os.path.join(self.directory, filename)
        np.save(outfile, pxvals) 
        
    def read_Efield_values(self, internal=True, filename="Efield_values.npy", override_directory=None):
        """Load complex per-pixel field values from a .npy file.

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
        """Compute a 3x3 right-handed rotation matrix.

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
        """Move the detector by relative increments.

        Applies additional rotations and translation to the current state.
        Order of operations matches get_rotation_detector: rotate by
        +two_theta about +y, then by +eta about +x, then translate along the
        detector normal so that the center is at the requested distance.

        Args:
          distance (float): Additional distance to add to the current distance.
          two_theta (float): Additional angle. Degrees if degrees=True.
          eta (float): Additional angle. Degrees if degrees=True.
          degrees (bool, optional): If True, inputs are in degrees; they are
            converted to radians internally. Default True.

        Returns:
          None
        """
        if degrees:
            two_theta = np.deg2rad(two_theta)
            eta = np.deg2rad(eta)
        self._two_theta += two_theta
        self._eta += eta
        self._distance  += distance

        # Compute the incremental rotation and apply it to the normal and pixel plane
        detector_rotation_np = self.get_rotation_detector(two_theta, eta)
        self._direction = detector_rotation_np @ self._direction

        # Update center location based on the accumulated orientation and distance
        self._center = self._distance * np.array([
            np.cos(self._two_theta),
            np.sin(self._two_theta) * np.sin(self._eta),
            np.sin(self._two_theta) * np.cos(self._eta)
        ])

        # Rotate and translate all pixel coordinates
        self._pixel_coordinates = detector_rotation_np @ self._pixel_coordinates + self._center[:, None]
        
    def position_detector_absolute(self, distance, two_theta, eta, degrees=True):
        """Set an absolute detector pose, rebuilding the pixel plane first.

        Recreates the base pixel layout via create_detector using the current
        shape and pixel_size, then sets absolute values of distance, two_theta,
        and eta. The pixel plane is then rotated and translated to the final
        pose.

        Args:
          distance (float): Absolute distance to set.
          two_theta (float): Absolute angle. Degrees if degrees=True.
          eta (float): Absolute angle. Degrees if degrees=True.
          degrees (bool, optional): If True, inputs are in degrees; they are
            converted to radians internally. Default True.

        Notes:
          - This method calls create_detector(self.shape, self.pixel_size) with
            default arguments. If you require a "ring" geometry, call
            create_detector with geometry="ring" prior to calling this method.

        Returns:
          None
        """
        self.create_detector(self.shape, self.pixel_size)
        if degrees:
            two_theta = np.deg2rad(two_theta)
            eta = np.deg2rad(eta)
        self._two_theta = two_theta
        self._eta = eta
        self._distance  = distance

        # Compute the absolute rotation and apply it to the normal and pixel plane
        detector_rotation_np = self.get_rotation_detector(self._two_theta, self._eta)
        self._direction = detector_rotation_np @ self._direction

        # Update center location based on the absolute orientation and distance
        self._center = self._distance * np.array([
            np.cos(self._two_theta),
            np.sin(self._two_theta) * np.sin(self._eta),
            np.sin(self._two_theta) * np.cos(self._eta)
        ])

        # Rotate and translate all pixel coordinates
        self._pixel_coordinates = detector_rotation_np @ self._pixel_coordinates + self._center[:, None]
        
    def get_rotation_detector(self, two_theta, eta):
        """Composite rotation used by this detector.

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
        """Return the detector center and normal vector.

        Returns:
          tuple[np.ndarray, np.ndarray]: (center, direction), where each has
          shape (3,).
        """
        return self._center,self._direction
    
    def input_pixel_values(self,pixel_values):
        """Assign complex pixel values and update derived fields.

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
        """Convert between Cartesian and angular coordinate systems.

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
        """Return detector coordinates arranged on the pixel grid.

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
        """Display a 2D image of pixel values on the detector plane.

        Uses imshow with the detector extent computed from shape * pixel_size.
        The method supports plotting Intensity, Phase, or Amplitude.

        Args:
          type (str): "Intensity", "Phase", or "Amplitude". Default "Intensity".
          title (str or None): Matplotlib title. If None, uses 'type'.
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
        
        # Compute extent from half-size in each dimension ----------------------
        detector_extent = self._shape*self._pixel_size/2
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1)
        im = ax1.imshow(
            plot_val,
            extent=[-detector_extent[0], detector_extent[0], -detector_extent[1], detector_extent[1]],
            origin='lower', 
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax1.set_xlabel("X (Å)")
        ax1.set_ylabel("Y (Å)")
        if title is None:
            ax1.set_title(type)
        else:
            ax1.set_title(title)
        ax1.axis('scaled')
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
        marker_size=2
    ):
        """Scatter plot of pixel values in diffraction-angle space.

        The x-axis is eta and the y-axis is two_theta. Values are drawn from
        Intensity, Phase, or Amplitude.

        Args:
          type (str): "Intensity", "Phase", or "Amplitude". Default "Intensity".
          title (str or None): Matplotlib title. If None, uses 'type'.
          scaling (str): "linear" or "log". For "log", applies log10 after
            replacing non-positive values with a small positive number.
          degrees (bool): If True, axes are shown in degrees; otherwise radians.
          figsize (tuple[float, float]): Figure size in inches. Default (8, 6).
          cmap (str or Colormap): Colormap used for scatter points.
          vmin (float or None): Lower color limit. Default None.
          vmax (float or None): Upper color limit. Default None.
          xlim (tuple[float, float] or None): X-axis limits. Default None.
          ylim (tuple[float, float] or None): Y-axis limits. Default None.
          marker_size (float): Matplotlib marker size passed to scatter. Default 2.

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

        # Colorbar adopts the same limits automatically
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(type)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(r"$\eta$" + (" (deg)" if degrees else " (rad)"))
        ax.set_ylabel(r"$2\theta$" + (" (deg)" if degrees else " (rad)"))
        if title is None:
            ax.set_title(type)
        else:
            ax.set_title(title)
        return fig, ax

    def plot_detector_position(self,elev=0,azim=90,figsize=(8, 8),title="Detector Position"):
        """Visualize detector center and pixels in 3D.

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
