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
        
    def create_detector(self, shape, pixel_size, geometry='rectangular', use_gpu=True):
        """
        Create detector pixels in either a rectangular or ring geometry on the YZ-plane.

        Parameters
        ----------
        shape : tuple of two ints
            Number of pixels in each dimension (e.g. (Ny, Nz) for rectangular,
            or (NtwoTheta, Neta) for ring).
        pixel_size : tuple of two floats
            Pixel size in each dimension (e.g. (dy, dz) for rectangular,
            or (dTwoTheta, dEta) for ring).
        geometry : str
            'rectangular' (default) or 'ring'.
        use_gpu : bool
            Whether to use cupy (GPU) if available.
        """
        self._shape = shape
        self._pixel_size = pixel_size
        self._center = np.array([0.0, 0.0, 0.0])
        self._direction = np.array([1.0, 0.0, 0.0])
        self._two_theta = 0
        self._eta = 0
        self._distance = 0
        self._geometry = geometry.lower()
        if geometry.lower() == 'rectangular':
            if cp is None or not use_gpu:
                y_lin = np.linspace(-(self.shape[0]/2)*self.pixel_size[0],
                                    +(self.shape[0]/2)*self.pixel_size[0],
                                    self.shape[0], dtype=np.float32)
                z_lin = np.linspace(-(self.shape[1]/2)*self.pixel_size[1],
                                    +(self.shape[1]/2)*self.pixel_size[1],
                                    self.shape[1], dtype=np.float32)
                Y, Z = np.meshgrid(y_lin, z_lin)
                X = np.full_like(Y, 0)
                self._pixel_coordinates = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
            else:
                y_lin = cp.linspace(-(self.shape[0]/2)*self.pixel_size[0],
                                    +(self.shape[0]/2)*self.pixel_size[0],
                                    self.shape[0], dtype=cp.float32)
                z_lin = cp.linspace(-(self.shape[1]/2)*self.pixel_size[1],
                                    +(self.shape[1]/2)*self.pixel_size[1],
                                    self.shape[1], dtype=cp.float32)
                Y, Z = cp.meshgrid(y_lin, z_lin)
                X = cp.full_like(Y, 0)
                self._pixel_coordinates = cp.vstack((X.ravel(), Y.ravel(), Z.ravel()))
        elif geometry.lower() == 'ring':
            # Inner radius so that circumference matches (Neta * pixel_size_eta)
            r_in = (self.shape[1] * self.pixel_size[1]) / (2 * np.pi)
            # Outer radius adds (NtwoTheta * pixel_size_twoTheta)
            r_out = r_in + self.shape[0] * self.pixel_size[0]
            # shape[0] = number of pixels in two_theta direction
            # shape[1] = number of pixels in eta direction
            if cp is None or not use_gpu:
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
                i_lin = cp.arange(self.shape[0], dtype=cp.float32)
                j_lin = cp.arange(self.shape[1], dtype=cp.float32)
                r_lin = r_in + (i_lin + 0.5) * self.pixel_size[0]
                phi_lin = 2.0 * cp.pi * (j_lin + 0.5) / self.shape[1]
                PHI, R = cp.meshgrid(phi_lin, r_lin, indexing='xy')
                Y = R * cp.cos(PHI)
                Z = R * cp.sin(PHI)
                X = cp.zeros_like(Y)
                self._pixel_coordinates = cp.vstack((X.ravel(), Y.ravel(), Z.ravel()))
        else:
            raise ValueError(f"Unknown geometry '{geometry}'. Choose 'rectangular' or 'ring'.")
        
    def read_detector_metadata(self, override_directory=None):
        """
        Reads the metadata JSON file from disk and restores
        this detector object's state.
        (Does not restore pixel values or pixel coordinates.)
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
        """
        Serializes the detector object's critical internal fields to disk 
        as human-readable JSON so that the state can be restored later.
        (Does not store pixel values or pixel coordinates.)
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
    
    ## Static Functions
    @staticmethod
    def get_rotation(axis,angle):
        """
        Return the 3x3 rotation matrix for rotation by 'angle' radians
        around the (normalized) 'axis'.
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
    def position_detector_relative(self, distance, two_theta, eta, degrees=True, use_gpu=True):
        if degrees:
            two_theta = np.deg2rad(two_theta)
            eta = np.deg2rad(eta)
        self._two_theta += two_theta
        self._eta        += eta
        self._distance  += distance
        detector_rotation_np = self.get_rotation_detector(two_theta, eta)
        self._direction = detector_rotation_np @ self._direction
        self._center = self._distance * np.array([
            np.cos(self._two_theta),
            np.sin(self._two_theta) * np.sin(self._eta),
            np.sin(self._two_theta) * np.cos(self._eta)
        ])
        if cp is None or not use_gpu:
            self._pixel_coordinates = detector_rotation_np @ self._pixel_coordinates + self._center[:, None]
        else:
            detector_rotation_cp = cp.asarray(detector_rotation_np)
            self._pixel_coordinates = detector_rotation_cp @ self._pixel_coordinates + cp.asarray(self._center[:, None])
        
    def position_detector_absolute(self, distance, two_theta, eta, degrees=True, use_gpu=True):
        self.create_detector(self.shape, self.pixel_size, use_gpu=use_gpu)
        if degrees:
            two_theta = np.deg2rad(two_theta)
            eta = np.deg2rad(eta)
        self._two_theta = two_theta
        self._eta        = eta
        self._distance  = distance
        detector_rotation_np = self.get_rotation_detector(self._two_theta, self._eta)
        self._direction = detector_rotation_np @ self._direction
        self._center = self._distance * np.array([
            np.cos(self._two_theta),
            np.sin(self._two_theta) * np.sin(self._eta),
            np.sin(self._two_theta) * np.cos(self._eta)
        ])
        if cp is None or not use_gpu:
            self._pixel_coordinates = detector_rotation_np @ self._pixel_coordinates + self._center[:, None]
        else:
            detector_rotation_cp = cp.asarray(detector_rotation_np)
            self._pixel_coordinates = detector_rotation_cp @ self._pixel_coordinates + cp.asarray(self._center[:, None])
        
    def get_rotation_detector(self, two_theta, eta):
        """
        Return the 3x3 rotation matrix for applying
        +two_theta around +y, then +eta around +x.
        """
        eta_matrix = self.get_rotation(np.array([-1.0, 0.0, 0.0]), eta)
        two_theta_matrix = self.get_rotation(eta_matrix@np.array([0.0, -1.0, 0.0]), two_theta)
        return eta_matrix @ two_theta_matrix
        
    def get_detector_position_cartesian(self):
        return self._center,self._direction
    
    def input_pixel_values(self,pixel_values):
        self._pixel_values = pixel_values
        self._pixel_phase = np.angle(self.pixel_values)
        self._pixel_amplitude = np.abs(self.pixel_values)
        self._pixel_intensity = self.pixel_amplitude**2

    def coordinate_conversion(self,data,input_system="cartesian",output_system="angular",units="deg"):
        """
        Returns values in either cartesian or angular coordinates, depending on
        the specified input_system and output_system.

        Parameters
        ----------
        data : ndarray
            Input coordinate array. The shape must be (3, N) where
            - For 'cartesian': data[0] = X, data[1] = Y, data[2] = Z
            - For 'angular': data[0] = eta, data[1] = 2theta, data[2] = distance
        input_system : str
            'cartesian' or 'angular'. Default = 'cartesian'.
        output_system : str
            'cartesian' or 'angular'. Default = 'cartesian'.

        Returns
        -------
        converted : ndarray
            The coordinate array after conversion. Same shape, (3, N).
        """
        if input_system == output_system:
            return data
        elif output_system == "cartesian":
            two_theta_pixels = data[0]
            eta_pixels = data[1]
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
        
    def get_detector_axis(self, system="angular", units="deg", axis=None):
        """
        Returns the detector coordinates (3, Nx, Ny) in either cartesian or angular
        form. If 'axis' is specified, return only that axis (0=x/eta, 1=y/2theta, 2=z/distance).

        Parameters
        ----------
        system : str
            "cartesian" or "angular". Default "angular".
        units : str
            "deg" or "rad" (only relevant for "angular" output).
        axis : None or int
            If None, return the full 3D array shaped (3, Nx, Ny).
            If an integer (0, 1, or 2), return just that axis with shape (Nx, Ny).
        """
        # 1) Get raw pixel coordinates
        if isinstance(self.pixel_coordinates, cp.ndarray):
            coords = self.pixel_coordinates.get()
        else:
            coords = self.pixel_coordinates
        # 2) Convert if needed
        if system == "angular":
            coords = self.coordinate_conversion(coords, output_system="angular", units=units)
        elif system != "cartesian":
            raise ValueError(f"Unknown system '{system}'. Use 'cartesian' or 'angular'.")
        # 3) Reshape from (3, Nx*Ny) -> (3, Nx, Ny), matching 2D pixel_data shape
        coords_reshaped = coords.reshape(3, self.shape[1], self.shape[0])
        # 4) If 'axis' is given, return only that slice; else return full (3, Nx, Ny)
        if axis is not None:
            return coords_reshaped[axis]
        else:
            return coords_reshaped

        
    def plot_detector(self,type="Intensity",scaling="linear",limits=np.array([0,1]),figsize=(8, 6)):
        import matplotlib.pyplot as plt
        import matplotlib.colors as pltcolor
        if type == "Intensity":
            plot_val = self.pixel_intensity
            cmap = 'gist_yarg'
        elif type == "Phase":
            plot_val = self.pixel_phase
            colors = ["white", "black", "white"]
            cmap=pltcolor.LinearSegmentedColormap.from_list("", colors)
        elif type == "Amplitude":
            plot_val = self.pixel_amplitude
            cmap = 'gist_yarg'
        if scaling == "log":
            plot_val = np.log(plot_val)
        
        detector_extent = self._shape*self._pixel_size/2
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1)
        im = ax1.imshow(
            plot_val,
            extent=[-detector_extent[0], detector_extent[0], -detector_extent[1], detector_extent[1]],
            origin='lower', cmap=cmap
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title(type)
        ax1.axis('scaled')
        fig.colorbar(im, ax=ax1)
        # fig.show()
        return fig, ax1
    
    def plot_detector_angles(self, type="Intensity", scaling="linear", degrees=True, figsize=(8, 6), use_gpu=True):
        """
        Compute and plot each pixel's value in diffraction-angle coordinates:
        - x-axis: eta
        - y-axis: two_theta
        
        Parameters
        ----------
        type : str
            Quantity to plot: "Intensity", "Phase", or "Amplitude".
        scaling : str
            Either "linear" or "log" scaling.
        degrees : bool
            If True, angles are plotted in degrees; if False, in radians.
        use_gpu : bool
            If True and cupy is available, then `self.pixel_coordinates` is on GPU
            memory; we must bring it to CPU to plot.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes._subplots.Axes3DSubplot (or 2D Axes)
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as pltcolor
        if type == "Intensity":
            plot_val = self.pixel_intensity
            cmap = 'gist_yarg'
        elif type == "Phase":
            plot_val = self.pixel_phase
            colors = ["white", "black", "white"]
            cmap = pltcolor.LinearSegmentedColormap.from_list("", colors)
        elif type == "Amplitude":
            plot_val = self.pixel_amplitude
            cmap = 'gist_yarg'
        else:
            raise ValueError(f"Unknown plot type '{type}'. Choose from 'Intensity', 'Phase', or 'Amplitude'.")
        if plot_val is None:
            raise ValueError(f"Detector pixel values for '{type}' have not been initialized.")
        if scaling == "log":
            plot_val = np.where(plot_val <= 0, 1e-20, plot_val)  # avoid log(0)
            plot_val = np.log(plot_val)
        coords = self.pixel_coordinates
        if coords is None:
            raise ValueError("Detector pixel coordinates are not initialized.")
        if cp is not None and use_gpu and isinstance(coords, cp.ndarray):
            coords = coords.get()
        x = coords[0]
        y = coords[1]
        z = coords[2]
        two_theta_pixels = np.arctan2(np.sqrt(y**2 + z**2), x)
        eta_pixels = np.arctan2(y, z) 
        if degrees:
            two_theta_pixels = np.degrees(two_theta_pixels)
            eta_pixels        = np.degrees(eta_pixels)
            
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(
            eta_pixels,
            two_theta_pixels,
            c=plot_val.ravel(),
            s=2,
            cmap=cmap,
            marker='.'
        )
        cbar = plt.colorbar(sc, ax=ax)
        ax.set_xlabel(r"$\eta$" + (" (deg)" if degrees else " (rad)"))
        ax.set_ylabel(r"$2\theta$" + (" (deg)" if degrees else " (rad)"))
        cbar.set_label(type)
        ax.set_title(type)
        ax.axis('scaled')
        return fig, ax

    def plot_detector_position(self,elev=0,azim=90,figsize=(8, 8)):
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
                self.pixel_coordinates[0, :].get(),
                self.pixel_coordinates[1, :].get(),
                self.pixel_coordinates[2, :].get(),
                c='b', marker='.'
            )
        ax1.view_init(elev=elev, azim=azim)
        ax1.set_proj_type('ortho')
        ax1.axis('scaled')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        return fig, ax1
    
    @property
    def shape(self):
        """
        Returns the detector shape in pixels.
        """
        if self._shape is None:
            print("self._shape has not been initialized yet")
        return self._shape
    
    @property
    def pixel_size(self):
        """
        Returns the pixel size in Angstroms.
        """
        if self._pixel_size is None:
            print("self._pixel_size has not been initialized yet")
        return self._pixel_size
    
    @property
    def pixel_coordinates(self):
        """
        Returns the detector pixel locations in either cartesian coordinates.
        """
        if self._pixel_coordinates is None:
            print("self._pixel_coordinates has not been initialized yet")
        return self._pixel_coordinates
    
    @property
    def pixel_values(self):
        """
        Returns the complex pixel values.
        """
        if self._pixel_values is None:
            print("self._pixel_values has not been initialized yet")
        return self._pixel_values
    
    @property
    def pixel_phase(self):
        """
        Returns the pixel phase values.
        """
        if self._pixel_phase is None:
            self._pixel_phase = np.angle(self.pixel_values)
        return self._pixel_phase
    
    @property
    def pixel_amplitude(self):
        """
        Returns the pixel amplitude values.
        """
        if self._pixel_amplitude is None:
            self._pixel_amplitude = np.abs(self.pixel_values)
        return self._pixel_amplitude
    
    @property
    def pixel_intensity(self):
        """
        Returns the pixel intensity values.
        """
        if self._pixel_intensity is None:
            self._pixel_intensity = np.abs(self.pixel_values)**2
        return self._pixel_intensity
    
    @property
    def two_theta(self):
        """
        Returns the detector two_theta values.
        """
        if self._two_theta is None:
            print("self._two_theta has not been initialized yet")
        return self._two_theta
    
    @property
    def eta(self):
        """
        Returns the detector eta values.
        """
        if self._eta is None:
            print("self._eta has not been initialized yet")
        return self._eta

    @property
    def distance(self):
        """
        Returns the detector distance values.
        """
        if self._distance is None:
            print("self._distance has not been initialized yet")
        return self._distance