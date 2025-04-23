# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import pickle
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
        self._nu = None
        self._pixel_coordinates = None
        self._pixel_values = None
        self._pixel_phase = None
        self._pixel_amplitude = None
        self._pixel_intensity = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        
    def create_detector(self,shape,pixel_size,use_gpu=True):
        self._shape = shape
        self._pixel_size = pixel_size
        self._center = np.array([0.0,0.0,0.0])
        self._direction = np.array([1.0,0.0,0.0])
        self._two_theta = 0
        self._nu = 0
        # Create detector pixel grid
        if cp is None or not use_gpu:
            y_lin = np.linspace(-(self.shape[0]/2)*self.pixel_size[0],
                                +(self.shape[0]/2)*self.pixel_size[0],
                                self.shape[0], dtype=np.float32)
            z_lin = np.linspace(-(self.shape[1]/2)*self.pixel_size[1],
                                +(self.shape[1]/2)*self.pixel_size[1],
                                self.shape[1], dtype=np.float32)
            Y,Z  = np.meshgrid(y_lin,z_lin)
            X = np.full_like(Y,0)
            self._pixel_coordinates = np.vstack((X.ravel(),Y.ravel(),Z.ravel()))
        else:
            y_lin = cp.linspace(-(self.shape[0]/2)*self.pixel_size[0],
                                +(self.shape[0]/2)*self.pixel_size[0],
                                self.shape[0], dtype=cp.float32)
            z_lin = cp.linspace(-(self.shape[1]/2)*self.pixel_size[1],
                                +(self.shape[1]/2)*self.pixel_size[1],
                                self.shape[1], dtype=cp.float32)
            Y,Z  = cp.meshgrid(y_lin,z_lin)
            X = cp.full_like(Y,0)
            self._pixel_coordinates = cp.vstack((X.ravel(),Y.ravel(),Z.ravel()))
    
    ## Data Handling Functions
    def write_detector_metadata(self): #incomplete
        detector_metadata = [self._shape]
    
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
    def position_detector_relative(self,distance,two_theta,nu,degrees=True,use_gpu=True):
        if degrees:
            two_theta = np.deg2rad(two_theta)
            nu = np.deg2rad(nu)
        self._two_theta += two_theta
        self._nu += nu
        detector_rotation_np = self.get_rotation_detector(two_theta,nu)
        self._direction = self._direction@detector_rotation_np.T
        self._center = distance*np.array([np.cos(self._two_theta),np.sin(self._two_theta)*np.sin(self._nu),np.sin(self._two_theta)*np.cos(self._nu)])
        if cp is None or not use_gpu:
            self._pixel_coordinates = detector_rotation_np @ self._pixel_coordinates + self._center[:,None]
        else:
            detector_rotation_cp = cp.asarray(detector_rotation_np)
            self._pixel_coordinates = detector_rotation_cp @ self._pixel_coordinates + cp.asarray(self._center[:,None])
        
    def position_detector_absolute(self,distance,two_theta,nu,degrees=True,use_gpu=True):
        self.create_detector(self.shape,self.pixel_size,use_gpu=use_gpu)
        if degrees:
            two_theta = np.deg2rad(two_theta)
            nu = np.deg2rad(nu)
        self._two_theta = self._two_theta
        self._nu = self._nu
        detector_rotation_np = self.get_rotation_detector(self._two_theta,self._nu)
        self._direction = self._direction@detector_rotation_np.T
        self._center = distance*self._direction + self._center
        if cp is None or not use_gpu:
            self._pixel_coordinates = detector_rotation_np @ self._pixel_coordinates + self._center[:,None]
        else:
            detector_rotation_cp = cp.asarray(detector_rotation_np)
            self._pixel_coordinates = detector_rotation_cp @ self._pixel_coordinates + cp.asarray(self._center[:,None])
        
    def get_rotation_detector(self,two_theta,nu):
        """
        Return the 3x3 rotation matrix for the detector.
        """
        two_theta_matrix = self.get_rotation(np.array([0.0,-1.0,0.0]),two_theta)
        nu_matrix = self.get_rotation(np.array([-1.0,0.0,0.0]),nu)
        return nu_matrix@two_theta_matrix
        
    def get_detector_position_cartesian(self):
        return self._center,self._direction
    
    def input_pixel_values(self,pixel_values):
        self._pixel_values = pixel_values
        self._pixel_phase = np.angle(self.pixel_values)
        self._pixel_amplitude = np.abs(self.pixel_values)
        self._pixel_intensity = self.pixel_amplitude**2
        
    def plot_detector(self,type="Intensity",scaling="linear",limits=np.array([0,1])):
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
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(1, 1, 1)
        im = ax1.imshow(
            plot_val,
            extent=[-detector_extent[0], detector_extent[0], -detector_extent[1], detector_extent[1]],
            origin='lower', cmap=cmap
        )
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title(type)
        fig.colorbar(im, ax=ax1)
        # fig.show()
        return fig, ax1
    
    def plot_detector_position(self,elev=0,azim=90):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 8))
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
        ax1.axis('equal')
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_ylabel("Z")
        
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
        Returns the detector pixel locations.
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
            print("self._pixel_phase has not been initialized yet")
        return self._pixel_phase
    
    @property
    def pixel_amplitude(self):
        """
        Returns the pixel amplitude values.
        """
        if self._pixel_amplitude is None:
            print("self._pixel_amplitude has not been initialized yet")
        return self._pixel_amplitude
    
    @property
    def pixel_intensity(self):
        """
        Returns the pixel intensity values.
        """
        if self._pixel_intensity is None:
            print("self._pixel_intensity has not been initialized yet")
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
    def nu(self):
        """
        Returns the detector nu values.
        """
        if self._nu is None:
            print("self._nu has not been initialized yet")
        return self._nu
