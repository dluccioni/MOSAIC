# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import os
import json

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class stage:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=os.getcwd()):
        """
        Initializes the stage object with rotation and translation properties set to None.
        The user must call create_stage(...) to define the stage convention and
        initialize the angles/translations.
        """
        self.directory = directory
        self._omega = None
        self._phi   = None
        self._chi   = None
        self._mu    = None
        self._translation = None
        self._rotation = None
        self._mode = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
    
    def create_stage(self, mode="goniometer"):
        """
        Create the stage with a specified mode/convention (e.g. 'goniometer').
        This will initialize all angles to zero and translation to [0,0,0].
        """
        self._mode = mode
        self._omega = 0.0
        self._phi   = 0.0
        self._chi   = 0.0
        self._mu    = 0.0
        self._rotation = np.eye(3, dtype=np.float32)
        self._translation = np.zeros(3, dtype=np.float32)
        print(f"Stage created with mode '{self._mode}' and all angles, translations set to 0.")

    def read_stage_metadata(self):
        """
        Reads the stage metadata JSON file from disk and restores
        this stage object's state.
        """
        metadata_filename = os.path.join(self.directory, "stage_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            stage_metadata = json.load(f)

        if stage_metadata["mode"] is not None:
            self._mode = stage_metadata["mode"]
        if stage_metadata["omega"] is not None:
            self._omega = float(stage_metadata["omega"])
        if stage_metadata["phi"] is not None:
            self._phi = float(stage_metadata["phi"])
        if stage_metadata["chi"] is not None:
            self._chi = float(stage_metadata["chi"])
        if stage_metadata["mu"] is not None:
            self._mu = float(stage_metadata["mu"])
        if stage_metadata["translation"] is not None:
            self._translation = np.array(stage_metadata["translation"], dtype=np.float32)
            
    ## Data Handling Functions
    def write_stage_metadata(self, override_directory=None):
        """
        Serializes the stage object's critical fields to disk
        as human-readable JSON so that the state can be restored later.
        """
        # Convert the translation array to a Python list for JSON
        if self._translation is not None:
            translation_list = self._translation.tolist()
        else:
            translation_list = None
        
        stage_metadata = {
            "mode": self._mode,
            "omega": self._omega,
            "phi": self._phi,
            "chi": self._chi,
            "mu": self._mu,
            "translation": translation_list,
        }
        
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "stage_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "stage_metadata.json")

        with open(metadata_filename, "w") as f:
            json.dump(stage_metadata, f, indent=4)
        print(f"Metadata written to {metadata_filename} in JSON format.")
    
    ## Static Functions
    @staticmethod
    def get_unit_corners():
        unit_corners = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]], dtype=np.float32)
        return unit_corners
    
    @staticmethod
    def get_axis_rotation(axis, angle):
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
                         [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]], dtype=np.float32)
    
    def get_rotation(self, omega, phi, chi, mu, mode=None, degrees=True):
        """
        Return the 3x3 rotation matrix corresponding to (omega, phi, chi, mu)
        under the specified stage convention (defaults to self._mode if not given).
        
        Example: 'goniometer' mode uses four consecutive rotations about
                 axes defined by the sequence of transforms:
                 - mu rotation about [0,-1,0]
                 - omega rotation about (mu-rotated) [0,0,-1]
                 - phi rotation about (mu-rotated) [0,-1,0]
                 - chi rotation about (mu-rotated) [-1,0,0]
        """
        if mode is None:
            mode = self._mode
        if degrees:
            omega = np.deg2rad(omega)
            phi   = np.deg2rad(phi)
            chi   = np.deg2rad(chi)
            mu    = np.deg2rad(mu)
        
        if mode == "goniometer":
            # 1) mu rotation
            mu_matrix = self.get_axis_rotation(np.array([0.0, -1.0, 0.0], dtype=np.float32), mu)
            # 2) omega rotation
            omega_axis = mu_matrix @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
            omega_matrix = self.get_axis_rotation(omega_axis, omega)
            # 3) phi rotation
            phi_axis = mu_matrix @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
            phi_matrix = self.get_axis_rotation(phi_axis, phi)
            # 4) chi rotation
            chi_axis = mu_matrix @ np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            chi_matrix = self.get_axis_rotation(chi_axis, chi)
            return mu_matrix @ omega_matrix @ chi_matrix @ phi_matrix
        else:
            return np.eye(3, dtype=np.float32)
    
    ## Main Functions
    def set_rotation_stage_absolute(self, omega=0, phi=0, chi=0, mu=0, degrees=True):
        """
        Set the stage rotation to absolute angles (omega, phi, chi, mu).
        Overwrites any existing angles. Resets the stage to these angles.
        """
        if self._mode is None:
            raise ValueError("Stage mode not set. Please call create_stage(...) before using this function.")
        
        # Overwrite internal angles
        self._omega = float(omega)
        self._phi   = float(phi)
        self._chi   = float(chi)
        self._mu    = float(mu)
        
        self._rotation = self.get_rotation(self._omega,self._phi,self._chi,self._mu,degrees=degrees)
        
        print(f"Stage rotation set ABSOLUTE to: omega={omega}, phi={phi}, chi={chi}, mu={mu} (degrees={degrees}).")
    
    def set_rotation_stage_relative(self, domega=0, dphi=0, dchi=0, dmu=0, degrees=True):
        """
        Rotate the stage by angles (domega, dphi, dchi, dmu) relative to current angles.
        """
        if self._mode is None:
            raise ValueError("Stage mode not set. Please call create_stage(...) before using this function.")
        
        # Update angles
        self._omega += domega
        self._phi   += dphi
        self._chi   += dchi
        self._mu    += dmu
        
        self._rotation = self.get_rotation(self._omega,self._phi,self._chi,self._mu,degrees=degrees)
        
        print(f"Stage rotation set RELATIVE by: domega={domega}, dphi={dphi}, dchi={dchi}, dmu={dmu} (degrees={degrees}).")
    
    def set_translation_stage_absolute(self, x=0, y=0, z=0):
        """
        Set the stage translation to an absolute coordinate (x, y, z).
        """
        if self._translation is None:
            raise ValueError("Stage translation not initialized. Please call create_stage(...) before using this function.")
        self._translation = np.array([x, y, z], dtype=np.float32)
        print(f"Stage translation set ABSOLUTE to: [{x}, {y}, {z}].")
    
    def set_translation_stage_relative(self, dx=0, dy=0, dz=0):
        """
        Move the stage by a relative vector (dx, dy, dz) from the current translation.
        """
        if self._translation is None:
            raise ValueError("Stage translation not initialized. Please call create_stage(...) before using this function.")
        self._translation += np.array([dx, dy, dz], dtype=np.float32)
        print(f"Stage translation set RELATIVE by: [{dx}, {dy}, {dz}].")
    
    def zero_stage(self):
        """
        Reset all angles (omega, phi, chi, mu) and translation to zero.
        """
        if self._mode is None:
            raise ValueError("Stage mode not set. Please call create_stage(...) before using this function.")
        
        self._omega = 0.0
        self._phi   = 0.0
        self._chi   = 0.0
        self._mu    = 0.0
        
        self._rotation = np.eye(3, dtype=np.float32)
        self._translation = np.zeros(3, dtype=np.float32)
        
        print("Stage angles and translation have been zeroed.")
    
    def plot_stage(self, elev=30, azim=45):
        """
        Plots a centered unit cube (edges from -0.5 to +0.5 in each axis) after
        applying the stage's rotation and translation. Also plots red lines along
        the X, Y, and Z axes originating from [0,0,0].
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Get the cube corners
        corners_local = self.get_unit_corners() - 0.5

        # Apply rotation + translation to each corner
        corners_lab = (corners_local @ self.rotation) + self.translation
        edges = [
            (0,1), (0,2), (0,3),
            (1,4), (1,5),
            (2,4), (2,6),
            (3,5), (3,6),
            (4,7), (5,7), (6,7)
        ]

        # Plot each edge of the cube as a black line
        for (i1, i2) in edges:
            x_vals = [corners_lab[i1, 0], corners_lab[i2, 0]]
            y_vals = [corners_lab[i1, 1], corners_lab[i2, 1]]
            z_vals = [corners_lab[i1, 2], corners_lab[i2, 2]]
            ax.plot(x_vals, y_vals, z_vals, 'k-',linewidth=3)

        # Plot red lines for the coordinate axes from origin
        ax.plot([0, 2], [0, 0], [0, 0], 'r-')
        ax.plot([0, 0], [0, 2], [0, 0], 'g-')
        ax.plot([0, 0], [0, 0], [0, 2], 'b-')
        ax.plot([0, -2], [0, 0], [0, 0], 'r--')
        ax.plot([0, 0], [0, -2], [0, 0], 'g--')
        ax.plot([0, 0], [0, 0], [0, -2], 'b--')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type('ortho')
        ax.axis('equal')

        plt.tight_layout()
        plt.show()
        return fig, ax
    
    ## Properties
    @property
    def omega(self):
        """
        Returns the stage omega angle.
        """
        if self._omega is None:
            print("Stage angles not initialized. Please call create_stage(...) first.")
        return self._omega
    
    @property
    def phi(self):
        """
        Returns the stage phi angle.
        """
        if self._phi is None:
            print("Stage angles not initialized. Please call create_stage(...) first.")
        return self._phi
    
    @property
    def chi(self):
        """
        Returns the stage chi angle.
        """
        if self._chi is None:
            print("Stage angles not initialized. Please call create_stage(...) first.")
        return self._chi
    
    @property
    def mu(self):
        """
        Returns the stage mu angle.
        """
        if self._mu is None:
            print("Stage angles not initialized. Please call create_stage(...) first.")
        return self._mu
    
    @property
    def translation(self):
        """
        Returns the stage translation vector [x, y, z].
        """
        if self._translation is None:
            print("Stage translation not initialized. Please call create_stage(...) first.")
        return self._translation

    @property
    def rotation(self):
        """
        Returns the stage rotation matrix (3x3).
        """
        if self._rotation is None:
            print("Stage rotation not initialized. Please call create_stage(...) first.")
        return self._rotation
    
    @property
    def mode(self):
        """
        Returns the stage mode (e.g. 'goniometer').
        """
        if self._mode is None:
            print("Stage mode not initialized. Please call create_stage(...) first.")
        return self._mode
