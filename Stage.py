# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
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
        self._name = None
        self._motor_name = None
        self._motor_type = None
        self._motor_coupling = None
        self._motor_axis = None
        self._motor_value = None
        self._motor_resolution = None
        self._rotation = None
        self._translation = None
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)
    
    def create_stage(self, name=['goniometer'], motor_name=['mu','phi','chi','omega','x','y','z'], motor_type=['R','R','R','R','T','T','T'], motor_coupling=[[None],[0],[0],[0,1,2],[0,1,2,3],[0,1,2,3],[0,1,2,3]], motor_axis=[[0.0,-1.0,0.0],[0.0,-1.0,0.0],[-1.0,0.0,0.0],[0.0,0.0,-1.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], motor_value=[0.0,0.0,0.0,0.0,0.0,0.0,0.0], motor_resolution=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
        """
        Create the stage with a specified mode/convention (e.g. 'goniometer').
        This will initialize all angles to zero and translation to [0,0,0].
        """
        self._name = name
        self._motor_name = np.array(motor_name)
        self._motor_type = np.array(motor_type)
        self._motor_coupling = motor_coupling
        self._motor_axis = np.array(motor_axis)
        self._motor_value = np.array(motor_value)
        self._motor_resolution = np.array(motor_resolution)
        self._rotation = np.eye(3, dtype=np.float32)
        self._translation = np.zeros(3, dtype=np.float32)
        print(f"{name} stage created with all angles and translations set to 0.")

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
        # Convert lists back to NumPy arrays (and normal Python objects where needed)
        if stage_metadata["name"] is not None:
            self._name = stage_metadata["name"]
        if stage_metadata["motor_name"] is not None:
            self._motor_name = np.array(stage_metadata["motor_name"])
        if stage_metadata["motor_type"] is not None:
            self._motor_type = np.array(stage_metadata["motor_type"])
        if stage_metadata["motor_coupling"] is not None:
            self._motor_coupling = stage_metadata["motor_coupling"]
        if stage_metadata["motor_axis"] is not None:
            self._motor_axis = np.array(stage_metadata["motor_axis"], dtype=np.float32)
        if stage_metadata["motor_value"] is not None:
            self._motor_value = np.array(stage_metadata["motor_value"], dtype=np.float32)
        if stage_metadata["motor_resolution"] is not None:
            self._motor_resolution = np.array(stage_metadata["motor_resolution"], dtype=np.float32)
        if stage_metadata["rotation"] is not None:
            self._rotation = np.array(stage_metadata["rotation"], dtype=np.float32)
        if stage_metadata["translation"] is not None:
            self._translation = np.array(stage_metadata["translation"], dtype=np.float32)
            
    ## Data Handling Functions
    def write_stage_metadata(self, override_directory=None):
        """
        Serializes the stage object's critical fields to disk
        as human-readable JSON so that the state can be restored later.
        """
        # Convert NumPy arrays to Python lists so JSON can handle them
        stage_metadata = {
            "name": self._name if self._name is not None else None,
            "motor_name": self._motor_name.tolist() if self._motor_name is not None else None,
            "motor_type": self._motor_type.tolist() if self._motor_type is not None else None,
            "motor_coupling": self._motor_coupling if self._motor_coupling is not None else None,
            "motor_axis": self._motor_axis.tolist() if self._motor_axis is not None else None,
            "motor_value": self._motor_value.tolist() if self._motor_value is not None else None,
            "motor_resolution": self._motor_resolution.tolist() if self._motor_resolution is not None else None,
            "rotation": self._rotation.tolist() if self._rotation is not None else None,
            "translation": self._translation.tolist() if self._translation is not None else None
        }
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "stage_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "stage_metadata.json")
        # Write as nicely formatted JSON
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
    
    ## Main Functions
    def get_motor_axis(self,motor_idx):
        """
        Return the motion axis for a given motor given the coupling and motor values provided.
        """
        motor_axis_arr = self._motor_axis.copy()
        for coupled in self._motor_coupling[motor_idx]:
            print(f"Motor ID:{motor_idx} axis array = {motor_axis_arr[motor_idx]}")
            if coupled is None:
                break
            else:
                motor_axis_arr[motor_idx] = self.get_axis_rotation(self.get_motor_axis(coupled),self._motor_value[coupled]) @ motor_axis_arr[motor_idx]
        return motor_axis_arr[motor_idx]    

    def get_rotation(self):
        """
        Return the 3x3 rotation matrix given the coupling and motor values provided.
        """
        motor_mask = (self._motor_type == "R")
        rotation_matrix = np.eye(3, dtype=np.float32)
        for motor_idx in np.where(motor_mask)[0]:
            rotation_matrix = self.get_axis_rotation(self.get_motor_axis(motor_idx), self._motor_value[motor_idx]) @ rotation_matrix
        self._rotation = rotation_matrix
        return self._rotation
    
    def get_translation(self):
        """
        Return the 3x1 translation array given the coupling and motor values provided.
        """
        motor_mask = (self._motor_type == "T")
        translation_arr = np.zeros(3, dtype=np.float32)
        for motor_idx in np.where(motor_mask)[0]:
            translation_arr += self.get_motor_axis(motor_idx)*self._motor_value[motor_idx]
        self._translation = translation_arr
        return self._translation
        
    def set_motor_value_relative(self, motor_value_rel=[0.0,0.0,0.0,0.0,0.0,0.0,0.0], degrees=True):
        motor_value_rel = np.array(motor_value_rel)
        if degrees is True:
            motor_mask = (self._motor_type == "R")
            motor_value_rel[motor_mask] = np.deg2rad(motor_value_rel[motor_mask])
            self._motor_value += motor_value_rel
        else:
            self._motor_value += motor_value_rel
        self.clip_motor_value()
        self.get_rotation()
        self.get_translation()
        self.display_motor_values(degrees=degrees)
    
    def set_motor_value_absolute(self, motor_value_abs=[0.0,0.0,0.0,0.0,0.0,0.0,0.0], degrees=True):
        motor_value_abs = np.array(motor_value_abs)
        if degrees is True:
            motor_mask = (self._motor_type == "R")
            motor_value_abs[motor_mask] = np.deg2rad(motor_value_abs[motor_mask])
            self._motor_value = motor_value_abs
        else:
            self._motor_value = motor_value_abs
        self.clip_motor_value()
        self.get_rotation()
        self.get_translation()
        self.display_motor_values(degrees=degrees)
    
    def set_single_motor_value_relative(self, name, value, degrees=True):
        """
        """
        motor_mask = (self._motor_name == name)
        if self._motor_type[motor_mask] == "R" and degrees is True:
            value = np.deg2rad(value)
        self._motor_value[motor_mask] += value
        self.clip_motor_value()
        self.get_rotation()
        self.get_translation()
        self.display_motor_values(degrees=degrees)
        
    def set_single_motor_value_absolute(self, name, value, degrees=True):
        """
        """
        motor_mask = (self._motor_name == name)
        if self._motor_type[motor_mask] == "R" and degrees is True:
            value = np.deg2rad(value)
        self._motor_value[motor_mask] = value
        self.clip_motor_value()
        self.get_rotation()
        self.get_translation()
        self.display_motor_values(degrees=degrees)
    
    def clip_motor_value(self):
        """
        Clips motor values according to resolution.
        """
        motor_mask = (self._motor_resolution is not None) & (self._motor_resolution != 0)
        self._motor_value[motor_mask] = self._motor_resolution[motor_mask]*int(self._motor_value[motor_mask]/self._motor_resolution[motor_mask])

    def zero_stage(self):
        """
        Reset all motor values to zero.
        """
        self._motor_value *= 0
        self._rotation = np.eye(3, dtype=np.float32)
        self._translation = np.zeros(3, dtype=np.float32)
        print("Stage angles and translation have been zeroed.")
        
    def display_motor_values(self,degrees=True):
        if degrees is True:
            motor_mask = (self._motor_type == "R")
            output = self._motor_value.copy()
            output[motor_mask] = np.rad2deg(self._motor_value[motor_mask])
        else:
            output = self._motor_value.copy()
        print("Motor States:")
        df = pd.DataFrame([output], columns=self._motor_name)
        df.index = ['Values']
        print(df)
    
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
        corners_lab = (corners_local @ self.get_rotation()) + self.get_translation()
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
    def motor_value(self):
        """
        Returns the motor values.
        """
        if self._motor_value is None:
            print("Stage motors not initialized. Please call create_stage(...) first.")
        return self._motor_value
    
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
    

