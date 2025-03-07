# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import pickle
from pymatgen.core import Structure

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class crystal:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,filepath):
        self.filepath = filepath
        self._lattice_matrix = None
        self._lattice_corners = None
        self._lattice_center = None
        self._lattice_lengths = None
        self._lattice_volume = None
        self._lattice_atom_cartesian = None
        self._species = None
        self._lattice_atom_fractional = None
        self._lattice_matrix_conventional = None
        self._lattice_orientation = None
        self._cumulative_rotation = np.eye(3)
        self._default_filenames = np.array(["crystal_metadata.npy"]) #sample_metadata will be a struct
    
    def get_lattice_from_cif(self):
        self.structure = Structure.from_file(self.filepath)
        self.to_primitive()
        self._lattice_matrix = (self.structure.lattice.matrix) #.T
        self._lattice_corners = self.get_unit_corners()@self.lattice_matrix
        self._lattice_center = self.lattice_matrix/2
        self._lattice_lengths = np.array(self.structure.lattice.abc)
        self._lattice_volume = self.structure.lattice.volume
        self._lattice_atom_fractional = self.structure.frac_coords
        self._lattice_atom_cartesian = self.structure.cart_coords
        self._species = np.array([site.specie.name for site in self.structure.sites])
        self.to_conventional()
        self._lattice_matrix_conventional = (self.structure.lattice.matrix) #.T
        self._lattice_lengths_conventional = np.array(self.structure.lattice.abc)
        self._lattice_volume_conventional = self.structure.lattice.volume
        self._lattice_orientation = np.linalg.inv(self.lattice_matrix_conventional)/np.linalg.norm(np.linalg.inv(self.lattice_matrix_conventional), axis=0, keepdims=True)
        del self.structure
        
    ## Data Saving Functions
    def write_crystal_metadata(self): 
        crystal_metadata = [self.lattice_matrix]
    
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
        [1, 1, 1]])
        return unit_corners
    
    @staticmethod
    def get_rotation(axis, angle):
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
    
    def get_rotation_matrix_vector_to_vector(self,vector1, vector2, eps=1e-8):
        # 1) Compute cos(theta) from the dot product
        cos_theta = np.dot(vector1, vector2)

        # 2) Check for nearly identical or opposite vectors
        if np.isclose(cos_theta, 1.0, atol=eps):
            # Vectors are almost identical => identity
            return np.eye(3)

        if np.isclose(cos_theta, -1.0, atol=eps):
            # Vectors are opposite => rotate 180° about some axis orthogonal to 'vector1'
            fallback_axis = np.array([1.0, 0.0, 0.0])
            if np.allclose(vector1, fallback_axis, atol=eps) or np.allclose(vector1, -fallback_axis, atol=eps):
                fallback_axis = np.array([0.0, 1.0, 0.0])
            orth_axis = np.cross(vector1, fallback_axis)
            orth_axis /= np.linalg.norm(orth_axis)
            return self.get_rotation(orth_axis, np.pi)

        # 3) General case => angle is the one between the vectors
        #    axis = (vector1 x vector2), sin(theta) = ||axis||
        axis = np.cross(vector1, vector2)
        sin_theta = np.linalg.norm(axis)
        axis /= sin_theta
        # angle in [0, π]
        angle = np.arctan2(sin_theta, cos_theta)
        return self.get_rotation(axis, angle)
    
    def get_rotation_matrix_vector_to_plane(self,rotation_axis, vector1, plane1, angle_selection='small', eps=1e-12):
        # 1) Normalize the axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # 2) Set up the trig equation A cos(θ) + B sin(θ) + C = 0
        dot_vn         = np.dot(vector1, plane1)
        cross_axis_v   = np.cross(rotation_axis, vector1)
        dot_cross_axis = np.dot(cross_axis_v, plane1)
        dot_axis_v     = np.dot(rotation_axis, vector1)
        dot_axis_avn   = np.dot(rotation_axis * dot_axis_v, plane1)

        eqA = dot_vn - dot_axis_avn
        eqB = dot_cross_axis
        eqC = dot_axis_avn

        Rval = np.sqrt(eqA**2 + eqB**2)
        if Rval < eps:
            # Either no solution or infinite solutions
            if abs(eqC) < eps:
                # Already orthogonal => identity
                return np.eye(3)
            raise ValueError("No solution: cannot bring 'vector1' into plane by rotating about 'rotation_axis'.")

        cos_val = -eqC / Rval
        if abs(cos_val) > 1:
            raise ValueError("No real solution: |cos(θ - a)| > 1.")
        cos_val = np.clip(cos_val, -1, 1)

        offset_angle   = np.arctan2(eqB, eqA)       # a
        principal_angle = np.arccos(cos_val)        # arccos(...)
        angle_candidates = [offset_angle + principal_angle, offset_angle - principal_angle]

        # 3) Decide which angle to use
        if angle_selection == 'small':
            theta = min(angle_candidates, key=abs)
        else:
            theta = max(angle_candidates, key=abs)
        # 4) Build the rotation matrix using the shared Rodrigues function
        return self.get_rotation(rotation_axis, theta)
    
    def to_conventional(self):
        self.structure = self.structure.to_conventional()
        
    def to_primitive(self):
        self.structure = self.structure.to_primitive()
    
    def align_axes(self,orientation_array,alignment_array=np.array([[0,0,1],[0,1,0]]).T):
        #Normalize alignment array
        alignment_array = alignment_array/np.linalg.norm(alignment_array, axis=0, keepdims=True)
        #Align first orientation_array axis with first alignment_array axis
        orientation_array_1 = self.lattice_matrix_conventional@orientation_array
        orientation_array_1 = orientation_array_1/np.linalg.norm(orientation_array_1, axis=0, keepdims=True)
        rotation_matrix_1 = self.get_rotation_matrix_vector_to_vector(orientation_array_1[:,0],alignment_array[:,0])
        self.rotate_crystal(rotation_matrix_1)
        #Align second axis with desired plane
        orientation_array_2 = self.lattice_matrix_conventional@orientation_array
        orientation_array_2 = orientation_array_2/np.linalg.norm(orientation_array_2, axis=0, keepdims=True)
        rotation_matrix_2 = self.get_rotation_matrix_vector_to_plane(alignment_array[:,0],orientation_array_2[:,1],alignment_array[:,1])
        self.rotate_crystal(rotation_matrix_2)
     
    def rotate_crystal(self, rotation_matrix,eps=1e-15):
        """
        Rotate the entire structure (atoms + lattice) using the computed rotation matrix.
        """
        rotation_matrix[np.abs(rotation_matrix)<eps] = 0
        self._lattice_matrix = rotation_matrix@self.lattice_matrix
        self._lattice_corners = self.get_unit_corners()@self.lattice_matrix.T
        self._lattice_center = self.lattice_matrix/2
        self._lattice_atom_cartesian = self.lattice_atom_fractional@self.lattice_matrix.T
        self._lattice_matrix_conventional = rotation_matrix@self.lattice_matrix_conventional
        self._lattice_orientation = np.linalg.inv(self.lattice_matrix_conventional)/np.linalg.norm(np.linalg.inv(self.lattice_matrix_conventional), axis=0, keepdims=True)
        self._cumulative_rotation = rotation_matrix@self.cumulative_rotation
    
    def get_dhkl(self,target_plane):
        """
        Return the interplanar spacing d for the (h,k,l) plane 
        given the direct lattice vectors.
        """
        reciprocal_lattice_vectors = [np.cross(self.lattice_matrix_conventional[(i+1)%3], self.lattice_matrix_conventional[(i+2)%3]) / self.lattice_volume_conventional for i in range(3)]
        G = sum(target_plane[i] * reciprocal_lattice_vectors[i] for i in range(3))
        return 1/np.linalg.norm(G)
    
    ## Properties  
    @property
    def default_filenames(self):
        """
        Return the default output filename.
        """
        if self._default_filenames is None:
            print("self._default_filenames has not been initialized yet")
        return self._default_filenames
    
    @property
    def lattice_matrix(self):
        """
        Return the 3x3 lattice matrix (as a NumPy array).
        """
        if self._lattice_matrix is None:
            print("self._lattice_matrix has not been initialized yet")
        return self._lattice_matrix
    
    @property
    def lattice_corners(self):
        """
        Return the 8x3 lattice corner positions (as a NumPy array).
        """
        if self._lattice_corners is None:
            print("self._lattice_corners has not been initialized yet")
        return self._lattice_corners
    
    @property
    def lattice_center(self):
        """
        Return the center position of the lattice cell (in Angstroms).
        """
        if self._lattice_center is None:
            print("self._lattice_center has not been initialized yet")
        return self._lattice_center
    
    @property
    def lattice_lengths(self):
        """
        Return the lengths of the a, b, c lattice vectors (in Angstroms).
        """
        if self._lattice_lengths is None:
            print("self._lattice_lengths has not been initialized yet")
        return self._lattice_lengths
    
    @property
    def lattice_volume(self):
        """
        Return the lattice volume (in Angstroms^3).
        """
        if self._lattice_volume is None:
            print("lattice_volume has not been initialized yet")
        return self._lattice_volume
    
    @property
    def lattice_atom_fractional(self):
        """
        Return the nx3 fractional positions of the atoms.
        """
        if self._lattice_atom_fractional is None:
            print("self._lattice_atom_fractional has not been initialized yet")
        return self._lattice_atom_fractional
    
    @property
    def lattice_atom_cartesian(self):
        """
        Return the nx3 cartesian positions of the atoms. (in Angstroms).
        """
        if self._lattice_atom_cartesian is None:
            print("self._lattice_atom_cartesian has not been initialized yet")
        return self._lattice_atom_cartesian
    
    @property
    def species(self):
        """
        Return the ordered atomic species of atoms in the primitve cell.
        """
        if self._species is None:
            print("self._species has not been initialized yet")
        return self._species
    
    @property
    def lattice_matrix_conventional(self):
        """
        Return the lattice matrix of the conventional cell. (in Angstroms).
        """
        if self._lattice_matrix_conventional is None:
            print("self._lattice_matrix_conventional has not been initialized yet")
        return self._lattice_matrix_conventional
    
    @property
    def lattice_lengths_conventional(self):
        """
        Return the lengths of the a, b, c conventional lattice vectors (in Angstroms).
        """
        if self._lattice_lengths_conventional is None:
            print("self._lattice_lengths_conventional has not been initialized yet")
        return self._lattice_lengths_conventional
    
    @property
    def lattice_volume_conventional(self):
        """
        Return the conventional lattice volume (in Angstroms^3).
        """
        if self._lattice_volume_conventional is None:
            print("self._lattice_volume_conventional has not been initialized yet")
        return self._lattice_volume_conventional
    
    @property
    def lattice_orientation(self):
        """
        Return the cartesian axis directions in the crystal system w.r.t. conventional cell.
        """
        if self._lattice_orientation is None:
            print("self._lattice_orientation has not been initialized yet")
        return self._lattice_orientation
    
    @property
    def cumulative_rotation(self):
        """
        Return the cumulative rotation applied to the lattice compared to the original CIF.
        """
        if self._cumulative_rotation is None:
            print("self._cumulative_rotation has not been initialized yet")
        return self._cumulative_rotation
