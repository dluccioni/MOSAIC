# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import json
import os
from Logging import logging
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class crystal(logging):
    
    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "get_lattice_from_cif",
        "read_crystal_metadata",
        "write_crystal_metadata",
        "to_conventional",
        "to_primitive",
        "get_cartesian_from_indices",
        "align_axes",
        "rotate_crystal",
        "get_dhkl",
    )
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,filepath):
        super().__init__(log_name="crystal")
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
        self._lattice_atom_fractional_conventional = None
        self._species_conventional = None
        self._lattice_orientation = None
        self._cumulative_rotation = np.eye(3)
        self._default_filenames = np.array(["crystal_metadata.npy"]) #sample_metadata will be a struct
    
    def get_lattice_from_cif(self):
        """
        Load crystal structure from the CIF file and initialize all lattice properties.

        Reads the CIF file at self.filepath, uses SpacegroupAnalyzer to get proper
        primitive and conventional cell representations with correct atom positions.
        """
        self.structure = Structure.from_file(self.filepath)

        # Use SpacegroupAnalyzer for proper symmetry-aware cell transformations
        sga = SpacegroupAnalyzer(self.structure)

        # Get primitive cell
        prim_structure = sga.get_primitive_standard_structure()
        self._lattice_matrix = prim_structure.lattice.matrix
        self._lattice_corners = self.get_unit_corners() @ self._lattice_matrix
        self._lattice_center = self._lattice_matrix / 2
        self._lattice_lengths = np.array(prim_structure.lattice.abc)
        self._lattice_volume = prim_structure.lattice.volume
        self._lattice_atom_fractional = prim_structure.frac_coords
        self._lattice_atom_cartesian = prim_structure.cart_coords
        self._species = np.array([site.specie.name for site in prim_structure.sites])

        # Get conventional cell
        conv_structure = sga.get_conventional_standard_structure()
        self._lattice_matrix_conventional = conv_structure.lattice.matrix
        self._lattice_lengths_conventional = np.array(conv_structure.lattice.abc)
        self._lattice_volume_conventional = conv_structure.lattice.volume
        self._lattice_atom_fractional_conventional = conv_structure.frac_coords
        self._species_conventional = np.array([site.specie.name for site in conv_structure.sites])
        self._lattice_orientation = np.linalg.inv(self._lattice_matrix_conventional) / np.linalg.norm(
            np.linalg.inv(self._lattice_matrix_conventional), axis=0, keepdims=True
        )

        del self.structure
        
    def read_crystal_metadata(self, override_directory=None):
        """
        Read crystal_metadata.json from disk and restore internal fields.

        Args:
            override_directory (str or None, optional): Directory to read from.
                If None, uses the directory of self.filepath. Defaults to None.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
        """
        if override_directory is not None:
            base_dir = override_directory
        else:
            base_dir = os.path.dirname(self.filepath)
        
        metadata_filename = os.path.join(base_dir, "crystal_metadata.json")
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")

        with open(metadata_filename, "r") as f:
            data = json.load(f)

        # Files written before the lattice matrices were made row-consistent
        # carry no convention key, and a rotated crystal in one of them holds
        # its lattice vectors as columns instead
        if data.get("lattice_convention") != "rows":
            print(f"{metadata_filename} predates the row lattice convention; "
                  "a rotated crystal in it will be read transposed")

        self.filepath = data["filepath"]
        
        if data["lattice_matrix"] is not None:
            self._lattice_matrix = np.array(data["lattice_matrix"], dtype=np.float64)
        if data["lattice_corners"] is not None:
            self._lattice_corners = np.array(data["lattice_corners"], dtype=np.float64)
        if data["lattice_center"] is not None:
            self._lattice_center = np.array(data["lattice_center"], dtype=np.float64)
        if data["lattice_lengths"] is not None:
            self._lattice_lengths = np.array(data["lattice_lengths"], dtype=np.float64)
        if data["lattice_volume"] is not None:
            self._lattice_volume = float(data["lattice_volume"])
        if data["lattice_atom_cartesian"] is not None:
            self._lattice_atom_cartesian = np.array(data["lattice_atom_cartesian"], dtype=np.float64)
        if data["species"] is not None:
            self._species = np.array(data["species"], dtype=object)
        if data["lattice_atom_fractional"] is not None:
            self._lattice_atom_fractional = np.array(data["lattice_atom_fractional"], dtype=np.float64)
        if data["lattice_matrix_conventional"] is not None:
            self._lattice_matrix_conventional = np.array(data["lattice_matrix_conventional"], dtype=np.float64)
        if data["lattice_lengths_conventional"] is not None:
            self._lattice_lengths_conventional = np.array(data["lattice_lengths_conventional"], dtype=np.float64)
        if data["lattice_volume_conventional"] is not None:
            self._lattice_volume_conventional = float(data["lattice_volume_conventional"])
        if data.get("lattice_atom_fractional_conventional") is not None:
            self._lattice_atom_fractional_conventional = np.array(data["lattice_atom_fractional_conventional"], dtype=np.float64)
        if data.get("species_conventional") is not None:
            self._species_conventional = np.array(data["species_conventional"], dtype=object)
        if data["lattice_orientation"] is not None:
            self._lattice_orientation = np.array(data["lattice_orientation"], dtype=np.float64)
        if data["cumulative_rotation"] is not None:
            self._cumulative_rotation = np.array(data["cumulative_rotation"], dtype=np.float64)
        
    ## Data Saving Functions
    def write_crystal_metadata(self, override_directory=None):
        """
        Serialize crystal internal fields to disk as human-readable JSON.

        Writes all critical crystal state to crystal_metadata.json so that
        it can be restored later via read_crystal_metadata.

        Args:
            override_directory (str or None, optional): Directory to write to.
                If None, uses the directory of self.filepath. Defaults to None.
        """
        if override_directory is not None:
            base_dir = override_directory
        else:
            base_dir = os.path.dirname(self.filepath)
            
        metadata_filename = os.path.join(base_dir, "crystal_metadata.json")

        crystal_metadata = {
            "lattice_convention":        "rows",
            "filepath":                  self.filepath,
            "lattice_matrix":            self._lattice_matrix.tolist() if self._lattice_matrix is not None else None,
            "lattice_corners":           self._lattice_corners.tolist() if self._lattice_corners is not None else None,
            "lattice_center":            self._lattice_center.tolist() if self._lattice_center is not None else None,
            "lattice_lengths":           self._lattice_lengths.tolist() if self._lattice_lengths is not None else None,
            "lattice_volume":            float(self._lattice_volume) if self._lattice_volume is not None else None,
            "lattice_atom_cartesian":    self._lattice_atom_cartesian.tolist() if self._lattice_atom_cartesian is not None else None,
            "species":                   self._species.tolist() if self._species is not None else None,
            "lattice_atom_fractional":   self._lattice_atom_fractional.tolist() if self._lattice_atom_fractional is not None else None,
            "lattice_matrix_conventional":
                self._lattice_matrix_conventional.tolist() if self._lattice_matrix_conventional is not None else None,
            "lattice_lengths_conventional":
                self._lattice_lengths_conventional.tolist() if self._lattice_lengths_conventional is not None else None,
            "lattice_volume_conventional":
                float(self._lattice_volume_conventional) if self._lattice_volume_conventional is not None else None,
            "lattice_atom_fractional_conventional":
                self._lattice_atom_fractional_conventional.tolist() if self._lattice_atom_fractional_conventional is not None else None,
            "species_conventional":
                self._species_conventional.tolist() if self._species_conventional is not None else None,
            "lattice_orientation":
                self._lattice_orientation.tolist() if self._lattice_orientation is not None else None,
            "cumulative_rotation":
                self._cumulative_rotation.tolist() if self._cumulative_rotation is not None else None
        }

        with open(metadata_filename, "w") as f:
            json.dump(crystal_metadata, f, indent=4)
        print(f"Metadata written to {metadata_filename}")
    
    ## Static Functions
    @staticmethod
    def get_unit_corners():
        """
        Return the 8 corner positions of a unit cube in fractional coordinates.

        Returns:
            np.ndarray: Array of shape (8, 3) with corner positions ranging
                from [0,0,0] to [1,1,1].
        """
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
        Compute a 3x3 rotation matrix using the Rodrigues formula.

        Args:
            axis (array_like): Rotation axis vector (will be normalized internally).
            angle (float): Rotation angle in radians.

        Returns:
            np.ndarray: 3x3 rotation matrix for rotation by angle around axis.
        """
        axis = axis / np.linalg.norm(axis)
        c = np.cos(angle)
        s = np.sin(angle)
        d = 1.0 - c
        x, y, z = axis
        return np.array([[c + d*x*x,     d*x*y - z*s,   d*x*z + y*s],
                         [d*y*x + z*s,   c + d*y*y,     d*y*z - x*s],
                         [d*z*x - y*s,   d*z*y + x*s,   c + d*z*z]])
    
    def get_rotation_matrix_vector_to_vector(self, vector1, vector2, eps=1e-8):
        """
        Compute rotation matrix that rotates vector1 to align with vector2.

        Args:
            vector1 (array_like): Source unit vector (3,).
            vector2 (array_like): Target unit vector (3,).
            eps (float, optional): Tolerance for detecting parallel/antiparallel
                vectors. Defaults to 1e-8.

        Returns:
            np.ndarray: 3x3 rotation matrix R such that R @ vector1 aligns with vector2.
        """
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
    
    def get_rotation_matrix_vector_to_plane(self, rotation_axis, vector1, plane1, angle_selection='small', eps=1e-12):
        """
        Compute rotation matrix to bring vector1 into a plane by rotating about an axis.

        Finds the rotation about rotation_axis that makes vector1 orthogonal to
        plane1's normal (i.e., brings vector1 into the plane).

        Args:
            rotation_axis (array_like): Axis to rotate around (3,).
            vector1 (array_like): Vector to rotate into the plane (3,).
            plane1 (array_like): Normal vector of the target plane (3,).
            angle_selection (str, optional): 'small' to pick the smaller angle,
                'large' to pick the larger angle. Defaults to 'small'.
            eps (float, optional): Tolerance for degenerate cases. Defaults to 1e-12.

        Returns:
            np.ndarray: 3x3 rotation matrix.

        Raises:
            ValueError: If no solution exists for the given geometry.
        """
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
        """
        Convert the internal pymatgen Structure to its conventional cell.
        """
        self.structure = self.structure.to_conventional()

    def to_primitive(self):
        """
        Convert the internal pymatgen Structure to its primitive cell.
        """
        self.structure = self.structure.to_primitive()

    def get_cartesian_from_indices(self, indices, index_type="plane"):
        """
        Convert Miller indices to Cartesian vectors in the current orientation.

        A set of indices names two different vectors, which coincide only for
        cubic and other orthogonal cells:
        - as a plane (h, k, l), the vector is the plane normal, h*a* + k*b* +
          l*c*, built from the reciprocal lattice;
        - as a direction [u, v, w], the vector is u*a + v*b + w*c.
        For alpha-quartz the two are 30 degrees apart for (100) and 23 degrees
        for (111), so which one is meant has to be stated rather than assumed.

        Args:
            indices (array_like): Shape (3,) or (3, N) array of Miller indices,
                as columns if two-dimensional.
            index_type (str, optional): 'plane' (default) treats the indices as
                (h, k, l) and returns plane normals; 'direction' treats them as
                [u, v, w] and returns real-space directions.

        Returns:
            np.ndarray: Cartesian vectors in the same shape as ``indices``.

        Raises:
            ValueError: If index_type is neither 'plane' nor 'direction'.
        """
        indices = np.asarray(indices, dtype=np.float64)
        if index_type == "plane":
            # Columns of the inverse are the reciprocal vectors a*, b*, c*
            return np.linalg.inv(self.lattice_matrix_conventional)@indices
        elif index_type == "direction":
            return self.lattice_matrix_conventional.T@indices
        else:
            raise ValueError(f"index_type must be 'plane' or 'direction', got '{index_type}'")

    def align_axes(self, orientation_array, alignment_array=np.array([[0, 0, 1], [0, 1, 0]]).T, index_type="plane"):
        """
        Align crystal axes to specified laboratory directions.

        Performs a two-step rotation: first aligns the primary crystal vector
        with the primary alignment direction, then rotates around that axis to
        bring the secondary vector into the specified plane.

        Args:
            orientation_array (np.ndarray): Shape (3, 2) array of Miller indices.
                Column 0 is the primary, column 1 is the secondary. They are
                converted to Cartesian vectors by `get_cartesian_from_indices`
                according to index_type.
            alignment_array (np.ndarray, optional): Shape (3, 2) array of target
                lab directions. Column 0 is the axis the primary is brought onto;
                column 1 is the NORMAL of the plane the secondary is brought into.
                Defaults to [[0,0,1],[0,1,0]].T, i.e. the primary onto +z and the
                secondary into the plane normal to +y.
            index_type (str, optional): 'plane' (default) reads the indices as
                (h, k, l) plane normals, which is what a reflection means;
                'direction' reads them as [u, v, w] real-space directions. The
                two are identical for cubic cells and differ otherwise.
        """
        # Normalize alignment array
        alignment_array = alignment_array / np.linalg.norm(alignment_array, axis=0, keepdims=True)
        #Align first orientation_array axis with first alignment_array axis
        orientation_array_1 = self.get_cartesian_from_indices(orientation_array, index_type)
        orientation_array_1 = orientation_array_1/np.linalg.norm(orientation_array_1, axis=0, keepdims=True)
        rotation_matrix_1 = self.get_rotation_matrix_vector_to_vector(orientation_array_1[:,0],alignment_array[:,0])
        self.rotate_crystal(rotation_matrix_1)
        #Align second axis with desired plane
        orientation_array_2 = self.get_cartesian_from_indices(orientation_array, index_type)
        orientation_array_2 = orientation_array_2/np.linalg.norm(orientation_array_2, axis=0, keepdims=True)
        rotation_matrix_2 = self.get_rotation_matrix_vector_to_plane(alignment_array[:,0],orientation_array_2[:,1],alignment_array[:,1])
        self.rotate_crystal(rotation_matrix_2)
     
    def rotate_crystal(self, rotation_matrix, eps=1e-15):
        """
        Rotate the entire crystal structure using a rotation matrix.

        Applies the rotation to all lattice vectors, atom positions, and updates
        the cumulative rotation tracker. Small values below eps are zeroed.

        The lattice matrices hold the Cartesian vectors a, b, c as their rows
        (pymatgen's convention, as loaded by get_lattice_from_cif), so a rotation
        that maps a column vector v to rotation_matrix@v acts on the matrix as
        M -> M@rotation_matrix.T.

        Args:
            rotation_matrix (np.ndarray): 3x3 rotation matrix to apply. The
                caller's array is not modified.
            eps (float, optional): Threshold below which matrix elements are
                set to zero. Defaults to 1e-15.
        """
        rotation_matrix = np.array(rotation_matrix, dtype=np.float64)
        rotation_matrix[np.abs(rotation_matrix)<eps] = 0
        self._lattice_matrix = self.lattice_matrix@rotation_matrix.T
        self._lattice_corners = self.get_unit_corners()@self.lattice_matrix
        self._lattice_center = self.lattice_matrix/2
        self._lattice_atom_cartesian = self.lattice_atom_fractional@self.lattice_matrix
        self._lattice_matrix_conventional = self.lattice_matrix_conventional@rotation_matrix.T
        self._lattice_orientation = np.linalg.inv(self.lattice_matrix_conventional)/np.linalg.norm(np.linalg.inv(self.lattice_matrix_conventional), axis=0, keepdims=True)
        self._cumulative_rotation = rotation_matrix@self.cumulative_rotation
    
    def get_dhkl(self, target_plane):
        """
        Calculate the interplanar spacing for a given Miller plane.

        Computes d_hkl using the reciprocal lattice vectors derived from
        the conventional cell.

        Args:
            target_plane (array_like): Miller indices (h, k, l) of the plane.

        Returns:
            float: Interplanar spacing d_hkl in Angstroms.
        """
        reciprocal_lattice_vectors = [np.cross(self.lattice_matrix_conventional[(i+1)%3], self.lattice_matrix_conventional[(i+2)%3]) / self.lattice_volume_conventional for i in range(3)]
        G = sum(target_plane[i] * reciprocal_lattice_vectors[i] for i in range(3))
        return 1/np.linalg.norm(G)
    
    ## Properties
    @property
    def default_filenames(self):
        """
        Return the default output filenames.

        Returns:
            np.ndarray: Array of default filename strings for metadata.
        """
        if self._default_filenames is None:
            print("self._default_filenames has not been initialized yet")
        return self._default_filenames

    @property
    def lattice_matrix(self):
        """
        Return the primitive cell lattice matrix.

        Returns:
            np.ndarray: 3x3 lattice matrix with row vectors a, b, c in Angstroms.
        """
        if self._lattice_matrix is None:
            print("self._lattice_matrix has not been initialized yet")
        return self._lattice_matrix

    @property
    def lattice_corners(self):
        """
        Return the corner positions of the primitive cell.

        Returns:
            np.ndarray: 8x3 array of corner positions in Angstroms.
        """
        if self._lattice_corners is None:
            print("self._lattice_corners has not been initialized yet")
        return self._lattice_corners

    @property
    def lattice_center(self):
        """
        Return the center position of the lattice cell.

        Returns:
            np.ndarray: 3x3 array representing half the lattice matrix in Angstroms.
        """
        if self._lattice_center is None:
            print("self._lattice_center has not been initialized yet")
        return self._lattice_center

    @property
    def lattice_lengths(self):
        """
        Return the primitive cell lattice vector lengths.

        Returns:
            np.ndarray: Array of (a, b, c) lengths in Angstroms.
        """
        if self._lattice_lengths is None:
            print("self._lattice_lengths has not been initialized yet")
        return self._lattice_lengths

    @property
    def lattice_volume(self):
        """
        Return the primitive cell volume.

        Returns:
            float: Lattice volume in Angstroms^3.
        """
        if self._lattice_volume is None:
            print("lattice_volume has not been initialized yet")
        return self._lattice_volume

    @property
    def lattice_atom_fractional(self):
        """
        Return the fractional coordinates of atoms in the primitive cell.

        Returns:
            np.ndarray: Shape (n_atoms, 3) array of fractional coordinates.
        """
        if self._lattice_atom_fractional is None:
            print("self._lattice_atom_fractional has not been initialized yet")
        return self._lattice_atom_fractional

    @property
    def lattice_atom_cartesian(self):
        """
        Return the Cartesian coordinates of atoms in the primitive cell.

        Returns:
            np.ndarray: Shape (n_atoms, 3) array of positions in Angstroms.
        """
        if self._lattice_atom_cartesian is None:
            print("self._lattice_atom_cartesian has not been initialized yet")
        return self._lattice_atom_cartesian

    @property
    def species(self):
        """
        Return the atomic species of atoms in the primitive cell.

        Returns:
            np.ndarray: Array of element symbol strings in site order.
        """
        if self._species is None:
            print("self._species has not been initialized yet")
        return self._species

    @property
    def lattice_matrix_conventional(self):
        """
        Return the conventional cell lattice matrix.

        Returns:
            np.ndarray: 3x3 lattice matrix with row vectors in Angstroms.
        """
        if self._lattice_matrix_conventional is None:
            print("self._lattice_matrix_conventional has not been initialized yet")
        return self._lattice_matrix_conventional

    @property
    def lattice_lengths_conventional(self):
        """
        Return the conventional cell lattice vector lengths.

        Returns:
            np.ndarray: Array of (a, b, c) lengths in Angstroms.
        """
        if self._lattice_lengths_conventional is None:
            print("self._lattice_lengths_conventional has not been initialized yet")
        return self._lattice_lengths_conventional

    @property
    def lattice_volume_conventional(self):
        """
        Return the conventional cell volume.

        Returns:
            float: Lattice volume in Angstroms^3.
        """
        if self._lattice_volume_conventional is None:
            print("self._lattice_volume_conventional has not been initialized yet")
        return self._lattice_volume_conventional

    @property
    def lattice_orientation(self):
        """
        Return the unit reciprocal-lattice vectors in Cartesian coordinates.

        The orientation is computed from the inverse of the conventional
        lattice matrix, normalized column-wise, so its columns are the unit
        normals of the (100), (010) and (001) planes. These coincide with the
        crystal axis directions only for cubic and other orthogonal cells.

        Returns:
            np.ndarray: 3x3 array whose columns are unit plane normals.
        """
        if self._lattice_orientation is None:
            print("self._lattice_orientation has not been initialized yet")
        return self._lattice_orientation

    @property
    def cumulative_rotation(self):
        """
        Return the cumulative rotation matrix applied to the crystal.

        Tracks all rotations applied since loading from the CIF file.

        Returns:
            np.ndarray: 3x3 rotation matrix (identity if no rotations applied).
        """
        if self._cumulative_rotation is None:
            print("self._cumulative_rotation has not been initialized yet")
        return self._cumulative_rotation
