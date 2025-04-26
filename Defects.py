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
class defects:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=None):
        self.directory = directory
        if self.directory is not None and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self._default_filenames = np.array(["defects_metadata.npy"])
        self._defect_history = []
        self._stacking_faults = None
        self._cracks = None
        
    def read_defect_metadata(self, override_directory=None):
        """
        Reads the defect metadata JSON file from disk and restores
        this defect object's state, including stacking faults and cracks
        if present.
        """
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "defects_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "defects_metadata.json")
        
        if not os.path.isfile(metadata_filename):
            raise FileNotFoundError(f"No JSON metadata file found at {metadata_filename}")
        
        with open(metadata_filename, "r") as f:
            defect_metadata = json.load(f)
        
        # Restore defect history
        self._defect_history = defect_metadata.get("defect_history", [])
        
        # Restore stacking_faults if present
        sf_data = defect_metadata.get("stacking_faults", None)
        if sf_data is not None:
            self._stacking_faults = self.stacking_fault(
                directory      = sf_data.get("directory", None),
                fault_number   = sf_data.get("fault_number", 0),
                fault_offset   = np.array(sf_data.get("fault_offset", [0,0,0]), dtype=np.float32),
                fault_normal   = np.array(sf_data.get("fault_normal", [0,0,1]), dtype=np.float32),
                interfault_spacing = sf_data.get("interfault_spacing", 0.0),
                burgers_vector = np.array(sf_data.get("burgers_vector", [0,0,0]), dtype=np.float32),
                fault_orientation = sf_data.get("fault_orientation", [0]),
                fault_gap      = sf_data.get("fault_gap", 0.0)
            )
        
        # Restore cracks if present
        crack_data = defect_metadata.get("cracks", None)
        if crack_data is not None:
            self._cracks = self.crack(
                directory    = crack_data.get("directory", None),
                crack_points = crack_data.get("crack_points", [])
            )
        print(f"Defect metadata read from {metadata_filename}.")
        
    ## Data Handling Functions
    def write_defect_metadata(self, override_directory=None):
        """
        Serializes the defect object's critical internal fields to disk
        as human-readable JSON so that the state can be restored later.
        """
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "defects_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "defects_metadata.json")

        # Prepare top-level defect data
        defect_metadata = {
            "defect_history": self._defect_history if self._defect_history else []
        }
        
        # If stacking_faults exist, store them
        if self._stacking_faults is not None:
            sf = self._stacking_faults
            defect_metadata["stacking_faults"] = {
                "directory": sf.directory,
                "fault_number": sf.fault_number,
                "fault_offset": sf.fault_offset.tolist(),
                "fault_normal": sf.fault_normal.tolist(),
                "interfault_spacing": sf.interfault_spacing,
                "burgers_vector": sf.burgers_vector.tolist(),
                "fault_orientation": sf.fault_orientation.tolist(),
                "fault_gap": sf.fault_gap
            }
        else:
            defect_metadata["stacking_faults"] = None

        # If cracks exist, store them
        if self._cracks is not None:
            cr = self._cracks
            # crack_points is an Nx3 array. Make sure to convert to list of lists
            crack_points_list = cr.crack_points.tolist() if len(cr.crack_points) > 0 else []
            defect_metadata["cracks"] = {
                "directory": cr.directory,
                "crack_points": crack_points_list
            }
        else:
            defect_metadata["cracks"] = None

        # Write as nicely formatted JSON
        with open(metadata_filename, "w") as f:
            json.dump(defect_metadata, f, indent=4)
        print(f"Defect metadata written to {metadata_filename} in JSON format.")

    def add_stacking_faults(self,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap):
        self._stacking_faults = self.stacking_fault(self.directory,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap)
    
    def add_cracks(self,crack_points):
        self._cracks = self.crack(self.directory,crack_points)
    
    ## Properties
    @property
    def stacking_faults(self):
        if self._stacking_faults is None:
            print("self._stacking_faults has not been initialized yet")
        return self._stacking_faults
    
    @property
    def cracks(self):
        if self._cracks is None:
            print("self._cracks has not been initialized yet")
        return self._cracks

    # -------------------------------------------------------------------------
    # Sub-Classes
    # -------------------------------------------------------------------------
    class stacking_fault():
        
        # -----------------------------------------------------------------------------
        # Functions
        # -----------------------------------------------------------------------------
        ## Initialization
        def __init__(self,directory,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap):
            self.directory = directory
            self.fault_number = fault_number
            self.fault_offset = fault_offset
            self.fault_normal = fault_normal/np.linalg.norm(fault_normal)
            self.interfault_spacing = interfault_spacing
            self.burgers_vector = burgers_vector
            self.fault_orientation = np.array([fault_orientation[i % len(fault_orientation)] for i in range(self.fault_number)])
            self.fault_gap = fault_gap
            self.global_fault_positions = None
            self.rotated_fault_normal = None
            self.rotated_burgers_vector = None
            self._global_fault_positions_cp = None
            self._rotated_burgers_vector_cp = None
            self._fault_gap_cp = None
            self._fault_normal_cp = None
            self._fault_orientation_cp = None
        
        ## Main Functions    
        def generate_global_positions(self,sample,crystal,plotting=False,use_gpu=False):
            # Calculates the position of stacking faults
            self.rotated_fault_normal = (crystal.lattice_matrix_conventional/crystal.lattice_lengths_conventional[:,None])@self.fault_normal
            self.rotated_burgers_vector = (crystal.lattice_matrix_conventional/crystal.lattice_lengths_conventional[:,None])@self.burgers_vector
            sample_center = sample.offset
            sample_center_proj = np.dot(sample_center,self.rotated_fault_normal)
            fault_offest_proj = np.dot(self.fault_offset,self.rotated_fault_normal)
            self.global_fault_positions = sample_center_proj + fault_offest_proj - (self.fault_number - 1)*(self.interfault_spacing+self.fault_gap)/2 + np.arange(self.fault_number, dtype=np.float32)*(self.interfault_spacing+self.fault_gap)

            # Set up GPU arrays if requested and available
            if cp is not None and use_gpu:
                self._global_fault_positions_cp = cp.asarray(self.global_fault_positions)
                self._rotated_burgers_vector_cp = cp.asarray(self.rotated_burgers_vector)
                self._fault_gap_cp = cp.float32(self.fault_gap)
                self._fault_normal_cp = cp.asarray(self.rotated_fault_normal)
                self._fault_orientation_cp = cp.asarray(self.fault_orientation, dtype=cp.int8)
            else:
                self._global_fault_positions_cp = None
                self._rotated_burgers_vector_cp = None
                self._fault_gap_cp = None
                self._fault_normal_cp = None
                self._fault_orientation_cp = None

            if plotting:
                self.plot_global_positions(sample)
            
        def plot_global_positions(self,sample,color='c',alpha=0.5,elev=0, azim=0):
            """
            Plot one or more planes (all sharing the same normal vector) intersecting
            a cuboid (defined by its 8 corners). self.global_fault_positions are scalars
            giving distance along self.rotated_fault_normal, so the plane equation is
                n . r = distance.
            """
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

            sample_corners = sample.corners
            
            edges = [
                (0,1), (0,2), (0,3),
                (1,4), (1,5),
                (2,4), (2,6),
                (3,5), (3,6),
                (4,7), (5,7), (6,7)
            ]
            segs = [(sample_corners[i], sample_corners[j]) for i, j in edges]

            # Prepare figure
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.add_collection3d(Line3DCollection(segs, colors='gray', lw=1))
            # Normalize the plane normal once
            n = np.array(self.rotated_fault_normal, float)
            n /= np.linalg.norm(n)

            # self.global_fault_positions is a 1D array of scalar offsets along n
            distances = np.atleast_1d(self.global_fault_positions)

            def intersect(p1, p2, dist):
                """
                Intersect the line segment [p1, p2] with the plane n·r = dist.
                p1, p2 are 3D, dist is a scalar.
                """
                d1 = n.dot(p1)
                d2 = n.dot(p2)
                denom = d2 - d1
                if abs(denom) < 1e-12:
                    return None
                t = (dist - d1) / denom
                return p1 + t*(p2 - p1) if (0 <= t <= 1) else None

            # Build a local 2D basis (u,v) in the plane for sorting intersection polygons
            v0 = np.array([1, 0, 0], dtype=float)
            if np.allclose(np.cross(n, v0), 0, atol=1e-12):
                v0 = np.array([0, 1, 0], dtype=float)
            u = np.cross(n, v0); u /= np.linalg.norm(u)
            v = np.cross(n, u)

            # For each plane distance, compute intersection polygon and plot
            for dist in distances:
                pts = []
                for i, j in edges:
                    ip = intersect(sample_corners[i], sample_corners[j], dist)
                    if ip is not None:
                        pts.append(ip)
                if len(pts) < 3:
                    continue  # Not enough points to form a polygon

                # Remove duplicates
                unique_pts = []
                for pt in pts:
                    if not any(np.allclose(pt, q, atol=1e-9) for q in unique_pts):
                        unique_pts.append(pt)
                pts = np.array(unique_pts)
                if len(pts) < 3:
                    continue

                # Sort intersection points around centroid (in 2D coordinates)
                plane_point_3d = dist * n
                pts_2d = [(np.dot(pt - plane_point_3d, u), np.dot(pt - plane_point_3d, v)) for pt in pts]
                ctr = np.mean(pts_2d, axis=0)
                angles = [np.arctan2(py - ctr[1], px - ctr[0]) for px, py in pts_2d]
                polygon = pts[np.argsort(angles)]

                ax.add_collection3d(Poly3DCollection([polygon], facecolors=color, edgecolors='k', alpha=alpha))

            # Set plot limits
            x, y, z = sample_corners[:, 0], sample_corners[:, 1], sample_corners[:, 2]
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_zlim(z.min(), z.max())
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("Stacking Fault Planes in Sample")
            plt.show()
            return fig, ax

        def apply_to_sample(self,sample,use_gpu=True):
            """
            Apply stacking faults to the full sample.
            Read chunk -> apply stacking faults to chunk -> write chunk
            """
            for i in range(sample.chunk_total):
                if cp is not None and use_gpu:
                    positions_chunk_cp = sample.load_chunk_positions(i+1,use_gpu=True)
                    positions_chunk_cp = self.apply_stacking_fault_chunk(positions_chunk_cp,use_gpu=True)
                    positions_chunk_np = cp.asnumpy(positions_chunk_cp)
                else:
                    positions_chunk_np = sample.load_chunk_positions(i+1,use_gpu=False)
                    positions_chunk_np = self.apply_stacking_fault_chunk(positions_chunk_np,use_gpu=False)

                sample.write_chunk_positions(positions_chunk_np,i+1,override_directory=self.directory)
                if self.directory is not None:
                    species_chunk_np = sample.load_chunk_species(i + 1, use_gpu=False)
                    sample.write_chunk_species(species_chunk_np,i+1,override_directory=self.directory)
            sample.write_sample_metadata(override_directory=self.directory)
            
        def apply_stacking_fault_chunk(self,positions_chunk,use_gpu=True):
            """
            Apply stacking faults to a chunk by shifting atoms that lie 'beyond' each fault plane.
            A small 'fault_gap' is added each time an atom crosses a fault plane.
            """
            if cp is not None and use_gpu and self._global_fault_positions_cp is not None:
                position_projection = cp.dot(positions_chunk, self._fault_normal_cp)
                mask = cp.array(position_projection[:, None] > self._global_fault_positions_cp[None, :], dtype=cp.int8)
                count_faults = cp.sum(mask * self._fault_orientation_cp, axis=1)  # orientation-based sum
                count_faults_abs = cp.sum(mask, axis=1)  # how many planes crossed, ignoring orientation
                positions_chunk = positions_chunk \
                    + count_faults[:, None] * self._rotated_burgers_vector_cp \
                    + count_faults_abs[:, None] * self._fault_normal_cp * self._fault_gap_cp \
                    - self._fault_normal_cp * self._fault_gap_cp * self.fault_number/2
                return positions_chunk
            else:
                position_projection = np.dot(positions_chunk, self.rotated_fault_normal)
                mask = (position_projection[:, None] > self.global_fault_positions[None, :]).astype(np.int8)
                count_faults = np.sum(mask * self.fault_orientation, axis=1)
                count_faults_abs = np.sum(mask, axis=1)
                positions_chunk = positions_chunk \
                    + count_faults[:, None] * self.rotated_burgers_vector \
                    + count_faults_abs[:, None] * self.rotated_fault_normal * self.fault_gap \
                    - self.rotated_fault_normal * self.fault_gap * self.fault_number/2
                return positions_chunk
        
    class crack():
        
        # -----------------------------------------------------------------------------
        # Functions
        # -----------------------------------------------------------------------------
        ## Initialization
        def __init__(self, directory, crack_points):
            """
            Parameters
            ----------
            crack_points : (N, 3) array-like
                Coordinates defining the exterior of a convex hull in 3D.
                The hull is assumed to be convex.
            """
            self.directory = directory
            from scipy.spatial import ConvexHull
            # Store input points
            self.crack_points = np.asarray(crack_points, dtype=float)
            # Build convex hull
            self.hull = ConvexHull(self.crack_points)
            # Keep a copy of the plane equations: shape (M, 4)
            # Each row [a, b, c, d] => a*x + b*y + c*z + d <= 0 for points inside
            self.hull_equations = self.hull.equations
            self._hull_equations_cp = None

        ## Main Functions
        def apply_to_sample(self, sample, use_gpu=True):
            """
            Loops over each chunk in the sample and removes all atoms lying inside the convex hull.
            """
            for i in range(sample.chunk_total):
                if cp is not None and use_gpu:
                    positions_chunk_cp = sample.load_chunk_positions(i + 1, use_gpu=True)
                    species_chunk_np = sample.load_chunk_species(i + 1, use_gpu=False)
                    positions_chunk_cp, species_chunk_np = self.apply_crack_chunk(positions_chunk_cp,species_chunk_np,use_gpu=True)
                    positions_chunk_np = cp.asnumpy(positions_chunk_cp)
                else:
                    positions_chunk_np = sample.load_chunk_positions(i + 1, use_gpu=False)
                    species_chunk_np = sample.load_chunk_species(i + 1, use_gpu=False)
                    positions_chunk_np, species_chunk_np = self.apply_crack_chunk(positions_chunk_np,species_chunk_np,use_gpu=False)

                sample.write_chunk_positions(positions_chunk_np,i+1,override_directory=self.directory)
                sample.write_chunk_species(species_chunk_np,i+1,override_directory=self.directory)
            sample.write_sample_metadata(override_directory=self.directory)

        def apply_crack_chunk(self, positions_chunk, species_chunk_np, use_gpu=False):
            """
            Removes all positions inside the convex hull by checking the half-space inequalities from self.hull_equations.
            """
            if cp is not None and use_gpu:
                if self._hull_equations_cp is None:
                    self._hull_equations_cp = cp.asarray(self.hull_equations)
                eq = self._hull_equations_cp
                normals = eq[:, :3]
                offsets = eq[:, 3]
                dot_vals = normals @ positions_chunk.T + offsets[:, None]
                inside_mask = cp.all(dot_vals <= 1e-12, axis=0)
                positions_chunk = positions_chunk[~inside_mask]
                species_chunk_np = species_chunk_np[~(inside_mask.get())]
                return positions_chunk, species_chunk_np
            else:
                eq = self.hull_equations
                normals = eq[:, :3]
                offsets = eq[:, 3]
                dot_vals = normals @ positions_chunk.T + offsets[:, None]
                inside_mask = np.all(dot_vals <= 1e-12, axis=0)
                positions_chunk = positions_chunk[~inside_mask]
                species_chunk_np = species_chunk_np[~inside_mask]
                return positions_chunk, species_chunk_np
        
        def plot_crack_geometry(self, sample, color='r', alpha=0.5, elev=0, azim=0):
            """
            Plot the sample as a wireframe, along with the triangular facets
            of the crack's convex hull.
            """
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            # 1) Plot sample wireframe -------------------------------------
            corners = sample.corners  # shape (8, 3)
            edges = [
                (0,1), (0,2), (0,3),
                (1,4), (1,5),
                (2,4), (2,6),
                (3,5), (3,6),
                (4,7), (5,7), (6,7)
            ]
            segs = [(corners[i], corners[j]) for i, j in edges]
            ax.add_collection3d(Line3DCollection(segs, colors='gray', lw=1))
            # 2) Plot the crack convex hull facets --------------------------
            hull_verts = []
            for simplex in self.hull.simplices:
                hull_verts.append(self.crack_points[simplex])
            poly = Poly3DCollection(hull_verts, facecolors=color, edgecolors='k', alpha=alpha)
            ax.add_collection3d(poly)
            # 3) Set axes limits -------------------------------------------
            x, y, z = corners[:, 0], corners[:, 1], corners[:, 2]
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_zlim(z.min(), z.max())
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("Crack Geometry in Sample")
            plt.show()
            return fig, ax
