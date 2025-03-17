# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import cupy as cp
import pickle
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
    
    ## Data Handling Functions
    def write_defect_metadata(self): #incomplete
        defect_metadata = [self.stacking_faults,self.cracks]

    def add_stacking_faults(self,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap):
        self._stacking_faults = self.stacking_fault(fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap)
    
    def add_cracks(self,crack_points):
        self._cracks = self.crack(crack_points)
    
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
        def __init__(self,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap):
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
        def generate_global_positions(self,sample,crystal,plotting=False):
            # Calculates the position of 
            self.rotated_fault_normal = (crystal.lattice_matrix_conventional/crystal.lattice_lengths_conventional[:,None])@self.fault_normal
            self.rotated_burgers_vector = (crystal.lattice_matrix_conventional/crystal.lattice_lengths_conventional[:,None])@self.burgers_vector
            sample_center = sample.offset
            sample_center_proj = np.dot(sample_center,self.rotated_fault_normal)
            fault_offest_proj = np.dot(self.fault_offset,self.rotated_fault_normal)
            self.global_fault_positions = sample_center_proj + fault_offest_proj - (self.fault_number - 1)*(self.interfault_spacing+self.fault_gap)/2 + np.arange(self.fault_number, dtype=np.float32)*(self.interfault_spacing+self.fault_gap)
            self._global_fault_positions_cp = cp.asarray(self.global_fault_positions)
            self._rotated_burgers_vector_cp = cp.asarray(self.rotated_burgers_vector)
            self._fault_gap_cp = cp.float32(self.fault_gap)
            self._fault_normal_cp = cp.asarray(self.rotated_fault_normal)
            self._fault_orientation_cp = cp.asarray(self.fault_orientation, dtype=cp.int8)
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

        def apply_to_sample(self,sample):
            """
            Apply stacking faults to the full sample.
            Read chunk -> apply stacking faults to chunk -> write chunk
            """
            for i in range(sample.chunk_total):
                positions_chunk_cp = sample.load_chunk_positions(i+1,gpu=True)
                if positions_chunk_cp.shape[0] > 0: # can remove this safeguard once chunking is accumulated in sample
                    positions_chunk_cp = self.apply_stacking_fault_chunk(positions_chunk_cp)
                    positions_chunk_np = cp.asnumpy(positions_chunk_cp)
                    sample.write_chunk_positions(positions_chunk_np,i+1)
        
        def apply_stacking_fault_chunk(self,positions_chunk_cp):
            """
            Apply stacking faults to a chunk by shifting atoms that lie 'beyond' each fault plane.
            A small 'fault_gap' is added each time an atom crosses a fault plane.
            """
            position_projection = cp.dot(positions_chunk_cp, self._fault_normal_cp)
            mask = cp.array(position_projection[:, None] > self._global_fault_positions_cp[None, :], dtype=cp.int8)
            count_faults = cp.sum(mask * self._fault_orientation_cp, axis=1)  # orientation-based sum
            count_faults_abs = cp.sum(mask, axis=1)  # how many planes crossed, ignoring orientation
            positions_chunk_cp = positions_chunk_cp \
                + count_faults[:, None] * self._rotated_burgers_vector_cp \
                + count_faults_abs[:, None] * self._fault_normal_cp * self._fault_gap_cp \
                - self._fault_normal_cp * self._fault_gap_cp * self.fault_number/2
            return positions_chunk_cp
        
    class crack():
        
        # -----------------------------------------------------------------------------
        # Functions
        # -----------------------------------------------------------------------------
        ## Initialization
        def __init__(self, crack_points):
            """
            Parameters
            ----------
            crack_points : (N, 3) array-like
                Coordinates defining the exterior of a convex hull in 3D.
                The hull is assumed to be convex.
            """
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
        def apply_to_sample(self, sample):
            """
            Loops over each chunk in the sample and removes all atoms lying inside the convex hull.
            """
            for i in range(sample.chunk_total):
                positions_chunk_cp = sample.load_chunk_positions(i + 1, gpu=True)
                species_chunk_np = sample.load_chunk_species(i + 1, gpu=False)
                positions_chunk_cp, species_chunk_np = self.apply_crack_chunk(positions_chunk_cp,species_chunk_np)
                sample.write_chunk_positions(cp.asnumpy(positions_chunk_cp), i + 1)
                sample.write_chunk_species(species_chunk_np,i+1)
            return

        def apply_crack_chunk(self, positions_chunk_cp,species_chunk_np):
            """
            Removes all positions inside the convex hull by checking the half-space inequalities from self.hull_equations.
            """
            if self._hull_equations_cp is None:
                self._hull_equations_cp = cp.asarray(self.hull_equations)

            eq = self._hull_equations_cp  # shape (M,4)
            # eq[:, :3] are the normal vectors, eq[:, 3] the offset
            # For each facet i, inside points satisfy (n_i . r + d_i) <= 0
            normals = eq[:, :3]            # shape (M,3)
            offsets = eq[:, 3]            # shape (M,)
            # Dot each facet's normal with each position
            # positions_chunk_cp shape: (N,3)
            # We'll do a matrix multiply: result shape => (M, N)
            dot_vals = normals @ positions_chunk_cp.T + offsets[:, None]
            # A point is inside the hull if dot_vals <= 1e-12 for *all* facets
            inside_mask = cp.all(dot_vals <= 1e-12, axis=0)
            # Remove all points that are inside the hull
            positions_chunk_cp = positions_chunk_cp[~inside_mask]
            species_chunk_np = species_chunk_np[~(inside_mask.get())]
            return positions_chunk_cp,species_chunk_np
        
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
            # Adjust edges if your sample corner order differs
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
            # hull.simplices => array of shape (F, 3) or (F, 4),
            #   listing the vertex indices for each facet.
            hull_verts = []
            for simplex in self.hull.simplices:
                hull_verts.append(self.crack_points[simplex])
            # Create a Poly3DCollection from these hull facets
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
        