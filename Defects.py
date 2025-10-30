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
from Logging import logging

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class defects(logging):
    
    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "read_defect_metadata",
        "write_defect_metadata",
        "add_stacking_faults",
        "add_cracks",
        "add_point_defects",
    )
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=None):
        super().__init__(log_name="defects")
        self.directory = directory
        if self.directory is not None and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self._default_filenames = np.array(["defects_metadata.npy"])
        self._defect_history = []
        self._stacking_faults = None
        self._cracks = None
        self._point_defects = None
        
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
            
        # Restore point_defects if present
        pd_data = defect_metadata.get("point_defects", None)
        if pd_data is not None:
            spec = pd_data.get("spec", {})
            self._point_defects = self.point_defect(
                directory=pd_data.get("directory", None),
                seed=pd_data.get("seed", None),
                region_min=pd_data.get("region_min", None),
                region_max=pd_data.get("region_max", None),

                vacancy_fraction=spec.get("vacancy_fraction", None),
                vacancy_count=spec.get("vacancy_count", None),
                vacancy_global_indices=spec.get("vacancy_global_indices", None),
                vacancy_positions=spec.get("vacancy_positions", None),
                vacancy_species_filter=spec.get("vacancy_species_filter", None),

                substitution_fraction=spec.get("substitution_fraction", None),
                substitution_count=spec.get("substitution_count", None),
                substitution_from=spec.get("substitution_from", None),
                substitution_to=spec.get("substitution_to", None),
                substitution_positions=spec.get("substitution_positions", None),
                substitution_global_indices=spec.get("substitution_global_indices", None),

                interstitial_count=spec.get("interstitial_count", None),
                interstitial_positions=spec.get("interstitial_positions", None),
                interstitial_species=spec.get("interstitial_species", None),
                interstitial_min_separation=spec.get("interstitial_min_separation", None),
            )
            # Restore applied logs if present
            applied = pd_data.get("applied", {})
            if applied:
                self._point_defects._applied_vacancies = [np.asarray(v, dtype=np.float32) for v in applied.get("vacancies", [])]
                self._point_defects._applied_substitutions = applied.get("substitutions", [])
                self._point_defects._applied_interstitials = applied.get("interstitials", [])
                
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
            
        # If point_defects exist, store them
        if self._point_defects is not None:
            pd = self._point_defects
            defect_metadata["point_defects"] = {
                "directory": pd.directory,
                "seed": pd.seed,
                "region_min": pd.region_min.tolist() if pd.region_min is not None else None,
                "region_max": pd.region_max.tolist() if pd.region_max is not None else None,
                "spec": {
                    "vacancy_fraction": pd.vacancy_fraction,
                    "vacancy_count": pd.vacancy_count,
                    "vacancy_global_indices": pd.vacancy_global_indices if pd.vacancy_global_indices is not None else None,
                    "vacancy_positions": pd.vacancy_positions.tolist() if pd.vacancy_positions is not None else None,
                    "vacancy_species_filter": pd.vacancy_species_filter if pd.vacancy_species_filter is not None else None,

                    "substitution_fraction": pd.substitution_fraction,
                    "substitution_count": pd.substitution_count,
                    "substitution_from": pd.substitution_from,
                    "substitution_to": pd.substitution_to,
                    "substitution_positions": pd.substitution_positions.tolist() if pd.substitution_positions is not None else None,
                    "substitution_global_indices": pd.substitution_global_indices if pd.substitution_global_indices is not None else None,

                    "interstitial_count": pd.interstitial_count,
                    "interstitial_positions": pd.interstitial_positions.tolist() if pd.interstitial_positions is not None else None,
                    "interstitial_species": pd.interstitial_species,
                    "interstitial_min_separation": pd.interstitial_min_separation,
                },
                "applied": {
                    "vacancies": [x.tolist() for x in pd._applied_vacancies],
                    "substitutions": [{"pos": s["pos"].tolist(), "from": s["from"], "to": s["to"]} for s in pd._applied_substitutions],
                    "interstitials": [{"pos": t["pos"].tolist(), "species": t["species"]} for t in pd._applied_interstitials],
                }
            }
        else:
            defect_metadata["point_defects"] = None

        # Write as nicely formatted JSON
        with open(metadata_filename, "w") as f:
            json.dump(defect_metadata, f, indent=4)
        print(f"Defect metadata written to {metadata_filename} in JSON format.")

    def add_stacking_faults(self,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap):
        self._stacking_faults = self.stacking_fault(self.directory,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap)
    
    def add_cracks(self,crack_points):
        self._cracks = self.crack(self.directory,crack_points)
        
    def add_point_defects(self, **kwargs):
        self._point_defects = self.point_defect(self.directory, **kwargs)
    
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
    
    @property
    def point_defects(self):
        if self._point_defects is None:
            print("self._point_defects has not been initialized yet")
        return self._point_defects

    # -------------------------------------------------------------------------
    # Sub-Classes
    # -------------------------------------------------------------------------
    class stacking_fault(logging):
        
        # -------------------------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------------------------
        __log_top__ = (
            "generate_global_positions",
            "apply_to_sample",
            "plot_global_positions",
            "apply_stacking_fault_chunk",
        )
        
        # -----------------------------------------------------------------------------
        # Functions
        # -----------------------------------------------------------------------------
        ## Initialization
        def __init__(self,directory,fault_number,fault_offset,fault_normal,interfault_spacing,burgers_vector,fault_orientation,fault_gap):
            super().__init__(log_name="stacking fault")
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
        def generate_global_positions(self,sample,crystal,plotting=False,use_gpu=True):
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
        
    class crack(logging):
        
        # -------------------------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------------------------
        __log_top__ = (
            "apply_to_sample",
            "apply_crack_chunk",
            "plot_crack_geometry",
        )
        
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
            super().__init__(log_name="crack")
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

        def apply_crack_chunk(self, positions_chunk, species_chunk_np, use_gpu=True):
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
        
    class point_defect(logging):
        """
        Create random or specific vacancies, substitutions, and interstitials.
        Provides a local relaxation routine and a plotting helper.

        Notes:
        - Works chunk-by-chunk using sample.load_chunk_positions/species and the paired
            write methods, same as stacking_fault/crack.  The relaxation step is local to
            the atoms present in each chunk (neighbors in other chunks are ignored).
        """

        __log_top__ = (
            "apply_to_sample",
            "relax_local_atoms",
            "plot_defects"
        )

        def __init__(self,
                    directory=None,
                    seed=None,
                    region_min=None,
                    region_max=None,
                    # vacancies
                    vacancy_fraction=None,
                    vacancy_count=None,
                    vacancy_global_indices=None,
                    vacancy_positions=None,
                    vacancy_species_filter=None,
                    # substitutions
                    substitution_fraction=None,
                    substitution_count=None,
                    substitution_from=None,
                    substitution_to=None,
                    substitution_positions=None,
                    substitution_global_indices=None,
                    # interstitials
                    interstitial_count=None,
                    interstitial_positions=None,
                    interstitial_species=None,
                    interstitial_min_separation=None,
                    # behavior
                    relax_after=False,
                    relax_params=None,
                    ):
            super().__init__(log_name="point_defect")
            self.directory = directory
            self.seed = None if seed is None else int(seed)

            # World-space bounding box to limit random/specific operations (optional)
            self.region_min = None if region_min is None else np.asarray(region_min, dtype=np.float32).reshape(3,)
            self.region_max = None if region_max is None else np.asarray(region_max, dtype=np.float32).reshape(3,)

            # Vacancy spec
            self.vacancy_fraction = float(vacancy_fraction) if vacancy_fraction is not None else None
            self.vacancy_count = int(vacancy_count) if vacancy_count is not None else None
            self.vacancy_global_indices = list(vacancy_global_indices) if vacancy_global_indices is not None else None
            self.vacancy_positions = None if vacancy_positions is None else np.asarray(vacancy_positions, dtype=np.float32).reshape(-1,3)
            self.vacancy_species_filter = None if vacancy_species_filter is None else list(vacancy_species_filter)

            # Substitution spec
            self.substitution_fraction = float(substitution_fraction) if substitution_fraction is not None else None
            self.substitution_count = int(substitution_count) if substitution_count is not None else None
            self.substitution_from = substitution_from
            self.substitution_to = substitution_to
            self.substitution_positions = None if substitution_positions is None else np.asarray(substitution_positions, dtype=np.float32).reshape(-1,3)
            self.substitution_global_indices = list(substitution_global_indices) if substitution_global_indices is not None else None

            # Interstitial spec
            self.interstitial_count = int(interstitial_count) if interstitial_count is not None else None
            self.interstitial_positions = None if interstitial_positions is None else np.asarray(interstitial_positions, dtype=np.float32).reshape(-1,3)
            self.interstitial_species = interstitial_species
            self.interstitial_min_separation = float(interstitial_min_separation) if interstitial_min_separation is not None else 0.0

            # Logs of what was applied (positions and species changes)
            self._applied_vacancies = []       # list of (3,) positions removed
            self._applied_substitutions = []   # list of {"pos":(3,), "from":str, "to":str}
            self._applied_interstitials = []   # list of {"pos":(3,), "species":str}

            # Optional immediate relaxation configuration
            self._relax_after = bool(relax_after)
            self._relax_params = relax_params if isinstance(relax_params, dict) else None

            # For random counts where a fraction is not given, we will gather global candidate
            # counts in a pre-pass to compute chunk-wise quotas.
            self._precomputed = {
                "vacancy_candidates_total": None,
                "substitution_candidates_total": None,
            }

        # ------------------------
        # Chainable convenience APIs (optional)
        def add_random_vacancies(self, fraction=None, count=None, species_filter=None):
            if fraction is not None: self.vacancy_fraction = float(fraction)
            if count is not None: self.vacancy_count = int(count)
            if species_filter is not None: self.vacancy_species_filter = list(species_filter)
            return self

        def add_specific_vacancies(self, positions=None, global_indices=None):
            if positions is not None:
                self.vacancy_positions = np.asarray(positions, dtype=np.float32).reshape(-1,3)
            if global_indices is not None:
                self.vacancy_global_indices = list(global_indices)
            return self

        def add_random_substitutions(self, fraction=None, count=None, from_species=None, to_species=None):
            if fraction is not None: self.substitution_fraction = float(fraction)
            if count is not None: self.substitution_count = int(count)
            if from_species is not None: self.substitution_from = from_species
            if to_species is not None: self.substitution_to = to_species
            return self

        def add_specific_substitutions(self, positions=None, global_indices=None, to_species=None, from_species=None):
            if positions is not None:
                self.substitution_positions = np.asarray(positions, dtype=np.float32).reshape(-1,3)
            if global_indices is not None:
                self.substitution_global_indices = list(global_indices)
            if to_species is not None:
                self.substitution_to = to_species
            if from_species is not None:
                self.substitution_from = from_species
            return self

        def add_random_interstitials(self, count, species, min_separation=0.0):
            self.interstitial_count = int(count)
            self.interstitial_species = species
            self.interstitial_min_separation = float(min_separation)
            return self

        def add_specific_interstitials(self, positions, species):
            self.interstitial_positions = np.asarray(positions, dtype=np.float32).reshape(-1,3)
            self.interstitial_species = species
            return self

        # ------------------------
        # Core pipeline

        def apply_to_sample(self, sample, use_gpu=False, tol_match=1e-4):
            """
            Stream over chunks:
            - random/specific vacancy deletions
            - random/specific substitutions (species swap)
            - random/specific interstitial insertions
            Then optionally relax locally and write back. Writes both positions and species arrays.

            Notes:
            - Uses sample.load_chunk_positions/species and write_* counterparts, preserving the
                chunked .npy layout, as in the other sub-classes.  :contentReference[oaicite:2]{index=2}
            """
            rng = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
            n_chunks = int(sample.chunk_total)

            # First pass: for random "count" (no fraction) compute candidate totals across all chunks.
            vac_need_fraction = (self.vacancy_count is not None and self.vacancy_fraction is None)
            sub_need_fraction = (self.substitution_count is not None and self.substitution_fraction is None)
            if vac_need_fraction or sub_need_fraction:
                vac_total_cand = 0
                sub_total_cand = 0
                for i in range(n_chunks):
                    pos_i = sample.load_chunk_positions(i+1, use_gpu=False)
                    spc_i = sample.load_chunk_species(i+1, use_gpu=False)
                    region_mask = self._region_mask(pos_i)
                    if vac_need_fraction:
                        vac_cand_mask = self._vacancy_candidate_mask(pos_i, spc_i, region_mask)
                        vac_total_cand += int(np.count_nonzero(vac_cand_mask))
                    if sub_need_fraction:
                        sub_cand_mask = self._substitution_candidate_mask(pos_i, spc_i, region_mask)
                        sub_total_cand += int(np.count_nonzero(sub_cand_mask))
                self._precomputed["vacancy_candidates_total"] = int(vac_total_cand)
                self._precomputed["substitution_candidates_total"] = int(sub_total_cand)

            # Fractions derived from counts if needed
            vacancy_fraction_eff = self.vacancy_fraction
            if vacancy_fraction_eff is None and self.vacancy_count is not None:
                denom = max(1, int(self._precomputed["vacancy_candidates_total"] or 0))
                vacancy_fraction_eff = float(self.vacancy_count) / float(denom)

            substitution_fraction_eff = self.substitution_fraction
            if substitution_fraction_eff is None and self.substitution_count is not None:
                denom = max(1, int(self._precomputed["substitution_candidates_total"] or 0))
                substitution_fraction_eff = float(self.substitution_count) / float(denom)

            # Global index windows for "specific by global index"
            global_start = 0

            # Interstitial random distribution across chunks
            interstitials_remaining_random = int(self.interstitial_count or 0)
            interstitials_specific_left = None if self.interstitial_positions is None else list(range(self.interstitial_positions.shape[0]))

            for i in range(n_chunks):
                # Load
                pos = sample.load_chunk_positions(i+1, use_gpu=False)
                spc = sample.load_chunk_species(i+1, use_gpu=False)

                N = pos.shape[0]
                region_mask = self._region_mask(pos)

                # Construct delete mask for vacancies
                delete_mask = np.zeros(N, dtype=bool)

                # Specific vacancies by global index
                if self.vacancy_global_indices is not None:
                    # local window
                    g0, g1 = global_start, global_start + N
                    local_hits = [g - g0 for g in self.vacancy_global_indices if (g >= g0 and g < g1)]
                    if local_hits:
                        delete_mask[np.asarray(local_hits, dtype=int)] = True

                # Specific vacancies by position (tolerant match)
                if self.vacancy_positions is not None and self.vacancy_positions.size > 0:
                    delete_mask |= self._indices_from_positions(pos, self.vacancy_positions, tol=tol_match)

                # Random vacancies (fractional on eligible candidates)
                if vacancy_fraction_eff is not None and vacancy_fraction_eff > 0.0:
                    vac_cand = self._vacancy_candidate_mask(pos, spc, region_mask) & (~delete_mask)
                    num_cand = int(np.count_nonzero(vac_cand))
                    if num_cand > 0:
                        k = int(np.round(vacancy_fraction_eff * num_cand))
                        if k > 0:
                            cand_idx = np.flatnonzero(vac_cand)
                            pick = rng.choice(cand_idx, size=min(k, cand_idx.size), replace=False)
                            delete_mask[pick] = True

                # Substitutions
                subs_mask_local = np.zeros(N, dtype=bool)
                subs_to_species = None  # filled only for substituted indices
                if self.substitution_from is not None and self.substitution_to is not None:
                    # Specific substitutions by global index
                    if self.substitution_global_indices is not None:
                        g0, g1 = global_start, global_start + N
                        local_hits = [g - g0 for g in self.substitution_global_indices if (g >= g0 and g < g1)]
                        if local_hits:
                            idx = np.asarray(local_hits, dtype=int)
                            subs_mask_local[idx] = True

                    # Specific substitutions by position
                    if self.substitution_positions is not None and self.substitution_positions.size > 0:
                        subs_mask_local |= self._indices_from_positions(pos, self.substitution_positions, tol=tol_match)

                    # Random substitutions
                    if substitution_fraction_eff is not None and substitution_fraction_eff > 0.0:
                        sub_cand = self._substitution_candidate_mask(pos, spc, region_mask) & (~delete_mask) & (~subs_mask_local)
                        num_cand = int(np.count_nonzero(sub_cand))
                        if num_cand > 0:
                            k = int(np.round(substitution_fraction_eff * num_cand))
                            if k > 0:
                                cand_idx = np.flatnonzero(sub_cand)
                                pick = rng.choice(cand_idx, size=min(k, cand_idx.size), replace=False)
                                subs_mask_local[pick] = True

                    # Commit substitution: change species at subs_mask_local
                    if np.any(subs_mask_local):
                        spc = spc.astype(object, copy=True)  # robust for string updates
                        from_mask = (spc == self.substitution_from)
                        apply_mask = subs_mask_local & from_mask & (~delete_mask)
                        if np.any(apply_mask):
                            # Log applied substitutions
                            for idx in np.flatnonzero(apply_mask):
                                self._applied_substitutions.append({
                                    "pos": pos[idx].copy(),
                                    "from": self.substitution_from,
                                    "to": self.substitution_to
                                })
                            spc[apply_mask] = self.substitution_to

                # Capture positions of atoms to be deleted (for plotting/relax)
                if np.any(delete_mask):
                    removed_positions = pos[delete_mask].copy()
                    for p in removed_positions:
                        self._applied_vacancies.append(p)

                # Remove vacancies
                keep_mask = ~delete_mask
                pos = pos[keep_mask]
                spc = spc[keep_mask]

                # Interstitials: specific
                new_pos_list = []
                new_spc_list = []
                if interstitials_specific_left is not None and len(interstitials_specific_left) > 0:
                    # simple round-robin allocation of specific positions across chunks
                    # compute a per-chunk slice size ~ evenly
                    slice_len = max(0, int(np.ceil(len(interstitials_specific_left) / float(n_chunks - i))))
                    if slice_len > 0:
                        sel_idx = interstitials_specific_left[:slice_len]
                        interstitials_specific_left = interstitials_specific_left[slice_len:]
                        P = self.interstitial_positions[sel_idx, :]
                        # region clip (optional)
                        if self.region_min is not None and self.region_max is not None:
                            rm = self._in_region_mask(P)
                            P = P[rm]
                        for p in P:
                            if self.interstitial_min_separation > 0.0:
                                if not self._ok_min_sep(p, pos, self.interstitial_min_separation):
                                    continue
                            new_pos_list.append(p)
                            new_spc_list.append(self.interstitial_species)
                            self._applied_interstitials.append({"pos": p.copy(), "species": self.interstitial_species})

                # Interstitials: random
                if interstitials_remaining_random > 0 and self.interstitial_species is not None:
                    # take a per-chunk share
                    to_take = int(np.ceil(interstitials_remaining_random / float(n_chunks - i)))
                    if to_take > 0:
                        # sample uniformly in region box if given, else sample box of sample
                        box_min, box_max = self._region_or_sample_box(sample)
                        # try to generate respecting min separation
                        tries = 0
                        added = 0
                        while added < to_take and tries < 50 * to_take:
                            tries += 1
                            p = rng.uniform(low=box_min, high=box_max, size=(3,)).astype(np.float32)
                            if self.region_min is not None and self.region_max is not None:
                                if not self._in_region_mask(p[None, :])[0]:
                                    continue
                            if self.interstitial_min_separation > 0.0:
                                if not self._ok_min_sep(p, pos, self.interstitial_min_separation):
                                    continue
                            new_pos_list.append(p)
                            new_spc_list.append(self.interstitial_species)
                            self._applied_interstitials.append({"pos": p.copy(), "species": self.interstitial_species})
                            added += 1
                        interstitials_remaining_random -= added

                # Append interstitials to chunk arrays
                if new_pos_list:
                    pos = np.concatenate([pos, np.asarray(new_pos_list, dtype=np.float32)], axis=0)
                    spc = np.concatenate([spc, np.asarray(new_spc_list, dtype=object)], axis=0)

                # Write updated arrays
                sample.write_chunk_positions(pos, i+1, override_directory=self.directory)
                sample.write_chunk_species(spc, i+1, override_directory=self.directory)

                # Advance global index window
                global_start += N

            # Finalize sample metadata in the chosen directory
            sample.write_sample_metadata(override_directory=self.directory)

            # Optional relaxation now
            if self._relax_after:
                params = self._relax_params if self._relax_params else {}
                self.relax_local_atoms(sample, **params)

        # ------------------------
        # Relaxation

        def relax_local_atoms(self,
                            sample,
                            r_cut=2.0,
                            strength=0.05,
                            iterations=2,
                            decay=0.8,
                            use_gpu=False):
            """
            Light-weight local relaxation around recorded defect centers using a
            simple radial update:
            - Interstitials push neighbors away
            - Vacancies pull neighbors toward the vacancy site
            - Substitutions pull or push based on a size heuristic (default: pull in)

            Update rule per center c and neighbor x within r_cut:
                w = exp(-(r/r_cut)^2)
                delta = sgn * strength * w * (x - c) / (r + 1e-12)
            summed over centers, with 'strength' decayed each iteration by 'decay'.

            Notes:
            - Operates chunk-by-chunk and only considers atoms present in the chunk.
            - Keeps atoms inside the sample AABB.
            """
            # Collect centers
            vacancy_centers = [np.asarray(p, dtype=np.float32).reshape(1,3) for p in self._applied_vacancies]
            sub_centers = [np.asarray(x["pos"], dtype=np.float32).reshape(1,3) for x in self._applied_substitutions]
            int_centers = [np.asarray(x["pos"], dtype=np.float32).reshape(1,3) for x in self._applied_interstitials]

            # Concatenate to arrays
            V = np.vstack(vacancy_centers) if vacancy_centers else np.zeros((0,3), dtype=np.float32)
            S = np.vstack(sub_centers) if sub_centers else np.zeros((0,3), dtype=np.float32)
            I = np.vstack(int_centers) if int_centers else np.zeros((0,3), dtype=np.float32)

            box_min, box_max = self._region_or_sample_box(sample)

            # Early out if nothing to do
            if V.shape[0] == 0 and S.shape[0] == 0 and I.shape[0] == 0:
                return

            n_chunks = int(sample.chunk_total)
            for it in range(int(iterations)):
                step = float(strength) * (float(decay) ** it)
                for i in range(n_chunks):
                    pos = sample.load_chunk_positions(i+1, use_gpu=False)
                    spc = sample.load_chunk_species(i+1, use_gpu=False)  # unchanged species

                    # Accumulate displacements
                    disp = np.zeros_like(pos, dtype=np.float32)

                    if I.shape[0] > 0:
                        dI = self._accumulate_radial_disp(pos, I, r_cut, +step)  # push
                        disp += dI
                    if V.shape[0] > 0:
                        dV = self._accumulate_radial_disp(pos, V, r_cut, -step)  # pull
                        disp += dV
                    if S.shape[0] > 0:
                        dS = self._accumulate_radial_disp(pos, S, r_cut, -0.5*step)  # mild pull by default
                        disp += dS

                    pos += disp

                    # Clamp to sample AABB
                    np.clip(pos, box_min, box_max, out=pos)

                    sample.write_chunk_positions(pos, i+1, override_directory=self.directory)
                    sample.write_chunk_species(spc, i+1, override_directory=self.directory)

            sample.write_sample_metadata(override_directory=self.directory)

        # ------------------------
        # Plotting

        def plot_defects(self, sample, elev=15, azim=-60, size=8):
            """
            Scatter-plot of vacancies, substitutions, interstitials in the sample AABB.
            """
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(size, size))
            ax = fig.add_subplot(111, projection='3d')

            # draw sample wireframe
            corners = sample.corners
            edges = [(0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)]
            for (a,b) in edges:
                x = [corners[a,0], corners[b,0]]
                y = [corners[a,1], corners[b,1]]
                z = [corners[a,2], corners[b,2]]
                ax.plot(x,y,z, c="gray", lw=1)

            # plot defects
            if self._applied_vacancies:
                P = np.vstack([np.asarray(p, dtype=np.float32) for p in self._applied_vacancies])
                ax.scatter(P[:,0], P[:,1], P[:,2], s=12, c="r", marker="x", label="vacancy")
            if self._applied_substitutions:
                P = np.vstack([np.asarray(x["pos"], dtype=np.float32) for x in self._applied_substitutions])
                ax.scatter(P[:,0], P[:,1], P[:,2], s=10, c="g", marker="o", label="substitution")
            if self._applied_interstitials:
                P = np.vstack([np.asarray(x["pos"], dtype=np.float32) for x in self._applied_interstitials])
                ax.scatter(P[:,0], P[:,1], P[:,2], s=10, c="m", marker="^", label="interstitial")

            ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
            ax.view_init(elev=elev, azim=azim)
            ax.legend(loc="best")
            plt.tight_layout()
            return fig, ax

        # ------------------------
        # Helpers

        def _region_mask(self, positions):
            if self.region_min is None or self.region_max is None:
                return np.ones(positions.shape[0], dtype=bool)
            p = positions
            r = (p[:,0] >= self.region_min[0]) & (p[:,0] <= self.region_max[0]) & \
                (p[:,1] >= self.region_min[1]) & (p[:,1] <= self.region_max[1]) & \
                (p[:,2] >= self.region_min[2]) & (p[:,2] <= self.region_max[2])
            return r

        def _in_region_mask(self, P):
            if self.region_min is None or self.region_max is None:
                return np.ones(P.shape[0], dtype=bool)
            r = (P[:,0] >= self.region_min[0]) & (P[:,0] <= self.region_max[0]) & \
                (P[:,1] >= self.region_min[1]) & (P[:,1] <= self.region_max[1]) & \
                (P[:,2] >= self.region_min[2]) & (P[:,2] <= self.region_max[2])
            return r

        def _vacancy_candidate_mask(self, pos, spc, region_mask):
            mask = region_mask.copy()
            if self.vacancy_species_filter is not None:
                spc_obj = spc.astype(object, copy=False)
                allowed = np.zeros(spc_obj.shape[0], dtype=bool)
                for s in self.vacancy_species_filter:
                    allowed |= (spc_obj == s)
                mask &= allowed
            return mask

        def _substitution_candidate_mask(self, pos, spc, region_mask):
            mask = region_mask.copy()
            if self.substitution_from is not None:
                spc_obj = spc.astype(object, copy=False)
                mask &= (spc_obj == self.substitution_from)
            return mask

        def _indices_from_positions(self, pos, target_positions, tol=1e-4):
            # Tolerant nearest match: any site within tol of a target is selected
            sel = np.zeros(pos.shape[0], dtype=bool)
            if target_positions.size == 0:
                return sel
            for t in target_positions:
                d = pos - t[None, :]
                r2 = np.sum(d*d, axis=1)
                hit = np.where(r2 <= float(tol*tol))[0]
                if hit.size > 0:
                    sel[hit[0]] = True
            return sel

        def _ok_min_sep(self, p, pos, r_min):
            if pos.shape[0] == 0:
                return True
            d = pos - p[None, :]
            r2 = np.sum(d*d, axis=1)
            return bool(np.all(r2 >= float(r_min*r_min)))

        def _region_or_sample_box(self, sample):
            if self.region_min is not None and self.region_max is not None:
                return self.region_min.copy(), self.region_max.copy()
            # sample box: centered at offset with lengths=dimensions
            dims = sample.dimensions.astype(np.float32)
            mn = (sample.offset - 0.5*dims).astype(np.float32)
            mx = mn + dims
            return mn, mx

        def _accumulate_radial_disp(self, pos, centers, r_cut, step_signed):
            # For each center, compute outward unit vector and weight
            if centers.shape[0] == 0 or pos.shape[0] == 0:
                return np.zeros_like(pos, dtype=np.float32)
            acc = np.zeros_like(pos, dtype=np.float32)
            rc = float(r_cut)
            for c in centers:
                v = pos - c[None, :]
                r = np.linalg.norm(v, axis=1)
                mask = (r > 1e-12) & (r <= rc)
                if not np.any(mask):
                    continue
                u = v[mask] / r[mask][:, None]
                w = np.exp(- (r[mask] / rc)**2).astype(np.float32)
                acc[mask] += (step_signed * w)[:, None] * u
            return acc
