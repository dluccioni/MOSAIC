# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
import json
import re
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
        "import_dislocation_network",
        "generate_nodal_field",
    )
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=None):
        """
        Initialize the defects object.

        Args:
            directory: Optional path to the directory for storing defect data.
                If provided and does not exist, it will be created.
        """
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
        Read defect metadata from a JSON file and restore object state.

        Reads the defect metadata JSON file from disk and restores this defect
        object's state, including stacking faults, cracks, and point defects
        if present.

        Args:
            override_directory: Optional directory path to read from instead
                of the default self.directory.

        Raises:
            FileNotFoundError: If the metadata file does not exist.
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
        Serialize the defect object's state to a JSON file on disk.

        Writes critical internal fields (defect history, stacking faults,
        cracks, point defects) to a human-readable JSON file so that the
        state can be restored later.

        Args:
            override_directory: Optional directory path to write to instead
                of the default self.directory.
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

    def add_stacking_faults(self, fault_number, fault_offset, fault_normal,
                            interfault_spacing, burgers_vector, fault_orientation, fault_gap):
        """
        Add stacking faults to the defect object.

        Args:
            fault_number: Number of stacking faults to create.
            fault_offset: Offset vector for the fault position in the sample.
            fault_normal: Normal vector to the stacking fault plane.
            interfault_spacing: Spacing between consecutive fault planes.
            burgers_vector: Burgers vector defining the fault displacement.
            fault_orientation: Orientation pattern for the faults (list of +1/-1).
            fault_gap: Gap size at each fault plane.
        """
        self._stacking_faults = self.stacking_fault(
            self.directory, fault_number, fault_offset, fault_normal,
            interfault_spacing, burgers_vector, fault_orientation, fault_gap)

    def add_cracks(self, crack_points):
        """
        Add a crack to the defect object.

        Creates a crack defined by a convex hull of the given points.
        Atoms inside the hull will be removed when applied to a sample.

        Args:
            crack_points: Array-like of shape (N, 3) defining the vertices
                of the convex hull representing the crack geometry.
        """
        self._cracks = self.crack(self.directory, crack_points)

    def add_point_defects(self, **kwargs):
        """
        Add point defects (vacancies, substitutions, interstitials).

        Creates a point_defect object with the specified parameters.
        See the point_defect class for full parameter documentation.

        Args:
            **kwargs: Keyword arguments passed to the point_defect constructor.
                Common parameters include vacancy_fraction, vacancy_count,
                substitution_from, substitution_to, interstitial_count, etc.
        """
        self._point_defects = self.point_defect(self.directory, **kwargs)
            
    def import_dislocation_network(self,
                                    filepath,
                                    crystal,
                                    burgers_magnitude=None,
                                    burgers_family="fcc_110_over_2",
                                    dtype=np.float32):
        """
        Parse an OpenDiS config file and reconstruct the dislocation network.

        Parses an OpenDiS config.*.data file, reconstructs the dislocation
        network, and caches arrays for GPU/CPU evaluation.

        Args:
            filepath: Path to the OpenDiS config.*.data file.
            crystal: Crystal object providing lattice information.
            burgers_magnitude: Magnitude of the Burgers vector. If None,
                computed from burgers_family and crystal lattice parameters.
            burgers_family: Family of Burgers vectors. Currently supports
                "fcc_110_over_2". Defaults to "fcc_110_over_2".
            dtype: NumPy dtype for stored arrays. Defaults to np.float32.

        Returns:
            dict: Summary with keys:
                - node_count: Number of nodes in the network.
                - segment_count: Number of segments.
                - bounds_min: Minimum coordinates of the bounding box.
                - bounds_max: Maximum coordinates of the bounding box.

        Raises:
            FileNotFoundError: If filepath does not exist.
            ValueError: If burgers_family is unknown and burgers_magnitude is None.

        Note:
            Caches the following attributes on self:
                - _opendis_nodes_xyz: (N, 3) node coordinates.
                - _opendis_segments: (M, 2) 0-based node index pairs.
                - _opendis_S0, _opendis_S1: (M, 3) segment endpoints.
                - _opendis_bvec: (M, 3) real-space Burgers vectors.
                - _opendis_tvec: (M, 3) unit line directions.
                - _opendis_mids: (M, 3) segment midpoints.
                - _opendis_halfL: (M,) half lengths.
                - _opendis_bounds: dict with "min" and "max" bounds.
        """
        import re, os

        if not os.path.isfile(filepath):
            raise FileNotFoundError("File not found: {}".format(filepath))

        # Resolve |b|
        if burgers_magnitude is None:
            if burgers_family == "fcc_110_over_2":
                a = float(np.asarray(crystal.lattice_lengths_conventional, dtype=np.float64)[0])
                burgers_magnitude = a / np.sqrt(2.0)
            else:
                raise ValueError("Unknown burgers_family '{}' and burgers_magnitude is None".format(burgers_family))
        bmag = float(burgers_magnitude)

        # Crystal-space -> real-space mapping
        L = np.asarray(crystal.lattice_matrix_conventional, dtype=np.float64)
        Lens = np.asarray(crystal.lattice_lengths_conventional, dtype=np.float64).reshape(3,1)
        Bmap = (L / Lens)  # 3x3

        # Read lines
        with open(filepath, "r", errors="ignore") as f:
            lines = f.read().splitlines()

        def _seek(tag):
            for i, ln in enumerate(lines):
                if ln.strip().startswith(tag):
                    return i
            return -1

        i_min = _seek("minCoordinates")
        i_max = _seek("maxCoordinates")
        if i_min < 0 or i_max < 0:
            raise ValueError("Could not find minCoordinates/maxCoordinates in {}".format(filepath))
        bounds_min = np.array([float(lines[i_min+1].strip()),
                            float(lines[i_min+2].strip()),
                            float(lines[i_min+3].strip())], dtype=np.float64)
        bounds_max = np.array([float(lines[i_max+1].strip()),
                            float(lines[i_max+2].strip()),
                            float(lines[i_max+3].strip())], dtype=np.float64)

        node_hdr = re.compile(r'^\s*(\d+),\s*(\d+)\s+([\-0-9Ee\.+]+)\s+([\-0-9Ee\.+]+)\s+([\-0-9Ee\.+]+)\s+(\d+)\s+(\d+)')
        arm_l1   = re.compile(r'^\s*(\d+),\s*(\d+)\s+([\-0-9Ee\.+]+)\s+([\-0-9Ee\.+]+)\s+([\-0-9Ee\.+]+)\s*$')
        arm_l2   = re.compile(r'^\s*([\-0-9Ee\.+]+)\s+([\-0-9Ee\.+]+)\s+([\-0-9Ee\.+]+)\s*$')

        nodes_xyz = {}
        arms_by_node = {}
        i = 0
        while i < len(lines):
            m = node_hdr.match(lines[i])
            if not m:
                i += 1
                continue
            dom, node_id, x, y, z, n_arms, _flag = m.groups()
            node_id = int(node_id)
            nodes_xyz[node_id] = (float(x), float(y), float(z))
            i += 1
            arms = []
            for _ in range(int(n_arms)):
                m1 = arm_l1.match(lines[i]); m2 = arm_l2.match(lines[i+1]) if (i+1) < len(lines) else None
                if not (m1 and m2):
                    raise ValueError("Malformed arm block after node {}".format(node_id))
                _dom2, nbr_id, bx, by, bz = m1.groups()
                nx, ny, nz = m2.groups()
                arms.append((int(nbr_id),
                            float(bx), float(by), float(bz),
                            float(nx), float(ny), float(nz)))
                i += 2
            arms_by_node[node_id] = arms

        # Deduplicate segments by sorted node-pair
        seg_keys = []
        seg_S0 = []
        seg_S1 = []
        seg_b = []
        seg_n = []
        seen = set()
        for ni, arm_list in arms_by_node.items():
            pi = np.asarray(nodes_xyz[ni], dtype=np.float64)
            for nbr, bx, by, bz, nx, ny, nz in arm_list:
                if ni == nbr:
                    continue
                key = (ni, nbr) if ni < nbr else (nbr, ni)
                if key in seen:
                    continue
                seen.add(key)
                pj = np.asarray(nodes_xyz[nbr], dtype=np.float64)

                b_dir_crys = np.array([bx, by, bz], dtype=np.float64)
                nrm = np.linalg.norm(b_dir_crys)
                if nrm == 0:
                    continue
                b_dir_crys /= nrm
                b_dir_real = (Bmap @ b_dir_crys.reshape(3,1)).reshape(3)
                bn = np.linalg.norm(b_dir_real)
                if bn == 0:
                    continue
                b_vec = (b_dir_real / bn) * bmag

                seg_keys.append(key)
                seg_S0.append(pi)
                seg_S1.append(pj)
                seg_b.append(b_vec.astype(np.float64))
                seg_n.append(np.array([nx, ny, nz], dtype=np.float64))

        S0 = np.asarray(seg_S0, dtype=dtype)
        S1 = np.asarray(seg_S1, dtype=dtype)
        Bv = np.asarray(seg_b, dtype=dtype)
        segs = np.asarray(seg_keys, dtype=np.int64)

        # Box size from the config file
        box = (bounds_max - bounds_min).astype(np.float64)

        # Start from the raw segment vectors
        Lvec = (S1 - S0).astype(np.float64)

        # Minimum-image convention: ensure each component of the segment
        # vector lies in [-L/2, L/2] in a periodic domain.
        if np.all(box > 0.0):
            for ax in range(3):
                L = box[ax]
                halfL = 0.5 * L
                d = Lvec[:, ax]
                d[d >  halfL] -= L
                d[d < -halfL] += L
                Lvec[:, ax] = d
            # Move the second endpoint into the nearest periodic image
            S1 = (S0 + Lvec).astype(dtype)

        Llen_raw = np.linalg.norm(Lvec, axis=1)
        max_len = 0.5 * np.max(box)   # e.g. anything ~box-size is suspicious
        keep = (Llen_raw <= max_len)
        if not np.all(keep):
            S0 = S0[keep, :]
            S1 = S1[keep, :]
            Bv = Bv[keep, :]
            segs = segs[keep, :]
            Lvec = Lvec[keep, :]

        # Unit line directions (after PBC fix and optional filtering)
        Llen = np.linalg.norm(Lvec, axis=1)
        Tvec = np.divide(Lvec, Llen[:, None],
                         where=(Llen[:, None] > 0)).astype(dtype)

        self._opendis_nodes_xyz = np.asarray(
            [nodes_xyz[k] for k in sorted(nodes_xyz.keys())], dtype=dtype
        )
        self._opendis_segments = segs
        self._opendis_S0 = S0
        self._opendis_S1 = S1
        self._opendis_bvec = Bv
        self._opendis_tvec = Tvec
        self._opendis_bounds = {"min": bounds_min.astype(dtype),
                                "max": bounds_max.astype(dtype)}

        mids = 0.5*(S0 + S1)
        halfL = 0.5*np.linalg.norm(S1 - S0, axis=1)
        self._opendis_mids = mids.astype(dtype)
        self._opendis_halfL = halfL.astype(dtype)

        return {
            "node_count": int(len(nodes_xyz)),
            "segment_count": int(S0.shape[0]),
            "bounds_min": bounds_min.tolist(),
            "bounds_max": bounds_max.tolist()
        }


    def clip_dislocation_network_to_sample(self, sample, margin=0.0, return_mask=False):
        """
        Clip dislocation network to the sample bounding box.

        Keeps only dislocation segments that intersect the sample axis-aligned
        bounding box (AABB) plus an optional margin. Rebuilds a compact node
        set and recomputes per-segment derived arrays.

        Args:
            sample: Sample object exposing a 'corners' attribute with shape
                (8, 3) defining the sample bounding box in the same frame.
            margin: Amount to expand the AABB by on each side. Defaults to 0.0.
            return_mask: If True, also return the boolean mask of kept segments.
                Defaults to False.

        Returns:
            dict: Summary with keys:
                - segments_before: Number of segments before clipping.
                - segments_after: Number of segments after clipping.
                - nodes_before: Number of nodes before clipping.
                - nodes_after: Number of nodes after clipping.

            If return_mask is True, returns a tuple (dict, mask) where mask is
            the boolean array indicating which segments were kept.

        Raises:
            RuntimeError: If dislocation network has not been imported.
            ValueError: If no segments are available to clip.

        Note:
            Updates the following attributes on self:
                _opendis_nodes_xyz, _opendis_segments, _opendis_S0, _opendis_S1,
                _opendis_bvec, _opendis_tvec, _opendis_mids, _opendis_halfL,
                _opendis_bounds.
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Dislocation network not initialized. Call import_dislocation_network(...) first.")

        # Current arrays (CPU/NumPy)
        S0 = np.asarray(self._opendis_S0, dtype=np.float64)
        S1 = np.asarray(self._opendis_S1, dtype=np.float64)
        Bv = np.asarray(self._opendis_bvec, dtype=np.float64)
        seg_idx = np.asarray(self._opendis_segments, dtype=np.int64)
        nodes = np.asarray(self._opendis_nodes_xyz, dtype=np.float64)

        M = int(S0.shape[0])
        if M == 0:
            raise ValueError("No segments available to clip.")

        # Sample AABB (+margin)
        corners = np.asarray(sample.corners, dtype=np.float64)
        cmin = corners.min(axis=0)
        cmax = corners.max(axis=0)
        m = float(max(0.0, margin))
        cmin = cmin - m
        cmax = cmax + m

        # Liang-Barsky segment-AABB intersection
        d = S1 - S0
        t0 = np.zeros((M,), dtype=np.float64)
        t1 = np.ones((M,), dtype=np.float64)
        valid = np.ones((M,), dtype=bool)

        for ax in range(3):
            p0 = S0[:, ax]
            di = d[:, ax]
            nz = (di != 0.0)
            inv = np.zeros_like(di, dtype=np.float64)
            inv[nz] = 1.0 / di[nz]

            tmin = (cmin[ax] - p0) * inv
            tmax = (cmax[ax] - p0) * inv
            tlow = np.minimum(tmin, tmax)
            thigh = np.maximum(tmin, tmax)

            t0 = np.maximum(t0, tlow)
            t1 = np.minimum(t1, thigh)

            # For segments parallel to this axis, require p0 within the slab
            if np.any(~nz):
                mask = ~nz
                valid[mask] &= (p0[mask] >= cmin[ax]) & (p0[mask] <= cmax[ax])

        keep = valid & (t0 <= t1)

        if not np.any(keep):
            return
            # raise ValueError("Clipping removed all segments. Increase margin or verify inputs.")

        # Filter segments and associated per-segment data
        seg_keep = seg_idx[keep, :]
        Bv_keep = Bv[keep, :]

        # Build compact node set and reindex segments
        used_nodes = np.unique(seg_keep.ravel())
        remap = -np.ones((nodes.shape[0],), dtype=np.int64)
        remap[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
        seg_new = remap[seg_keep]

        nodes_new = nodes[used_nodes, :]

        # Recompute S0/S1 from the compact node set
        S0_new = nodes_new[seg_new[:, 0], :]
        S1_new = nodes_new[seg_new[:, 1], :]

        # Recompute derived per-segment quantities
        Lvec = S1_new - S0_new
        Llen = np.linalg.norm(Lvec, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Tvec_new = np.divide(Lvec, Llen[:, None], out=np.zeros_like(Lvec), where=(Llen[:, None] > 0))
        mids_new = 0.5 * (S0_new + S1_new)
        halfL_new = 0.5 * Llen

        # Update bounds from segment endpoints
        pts = np.vstack([S0_new, S1_new])
        bmin = pts.min(axis=0).astype(np.float32)
        bmax = pts.max(axis=0).astype(np.float32)

        # Commit to self
        dt = np.dtype(np.float32)
        self._opendis_nodes_xyz = nodes_new.astype(dt, copy=False)
        self._opendis_segments = seg_new.astype(np.int64, copy=False)
        self._opendis_S0 = S0_new.astype(dt, copy=False)
        self._opendis_S1 = S1_new.astype(dt, copy=False)
        self._opendis_bvec = Bv_keep.astype(dt, copy=False)
        self._opendis_tvec = Tvec_new.astype(dt, copy=False)
        self._opendis_mids = mids_new.astype(dt, copy=False)
        self._opendis_halfL = halfL_new.astype(dt, copy=False)
        self._opendis_bounds = {"min": bmin, "max": bmax}

        info = {
            "segments_before": int(M),
            "segments_after": int(S0_new.shape[0]),
            "nodes_before": int(nodes.shape[0]),
            "nodes_after": int(nodes_new.shape[0]),
        }
        if return_mask:
            return info, keep
        return info


    def zero_dislocation_network(self, mode="aabb_min_to_origin"):
        """
        Translate the dislocation network to align with the origin.

        Translates the network so a chosen reference point of its axis-aligned
        bounding box (AABB) is at the origin.

        Args:
            mode: Translation mode. Options are:
                - "aabb_min_to_origin": Move the AABB minimum corner to (0,0,0).
                - "aabb_center_to_origin": Move the AABB center to (0,0,0).
                Defaults to "aabb_min_to_origin".

        Returns:
            dict: Contains "translation" key with the applied translation vector
                as a numpy array of shape (3,).

        Raises:
            RuntimeError: If dislocation network has not been imported.
            ValueError: If mode is not recognized.
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Dislocation network not initialized. Call import_dislocation_network(...) first.")

        S0 = np.asarray(self._opendis_S0, dtype=np.float64)
        S1 = np.asarray(self._opendis_S1, dtype=np.float64)
        pts = np.vstack([S0, S1])

        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        if mode == "aabb_min_to_origin":
            t = -bmin
        elif mode == "aabb_center_to_origin":
            t = -0.5 * (bmin + bmax)
        else:
            raise ValueError('mode must be "aabb_min_to_origin" or "aabb_center_to_origin"')

        self.transform_dislocation_network(translate=t, position_scale=1.0)
        return {"translation": t.astype(np.float32, copy=False)}


    def transform_dislocation_network(self,
                                      position_scale=1.0,
                                      translate=None,
                                      rotate_axis=None,
                                      rotate_angle=None,
                                      rotate_matrix=None,
                                      degrees=True):
        """
        Apply scale, rotation, and translation to the dislocation network.

        Applies an isotropic scale, optional rotation, and optional translation
        to the dislocation network. Recomputes all derived arrays for consistency.

        The transformation order is: scale -> rotate -> translate.

        Args:
            position_scale: Isotropic scale factor for all position-like data.
                Defaults to 1.0.
            translate: 3-element sequence for translation (applied after rotation).
                If None, no translation is applied.
            rotate_axis: 3-element sequence defining the axis for axis-angle rotation.
                Requires rotate_angle to be set. Ignored if rotate_matrix is provided.
            rotate_angle: Rotation angle around rotate_axis. Interpreted in degrees
                unless degrees=False. Ignored if rotate_matrix is provided.
            rotate_matrix: Explicit 3x3 rotation matrix. If provided, rotate_axis
                and rotate_angle are ignored.
            degrees: If True, interpret rotate_angle in degrees. Defaults to True.

        Raises:
            RuntimeError: If dislocation network has not been imported.
            ValueError: If rotate_matrix is not 3x3 or rotate_axis is zero-length.

        Note:
            Updates the following arrays on self:
                - Positions: _opendis_nodes_xyz, _opendis_S0, _opendis_S1, _opendis_mids
                - Directions: _opendis_tvec (recomputed from S1-S0)
                - Magnitudes: _opendis_halfL (scaled), _opendis_bvec (rotated and scaled)
                - Bounds: _opendis_bounds
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Dislocation network not initialized. Call import_dislocation_network(...) first.")

        def _build_R(rotate_axis, rotate_angle, rotate_matrix, degrees):
            if rotate_matrix is not None:
                Rm = np.asarray(rotate_matrix, dtype=np.float64)
                if Rm.shape != (3, 3):
                    raise ValueError("rotate_matrix must be 3x3")
                return Rm
            if rotate_axis is None or rotate_angle is None:
                return None
            axis = np.asarray(rotate_axis, dtype=np.float64).reshape(3,)
            nrm = np.linalg.norm(axis)
            if nrm == 0.0:
                raise ValueError("rotate_axis must be non-zero")
            axis = axis / nrm
            ang = float(rotate_angle)
            if degrees:
                ang = np.deg2rad(ang)
            c = np.cos(ang)
            s = np.sin(ang)
            d1 = 1.0 - c
            x, y, z = axis[0], axis[1], axis[2]
            Rm = np.empty((3, 3), dtype=np.float64)
            Rm[0, 0] = c + d1*x*x
            Rm[0, 1] = d1*x*y - z*s
            Rm[0, 2] = d1*x*z + y*s
            Rm[1, 0] = d1*y*x + z*s
            Rm[1, 1] = c + d1*y*y
            Rm[1, 2] = d1*y*z - x*s
            Rm[2, 0] = d1*z*x - y*s
            Rm[2, 1] = d1*z*y + x*s
            Rm[2, 2] = c + d1*z*z
            return Rm

        R = _build_R(rotate_axis, rotate_angle, rotate_matrix, bool(degrees))
        s = float(position_scale if position_scale is not None else 1.0)
        t = None if translate is None else np.asarray(translate, dtype=np.float64).reshape(1, 3)

        # Fetch arrays
        nodes = np.asarray(self._opendis_nodes_xyz, dtype=np.float64)
        S0 = np.asarray(self._opendis_S0, dtype=np.float64)
        S1 = np.asarray(self._opendis_S1, dtype=np.float64)
        Bv = np.asarray(self._opendis_bvec, dtype=np.float64)
        seg_idx = np.asarray(self._opendis_segments, dtype=np.int64)

        # Transform positions: scale, rotate, then translate
        def _xform_pos(P):
            out = P * s
            if R is not None:
                out = out @ R
            if t is not None:
                out = out + t
            return out

        # Transform vectors: scale and rotate (no translation)
        def _xform_vec(V):
            out = V * s
            if R is not None:
                out = out @ R
            return out

        nodes_new = _xform_pos(nodes)
        S0_new = _xform_pos(S0)
        S1_new = _xform_pos(S1)

        # Recompute T, mids, half-lengths from transformed endpoints
        Lvec = S1_new - S0_new
        Llen = np.linalg.norm(Lvec, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            Tvec_new = np.divide(Lvec, Llen[:, None], out=np.zeros_like(Lvec), where=(Llen[:, None] > 0))
        mids_new = 0.5 * (S0_new + S1_new)
        halfL_new = 0.5 * Llen

        # Transform Burgers vectors as real-space vectors
        Bv_new = _xform_vec(Bv)

        # Update bounds from transformed endpoints
        pts = np.vstack([S0_new, S1_new])
        bmin = pts.min(axis=0).astype(np.float32)
        bmax = pts.max(axis=0).astype(np.float32)

        # Commit back (float32 storage by convention)
        dt = np.dtype(np.float32)
        self._opendis_nodes_xyz = nodes_new.astype(dt, copy=False)
        self._opendis_S0 = S0_new.astype(dt, copy=False)
        self._opendis_S1 = S1_new.astype(dt, copy=False)
        self._opendis_bvec = Bv_new.astype(dt, copy=False)
        self._opendis_tvec = Tvec_new.astype(dt, copy=False)
        self._opendis_mids = mids_new.astype(dt, copy=False)
        self._opendis_halfL = halfL_new.astype(dt, copy=False)
        self._opendis_segments = seg_idx  # unchanged
        self._opendis_bounds = {"min": bmin, "max": bmax}

    def generate_nodal_field(self,
                             crystal,
                             mu,
                             nu,
                             grid_shape=(64, 64, 64),
                             bounds=None,
                             padding=0.0,
                             core_radius=5.0,
                             r_cut=None,
                             scale=1.0,
                             write_directory=None,
                             nodes_filename="opendis_nodes_fe.txt",
                             conn_filename="opendis_tet4.txt",
                             use_gpu=True,
                             one_based_connectivity=True,
                             file_format="npy",
                             float_fmt="%.9e",
                             chunk_rows=2000000,
                             dtype=np.float32,
                             mode="LR+SR"):
        """
        Compute a nodal displacement field from the dislocation network.

        Computes a nodal displacement field U on a regular cell-centered grid
        from the imported OpenDiS dislocation network.

        Args:
            crystal: Crystal object (deprecated, may be removed in future).
            mu: Shear modulus of the material.
            nu: Poisson's ratio (must be in range (0, 0.5)).
            grid_shape: Shape of the output grid as (nx, ny, nz).
                Defaults to (64, 64, 64).
            bounds: Explicit bounds as ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
                If None, uses the dislocation network bounds plus padding.
            padding: Padding to add around the network bounds. Defaults to 0.0.
            core_radius: Physical dislocation core radius (a_phys).
                Defaults to 5.0.
            r_cut: Short-range neighbor radius for LR+SR mode. If None,
                automatically computed from grid spacing. Defaults to None.
            scale: Scale factor for the displacement field. Defaults to 1.0.
            write_directory: Directory for output files. If None, uses
                self.directory. Defaults to None.
            nodes_filename: Filename for node output. Defaults to "opendis_nodes_fe.txt".
            conn_filename: Filename for connectivity output. Defaults to "opendis_tet4.txt".
            use_gpu: If True and CuPy is available, use GPU acceleration.
                Defaults to True.
            one_based_connectivity: If True, connectivity indices start at 1.
                Defaults to True.
            file_format: Output format ("txt", "npy", or "npz"). Defaults to "npy".
            float_fmt: Float format string for text output. Defaults to "%.9e".
            chunk_rows: Number of rows per write chunk. Defaults to 2000000.
            dtype: NumPy dtype for output arrays. Defaults to np.float32.
            mode: Computation mode. Options are:
                - "LR": Spectral long-range only using periodic FFTs and
                    Navier inversion.
                - "SR": Continuum-mechanics only using non-singular displacement
                    gradient line-integral formulation (ExaDiS-style).
                - "LR+SR": Hybrid mode with spectral LR plus analytic near-core
                    correction (default).

        Returns:
            dict: Contains:
                - Xref: Reference node positions as (N, 3) array.
                - U: Displacement field as (N, 3) array.
                - conn: Tet4 connectivity array.
                - nodes_path: Path to written nodes file.
                - conn_path: Path to written connectivity file.

        Raises:
            RuntimeError: If dislocation network has not been imported.
            ValueError: If mode is invalid or nu is out of range.
        """
        # --------------------------
        # Preconditions and inputs
        # --------------------------
        if not hasattr(self, "_opendis_S0"):
            raise RuntimeError("Call import_dislocation_network(...) first.")
        if mode not in ("LR", "SR", "LR+SR"):
            raise ValueError('mode must be one of "LR", "SR", or "LR+SR"')
        if not (0.0 < float(nu) < 0.5):
            raise ValueError("nu must be in (0, 0.5)")

        nu = float(nu)
        mu_ = float(mu)
        lam = 2.0 * mu_ * nu / max(1.0 - 2.0*nu, 1e-12)
        a_phys = float(core_radius)

        out_dir = write_directory if write_directory is not None else (self.directory if self.directory else ".")
        os.makedirs(out_dir, exist_ok=True)

        # --------------------------
        # Bounds and grid (cell-centered)
        # --------------------------
        if bounds is None:
            bmin = np.asarray(self._opendis_bounds["min"], dtype=np.float64)
            bmax = np.asarray(self._opendis_bounds["max"], dtype=np.float64)
            if float(padding) != 0.0:
                pad = float(padding)
                bmin = bmin - pad
                bmax = bmax + pad
            bounds = ((float(bmin[0]), float(bmax[0])),
                    (float(bmin[1]), float(bmax[1])),
                    (float(bmin[2]), float(bmax[2])))
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        nx, ny, nz = [int(v) for v in grid_shape]
        if min(nx, ny, nz) < 2:
            raise ValueError("grid_shape must be >= 2")

        Lx = float(xmax - xmin); Ly = float(ymax - ymin); Lz = float(zmax - zmin)
        dx = Lx / nx; dy = Ly / ny; dz = Lz / nz

        # a_grid for LR spreading; SR-only does not use it
        a_grid = 2.0 * min(dx, dy, dz)

        # default SR neighbor radius for LR+SR; for SR-only we include all segments
        if r_cut is None:
            mean_h = (dx + dy + dz) / 3.0
            r_cut = 2.0 * mean_h
        R_c = float(r_cut)

        # Cell-centered nodes (reference positions)
        xs = np.linspace(xmin + 0.5*dx, xmax - 0.5*dx, nx, dtype=dtype)
        ys = np.linspace(ymin + 0.5*dy, ymax - 0.5*dy, ny, dtype=dtype)
        zs = np.linspace(zmin + 0.5*dz, zmax - 0.5*dz, nz, dtype=dtype)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
        Xref = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1).astype(dtype, copy=False)

        # Simple structured Tet4 connectivity (unchanged)
        def _grid_tet4(nx, ny, nz, one_based=True):
            ex, ey, ez = nx-1, ny-1, nz-1
            elems = []
            def nid(i,j,k, ii,jj,kk):
                return (ii+i)*ny*nz + (jj+j)*nz + (kk+k)
            for ii in range(ex):
                for jj in range(ey):
                    for kk in range(ez):
                        v000 = nid(0,0,0, ii,jj,kk)
                        v100 = nid(1,0,0, ii,jj,kk)
                        v010 = nid(0,1,0, ii,jj,kk)
                        v110 = nid(1,1,0, ii,jj,kk)
                        v001 = nid(0,0,1, ii,jj,kk)
                        v101 = nid(1,0,1, ii,jj,kk)
                        v011 = nid(0,1,1, ii,jj,kk)
                        v111 = nid(1,1,1, ii,jj,kk)
                        elems.extend([
                            [v000, v100, v110, v111],
                            [v000, v110, v010, v111],
                            [v000, v010, v011, v111],
                            [v000, v011, v001, v111],
                            [v000, v001, v101, v111],
                            [v000, v101, v100, v111],
                        ])
            conn = np.asarray(elems, dtype=np.int64)
            if one_based:
                conn = conn + 1
            return conn

        conn = _grid_tet4(nx, ny, nz, one_based=bool(one_based_connectivity))

        # --------------------------
        # Dislocation arrays (NumPy on host)
        # --------------------------
        S0  = np.asarray(self._opendis_S0, dtype=np.float64)
        S1  = np.asarray(self._opendis_S1, dtype=np.float64)
        Bv  = np.asarray(self._opendis_bvec, dtype=np.float64)
        MID = np.asarray(self._opendis_mids, dtype=np.float64)
        HL  = np.asarray(self._opendis_halfL, dtype=np.float64)
        Ns = int(S0.shape[0])
        if Ns == 0:
            raise ValueError("No dislocation segments loaded.")

        # --------------------------
        # GPU availability
        # --------------------------
        gpu_ok = bool(use_gpu and (cp is not None))
        if gpu_ok:
            try:
                _ = cp.cuda.runtime.getDeviceCount()
            except Exception:
                gpu_ok = False

        # --------------------------
        # Helper: spectral LR solver (unchanged)
        # --------------------------
        def _fft_lr(alpha_arr, use_gpu_fft=True, ns_kernel="exp"):
            if use_gpu_fft:
                aa = alpha_arr
                kx = 2.0*np.pi*cp.fft.fftfreq(nx, d=dx)
                ky = 2.0*np.pi*cp.fft.fftfreq(ny, d=dy)
                kz = 2.0*np.pi*cp.fft.fftfreq(nz, d=dz)
                KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing="ij")
                K2 = KX*KX + KY*KY + KZ*KZ
                K = cp.sqrt(K2)
                K2[0,0,0] = 1.0

                Phi = cp.exp(-a_grid * K) if ns_kernel != "helmholtz" else 1.0 / (1.0 + (a_grid*a_grid)*K2)
                Ak = cp.fft.fftn(aa, axes=(0,1,2))
                Ak_ns = Phi[...,None,None] * Ak

                beta_p = cp.zeros_like(Ak_ns, dtype=cp.complex64)
                for i in range(3):
                    arow = Ak_ns[..., i, :]
                    cx = cp.stack([
                        KY*arow[...,2] - KZ*arow[...,1],
                        KZ*arow[...,0] - KX*arow[...,2],
                        KX*arow[...,1] - KY*arow[...,0],
                    ], axis=-1)
                    beta_p[..., i, :] = 1j * cx / K2[...,None]
                beta_p[0,0,0,...] = 0.0

                ep = 0.5*(beta_p + cp.transpose(beta_p, (0,1,2,4,3)))
                tr_ep = ep[...,0,0] + ep[...,1,1] + ep[...,2,2]
                kvec = cp.stack([KX, KY, KZ], axis=-1)

                RHS = cp.zeros((nx,ny,nz,3), dtype=cp.complex64)
                for i in range(3):
                    term1 = lam * kvec[...,i] * tr_ep
                    term2 = 2.0*mu_ * (kvec[...,0]*ep[...,i,0] + kvec[...,1]*ep[...,i,1] + kvec[...,2]*ep[...,i,2])
                    RHS[..., i] = 1j * (term1 + term2)

                denom = mu_ * K2
                cfac = (lam + mu_) / (lam + 2.0*mu_ + 1e-30)
                kk_over_k2 = cp.stack([kvec[...,0]/K2, kvec[...,1]/K2, kvec[...,2]/K2], axis=-1)

                Uhat = cp.zeros_like(RHS)
                for c in range(3):
                    base = RHS[..., c] / denom
                    corr = cfac * ( kk_over_k2[...,0]*RHS[...,0] + kk_over_k2[...,1]*RHS[...,1] + kk_over_k2[...,2]*RHS[...,2] )
                    Uhat[..., c] = base - corr
                Uhat[0,0,0,:] = 0.0

                u_long = cp.real(cp.fft.ifftn(Uhat, axes=(0,1,2))).astype(cp.float32)
                return u_long
            else:
                aa = alpha_arr
                kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
                ky = 2.0*np.pi*np.fft.fftfreq(ny, d=dy)
                kz = 2.0*np.pi*np.fft.fftfreq(nz, d=dz)
                KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
                K2 = KX*KX + KY*KY + KZ*KZ
                K = np.sqrt(K2)
                K2[0,0,0] = 1.0

                Phi = np.exp(-a_grid * K)
                Ak = np.fft.fftn(aa, axes=(0,1,2))
                Ak_ns = Phi[...,None,None] * Ak

                beta_p = np.zeros_like(Ak_ns, dtype=np.complex64)
                for i in range(3):
                    arow = Ak_ns[..., i, :]
                    cx = np.stack([
                        KY*arow[...,2] - KZ*arow[...,1],
                        KZ*arow[...,0] - KX*arow[...,2],
                        KX*arow[...,1] - KY*arow[...,0],
                    ], axis=-1)
                    beta_p[..., i, :] = 1j * cx / K2[...,None]
                beta_p[0,0,0,...] = 0.0

                ep = 0.5*(beta_p + np.transpose(beta_p, (0,1,2,4,3)))
                tr_ep = ep[...,0,0] + ep[...,1,1] + ep[...,2,2]
                kvec = np.stack([KX, KY, KZ], axis=-1)

                RHS = np.zeros((nx,ny,nz,3), dtype=np.complex64)
                for i in range(3):
                    term1 = lam * kvec[...,i] * tr_ep
                    term2 = 2.0*mu_ * (kvec[...,0]*ep[...,i,0] + kvec[...,1]*ep[...,i,1] + kvec[...,2]*ep[...,i,2])
                    RHS[..., i] = 1j * (term1 + term2)

                denom = mu_ * K2
                cfac = (lam + mu_) / (lam + 2.0*mu_ + 1e-30)
                kk_over_k2 = np.stack([kvec[...,0]/K2, kvec[...,1]/K2, kvec[...,2]/K2], axis=-1)
                Uhat = np.zeros_like(RHS)
                for c in range(3):
                    base = RHS[..., c] / denom
                    corr = cfac * ( kk_over_k2[...,0]*RHS[...,0] + kk_over_k2[...,1]*RHS[...,1] + kk_over_k2[...,2]*RHS[...,2] )
                    Uhat[..., c] = base - corr
                Uhat[0,0,0,:] = 0.0

                u_long = np.real(np.fft.ifftn(Uhat, axes=(0,1,2))).astype(np.float32)
                return u_long

        # --------------------------
        # LR branch (alpha deposition + spectral solve), used when mode has LR
        # --------------------------
        lr_needed = (mode in ("LR", "LR+SR"))
        if lr_needed:
            # Deposit Nye tensor alpha to grid (CIC), GPU if available
            if gpu_ok:
                alpha = cp.zeros((nx, ny, nz, 3, 3), dtype=cp.float32)

                cic_kernel_src = r'''
                extern "C" __global__
                void deposit_alpha_cic(
                    const int Ns,
                    const float *s0x, const float *s0y, const float *s0z,
                    const float *s1x, const float *s1y, const float *s1z,
                    const float *bx,  const float *by,  const float *bz,
                    const int nx, const int ny, const int nz,
                    const float xmin, const float ymin, const float zmin,
                    const float dx, const float dy, const float dz,
                    const float oversamp,
                    float *alpha)   // flattened [nx,ny,nz,3,3]
                {
                    int sid = blockDim.x * blockIdx.x + threadIdx.x;
                    if (sid >= Ns) return;

                    float x0 = s0x[sid], y0 = s0y[sid], z0 = s0z[sid];
                    float x1 = s1x[sid], y1 = s1y[sid], z1 = s1z[sid];
                    float tx = x1 - x0, ty = y1 - y0, tz = z1 - z0;
                    float L2 = tx*tx + ty*ty + tz*tz;
                    if (!(L2 > 1.0e-20f)) return;
                    float Linv = rsqrtf(L2);
                    float L = 1.0f / Linv;
                    tx *= Linv; ty *= Linv; tz *= Linv;

                    float btx[3][3];
                    btx[0][0] = bx[sid]*tx; btx[0][1] = bx[sid]*ty; btx[0][2] = bx[sid]*tz;
                    btx[1][0] = by[sid]*tx; btx[1][1] = by[sid]*ty; btx[1][2] = by[sid]*tz;
                    btx[2][0] = bz[sid]*tx; btx[2][1] = bz[sid]*ty; btx[2][2] = bz[sid]*tz;

                    float hmin = fminf(dx, fminf(dy, dz));
                    int nsub = (int)fmaxf(1.0f, oversamp * L / hmin) + 1;
                    float ds = L / (float)nsub;

                    for (int m = 0; m <= nsub; ++m) {
                        float s = (float)m * ds;
                        float xs = x0 + s*tx;
                        float ys = y0 + s*ty;
                        float zs = z0 + s*tz;

                        float qx = (xs - xmin) / dx - 0.5f;
                        float qy = (ys - ymin) / dy - 0.5f;
                        float qz = (zs - zmin) / dz - 0.5f;

                        int i0 = (int)floorf(qx);
                        int j0 = (int)floorf(qy);
                        int k0 = (int)floorf(qz);

                        float fx = qx - (float)i0;
                        float fy = qy - (float)j0;
                        float fz = qz - (float)k0;

                        float wx[2] = {1.0f - fx, fx};
                        float wy[2] = {1.0f - fy, fy};
                        float wz[2] = {1.0f - fz, fz};

                        float wscale = ds / (dx*dy*dz);

                        for (int kk = 0; kk < 2; ++kk) {
                            int k = (k0 + kk) % nz; if (k < 0) k += nz;
                            float wk = wz[kk];
                            for (int jj = 0; jj < 2; ++jj) {
                                int j = (j0 + jj) % ny; if (j < 0) j += ny;
                                float wj = wy[jj];
                                for (int ii = 0; ii < 2; ++ii) {
                                    int i = (i0 + ii) % nx; if (i < 0) i += nx;
                                    float wi = wx[ii];
                                    float w = wscale * wi * wj * wk;

                                    size_t base = (((size_t)i*ny + (size_t)j)*nz + (size_t)k)*9;
                                    for (int p = 0; p < 3; ++p) {
                                        for (int q = 0; q < 3; ++q) {
                                            atomicAdd(&alpha[base + p*3 + q], w * btx[p][q]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }'''.strip("\n")
                mod_alpha = cp.RawModule(code=cic_kernel_src, backend='nvcc',
                                        options=('--gpu-architecture=native','-O3','--use_fast_math'))
                deposit_alpha = mod_alpha.get_function('deposit_alpha_cic')

                s0x = cp.asarray(S0[:,0], dtype=cp.float32); s0y = cp.asarray(S0[:,1], dtype=cp.float32); s0z = cp.asarray(S0[:,2], dtype=cp.float32)
                s1x = cp.asarray(S1[:,0], dtype=cp.float32); s1y = cp.asarray(S1[:,1], dtype=cp.float32); s1z = cp.asarray(S1[:,2], dtype=cp.float32)
                bx  = cp.asarray(Bv[:,0], dtype=cp.float32); by  = cp.asarray(Bv[:,1], dtype=cp.float32); bz  = cp.asarray(Bv[:,2], dtype=cp.float32)

                threads = 256
                blocks  = (Ns + threads - 1) // threads
                deposit_alpha((blocks,), (threads,),
                            (np.int32(Ns),
                            s0x, s0y, s0z, s1x, s1y, s1z, bx, by, bz,
                            np.int32(nx), np.int32(ny), np.int32(nz),
                            np.float32(xmin), np.float32(ymin), np.float32(zmin),
                            np.float32(dx), np.float32(dy), np.float32(dz),
                            np.float32(1.0),
                            alpha))
            else:
                alpha = np.zeros((nx, ny, nz, 3, 3), dtype=np.float32)
                hmin = min(dx, dy, dz)
                for s in range(Ns):
                    p0 = S0[s]; p1 = S1[s]
                    t = p1 - p0
                    L2 = float(np.dot(t, t))
                    if L2 < 1e-20:
                        continue
                    Linv = 1.0/np.sqrt(L2); L = 1.0/Linv; t = t*Linv
                    btx = np.outer(Bv[s].astype(np.float32), t.astype(np.float32))
                    nsub = int(max(1.0, L / hmin)) + 1
                    ds = L / float(nsub)
                    for m in range(nsub+1):
                        sparam = m * ds
                        xs_, ys_, zs_ = (p0 + sparam * t)
                        qx = (xs_ - xmin)/dx - 0.5
                        qy = (ys_ - ymin)/dy - 0.5
                        qz = (zs_ - zmin)/dz - 0.5
                        i0 = int(np.floor(qx)); j0 = int(np.floor(qy)); k0 = int(np.floor(qz))
                        fx = qx - i0; fy = qy - j0; fz = qz - k0
                        wx = np.array([1.0-fx, fx], dtype=np.float32)
                        wy = np.array([1.0-fy, fy], dtype=np.float32)
                        wz = np.array([1.0-fz, fz], dtype=np.float32)
                        wscale = ds / (dx*dy*dz)
                        for kk in range(2):
                            k = (k0+kk) % nz; wk = wz[kk]
                            for jj in range(2):
                                j = (j0+jj) % ny; wj = wy[jj]
                                for ii in range(2):
                                    i = (i0+ii) % nx; wi = wx[ii]
                                    w = wscale * wi * wj * wk
                                    alpha[i,j,k,...] += w * btx

            # Spectral LR solve
            u_long = _fft_lr(alpha, use_gpu_fft=gpu_ok, ns_kernel="exp")

        # --------------------------
        # SR branch
        # --------------------------
        sr_needed = (mode in ("SR", "LR+SR"))
        U = None  # will be set in next branches

        if sr_needed and mode == "LR+SR":
            # Existing LR+SR: analytic delta dudx [a_phys - a_grid] near segments, then integrate to Ucorr, add to LR.
            if gpu_ok:
                sr_kernel_src = r'''
                extern "C" __global__
                void sr_accumulate_diff(
                    const int Npts,
                    const float *px, const float *py, const float *pz,
                    const int Ns,
                    const float *s0x, const float *s0y, const float *s0z,
                    const float *s1x, const float *s1y, const float *s1z,
                    const float *bx,  const float *by,  const float *bz,
                    const float *mid_x, const float *mid_y, const float *mid_z,
                    const float *halfL, const float Rc,
                    const float a_phys, const float a_grid, const float nu,
                    float *G00, float *G01, float *G02,
                    float *G10, float *G11, float *G12,
                    float *G20, float *G21, float *G22)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    if (i >= Npts) return;

                    float x = px[i], y = py[i], z = pz[i];

                    float g00=0.f,g01=0.f,g02=0.f;
                    float g10=0.f,g11=0.f,g12=0.f;
                    float g20=0.f,g21=0.f,g22=0.f;

                    const float m8pi = -0.125f / 3.14159265358979323846f;
                    const float m8pinu = m8pi / (1.0f - nu);

                    for (int s=0; s<Ns; ++s) {
                        float dxm = x - mid_x[s];
                        float dym = y - mid_y[s];
                        float dzm = z - mid_z[s];
                        float rad = Rc + halfL[s];
                        if (dxm*dxm + dym*dym + dzm*dzm > rad*rad) continue;

                        float sx = s0x[s], sy = s0y[s], sz = s0z[s];
                        float ex = s1x[s], ey = s1y[s], ez = s1z[s];
                        float tx = ex - sx, ty = ey - sy, tz = ez - sz;
                        float L2 = tx*tx + ty*ty + tz*tz;
                        if (!(L2 > 1.0e-20f)) continue;
                        float Linv = rsqrtf(L2);
                        float L = 1.0f / Linv;
                        tx*=Linv; ty*=Linv; tz*=Linv;

                        float Rx = x - sx, Ry = y - sy, Rz = z - sz;
                        float Rdt = Rx*tx + Ry*ty + Rz*tz;

                        float p0x = sx + Rdt*tx;
                        float p0y = sy + Rdt*ty;
                        float p0z = sz + Rdt*tz;

                        float dx = Rx - (p0x - sx);
                        float dy = Ry - (p0y - sy);
                        float dz = Rz - (p0z - sz);
                        float s1p = -Rdt;
                        float s2p =  L - Rdt;

                        float bxv = bx[s]; float byv = by[s]; float bzv = bz[s];

                        auto compute_d = [&] (float a2,
                                            float &d00,float &d01,float &d02,
                                            float &d10,float &d11,float &d12,
                                            float &d20,float &d21,float &d22) {
                            float d2   = dx*dx + dy*dy + dz*dz;
                            float da2  = d2 + a2;
                            float da2inv = 1.0f / da2;

                            float Ra1 = sqrtf(s1p*s1p + da2);
                            float Ra2 = sqrtf(s2p*s2p + da2);
                            float Ra1inv = 1.0f / fmaxf(Ra1, 1.0e-38f);
                            float Ra2inv = 1.0f / fmaxf(Ra2, 1.0e-38f);
                            float Ra1inv3 = Ra1inv*Ra1inv*Ra1inv;
                            float Ra2inv3 = Ra2inv*Ra2inv*Ra2inv;

                            float J03 = da2inv*(s2p*Ra2inv - s1p*Ra1inv);
                            float J13 = -Ra2inv + Ra1inv;
                            float J15 = -(1.0f/3.0f)*(Ra2inv3 - Ra1inv3);
                            float J25 = (1.0f/3.0f)*da2inv*(s2p*s2p*s2p*Ra2inv3 - s1p*s1p*s1p*Ra1inv3);
                            float J05 = da2inv*(2.0f*J25 + s2p*Ra2inv3 - s1p*Ra1inv3);
                            float J35 = 2.0f*da2*J15 - s2p*s2p*Ra2inv3 + s1p*s1p*Ra1inv3;

                            float A0 = 3.0f*a2*(dx*J05 - tx*J15) + 2.0f*(dx*J03 - tx*J13);
                            float A1 = 3.0f*a2*(dy*J05 - ty*J15) + 2.0f*(dy*J03 - ty*J13);
                            float A2 = 3.0f*a2*(dz*J05 - tz*J15) + 2.0f*(dz*J03 - tz*J13);

                            float U1_00 = (-A2*ty*bxv) - (-A1*tz*bxv);
                            float U1_01 = (-A0*tz*byv) - (-A2*tx*byv);
                            float U1_02 = (-A1*tx*bzv) - (-A0*ty*bzv);

                            float U2_00 = (-A0*tz*byv) - (-A0*ty*bzv);
                            float U2_11 = (-A1*tx*bzv) - (-A1*tz*bxv);
                            float U2_22 = (-A2*ty*bxv) - (-A2*tx*byv);

                            float t0=tx, t1=ty, t2=tz;
                            float d0=dx, d1=dy, d2c=dz;

                            float B111 = -3.f*d0*J03 + 3.f*t0*J13 + 3.f*d0*d0*d0*J05 - 9.f*(d0*d0*t0)*J15 + 9.f*(d0*t0*t0)*J25 - 3.f*t0*t0*t0*J35;
                            float B222 = -3.f*d1*J03 + 3.f*t1*J13 + 3.f*d1*d1*d1*J05 - 9.f*(d1*d1*t1)*J15 + 9.f*(d1*t1*t1)*J25 - 3.f*t1*t1*t1*J35;
                            float B333 = -3.f*d2c*J03 + 3.f*t2*J13 + 3.f*d2c*d2c*d2c*J05 - 9.f*(d2c*d2c*t2)*J15 + 9.f*(d2c*t2*t2)*J25 - 3.f*t2*t2*t2*J35;

                            float B112 = -d1*J03 + t1*J13 + 3.f*d0*d0*d1*J05
                                    -3.f*(d0*d0*t1 + d0*t0*d1 + t0*d0*d1)*J15
                                    +3.f*(t0*t0*d1 + t0*d0*t1 + d0*t0*t1)*J25
                                    -3.f*t0*t0*t1*J35;

                            float B113 = -d2c*J03 + t2*J13 + 3.f*d0*d0*d2c*J05
                                    -3.f*(d0*d0*t2 + d0*t0*d2c + t0*d0*d2c)*J15
                                    +3.f*(t0*t0*d2c + t0*d0*t2 + d0*t0*t2)*J25
                                    -3.f*t0*t0*t2*J35;

                            float B221 = -d0*J03 + t0*J13 + 3.f*d1*d1*d0*J05
                                    -3.f*(d1*d1*t0 + d1*t1*d0 + t1*d1*d0)*J15
                                    +3.f*(t1*t1*d0 + t1*d1*t0 + d1*t1*t0)*J25
                                    -3.f*t1*t1*t0*J35;

                            float B223 = -d2c*J03 + t2*J13 + 3.f*d1*d1*d2c*J05
                                    -3.f*(d1*d1*t2 + d1*t1*d2c + t1*d1*d2c)*J15
                                    +3.f*(t1*t1*d2c + t1*d1*t2 + d1*t1*t2)*J25
                                    -3.f*t1*t1*t2*J35;

                            float B331 = -d0*J03 + t0*J13 + 3.f*d2c*d2c*d0*J05
                                    -3.f*(d2c*d2c*t0 + d2c*t2*d0 + t2*d2c*d0)*J15
                                    +3.f*(t2*t2*d0 + t2*d2c*t0 + d2c*t2*t0)*J25
                                    -3.f*t2*t2*t0*J35;

                            float B332 = -d1*J03 + t1*J13 + 3.f*d2c*d2c*d1*J05
                                    -3.f*(d2c*d2c*t1 + d2c*t2*d1 + t2*d2c*d1)*J15
                                    +3.f*(t2*t2*d1 + t2*d2c*t1 + d2c*t2*t1)*J25
                                    -3.f*t2*t2*t1*J35;

                            float B123 =  3.f*d0*d1*d2c*J05
                                    -3.f*(d0*d1*t2 + d0*t1*d2c + t0*d1*d2c)*J15
                                    +3.f*(t0*t1*d2c + t0*d1*t2 + d0*t1*t2)*J25
                                    -3.f*t0*t1*t2*J35;

                            float U3_00 = (B112*t2 - B113*t1)*bxv + (B113*t0 - B111*t2)*byv + (B111*t1 - B112*t0)*bzv;
                            float U3_11 = (B222*t2 - B223*t1)*bxv + (B223*t0 - B221*t2)*byv + (B221*t1 - B222*t0)*bzv;
                            float U3_22 = (B332*t2 - B333*t1)*bxv + (B333*t0 - B331*t2)*byv + (B331*t1 - B332*t0)*bzv;
                            float U3_01 = (B221*t2 - B123*t1)*bxv + (B123*t0 - B112*t2)*byv + (B112*t1 - B221*t0)*bzv;
                            float U3_02 = (B123*t2 - B331*t1)*bxv + (B331*t0 - B113*t2)*byv + (B113*t1 - B123*t0)*bzv;
                            float U3_12 = (B223*t2 - B332*t1)*bxv + (B332*t0 - B123*t2)*byv + (B123*t1 - B223*t0)*bzv;

                            d00 = m8pi*(U1_00 + U2_00) + m8pinu*U3_00;
                            d01 = m8pi*(U1_01 + 0.0f)  + m8pinu*U3_01;
                            d02 = m8pi*(U1_02 + 0.0f)  + m8pinu*U3_02;

                            d10 = m8pinu*U3_01;
                            d11 = m8pi*(0.0f + U2_11) + m8pinu*U3_11;
                            d12 = m8pinu*U3_12;

                            d20 = m8pinu*U3_02;
                            d21 = m8pinu*U3_12;
                            d22 = m8pi*(0.0f + U2_22) + m8pinu*U3_22;
                        };

                        float d00p,d01p,d02p,d10p,d11p,d12p,d20p,d21p,d22p;
                        float d00g,d01g,d02g,d10g,d11g,d12g,d20g,d21g,d22g;

                        compute_d(a_phys*a_phys, d00p,d01p,d02p,d10p,d11p,d12p,d20p,d21p,d22p);
                        compute_d(a_grid*a_grid, d00g,d01g,d02g,d10g,d11g,d12g,d20g,d21g,d22g);

                        g00 += (d00p - d00g); g01 += (d01p - d01g); g02 += (d02p - d02g);
                        g10 += (d10p - d10g); g11 += (d11p - d11g); g12 += (d12p - d12g);
                        g20 += (d20p - d20g); g21 += (d21p - d21g); g22 += (d22p - d22g);
                    }

                    G00[i]=g00; G01[i]=g01; G02[i]=g02;
                    G10[i]=g10; G11[i]=g11; G12[i]=g12;
                    G20[i]=g20; G21[i]=g21; G22[i]=g22;
                }'''.strip("\n")
                mod_sr = cp.RawModule(code=sr_kernel_src, backend='nvcc',
                                    options=('--gpu-architecture=native','-O3','--use_fast_math'))
                sr_accum = mod_sr.get_function('sr_accumulate_diff')

                px = cp.asarray(Xref[:,0], dtype=cp.float32)
                py = cp.asarray(Xref[:,1], dtype=cp.float32)
                pz = cp.asarray(Xref[:,2], dtype=cp.float32)

                s0x = cp.asarray(S0[:,0], dtype=cp.float32); s0y = cp.asarray(S0[:,1], dtype=cp.float32); s0z = cp.asarray(S0[:,2], dtype=cp.float32)
                s1x = cp.asarray(S1[:,0], dtype=cp.float32); s1y = cp.asarray(S1[:,1], dtype=cp.float32); s1z = cp.asarray(S1[:,2], dtype=cp.float32)
                bx  = cp.asarray(Bv[:,0], dtype=cp.float32); by  = cp.asarray(Bv[:,1], dtype=cp.float32); bz  = cp.asarray(Bv[:,2], dtype=cp.float32)
                mx  = cp.asarray(MID[:,0], dtype=cp.float32); my  = cp.asarray(MID[:,1], dtype=cp.float32); mz  = cp.asarray(MID[:,2], dtype=cp.float32)
                hl  = cp.asarray(HL,        dtype=cp.float32)

                G00 = cp.zeros(px.size, dtype=cp.float32); G01 = cp.zeros_like(G00); G02 = cp.zeros_like(G00)
                G10 = cp.zeros_like(G00); G11 = cp.zeros_like(G00); G12 = cp.zeros_like(G00)
                G20 = cp.zeros_like(G00); G21 = cp.zeros_like(G00); G22 = cp.zeros_like(G00)

                threads = 256
                blocks  = (px.size + threads - 1) // threads
                sr_accum((blocks,), (threads,),
                        (np.int32(px.size),
                        px, py, pz,
                        np.int32(Ns),
                        s0x, s0y, s0z, s1x, s1y, s1z,
                        bx,  by,  bz,
                        mx,  my,  mz,
                        hl,  cp.float32(R_c),
                        cp.float32(a_phys), cp.float32(a_grid), cp.float32(nu),
                        G00, G01, G02, G10, G11, G12, G20, G21, G22))

                deltaG = cp.stack([
                            cp.stack([G00,G01,G02], axis=1),
                            cp.stack([G10,G11,G12], axis=1),
                            cp.stack([G20,G21,G22], axis=1)
                        ], axis=1).transpose(0,2,1)  # (N,3,3)
                deltaG = deltaG.reshape(nx,ny,nz,3,3)

                # Integrate deltaG to Ucorr in Fourier domain
                kx = 2.0*np.pi*cp.fft.fftfreq(nx, d=dx)
                ky = 2.0*np.pi*cp.fft.fftfreq(ny, d=dy)
                kz = 2.0*np.pi*cp.fft.fftfreq(nz, d=dz)
                KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing="ij")
                K2 = KX*KX + KY*KY + KZ*KZ
                K2[0,0,0] = 1.0

                Ucorr = cp.zeros((nx,ny,nz,3), dtype=cp.float32)
                for i in range(3):
                    Gk0 = cp.fft.fftn(deltaG[..., i, 0]); Gk1 = cp.fft.fftn(deltaG[..., i, 1]); Gk2 = cp.fft.fftn(deltaG[..., i, 2])
                    Uk  = (1j*(KX*Gk0 + KY*Gk1 + KZ*Gk2)) / K2
                    Uk[0,0,0] = 0.0
                    Ucorr[..., i] = cp.real(cp.fft.ifftn(Uk)).astype(cp.float32)

                U = (u_long + Ucorr).reshape(nx*ny*nz, 3).get().astype(dtype, copy=False)
            else:
                # CPU LR+SR delta dudx
                px = Xref[:,0].astype(np.float64); py = Xref[:,1].astype(np.float64); pz = Xref[:,2].astype(np.float64)
                Gsum = np.zeros((px.size,3,3), dtype=np.float64)

                def _dudx_seg(px, py, pz, p1, p2, b, a, nu):
                    t = p2 - p1
                    L2 = float(np.dot(t,t))
                    if L2 < 1e-20:
                        z = np.zeros_like(px)
                        return (z,z,z,z,z,z,z,z,z)
                    Linv = 1.0/np.sqrt(L2); L = 1.0/Linv; t = t*Linv
                    Rx = px - p1[0]; Ry = py - p1[1]; Rz = pz - p1[2]
                    Rdt = Rx*t[0] + Ry*t[1] + Rz*t[2]
                    p0x = p1[0] + Rdt*t[0]; p0y = p1[1] + Rdt*t[1]; p0z = p1[2] + Rdt*t[2]
                    dx = Rx - (p0x - p1[0]); dy = Ry - (p0y - p1[1]); dz = Rz - (p0z - p1[2])
                    s1 = -Rdt; s2 = L - Rdt

                    a2  = a*a
                    d2  = dx*dx + dy*dy + dz*dz
                    da2 = d2 + a2
                    da2inv = 1.0/da2
                    Ra1 = np.sqrt(s1*s1 + da2); Ra2 = np.sqrt(s2*s2 + da2)
                    Ra1inv = 1.0/np.maximum(Ra1,1e-38); Ra2inv = 1.0/np.maximum(Ra2,1e-38)
                    Ra1inv3 = Ra1inv*Ra1inv*Ra1inv; Ra2inv3 = Ra2inv*Ra2inv*Ra2inv

                    J03 = da2inv*(s2*Ra2inv - s1*Ra1inv)
                    J13 = -Ra2inv + Ra1inv
                    J15 = -(1.0/3.0)*(Ra2inv3 - Ra1inv3)
                    J25 = (1.0/3.0)*da2inv*(s2*s2*s2*Ra2inv3 - s1*s1*s1*Ra1inv3)
                    J05 = da2inv*(2.0*J25 + s2*Ra2inv3 - s1*Ra1inv3)
                    J35 = 2.0*da2*J15 - s2*s2*Ra2inv3 + s1*s1*Ra1inv3

                    A0 = 3.0*a2*(dx*J05 - t[0]*J15) + 2.0*(dx*J03 - t[0]*J13)
                    A1 = 3.0*a2*(dy*J05 - t[1]*J15) + 2.0*(dy*J03 - t[1]*J13)
                    A2 = 3.0*a2*(dz*J05 - t[2]*J15) + 2.0*(dz*J03 - t[2]*J13)

                    U1_00 = (-A2*t[1]*b[0]) - (-A1*t[2]*b[0])
                    U1_01 = (-A0*t[2]*b[1]) - (-A2*t[0]*b[1])
                    U1_02 = (-A1*t[0]*b[2]) - (-A0*t[1]*b[2])

                    U2_00 = (-A0*t[2]*b[1]) - (-A0*t[1]*b[2])
                    U2_11 = (-A1*t[0]*b[2]) - (-A1*t[2]*b[0])
                    U2_22 = (-A2*t[1]*b[0]) - (-A2*t[0]*b[1])

                    t0,t1,t2 = t[0],t[1],t[2]; d0,d1,d2c = dx,dy,dz
                    B111 = -3.0*d0*J03 + 3.0*t0*J13 + 3.0*d0*d0*d0*J05 - 9.0*(d0*d0*t0)*J15 + 9.0*(d0*t0*t0)*J25 - 3.0*t0*t0*t0*J35
                    B222 = -3.0*d1*J03 + 3.0*t1*J13 + 3.0*d1*d1*d1*J05 - 9.0*(d1*d1*t1)*J15 + 9.0*(d1*t1*t1)*J25 - 3.0*t1*t1*t1*J35
                    B333 = -3.0*d2c*J03 + 3.0*t2*J13 + 3.0*d2c*d2c*d2c*J05 - 9.0*(d2c*d2c*t2)*J15 + 9.0*(d2c*t2*t2)*J25 - 3.0*t2*t2*t2*J35
                    B112 = -d1*J03 + t1*J13 + 3.0*d0*d0*d1*J05 -3.0*(d0*d0*t1 + d0*t0*d1 + t0*d0*d1)*J15 +3.0*(t0*t0*d1 + t0*d0*t1 + d0*t0*t1)*J25 -3.0*t0*t0*t1*J35
                    B113 = -d2c*J03 + t2*J13 + 3.0*d0*d0*d2c*J05 -3.0*(d0*d0*t2 + d0*t0*d2c + t0*d0*d2c)*J15 +3.0*(t0*t0*d2c + t0*d0*t2 + d0*t0*t2)*J25 -3.0*t0*t0*t2*J35
                    B221 = -d0*J03 + t0*J13 + 3.0*d1*d1*d0*J05 -3.0*(d1*d1*t0 + d1*t1*d0 + t1*d1*d0)*J15 +3.0*(t1*t1*d0 + t1*d1*t0 + d1*t1*t0)*J25 -3.0*t1*t1*t0*J35
                    B223 = -d2c*J03 + t2*J13 + 3.0*d1*d1*d2c*J05 -3.0*(d1*d1*t2 + d1*t1*d2c + t1*d1*d2c)*J15 +3.0*(t1*t1*d2c + t1*d1*t2 + d1*t1*t2)*J25 -3.0*t1*t1*t2*J35
                    B331 = -d0*J03 + t0*J13 + 3.0*d2c*d2c*d0*J05 -3.0*(d2c*d2c*t0 + d2c*t2*d0 + t2*d2c*d0)*J15 +3.0*(t2*t2*d0 + t2*d2c*t0 + d2c*t2*t0)*J25 -3.0*t2*t2*t0*J35
                    B332 = -d1*J03 + t1*J13 + 3.0*d2c*d2c*d1*J05 -3.0*(d2c*d2c*t1 + d2c*t2*d1 + t2*d2c*d1)*J15 +3.0*(t2*t2*d1 + t2*d2c*t1 + d2c*t2*t1)*J25 -3.0*t2*t2*t1*J35
                    B123 =  3.0*d0*d1*d2c*J05 -3.0*(d0*d1*t2 + d0*t1*d2c + t0*d1*d2c)*J15 +3.0*(t0*t1*d2c + t0*d1*t2 + d0*t1*t2)*J25 -3.0*t0*t1*t2*J35

                    U3_00 = (B112*t2 - B113*t1)*b[0] + (B113*t0 - B111*t2)*b[1] + (B111*t1 - B112*t0)*b[2]
                    U3_11 = (B222*t2 - B223*t1)*b[0] + (B223*t0 - B221*t2)*b[1] + (B221*t1 - B222*t0)*b[2]
                    U3_22 = (B332*t2 - B333*t1)*b[0] + (B333*t0 - B331*t2)*b[1] + (B331*t1 - B332*t0)*b[2]
                    U3_01 = (B221*t2 - B123*t1)*b[0] + (B123*t0 - B112*t2)*b[1] + (B112*t1 - B221*t0)*b[2]
                    U3_02 = (B123*t2 - B331*t1)*b[0] + (B331*t0 - B113*t2)*b[1] + (B113*t1 - B123*t0)*b[2]
                    U3_12 = (B223*t2 - B332*t1)*b[0] + (B332*t0 - B123*t2)*b[1] + (B123*t1 - B223*t0)*b[2]

                    m8pi   = -0.125/np.pi
                    m8pinu = m8pi/(1.0 - nu)
                    d00 = m8pi*(U1_00 + U2_00) + m8pinu*U3_00
                    d01 = m8pi*(U1_01 + 0.0)   + m8pinu*U3_01
                    d02 = m8pi*(U1_02 + 0.0)   + m8pinu*U3_02
                    d10 = m8pinu*U3_01
                    d11 = m8pi*(0.0 + U2_11) + m8pinu*U3_11
                    d12 = m8pinu*U3_12
                    d20 = m8pinu*U3_02
                    d21 = m8pinu*U3_12
                    d22 = m8pi*(0.0 + U2_22) + m8pinu*U3_22
                    return (d00,d01,d02,d10,d11,d12,d20,d21,d22)

                for s in range(Ns):
                    p0 = S0[s]; p1 = S1[s]; b = Bv[s]
                    dxm = px - MID[s,0]; dym = py - MID[s,1]; dzm = pz - MID[s,2]
                    rad = R_c + HL[s]
                    msk = (dxm*dxm + dym*dym + dzm*dzm) <= (rad*rad)
                    if not np.any(msk):
                        continue
                    d00p,d01p,d02p,d10p,d11p,d12p,d20p,d21p,d22p = _dudx_seg(px[msk], py[msk], pz[msk], p0, p1, b, a_phys, nu)
                    d00g,d01g,d02g,d10g,d11g,d12g,d20g,d21g,d22g = _dudx_seg(px[msk], py[msk], pz[msk], p0, p1, b, a_grid, nu)
                    Gsum[msk,0,0] += (d00p-d00g); Gsum[msk,0,1] += (d01p-d01g); Gsum[msk,0,2] += (d02p-d02g)
                    Gsum[msk,1,0] += (d10p-d10g); Gsum[msk,1,1] += (d11p-d11g); Gsum[msk,1,2] += (d12p-d12g)
                    Gsum[msk,2,0] += (d20p-d20g); Gsum[msk,2,1] += (d21p-d21g); Gsum[msk,2,2] += (d22p-d22g)

                deltaG = Gsum.reshape(nx,ny,nz,3,3)

                # Integrate deltaG to Ucorr
                kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
                ky = 2.0*np.pi*np.fft.fftfreq(ny, d=dy)
                kz = 2.0*np.pi*np.fft.fftfreq(nz, d=dz)
                KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
                K2 = KX*KX + KY*KY + KZ*KZ
                K2[0,0,0] = 1.0

                Ucorr = np.zeros((nx,ny,nz,3), dtype=np.float32)
                for i in range(3):
                    Gk0 = np.fft.fftn(deltaG[..., i, 0]); Gk1 = np.fft.fftn(deltaG[..., i, 1]); Gk2 = np.fft.fftn(deltaG[..., i, 2])
                    Uk  = (1j*(KX*Gk0 + KY*Gk1 + KZ*Gk2)) / K2
                    Uk[0,0,0] = 0.0
                    Ucorr[..., i] = np.real(np.fft.ifftn(Uk)).astype(np.float32)

                U = (u_long + Ucorr).reshape(nx*ny*nz, 3).astype(dtype, copy=False)

        elif sr_needed and mode == "SR":
            # New SR-only (ExaDiS-style dudx): compute full non-singular dudx(a_phys), integrate to U
            if gpu_ok:
                sr_full_src = r'''
                extern "C" __global__
                void sr_accumulate_full(
                    const int Npts,
                    const float *px, const float *py, const float *pz,
                    const int Ns,
                    const float *s0x, const float *s0y, const float *s0z,
                    const float *s1x, const float *s1y, const float *s1z,
                    const float *bx,  const float *by,  const float *bz,
                    const float *mid_x, const float *mid_y, const float *mid_z,
                    const float *halfL,
                    const float Rc_effective,   // if very large, effectively includes all segments
                    const float a_phys, const float nu,
                    float *G00, float *G01, float *G02,
                    float *G10, float *G11, float *G12,
                    float *G20, float *G21, float *G22)
                {
                    int i = blockDim.x * blockIdx.x + threadIdx.x;
                    if (i >= Npts) return;

                    float x = px[i], y = py[i], z = pz[i];

                    float g00=0.f,g01=0.f,g02=0.f;
                    float g10=0.f,g11=0.f,g12=0.f;
                    float g20=0.f,g21=0.f,g22=0.f;

                    const float m8pi   = -0.125f / 3.14159265358979323846f;
                    const float m8pinu = m8pi / (1.0f - nu);
                    const float a2 = a_phys * a_phys;

                    for (int s=0; s<Ns; ++s) {
                        // optional midpoint culling; if Rc_effective is very large, this test passes for all
                        float dxm = x - mid_x[s];
                        float dym = y - mid_y[s];
                        float dzm = z - mid_z[s];
                        float rad = Rc_effective + halfL[s];
                        if (dxm*dxm + dym*dym + dzm*dzm > rad*rad) continue;

                        float sx = s0x[s], sy = s0y[s], sz = s0z[s];
                        float ex = s1x[s], ey = s1y[s], ez = s1z[s];
                        float tx = ex - sx, ty = ey - sy, tz = ez - sz;
                        float L2 = tx*tx + ty*ty + tz*tz;
                        if (!(L2 > 1.0e-20f)) continue;
                        float Linv = rsqrtf(L2);
                        float L = 1.0f / Linv;
                        tx*=Linv; ty*=Linv; tz*=Linv;

                        float Rx = x - sx, Ry = y - sy, Rz = z - sz;
                        float Rdt = Rx*tx + Ry*ty + Rz*tz;

                        float p0x = sx + Rdt*tx;
                        float p0y = sy + Rdt*ty;
                        float p0z = sz + Rdt*tz;

                        float dx = Rx - (p0x - sx);
                        float dy = Ry - (p0y - sy);
                        float dz = Rz - (p0z - sz);
                        float s1p = -Rdt;
                        float s2p =  L - Rdt;

                        float bxv = bx[s]; float byv = by[s]; float bzv = bz[s];

                        float d2   = dx*dx + dy*dy + dz*dz;
                        float da2  = d2 + a2;
                        float da2inv = 1.0f / da2;

                        float Ra1 = sqrtf(s1p*s1p + da2);
                        float Ra2 = sqrtf(s2p*s2p + da2);
                        float Ra1inv = 1.0f / fmaxf(Ra1, 1.0e-38f);
                        float Ra2inv = 1.0f / fmaxf(Ra2, 1.0e-38f);
                        float Ra1inv3 = Ra1inv*Ra1inv*Ra1inv;
                        float Ra2inv3 = Ra2inv*Ra2inv*Ra2inv;

                        float J03 = da2inv*(s2p*Ra2inv - s1p*Ra1inv);
                        float J13 = -Ra2inv + Ra1inv;
                        float J15 = -(1.0f/3.0f)*(Ra2inv3 - Ra1inv3);
                        float J25 = (1.0f/3.0f)*da2inv*(s2p*s2p*s2p*Ra2inv3 - s1p*s1p*s1p*Ra1inv3);
                        float J05 = da2inv*(2.0f*J25 + s2p*Ra2inv3 - s1p*Ra1inv3);
                        float J35 = 2.0f*da2*J15 - s2p*s2p*Ra2inv3 + s1p*s1p*Ra1inv3;

                        float A0 = 3.0f*a2*(dx*J05 - tx*J15) + 2.0f*(dx*J03 - tx*J13);
                        float A1 = 3.0f*a2*(dy*J05 - ty*J15) + 2.0f*(dy*J03 - ty*J13);
                        float A2 = 3.0f*a2*(dz*J05 - tz*J15) + 2.0f*(dz*J03 - tz*J13);

                        float U1_00 = (-A2*ty*bxv) - (-A1*tz*bxv);
                        float U1_01 = (-A0*tz*byv) - (-A2*tx*byv);
                        float U1_02 = (-A1*tx*bzv) - (-A0*ty*bzv);

                        float U2_00 = (-A0*tz*byv) - (-A0*ty*bzv);
                        float U2_11 = (-A1*tx*bzv) - (-A1*tz*bxv);
                        float U2_22 = (-A2*ty*bxv) - (-A2*tx*byv);

                        float t0=tx, t1=ty, t2=tz;
                        float d0=dx, d1=dy, d2c=dz;

                        float B111 = -3.f*d0*J03 + 3.f*t0*J13 + 3.f*d0*d0*d0*J05 - 9.f*(d0*d0*t0)*J15 + 9.f*(d0*t0*t0)*J25 - 3.f*t0*t0*t0*J35;
                        float B222 = -3.f*d1*J03 + 3.f*t1*J13 + 3.f*d1*d1*d1*J05 - 9.f*(d1*d1*t1)*J15 + 9.f*(d1*t1*t1)*J25 - 3.f*t1*t1*t1*J35;
                        float B333 = -3.f*d2c*J03 + 3.f*t2*J13 + 3.f*d2c*d2c*d2c*J05 - 9.f*(d2c*d2c*t2)*J15 + 9.f*(d2c*t2*t2)*J25 - 3.f*t2*t2*t2*J35;

                        float B112 = -d1*J03 + t1*J13 + 3.f*d0*d0*d1*J05
                                -3.f*(d0*d0*t1 + d0*t0*d1 + t0*d0*d1)*J15
                                +3.f*(t0*t0*d1 + t0*d0*t1 + d0*t0*t1)*J25
                                -3.f*t0*t0*t1*J35;

                        float B113 = -d2c*J03 + t2*J13 + 3.f*d0*d0*d2c*J05
                                -3.f*(d0*d0*t2 + d0*t0*d2c + t0*d0*d2c)*J15
                                +3.f*(t0*t0*d2c + t0*d0*t2 + d0*t0*t2)*J25
                                -3.f*t0*t0*t2*J35;

                        float B221 = -d0*J03 + t0*J13 + 3.f*d1*d1*d0*J05
                                -3.f*(d1*d1*t0 + d1*t1*d0 + t1*d1*d0)*J15
                                +3.f*(t1*t1*d0 + t1*d1*t0 + d1*t1*t0)*J25
                                -3.f*t1*t1*t0*J35;

                        float B223 = -d2c*J03 + t2*J13 + 3.f*d1*d1*d2c*J05
                                -3.f*(d1*d1*t2 + d1*t1*d2c + t1*d1*d2c)*J15
                                +3.f*(t1*t1*d2c + t1*d1*t2 + d1*t1*t2)*J25
                                -3.f*t1*t1*t2*J35;

                        float B331 = -d0*J03 + t0*J13 + 3.f*d2c*d2c*d0*J05
                                -3.f*(d2c*d2c*t0 + d2c*t2*d0 + t2*d2c*d0)*J15
                                +3.f*(t2*t2*d0 + t2*d2c*t0 + d2c*t2*t0)*J25
                                -3.f*t2*t2*t0*J35;

                        float B332 = -d1*J03 + t1*J13 + 3.f*d2c*d2c*d1*J05
                                -3.f*(d2c*d2c*t1 + d2c*t2*d1 + t2*d2c*d1)*J15
                                +3.f*(t2*t2*d1 + t2*d2c*t1 + d2c*t2*t1)*J25
                                -3.f*t2*t2*t1*J35;

                        float B123 =  3.f*d0*d1*d2c*J05
                                -3.f*(d0*d1*t2 + d0*t1*d2c + t0*d1*d2c)*J15
                                +3.f*(t0*t1*d2c + t0*d1*t2 + d0*t1*t2)*J25
                                -3.f*t0*t1*t2*J35;

                        float U3_00 = (B112*t2 - B113*t1)*bxv + (B113*t0 - B111*t2)*byv + (B111*t1 - B112*t0)*bzv;
                        float U3_11 = (B222*t2 - B223*t1)*bxv + (B223*t0 - B221*t2)*byv + (B221*t1 - B222*t0)*bzv;
                        float U3_22 = (B332*t2 - B333*t1)*bxv + (B333*t0 - B331*t2)*byv + (B331*t1 - B332*t0)*bzv;
                        float U3_01 = (B221*t2 - B123*t1)*bxv + (B123*t0 - B112*t2)*byv + (B112*t1 - B221*t0)*bzv;
                        float U3_02 = (B123*t2 - B331*t1)*bxv + (B331*t0 - B113*t2)*byv + (B113*t1 - B123*t0)*bzv;
                        float U3_12 = (B223*t2 - B332*t1)*bxv + (B332*t0 - B123*t2)*byv + (B123*t1 - B223*t0)*bzv;

                        float d00 = m8pi*(U1_00 + U2_00) + m8pinu*U3_00;
                        float d01 = m8pi*(U1_01 + 0.0f)  + m8pinu*U3_01;
                        float d02 = m8pi*(U1_02 + 0.0f)  + m8pinu*U3_02;

                        float d10 = m8pinu*U3_01;
                        float d11 = m8pi*(0.0f + U2_11) + m8pinu*U3_11;
                        float d12 = m8pinu*U3_12;

                        float d20 = m8pinu*U3_02;
                        float d21 = m8pinu*U3_12;
                        float d22 = m8pi*(0.0f + U2_22) + m8pinu*U3_22;

                        g00 += d00; g01 += d01; g02 += d02;
                        g10 += d10; g11 += d11; g12 += d12;
                        g20 += d20; g21 += d21; g22 += d22;
                    }

                    G00[i]=g00; G01[i]=g01; G02[i]=g02;
                    G10[i]=g10; G11[i]=g11; G12[i]=g12;
                    G20[i]=g20; G21[i]=g21; G22[i]=g22;
                }'''.strip("\n")

                mod_full = cp.RawModule(code=sr_full_src, backend='nvcc',
                                        options=('--gpu-architecture=native','-O3','--use_fast_math'))
                sr_full = mod_full.get_function('sr_accumulate_full')

                # Inputs to GPU
                px = cp.asarray(Xref[:,0], dtype=cp.float32)
                py = cp.asarray(Xref[:,1], dtype=cp.float32)
                pz = cp.asarray(Xref[:,2], dtype=cp.float32)

                s0x = cp.asarray(S0[:,0], dtype=cp.float32); s0y = cp.asarray(S0[:,1], dtype=cp.float32); s0z = cp.asarray(S0[:,2], dtype=cp.float32)
                s1x = cp.asarray(S1[:,0], dtype=cp.float32); s1y = cp.asarray(S1[:,1], dtype=cp.float32); s1z = cp.asarray(S1[:,2], dtype=cp.float32)
                bx  = cp.asarray(Bv[:,0], dtype=cp.float32); by  = cp.asarray(Bv[:,1], dtype=cp.float32); bz  = cp.asarray(Bv[:,2], dtype=cp.float32)
                mx  = cp.asarray(MID[:,0], dtype=cp.float32); my  = cp.asarray(MID[:,1], dtype=cp.float32); mz  = cp.asarray(MID[:,2], dtype=cp.float32)
                hl  = cp.asarray(HL,        dtype=cp.float32)

                G00 = cp.zeros(px.size, dtype=cp.float32); G01 = cp.zeros_like(G00); G02 = cp.zeros_like(G00)
                G10 = cp.zeros_like(G00); G11 = cp.zeros_like(G00); G12 = cp.zeros_like(G00)
                G20 = cp.zeros_like(G00); G21 = cp.zeros_like(G00); G22 = cp.zeros_like(G00)

                threads = 256
                blocks  = (px.size + threads - 1) // threads

                # Choose an effective Rc that surely includes every segment relative to any grid node:
                # use the grid AABB diagonal plus max half-length.
                diag = float(np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz))
                Rc_all = np.float32(diag + (float(np.max(HL)) if HL.size else 0.0) + 1.0)

                sr_full((blocks,), (threads,),
                        (np.int32(px.size),
                        px, py, pz,
                        np.int32(Ns),
                        s0x, s0y, s0z, s1x, s1y, s1z,
                        bx,  by,  bz,
                        mx,  my,  mz,
                        hl,
                        cp.float32(Rc_all),
                        cp.float32(a_phys), cp.float32(nu),
                        G00, G01, G02, G10, G11, G12, G20, G21, G22))

                Gfull = cp.stack([
                            cp.stack([G00,G01,G02], axis=1),
                            cp.stack([G10,G11,G12], axis=1),
                            cp.stack([G20,G21,G22], axis=1)
                        ], axis=1).transpose(0,2,1)  # (N,3,3)
                Gfull = Gfull.reshape(nx,ny,nz,3,3)

                # Integrate dudx -> U (FFT LSQ, U(k=0)=0)
                kx = 2.0*np.pi*cp.fft.fftfreq(nx, d=dx)
                ky = 2.0*np.pi*cp.fft.fftfreq(ny, d=dy)
                kz = 2.0*np.pi*cp.fft.fftfreq(nz, d=dz)
                KX, KY, KZ = cp.meshgrid(kx, ky, kz, indexing="ij")
                K2 = KX*KX + KY*KY + KZ*KZ
                K2[0,0,0] = 1.0

                Ugrid = cp.zeros((nx,ny,nz,3), dtype=cp.float32)
                for i in range(3):
                    Gk0 = cp.fft.fftn(Gfull[..., i, 0]); Gk1 = cp.fft.fftn(Gfull[..., i, 1]); Gk2 = cp.fft.fftn(Gfull[..., i, 2])
                    Uk  = (1j*(KX*Gk0 + KY*Gk1 + KZ*Gk2)) / K2
                    Uk[0,0,0] = 0.0
                    Ugrid[..., i] = cp.real(cp.fft.ifftn(Uk)).astype(cp.float32)

                U = Ugrid.reshape(nx*ny*nz, 3).get().astype(dtype, copy=False)
            else:
                # CPU SR-only: compute full dudx(a_phys) for all segments/nodes; integrate to U
                px = Xref[:,0].astype(np.float64); py = Xref[:,1].astype(np.float64); pz = Xref[:,2].astype(np.float64)
                Gsum = np.zeros((px.size,3,3), dtype=np.float64)

                def _dudx_seg_full(px, py, pz, p1, p2, b, a, nu):
                    t = p2 - p1
                    L2 = float(np.dot(t,t))
                    if L2 < 1e-20:
                        z = np.zeros_like(px)
                        return (z,z,z,z,z,z,z,z,z)
                    Linv = 1.0/np.sqrt(L2); L = 1.0/Linv; t = t*Linv
                    Rx = px - p1[0]; Ry = py - p1[1]; Rz = pz - p1[2]
                    Rdt = Rx*t[0] + Ry*t[1] + Rz*t[2]
                    p0x = p1[0] + Rdt*t[0]; p0y = p1[1] + Rdt*t[1]; p0z = p1[2] + Rdt*t[2]
                    dx = Rx - (p0x - p1[0]); dy = Ry - (p0y - p1[1]); dz = Rz - (p0z - p1[2])
                    s1 = -Rdt; s2 = L - Rdt

                    a2  = a*a
                    d2  = dx*dx + dy*dy + dz*dz
                    da2 = d2 + a2
                    da2inv = 1.0/da2
                    Ra1 = np.sqrt(s1*s1 + da2); Ra2 = np.sqrt(s2*s2 + da2)
                    Ra1inv = 1.0/np.maximum(Ra1,1e-38); Ra2inv = 1.0/np.maximum(Ra2,1e-38)
                    Ra1inv3 = Ra1inv*Ra1inv*Ra1inv; Ra2inv3 = Ra2inv*Ra2inv*Ra2inv

                    J03 = da2inv*(s2*Ra2inv - s1*Ra1inv)
                    J13 = -Ra2inv + Ra1inv
                    J15 = -(1.0/3.0)*(Ra2inv3 - Ra1inv3)
                    J25 = (1.0/3.0)*da2inv*(s2*s2*s2*Ra2inv3 - s1*s1*s1*Ra1inv3)
                    J05 = da2inv*(2.0*J25 + s2*Ra2inv3 - s1*Ra1inv3)
                    J35 = 2.0*da2*J15 - s2*s2*Ra2inv3 + s1*s1*Ra1inv3

                    A0 = 3.0*a2*(dx*J05 - t[0]*J15) + 2.0*(dx*J03 - t[0]*J13)
                    A1 = 3.0*a2*(dy*J05 - t[1]*J15) + 2.0*(dy*J03 - t[1]*J13)
                    A2 = 3.0*a2*(dz*J05 - t[2]*J15) + 2.0*(dz*J03 - t[2]*J13)

                    U1_00 = (-A2*t[1]*b[0]) - (-A1*t[2]*b[0])
                    U1_01 = (-A0*t[2]*b[1]) - (-A2*t[0]*b[1])
                    U1_02 = (-A1*t[0]*b[2]) - (-A0*t[1]*b[2])

                    U2_00 = (-A0*t[2]*b[1]) - (-A0*t[1]*b[2])
                    U2_11 = (-A1*t[0]*b[2]) - (-A1*t[2]*b[0])
                    U2_22 = (-A2*t[1]*b[0]) - (-A2*t[0]*b[1])

                    t0,t1,t2 = t[0],t[1],t[2]; d0,d1,d2c = dx,dy,dz
                    B111 = -3.0*d0*J03 + 3.0*t0*J13 + 3.0*d0*d0*d0*J05 - 9.0*(d0*d0*t0)*J15 + 9.0*(d0*t0*t0)*J25 - 3.0*t0*t0*t0*J35
                    B222 = -3.0*d1*J03 + 3.0*t1*J13 + 3.0*d1*d1*d1*J05 - 9.0*(d1*d1*t1)*J15 + 9.0*(d1*t1*t1)*J25 - 3.0*t1*t1*t1*J35
                    B333 = -3.0*d2c*J03 + 3.0*t2*J13 + 3.0*d2c*d2c*d2c*J05 - 9.0*(d2c*d2c*t2)*J15 + 9.0*(d2c*t2*t2)*J25 - 3.0*t2*t2*t2*J35
                    B112 = -d1*J03 + t1*J13 + 3.0*d0*d0*d1*J05 -3.0*(d0*d0*t1 + d0*t0*d1 + t0*d0*d1)*J15 +3.0*(t0*t0*d1 + t0*d0*t1 + d0*t0*t1)*J25 -3.0*t0*t0*t1*J35
                    B113 = -d2c*J03 + t2*J13 + 3.0*d0*d0*d2c*J05 -3.0*(d0*d0*t2 + d0*t0*d2c + t0*d0*d2c)*J15 +3.0*(t0*t0*d2c + t0*d0*t2 + d0*t0*t2)*J25 -3.0*t0*t0*t2*J35
                    B221 = -d0*J03 + t0*J13 + 3.0*d1*d1*d0*J05 -3.0*(d1*d1*t0 + d1*t1*d0 + t1*d1*d0)*J15 +3.0*(t1*t1*d0 + t1*d1*t0 + d1*t1*t0)*J25 -3.0*t1*t1*t0*J35
                    B223 = -d2c*J03 + t2*J13 + 3.0*d1*d1*d2c*J05 -3.0*(d1*d1*t2 + d1*t1*d2c + t1*d1*d2c)*J15 +3.0*(t1*t1*d2c + t1*d1*t2 + d1*t1*t2)*J25 -3.0*t1*t1*t2*J35
                    B331 = -d0*J03 + t0*J13 + 3.0*d2c*d2c*d0*J05 -3.0*(d2c*d2c*t0 + d2c*t2*d0 + t2*d2c*d0)*J15 +3.0*(t2*t2*d0 + t2*d2c*t0 + d2c*t2*t0)*J25 -3.0*t2*t2*t0*J35
                    B332 = -d1*J03 + t1*J13 + 3.0*d2c*d2c*d1*J05 -3.0*(d2c*d2c*t1 + d2c*t2*d1 + t2*d2c*d1)*J15 +3.0*(t2*t2*d1 + t2*d2c*t1 + d2c*t2*t1)*J25 -3.0*t2*t2*t1*J35
                    B123 =  3.0*d0*d1*d2c*J05 -3.0*(d0*d1*t2 + d0*t1*d2c + t0*d1*d2c)*J15 +3.0*(t0*t1*d2c + t0*d1*t2 + d0*t1*t2)*J25 -3.0*t0*t1*t2*J35

                    U3_00 = (B112*t2 - B113*t1)*b[0] + (B113*t0 - B111*t2)*b[1] + (B111*t1 - B112*t0)*b[2]
                    U3_11 = (B222*t2 - B223*t1)*b[0] + (B223*t0 - B221*t2)*b[1] + (B221*t1 - B222*t0)*b[2]
                    U3_22 = (B332*t2 - B333*t1)*b[0] + (B333*t0 - B331*t2)*b[1] + (B331*t1 - B332*t0)*b[2]
                    U3_01 = (B221*t2 - B123*t1)*b[0] + (B123*t0 - B112*t2)*b[1] + (B112*t1 - B221*t0)*b[2]
                    U3_02 = (B123*t2 - B331*t1)*b[0] + (B331*t0 - B113*t2)*b[1] + (B113*t1 - B123*t0)*b[2]
                    U3_12 = (B223*t2 - B332*t1)*b[0] + (B332*t0 - B123*t2)*b[1] + (B123*t1 - B223*t0)*b[2]

                    m8pi   = -0.125/np.pi
                    m8pinu = m8pi/(1.0 - nu)
                    d00 = m8pi*(U1_00 + U2_00) + m8pinu*U3_00
                    d01 = m8pi*(U1_01 + 0.0)   + m8pinu*U3_01
                    d02 = m8pi*(U1_02 + 0.0)   + m8pinu*U3_02
                    d10 = m8pinu*U3_01
                    d11 = m8pi*(0.0 + U2_11) + m8pinu*U3_11
                    d12 = m8pinu*U3_12
                    d20 = m8pinu*U3_02
                    d21 = m8pinu*U3_12
                    d22 = m8pi*(0.0 + U2_22) + m8pinu*U3_22
                    return (d00,d01,d02,d10,d11,d12,d20,d21,d22)

                # Include all segments by using an effective large radius like the diagonal size of the box
                diag = float(np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz))
                Rc_all = diag + (float(np.max(HL)) if HL.size else 0.0) + 1.0

                for s in range(Ns):
                    p0 = S0[s]; p1 = S1[s]; b = Bv[s]
                    dxm = px - MID[s,0]; dym = py - MID[s,1]; dzm = pz - MID[s,2]
                    rad = Rc_all + HL[s]
                    msk = (dxm*dxm + dym*dym + dzm*dzm) <= (rad*rad)
                    if not np.any(msk):
                        continue
                    d00,d01,d02,d10,d11,d12,d20,d21,d22 = _dudx_seg_full(px[msk], py[msk], pz[msk], p0, p1, b, a_phys, nu)
                    Gsum[msk,0,0] += d00; Gsum[msk,0,1] += d01; Gsum[msk,0,2] += d02
                    Gsum[msk,1,0] += d10; Gsum[msk,1,1] += d11; Gsum[msk,1,2] += d12
                    Gsum[msk,2,0] += d20; Gsum[msk,2,1] += d21; Gsum[msk,2,2] += d22

                Gfull = Gsum.reshape(nx,ny,nz,3,3)

                # Integrate dudx -> U
                kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
                ky = 2.0*np.pi*np.fft.fftfreq(ny, d=dy)
                kz = 2.0*np.pi*np.fft.fftfreq(nz, d=dz)
                KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
                K2 = KX*KX + KY*KY + KZ*KZ
                K2[0,0,0] = 1.0

                Ugrid = np.zeros((nx,ny,nz,3), dtype=np.float32)
                for i in range(3):
                    Gk0 = np.fft.fftn(Gfull[..., i, 0]); Gk1 = np.fft.fftn(Gfull[..., i, 1]); Gk2 = np.fft.fftn(Gfull[..., i, 2])
                    Uk  = (1j*(KX*Gk0 + KY*Gk1 + KZ*Gk2)) / K2
                    Uk[0,0,0] = 0.0
                    Ugrid[..., i] = np.real(np.fft.ifftn(Uk)).astype(np.float32)

                U = Ugrid.reshape(nx*ny*nz, 3).astype(dtype, copy=False)

        # --------------------------
        # LR-only branch
        # --------------------------
        if mode == "LR":
            if gpu_ok:
                U = u_long.reshape(nx*ny*nz, 3).get().astype(dtype, copy=False)
            else:
                U = u_long.reshape(nx*ny*nz, 3).astype(dtype, copy=False)

        # --------------------------
        # Final scaling and write-out
        # --------------------------
        if float(scale) != 1.0:
            U *= float(scale)

        nodes_path = os.path.join(out_dir, nodes_filename)
        conn_path  = os.path.join(out_dir, conn_filename)

        xyzu = np.concatenate([Xref.astype(np.float32, copy=False), U.astype(np.float32, copy=False)], axis=1)
        conn0 = (conn - 1) if one_based_connectivity else conn
        conn0 = conn0.astype(np.int32, copy=False)

        def _write_nodes(path, arr, fmt, chunk_rows):
            if file_format == "txt":
                with open(path, "w", buffering=64*1024*1024) as f:
                    for s0 in range(0, arr.shape[0], int(chunk_rows)):
                        np.savetxt(f, arr[s0:s0+int(chunk_rows)], fmt=fmt)
            elif file_format == "npy":
                if not path.endswith(".npy"):
                    path = path + ".npy"
                np.save(path, arr, allow_pickle=False)
                return path
            elif file_format == "npz":
                if not path.endswith(".npz"):
                    path = path + ".npz"
                np.savez_compressed(path, xyzu=arr)
                return path
            else:
                raise ValueError("file_format must be 'txt', 'npy', or 'npz'")
            return path

        def _write_conn(path, conn_arr):
            if file_format == "txt":
                with open(path, "w", buffering=32*1024*1024) as f:
                    np.savetxt(f, (conn if one_based_connectivity else conn0).astype(np.int64, copy=False), fmt="%d")
            elif file_format == "npy":
                if not path.endswith(".npy"):
                    path = path + ".npy"
                np.save(path, conn0, allow_pickle=False)
                return path
            elif file_format == "npz":
                if not path.endswith(".npz"):
                    path = path + ".npz"
                np.savez_compressed(path, tet4=conn0)
                return path
            else:
                raise ValueError("file_format must be 'txt', 'npy', or 'npz'")
            return path

        nodes_path = _write_nodes(nodes_path, xyzu, float_fmt, chunk_rows)
        conn_path  = _write_conn(conn_path, conn0)

        return {
            "Xref": Xref.astype(dtype, copy=False),
            "U": U.astype(dtype, copy=False),
            "conn": conn,
            "nodes_path": nodes_path,
            "conn_path": conn_path
        }
        
    # Helpers used by finalize_dislocation_sample
    def _nearest_neighbor_distance_from_crystal(self, crystal):
        """
        Estimate nearest-neighbor distance from the crystal structure.

        Computes the nearest-neighbor distance d0 by checking pair distances
        among atoms in a 3x3x3 supercell of the unit cell.

        Args:
            crystal: Crystal object providing lattice_matrix and
                lattice_atom_cartesian attributes.

        Returns:
            float: Estimated nearest-neighbor distance.
        """
        R = np.asarray(crystal.lattice_matrix, dtype=np.float64)
        basis = np.asarray(crystal.lattice_atom_cartesian, dtype=np.float64)  # (A,3)
        # supercell translations in [-1,0,1]^3
        shifts = np.array(np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1], indexing="ij")).reshape(3, -1).T
        T = shifts @ R  # (27,3)
        pts = (basis[None, :, :] + T[:, None, :]).reshape(-1, 3)  # (27*A, 3)
        # distance to original basis
        dmin = np.inf
        for p in basis:
            d = np.linalg.norm(pts - p[None, :], axis=1)
            d = d[(d > 1e-6)]
            if d.size:
                dmin = min(dmin, float(d.min()))
        if not np.isfinite(dmin):
            dmin = float(np.linalg.norm(R[:, 0]))  # fallback
        return float(dmin)

    def _ensure_cuda_segU_phys_kernel(self, gp_ng=4, dtype="float32"):
        """
        Build and cache a CUDA kernel for near-core displacement.

        Builds and caches a CUDA kernel that accumulates physically-accurate
        near-core displacement by Gauss-Legendre line integration along
        dislocation segments.

        Args:
            gp_ng: Number of Gauss points (4 or 8). Defaults to 4.
            dtype: Data type string ("float32"). Defaults to "float32".

        Returns:
            The compiled CUDA kernel function, or None if CuPy is unavailable.

        Note:
            Cache key is ("segU_phys", dtype, gp_ng). The kernel expects
            Gauss xi and w arrays as arguments.
        """
        if cp is None:
            return None
        if not hasattr(self, "_opendis_cuda_phys"):
            self._opendis_cuda_phys = {}

        key = ("segU_phys", str(dtype), int(gp_ng))
        if key in self._opendis_cuda_phys:
            return self._opendis_cuda_phys[key]

        src = r'''
        extern "C" __global__
        void segU_phys(
            const float* __restrict__ X,   // (Npts,3)
            const float* __restrict__ S0,  // (Ns,3)
            const float* __restrict__ S1,  // (Ns,3)
            const float* __restrict__ B,   // (Ns,3)
            const float* __restrict__ T,   // (Ns,3)
            const float* __restrict__ MID, // (Ns,3)
            const float* __restrict__ HL,  // (Ns,)
            const int Ns, const int Npts, const float rcut,
            const float* __restrict__ xi,  // (NG,)
            const float* __restrict__ wg,  // (NG,)
            const int NG,
            const float A, const float c1, const float a2, const float scale,
            float* __restrict__ Uout       // (Npts,3)
        ){
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= Npts) return;

            float x = X[3*i+0];
            float y = X[3*i+1];
            float z = X[3*i+2];

            float ux = 0.0f, uy = 0.0f, uz = 0.0f;

            for (int s=0; s<Ns; ++s){
                // midpoint culling
                float mx = MID[3*s+0], my = MID[3*s+1], mz = MID[3*s+2];
                float dxm = x - mx, dym = y - my, dzm = z - mz;
                float rad = rcut + HL[s];
                if (dxm*dxm + dym*dym + dzm*dzm > rad*rad) continue;

                float s0x = S0[3*s+0], s0y = S0[3*s+1], s0z = S0[3*s+2];
                float s1x = S1[3*s+0], s1y = S1[3*s+1], s1z = S1[3*s+2];
                float Lx = s1x - s0x, Ly = s1y - s0y, Lz = s1z - s0z;
                float L2 = Lx*Lx + Ly*Ly + Lz*Lz;
                if (!(L2 > 1.0e-20f)) continue;
                float L = sqrtf(L2);
                float seg_scale = 0.5f * L;

                float bx = B[3*s+0], by = B[3*s+1], bz = B[3*s+2];
                float tx = T[3*s+0], ty = T[3*s+1], tz = T[3*s+2];
                float dotBT = bx*tx + by*ty + bz*tz;

                for (int g=0; g<NG; ++g){
                    float u = 0.5f*(xi[g] + 1.0f);
                    float px = s0x + u*Lx;
                    float py = s0y + u*Ly;
                    float pz = s0z + u*Lz;

                    float Rx = x - px;
                    float Ry = y - py;
                    float Rz = z - pz;

                    float dotBR = bx*Rx + by*Ry + bz*Rz;
                    float dotTR = tx*Rx + ty*Ry + tz*Rz;

                    float r2   = Rx*Rx + Ry*Ry + Rz*Rz + a2;
                    float invR = rsqrtf(r2);
                    float invR3 = invR*invR*invR;
                    float invR5 = invR3*invR*invR;

                    // Same integrand as CPU fallback helper
                    float Cx = (-c1)*bx*dotTR*invR3 + (tx*dotBR + dotBT*Rx)*invR3 - 3.0f*Rx*dotBR*dotTR*invR5;
                    float Cy = (-c1)*by*dotTR*invR3 + (ty*dotBR + dotBT*Ry)*invR3 - 3.0f*Ry*dotBR*dotTR*invR5;
                    float Cz = (-c1)*bz*dotTR*invR3 + (tz*dotBR + dotBT*Rz)*invR3 - 3.0f*Rz*dotBR*dotTR*invR5;

                    float w = wg[g] * seg_scale * A;
                    ux += w * Cx;
                    uy += w * Cy;
                    uz += w * Cz;
                }
            }

            Uout[3*i+0] = scale * ux;
            Uout[3*i+1] = scale * uy;
            Uout[3*i+2] = scale * uz;
        }
        ''';

        mod = cp.RawModule(
            code=src,
            backend='nvcc',
            options=('--gpu-architecture=native', '-O3')
        )
        fn = mod.get_function('segU_phys')
        self._opendis_cuda_phys[key] = fn
        return fn

    def _ensure_cuda_helpers_for_cleanup(self):
        """
        Build and cache CUDA helper kernels for cleanup operations.

        Creates and caches two CUDA kernels:
            - flag_near_segments: Flags atoms near any dislocation segment.
            - relax_repulsive_sorted: Performs repulsive relaxation using cell lists.

        Returns:
            dict: Dictionary of compiled CUDA kernel functions, or None if
                CuPy is unavailable.
        """
        if (cp is None):
            return None
        if hasattr(self, "_cleanup_cuda"):
            return self._cleanup_cuda
        # Kernel 1: flag atoms near any dislocation segment by midpoint radius test
        src_flag = r'''
        extern "C" __global__
        void flag_near_segments(const float* __restrict__ X,     // (M,3)
                                const float* __restrict__ MID,   // (Ns,3)
                                const float* __restrict__ HL,    // (Ns,)
                                const int Ns,
                                const float rcut,
                                unsigned char* __restrict__ mask,
                                const int M)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= M) return;
            float xi = X[3*i+0];
            float yi = X[3*i+1];
            float zi = X[3*i+2];
            unsigned char f = 0;
            for (int s=0; s<Ns; ++s){
                float mx = MID[3*s+0], my = MID[3*s+1], mz = MID[3*s+2];
                float dx = xi - mx, dy = yi - my, dz = zi - mz;
                float rad = rcut + HL[s];
                if (dx*dx + dy*dy + dz*dz <= rad*rad){ f = 1; break; }
            }
            mask[i] = f;
        }'''
        # Kernel 2: single relaxation step using cell lists (repulsive only)
        # Operates on sorted positions to traverse 27 neighboring cells quickly.
        src_relax = r'''
        extern "C" __global__
        void relax_repulsive_sorted(
            const float* __restrict__ pos_sorted,   // (N,3)
            const int*   __restrict__ cell_start,   // (Nc,)
            const int*   __restrict__ cell_end,     // (Nc,)
            const float* __restrict__ bbox_min,     // (3,)
            const float  cell_size,
            const int nx, const int ny, const int nz,
            const float d0, const float k_rep, const float dt,
            float* __restrict__ out_sorted,         // (N,3)
            const int N)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= N) return;

            float xi = pos_sorted[3*i+0];
            float yi = pos_sorted[3*i+1];
            float zi = pos_sorted[3*i+2];

            // compute this particle's cell
            int cx = (int)floorf((xi - bbox_min[0]) / cell_size);
            int cy = (int)floorf((yi - bbox_min[1]) / cell_size);
            int cz = (int)floorf((zi - bbox_min[2]) / cell_size);
            if (cx < 0) cx = 0; else if (cx >= nx) cx = nx-1;
            if (cy < 0) cy = 0; else if (cy >= ny) cy = ny-1;
            if (cz < 0) cz = 0; else if (cz >= nz) cz = nz-1;

            float dx_acc = 0.0f, dy_acc = 0.0f, dz_acc = 0.0f;

            // visit 27 neighbor cells
            for (int dzc=-1; dzc<=1; ++dzc){
                int zc = cz + dzc; if (zc < 0 || zc >= nz) continue;
                for (int dyc=-1; dyc<=1; ++dyc){
                    int yc = cy + dyc; if (yc < 0 || yc >= ny) continue;
                    for (int dxc=-1; dxc<=1; ++dxc){
                        int xc = cx + dxc; if (xc < 0 || xc >= nx) continue;
                        int cid = zc*(nx*ny) + yc*nx + xc;
                        int s0 = cell_start[cid];
                        int s1 = cell_end[cid];
                        for (int j = s0; j < s1; ++j){
                            if (j == i) continue; // skip self in the same cell
                            float xj = pos_sorted[3*j+0];
                            float yj = pos_sorted[3*j+1];
                            float zj = pos_sorted[3*j+2];
                            float dx = xi - xj, dy = yi - yj, dz = zi - zj;
                            float r2 = dx*dx + dy*dy + dz*dz;
                            if (r2 > 1e-20f){
                                float r = sqrtf(r2);
                                if (r < d0){
                                    // linear repulsion toward target distance
                                    float s = (d0 - r) * (k_rep / r);
                                    dx_acc += s * dx;
                                    dy_acc += s * dy;
                                    dz_acc += s * dz;
                                }
                            }
                        }
                    }
                }
            }

            out_sorted[3*i+0] = xi + dt * dx_acc;
            out_sorted[3*i+1] = yi + dt * dy_acc;
            out_sorted[3*i+2] = zi + dt * dz_acc;
        }'''
        mod_flag  = cp.RawModule(code=src_flag, backend="nvcc",
                                 options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true'))
        mod_relax = cp.RawModule(code=src_relax, backend="nvcc",
                                 options=('--gpu-architecture=native', '-O3', '--ftz=true', '--fmad=true'))
        self._cleanup_cuda = {
            "flag_near_segments": mod_flag.get_function("flag_near_segments"),
            "relax_repulsive_sorted": mod_relax.get_function("relax_repulsive_sorted"),
        }
        return self._cleanup_cuda

    def _accum_segment_displacement_phys_cpu(self, X, S0f, S1f, Bf, Tf, MIDf, HLf,
                                             xis, ws, A32, c132, a232, rcf, sc32):
        """
        Compute near-core displacement on CPU (fallback for CUDA).

        CPU implementation for near-core displacement evaluation using the same
        Gauss-Legendre line integration as the CUDA kernel.

        Args:
            X: Query positions as (N, 3) array.
            S0f: Segment start points as (Ns, 3) array.
            S1f: Segment end points as (Ns, 3) array.
            Bf: Burgers vectors as (Ns, 3) array.
            Tf: Unit tangent vectors as (Ns, 3) array.
            MIDf: Segment midpoints as (Ns, 3) array.
            HLf: Segment half-lengths as (Ns,) array.
            xis: Gauss quadrature nodes as (NG,) array.
            ws: Gauss quadrature weights as (NG,) array.
            A32: Material constant A (float32).
            c132: Material constant c1 (float32).
            a232: Core radius squared (float32).
            rcf: Cutoff radius (float32).
            sc32: Scale factor (float32).

        Returns:
            numpy.ndarray: Displacement field as (N, 3) float32 array.
        """
        U = np.zeros_like(X, dtype=np.float32)
        NG = int(xis.shape[0])
        for i in range(X.shape[0]):
            xi, yi, zi = X[i, 0], X[i, 1], X[i, 2]
            ux = 0.0; uy = 0.0; uz = 0.0
            for s in range(S0f.shape[0]):
                # midpoint culling
                mx, my, mz = MIDf[s, 0], MIDf[s, 1], MIDf[s, 2]
                dxm = xi - mx; dym = yi - my; dzm = zi - mz
                if (dxm*dxm + dym*dym + dzm*dzm) > (rcf + HLf[s])*(rcf + HLf[s]):
                    continue
                s0x, s0y, s0z = S0f[s, 0], S0f[s, 1], S0f[s, 2]
                Lx = S1f[s, 0] - s0x; Ly = S1f[s, 1] - s0y; Lz = S1f[s, 2] - s0z
                L = np.sqrt(Lx*Lx + Ly*Ly + Lz*Lz).astype(np.float32)
                if L <= 1e-16:
                    continue
                bx, by, bz = Bf[s, 0], Bf[s, 1], Bf[s, 2]
                tx, ty, tz = Tf[s, 0], Tf[s, 1], Tf[s, 2]
                seg_scale = 0.5 * L
                for g in range(NG):
                    float_u = 0.5*(xis[g] + 1.0)
                    px = s0x + float_u*Lx; py = s0y + float_u*Ly; pz = s0z + float_u*Lz
                    Rx = xi - px; Ry = yi - py; Rz = zi - pz
                    dotBR = bx*Rx + by*Ry + bz*Rz
                    dotTR = tx*Rx + ty*Ry + tz*Rz
                    dotBT = bx*tx + by*ty + bz*tz
                    r2 = Rx*Rx + Ry*Ry + Rz*Rz + a232
                    invR = 1.0/np.sqrt(r2)
                    invR3 = invR*invR*invR
                    invR5 = invR3*invR*invR
                    Cx = (-c132) * bx * dotTR * invR3 + (tx * dotBR + dotBT * Rx) * invR3 - 3.0 * Rx * dotBR * dotTR * invR5
                    Cy = (-c132) * by * dotTR * invR3 + (ty * dotBR + dotBT * Ry) * invR3 - 3.0 * Ry * dotBR * dotTR * invR5
                    Cz = (-c132) * bz * dotTR * invR3 + (tz * dotBR + dotBT * Rz) * invR3 - 3.0 * Rz * dotBR * dotTR * invR5
                    w = ws[g] * seg_scale * A32
                    ux += w * Cx; uy += w * Cy; uz += w * Cz
            U[i, 0] = sc32 * ux
            U[i, 1] = sc32 * uy
            U[i, 2] = sc32 * uz
        return U

    def finalize_dislocation_sample(self,
                                    crystal,
                                    deformed_sample,
                                    pristine_sample=None,
                                    output_directory=None,
                                    mu=1.0,
                                    nu=0.33,
                                    core_radius=5.0,
                                    near_factor=1.5,
                                    gauss_points=4,
                                    relax_steps=3,
                                    relax_dt=0.125,
                                    relax_k=0.5,
                                    use_gpu=True,
                                    dtype=np.float32):
        """
        Finalize dislocation sample with near-core correction and relaxation.

        Performs optimized near-core finalization with:
            - Cached CUDA line-integral kernel
            - Sub-chunking for large near-core sets to avoid OOM
            - Streaming CPU midpoint test to avoid NxNs allocations

        Args:
            crystal: Crystal object for material properties.
            deformed_sample: Sample object with deformed atomic positions.
            pristine_sample: Optional pristine sample for exact near-core
                recomputation. If None, near-core positions are not recalculated.
            output_directory: Directory for output files. If None, uses
                self.directory.
            mu: Shear modulus. Defaults to 1.0.
            nu: Poisson's ratio (must be in (0, 0.5)). Defaults to 0.33.
            core_radius: Physical core radius. Defaults to 5.0.
            near_factor: Factor multiplied by core_radius for near-core cutoff.
                Defaults to 1.5.
            gauss_points: Number of Gauss-Legendre integration points (4 or 8).
                Defaults to 4.
            relax_steps: Number of repulsive relaxation iterations. Defaults to 3.
            relax_dt: Time step for relaxation. Defaults to 0.125.
            relax_k: Spring constant for repulsive relaxation. Defaults to 0.5.
            use_gpu: If True and CuPy is available, use GPU acceleration.
                Defaults to True.
            dtype: NumPy dtype for output arrays. Defaults to np.float32.

        Raises:
            RuntimeError: If OpenDiS network has not been imported.
            ValueError: If nu is out of range (0, 0.5).

        Note:
            The finalization procedure:
                1. Flags atoms near any segment by midpoint radius test.
                2. If pristine_sample provided, recomputes exact near-core
                   displacements using Gauss-Legendre line integration.
                3. Performs lightweight repulsive relaxation to resolve overlaps.
        """
        if not hasattr(self, "_opendis_S0"):
            raise RuntimeError("OpenDiS network must be imported first (prepare_opendis_fe or import_dislocation_network).")

        out_dir = output_directory if output_directory is not None else (self.directory if self.directory else ".")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Material constants for near-core evaluation
        mu = float(mu)
        nu = float(nu)
        if not (0.0 < nu < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        gp_ng = int(gauss_points) if int(gauss_points) in (4, 8) else 4
        a = float(core_radius)
        a2 = a*a
        c1 = float(3.0 - 4.0*nu)
        A = float(1.0 / (16.0*np.pi*mu*(1.0 - nu)))
        near_rcut = float(near_factor) * a

        # Dislocation arrays (kept in float32 for GPU)
        S0  = np.asarray(self._opendis_S0,   dtype=np.float32)
        S1  = np.asarray(self._opendis_S1,   dtype=np.float32)
        Bv  = np.asarray(self._opendis_bvec, dtype=np.float32)
        Tv  = np.asarray(self._opendis_tvec, dtype=np.float32)
        MID = np.asarray(self._opendis_mids, dtype=np.float32)
        HL  = np.asarray(self._opendis_halfL, dtype=np.float32)
        Ns = int(S0.shape[0])
        if Ns == 0:
            raise ValueError("No dislocation segments loaded.")

        # Relaxation target spacing from crystal
        d0 = self._nearest_neighbor_distance_from_crystal(crystal)

        # Prepare GPU env (matches generate_nodal_field pattern)
        gpu_ok = bool(use_gpu and (cp is not None))
        if gpu_ok:
            try:
                _ = cp.cuda.runtime.getDeviceCount()
            except Exception:
                gpu_ok = False

        # Precompute Gauss tables (host + device)
        if gp_ng == 4:
            xinp = np.array([-0.8611363115940526, -0.3399810435848563,
                            0.3399810435848563,  0.8611363115940526], dtype=np.float32)
            wgnp = np.array([ 0.3478548451374539,  0.6521451548625461,
                            0.6521451548625461,  0.3478548451374539], dtype=np.float32)
        else:
            xinp = np.array([-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                            0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363], dtype=np.float32)
            wgnp = np.array([ 0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620,
                            0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763], dtype=np.float32)

        if gpu_ok:
            # Helpers used also in generate_nodal_field family
            cuda_helpers = self._ensure_cuda_helpers_for_cleanup()

            # Segment data to device (one-time)
            S0g  = cp.asarray(S0);  S1g  = cp.asarray(S1)
            Bg   = cp.asarray(Bv);  Tg   = cp.asarray(Tv)
            MIDg = cp.asarray(MID); HLg  = cp.asarray(HL)

            # Gauss nodes/weights to device
            xig  = cp.asarray(xinp)
            wgg  = cp.asarray(wgnp)

            # Compile and cache the line-integral kernel
            segU_kernel = self._ensure_cuda_segU_phys_kernel(gp_ng=gp_ng, dtype="float32")

        # CPU helper alias
        def _cpu_accum_batch(Xbatchnp):
            return self._accum_segment_displacement_phys_cpu(
                Xbatchnp.astype(np.float32, copy=False),
                S0.astype(np.float32, copy=False),
                S1.astype(np.float32, copy=False),
                Bv.astype(np.float32, copy=False),
                Tv.astype(np.float32, copy=False),
                MID.astype(np.float32, copy=False),
                HL.astype(np.float32, copy=False),
                xinp.astype(np.float32, copy=False),
                wgnp.astype(np.float32, copy=False),
                np.float32(A), np.float32(c1), np.float32(a2),
                np.float32(near_rcut), np.float32(1.0)
            )

        # Adaptive GPU batch size to prevent OOM
        def _gpu_batch_cap(default_cap=500000):
            if not gpu_ok:
                return 0
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                # very conservative per-point footprint: ~ 2 arrays of 3 floats + overhead
                bytes_per_pt = 3*4 + 3*4 + 64  # ~ 88 bytes
                cap = int(0.6 * free_b / max(bytes_per_pt, 1))
                cap = max(32768, min(cap, default_cap))
                return cap
            except Exception:
                return default_cap

        # CPU batch size (smaller to keep cache friendly)
        CPU_BATCH = 200000

        # Process chunks
        K = int(deformed_sample.chunk_total)
        for k in range(K):
            # Load deformed positions and species
            if gpu_ok:
                Xg = deformed_sample.load_chunk_positions(k+1, use_gpu=True)  # cp.ndarray (N,3)
                spcnp = deformed_sample.load_chunk_species(k+1, use_gpu=False)
                M = int(Xg.shape[0])

                # Flag near-core atoms
                flags = cp.empty((M,), dtype=cp.uint8)
                threads = 256
                blocks = (M + threads - 1)//threads
                self._ensure_cuda_helpers_for_cleanup()["flag_near_segments"](
                    (blocks,), (threads,),
                    (Xg.astype(cp.float32).ravel(),
                    MIDg.ravel(), HLg.ravel(),
                    np.int32(int(HLg.shape[0])),
                    cp.float32(near_rcut),
                    flags, np.int32(M))
                )
                near_idx = cp.where(flags != 0)[0]

                # If pristine provided: recompute exact near-core U and reset those atoms
                if pristine_sample is not None and near_idx.size > 0:
                    # Gather pristine Xref for the flagged subset (host -> device)
                    pos_refnp = pristine_sample.load_chunk_positions(k+1, use_gpu=False)
                    near_idxnp = near_idx.get()
                    # Sub-chunk to avoid OOM
                    BATCH = _gpu_batch_cap()
                    for s0 in range(0, near_idxnp.size, BATCH):
                        s1 = min(near_idxnp.size, s0 + BATCH)
                        idx_slice = near_idxnp[s0:s1]
                        Xref_sel = cp.asarray(pos_refnp[idx_slice, :], dtype=cp.float32)
                        Uout = cp.zeros((int(idx_slice.size), 3), dtype=cp.float32)

                        # Launch kernel
                        npts = np.int32(int(idx_slice.size))
                        Ns32 = np.int32(int(S0g.shape[0]))
                        segU_kernel(
                            ((int(idx_slice.size) + 255)//256,), (256,),
                            (
                                Xref_sel.ravel(),
                                S0g.ravel(), S1g.ravel(), Bg.ravel(), Tg.ravel(),
                                MIDg.ravel(), HLg.ravel(),
                                Ns32, npts, cp.float32(near_rcut),
                                xig, wgg, np.int32(gp_ng),
                                cp.float32(A), cp.float32(c1), cp.float32(a2), cp.float32(1.0),
                                Uout.ravel()
                            )
                        )
                        # Write back corrected positions: X = Xref + Uexact
                        Xg[cp.asarray(idx_slice), :] = Xref_sel + Uout

                # GPU relaxation via cell list (already chunked by construction)
                sorted_pos, sorted_idx, cell_start, cell_end, bbmin, cell_size, nx, ny, nz = \
                    deformed_sample.build_cell_list_gpu(Xg.astype(cp.float32), r_cut=float(d0))

                out_sorted = cp.empty_like(sorted_pos)
                threads3 = 256
                blocks3 = (int(sorted_pos.shape[0]) + threads3 - 1)//threads3
                relax_fn = self._ensure_cuda_helpers_for_cleanup()["relax_repulsive_sorted"]
                for _ in range(int(relax_steps)):
                    relax_fn(
                        (blocks3,), (threads3,),
                        (sorted_pos.ravel(),
                        cell_start.ravel(), cell_end.ravel(),
                        bbmin.ravel(),
                        cp.float32(cell_size),
                        np.int32(int(nx)), np.int32(int(ny)), np.int32(int(nz)),
                        cp.float32(float(d0)), cp.float32(float(relax_k)), cp.float32(float(relax_dt)),
                        out_sorted.ravel(),
                        np.int32(int(sorted_pos.shape[0])))
                    )
                    sorted_pos, out_sorted = out_sorted, sorted_pos  # ping-pong

                # Unsort back and persist
                pos_out = cp.empty_like(Xg)
                pos_out[sorted_idx, :] = sorted_pos
                posnp = pos_out.get()

            else:
                # CPU path (streaming midpoint test + sub-chunked line integral)
                posnp = np.asarray(deformed_sample.load_chunk_positions(k+1, use_gpu=False), dtype=np.float32)
                spcnp = deformed_sample.load_chunk_species(k+1, use_gpu=False)
                M = int(posnp.shape[0])

                # Streaming midpoint cull to avoid NxNs memory
                near_mask = np.zeros(M, dtype=bool)
                # Expand each segment's sphere once
                for s in range(Ns):
                    rad = near_rcut + HL[s]
                    d = posnp - MID[s]  # (M,3)
                    dist2 = (d[:, 0]*d[:, 0] + d[:, 1]*d[:, 1] + d[:, 2]*d[:, 2])
                    near_mask |= (dist2 <= rad*rad)

                if pristine_sample is not None and np.any(near_mask):
                    Xref_all = np.asarray(pristine_sample.load_chunk_positions(k+1, use_gpu=False), dtype=np.float32)
                    idx_all = np.flatnonzero(near_mask)
                    # Sub-chunk CPU integration
                    for s0 in range(0, idx_all.size, CPU_BATCH):
                        s1 = min(idx_all.size, s0 + CPU_BATCH)
                        idx_slice = idx_all[s0:s1]
                        Xsel = Xref_all[idx_slice, :]
                        Usel = _cpu_accum_batch(Xsel)
                        posnp[idx_slice, :] = Xsel + Usel

                # Simple CPU relaxation (reuse existing local scheme, but keep it light)
                cs = float(d0)
                bbmin = posnp.min(axis=0)
                idx_grid = np.floor((posnp - bbmin)/max(cs, 1e-12)).astype(np.int64)
                key = idx_grid[:, 0] + 104729*idx_grid[:, 1] + 130363*idx_grid[:, 2]
                from collections import defaultdict
                buckets = defaultdict(list)
                for ii, kk in enumerate(key):
                    buckets[int(kk)].append(ii)
                neighbors = [(dx,dy,dz) for dx in (-1,0,1) for dy in (-1,0,1) for dz in (-1,0,1)]
                for _ in range(int(relax_steps)):
                    disp = np.zeros_like(posnp, dtype=np.float32)
                    for ii, kk in enumerate(key):
                        ix, iy, iz = idx_grid[ii]
                        for dx,dy,dz in neighbors:
                            kk2 = (ix+dx) + 104729*(iy+dy) + 130363*(iz+dz)
                            for jj in buckets.get(int(kk2), []):
                                if jj == ii:
                                    continue
                                r = posnp[ii] - posnp[jj]
                                r2 = float(np.dot(r, r))
                                if r2 > 1e-20:
                                    rr = np.sqrt(r2)
                                    if rr < d0:
                                        s = (d0 - rr) * (relax_k/rr)
                                        disp[ii] += s * r
                    posnp += relax_dt * disp

            # Write chunk results
            deformed_sample.write_chunk_positions(posnp.astype(dtype, copy=False), k+1, override_directory=out_dir)
            deformed_sample.write_chunk_species(spcnp, k+1, override_directory=out_dir)

        # Finalize metadata
        deformed_sample.write_sample_metadata(override_directory=out_dir)
        
    def visualize_dislocation_network(
        self,
        sample=None,
        out_path=None,
        mode="matplotlib",
        show=True,
        render_tubes=True,
        line_width=2.0,
        color_mode="defect_type",
        uniform_color=(0.25, 0.25, 0.25),
        cmap=None,
        clim=None,
        type_thresholds=None,
        float_fmt="%.9e",
        clip_to_sample=True,
        clip_margin=0.0,
        use_minimum_image=True,
        length_filter_percentile=100.0,
        elev=None,
        azim=None,
    ):
        """
        Visualize or export the dislocation network.

        Creates a visualization of the current OpenDiS dislocation network using
        matplotlib, pyvista, or VTK output.

        Args:
            sample: Optional sample object for clipping and bounds. If provided
                and clip_to_sample is True, segments are clipped to the sample AABB.
            out_path: Output file path. If None, no file is written.
            mode: Visualization backend. Options are:
                - "matplotlib": 3D line plot using matplotlib (default).
                - "pyvista": Interactive 3D visualization using PyVista.
                - "vtk": Export to VTK legacy format.
            show: If True, display the visualization interactively. Defaults to True.
            render_tubes: If True and mode="pyvista", render lines as 3D tubes.
                Defaults to True.
            line_width: Line width for rendering. Defaults to 2.0.
            color_mode: Coloring scheme. Options are:
                - "uniform": Single color for all segments.
                - "signed_burgers": Color by signed Burgers vector component.
                - "burgers_magnitude": Color by Burgers vector magnitude.
                - "length": Color by segment length.
                - "defect_type": Color by defect type (edge/screw/partial) (default).
            uniform_color: RGB tuple for uniform coloring. Defaults to (0.25, 0.25, 0.25).
            cmap: Colormap name. If None, a sensible default is chosen per mode.
            clim: Color limits as (vmin, vmax). If None, auto-scaled.
            type_thresholds: Dictionary defining edge/screw classification thresholds.
                Example: {"edge": 0.2, "screw": 0.8}. Defaults to None.
            float_fmt: Float format for VTK text output. Defaults to "%.9e".
            clip_to_sample: If True and sample is provided, clip segments to
                sample AABB. Defaults to True.
            clip_margin: Margin to expand the sample AABB before clipping.
                Defaults to 0.0.
            use_minimum_image: If True, remap segments using minimum-image
                convention for periodic boundaries. Defaults to True.
            length_filter_percentile: Drop segments longer than this percentile
                (0-100). Defaults to 100.0 (keep all).
            elev: Elevation angle for matplotlib view. Defaults to None.
            azim: Azimuth angle for matplotlib view. Defaults to None.

        Returns:
            dict: Contains:
                - path: Path to written file (if any).
                - backend: Visualization backend used.
                - coloring: Color mode used.

        Raises:
            RuntimeError: If dislocation network has not been imported.
            ValueError: If color_mode is not recognized.
        """

        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError(
                "Dislocation network not initialized. "
                "Call import_dislocation_network(...) first."
            )

        # ---------------- Base arrays from imported OpenDiS network ----------------
        nodes = np.asarray(self._opendis_nodes_xyz, dtype=np.float32)
        segs  = np.asarray(self._opendis_segments,   dtype=np.int64)
        S0    = np.asarray(self._opendis_S0,         dtype=np.float64)
        S1    = np.asarray(self._opendis_S1,         dtype=np.float64)

        if hasattr(self, "_opendis_tvec") and self._opendis_tvec is not None:
            tvec = np.asarray(self._opendis_tvec, dtype=np.float64)
        else:
            Lvec0 = S1 - S0
            Llen0 = np.linalg.norm(Lvec0, axis=1)
            tvec = np.divide(
                Lvec0,
                Llen0[:, None],
                out=np.zeros_like(Lvec0),
                where=(Llen0[:, None] > 0),
            )

        if hasattr(self, "_opendis_bvec") and self._opendis_bvec is not None:
            bvec = np.asarray(self._opendis_bvec, dtype=np.float64)
        else:
            bvec = np.zeros_like(S0, dtype=np.float64)

        rebuilt_pairwise = False

        # ---------------- Optional: clip to sample AABB ----------------
        if (sample is not None) and clip_to_sample:
            corners = np.asarray(sample.corners, dtype=np.float64)
            cmin = corners.min(axis=0) - float(clip_margin)
            cmax = corners.max(axis=0) + float(clip_margin)

            p0 = S0
            p1 = S1
            d = p1 - p0
            t0 = np.zeros(p0.shape[0], dtype=np.float64)
            t1 = np.ones(p0.shape[0],  dtype=np.float64)
            valid = np.ones(p0.shape[0], dtype=bool)

            # Liang-Barsky style clipping against axis-aligned box
            for ax in range(3):
                p0a = p0[:, ax]
                da  = d[:, ax]
                nz = np.abs(da) > 1e-20
                inv = np.zeros_like(da)
                inv[nz] = 1.0 / da[nz]

                tmin = (cmin[ax] - p0a) * inv
                tmax = (cmax[ax] - p0a) * inv
                tlow  = np.minimum(tmin, tmax)
                thigh = np.maximum(tmin, tmax)

                t0 = np.maximum(t0, tlow)
                t1 = np.minimum(t1, thigh)

                if np.any(~nz):
                    mask = ~nz
                    valid[mask] &= (p0a[mask] >= cmin[ax]) & (p0a[mask] <= cmax[ax])

            keep = valid & (t0 <= t1)

            if not np.any(keep):
                nodes = np.zeros((0, 3), dtype=np.float32)
                segs  = np.zeros((0, 2), dtype=np.int64)
                S0    = np.zeros((0, 3), dtype=np.float64)
                S1    = np.zeros((0, 3), dtype=np.float64)
                tvec  = np.zeros((0, 3), dtype=np.float64)
                bvec  = np.zeros((0, 3), dtype=np.float64)
            else:
                t0c = np.clip(t0[keep], 0.0, 1.0)[:, None]
                t1c = np.clip(t1[keep], 0.0, 1.0)[:, None]
                p0k = p0[keep]
                dk  = d[keep]
                S0c = p0k + t0c * dk
                S1c = p0k + t1c * dk

                M = S0c.shape[0]
                nodes = np.empty((2 * M, 3), dtype=np.float32)
                nodes[0::2] = S0c.astype(np.float32, copy=False)
                nodes[1::2] = S1c.astype(np.float32, copy=False)
                segs = np.column_stack([
                    np.arange(0, 2 * M, 2, dtype=np.int64),
                    np.arange(1, 2 * M, 2, dtype=np.int64),
                ])

                S0 = S0c
                S1 = S1c
                Lvec = S1 - S0
                Llen = np.linalg.norm(Lvec, axis=1)
                tvec = np.divide(
                    Lvec,
                    Llen[:, None],
                    out=np.zeros_like(Lvec),
                    where=(Llen[:, None] > 0),
                )
                bvec = bvec[keep]
                rebuilt_pairwise = True

        # ---------------- Optional: minimum-image convention for PBC ----------------
        # This remaps each segment to its shortest periodic image for visualization.
        if use_minimum_image and S0.shape[0] > 0:
            # Decide which bounds define the periodic cell:
            if sample is not None and hasattr(sample, "corners"):
                corners_cell = np.asarray(sample.corners, dtype=np.float64)
                bmin_mi = corners_cell.min(axis=0)
                bmax_mi = corners_cell.max(axis=0)
            elif hasattr(self, "_opendis_bounds") and getattr(self, "_opendis_bounds") is not None:
                bmin_mi = np.asarray(self._opendis_bounds["min"], dtype=np.float64)
                bmax_mi = np.asarray(self._opendis_bounds["max"], dtype=np.float64)
            else:
                pts_cell = np.vstack([S0, S1])
                bmin_mi = pts_cell.min(axis=0)
                bmax_mi = pts_cell.max(axis=0)

            L = bmax_mi - bmin_mi
            L_safe = np.where(L > 0.0, L, 1.0)

            d = S1 - S0
            # nearest-integer shift per component: standard minimum-image
            shift = np.rint(d / L_safe)
            for ax in range(3):
                if (not np.isfinite(L[ax])) or (L[ax] <= 0.0):
                    shift[:, ax] = 0.0
            d_mi = d - shift * L
            S1 = S0 + d_mi

            # For consistency across backends, rebuild pairwise node list
            M = S0.shape[0]
            nodes = np.empty((2 * M, 3), dtype=np.float32)
            nodes[0::2] = S0.astype(np.float32, copy=False)
            nodes[1::2] = S1.astype(np.float32, copy=False)
            segs = np.column_stack([
                np.arange(0, 2 * M, 2, dtype=np.int64),
                np.arange(1, 2 * M, 2, dtype=np.int64),
            ])

            Lvec = S1 - S0
            Llen = np.linalg.norm(Lvec, axis=1)
            tvec = np.divide(
                Lvec,
                Llen[:, None],
                out=np.zeros_like(Lvec),
                where=(Llen[:, None] > 0),
            )
            rebuilt_pairwise = True

        # ---------------- Length outlier filter ----------------
        if S0.shape[0] > 0:
            raw_length = np.linalg.norm(S1 - S0, axis=1).astype(np.float64)
        else:
            raw_length = np.zeros((0,), dtype=np.float64)

        do_len_filter = (
            length_filter_percentile is not None
            and float(length_filter_percentile) > 0.0
            and float(length_filter_percentile) < 100.0
            and raw_length.size > 0
        )
        if do_len_filter:
            p = float(length_filter_percentile)
            finite = np.isfinite(raw_length)
            if np.any(finite):
                thr = np.percentile(raw_length[finite], p)
                keep_len = finite & (raw_length <= thr)
                if not np.any(keep_len):
                    keep_len = np.ones_like(raw_length, dtype=bool)

                if rebuilt_pairwise:
                    S0   = S0[keep_len]
                    S1   = S1[keep_len]
                    tvec = tvec[keep_len]
                    bvec = bvec[keep_len]
                    M = S0.shape[0]
                    nodes = np.empty((2 * M, 3), dtype=np.float32)
                    nodes[0::2] = S0.astype(np.float32, copy=False)
                    nodes[1::2] = S1.astype(np.float32, copy=False)
                    segs = np.column_stack([
                        np.arange(0, 2 * M, 2, dtype=np.int64),
                        np.arange(1, 2 * M, 2, dtype=np.int64),
                    ])
                else:
                    segs = segs[keep_len, :]
                    S0   = S0[keep_len]
                    S1   = S1[keep_len]
                    tvec = tvec[keep_len]
                    bvec = bvec[keep_len]

                raw_length = np.linalg.norm(S1 - S0, axis=1).astype(np.float64)

        # ---------------- Per-segment scalar fields ----------------
        length = raw_length.astype(np.float32)
        bmag   = np.linalg.norm(bvec, axis=1).astype(np.float32)
        bt     = np.einsum("ij,ij->i", bvec, tvec, dtype=np.float64)

        character = np.divide(
            bt,
            np.maximum(bmag, 1e-12),
            out=np.zeros_like(bt),
            where=(bmag > 0),
        ).astype(np.float32)

        screw_fraction = np.abs(character).astype(np.float32)
        edge_fraction  = np.sqrt(
            np.clip(1.0 - np.minimum(1.0, screw_fraction ** 2), 0.0, 1.0)
        ).astype(np.float32)
        bt_sign        = np.sign(bt).astype(np.float32)

        thr = {"edge": 0.2, "screw": 0.8}
        if isinstance(type_thresholds, dict):
            thr.update({
                k: float(v)
                for k, v in type_thresholds.items()
                if k in ("edge", "screw")
            })
        abs_chi  = np.abs(character)
        type_code = np.full(character.shape, 2, dtype=np.int32)  # 2=partial
        type_code[abs_chi <= thr["edge"]] = 0                      # 0=edge
        type_code[abs_chi >= thr["screw"]] = 1                     # 1=screw
        bad = (~np.isfinite(character)) | (length <= 1e-12) | (bmag <= 1e-12)
        type_code[bad] = 3                                         # 3=other

        # ---------------- Coloring selection ----------------
        cmode = str(color_mode).lower()
        is_categorical = False
        if cmode in ("uniform", "solid", "constant"):
            C = None
            C_name = "uniform"
        elif cmode in ("signed_burgers", "bt", "screw_component"):
            C = bt.astype(np.float32)
            C_name = "signed_burgers"
        elif cmode in ("burgers_magnitude", "burgers", "bmag", "mag_burgers", "mag-burgers"):
            C = bmag.astype(np.float32)
            C_name = "burgers_magnitude"
        elif cmode in ("length", "len", "l"):
            C = length
            C_name = "length"
        elif cmode in ("defect_type", "type", "category", "classes"):
            C = type_code.astype(np.float32)
            C_name = "defect_type"
            is_categorical = True
        else:
            raise ValueError(
                "color_mode must be one of "
                "'uniform', 'signed_burgers', 'burgers_magnitude', "
                "'length', or 'defect_type'."
            )

        # Colormap and limits
        if cmap is None:
            if C_name == "signed_burgers":
                cmap = "coolwarm"
            elif C_name == "burgers_magnitude":
                cmap = "viridis"
            else:
                cmap = "tab10" if is_categorical else "viridis"

        if (not is_categorical) and (C is not None) and (C.size > 0):
            if clim is None:
                if C_name == "signed_burgers":
                    vmax = float(np.max(np.abs(C))) or 1.0
                    vmin = -vmax
                else:
                    vmin = float(np.nanmin(C))
                    vmax = float(np.nanmax(C))
                    if ((not np.isfinite(vmin)) or
                        (not np.isfinite(vmax)) or
                        vmin == vmax):
                        vmin, vmax = 0.0, 1.0
            else:
                vmin = float(clim[0])
                vmax = float(clim[1])
        else:
            vmin = vmax = None

        # Output path and backend
        out_dir = self.directory if getattr(self, "directory", None) else "."
        os.makedirs(out_dir, exist_ok=True)
        if out_path is None:
            out_path = os.path.join(out_dir, "dislocation_network.vtk")
        backend = str(mode).lower()

        # ---------------- Helpers for AABB overlay ----------------
        def _cube_edges():
            return [
                (0, 1), (0, 2), (0, 3),
                (1, 4), (1, 5),
                (2, 4), (2, 6),
                (3, 5), (3, 6),
                (4, 7), (5, 7), (6, 7),
            ]

        def _corners_from_minmax(bmin, bmax):
            bmin = np.asarray(bmin, dtype=float)
            bmax = np.asarray(bmax, dtype=float)
            return np.array([
                [bmin[0], bmin[1], bmin[2]],
                [bmax[0], bmin[1], bmin[2]],
                [bmin[0], bmax[1], bmin[2]],
                [bmin[0], bmin[1], bmax[2]],
                [bmax[0], bmax[1], bmin[2]],
                [bmax[0], bmin[1], bmax[2]],
                [bmin[0], bmax[1], bmax[2]],
                [bmax[0], bmax[1], bmax[2]],
            ], dtype=float)

        def _get_overlay_corners():
            # sample AABB if provided, else network bounds
            if sample is not None:
                return np.asarray(sample.corners, dtype=float)
            if hasattr(self, "_opendis_bounds") and getattr(self, "_opendis_bounds") is not None:
                bmin = np.asarray(self._opendis_bounds["min"], dtype=float)
                bmax = np.asarray(self._opendis_bounds["max"], dtype=float)
            else:
                if S0.size and S1.size:
                    pts = np.vstack([S0, S1])
                else:
                    pts = np.zeros((2, 3), dtype=float)
                bmin = pts.min(axis=0)
                bmax = pts.max(axis=0)
            return _corners_from_minmax(bmin, bmax)

        # ---------------- 1) Legacy VTK backend (with AABB overlay) ----------------
        def _write_legacy_vtk(path):
            M_net = int(segs.shape[0])
            N_net = int(nodes.shape[0])

            # AABB geometry
            corners = _get_overlay_corners()
            edges   = _cube_edges()
            N_box   = int(corners.shape[0])  # 8
            M_box   = len(edges)             # 12

            # Points: network nodes + 8 AABB corners
            pts_all = np.vstack([
                nodes.astype(np.float64, copy=False),
                corners.astype(np.float64, copy=False),
            ])
            # Lines: network segs + AABB edges (indices offset by N_net)
            segs_box = np.array(
                [[e[0] + N_net, e[1] + N_net] for e in edges],
                dtype=np.int64,
            )
            segs_all = np.vstack([
                segs.astype(np.int64, copy=False),
                segs_box,
            ])

            # Helper for unit directions
            def _unit_dir(a, b):
                v = b - a
                n = np.linalg.norm(v)
                if n > 0:
                    return v / n
                return np.zeros_like(v)

            length_box = np.array([
                np.linalg.norm(corners[j] - corners[i])
                for (i, j) in edges
            ], dtype=np.float32)
            tvec_box = np.array([
                _unit_dir(corners[i], corners[j])
                for (i, j) in edges
            ], dtype=np.float32)
            bvec_box = np.zeros((M_box, 3), dtype=np.float32)
            bmag_box = np.zeros((M_box,),   dtype=np.float32)
            character_box      = np.zeros((M_box,), dtype=np.float32)
            screw_fraction_box = np.zeros((M_box,), dtype=np.float32)
            edge_fraction_box  = np.zeros((M_box,), dtype=np.float32)
            bt_sign_box        = np.zeros((M_box,), dtype=np.float32)
            type_code_box      = np.full((M_box,), 3, dtype=np.int32)
            signed_burgers_box = np.zeros((M_box,), dtype=np.float32)

            if C is None or is_categorical or (C.size == 0):
                coloring_box = np.zeros((M_box,), dtype=np.float32)
            else:
                coloring_box = np.zeros((M_box,), dtype=np.float32)

            # Concatenate network and box fields
            length_all = np.concatenate(
                [length.astype(np.float32, copy=False), length_box],
                axis=0,
            )
            bmag_all = np.concatenate(
                [bmag.astype(np.float32, copy=False), bmag_box],
                axis=0,
            )
            character_all = np.concatenate(
                [character.astype(np.float32, copy=False), character_box],
                axis=0,
            )
            screw_fraction_all = np.concatenate(
                [screw_fraction.astype(np.float32, copy=False),
                 screw_fraction_box],
                axis=0,
            )
            edge_fraction_all = np.concatenate(
                [edge_fraction.astype(np.float32, copy=False),
                 edge_fraction_box],
                axis=0,
            )
            bt_sign_all = np.concatenate(
                [bt_sign.astype(np.float32, copy=False), bt_sign_box],
                axis=0,
            )
            type_code_all = np.concatenate(
                [type_code.astype(np.int32, copy=False), type_code_box],
                axis=0,
            )
            signed_burgers_all = np.concatenate(
                [bt.astype(np.float32, copy=False), signed_burgers_box],
                axis=0,
            )
            tvec_all = np.vstack([
                tvec.astype(np.float32, copy=False),
                tvec_box,
            ])
            bvec_all = np.vstack([
                bvec.astype(np.float32, copy=False),
                bvec_box,
            ])
            if C is None or is_categorical or (C.size == 0):
                coloring_all = np.concatenate(
                    [np.zeros((M_net,), dtype=np.float32), coloring_box],
                    axis=0,
                )
            else:
                C_all = np.concatenate(
                    [C.astype(np.float32, copy=False), coloring_box],
                    axis=0,
                )
                coloring_all = C_all

            is_bounds = np.concatenate(
                [np.zeros((M_net,), dtype=np.int32),
                 np.ones((M_box,), dtype=np.int32)],
                axis=0,
            )

            M_all = int(segs_all.shape[0])
            N_all = int(pts_all.shape[0])

            with open(path, "w") as f:
                f.write("# vtk DataFile Version 4.2\n")
                f.write("Dislocation network with AABB overlay\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")
                f.write(f"POINTS {N_all} float\n")
                for p in pts_all.astype(np.float64):
                    f.write(
                        (float_fmt + " " + float_fmt + " " + float_fmt + "\n")
                        % (p[0], p[1], p[2])
                    )
                f.write(f"LINES {M_all} {3 * M_all}\n")
                for a, b in segs_all:
                    f.write(f"2 {int(a)} {int(b)}\n")

                f.write(f"CELL_DATA {M_all}\n")
                f.write("VECTORS burgers float\n")
                for v in bvec_all.astype(np.float64):
                    f.write(
                        (float_fmt + " " + float_fmt + " " + float_fmt + "\n")
                        % (v[0], v[1], v[2])
                    )
                f.write("VECTORS tangent float\n")
                for v in tvec_all.astype(np.float64):
                    f.write(
                        (float_fmt + " " + float_fmt + " " + float_fmt + "\n")
                        % (v[0], v[1], v[2])
                    )

                def _w_scalar(name, arr):
                    f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                    for v in arr.astype(np.float64):
                        f.write((float_fmt + "\n") % v)

                _w_scalar("length",         length_all)
                _w_scalar("bmag",           bmag_all)
                _w_scalar("character",      character_all)
                _w_scalar("screw_fraction", screw_fraction_all)
                _w_scalar("edge_fraction",  edge_fraction_all)
                _w_scalar("bt_sign",        bt_sign_all)
                _w_scalar("signed_burgers", signed_burgers_all)

                f.write("SCALARS defect_type int 1\nLOOKUP_TABLE default\n")
                for v in type_code_all:
                    f.write(f"{int(v)}\n")

                # mark which cells are the AABB overlay
                f.write("SCALARS is_bounds int 1\nLOOKUP_TABLE default\n")
                for v in is_bounds:
                    f.write(f"{int(v)}\n")

                _w_scalar("coloring", coloring_all.astype(np.float32))

            return path

        # ---------------- 2) PyVista backend (with AABB overlay) ----------------
        def _do_pyvista(path, show_flag):
            try:
                import pyvista as pv

                try:
                    cell_arr = pv.CellArray.from_regular_cells(segs)
                except Exception:
                    lines_long = np.empty((segs.shape[0], 3), dtype=np.int64)
                    lines_long[:, 0] = 2
                    lines_long[:, 1:] = segs
                    cell_arr = lines_long.ravel()

                mesh = pv.PolyData(nodes, lines=cell_arr)

                # Attach data
                mesh.cell_data["length"]         = length
                mesh.cell_data["bmag"]           = bmag
                mesh.cell_data["character"]      = character
                mesh.cell_data["screw_fraction"] = screw_fraction
                mesh.cell_data["edge_fraction"]  = edge_fraction
                mesh.cell_data["bt_sign"]        = bt_sign
                mesh.cell_data["signed_burgers"] = bt.astype(np.float32)
                mesh.cell_data["defect_type"]    = type_code
                mesh.cell_data["burgers"] = bvec.astype(np.float32, copy=False)
                mesh.cell_data["tangent"] = tvec.astype(np.float32, copy=False)

                # Save if file extension matches
                if path.lower().endswith((".vtp", ".vtk")):
                    try:
                        mesh.save(path, binary=True)
                    except TypeError:
                        mesh.save(path)

                if show_flag:
                    pl = pv.Plotter()

                    if C is None or (C.size == 0):
                        pl.add_mesh(
                            mesh,
                            color=uniform_color,
                            render_lines_as_tubes=bool(render_tubes),
                            line_width=float(line_width),
                        )
                    elif is_categorical:
                        palette = ["#34a853", "#1a73e8", "#fbbc05", "#9aa0a6"]
                        try:
                            pl.add_mesh(
                                mesh,
                                scalars="defect_type",
                                cmap=palette,
                                categories=True,
                                clim=(-0.5, 3.5),
                                render_lines_as_tubes=bool(render_tubes),
                                line_width=float(line_width),
                            )
                        except TypeError:
                            pl.add_mesh(
                                mesh,
                                scalars="defect_type",
                                cmap=palette,
                                clim=(-0.5, 3.5),
                                render_lines_as_tubes=bool(render_tubes),
                                line_width=float(line_width),
                            )
                    else:
                        scalar_name = (
                            "signed_burgers"
                            if C_name == "signed_burgers"
                            else ("bmag" if C_name == "burgers_magnitude" else C_name)
                        )
                        pl.add_mesh(
                            mesh,
                            scalars=scalar_name,
                            cmap=cmap,
                            clim=(vmin, vmax) if (vmin is not None) else None,
                            render_lines_as_tubes=bool(render_tubes),
                            line_width=float(line_width),
                        )

                    # AABB overlay
                    corners = _get_overlay_corners()
                    edges   = _cube_edges()
                    lines_long = np.empty((len(edges), 3), dtype=np.int64)
                    lines_long[:, 0] = 2
                    lines_long[:, 1:] = np.array(edges, dtype=np.int64)
                    box = pv.PolyData(corners, lines=lines_long)
                    pl.add_mesh(
                        box,
                        color="black",
                        render_lines_as_tubes=False,
                        line_width=max(1.0, float(line_width) * 0.6),
                        name="aabb_wireframe",
                    )

                    # Optional camera orientation
                    if (elev is not None) or (azim is not None):
                        if sample is not None and clip_to_sample:
                            c = np.asarray(sample.corners, dtype=float)
                            bmin = c.min(axis=0); bmax = c.max(axis=0)
                        elif hasattr(self, "_opendis_bounds") and getattr(self, "_opendis_bounds") is not None:
                            bmin = np.asarray(self._opendis_bounds["min"], dtype=float)
                            bmax = np.asarray(self._opendis_bounds["max"], dtype=float)
                        else:
                            if S0.size and S1.size:
                                pts = np.vstack([S0, S1])
                            else:
                                pts = np.zeros((2, 3), dtype=float)
                            bmin = pts.min(axis=0); bmax = pts.max(axis=0)

                        center = 0.5 * (bmin + bmax)
                        diag   = float(np.linalg.norm(bmax - bmin)) or 1.0
                        dist   = 2.5 * diag

                        el = float(elev if elev is not None else 30.0)
                        az = float(azim if azim is not None else -60.0)
                        ce = np.cos(np.deg2rad(el)); se = np.sin(np.deg2rad(el))
                        ca = np.cos(np.deg2rad(az)); sa = np.sin(np.deg2rad(az))
                        dirvec = np.array([ce * ca, ce * sa, se], dtype=float)
                        pos    = center + dist * dirvec
                        up = (np.array([0.0, 1.0, 0.0], dtype=float)
                              if abs(se) > 0.99
                              else np.array([0.0, 0.0, 1.0], dtype=float))
                        try:
                            pl.camera_position = [
                                tuple(pos.tolist()),
                                tuple(center.tolist()),
                                tuple(up.tolist()),
                            ]
                        except Exception:
                            try:
                                pl.camera.position    = tuple(pos.tolist())
                                pl.camera.focal_point = tuple(center.tolist())
                                pl.camera.view_up     = tuple(up.tolist())
                            except Exception:
                                pass

                    pl.show()

                return path
            except ImportError as e:
                raise RuntimeError(
                    "PyVista is not installed. Use mode='vtk' or 'matplotlib', "
                    "or install with `pip install pyvista`."
                ) from e

        # ---------------- 3) Matplotlib backend (with AABB overlay) ----------------
        def _do_matplotlib(path_png, show_flag):
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Line3DCollection
            from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
            from matplotlib.cm import ScalarMappable

            segments = np.stack(
                [S0.astype(np.float32), S1.astype(np.float32)], axis=1
            )
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection="3d")

            lc = Line3DCollection(segments, linewidths=float(line_width))
            cbar = None
            if C is None or (C.size == 0):
                lc.set_color(uniform_color)
            elif is_categorical:
                colors = np.array([
                    [0.2039, 0.6588, 0.3255, 1.0],
                    [0.1020, 0.4510, 0.9020, 1.0],
                    [0.9843, 0.7373, 0.2039, 1.0],
                    [0.6039, 0.6039, 0.6039, 1.0],
                ])
                cmap_cat = ListedColormap(colors)
                bounds   = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
                norm     = BoundaryNorm(bounds, cmap_cat.N)
                lc.set_cmap(cmap_cat)
                lc.set_norm(norm)
                lc.set_array(type_code.astype(float))
                sm = ScalarMappable(norm=norm, cmap=cmap_cat)
                sm.set_array(type_code)
                cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3], pad=0.02)
                cbar.ax.set_yticklabels(["edge", "screw", "partial", "other"])
            else:
                norm = Normalize(vmin=vmin, vmax=vmax)
                lc.set_cmap(cmap)
                lc.set_norm(norm)
                lc.set_array(C)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array(C)
                label_map = {
                    "signed_burgers": "b.t",
                    "burgers_magnitude": "|b|",
                    "length": "segment length",
                }
                label = label_map.get(C_name, C_name)
                cbar = fig.colorbar(sm, ax=ax, pad=0.02)
                cbar.set_label(label)
            ax.add_collection3d(lc)

            # AABB overlay
            corners = _get_overlay_corners().astype(np.float32)
            edges   = _cube_edges()
            aabb_segs = np.array(
                [[corners[i], corners[j]] for (i, j) in edges],
                dtype=np.float32,
            )
            aabb_lc = Line3DCollection(
                aabb_segs,
                colors="k",
                linewidths=max(1.0, float(line_width) * 0.6),
            )
            ax.add_collection3d(aabb_lc)

            # Axes limits
            if sample is not None and clip_to_sample:
                c = np.asarray(sample.corners, dtype=float)
                bmin = c.min(axis=0); bmax = c.max(axis=0)
                ax.set_xlim(bmin[0], bmax[0])
                ax.set_ylim(bmin[1], bmax[1])
                ax.set_zlim(bmin[2], bmax[2])
            elif hasattr(self, "_opendis_bounds") and getattr(self, "_opendis_bounds") is not None:
                bmin = np.asarray(self._opendis_bounds["min"], dtype=float)
                bmax = np.asarray(self._opendis_bounds["max"], dtype=float)
                ax.set_xlim(bmin[0], bmax[0])
                ax.set_ylim(bmin[1], bmax[1])
                ax.set_zlim(bmin[2], bmax[2])
            else:
                if S0.size and S1.size:
                    pts = np.vstack([S0, S1])
                else:
                    pts = np.zeros((2, 3))
                ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
                ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
                ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())

            # Equal aspect if possible
            try:
                xr = ax.get_xlim()
                yr = ax.get_ylim()
                zr = ax.get_zlim()
                ax.set_box_aspect(
                    (abs(xr[1] - xr[0]),
                     abs(yr[1] - yr[0]),
                     abs(zr[1] - zr[0]))
                )
            except Exception:
                pass

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            if (elev is not None) or (azim is not None):
                cur_elev = getattr(ax, "elev", 30)
                cur_azim = getattr(ax, "azim", -60)
                ax.view_init(
                    elev=(elev if elev is not None else cur_elev),
                    azim=(azim if azim is not None else cur_azim),
                )

            if path_png is not None:
                fig.savefig(path_png, dpi=300, bbox_inches="tight")
            if show_flag:
                plt.show()
            else:
                plt.close(fig)
            return path_png

        # ---------------- Dispatch ----------------
        written = None
        if backend == "vtk":
            written = _write_legacy_vtk(out_path)
        elif backend == "pyvista":
            written = _do_pyvista(out_path, bool(show))
        elif backend == "matplotlib":
            written = _do_matplotlib(out_path, bool(show))
        else:
            raise ValueError("mode must be 'vtk', 'pyvista', or 'matplotlib'.")

        return {
            "points":   int(nodes.shape[0]),
            "segments": int(segs.shape[0]),
            "path":     written,
            "backend":  backend,
            "coloring": cmode,
        }
    
    ## Properties
    @property
    def stacking_faults(self):
        """stacking_fault: The stacking fault object, or None if not initialized."""
        if self._stacking_faults is None:
            print("self._stacking_faults has not been initialized yet")
        return self._stacking_faults

    @property
    def cracks(self):
        """crack: The crack object, or None if not initialized."""
        if self._cracks is None:
            print("self._cracks has not been initialized yet")
        return self._cracks

    @property
    def point_defects(self):
        """point_defect: The point defect object, or None if not initialized."""
        if self._point_defects is None:
            print("self._point_defects has not been initialized yet")
        return self._point_defects

    # -------------------------------------------------------------------------
    # Sub-Classes
    # -------------------------------------------------------------------------
    class stacking_fault(logging):
        """
        Represents stacking faults in a crystalline sample.

        Stacking faults are planar defects where the regular stacking sequence
        of atomic planes is disrupted. This class handles the creation,
        positioning, and application of stacking faults to atomic samples.

        Attributes:
            directory: Output directory for modified sample data.
            fault_number: Number of stacking fault planes.
            fault_offset: Offset vector for fault positioning.
            fault_normal: Normalized normal vector to fault planes.
            interfault_spacing: Spacing between consecutive fault planes.
            burgers_vector: Burgers vector defining the fault displacement.
            fault_orientation: Array of orientations (+1/-1) for each fault.
            fault_gap: Gap size at each fault plane.
        """

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
        def __init__(self, directory, fault_number, fault_offset, fault_normal,
                     interfault_spacing, burgers_vector, fault_orientation, fault_gap):
            """
            Initialize a stacking fault configuration.

            Args:
                directory: Output directory for storing modified sample data.
                fault_number: Number of stacking fault planes to create.
                fault_offset: 3D offset vector for positioning faults in the sample.
                fault_normal: 3D normal vector to the stacking fault planes.
                interfault_spacing: Distance between consecutive fault planes.
                burgers_vector: 3D Burgers vector defining the fault displacement.
                fault_orientation: List/array of orientations (+1 or -1) for each
                    fault plane. Values are cycled if fewer than fault_number.
                fault_gap: Gap size to add at each fault plane crossing.
            """
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
        def generate_global_positions(self, sample, crystal, plotting=False, use_gpu=True):
            """
            Calculate the global positions of stacking fault planes.

            Computes the positions of stacking fault planes in the sample
            coordinate system based on the crystal orientation and sample geometry.

            Args:
                sample: Sample object providing offset and geometry information.
                crystal: Crystal object providing lattice transformation matrices.
                plotting: If True, display a plot of the fault planes. Defaults to False.
                use_gpu: If True and CuPy is available, prepare GPU arrays for
                    later processing. Defaults to True.
            """
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
            
        def plot_global_positions(self, sample, color='c', alpha=0.5, elev=0, azim=0):
            """
            Plot stacking fault planes intersecting the sample cuboid.

            Visualizes one or more stacking fault planes (all sharing the same
            normal vector) intersecting the sample cuboid defined by its corners.

            Args:
                sample: Sample object with a 'corners' attribute of shape (8, 3).
                color: Matplotlib color for the fault plane polygons. Defaults to 'c'.
                alpha: Transparency of the fault plane polygons. Defaults to 0.5.
                elev: Elevation angle for 3D view. Defaults to 0.
                azim: Azimuth angle for 3D view. Defaults to 0.

            Returns:
                tuple: (fig, ax) matplotlib figure and axes objects.
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

        def apply_to_sample(self, sample, use_gpu=True):
            """
            Apply stacking faults to all chunks of a sample.

            Iterates through all sample chunks, applies stacking fault
            displacements to atomic positions, and writes the modified data.

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True and CuPy is available, use GPU acceleration.
                    Defaults to True.
            """
            for i in range(sample.chunk_total):
                if cp is not None and use_gpu:
                    positions_chunk_cp = sample.load_chunk_positions(i+1,use_gpu=True)
                    positions_chunk_cp = self.apply_stacking_fault_chunk(positions_chunk_cp,use_gpu=True)
                    positions_chunknp = cp.asnumpy(positions_chunk_cp)
                else:
                    positions_chunknp = sample.load_chunk_positions(i+1,use_gpu=False)
                    positions_chunknp = self.apply_stacking_fault_chunk(positions_chunknp,use_gpu=False)

                sample.write_chunk_positions(positions_chunknp,i+1,override_directory=self.directory)
                if self.directory is not None:
                    species_chunknp = sample.load_chunk_species(i + 1, use_gpu=False)
                    sample.write_chunk_species(species_chunknp,i+1,override_directory=self.directory)
            sample.write_sample_metadata(override_directory=self.directory)
            
        def apply_stacking_fault_chunk(self, positions_chunk, use_gpu=True):
            """
            Apply stacking fault displacements to a single chunk of positions.

            Shifts atoms that lie beyond each fault plane by the Burgers vector,
            accounting for the fault orientation. Adds a small gap at each fault
            plane crossing.

            Args:
                positions_chunk: Array of atomic positions with shape (N, 3).
                    Can be numpy array or cupy array depending on use_gpu.
                use_gpu: If True and CuPy is available, perform computation on GPU.
                    Defaults to True.

            Returns:
                Modified positions array with stacking fault displacements applied.
                Same type (numpy or cupy) as the input.
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
        """
        Represents a crack defect as a convex hull region.

        Cracks are modeled as convex hulls in 3D space. Atoms falling inside
        the hull are removed when the crack is applied to a sample.

        Attributes:
            directory: Output directory for modified sample data.
            crack_points: Array of vertices defining the convex hull.
            hull: scipy.spatial.ConvexHull object for the crack geometry.
            hull_equations: Plane equations for the hull facets.
        """

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
            Initialize a crack defect from convex hull vertices.

            Args:
                directory: Output directory for storing modified sample data.
                crack_points: Array-like of shape (N, 3) defining the vertices
                    of a convex hull in 3D. The hull must be convex.
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
            Apply the crack to all chunks of a sample.

            Iterates through all sample chunks and removes atoms that fall
            inside the crack's convex hull.

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True and CuPy is available, use GPU acceleration.
                    Defaults to True.
            """
            for i in range(sample.chunk_total):
                if cp is not None and use_gpu:
                    positions_chunk_cp = sample.load_chunk_positions(i + 1, use_gpu=True)
                    species_chunknp = sample.load_chunk_species(i + 1, use_gpu=False)
                    positions_chunk_cp, species_chunknp = self.apply_crack_chunk(positions_chunk_cp,species_chunknp,use_gpu=True)
                    positions_chunknp = cp.asnumpy(positions_chunk_cp)
                else:
                    positions_chunknp = sample.load_chunk_positions(i + 1, use_gpu=False)
                    species_chunknp = sample.load_chunk_species(i + 1, use_gpu=False)
                    positions_chunknp, species_chunknp = self.apply_crack_chunk(positions_chunknp,species_chunknp,use_gpu=False)

                sample.write_chunk_positions(positions_chunknp,i+1,override_directory=self.directory)
                sample.write_chunk_species(species_chunknp,i+1,override_directory=self.directory)
            sample.write_sample_metadata(override_directory=self.directory)

        def apply_crack_chunk(self, positions_chunk, species_chunknp, use_gpu=True):
            """
            Remove atoms inside the crack's convex hull from a single chunk.

            Tests positions against the half-space inequalities defined by the
            convex hull facets and removes atoms that satisfy all inequalities
            (i.e., are inside the hull).

            Args:
                positions_chunk: Array of atomic positions with shape (N, 3).
                    Can be numpy array or cupy array depending on use_gpu.
                species_chunknp: Numpy array of species labels with shape (N,).
                use_gpu: If True and CuPy is available, perform computation on GPU.
                    Defaults to True.

            Returns:
                tuple: (positions, species) arrays with atoms inside the crack removed.
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
                species_chunknp = species_chunknp[~(inside_mask.get())]
                return positions_chunk, species_chunknp
            else:
                eq = self.hull_equations
                normals = eq[:, :3]
                offsets = eq[:, 3]
                dot_vals = normals @ positions_chunk.T + offsets[:, None]
                inside_mask = np.all(dot_vals <= 1e-12, axis=0)
                positions_chunk = positions_chunk[~inside_mask]
                species_chunknp = species_chunknp[~inside_mask]
                return positions_chunk, species_chunknp
        
        def plot_crack_geometry(self, sample, color='r', alpha=0.5, elev=0, azim=0):
            """
            Plot the crack geometry within the sample bounding box.

            Displays the sample as a wireframe and overlays the triangular
            facets of the crack's convex hull.

            Args:
                sample: Sample object with a 'corners' attribute of shape (8, 3).
                color: Matplotlib color for the crack facets. Defaults to 'r'.
                alpha: Transparency of the crack facets. Defaults to 0.5.
                elev: Elevation angle for 3D view. Defaults to 0.
                azim: Azimuth angle for 3D view. Defaults to 0.

            Returns:
                tuple: (fig, ax) matplotlib figure and axes objects.
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
        Handles point defects: vacancies, substitutions, and interstitials.

        Creates and applies point defects to atomic samples. Supports both
        random and specific placement of defects, with optional local relaxation.

        Attributes:
            directory: Output directory for modified sample data.
            seed: Random seed for reproducibility.
            region_min: Minimum corner of the region for defect placement.
            region_max: Maximum corner of the region for defect placement.
            vacancy_fraction: Fraction of candidate atoms to remove as vacancies.
            vacancy_count: Exact number of vacancies to create.
            substitution_from: Species to replace in substitutions.
            substitution_to: Replacement species for substitutions.
            interstitial_count: Number of interstitial atoms to add.
            interstitial_species: Species of interstitial atoms.

        Note:
            Works chunk-by-chunk using sample.load_chunk_positions/species and
            the paired write methods, same as stacking_fault/crack. The relaxation
            step is local to the atoms present in each chunk.
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
                     vacancy_fraction=None,
                     vacancy_count=None,
                     vacancy_global_indices=None,
                     vacancy_positions=None,
                     vacancy_species_filter=None,
                     substitution_fraction=None,
                     substitution_count=None,
                     substitution_from=None,
                     substitution_to=None,
                     substitution_positions=None,
                     substitution_global_indices=None,
                     interstitial_count=None,
                     interstitial_positions=None,
                     interstitial_species=None,
                     interstitial_min_separation=None,
                     relax_after=False,
                     relax_params=None):
            """
            Initialize a point defect configuration.

            Args:
                directory: Output directory for storing modified sample data.
                seed: Random seed for reproducibility. Defaults to None.
                region_min: 3D array defining minimum corner of defect region.
                    If None, applies to entire sample.
                region_max: 3D array defining maximum corner of defect region.
                    If None, applies to entire sample.
                vacancy_fraction: Fraction of eligible atoms to remove (0-1).
                vacancy_count: Exact number of vacancies to create.
                vacancy_global_indices: List of global atom indices to remove.
                vacancy_positions: Array of (N, 3) positions to match for removal.
                vacancy_species_filter: List of species eligible for vacancies.
                substitution_fraction: Fraction of eligible atoms to substitute.
                substitution_count: Exact number of substitutions to make.
                substitution_from: Species to replace.
                substitution_to: Replacement species.
                substitution_positions: Array of (N, 3) positions to substitute.
                substitution_global_indices: List of global indices to substitute.
                interstitial_count: Number of random interstitials to add.
                interstitial_positions: Array of (N, 3) specific interstitial positions.
                interstitial_species: Species of interstitial atoms.
                interstitial_min_separation: Minimum separation from existing atoms.
                relax_after: If True, perform local relaxation after applying defects.
                relax_params: Dictionary of relaxation parameters.
            """
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
            """
            Add random vacancies to the defect configuration.

            Args:
                fraction: Fraction of eligible atoms to remove (0-1).
                count: Exact number of vacancies to create.
                species_filter: List of species eligible for vacancy creation.

            Returns:
                self: For method chaining.
            """
            if fraction is not None: self.vacancy_fraction = float(fraction)
            if count is not None: self.vacancy_count = int(count)
            if species_filter is not None: self.vacancy_species_filter = list(species_filter)
            return self

        def add_specific_vacancies(self, positions=None, global_indices=None):
            """
            Add vacancies at specific positions or indices.

            Args:
                positions: Array of (N, 3) positions to match for removal.
                global_indices: List of global atom indices to remove.

            Returns:
                self: For method chaining.
            """
            if positions is not None:
                self.vacancy_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            if global_indices is not None:
                self.vacancy_global_indices = list(global_indices)
            return self

        def add_random_substitutions(self, fraction=None, count=None, from_species=None, to_species=None):
            """
            Add random substitutional defects.

            Args:
                fraction: Fraction of eligible atoms to substitute (0-1).
                count: Exact number of substitutions to make.
                from_species: Species to replace.
                to_species: Replacement species.

            Returns:
                self: For method chaining.
            """
            if fraction is not None: self.substitution_fraction = float(fraction)
            if count is not None: self.substitution_count = int(count)
            if from_species is not None: self.substitution_from = from_species
            if to_species is not None: self.substitution_to = to_species
            return self

        def add_specific_substitutions(self, positions=None, global_indices=None, to_species=None, from_species=None):
            """
            Add substitutions at specific positions or indices.

            Args:
                positions: Array of (N, 3) positions to substitute.
                global_indices: List of global atom indices to substitute.
                to_species: Replacement species.
                from_species: Species to replace.

            Returns:
                self: For method chaining.
            """
            if positions is not None:
                self.substitution_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            if global_indices is not None:
                self.substitution_global_indices = list(global_indices)
            if to_species is not None:
                self.substitution_to = to_species
            if from_species is not None:
                self.substitution_from = from_species
            return self

        def add_random_interstitials(self, count, species, min_separation=0.0):
            """
            Add random interstitial atoms.

            Args:
                count: Number of interstitial atoms to add.
                species: Species of the interstitial atoms.
                min_separation: Minimum separation from existing atoms.
                    Defaults to 0.0.

            Returns:
                self: For method chaining.
            """
            self.interstitial_count = int(count)
            self.interstitial_species = species
            self.interstitial_min_separation = float(min_separation)
            return self

        def add_specific_interstitials(self, positions, species):
            """
            Add interstitial atoms at specific positions.

            Args:
                positions: Array of (N, 3) interstitial positions.
                species: Species of the interstitial atoms.

            Returns:
                self: For method chaining.
            """
            self.interstitial_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
            self.interstitial_species = species
            return self

        # ------------------------
        # Core pipeline

        def apply_to_sample(self, sample, use_gpu=False, tol_match=1e-4):
            """
            Apply all configured point defects to a sample.

            Streams over chunks and applies:
                - Random and specific vacancy deletions
                - Random and specific substitutions (species swap)
                - Random and specific interstitial insertions

            Then optionally performs local relaxation and writes back modified
            positions and species arrays.

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True, use GPU acceleration where available.
                    Defaults to False.
                tol_match: Tolerance for position matching when applying
                    specific defects by position. Defaults to 1e-4.

            Note:
                Uses sample.load_chunk_positions/species and write_* counterparts,
                preserving the chunked .npy layout.
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
            Perform local relaxation around defect sites.

            Light-weight local relaxation around recorded defect centers using
            a simple radial update rule:
                - Interstitials push neighbors away
                - Vacancies pull neighbors toward the vacancy site
                - Substitutions pull neighbors mildly (size heuristic)

            Args:
                sample: Sample object for loading/writing chunk data.
                r_cut: Cutoff radius for neighbor interactions. Defaults to 2.0.
                strength: Base displacement strength. Defaults to 0.05.
                iterations: Number of relaxation iterations. Defaults to 2.
                decay: Strength decay factor per iteration. Defaults to 0.8.
                use_gpu: If True, use GPU acceleration (currently not implemented).
                    Defaults to False.

            Note:
                The update rule per center c and neighbor x within r_cut is:
                    w = exp(-(r/r_cut)^2)
                    delta = sgn * strength * w * (x - c) / (r + 1e-12)
                where sgn is +1 for interstitials (push), -1 for vacancies (pull),
                and -0.5 for substitutions. Strength decays each iteration by decay.
                Operates chunk-by-chunk; atoms are clamped to sample AABB.
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
            Visualize point defects in the sample.

            Creates a 3D scatter plot showing vacancies, substitutions, and
            interstitials within the sample bounding box.

            Args:
                sample: Sample object with a 'corners' attribute of shape (8, 3).
                elev: Elevation angle for 3D view. Defaults to 15.
                azim: Azimuth angle for 3D view. Defaults to -60.
                size: Figure size in inches. Defaults to 8.

            Returns:
                tuple: (fig, ax) matplotlib figure and axes objects.
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
            """
            Create a boolean mask for positions within the defined region.

            Args:
                positions: Array of positions with shape (N, 3).

            Returns:
                numpy.ndarray: Boolean mask of shape (N,). True if inside region.
            """
            if self.region_min is None or self.region_max is None:
                return np.ones(positions.shape[0], dtype=bool)
            p = positions
            r = (p[:, 0] >= self.region_min[0]) & (p[:, 0] <= self.region_max[0]) & \
                (p[:, 1] >= self.region_min[1]) & (p[:, 1] <= self.region_max[1]) & \
                (p[:, 2] >= self.region_min[2]) & (p[:, 2] <= self.region_max[2])
            return r

        def _in_region_mask(self, P):
            """
            Check if positions are within the defined region.

            Args:
                P: Array of positions with shape (N, 3).

            Returns:
                numpy.ndarray: Boolean mask of shape (N,). True if inside region.
            """
            if self.region_min is None or self.region_max is None:
                return np.ones(P.shape[0], dtype=bool)
            r = (P[:, 0] >= self.region_min[0]) & (P[:, 0] <= self.region_max[0]) & \
                (P[:, 1] >= self.region_min[1]) & (P[:, 1] <= self.region_max[1]) & \
                (P[:, 2] >= self.region_min[2]) & (P[:, 2] <= self.region_max[2])
            return r

        def _vacancy_candidate_mask(self, pos, spc, region_mask):
            """
            Identify atoms eligible for vacancy creation.

            Args:
                pos: Array of positions with shape (N, 3).
                spc: Array of species labels with shape (N,).
                region_mask: Boolean mask for region filtering.

            Returns:
                numpy.ndarray: Boolean mask of shape (N,).
            """
            mask = region_mask.copy()
            if self.vacancy_species_filter is not None:
                spc_obj = spc.astype(object, copy=False)
                allowed = np.zeros(spc_obj.shape[0], dtype=bool)
                for s in self.vacancy_species_filter:
                    allowed |= (spc_obj == s)
                mask &= allowed
            return mask

        def _substitution_candidate_mask(self, pos, spc, region_mask):
            """
            Identify atoms eligible for substitution.

            Args:
                pos: Array of positions with shape (N, 3).
                spc: Array of species labels with shape (N,).
                region_mask: Boolean mask for region filtering.

            Returns:
                numpy.ndarray: Boolean mask of shape (N,).
            """
            mask = region_mask.copy()
            if self.substitution_from is not None:
                spc_obj = spc.astype(object, copy=False)
                mask &= (spc_obj == self.substitution_from)
            return mask

        def _indices_from_positions(self, pos, target_positions, tol=1e-4):
            """
            Find atoms matching target positions within tolerance.

            Args:
                pos: Array of positions with shape (N, 3).
                target_positions: Array of target positions with shape (M, 3).
                tol: Distance tolerance for matching. Defaults to 1e-4.

            Returns:
                numpy.ndarray: Boolean mask of shape (N,).
            """
            sel = np.zeros(pos.shape[0], dtype=bool)
            if target_positions.size == 0:
                return sel
            for t in target_positions:
                d = pos - t[None, :]
                r2 = np.sum(d * d, axis=1)
                hit = np.where(r2 <= float(tol * tol))[0]
                if hit.size > 0:
                    sel[hit[0]] = True
            return sel

        def _ok_min_sep(self, p, pos, r_min):
            """
            Check if a point satisfies minimum separation from all positions.

            Args:
                p: Single position with shape (3,).
                pos: Array of existing positions with shape (N, 3).
                r_min: Minimum required separation distance.

            Returns:
                bool: True if minimum separation is satisfied.
            """
            if pos.shape[0] == 0:
                return True
            d = pos - p[None, :]
            r2 = np.sum(d * d, axis=1)
            return bool(np.all(r2 >= float(r_min * r_min)))

        def _region_or_sample_box(self, sample):
            """
            Get the bounding box for defect placement.

            Args:
                sample: Sample object providing dimensions and offset.

            Returns:
                tuple: (min_corner, max_corner) as numpy arrays.
            """
            if self.region_min is not None and self.region_max is not None:
                return self.region_min.copy(), self.region_max.copy()
            # sample box: centered at offset with lengths=dimensions
            dims = sample.dimensions.astype(np.float32)
            mn = (sample.offset - 0.5 * dims).astype(np.float32)
            mx = mn + dims
            return mn, mx

        def _accumulate_radial_disp(self, pos, centers, r_cut, step_signed):
            """
            Accumulate radial displacements from defect centers.

            Args:
                pos: Array of positions with shape (N, 3).
                centers: Array of defect center positions with shape (M, 3).
                r_cut: Cutoff radius for interactions.
                step_signed: Signed displacement step (+1 for push, -1 for pull).

            Returns:
                numpy.ndarray: Accumulated displacement vectors with shape (N, 3).
            """
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
                w = np.exp(-(r[mask] / rc) ** 2).astype(np.float32)
                acc[mask] += (step_signed * w)[:, None] * u
            return acc
