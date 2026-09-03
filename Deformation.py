# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import os
import gc
import json
import threading
from Logging import logging
import numpy as np
try:
    import cupy as cp  # Optional GPU backend
except ImportError:
    cp = None
from cffi import FFI
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED


# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class deformation(logging):

    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "import_deformation_field",
        "clip_field",
        "clip_field_to_sample",
        "apply_deformation_chunked",
        "import_fe_nodal_field",
        "import_fe_connectivity",
        "apply_fe_nodal_field",
        "transform_fe_nodal_field",
        "plot_field_and_sample_edges_3d",
        "plot_mesh_and_sample_edges_3d",
    )
    
    # -------------------------------------------------------------------------
    # Functions
    # -------------------------------------------------------------------------
    # Initialization
    def __init__(self, directory=None):
        """
        Initialize the deformation helper.

        Args:
            directory (str or None): Optional directory. If provided and it
                does not exist, it is created.
        """
        super().__init__(log_name="deformation")
        self.directory = directory
        self._Xref = None         # shape (N, 3) reference nodal coordinates
        self._Xcurr = None        # shape (N, 3) current nodal coordinates
        self._elem_nodes = None   # shape (E, k) element connectivity (0-based)
        self._Xref_import = None  # import-order reference coordinates (host, float64)
        self._mesh_points = None  # COMSOL mesh point coordinates (host, float64)
        self._fe_field_file = None
        self._fe_mesh_file = None
        self._fe_position_scale = 1.0
        self._fe_nodes_matched = False
        if self.directory is not None and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    # Deformation Gradient Field
    def import_deformation_field(
        self,
        filepath,
        columns=None,
        preset=None,
        delimiter=None,
        header_lines=None,
        position_scale=1.0,
        use_gpu=True,
        comments="%",
        drop_nan_rows=True,
        dtype=np.float32,
    ):
        """
        Import a deformation gradient tensor field from a text file.

        This loader reads tabular text data containing x, y, z coordinates and
        the 3x3 deformation gradient components F11..F33. It returns two arrays:
        positions of shape (N, 3) and F of shape (N, 9). The 9 entries are in
        row-major order:
            [F11, F12, F13, F21, F22, F23, F31, F32, F33].

        You can supply an explicit column mapping via `columns` or choose a
        ready-made `preset`. Any values provided in `columns` override the
        chosen preset.

        Args:
            filepath (str): Path to the input text file.
            columns (dict or sequence, optional): Either a dict mapping the keys
                {"x","y","z","F11","F12","F13","F21","F22","F23","F31","F32","F33"}
                to 0-based column indices, or a sequence of 12 indices in the
                order [x, y, z, F11, F12, F13, F21, F22, F23, F31, F32, F33].
                If None, a preset must be provided or the function assumes the
                default sequential order 0..11.
            preset (str, optional): Name of a preset that can also supply
                defaults for delimiter, header_lines, and columns.
            delimiter (str or None, optional): Column delimiter. Use None for
                any whitespace. If not provided, taken from the preset.
            header_lines (int, optional): Number of header lines to skip. If not
                provided, taken from the preset. Defaults to 0 if nothing else
                is specified.
            position_scale (float, optional): Multiply positions by this factor
                to convert to desired units. For example, 1e-3 converts mm to m.
                Defaults to 1.0.
            use_gpu (bool, optional): If True and CuPy is available, return CuPy
                arrays; otherwise return NumPy arrays. Defaults to True.
            comments (str, optional): Comment character for NumPy text loading.
                Defaults to "%".
            drop_nan_rows (bool, optional): If True, drop any row containing NaN
                in the selected columns. If False and NaNs are present, a
                ValueError is raised. Defaults to True.
            dtype (numpy dtype, optional): Floating dtype of returned arrays.
                Defaults to np.float32.

        Returns:
            tuple:
                positions (ndarray): Shape (N, 3).
                F (ndarray): Shape (N, 9), row-major deformation gradient.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the column mapping is invalid or NaNs are present
                and drop_nan_rows is False.

        Notes:
            - Column indices are 0-based.
            - F is dimensionless; positions are scaled by `position_scale`.
            - Arrays are made contiguous and cast to `dtype`.
        """
        import os

        if not os.path.isfile(filepath):
            raise FileNotFoundError("File not found: {}".format(filepath))

        # Common import presets; adjust or extend as needed for your sources.
        DEFORMATION_IMPORT_PRESETS = {
            "comsol_csv": {
                "delimiter": ",",
                "header_lines": 6,
                "columns": {
                    "x": 0, "y": 1, "z": 2,
                    "F11": 3, "F12": 4, "F13": 5,
                    "F21": 6, "F22": 7, "F23": 8,
                    "F31": 9, "F32": 10, "F33": 11,
                },
            },
            "abaqus_dat": {
                "delimiter": None,
                "header_lines": 0,
                "columns": {
                    "x": 0, "y": 1, "z": 2,
                    "F11": 3, "F12": 4, "F13": 5,
                    "F21": 6, "F22": 7, "F23": 8,
                    "F31": 9, "F32": 10, "F33": 11,
                },
            },
            "ansys_csv": {
                "delimiter": ",",
                "header_lines": 1,
                "columns": {
                    "x": 0, "y": 1, "z": 2,
                    "F11": 6, "F12": 7, "F13": 8,
                    "F21": 9, "F22": 10, "F23": 11,
                    "F31": 12, "F32": 13, "F33": 14,
                },
            },
        }

        # 1) Resolve preset defaults, then apply user overrides.
        preset_cfg = DEFORMATION_IMPORT_PRESETS.get(preset, {}) if preset else {}
        delim = delimiter if delimiter is not None else preset_cfg.get("delimiter", None)
        skip = header_lines if header_lines is not None else int(preset_cfg.get("header_lines", 0))

        required_keys = [
            "x", "y", "z",
            "F11", "F12", "F13",
            "F21", "F22", "F23",
            "F31", "F32", "F33",
        ]

        # Normalize columns specification to a dict mapping names -> indices.
        colmap = None
        if isinstance(columns, dict):
            colmap = dict(columns)
        elif columns is None:
            preset_cols = preset_cfg.get("columns", None)
            if isinstance(preset_cols, dict):
                colmap = dict(preset_cols)
            elif isinstance(preset_cols, (list, tuple)) and len(preset_cols) == 12:
                colmap = dict(zip(required_keys, preset_cols))
            else:
                # Default to sequential 0..11 if nothing provided.
                colmap = dict(zip(required_keys, range(12)))
        else:
            # columns is a sequence of length 12.
            if not hasattr(columns, "__len__") or len(columns) != 12:
                raise ValueError("columns must be a dict or a 12-length sequence")
            colmap = dict(zip(required_keys, list(columns)))

        # Validate and build ordered usecols in the exact order we want back.
        try:
            usecols = [int(colmap[k]) for k in required_keys]
        except KeyError as e:
            raise ValueError("Missing column mapping for key: {}".format(str(e))) from None
        if len(set(usecols)) != 12:
            raise ValueError("Column indices must be unique across x,y,z and F entries")

        # 2) Load the selected columns from text on CPU; move to GPU later if requested.
        try:
            data = np.genfromtxt(
                filepath,
                delimiter=delim,
                comments=comments,
                skip_header=skip,
                usecols=usecols,
                dtype=np.float64,   # parse in float64 to minimize round-off
                invalid_raise=False,
            )
        except Exception as exc:
            raise ValueError("Failed to parse {}: {}".format(filepath, exc))

        if data is None:
            raise ValueError("No data read from file: {}".format(filepath))

        # Ensure 2D shape even for single-row files.
        data = np.atleast_2d(data)
        if data.shape[1] != 12:
            raise ValueError("Expected 12 columns after selection, got {}".format(data.shape[1]))

        # 3) Optional NaN handling.
        if np.isnan(data).any():
            if drop_nan_rows:
                mask = ~np.isnan(data).any(axis=1)
                data = data[mask, :]
            else:
                raise ValueError("NaNs detected in the selected columns")

        # 4) Split and scale.
        pos = data[:, 0:3].astype(dtype, copy=False)
        if position_scale != 1.0:
            pos *= np.float32(position_scale) if dtype == np.float32 else np.float64(position_scale)
        F = data[:, 3:].astype(dtype, copy=False)  # shape (N, 9), row-major

        # 5) Move to GPU if requested and available.
        if use_gpu and (cp is not None):
            pos = cp.asarray(pos, dtype=dtype)
            F = cp.asarray(F, dtype=dtype)

        return pos, F

    # Field transforms (CPU/GPU)
    def _select_backend(self, use_gpu):
        """
        Select numpy or cupy module.

        Args:
            use_gpu (bool): If True and CuPy is available, return CuPy; else NumPy.

        Returns:
            module: numpy or cupy.
        """
        return cp if (use_gpu and (cp is not None)) else np

    def _infer_dtype(self, positions, F, dtype):
        """
        Infer a floating dtype from inputs or fallback to float32.

        Args:
            positions (ndarray or None): Position array used to infer dtype.
            F (ndarray or None): Tensor array used to infer dtype.
            dtype (numpy dtype or None): Explicit dtype override.

        Returns:
            numpy.dtype: Resolved dtype.
        """
        if dtype is not None:
            return np.dtype(dtype)
        if hasattr(positions, "dtype"):
            return positions.dtype
        if hasattr(F, "dtype"):
            return F.dtype
        return np.float32

    # rotation utilities
    def build_rotation_matrix(
        self,
        rotate_axis=None,
        rotate_angle=None,
        rotate_matrix=None,
        degrees=True,
        use_gpu=True,
        dtype=None,
    ):
        """
        Build a 3x3 rotation matrix.

        Either construct from axis-angle or validate and return a provided
        rotation matrix.

        Args:
            rotate_axis (sequence or None): Length-3 axis for axis-angle rotation.
            rotate_angle (float or None): Rotation angle. Interpreted in degrees
                if `degrees` is True.
            rotate_matrix (array-like or None): If provided, use this 3x3 matrix
                directly.
            degrees (bool): Interpret rotate_angle in degrees if True.
            use_gpu (bool): Return a CuPy array if True and CuPy is available.
            dtype (numpy dtype or None): Floating dtype. If None, use float32.

        Returns:
            ndarray or None: 3x3 rotation matrix in the selected backend, or None
            if no rotation is requested.

        Raises:
            ValueError: If rotate_matrix is not 3x3, or axis is zero-length.
        """
        xp = self._select_backend(use_gpu)
        dtype = np.dtype(np.float32 if dtype is None else dtype)

        if rotate_matrix is not None:
            R = xp.asarray(rotate_matrix, dtype=dtype)
            if R.shape != (3, 3):
                raise ValueError("rotate_matrix must be 3x3")
            return R

        if (rotate_axis is None) or (rotate_angle is None):
            return None

        # Normalize axis and construct Rodrigues rotation.
        axis = xp.asarray(rotate_axis, dtype=dtype).reshape(3)
        nrm2 = axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]
        if xp is np:
            if float(nrm2) == 0.0:
                raise ValueError("rotate_axis must be non-zero")
        else:
            if float(cp.asnumpy(nrm2)) == 0.0:
                raise ValueError("rotate_axis must be non-zero")

        nrm = xp.sqrt(nrm2)
        axis = axis / nrm

        ang = float(rotate_angle)
        if degrees:
            ang = xp.deg2rad(ang)
        c = xp.cos(ang)
        s = xp.sin(ang)
        d1 = 1.0 - c
        x, y, z = axis[0], axis[1], axis[2]

        R = xp.empty((3, 3), dtype=dtype)
        R[0, 0] = c + d1*x*x
        R[0, 1] = d1*x*y - z*s
        R[0, 2] = d1*x*z + y*s
        R[1, 0] = d1*y*x + z*s
        R[1, 1] = c + d1*y*y
        R[1, 2] = d1*y*z - x*s
        R[2, 0] = d1*z*x - y*s
        R[2, 1] = d1*z*y + x*s
        R[2, 2] = c + d1*z*z
        return R

    # positions transforms
    def scale_positions(self, positions, scale=1.0, use_gpu=True, dtype=None, copy=True):
        """
        Scale positions isotropically.

        Args:
            positions (ndarray): Array of shape (N, 3).
            scale (float): Isotropic scale factor. 1.0 leaves positions unchanged.
            use_gpu (bool): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on a copy.

        Returns:
            ndarray: Scaled positions, same backend as selected.
        """
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(positions, None, dtype)
        P = xp.asarray(positions, dtype=dtype)
        if copy:
            P = P.copy()
        if scale is not None and scale != 1.0:
            P *= dtype.type(scale)
        return P

    def rotate_positions(self, positions, R, origin=(0.0, 0.0, 0.0), use_gpu=True, dtype=None, copy=True):
        """
        Rotate positions with row-vector convention: (p - origin) @ R + origin.

        Args:
            positions (ndarray): Array of shape (N, 3).
            R (ndarray or None): 3x3 rotation matrix. If None, returns positions.
            origin (sequence): Length-3 rotation origin.
            use_gpu (bool): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on a copy.

        Returns:
            ndarray: Rotated positions, same backend as selected.
        """
        if R is None:
            return positions.copy() if copy else positions
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(positions, None, dtype)
        P = xp.asarray(positions, dtype=dtype)
        if copy:
            P = P.copy()
        org = xp.asarray(origin, dtype=dtype).reshape(1, 3)
        return xp.matmul(P - org, R) + org

    def translate_positions(self, positions, translate, use_gpu=True, dtype=None, copy=True):
        """
        Translate positions by a vector.

        Args:
            positions (ndarray): Array of shape (N, 3).
            translate (sequence or None): Length-3 translation vector. If None,
                returns positions unchanged (or a copy if copy=True).
            use_gpu (bool): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on a copy.

        Returns:
            ndarray: Translated positions, same backend as selected.
        """
        if translate is None:
            return positions.copy() if copy else positions
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(positions, None, dtype)
        P = xp.asarray(positions, dtype=dtype)
        if copy:
            P = P.copy()
        t = xp.asarray(translate, dtype=dtype).reshape(1, 3)
        return P + t

    # F tensor transforms
    def scale_F_about_identity(self, F, alpha=None, use_gpu=True, dtype=None, copy=True):
        """
        Scale deformation gradients about identity: F' = I + alpha * (F - I).

        Args:
            F (ndarray): Array of shape (N, 9) with row-major 3x3 entries.
            alpha (float or None): Amplitude scale. If None, return F unchanged.
            use_gpu (bool): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on a copy.

        Returns:
            ndarray: Scaled F with shape (N, 9).
        """
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(None, F, dtype)
        F9 = xp.asarray(F, dtype=dtype)
        if copy:
            F9 = F9.copy()
        if alpha is None:
            return F9
        N = F9.shape[0]
        Fm = F9.reshape(N, 3, 3)
        I = xp.eye(3, dtype=dtype)
        a = dtype.type(float(alpha))
        Fm = I[None, :, :] + a * (Fm - I[None, :, :])
        return Fm.reshape(N, 9)

    def rotate_F_tensors(self, F, R, use_gpu=True, dtype=None, copy=True):
        """
        Rotate second-order tensors with row-vector convention: F' = R.T @ F @ R.

        Args:
            F (ndarray): Array of shape (N, 9) with row-major 3x3 entries.
            R (ndarray or None): 3x3 rotation matrix. If None, return F unchanged.
            use_gpu (bool): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on a copy.

        Returns:
            ndarray: Rotated F with shape (N, 9).
        """
        if R is None:
            return F.copy() if copy else F
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(None, F, dtype)
        F9 = xp.asarray(F, dtype=dtype)
        if copy:
            F9 = F9.copy()
        N = F9.shape[0]
        Fm = F9.reshape(N, 3, 3)
        RT = R.T
        Fm = xp.matmul(RT, Fm)
        Fm = xp.matmul(Fm, R)
        return Fm.reshape(N, 9)

    # clipping
    def _parse_clip_bounds(self, clip_bounds):
        """
        Normalize clip bounds to (xmin, xmax, ymin, ymax, zmin, zmax).

        Args:
            clip_bounds (sequence or None): Either a flat 6-length list
                [xmin, xmax, ymin, ymax, zmin, zmax] or a 3-tuple of
                pairs ((xmin, xmax), (ymin, ymax), (zmin, zmax)).

        Returns:
            tuple or None: (xmin, xmax, ymin, ymax, zmin, zmax) or None.

        Raises:
            ValueError: If the input shape is invalid.
        """
        if clip_bounds is None:
            return None
        b = clip_bounds
        if (hasattr(b, "__len__") and len(b) == 6):
            return float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]), float(b[5])
        if (hasattr(b, "__len__") and len(b) == 3 and
                all(hasattr(bb, "__len__") and len(bb) == 2 for bb in b)):
            return float(b[0][0]), float(b[0][1]), float(b[1][0]), float(b[1][1]), float(b[2][0]), float(b[2][1])
        raise ValueError("clip_bounds must be ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or [xmin,xmax,ymin,ymax,zmin,zmax]")

    def clip_field(
        self,
        positions,
        F,
        clip_bounds,
        mode="drop",
        return_mask=False,
        use_gpu=True,
        dtype=None,
        copy=True,
    ):
        """
        Clip or clamp a field to a bounding box.

        Args:
            positions (ndarray): Array of shape (N, 3).
            F (ndarray): Array of shape (N, 9).
            clip_bounds (sequence): ((xmin,xmax),(ymin,ymax),(zmin,zmax)) or
                [xmin,xmax,ymin,ymax,zmin,zmax].
            mode (str): "drop" to remove out-of-bounds rows, "clamp" to clamp
                coordinates into bounds, or "none" to leave arrays untouched.
            return_mask (bool): If True, also return a boolean mask of in-bounds rows.
            use_gpu (bool): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on copies.

        Returns:
            tuple:
                positions_out (ndarray): Possibly clipped or clamped positions.
                F_out (ndarray): Filtered F if mode="drop", otherwise original F.
                mask (ndarray, optional): Returned if return_mask=True.

        Raises:
            ValueError: If inputs are missing or mode is invalid.
        """
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(positions, F, dtype)

        if positions is None or F is None:
            raise ValueError("positions and F must be provided")

        P = xp.asarray(positions, dtype=dtype)
        F9 = xp.asarray(F, dtype=dtype)
        if copy:
            P = P.copy()
            F9 = F9.copy()

        bounds = self._parse_clip_bounds(clip_bounds)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        inside = (
            (P[:, 0] >= xmin) & (P[:, 0] <= xmax) &
            (P[:, 1] >= ymin) & (P[:, 1] <= ymax) &
            (P[:, 2] >= zmin) & (P[:, 2] <= zmax)
        )

        m = mode.lower()
        if m == "drop":
            P = P[inside, :]
            F9 = F9[inside, :]
        elif m == "clamp":
            P[:, 0] = xp.clip(P[:, 0], xmin, xmax)
            P[:, 1] = xp.clip(P[:, 1], ymin, ymax)
            P[:, 2] = xp.clip(P[:, 2], zmin, zmax)
        elif m == "none":
            pass
        else:
            raise ValueError('mode must be "drop", "clamp", or "none"')

        if return_mask:
            return P, F9, inside
        return P, F9

    def clip_field_to_sample(
        self,
        field_positions,
        field_F,
        sample,
        margin=0.0,
        use_gpu=True,
        dtype=None,
        copy=True,
        return_mask=False,
    ):
        """
        Clip a deformation field to the sample axis-aligned bounding box.

        The bounding box is derived from sample.corners. A non-negative margin
        grows the box by the same amount in all directions.

        Args:
            field_positions (ndarray): Shape (N, 3). NumPy or CuPy.
            field_F (ndarray): Shape (N, 9). Row-major [F11..F33]. NumPy or CuPy.
            sample (object): Instance exposing an 8x3 `corners` array.
            margin (float, optional): Non-negative expansion. Defaults to 0.0.
            use_gpu (bool, optional): If True and CuPy is available, operate on GPU.
            dtype (numpy dtype or None, optional): Output dtype; inferred if None.
            copy (bool, optional): If True, return new arrays; else may return views.
            return_mask (bool, optional): If True, also return the boolean mask.

        Returns:
            tuple:
                positions_out (ndarray): Clipped positions.
                F_out (ndarray): Clipped F, matching positions_out length.
                mask (ndarray, optional): If requested, shape (N,).

        Raises:
            ValueError: If no field points remain or shapes are invalid.

        Notes:
            This uses an axis-aligned box from min/max over sample.corners.
            Designed to be called before kNN to reduce the field size.
        """
        # Select backend.
        xp = cp if (use_gpu and (cp is not None)) else np

        # Infer dtype.
        if dtype is None:
            if hasattr(field_positions, "dtype"):
                dtype = field_positions.dtype
            elif hasattr(field_F, "dtype"):
                dtype = field_F.dtype
            else:
                dtype = np.float32
        dtype = np.dtype(dtype)

        # Cast arrays and validate shapes.
        P = xp.asarray(field_positions, dtype=dtype)
        F = xp.asarray(field_F, dtype=dtype)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("field_positions must have shape (N, 3)")
        if F.ndim != 2 or F.shape[1] != 9 or F.shape[0] != P.shape[0]:
            raise ValueError("field_F must have shape (N, 9) matching field_positions")

        # Compute AABB from sample corners (corners are a NumPy array).
        corners = sample.corners  # shape (8, 3)
        cmin = corners.min(axis=0).astype(np.float64)
        cmax = corners.max(axis=0).astype(np.float64)
        if margin is not None and margin > 0.0:
            cmin = cmin - float(margin)
            cmax = cmax + float(margin)

        cmin_xp = xp.asarray(cmin, dtype=dtype)
        cmax_xp = xp.asarray(cmax, dtype=dtype)

        # In-bounds mask for AABB.
        mask = (
            (P[:, 0] >= cmin_xp[0]) & (P[:, 0] <= cmax_xp[0]) &
            (P[:, 1] >= cmin_xp[1]) & (P[:, 1] <= cmax_xp[1]) &
            (P[:, 2] >= cmin_xp[2]) & (P[:, 2] <= cmax_xp[2])
        )

        # Clip arrays.
        P_out = P[mask, :]
        F_out = F[mask, :]
        if P_out.shape[0] == 0:
            raise ValueError("clip_field_to_sample removed all field points; increase margin or check inputs")

        if copy:
            P_out = P_out.copy()
            F_out = F_out.copy()

        if return_mask:
            return P_out, F_out, mask
        return P_out, F_out

    def _ensure_field_cuda_kernels(self, dtype=np.float32, k=8):
        """
        Compile and cache CUDA RawKernels for field kNN and application.

        Args:
            dtype (numpy dtype): Floating dtype (np.float32 or np.float64).
            k (int): Number of neighbors (1..32 recommended).

        Notes:
            If CuPy is not available, this is a no-op.
        """
        if (cp is None):
            return
        dt = np.dtype(dtype)
        key = ("field_kernels", dt.name, int(k))
        if hasattr(self, "_field_kernels") and key in self._field_kernels:
            return
        if not hasattr(self, "_field_kernels"):
            self._field_kernels = {}

        T = "double" if dt == np.float64 else "float"
        POW = "pow" if dt == np.float64 else "powf"
        SQRT = "sqrt" if dt == np.float64 else "sqrtf"
        INF = "1e300" if dt == np.float64 else "1e30f"

        # CUDA kernel sources for kNN, weighting, and application.
        src_knn = r'''
        extern "C" __global__
        void knn_topk_sqdist(const %(T)s* __restrict__ P, const int N,
                             const %(T)s* __restrict__ X, const int M,
                             int* __restrict__ out_idx,
                             %(T)s* __restrict__ out_d2)
        {
            int tid = threadIdx.x;
            int q = blockIdx.x * blockDim.x + tid;

            %(T)s qx = 0, qy = 0, qz = 0;
            if (q < M) {
                qx = X[3*q+0];
                qy = X[3*q+1];
                qz = X[3*q+2];
            }

            extern __shared__ unsigned char smem_raw[];
            %(T)s* sPx = (%(T)s*)smem_raw;
            %(T)s* sPy = sPx + blockDim.x;
            %(T)s* sPz = sPy + blockDim.x;

            const %(T)s INFV = (%(INF)s);
            %(T)s best_d[%(K)d];
            int   best_i[%(K)d];
            #pragma unroll
            for (int a=0; a<%(K)d; ++a) { best_d[a] = INFV; best_i[a] = -1; }

            for (int j0 = 0; j0 < N; j0 += blockDim.x) {
                int pj = j0 + tid;
                if (pj < N) {
                    sPx[tid] = P[3*pj+0];
                    sPy[tid] = P[3*pj+1];
                    sPz[tid] = P[3*pj+2];
                }
                __syncthreads();

                int tileCount = min(blockDim.x, N - j0);
                if (q < M) {
                    for (int t=0; t<tileCount; ++t) {
                        %(T)s dx = qx - sPx[t];
                        %(T)s dy = qy - sPy[t];
                        %(T)s dz = qz - sPz[t];
                        %(T)s d2 = dx*dx + dy*dy + dz*dz;
                        int id = j0 + t;

                        int ins = -1;
                        for (int a=%(K)d-1; a>=0; --a) {
                            if (d2 < best_d[a]) ins = a;
                        }
                        if (ins >= 0) {
                            for (int a=%(K)d-1; a>ins; --a) {
                                best_d[a] = best_d[a-1];
                                best_i[a] = best_i[a-1];
                            }
                            best_d[ins] = d2;
                            best_i[ins] = id;
                        }
                    }
                }
                __syncthreads();
            }

            if (q < M) {
                for (int a=0; a<%(K)d; ++a) {
                    out_idx[q*%(K)d + a] = best_i[a];
                    out_d2[q*%(K)d + a] = best_d[a];
                }
            }
        }
        ''' % {"T": T, "K": int(k), "INF": INF}

        src_weight = r'''
        extern "C" __global__
        void weighted_sum_F9_knn(const int* __restrict__ idx,
                                 const %(T)s* __restrict__ d2,
                                 const %(T)s* __restrict__ F9_field,
                                 const int M,
                                 const %(T)s power,
                                 const %(T)s eps,
                                 %(T)s* __restrict__ F9_out)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= M) return;

            bool has_zero = false;
            int zero_j = -1;
            %(T)s eps2 = eps * eps;
            for (int j=0; j<%(K)d; ++j) {
                if (d2[i*%(K)d + j] <= eps2) { has_zero = true; zero_j = j; break; }
            }

            %(T)s s0=0, s1=0, s2=0, s3=0, s4=0, s5=0, s6=0, s7=0, s8=0;
            %(T)s wsum = 0;

            if (has_zero) {
                int id = idx[i*%(K)d + zero_j];
                const %(T)s* f = &F9_field[id*9];
                s0=f[0]; s1=f[1]; s2=f[2];
                s3=f[3]; s4=f[4]; s5=f[5];
                s6=f[6]; s7=f[7]; s8=f[8];
                wsum = 1;
            } else {
                for (int j=0; j<%(K)d; ++j) {
                    int id = idx[i*%(K)d + j];
                    %(T)s dij = %(SQRT)s(d2[i*%(K)d + j]);
                    %(T)s w = (%(T)s)1 / %(POW)s(dij + eps, power);
                    wsum += w;
                    const %(T)s* f = &F9_field[id*9];
                    s0 += w * f[0]; s1 += w * f[1]; s2 += w * f[2];
                    s3 += w * f[3]; s4 += w * f[4]; s5 += w * f[5];
                    s6 += w * f[6]; s7 += w * f[7]; s8 += w * f[8];
                }
            }

            %(T)s inv = (wsum > (%(T)s)0) ? ((%(T)s)1/wsum) : (%(T)s)0;
            %(T)s* out = &F9_out[i*9];
            out[0]=s0*inv; out[1]=s1*inv; out[2]=s2*inv;
            out[3]=s3*inv; out[4]=s4*inv; out[5]=s5*inv;
            out[6]=s6*inv; out[7]=s7*inv; out[8]=s8*inv;
        }
        ''' % {"T": T, "K": int(k), "POW": POW, "SQRT": SQRT}

        src_apply = r'''
        extern "C" __global__
        void apply_affine_rowwise_T(const %(T)s* __restrict__ F9,
                                    const %(T)s* __restrict__ pos,
                                    const %(T)s* __restrict__ origin,
                                    %(T)s* __restrict__ out,
                                    const int n)
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            const %(T)s* f = &F9[i*9];
            %(T)s x = pos[i*3+0] - origin[0];
            %(T)s y = pos[i*3+1] - origin[1];
            %(T)s z = pos[i*3+2] - origin[2];
            out[i*3+0] = origin[0] + f[0]*x + f[1]*y + f[2]*z;
            out[i*3+1] = origin[1] + f[3]*x + f[4]*y + f[5]*z;
            out[i*3+2] = origin[2] + f[6]*x + f[7]*y + f[8]*z;
        }
        ''' % {"T": T}

        kernels = {
            "knn_topk_sqdist": cp.RawKernel(src_knn, "knn_topk_sqdist"),
            "weighted_sum_F9_knn": cp.RawKernel(src_weight, "weighted_sum_F9_knn"),
            "apply_affine_rowwise_T": cp.RawKernel(src_apply, "apply_affine_rowwise_T"),
            "dtype": dt,
            "k": int(k),
        }
        self._field_kernels[key] = kernels

    def _get_field_cuda_kernels(self, dtype, k):
        """
        Retrieve compiled CUDA kernels for field operations.

        Args:
            dtype (numpy dtype): Floating dtype.
            k (int): Neighbor count for kNN.

        Returns:
            dict or None: Kernel bundle if CuPy available; else None.
        """
        if (cp is None):
            return None
        self._ensure_field_cuda_kernels(dtype=dtype, k=k)
        return self._field_kernels[("field_kernels", np.dtype(dtype).name, int(k))]

    def _get_cell_cull_kernel(self):
        """
        Compile (once) and return the cell-list candidate gathering kernel.

        The kernel appends the node indices of every cell in an integer cell
        range to an output buffer. It has no dtype or K dependence, so it is
        shared by every kernel bundle.

        Returns:
            cupy.RawKernel or None: None if CuPy is unavailable.
        """
        if cp is None:
            return None
        kern = getattr(self, "_cell_cull_kernel", None)
        if kern is not None:
            return kern
        src = r'''
        extern "C" __global__
        void gather_cell_candidates(
            const int* __restrict__ sortedIdx,
            const int* __restrict__ cell_start,
            const int* __restrict__ cell_end,
            const int cx0, const int cy0, const int cz0,
            const int cx1, const int cy1, const int cz1,
            const int nx, const int ny, const int nz,
            const int n_cells_total,
            int* __restrict__ out_indices,
            int* __restrict__ out_count)
        {
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            int ncx = cx1 - cx0 + 1;
            int ncy = cy1 - cy0 + 1;
            int ncz = cz1 - cz0 + 1;
            int total_cells = ncx * ncy * ncz;

            if (tid >= total_cells) return;

            // Compute 3D cell indices
            int local_cx = tid % ncx;
            int local_cy = (tid / ncx) % ncy;
            int local_cz = tid / (ncx * ncy);

            int cx = cx0 + local_cx;
            int cy = cy0 + local_cy;
            int cz = cz0 + local_cz;

            int cid = cz * (nx * ny) + cy * nx + cx;

            if (cid < 0 || cid >= n_cells_total) return;

            int s = cell_start[cid];
            int e = cell_end[cid];

            if (s < 0 || e < s) return;

            // Atomic append to output
            int write_pos = atomicAdd(out_count, e - s);
            for (int i = s; i < e; i++) {
                out_indices[write_pos + (i - s)] = sortedIdx[i];
            }
        }
        '''
        self._cell_cull_kernel = cp.RawKernel(src, "gather_cell_candidates")
        return self._cell_cull_kernel

    def _ensure_fe_nodal_cuda_kernels(self, dtype=np.float32, k=48):
        """
        Compile and cache optimized CUDA kernels for FE nodal field operations.

        Creates kernels for:
        - kNN over node positions (sorted top-K list per query)
        - GPU-native cell culling
        - Fused MLS basis + weighted normal equations
        - Batched 10x10 Cholesky solver with a per-row status flag

        Args:
            dtype (numpy dtype): Floating dtype (np.float32 or np.float64).
            k (int): Number of neighbors (up to 64 recommended).
        """
        if cp is None:
            return
        dt = np.dtype(dtype)
        key = ("fe_nodal_kernels", dt.name, int(k))
        if hasattr(self, "_fe_nodal_kernels") and key in self._fe_nodal_kernels:
            return
        if not hasattr(self, "_fe_nodal_kernels"):
            self._fe_nodal_kernels = {}

        T = "double" if dt == np.float64 else "float"
        POW = "pow" if dt == np.float64 else "powf"
        SQRT = "sqrt" if dt == np.float64 else "sqrtf"
        INF = "1e300" if dt == np.float64 else "1e30f"
        K = int(k)

        # kNN kernel: per-thread sorted top-K list filled by insertion.
        src_knn_insert = r'''
        extern "C" __global__
        void knn_insert_sqdist(const %(T)s* __restrict__ P, const int N,
                                const %(T)s* __restrict__ X, const int M,
                                int* __restrict__ out_idx,
                                %(T)s* __restrict__ out_d2)
        {
            int tid = threadIdx.x;
            int q = blockIdx.x * blockDim.x + tid;

            %(T)s qx = 0, qy = 0, qz = 0;
            if (q < M) {
                qx = X[3*q+0];
                qy = X[3*q+1];
                qz = X[3*q+2];
            }

            extern __shared__ unsigned char smem_raw[];
            %(T)s* sPx = (%(T)s*)smem_raw;
            %(T)s* sPy = sPx + blockDim.x;
            %(T)s* sPz = sPy + blockDim.x;

            const %(T)s INFV = (%(INF)s);
            %(T)s best_d[%(K)d];
            int   best_i[%(K)d];
            #pragma unroll
            for (int a=0; a<%(K)d; ++a) { best_d[a] = INFV; best_i[a] = -1; }

            for (int j0 = 0; j0 < N; j0 += blockDim.x) {
                int pj = j0 + tid;
                if (pj < N) {
                    sPx[tid] = P[3*pj+0];
                    sPy[tid] = P[3*pj+1];
                    sPz[tid] = P[3*pj+2];
                }
                __syncthreads();

                int tileCount = min(blockDim.x, N - j0);
                if (q < M) {
                    for (int t=0; t<tileCount; ++t) {
                        %(T)s dx = qx - sPx[t];
                        %(T)s dy = qy - sPy[t];
                        %(T)s dz = qz - sPz[t];
                        %(T)s d2 = dx*dx + dy*dy + dz*dz;
                        int id = j0 + t;

                        // Insertion into the sorted top-K list (ascending d2).
                        if (d2 < best_d[%(K)d-1]) {
                            int ins = -1;
                            for (int a=%(K)d-1; a>=0; --a) {
                                if (d2 < best_d[a]) ins = a;
                            }
                            for (int a=%(K)d-1; a>ins; --a) {
                                best_d[a] = best_d[a-1];
                                best_i[a] = best_i[a-1];
                            }
                            best_d[ins] = d2;
                            best_i[ins] = id;
                        }
                    }
                }
                __syncthreads();
            }

            if (q < M) {
                #pragma unroll
                for (int a=0; a<%(K)d; ++a) {
                    out_idx[q*%(K)d + a] = best_i[a];
                    out_d2[q*%(K)d + a] = best_d[a];
                }
            }
        }
        ''' % {"T": T, "K": K, "INF": INF}

        # Fused MLS basis + weighted normal equations kernel
        src_mls_fused = r'''
        extern "C" __global__
        void mls_fused_weighted_neq(
            const %(T)s* __restrict__ Xq,      // (M, 3) query points
            const %(T)s* __restrict__ P_nodes, // (N, 3) node positions
            const %(T)s* __restrict__ U_nodes, // (N, 3) node displacements
            const int* __restrict__ idx,       // (M, k) neighbor indices
            const %(T)s* __restrict__ d2,      // (M, k) squared distances
            const %(T)s power,
            const %(T)s eps,
            const %(T)s reg,
            const int M,
            const int N,
            %(T)s* __restrict__ A_out,         // (M, 10, 10) normal equation matrices
            %(T)s* __restrict__ b_out)         // (M, 10, 3) RHS vectors
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= M) return;

            %(T)s qx = Xq[3*i+0];
            %(T)s qy = Xq[3*i+1];
            %(T)s qz = Xq[3*i+2];

            // Compute median distance for normalization
            %(T)s d_sorted[%(K)d];
            #pragma unroll
            for (int j = 0; j < %(K)d; j++) {
                d_sorted[j] = %(SQRT)s(d2[i*%(K)d + j] + eps);
            }

            // Simple bubble sort for median (small k)
            #pragma unroll
            for (int pass = 0; pass < %(K)d/2; pass++) {
                #pragma unroll
                for (int j = 0; j < %(K)d-1; j++) {
                    if (d_sorted[j] > d_sorted[j+1]) {
                        %(T)s tmp = d_sorted[j];
                        d_sorted[j] = d_sorted[j+1];
                        d_sorted[j+1] = tmp;
                    }
                }
            }
            %(T)s h = d_sorted[%(K)d/2];
            h = (h > (%(T)s)1e-9) ? h : (%(T)s)1e-9;
            %(T)s invh = (%(T)s)1 / h;

            // Initialize accumulators for A (10x10 symmetric) and b (10x3)
            %(T)s A[55]; // Upper triangle of 10x10 symmetric matrix
            %(T)s b[30]; // 10x3 matrix
            %(T)s wsum = 0;

            #pragma unroll
            for (int a = 0; a < 55; a++) A[a] = 0;
            #pragma unroll
            for (int a = 0; a < 30; a++) b[a] = 0;

            // Check for exact hits
            %(T)s eps2 = eps * eps;
            bool has_zero = false;
            int zero_idx = -1;

            #pragma unroll
            for (int j = 0; j < %(K)d; j++) {
                if (d2[i*%(K)d + j] <= eps2) {
                    has_zero = true;
                    zero_idx = j;
                    break;
                }
            }

            if (!has_zero) {
                // Accumulate weighted normal equations
                #pragma unroll
                for (int j = 0; j < %(K)d; j++) {
                    int nid = idx[i*%(K)d + j];
                    if (nid < 0 || nid >= N) continue;

                    %(T)s px = P_nodes[3*nid+0];
                    %(T)s py = P_nodes[3*nid+1];
                    %(T)s pz = P_nodes[3*nid+2];

                    %(T)s ux = U_nodes[3*nid+0];
                    %(T)s uy = U_nodes[3*nid+1];
                    %(T)s uz = U_nodes[3*nid+2];

                    // Normalized local coordinates
                    %(T)s sx = (px - qx) * invh;
                    %(T)s sy = (py - qy) * invh;
                    %(T)s sz = (pz - qz) * invh;

                    // 10-term quadratic basis
                    %(T)s basis[10];
                    basis[0] = 1;
                    basis[1] = sx;
                    basis[2] = sy;
                    basis[3] = sz;
                    basis[4] = sx*sx;
                    basis[5] = sy*sy;
                    basis[6] = sz*sz;
                    basis[7] = sx*sy;
                    basis[8] = sx*sz;
                    basis[9] = sy*sz;

                    // Weight
                    %(T)s dij = %(SQRT)s(d2[i*%(K)d + j]);
                    %(T)s w = (%(T)s)1 / %(POW)s(dij + eps, power);
                    wsum += w;

                    // A += w * basis * basis^T (upper triangle only)
                    int idx_a = 0;
                    #pragma unroll
                    for (int row = 0; row < 10; row++) {
                        #pragma unroll
                        for (int col = row; col < 10; col++) {
                            A[idx_a++] += w * basis[row] * basis[col];
                        }
                    }

                    // b += w * basis * U^T
                    #pragma unroll
                    for (int row = 0; row < 10; row++) {
                        b[row*3 + 0] += w * basis[row] * ux;
                        b[row*3 + 1] += w * basis[row] * uy;
                        b[row*3 + 2] += w * basis[row] * uz;
                    }
                }

                // Add regularization to diagonal
                %(T)s reg_val = reg * wsum;
                int diag_indices[10] = {0, 10, 19, 27, 34, 40, 45, 49, 52, 54};
                #pragma unroll
                for (int d = 0; d < 10; d++) {
                    A[diag_indices[d]] += reg_val;
                }
            } else {
                // Exact hit: set identity for A, U for b
                int nid = idx[i*%(K)d + zero_idx];
                %(T)s ux = U_nodes[3*nid+0];
                %(T)s uy = U_nodes[3*nid+1];
                %(T)s uz = U_nodes[3*nid+2];

                // A = identity (upper triangle)
                int idx_a = 0;
                #pragma unroll
                for (int row = 0; row < 10; row++) {
                    #pragma unroll
                    for (int col = row; col < 10; col++) {
                        A[idx_a++] = (row == col) ? (%(T)s)1 : (%(T)s)0;
                    }
                }

                // b = [U, 0, 0, ...]
                b[0] = ux; b[1] = uy; b[2] = uz;
                #pragma unroll
                for (int a = 3; a < 30; a++) b[a] = 0;
            }

            // Write out A (expand upper triangle to full 10x10)
            int idx_a = 0;
            #pragma unroll
            for (int row = 0; row < 10; row++) {
                #pragma unroll
                for (int col = row; col < 10; col++) {
                    A_out[i*100 + row*10 + col] = A[idx_a];
                    if (row != col) {
                        A_out[i*100 + col*10 + row] = A[idx_a];
                    }
                    idx_a++;
                }
            }

            // Write out b
            #pragma unroll
            for (int a = 0; a < 30; a++) {
                b_out[i*30 + a] = b[a];
            }
        }
        ''' % {"T": T, "K": K, "POW": POW, "SQRT": SQRT}

        # Custom batched 10x10 Cholesky solver kernel
        src_cholesky_10x10 = r'''
        extern "C" __global__
        void batched_cholesky_solve_10x10(
            const %(T)s* __restrict__ A,  // (M, 10, 10)
            const %(T)s* __restrict__ b,  // (M, 10, 3)
            const int M,
            %(T)s* __restrict__ x_out,    // (M, 10, 3)
            int* __restrict__ status_out) // (M,) 1 if the factorization succeeded
        {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= M) return;

            // Local copies
            %(T)s L[100]; // 10x10 lower triangular
            %(T)s rhs[30]; // 10x3
            %(T)s y[30]; // 10x3 intermediate
            %(T)s x[30]; // 10x3 solution

            // Load A and b
            #pragma unroll
            for (int a = 0; a < 100; a++) L[a] = A[i*100 + a];
            #pragma unroll
            for (int a = 0; a < 30; a++) rhs[a] = b[i*30 + a];

            // Cholesky decomposition: A = L * L^T
            bool success = true;
            #pragma unroll
            for (int j = 0; j < 10; j++) {
                %(T)s sum = L[j*10 + j];
                #pragma unroll
                for (int k = 0; k < j; k++) {
                    sum -= L[j*10 + k] * L[j*10 + k];
                }
                if (sum <= (%(T)s)0) {
                    success = false;
                    break;
                }
                L[j*10 + j] = %(SQRT)s(sum);

                #pragma unroll
                for (int i_row = j + 1; i_row < 10; i_row++) {
                    sum = L[i_row*10 + j];
                    #pragma unroll
                    for (int k = 0; k < j; k++) {
                        sum -= L[i_row*10 + k] * L[j*10 + k];
                    }
                    L[i_row*10 + j] = sum / L[j*10 + j];
                }
            }

            if (success) {
                // Forward substitution: L * y = b (for each of 3 columns)
                #pragma unroll
                for (int col = 0; col < 3; col++) {
                    #pragma unroll
                    for (int i_row = 0; i_row < 10; i_row++) {
                        %(T)s sum = rhs[i_row*3 + col];
                        #pragma unroll
                        for (int k = 0; k < i_row; k++) {
                            sum -= L[i_row*10 + k] * y[k*3 + col];
                        }
                        y[i_row*3 + col] = sum / L[i_row*10 + i_row];
                    }
                }

                // Backward substitution: L^T * x = y (for each of 3 columns)
                #pragma unroll
                for (int col = 0; col < 3; col++) {
                    #pragma unroll
                    for (int i_row = 9; i_row >= 0; i_row--) {
                        %(T)s sum = y[i_row*3 + col];
                        #pragma unroll
                        for (int k = i_row + 1; k < 10; k++) {
                            sum -= L[k*10 + i_row] * x[k*3 + col];
                        }
                        x[i_row*3 + col] = sum / L[i_row*10 + i_row];
                    }
                }
            } else {
                // Not positive definite: zero the row and flag it for the caller.
                #pragma unroll
                for (int a = 0; a < 30; a++) x[a] = 0;
            }

            // Write output
            #pragma unroll
            for (int a = 0; a < 30; a++) {
                x_out[i*30 + a] = x[a];
            }
            status_out[i] = success ? 1 : 0;
        }
        ''' % {"T": T, "SQRT": SQRT}

        # Compile all kernels
        kernels = {
            "knn": cp.RawKernel(src_knn_insert, "knn_insert_sqdist"),
            "cell_cull": self._get_cell_cull_kernel(),
            "mls_fused": cp.RawKernel(src_mls_fused, "mls_fused_weighted_neq"),
            "cholesky_solve": cp.RawKernel(src_cholesky_10x10, "batched_cholesky_solve_10x10"),
            "dtype": dt,
            "k": K,
        }
        self._fe_nodal_kernels[key] = kernels

    def _get_fe_nodal_cuda_kernels(self, dtype, k):
        """
        Retrieve compiled optimized CUDA kernels for FE nodal operations.

        Args:
            dtype (numpy dtype): Floating dtype.
            k (int): Neighbor count for kNN.

        Returns:
            dict or None: Kernel bundle if CuPy available; else None.
        """
        if cp is None:
            return None
        self._ensure_fe_nodal_cuda_kernels(dtype=dtype, k=k)
        return self._fe_nodal_kernels[("fe_nodal_kernels", np.dtype(dtype).name, int(k))]

    def apply_deformation_chunked(
        self,
        field_positions,
        field_F,
        sample,
        chunk_size=200000,
        k=8,
        origin=(0.0, 0.0, 0.0),
        use_gpu=True,
        power=2.0,
        threads=None,
        tile_size=None,
        yield_chunks=False,
        dtype=None,
        clip_to_sample=True,
        clip_margin=10.0,
        use_cell_list=True,
        cell_r_cut=None,
        cell_pad_cells=1,
        force=False,
    ):
        """
        Apply a deformation gradient field to sample points in chunks.

        Each point receives the inverse-distance-weighted average of the k
        nearest field tensors and is mapped by x' = origin + F (x - origin).
        The GPU path uses custom CUDA kernels for kNN, weighting and
        application, with optional cell-list culling of the field; the CPU
        path uses a k-d tree for the kNN.

        Args:
            field_positions (ndarray): Field node positions, shape (Nf, 3).
            field_F (ndarray): Field F tensors, shape (Nf, 9), row-major.
            sample (Sample, ndarray or iterable): A Sample object with chunked
                position IO, a single (M, 3) array, or an iterable of (Mi, 3)
                chunks. For a Sample the stored chunks are read raw, deformed,
                written back, and the sample metadata is updated; nothing is
                returned.
            chunk_size (int): Chunk size used when `sample` is a single array.
            k (int): Number of neighbors for the inverse-distance-weighted
                interpolation of the deformation gradient. Defaults to 8.
            origin (sequence): 3-vector origin for affine application.
            use_gpu (bool): Use CUDA path if True and CuPy is available.
            power (float): IDW power for weighting distances.
            threads (int or None): Reserved; not used.
            tile_size (int or None): Kept for API compatibility; not used.
            yield_chunks (bool): If True, return a generator of output chunks
                (array or iterable input only).
            dtype (numpy dtype or None): Output dtype; inferred if None.
            clip_to_sample (bool): If True and `sample` exposes `corners`, clip
                the field to the sample AABB first.
            clip_margin (float): Margin for clipping AABB.
            use_cell_list (bool): If True (GPU) and `sample` provides
                `build_cell_list_gpu`, build a cell list over the field.
            cell_r_cut (float or None): Optional cell list cutoff radius.
            cell_pad_cells (int): Halo cells around each chunk AABB for culling.
            force (bool): For Sample input, apply even if the same field was
                already applied to the sample.

        Returns:
            None, ndarray, list or generator:
                - Sample input: None (chunks rewritten on disk).
                - ndarray input, yield_chunks=False: one array of deformed
                  positions (CuPy on GPU, NumPy on CPU).
                - ndarray input, yield_chunks=True: generator of chunk outputs.
                - iterable input: list of outputs (generator if yield_chunks).

        Raises:
            ValueError: On invalid shapes or parameters.
            RuntimeError: For a Sample in streaming mode, or one to which this
                field was already applied when `force` is False.
        """
        import hashlib

        is_sample = (hasattr(sample, "load_chunk_positions")
                     and hasattr(sample, "write_chunk_positions")
                     and hasattr(sample, "chunk_total"))
        if is_sample and getattr(sample, "_streaming_mode", False):
            raise RuntimeError("apply_deformation_chunked: the sample is in streaming mode; chunks are "
                               "regenerated on demand and stored positions are never read, so a "
                               "deformation cannot be applied.")

        # Backend and dtype resolution.
        if (use_gpu and cp is None):
            use_gpu = False
        xp = cp if use_gpu else np
        if dtype is None:
            dtype = field_F.dtype if hasattr(field_F, "dtype") else np.float32
        dtype = np.dtype(dtype)
        f32 = (dtype == np.float32)

        # Cast field arrays and validate shapes.
        P_all = xp.asarray(field_positions, dtype=dtype)
        F_all = xp.asarray(field_F, dtype=dtype)
        if P_all.ndim != 2 or P_all.shape[1] != 3:
            raise ValueError("field_positions must have shape (N,3)")
        if F_all.ndim != 2 or F_all.shape[1] != 9 or F_all.shape[0] != P_all.shape[0]:
            raise ValueError("field_F must have shape (N,9) matching field_positions")
        if k <= 0 or k > max(1, P_all.shape[0]):
            raise ValueError("k must be in [1, N_field]")
        n_field_total = int(P_all.shape[0])

        # Modification record for Sample input: identifies the field by content.
        params = None
        if is_sample:
            h = hashlib.sha1()
            P_host = cp.asnumpy(P_all) if use_gpu else np.asarray(P_all)
            F_host = cp.asnumpy(F_all) if use_gpu else np.asarray(F_all)
            h.update(np.ascontiguousarray(P_host, dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(F_host, dtype=np.float32).tobytes())
            del P_host, F_host
            params = {
                "field_digest": h.hexdigest()[:16],
                "n_field": n_field_total,
                "k": int(k),
                "power": float(power),
                "origin": [float(v) for v in np.asarray(origin, dtype=np.float64).reshape(3)],
            }
            if (not force) and hasattr(sample, "has_modification") and sample.has_modification("deformation_field", params):
                raise RuntimeError("apply_deformation_chunked: this field was already applied to the sample "
                                   "(see sample_metadata.json); pass force=True to apply it again.")

        # Optional one-time field clip to sample AABB.
        if clip_to_sample and hasattr(sample, "corners"):
            P_all, F_all = self.clip_field_to_sample(
                P_all, F_all, sample, margin=float(clip_margin), use_gpu=use_gpu, dtype=dtype, copy=False
            )  # raises if empty
            if k > int(P_all.shape[0]):
                raise ValueError("k exceeds the number of field points inside the sample AABB (+margin)")

        origin = xp.asarray(origin, dtype=dtype).reshape(3)
        eps = dtype.type(1e-12)

        # Build the per-chunk processing function for the chosen backend.
        if use_gpu:
            kern = self._get_field_cuda_kernels(dtype=dtype, k=k)
            sizeof_T = 8 if dtype == np.float64 else 4
            block = 128

            def _knn_gpu(X, P_sub):
                M = int(X.shape[0]); N = int(P_sub.shape[0])
                idx = cp.empty((M, k), dtype=cp.int32)
                d2 = cp.empty((M, k), dtype=dtype)
                grid = (M + block - 1) // block
                smem = 3 * block * sizeof_T  # x,y,z per candidate in shared memory
                kern["knn_topk_sqdist"]((grid,), (block,),
                    (P_sub.ravel(), np.int32(N),
                     X.ravel(), np.int32(M),
                     idx.ravel(), d2.ravel()),
                    shared_mem=smem)
                return idx, d2

            def _weight_F_gpu(idx, d2):
                M = int(idx.shape[0])
                F9 = cp.empty((M, 9), dtype=dtype)
                grid = (M + block - 1) // block
                kern["weighted_sum_F9_knn"]((grid,), (block,),
                    (idx.ravel(), d2.ravel(), F_all.ravel(), np.int32(M),
                     dtype.type(float(power)), dtype.type(float(eps)),
                     F9.ravel()))
                return F9

            def _apply_gpu(F9, X):
                M = int(X.shape[0])
                out = cp.empty_like(X)
                grid = (M + block - 1) // block
                kern["apply_affine_rowwise_T"]((grid,), (block,),
                    (F9.ravel(), X.ravel(), origin, out.ravel(), np.int32(M)))
                return out

            # Optional GPU cell list over the field for candidate culling.
            field_cells = None
            if use_cell_list and hasattr(sample, "build_cell_list_gpu"):
                if cell_r_cut is None:
                    cell_r_cut = 3.0 * self._median_nn_spacing(cp.asnumpy(P_all))
                cell_r_cut = float(cell_r_cut)
                (sortedP, sortedIdx, cell_start, cell_end,
                 bb_min, cell_size, nx, ny, nz) = sample.build_cell_list_gpu(
                    P_all.astype(cp.float32, copy=False), cell_r_cut)
                field_cells = {
                    "sortedIdx": sortedIdx,
                    "cell_start": cell_start,
                    "cell_end": cell_end,
                    "bb_min": bb_min.astype(dtype, copy=False),
                    "cell_size": float(cell_size),
                    "nx": int(nx), "ny": int(ny), "nz": int(nz),
                    "cand": cp.empty((max(1, int(P_all.shape[0])),), dtype=cp.int32),
                    "count": cp.zeros((1,), dtype=cp.int32),
                    "kernel": self._get_cell_cull_kernel(),
                }

            def _candidate_indices_from_chunk_AABB(X_chunk):
                # Gather field indices from the cells covering the chunk AABB plus halo.
                cs = field_cells["cell_size"]
                halo = float(cs * max(0, int(cell_pad_cells)))
                nxv, nyv, nzv = field_cells["nx"], field_cells["ny"], field_cells["nz"]
                lo = cp.floor((cp.min(X_chunk, axis=0) - halo - field_cells["bb_min"]) / cs)
                hi = cp.floor((cp.max(X_chunk, axis=0) + halo - field_cells["bb_min"]) / cs)
                cb = cp.concatenate([lo, hi]).get()
                cx0 = max(0, min(nxv - 1, int(cb[0])))
                cy0 = max(0, min(nyv - 1, int(cb[1])))
                cz0 = max(0, min(nzv - 1, int(cb[2])))
                cx1 = max(0, min(nxv - 1, int(cb[3])))
                cy1 = max(0, min(nyv - 1, int(cb[4])))
                cz1 = max(0, min(nzv - 1, int(cb[5])))
                total_cells = (cx1 - cx0 + 1) * (cy1 - cy0 + 1) * (cz1 - cz0 + 1)
                if total_cells <= 0:
                    return cp.zeros((0,), dtype=cp.int32)
                field_cells["count"][0] = 0
                grid_cells = (total_cells + block - 1) // block
                field_cells["kernel"](
                    (grid_cells,), (block,),
                    (field_cells["sortedIdx"], field_cells["cell_start"], field_cells["cell_end"],
                     np.int32(cx0), np.int32(cy0), np.int32(cz0),
                     np.int32(cx1), np.int32(cy1), np.int32(cz1),
                     np.int32(nxv), np.int32(nyv), np.int32(nzv),
                     np.int32(int(field_cells["cell_start"].shape[0])),
                     field_cells["cand"], field_cells["count"])
                )
                count = int(field_cells["count"].get())
                if count == 0:
                    return cp.zeros((0,), dtype=cp.int32)
                return field_cells["cand"][:count].copy()

            def _process_chunk(Xchunk):
                X = cp.asarray(Xchunk, dtype=dtype)
                if X.ndim != 2 or X.shape[1] != 3:
                    raise ValueError("Each sample chunk must have shape (?,3)")
                if int(X.shape[0]) == 0:
                    return cp.empty((0, 3), dtype=dtype)
                if field_cells is not None:
                    cand_idx = _candidate_indices_from_chunk_AABB(X)
                    if int(cand_idx.size) >= k:
                        idx_sub, d2 = _knn_gpu(X, P_all[cand_idx])
                        idx = cp.take(cand_idx, idx_sub)
                    else:
                        idx, d2 = _knn_gpu(X, P_all)
                else:
                    idx, d2 = _knn_gpu(X, P_all)
                F9 = _weight_F_gpu(idx, d2)
                return _apply_gpu(F9, X)

        else:
            from scipy.spatial import cKDTree

            # Optional CFFI micro-kernels for float32 speedups.
            def _ensure_cffi():
                if not hasattr(self, "_cffi_lib"):
                    try:
                        ffi = FFI()
                        csrc = r"""
                        #include <stddef.h>
                        void weighted_sum_F9(const int* idx,
                                             const float* w,
                                             const float* F9_field,
                                             int n_rows, int k,
                                             float* F9_out)
                        {
                            for (int i = 0; i < n_rows; ++i) {
                                float sums[9] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
                                float wsum = 0.0f;
                                for (int j = 0; j < k; ++j) {
                                    int id = idx[i*k + j];
                                    float weight = w[i*k + j];
                                    wsum += weight;
                                    const float* f = &F9_field[(size_t)id * 9];
                                    sums[0] += weight * f[0];
                                    sums[1] += weight * f[1];
                                    sums[2] += weight * f[2];
                                    sums[3] += weight * f[3];
                                    sums[4] += weight * f[4];
                                    sums[5] += weight * f[5];
                                    sums[6] += weight * f[6];
                                    sums[7] += weight * f[7];
                                    sums[8] += weight * f[8];
                                }
                                float inv = (wsum > 0.0f) ? (1.0f / wsum) : 0.0f;
                                float* out = &F9_out[(size_t)i * 9];
                                out[0] = sums[0] * inv;
                                out[1] = sums[1] * inv;
                                out[2] = sums[2] * inv;
                                out[3] = sums[3] * inv;
                                out[4] = sums[4] * inv;
                                out[5] = sums[5] * inv;
                                out[6] = sums[6] * inv;
                                out[7] = sums[7] * inv;
                                out[8] = sums[8] * inv;
                            }
                        }
                        void apply_affine_rowwise(const float* F9,
                                                  const float* pos,
                                                  const float* origin,
                                                  float* out,
                                                  size_t n)
                        {
                            for (size_t i = 0; i < n; ++i) {
                                const float* f = &F9[i*9];
                                float x = pos[i*3+0] - origin[0];
                                float y = pos[i*3+1] - origin[1];
                                float z = pos[i*3+2] - origin[2];
                                out[i*3+0] = origin[0] + f[0]*x + f[1]*y + f[2]*z;
                                out[i*3+1] = origin[1] + f[3]*x + f[4]*y + f[5]*z;
                                out[i*3+2] = origin[2] + f[6]*x + f[7]*y + f[8]*z;
                            }
                        }
                        """
                        ffi.cdef("""
                            void weighted_sum_F9(const int* idx,
                                                 const float* w,
                                                 const float* F9_field,
                                                 int n_rows, int k,
                                                 float* F9_out);
                            void apply_affine_rowwise(const float* F9,
                                                      const float* pos,
                                                      const float* origin,
                                                      float* out,
                                                      size_t n);
                        """)
                        self._cffi_lib = ffi.verify(csrc, extra_compile_args=["-O3"])
                        self._cffi = ffi
                    except Exception:
                        self._cffi_lib = None
                        self._cffi = None

            def _weighted_F_cpu(idx, dists, F_field):
                # Inverse-distance weighting of F9. Handle zero-distance rows.
                M = idx.shape[0]
                w = 1.0 / (np.power(dists, power, dtype=np.float64).astype(dtype, copy=False) + eps)
                zero = dists <= eps
                if zero.any():
                    row_has_zero = zero.any(axis=1)
                    zpos = zero[row_has_zero, :].argmax(axis=1)
                    w[row_has_zero, :] = 0
                    for r, c in enumerate(zpos):
                        w[np.where(row_has_zero)[0][r], int(c)] = 1.0
                if getattr(self, "_cffi_lib", None) is not None and f32:
                    ffi = self._cffi
                    lib = self._cffi_lib
                    idx_i32 = np.asarray(idx, dtype=np.int32, order="C")
                    w_f32 = np.asarray(w, dtype=np.float32, order="C")
                    F_field_f32 = np.asarray(F_field, dtype=np.float32, order="C")
                    out = np.empty((M, 9), dtype=np.float32, order="C")
                    lib.weighted_sum_F9(
                        ffi.from_buffer("int[]", idx_i32),
                        ffi.from_buffer("float[]", w_f32),
                        ffi.from_buffer("float[]", F_field_f32),
                        int(M), int(idx.shape[1]),
                        ffi.from_buffer("float[]", out),
                    )
                    return out.astype(dtype, copy=False)
                else:
                    F_neighbors = F_field[idx]
                    wsum = w.sum(axis=1, keepdims=True)
                    Fw = (F_neighbors * w[..., None]).sum(axis=1)
                    Fw = Fw / np.maximum(wsum, eps)
                    return Fw.astype(dtype, copy=False)

            def _apply_F_cpu(F9, X):
                # Apply per-row affine transforms with origin translation.
                M = X.shape[0]
                if getattr(self, "_cffi_lib", None) is not None and f32:
                    ffi = self._cffi
                    lib = self._cffi_lib
                    F9_f32 = np.asarray(F9, dtype=np.float32, order="C")
                    X_f32 = np.asarray(X, dtype=np.float32, order="C")
                    out = np.empty_like(X_f32)
                    org = np.asarray(origin, dtype=np.float32)
                    lib.apply_affine_rowwise(
                        ffi.from_buffer("float[]", F9_f32),
                        ffi.from_buffer("float[]", X_f32),
                        ffi.from_buffer("float[]", org),
                        ffi.from_buffer("float[]", out),
                        int(M),
                    )
                    return out.astype(dtype, copy=False)
                else:
                    dx = X[:, 0] - origin[0]
                    dy = X[:, 1] - origin[1]
                    dz = X[:, 2] - origin[2]
                    x0 = origin[0] + (F9[:, 0] * dx + F9[:, 1] * dy + F9[:, 2] * dz)
                    x1 = origin[1] + (F9[:, 3] * dx + F9[:, 4] * dy + F9[:, 5] * dz)
                    x2 = origin[2] + (F9[:, 6] * dx + F9[:, 7] * dy + F9[:, 8] * dz)
                    return np.stack([x0, x1, x2], axis=1).astype(dtype, copy=False)

            _ensure_cffi()
            P_np = np.ascontiguousarray(np.asarray(P_all, dtype=dtype))
            F_np = np.ascontiguousarray(np.asarray(F_all, dtype=dtype))
            tree = cKDTree(P_np)

            def _process_chunk(Xchunk):
                X = np.asarray(Xchunk, dtype=dtype)
                if X.ndim != 2 or X.shape[1] != 3:
                    raise ValueError("Each sample chunk must have shape (?,3)")
                if X.shape[0] == 0:
                    return np.empty((0, 3), dtype=dtype)
                d, idx = tree.query(X, k=k)
                if k == 1:
                    d = d[:, None]
                    idx = idx[:, None]
                F9 = _weighted_F_cpu(idx.astype(np.int32), d.astype(dtype), F_np)
                return _apply_F_cpu(F9, X)

        # Sample input: rewrite the stored chunks in place.
        if is_sample:
            if sample.chunk_total is None:
                raise ValueError("Sample is not initialized. Ensure sample metadata is loaded.")
            gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
            for chunk_i in range(1, int(sample.chunk_total) + 1):
                X = sample.load_chunk_positions(chunk_i, use_gpu=use_gpu, raw=True)
                if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
                    continue
                out = _process_chunk(X)
                out_np = cp.asnumpy(out) if use_gpu else np.asarray(out)
                gmin = np.minimum(gmin, out_np.min(axis=0).astype(np.float64))
                gmax = np.maximum(gmax, out_np.max(axis=0).astype(np.float64))
                sample.write_chunk_positions(out_np, chunk_i)
                del X, out, out_np
            self._finalize_sample_aabb(sample, gmin, gmax)
            if hasattr(sample, "record_modification"):
                sample.record_modification("deformation_field", params)
            return None

        # Array or iterable input.
        def _as_iter():
            if hasattr(sample, "shape") and sample.ndim == 2 and sample.shape[1] == 3:
                total = sample.shape[0]
                for s in range(0, total, int(chunk_size)):
                    yield sample[s:s+int(chunk_size)]
            else:
                for blk in sample:
                    yield blk

        iterator = _as_iter()
        if yield_chunks:
            def _gen():
                for Xc in iterator:
                    yield _process_chunk(Xc)
            return _gen()

        outs = [_process_chunk(Xc) for Xc in iterator]
        if len(outs) == 0:
            return xp.empty((0, 3), dtype=dtype)
        if hasattr(sample, "shape") and sample.ndim == 2:
            return xp.concatenate(outs, axis=0)
        return outs

    def plot_field_and_sample_edges_3d(
        self,
        sample,
        field_positions,
        elev=20,
        azim=35,
        show_projections=True,
        projection_plane="min",
        sample_color="C0",
        field_color="C3",
        linewidth=1.8,
        proj_linewidth=1.2,
        proj_alpha=0.5,
        figsize=(7, 7),
        use_gpu=True,
    ):
        """
        Plot sample edges and field AABB edges in 3D.

        Draws two wireframes:
          - The sample box edges from sample.corners.
          - The field axis-aligned bounding box edges from field_positions.

        Optionally draws the XY, YZ, and XZ projection rectangles for each box
        on outer faces (either global min or global max planes).

        Args:
            sample (object): Exposes an 8x3 `corners` array in sample coordinates.
            field_positions (ndarray): Shape (N, 3) FE node positions.
            elev (float): Matplotlib 3D elevation (degrees). Defaults to 20.
            azim (float): Matplotlib 3D azimuth (degrees). Defaults to 35.
            show_projections (bool): If True, overlay XY, YZ, XZ rectangles.
            projection_plane (str): "min" or "max" selection for projection planes.
            sample_color (str): Color for sample edges.
            field_color (str): Color for field edges.
            linewidth (float): Line width for wireframe edges.
            proj_linewidth (float): Line width for projection rectangles.
            proj_alpha (float): Alpha for projection rectangle lines.
            figsize (tuple): Figure size.
            use_gpu (bool): If True and CuPy available, accepts CuPy arrays.

        Returns:
            tuple: (fig, ax) Matplotlib figure and 3D axes.

        Raises:
            ValueError: If field_positions does not have shape (N, 3).
        """
        import numpy as np
        try:
            import cupy as cp
        except Exception:
            cp = None
        import matplotlib.pyplot as plt

        # Pull field positions to CPU if needed for plotting.
        if use_gpu and (cp is not None) and isinstance(field_positions, cp.ndarray):
            P = cp.asnumpy(field_positions)
        else:
            P = np.asarray(field_positions)

        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("field_positions must have shape (N, 3)")

        # Sample corners already on CPU by design.
        C_sample = np.asarray(sample.corners, dtype=np.float64)

        # Field AABB corners in a consistent cube-corner ordering.
        mn = P.min(axis=0)
        mx = P.max(axis=0)
        C_field = np.array([
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2]],
        ], dtype=np.float64)

        # Twelve cube edges for this ordering.
        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]

        def _draw_edges(ax, corners, color, lw, ls="-", alpha_val=1.0):
            # Draw all edges between paired corner indices.
            for i, j in edges:
                x = [corners[i, 0], corners[j, 0]]
                y = [corners[i, 1], corners[j, 1]]
                z = [corners[i, 2], corners[j, 2]]
                ax.plot(x, y, z, color=color, linewidth=lw, linestyle=ls, alpha=alpha_val)

        def _draw_projection_rects(ax, corners, which_plane, color):
            # Draw rectangles on XY at z_plane, on YZ at x_plane, and on XZ at y_plane.
            xmin, ymin, zmin = corners.min(axis=0)
            xmax, ymax, zmax = corners.max(axis=0)

            def _rect_xy(z_plane):
                pts = np.array([
                    [xmin, ymin, z_plane],
                    [xmax, ymin, z_plane],
                    [xmax, ymax, z_plane],
                    [xmin, ymax, z_plane],
                ])
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    ax.plot([pts[a, 0], pts[b, 0]],
                            [pts[a, 1], pts[b, 1]],
                            [pts[a, 2], pts[b, 2]],
                            color=color, linestyle="--", linewidth=proj_linewidth, alpha=proj_alpha)

            def _rect_yz(x_plane):
                pts = np.array([
                    [x_plane, ymin, zmin],
                    [x_plane, ymax, zmin],
                    [x_plane, ymax, zmax],
                    [x_plane, ymin, zmax],
                ])
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    ax.plot([pts[a, 0], pts[b, 0]],
                            [pts[a, 1], pts[b, 1]],
                            [pts[a, 2], pts[b, 2]],
                            color=color, linestyle="--", linewidth=proj_linewidth, alpha=proj_alpha)

            def _rect_xz(y_plane):
                pts = np.array([
                    [xmin, y_plane, zmin],
                    [xmax, y_plane, zmin],
                    [xmax, y_plane, zmax],
                    [xmin, y_plane, zmax],
                ])
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                    ax.plot([pts[a, 0], pts[b, 0]],
                            [pts[a, 1], pts[b, 1]],
                            [pts[a, 2], pts[b, 2]],
                            color=color, linestyle="--", linewidth=proj_linewidth, alpha=proj_alpha)

            x_plane_val, y_plane_val, z_plane_val = which_plane
            _rect_xy(z_plane_val)
            _rect_yz(x_plane_val)
            _rect_xz(y_plane_val)

        # Build figure and 3D axis.
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

        # Draw wireframes.
        _draw_edges(ax, C_sample, sample_color, linewidth, ls="-", alpha_val=1.0)
        _draw_edges(ax, C_field, field_color, linewidth, ls="-", alpha_val=1.0)

        # Combined limits and aspect.
        all_pts = np.vstack([C_sample, C_field])
        mn_all = all_pts.min(axis=0)
        mx_all = all_pts.max(axis=0)
        span = mx_all - mn_all
        pad = 0.03 * (np.linalg.norm(span) if np.all(span > 0) else 1.0)
        ax.set_xlim(mn_all[0] - pad, mx_all[0] + pad)
        ax.set_ylim(mn_all[1] - pad, mx_all[1] + pad)
        ax.set_zlim(mn_all[2] - pad, mx_all[2] + pad)
        try:
            ax.set_box_aspect((span[0] + 1e-12, span[1] + 1e-12, span[2] + 1e-12))
        except Exception:
            pass

        # Draw projection rectangles on chosen outer planes.
        if show_projections:
            if isinstance(projection_plane, str):
                sel = projection_plane.strip().lower()
                if sel == "min":
                    which = (mn_all[0], mn_all[1], mn_all[2])
                elif sel == "max":
                    which = (mx_all[0], mx_all[1], mx_all[2])
                else:
                    raise ValueError('projection_plane must be "min" or "max"')
            else:
                if not (hasattr(projection_plane, "__len__") and len(projection_plane) == 3):
                    raise ValueError("projection_plane must be 'min', 'max', or a 3-tuple of plane values")
                which = tuple(float(v) for v in projection_plane)

            _draw_projection_rects(ax, C_sample, which, sample_color)
            _draw_projection_rects(ax, C_field, which, field_color)

        plt.tight_layout()
        return fig, ax

    # Nodal Deformation Field ----------------------------------------------------
    # Kernel management
    def _ensure_fe_kernels(self, dtype=np.float32):
        """
        Compile and cache CuPy RawKernels for FE tet mapping.

        Args:
            dtype (numpy dtype): Floating dtype (np.float32 or np.float64).

        Notes:
            If CuPy is not available, this is a no-op.
        """
        if (cp is None):
            return
        key = ("fe_tet_kernels", np.dtype(dtype).name)
        if hasattr(self, "_fe_kernels_cache") and key in self._fe_kernels_cache:
            return
        if not hasattr(self, "_fe_kernels_cache"):
            self._fe_kernels_cache = {}

        c_real = "double" if np.dtype(dtype) == np.float64 else "float"

        # Precompute inverse matrices for each tet element in reference space.
        src_precompute = r'''
        extern "C" __global__
        void precompute_tet_minv(const int* __restrict__ elem_nodes,
                                 const %(T)s* __restrict__ Xnodes, // (n_nodes,3)
                                 %(T)s* __restrict__ v0,            // (n_elem,3)
                                 %(T)s* __restrict__ Minv,          // (n_elem,9)
                                 const int n_elem)
        {
            int e = blockDim.x * blockIdx.x + threadIdx.x;
            if (e >= n_elem) return;

            int n0 = elem_nodes[4*e+0];
            int n1 = elem_nodes[4*e+1];
            int n2 = elem_nodes[4*e+2];
            int n3 = elem_nodes[4*e+3];

            const %(T)s X0x = Xnodes[3*n0+0];
            const %(T)s X0y = Xnodes[3*n0+1];
            const %(T)s X0z = Xnodes[3*n0+2];
            const %(T)s a0 = Xnodes[3*n1+0] - X0x;
            const %(T)s a1 = Xnodes[3*n1+1] - X0y;
            const %(T)s a2 = Xnodes[3*n1+2] - X0z;
            const %(T)s b0 = Xnodes[3*n2+0] - X0x;
            const %(T)s b1 = Xnodes[3*n2+1] - X0y;
            const %(T)s b2 = Xnodes[3*n2+2] - X0z;
            const %(T)s c0 = Xnodes[3*n3+0] - X0x;
            const %(T)s c1 = Xnodes[3*n3+1] - X0y;
            const %(T)s c2 = Xnodes[3*n3+2] - X0z;

            %(T)s cx = b1*c2 - b2*c1;
            %(T)s cy = b2*c0 - b0*c2;
            %(T)s cz = b0*c1 - b1*c0;
            %(T)s det = a0*cx + a1*cy + a2*cz;

            %(T)s r0x = cx/det, r0y = cy/det, r0z = cz/det;

            %(T)s r1x = c1*a2 - c2*a1;
            %(T)s r1y = c2*a0 - c0*a2;
            %(T)s r1z = c0*a1 - c1*a0;
            r1x /= det; r1y /= det; r1z /= det;

            %(T)s r2x = a1*b2 - a2*b1;
            %(T)s r2y = a2*b0 - a0*b2;
            %(T)s r2z = a0*b1 - a1*b0;
            r2x /= det; r2y /= det; r2z /= det;

            v0[3*e+0] = X0x; v0[3*e+1] = X0y; v0[3*e+2] = X0z;

            Minv[9*e+0] = r0x; Minv[9*e+1] = r0y; Minv[9*e+2] = r0z;
            Minv[9*e+3] = r1x; Minv[9*e+4] = r1y; Minv[9*e+5] = r1z;
            Minv[9*e+6] = r2x; Minv[9*e+7] = r2y; Minv[9*e+8] = r2z;
        }
        ''' % {"T": c_real}

        # Compute barycentric weights per atom given its containing tet id.
        src_bary = r'''
        extern "C" __global__
        void tet_barycentric(const int* __restrict__ atom_elem,
                             const %(T)s* __restrict__ X,     // (n_atoms,3) reference atoms
                             const %(T)s* __restrict__ v0,    // (n_elem,3)
                             const %(T)s* __restrict__ Minv,  // (n_elem,9)
                             %(T)s* __restrict__ wts,         // (n_atoms,4)
                             const int n_atoms)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n_atoms) return;
            int e = atom_elem[i];
            %(T)s dx = X[3*i+0] - v0[3*e+0];
            %(T)s dy = X[3*i+1] - v0[3*e+1];
            %(T)s dz = X[3*i+2] - v0[3*e+2];

            const %(T)s m00 = Minv[9*e+0], m01 = Minv[9*e+1], m02 = Minv[9*e+2];
            const %(T)s m10 = Minv[9*e+3], m11 = Minv[9*e+4], m12 = Minv[9*e+5];
            const %(T)s m20 = Minv[9*e+6], m21 = Minv[9*e+7], m22 = Minv[9*e+8];

            %(T)s l1 = m00*dx + m01*dy + m02*dz;
            %(T)s l2 = m10*dx + m11*dy + m12*dz;
            %(T)s l3 = m20*dx + m21*dy + m22*dz;
            %(T)s l0 = (%(T)s)1 - (l1 + l2 + l3);

            wts[4*i+0] = l0;
            wts[4*i+1] = l1;
            wts[4*i+2] = l2;
            wts[4*i+3] = l3;
        }
        ''' % {"T": c_real}

        # Gather node indices for each atom's element.
        src_gather_nodes = r'''
        extern "C" __global__
        void gather_atom_nodes(const int* __restrict__ atom_elem,
                               const int* __restrict__ elem_nodes, // (n_elem,4)
                               int* __restrict__ atom_nodes,       // (n_atoms,4)
                               const int n_atoms)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n_atoms) return;
            int e = atom_elem[i];
            atom_nodes[4*i+0] = elem_nodes[4*e+0];
            atom_nodes[4*i+1] = elem_nodes[4*e+1];
            atom_nodes[4*i+2] = elem_nodes[4*e+2];
            atom_nodes[4*i+3] = elem_nodes[4*e+3];
        }
        ''';

        # Interpolate current nodal positions to atom positions using weights.
        src_interpolate = r'''
        extern "C" __global__
        void interpolate_tet(const int* __restrict__ atom_nodes,   // (n_atoms,4)
                             const %(T)s* __restrict__ wts,        // (n_atoms,4)
                             const %(T)s* __restrict__ Xnodes,     // (n_nodes,3) current
                             %(T)s* __restrict__ Xout,             // (n_atoms,3)
                             const int n_atoms)
        {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            if (i >= n_atoms) return;

            int n0 = atom_nodes[4*i+0];
            int n1 = atom_nodes[4*i+1];
            int n2 = atom_nodes[4*i+2];
            int n3 = atom_nodes[4*i+3];

            %(T)s w0 = wts[4*i+0];
            %(T)s w1 = wts[4*i+1];
            %(T)s w2 = wts[4*i+2];
            %(T)s w3 = wts[4*i+3];

            %(T)s x = w0*Xnodes[3*n0+0] + w1*Xnodes[3*n1+0] + w2*Xnodes[3*n2+0] + w3*Xnodes[3*n3+0];
            %(T)s y = w0*Xnodes[3*n0+1] + w1*Xnodes[3*n1+1] + w2*Xnodes[3*n2+1] + w3*Xnodes[3*n3+1];
            %(T)s z = w0*Xnodes[3*n0+2] + w1*Xnodes[3*n1+2] + w2*Xnodes[3*n2+2] + w3*Xnodes[3*n3+2];

            Xout[3*i+0] = x;
            Xout[3*i+1] = y;
            Xout[3*i+2] = z;
        }
        ''' % {"T": c_real}

        kernels = {
            "precompute_tet_minv": cp.RawKernel(src_precompute, "precompute_tet_minv"),
            "tet_barycentric": cp.RawKernel(src_bary, "tet_barycentric"),
            "gather_atom_nodes": cp.RawKernel(src_gather_nodes, "gather_atom_nodes"),
            "interpolate_tet": cp.RawKernel(src_interpolate, "interpolate_tet"),
            "dtype": np.dtype(dtype),
        }
        self._fe_kernels_cache[key] = kernels

    def _get_fe_kernels(self, dtype):
        """
        Retrieve compiled CUDA kernels for FE tet mapping.

        Args:
            dtype (numpy dtype): Floating dtype.

        Returns:
            dict or None: Kernel bundle if CuPy available; else None.
        """
        if (cp is None):
            return None
        self._ensure_fe_kernels(dtype=dtype)
        return self._fe_kernels_cache[("fe_tet_kernels", np.dtype(dtype).name)]

    # FE importers ---------------------------------------------------------------
    def import_fe_nodal_field(self,
                            filepath,
                            columns=None,
                            preset=None,
                            delimiter=None,
                            header_lines=None,
                            position_scale=1.0,
                            use_gpu=True,
                            comments="%",
                            drop_nan_rows=True,
                            dtype=np.float32):
        """
        Import FE nodal field (reference nodes plus displacement or current nodes).

        Supports two column patterns:
            A) x, y, z, u1, u2, u3  -> returns Xref, Xref+U
            B) x0, y0, z0, x1, y1, z1 -> returns Xref, Xcur

        Presets:
            - "comsol_nodes_txt": whitespace, comments="%", header_lines=0,
              columns: x(0) y(1) z(2) u1(3) u2(4) u3(5)

        For COMSOL nodal tables using nm units, set position_scale=1e-9.

        Args:
            filepath (str): Path to input file.
            columns (dict or None): Column mapping. If None, use `preset`.
            preset (str or None): Import preset name.
            delimiter (str or None): Column delimiter or None for whitespace.
            header_lines (int or None): Number of header rows to skip.
            position_scale (float): Isotropic unit scale for positions.
            use_gpu (bool): If True and CuPy available, return CuPy arrays.
            comments (str): Comment character for text loading.
            drop_nan_rows (bool): Drop rows containing NaNs if True.
            dtype (numpy dtype): Target dtype for outputs.

        Returns:
            tuple: (Xref, Xcurr) each of shape (N, 3).

        Raises:
            FileNotFoundError: If file is missing.
            ValueError: On invalid columns or data.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File not found: {}".format(filepath))
        
        # Fast-path: binary .npy/.npz produced by generate_nodal_field(file_format="npy"/"npz")
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".npy", ".npz"):
            if ext == ".npy":
                arr = np.load(filepath)
            else:
                z = np.load(filepath)
                # prefer stable keys, else take the first array
                for k in ("xyzu", "nodes", "array"):
                    if k in z.files:
                        arr = z[k]
                        break
                else:
                    arr = z[z.files[0]]
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != 6:
                raise ValueError("binary nodes file must be shape (N,6) with columns x y z u1 u2 u3")
            Xref = arr[:, 0:3].astype(dtype, copy=False)
            U    = arr[:, 3:6].astype(dtype, copy=False)
            Xcurr = Xref + U
            if use_gpu and (cp is not None):
                Xref = cp.asarray(Xref, dtype=dtype)
                Xcurr = cp.asarray(Xcurr, dtype=dtype)
            self._Xref = Xref
            self._Xcurr = Xcurr
            self._record_fe_field_import(filepath, 1.0)
            return self._Xref, self._Xcurr

        PRESETS = {
            "comsol_nodes_csv": {
                "delimiter": ",",
                "header_lines": 0,
                "columns": {"x":0,"y":1,"z":2,"u1":3,"u2":4,"u3":5},
            },
            "comsol_nodes_txt": {
                "delimiter": None,
                "header_lines": 0,
                "columns": {"x":0,"y":1,"z":2,"u1":3,"u2":4,"u3":5},
            },
            "generic_xyzu": {
                "delimiter": None,
                "header_lines": 0,
                "columns": {"x":0,"y":1,"z":2,"u1":3,"u2":4,"u3":5},
            },
            "generic_x0x1": {
                "delimiter": None,
                "header_lines": 0,
                "columns": {"x0":0,"y0":1,"z0":2,"x1":3,"y1":4,"z1":5},
            },
        }

        cfg = PRESETS.get(preset, {}) if preset else {}
        delim = delimiter if delimiter is not None else cfg.get("delimiter", None)
        skip = header_lines if header_lines is not None else int(cfg.get("header_lines", 0))

        if isinstance(columns, dict):
            colmap = dict(columns)
        else:
            preset_cols = cfg.get("columns", None)
            colmap = dict(preset_cols) if isinstance(preset_cols, dict) else None

        if colmap is None:
            raise ValueError("columns or a valid preset must be provided")

        has_u = all(k in colmap for k in ("x","y","z","u1","u2","u3"))
        has_x0x1 = all(k in colmap for k in ("x0","y0","z0","x1","y1","z1"))
        if not (has_u or has_x0x1):
            raise ValueError("columns must provide either {x,y,z,u1,u2,u3} or {x0,y0,z0,x1,y1,z1}")

        usecols = (
            [colmap["x"], colmap["y"], colmap["z"], colmap["u1"], colmap["u2"], colmap["u3"]]
            if has_u else
            [colmap["x0"], colmap["y0"], colmap["z0"], colmap["x1"], colmap["y1"], colmap["z1"]]
        )

        data = np.genfromtxt(
            filepath,
            delimiter=delim,
            comments=comments,
            skip_header=skip,
            usecols=usecols,
            dtype=np.float64,
            invalid_raise=False
        )
        data = np.atleast_2d(data)
        if data.shape[1] != 6:
            raise ValueError("expected 6 selected columns, got {}".format(data.shape[1]))

        if np.isnan(data).any():
            if drop_nan_rows:
                data = data[~np.isnan(data).any(axis=1)]
            else:
                raise ValueError("NaNs detected in nodal file")

        if has_u:
            Xref = data[:, 0:3]
            U = data[:, 3:6]
            Xcurr = Xref + U
        else:
            Xref = data[:, 0:3]
            Xcurr = data[:, 3:6]

        if position_scale != 1.0:
            s = float(position_scale)
            Xref = Xref * s
            Xcurr = Xcurr * s

        Xref = Xref.astype(dtype, copy=False)
        Xcurr = Xcurr.astype(dtype, copy=False)

        if use_gpu and (cp is not None):
            Xref = cp.asarray(Xref, dtype=dtype)
            Xcurr = cp.asarray(Xcurr, dtype=dtype)

        self._Xref = Xref
        self._Xcurr = Xcurr
        self._record_fe_field_import(filepath, position_scale)
        return self._Xref, self._Xcurr

    def import_fe_connectivity(self,
                            filepath,
                            columns=None,
                            preset=None,
                            delimiter=None,
                            header_lines=None,
                            one_based=True,
                            use_gpu=True,
                            dtype=np.int32):
        """
        Import FE element connectivity for 4-node elements.

        Supported modes:

            1) preset == "comsol_mesh_txt"
               - Parses a COMSOL mesh text export (for example, Mesh1.txt).
               - Auto-detects and prefers the "tet" block; if none, falls back
                 to a "quad" block.
               - Requires number of nodes per element == 4.
               - Uses "lowest mesh point index" to determine 0-based vs 1-based
                 indexing when available; otherwise falls back to `one_based`.
               - Returns an (Ne, 4) array of 0-based node indices.

            2) Generic CSV/whitespace
               - Loads exactly 4 integer columns according to delimiter/header.
               - Applies 1->0 index shift if one_based=True.

        Args:
            filepath (str): Path to input file.
            columns (sequence or None): Custom 4-column mapping for generic mode.
            preset (str or None): "comsol_mesh_txt" or a generic preset.
            delimiter (str or None): Column delimiter for generic mode.
            header_lines (int or None): Header lines to skip for generic mode.
            one_based (bool): If True, subtract 1 from loaded indices in generic mode.
            use_gpu (bool): If True and CuPy available, return CuPy arrays.
            dtype (numpy dtype): Output integer dtype.

        Returns:
            ndarray: Element connectivity, shape (E, 4), 0-based.

        Raises:
            FileNotFoundError: If file is missing.
            ValueError: On invalid format or content.
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File not found: {}".format(filepath))
        
        # Fast-path: binary .npy/.npz connectivity (0-based inside the file)
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".npy", ".npz"):
            if ext == ".npy":
                arr = np.load(filepath)
            else:
                z = np.load(filepath)
                arr = z["tet4"] if "tet4" in z.files else z[z.files[0]]
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("binary connectivity must be shape (E,4)")
            elem_nodes = arr.astype(dtype, copy=False)  # already 0-based
            if use_gpu and (cp is not None):
                elem_nodes = cp.asarray(elem_nodes, dtype=dtype)
            self._elem_nodes = elem_nodes
            self._fe_mesh_file = os.path.basename(filepath)
            self._mesh_points = None
            return elem_nodes

        if preset == "comsol_mesh_txt":
            # Read entire file to parse COMSOL metadata and element blocks.
            with open(filepath, "r") as f:
                lines = f.read().splitlines()

            def _left_int(s):
                # Parse integer in the left segment before any "#".
                left = s.split("#", 1)[0].strip()
                if not left:
                    raise ValueError("expected integer on line: {}".format(s))
                tok = left.split()[0]
                try:
                    return int(tok)
                except ValueError:
                    return int(float(tok))

            # 1) Determine index base from "lowest mesh point index", if present.
            lowest_index = None
            for ln in lines:
                t = ln.strip().lower()
                if "lowest mesh point index" in t:
                    lowest_index = _left_int(ln)
                    break

            # 1b) Mesh point coordinates. Connectivity indexes this order, which
            #     is not the row order of a COMSOL nodal export.
            npts = None
            for ln in lines:
                if "number of mesh points" in ln.lower():
                    npts = _left_int(ln)
                    break
            mp_i = -1
            for i, ln in enumerate(lines):
                if "# Mesh point coordinates" in ln:
                    mp_i = i
                    break
            if npts is None or mp_i < 0:
                raise ValueError("mesh point block not found in COMSOL mesh")
            rows = []
            j = mp_i + 1
            while j < len(lines) and len(rows) < npts:
                s = lines[j].strip()
                j += 1
                if not s or s.startswith("#"):
                    continue
                rows.append(s.split("#", 1)[0].split()[:3])
            if len(rows) != npts:
                raise ValueError("expected {} mesh points, read {}".format(npts, len(rows)))
            mesh_points = np.asarray(rows, dtype=np.float64)
            if mesh_points.ndim != 2 or mesh_points.shape[1] != 3:
                raise ValueError("mesh points must have 3 coordinates")

            # 2) Locate an element block: prefer "tet"; else "quad".
            tet_i = -1
            quad_i = -1
            for i, ln in enumerate(lines):
                s = ln.strip().lower()
                if "# type name" in s:
                    # COMSOL writes "<sdim> <etype> # type name"
                    left = s.split("#", 1)[0].strip()
                    toks = left.split()
                    etype = toks[-1] if toks else ""
                    if etype == "tet" and tet_i < 0:
                        tet_i = i
                    elif etype == "quad" and quad_i < 0:
                        quad_i = i

            if tet_i >= 0:
                block_i = tet_i
                elem_type = "tet"
            elif quad_i >= 0:
                block_i = quad_i
                elem_type = "quad"
            else:
                raise ValueError("no 'tet' or 'quad' block found in COMSOL mesh")

            # 3) Extract counts inside the chosen block.
            j = block_i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                raise ValueError("unexpected end of file while reading nodes-per-element")
            nper = _left_int(lines[j])  # "4 # number of nodes per element"
            j += 1

            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                raise ValueError("unexpected end of file while reading number of elements")
            nelems = _left_int(lines[j])  # ". # number of elements"
            j += 1

            # Advance to "# Elements".
            while j < len(lines) and "# Elements" not in lines[j]:
                j += 1
            if j >= len(lines):
                raise ValueError("Elements header not found in {} block".format(elem_type))
            j += 1  # First element line

            if nper != 4:
                raise ValueError("expected 4 nodes per element in {} block, got {}".format(elem_type, nper))

            # 4) Read exactly 'nelems' connectivity lines (first 4 integers per line).
            arr = np.empty((nelems, 4), dtype=np.int64)
            k = 0
            while j < len(lines) and k < nelems:
                raw = lines[j].strip()
                j += 1
                if not raw or raw.startswith("#"):
                    continue
                left = raw.split("#", 1)[0].strip()
                if not left:
                    continue
                parts = left.replace("\t", " ").split()
                if len(parts) < 4:
                    raise ValueError("bad {} line at element {}: '{}'".format(elem_type, k, raw))
                try:
                    e = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])]
                except ValueError:
                    e = [int(float(parts[0])), int(float(parts[1])),
                         int(float(parts[2])), int(float(parts[3]))]
                arr[k, :] = e
                k += 1

            if k != nelems:
                raise ValueError("expected {} {} elements, read {}".format(nelems, elem_type, k))

            # 5) Adjust indexing to 0-based.
            if lowest_index is None:
                local_one_based = bool(one_based)
            else:
                local_one_based = (int(lowest_index) == 1)
            if local_one_based:
                arr -= 1

            elem_nodes = arr.astype(dtype, copy=False)
            if use_gpu and (cp is not None):
                elem_nodes = cp.asarray(elem_nodes, dtype=dtype)

            self._elem_nodes = elem_nodes
            self._fe_mesh_file = os.path.basename(filepath)
            self._mesh_points = mesh_points
            self._fe_nodes_matched = False
            if self._Xref is not None:
                self._match_fe_nodes_to_mesh()
            return elem_nodes

        # ---------------- Generic whitespace/CSV fallback ----------------
        PRESETS = {
            "generic_tet4_csv":  {"delimiter": ",",  "header_lines": 1, "columns": [0, 1, 2, 3]},
            "generic_tet4_ws":   {"delimiter": None, "header_lines": 0, "columns": [0, 1, 2, 3]},
            "generic_quad4_csv": {"delimiter": ",",  "header_lines": 1, "columns": [0, 1, 2, 3]},
            "generic_quad4_ws":  {"delimiter": None, "header_lines": 0, "columns": [0, 1, 2, 3]},
        }

        cfg = PRESETS.get(preset, {}) if preset else {}
        delim = delimiter if delimiter is not None else cfg.get("delimiter", None)
        skip = header_lines if header_lines is not None else int(cfg.get("header_lines", 0))

        if columns is None:
            cols = cfg.get("columns", [0, 1, 2, 3])
        else:
            if not hasattr(columns, "__len__") or len(columns) != 4:
                raise ValueError("columns must be a 4-length sequence for 4-node connectivity")
            cols = list(columns)

        data = np.genfromtxt(
            filepath,
            delimiter=delim,
            comments="%",
            skip_header=skip,
            usecols=cols,
            dtype=np.int64,
            invalid_raise=False
        )
        data = np.atleast_2d(data)
        if data.shape[1] != 4:
            raise ValueError("expected 4 columns for 4-node connectivity")

        if one_based:
            data = data - 1

        elem_nodes = data.astype(dtype, copy=False)
        if use_gpu and (cp is not None):
            elem_nodes = cp.asarray(elem_nodes, dtype=dtype)

        self._elem_nodes = elem_nodes
        self._fe_mesh_file = os.path.basename(filepath)
        self._mesh_points = None
        return elem_nodes

    def _record_fe_field_import(self, filepath, position_scale):
        """
        Store provenance of a freshly imported nodal field and match it to the
        mesh points when a COMSOL mesh is already loaded.

        Args:
            filepath (str): Nodal field file that was imported.
            position_scale (float): Scale applied to the file positions.
        """
        X = self._Xref
        if (cp is not None) and isinstance(X, cp.ndarray):
            X = cp.asnumpy(X)
        self._Xref_import = np.array(X, dtype=np.float64, copy=True)
        self._fe_field_file = os.path.basename(filepath)
        self._fe_position_scale = float(position_scale)
        self._fe_nodes_matched = False
        if getattr(self, "_mesh_points", None) is not None:
            self._match_fe_nodes_to_mesh()

    def _match_fe_nodes_to_mesh(self):
        """
        Reorder the nodal field rows to the mesh point order of the loaded mesh.

        COMSOL nodal exports and mesh exports list the same nodes in different
        orders, while element connectivity indexes the mesh order. Each mesh
        point is matched to the nodal row at the same coordinate (scaled by the
        nodal-field position scale) and `_Xref`, `_Xcurr` are permuted to mesh
        order. Matching is skipped when no mesh points or no import-order
        coordinates are available (binary or generic inputs, or a clipped field).

        Raises:
            ValueError: If the counts differ, a mesh point has no nodal row within
                1e-6 of the mesh extent, or the match is not one-to-one.
        """
        from scipy.spatial import cKDTree
        if getattr(self, "_fe_nodes_matched", False):
            return
        mesh = getattr(self, "_mesh_points", None)
        Ximp = getattr(self, "_Xref_import", None)
        if mesh is None or Ximp is None or self._Xref is None:
            return
        scale = float(getattr(self, "_fe_position_scale", 1.0))
        mesh_s = mesh * scale
        n_mesh = int(mesh_s.shape[0])
        n_rows = int(Ximp.shape[0])
        if n_mesh != n_rows or int(self._Xref.shape[0]) != n_rows:
            raise ValueError("nodal field has {} rows but the mesh has {} points".format(n_rows, n_mesh))

        extent = float(np.max(mesh_s.max(axis=0) - mesh_s.min(axis=0)))
        tol = 1e-6 * max(extent, 1e-300)
        d, row = cKDTree(Ximp).query(mesh_s, k=1)
        dmax = float(d.max()) if d.size else 0.0
        if dmax > tol:
            raise ValueError("nodal field rows do not coincide with the mesh points: "
                             "max distance {:.3g} exceeds tolerance {:.3g}".format(dmax, tol))
        if np.unique(row).size != row.size:
            raise ValueError("nodal field rows and mesh points are not in one-to-one correspondence")

        perm = np.asarray(row, dtype=np.int64)
        n_moved = int(np.count_nonzero(perm != np.arange(n_rows)))
        if n_moved > 0:
            def _take(arr):
                if (cp is not None) and isinstance(arr, cp.ndarray):
                    return arr[cp.asarray(perm)]
                return np.asarray(arr)[perm]
            self._Xref = _take(self._Xref)
            if self._Xcurr is not None:
                self._Xcurr = _take(self._Xcurr)
            self._Xref_import = Ximp[perm]
        self._fe_nodes_matched = True
        self._log("normal", "matched {} nodal rows to mesh points (max distance {:.3g}, "
                            "{} rows reordered)".format(n_rows, dmax, n_moved))

    # FE transforms --------------------------------------------------------------
    def zero_fe_nodal_field(self, center_mode="bbox", return_shift=False):
        """
        Translate the FE nodal field so that its center is at the origin.

        The center is computed from the reference nodal coordinates (self._Xref)
        and the same translation is applied to the current coordinates
        (self._Xcurr) if present, preserving displacements.

        Args:
            center_mode (str): Either "bbox" for the axis-aligned bounding-box
                center (default) or "mean" for the arithmetic mean.
            return_shift (bool): If True, also return the 3-vector shift applied
                as a NumPy array of dtype float.

        Returns:
            tuple:
                (Xref_out, Xcurr_out) or
                (Xref_out, Xcurr_out, shift) if return_shift is True.

        Raises:
            ValueError: If the FE nodal field is not initialized or shapes are invalid.
        """
        if self._Xref is None:
            raise ValueError("FE nodal field is not initialized. Call import_fe_nodal_field(...) first.")

        # Select backend from existing storage (do not move data between CPU/GPU).
        xp = cp if ((cp is not None) and isinstance(self._Xref, cp.ndarray)) else np
        Xr = self._Xref

        if Xr.ndim != 2 or Xr.shape[1] != 3:
            raise ValueError("Xref must have shape (N,3)")

        mode = str(center_mode).strip().lower()
        if mode == "bbox":
            mn = xp.min(Xr, axis=0)
            mx = xp.max(Xr, axis=0)
            c = (mn + mx) * 0.5
        elif mode == "mean":
            c = xp.mean(Xr, axis=0)
        else:
            raise ValueError('center_mode must be "bbox" or "mean"')

        # Convert to plain Python floats so we can reuse the same shift for both backends.
        if xp is np:
            cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        else:
            cx, cy, cz = float(c[0].item()), float(c[1].item()), float(c[2].item())

        def _shift_for(arr):
            # Build a shift vector with the correct backend and dtype for 'arr'.
            if (cp is not None) and isinstance(arr, cp.ndarray):
                return cp.asarray([cx, cy, cz], dtype=arr.dtype)
            else:
                return np.asarray([cx, cy, cz], dtype=arr.dtype if hasattr(arr, "dtype") else np.float32)

        # Apply translation to reference and current nodes.
        shift_ref = _shift_for(self._Xref)
        Xref_out = self._Xref - shift_ref
        self._Xref = Xref_out

        Xcurr_out = None
        if self._Xcurr is not None:
            shift_cur = _shift_for(self._Xcurr)
            Xcurr_out = self._Xcurr - shift_cur
            self._Xcurr = Xcurr_out

        if return_shift:
            shift_np = np.array([cx, cy, cz], dtype=float)
            return self._Xref, self._Xcurr, shift_np
        return self._Xref, self._Xcurr
    
    def transform_fe_nodal_field(self,
                                 Xref,
                                 Xcur,
                                 position_scale=1.0,
                                 disp_alpha=None,
                                 translate=None,
                                 rotate_axis=None,
                                 rotate_angle=None,
                                 rotate_matrix=None,
                                 degrees=True,
                                 origin=(0.0, 0.0, 0.0),
                                 use_gpu=True,
                                 dtype=None,
                                 copy=True,
                                 inplace=True):
        """
        Transform FE nodal reference/current positions consistently.

        Steps:
          1) optional isotropic scale of both Xref and Xcur
          2) optional displacement amplitude scaling: Xcur := Xref + alpha*(Xcur - Xref)
          3) optional rigid rotation about origin (both)
          4) optional translation (both)

        With `inplace` the results replace the stored `_Xref` and `_Xcurr`, as
        `zero_fe_nodal_field` and `clip_fe_mesh_to_sample` do.

        Args:
            Xref (ndarray): Reference nodal positions, shape (N, 3).
            Xcur (ndarray): Current nodal positions, shape (N, 3).
            position_scale (float): Isotropic scale factor for both arrays.
            disp_alpha (float or None): If provided, scale displacement amplitude.
            translate (sequence or None): Optional translation vector length 3.
            rotate_axis (sequence or None): Axis for axis-angle rotation.
            rotate_angle (float or None): Rotation angle (deg if `degrees` True).
            rotate_matrix (ndarray or None): Use this 3x3 rotation directly.
            degrees (bool): Interpret rotate_angle in degrees if True.
            origin (sequence): Rotation origin, length 3.
            use_gpu (bool): Use CuPy if available.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            copy (bool): If True, operate on copies.
            inplace (bool): If True, store the outputs as the field used by
                the apply and clip methods.

        Returns:
            tuple: (Xref_out, Xcur_out), both shape (N, 3).
        """
        xp = self._select_backend(use_gpu)
        dtype = self._infer_dtype(Xref, Xcur, dtype)
        Xr = xp.asarray(Xref, dtype=dtype)
        Xc = xp.asarray(Xcur, dtype=dtype)
        if copy:
            Xr = Xr.copy()
            Xc = Xc.copy()

        if position_scale is not None and position_scale != 1.0:
            s = dtype.type(float(position_scale))
            Xr *= s
            Xc *= s

        if disp_alpha is not None:
            a = dtype.type(float(disp_alpha))
            Xc = Xr + a * (Xc - Xr)

        R = self.build_rotation_matrix(
            rotate_axis=rotate_axis,
            rotate_angle=rotate_angle,
            rotate_matrix=rotate_matrix,
            degrees=degrees,
            use_gpu=use_gpu,
            dtype=dtype,
        )
        if R is not None:
            Xr = self.rotate_positions(Xr, R, origin=origin, use_gpu=use_gpu, dtype=dtype, copy=False)
            Xc = self.rotate_positions(Xc, R, origin=origin, use_gpu=use_gpu, dtype=dtype, copy=False)

        if translate is not None:
            Xr = self.translate_positions(Xr, translate, use_gpu=use_gpu, dtype=dtype, copy=False)
            Xc = self.translate_positions(Xc, translate, use_gpu=use_gpu, dtype=dtype, copy=False)

        if inplace:
            self._Xref = Xr
            self._Xcurr = Xc
        return Xr, Xc

    # FE clipping ----------------------------------------------------------------
    def clip_fe_mesh_to_sample(self,
                            sample,
                            Xref_nodes=None,
                            elem_nodes=None,
                            margin=None,
                            use_gpu=True):
        """
        Clip FE mesh to elements intersecting the sample AABB (+margin).

        The test is performed in REFERENCE space and keeps entire elements. An
        element is kept if:
            - any of its nodes lie inside the sample AABB (+margin), or
            - the element AABB intersects the sample AABB (+margin).

        If a COMSOL mesh is loaded, the nodal rows are first matched to the
        mesh point order so the connectivity indexes the right nodes.

        Args:
            sample (object): Exposes `corners` (8x3) in same frame as Xref.
            Xref_nodes (ndarray or None): Reference nodal coords; defaults to self.Xref.
            elem_nodes (ndarray or None): Element connectivity; defaults to self.elem_nodes.
            margin (float or None): Non-negative expansion added to the sample
                AABB on all sides. None uses 3 x the median node spacing so
                atoms near the sample faces keep a full MLS stencil.
            use_gpu (bool): If True and CuPy available, do computations on GPU.

        Returns:
            tuple: (Xref_out, Xcurr_out_or_None, elem_out).

        Raises:
            ValueError: If inputs are missing/invalid or the clip removes all elements.
        """
        # Select backend.
        xp = cp if (use_gpu and (cp is not None)) else np

        # Match nodal rows to the mesh order before indexing with connectivity.
        if Xref_nodes is None and elem_nodes is None:
            self._match_fe_nodes_to_mesh()

        # Resolve inputs and validate.
        Xref_nodes = self.Xref if Xref_nodes is None else Xref_nodes
        elem_nodes = self.elem_nodes if elem_nodes is None else elem_nodes
        if Xref_nodes is None or elem_nodes is None:
            raise ValueError("Xref_nodes and elem_nodes must be provided or initialized in the class.")

        if margin is None:
            margin = 3.0 * self._fe_node_spacing()
            self._log("normal", "clip_fe_mesh_to_sample: margin set to {:.4g} (3 x node spacing)".format(margin))

        Xn = xp.asarray(Xref_nodes)
        En = xp.asarray(elem_nodes)
        if Xn.ndim != 2 or Xn.shape[1] != 3:
            raise ValueError("Xref_nodes must have shape (N,3)")
        if En.ndim != 2:
            raise ValueError("elem_nodes must have shape (E,k)")
        k = int(En.shape[1])

        # Build sample AABB from corners (+margin).
        corners = np.asarray(sample.corners, dtype=np.float64)
        cmin = corners.min(axis=0)
        cmax = corners.max(axis=0)
        if margin is not None and float(margin) > 0.0:
            m = float(margin)
            cmin = cmin - m
            cmax = cmax + m
        cmin_xp = xp.asarray(cmin, dtype=Xn.dtype)
        cmax_xp = xp.asarray(cmax, dtype=Xn.dtype)

        # Criterion A: any node of the element lies inside sample AABB (+margin).
        inside_nodes = (
            (Xn[:, 0] >= cmin_xp[0]) & (Xn[:, 0] <= cmax_xp[0]) &
            (Xn[:, 1] >= cmin_xp[1]) & (Xn[:, 1] <= cmax_xp[1]) &
            (Xn[:, 2] >= cmin_xp[2]) & (Xn[:, 2] <= cmax_xp[2])
        )
        any_node_inside = inside_nodes[En].any(axis=1)

        # Criterion B: element AABB intersects sample AABB (+margin).
        Xe = Xn[En]  # (E, k, 3)
        emin = Xe.min(axis=1)  # (E, 3)
        emax = Xe.max(axis=1)  # (E, 3)
        aabb_intersect = (
            (emax[:, 0] >= cmin_xp[0]) & (emin[:, 0] <= cmax_xp[0]) &
            (emax[:, 1] >= cmin_xp[1]) & (emin[:, 1] <= cmax_xp[1]) &
            (emax[:, 2] >= cmin_xp[2]) & (emin[:, 2] <= cmax_xp[2])
        )

        keep_elem = any_node_inside | aabb_intersect

        # Robust truth check across backends.
        def _any(arr):
            if xp is np:
                return bool(arr.any())
            else:
                return bool(cp.asnumpy(arr.any()))

        if not _any(keep_elem):
            raise ValueError("No elements intersect the sample AABB (+margin). Increase margin or verify inputs.")

        # Keep elements and all their nodes; build compact node set and reindex.
        En_keep = En[keep_elem, :]
        keep_nodes_unique = xp.unique(En_keep.ravel())

        # Map old node indices -> new [0..n_keep-1].
        map_dtype = xp.int64 if xp is np else cp.int64
        new_ids = xp.arange(int(keep_nodes_unique.shape[0]), dtype=map_dtype)
        remap = -xp.ones((Xn.shape[0],), dtype=map_dtype)
        remap[keep_nodes_unique] = new_ids
        En_new = remap[En_keep].astype(xp.int32, copy=False)

        # Slice node arrays.
        Xref_new = Xn[keep_nodes_unique]

        # Slice current configuration if present.
        Xcurr_new = None
        if self._Xcurr is not None:
            if (cp is not None) and isinstance(self._Xcurr, cp.ndarray):
                idx_gpu = keep_nodes_unique if isinstance(keep_nodes_unique, cp.ndarray) else cp.asarray(keep_nodes_unique)
                Xcurr_new = self._Xcurr[idx_gpu]
            else:
                idx_cpu = cp.asnumpy(keep_nodes_unique) if ((cp is not None) and isinstance(keep_nodes_unique, cp.ndarray)) else np.asarray(keep_nodes_unique)
                Xcurr_new = np.asarray(self._Xcurr)[idx_cpu]

        # Update class properties. Rows no longer correspond to the mesh points.
        self._Xref = Xref_new
        if Xcurr_new is not None:
            self._Xcurr = Xcurr_new
        self._elem_nodes = En_new
        self._Xref_import = None

        return self._Xref, (self._Xcurr if self._Xcurr is not None else None), self._elem_nodes
    
    # FE apply (atoms) -----------------------------------------------------------
    def _estimate_mls_batch_bytes(self, n_rows, k, dtype):
        """
        Heuristic helper for sizing MLS mini-batches.

        Not used outside this module. Returns a conservative per-row byte estimate
        to keep temporary arrays (B, A, b, neighbors, weights) under a few hundred MB.
        """
        bytes_per = 8 if np.dtype(dtype) == np.float64 else 4
        m = 10  # quadratic basis size in 3D
        # Crude per-row accounting: B(M,k,m) + A(M,m,m) + b(M,m,3) + neighbors/weights
        per_row = bytes_per * (k * m + m * m + m * 3 + k * 4)
        return int(max(1, per_row))
    
    def _mls_quadratic_displacement(self,
                                    Xq,          # (M,3) query points
                                    P_nodes,     # (N,3) nodal positions
                                    U_nodes,     # (N,3) nodal displacements
                                    idx,         # (M,k) neighbor indices per query
                                    d2,          # (M,k) neighbor squared distances
                                    power=2.0,
                                    eps=1e-12,
                                    reg=1e-6,
                                    use_gpu=True,
                                    dtype=None):
        """
        Compute MLS-quadratic displacement per query from nodal data.

        The fit is performed in query-centered coordinates. For each query i,
        define shifts s_ij = P_nodes[idx[i,j]] - Xq[i], and the 10-term basis
        p(s) = [1, sx, sy, sz, sx^2, sy^2, sz^2, sx*sy, sx*sz, sy*sz].
        With weights w_ij = 1 / (sqrt(d2_ij) + eps)^power, the normal equations are:
            A_i a_i = b_i
        where
            A_i = sum_j w_ij * p(s_ij) p(s_ij)^T   (shape 10x10)
            b_i = sum_j w_ij * p(s_ij) U_ij^T      (shape 10x3)
        The predicted displacement at the query is a_i[0,:] (constant term).

        Exact-hit handling:
        If any distance is <= eps for a query, return that neighbor's U directly.

        Args:
            Xq (ndarray): Query points, shape (M,3).
            P_nodes (ndarray): Node positions, shape (N,3).
            U_nodes (ndarray): Node displacements, shape (N,3).
            idx (ndarray): Neighbor indices, shape (M,k), int32/64.
            d2 (ndarray): Squared distances, shape (M,k).
            power (float): Inverse-distance weight power.
            eps (float): Small constant for stability.
            reg (float): Tikhonov regularization scaling applied as
                        A_i += (reg * sum_j w_ij) * I. Defaults to 1e-6.
            use_gpu (bool): Use CuPy if True and available, else NumPy.
            dtype (dtype or None): Floating dtype to enforce.

        Returns:
            ndarray: Predicted displacements at queries, shape (M,3).
        """
        import numpy as np
        try:
            import cupy as cp
        except Exception:
            cp = None

        xp = cp if (use_gpu and (cp is not None)) else np
        T = np.dtype(dtype or np.float32)

        # Gather neighbors and distances
        Pn = P_nodes[idx]                    # (M,k,3)
        Un = U_nodes[idx]                    # (M,k,3)
        d2 = d2.astype(T, copy=False)        # (M,k)
        d = xp.sqrt(d2 + T.type(eps))        # (M,k)
        w = T.type(1.0) / xp.power(d + T.type(eps), T.type(power))  # (M,k)
        wsum = xp.maximum(w.sum(axis=1), T.type(1e-20))             # (M,)

        # Identify exact hits and prebuild a mask
        zero = (d2 <= T.type(eps) * T.type(eps))
        rows_zero = zero.any(axis=1)  # (M,)

        # Query-centered coordinates and per-row normalization
        M = Xq.shape[0]
        s = Pn - Xq[:, None, :]                    # (M,k,3)
        # h: robust local scale (median neighbor distance)
        h = xp.median(d, axis=1)                   # (M,)
        h = xp.maximum(h, T.type(1e-9))            # avoid div-by-zero
        invh = T.type(1.0) / h
        sN = s * invh[:, None, None]               # normalized local coords

        sx = sN[:, :, 0]; sy = sN[:, :, 1]; sz = sN[:, :, 2]
        # Build 10-term quadratic basis in normalized coords
        B = xp.stack([
            xp.ones_like(sx),
            sx, sy, sz,
            sx*sx, sy*sy, sz*sz,
            sx*sy, sx*sz, sy*sz
        ], axis=2)                                 # (M,k,10)

        # Weighted normal equations: A = B^T W B, b = B^T W U
        # Use einsum available in NumPy/CuPy
        A = xp.einsum('mki,mk,mkj->mij', B, w, B)  # (M,10,10)
        b = xp.einsum('mki,mk,mkq->miq', B, w, Un) # (M,10,3)

        # Scale-aware regularization
        I = xp.eye(10, dtype=T)[None, :, :]
        A = A + (T.type(reg) * wsum)[:, None, None] * I

        # Solve; fall back to IDW if needed
        U_pred = xp.empty((M, 3), dtype=T)
        solved_mask = xp.ones((M,), dtype=bool if xp is np else cp.bool_)
        try:
            coef = xp.linalg.solve(A, b)           # (M,10,3)
            U_pred[:] = coef[:, 0, :]              # constant term at query
        except Exception:
            # Mark all rows as unsolved; fallback below
            if xp is np:
                solved_mask[:] = False
            else:
                solved_mask = cp.zeros((M,), dtype=cp.bool_)

        # Fallback for any failed rows or rows with NaN/Inf in result
        if xp is np:
            bad = (~solved_mask) | (~np.isfinite(U_pred).all(axis=1))
        else:
            bad = (~solved_mask) | (~cp.isfinite(U_pred).all(axis=1))
        if (xp is np and bad.any()) or (xp is cp and bool(cp.any(bad))):
            # Simple IDW average of neighbor displacements
            w_idw = w[bad, :]
            wsum_idw = xp.maximum(w_idw.sum(axis=1, keepdims=True), T.type(1e-20))
            U_pred_bad = (Un[bad, :, :] * w_idw[:, :, None]).sum(axis=1) / wsum_idw
            U_pred = U_pred.copy()  # ensure we can assign even on CuPy view
            U_pred[bad, :] = U_pred_bad

        # Clamp clearly nonphysical predictions using local envelope
        # u_cap = factor * max neighbor displacement magnitude
        norms = xp.linalg.norm(Un, axis=2)                 # (M,k)
        u_cap = T.type(8.0) * xp.maximum(norms.max(axis=1), T.type(1e-9))  # (M,)
        up_norm = xp.linalg.norm(U_pred, axis=1)           # (M,)
        if xp is np:
            mask = up_norm > u_cap
            if mask.any():
                scale = (u_cap[mask] / (up_norm[mask] + T.type(1e-20)))[:, None]
                U_pred[mask, :] *= scale
        else:
            mask = up_norm > u_cap
            if bool(cp.any(mask)):
                scale = (u_cap[mask] / (up_norm[mask] + T.type(1e-20)))[:, None]
                U_pred[mask, :] *= scale

        # Exact-hit override: if a node coincides with the query, use its nodal U
        if (xp is np and rows_zero.any()) or (xp is cp and bool(cp.any(rows_zero))):
            if xp is np:
                row_ids = np.where(rows_zero)[0]
                zpos = zero[rows_zero, :].argmax(axis=1)
                direct_ids = idx[rows_zero, :][np.arange(row_ids.shape[0]), zpos]
                U_pred[row_ids, :] = U_nodes[direct_ids, :]
            else:
                row_ids = cp.where(rows_zero)[0]
                zpos = cp.argmax(zero[rows_zero, :], axis=1)
                idx_rows = idx[rows_zero, :]
                rr = cp.arange(idx_rows.shape[0], dtype=cp.int32)
                direct_ids = idx_rows[rr, zpos]
                U_pred[row_ids, :] = U_nodes[direct_ids, :]

        return U_pred.astype(T, copy=False)

    @staticmethod
    def _median_nn_spacing(P, max_queries=200000):
        """
        Median nearest-neighbour distance of a point set.

        Args:
            P (ndarray): Host array of shape (N, 3).
            max_queries (int): Upper bound on the number of points whose nearest
                neighbour is evaluated; larger sets are sampled with a fixed
                stride. The tree itself always holds every point.

        Returns:
            float: Median nearest-neighbour distance, or 1.0 for fewer than two points.
        """
        from scipy.spatial import cKDTree
        P = np.asarray(P, dtype=np.float64)
        n = int(P.shape[0])
        if n < 2:
            return 1.0
        stride = max(1, n // int(max_queries))
        d, _ = cKDTree(P).query(P[::stride], k=2)
        nn = d[:, 1]
        nn = nn[np.isfinite(nn)]
        if nn.size == 0:
            return 1.0
        return float(np.median(nn))

    def _fe_node_spacing(self):
        """
        Estimate the FE node spacing as the median nearest-neighbour distance
        of the reference nodes.

        Returns:
            float: Median nearest-neighbour distance, or 1.0 for fewer than two nodes.
        """
        if self._Xref is None:
            raise ValueError("FE nodal field is not initialized. Call import_fe_nodal_field(...) first.")
        P = self._Xref
        if (cp is not None) and isinstance(P, cp.ndarray):
            P = cp.asnumpy(P)
        return self._median_nn_spacing(P)

    def _fe_modification_params(self, k):
        """
        Parameters that identify an FE nodal field application for the sample
        modification record.

        Args:
            k (int): Neighbour count used by the MLS fit.

        Returns:
            dict: JSON-serialisable parameters.
        """
        return {
            "mesh_file": getattr(self, "_fe_mesh_file", None),
            "field_file": getattr(self, "_fe_field_file", None),
            "n_nodes": int(self._Xref.shape[0]),
            "k": int(k),
            "scale": float(getattr(self, "_fe_position_scale", 1.0)),
        }

    def _finalize_sample_aabb(self, sample, gmin, gmax):
        """
        Set the sample box to the given atom AABB and write the metadata file.

        Args:
            sample (object): Sample whose chunk files were rewritten.
            gmin (ndarray): Global minimum corner, shape (3,).
            gmax (ndarray): Global maximum corner, shape (3,).
        """
        if np.all(np.isfinite(gmin)) and np.all(np.isfinite(gmax)):
            new_dims = (np.asarray(gmax) - np.asarray(gmin)).astype(np.float32)
            new_offs = ((np.asarray(gmin) + np.asarray(gmax)) * 0.5).astype(np.float32)
            sample._dimensions = new_dims
            sample._offset = new_offs
            sample._matrix = np.diag(new_dims.astype(np.float32))
            sample._corners = (sample.get_unit_corners() @ sample._matrix) - (new_dims * 0.5) + new_offs
        if hasattr(sample, "write_sample_metadata"):
            sample.write_sample_metadata()

    def apply_fe_nodal_field(self, sample, use_gpu=True, outside_factor=2.0, force=False, reg=1e-6):
        """
        Apply FE nodal mapping to atom positions using an MLS quadratic fit.

        Uses a moving-least-squares (MLS) quadratic fit of the nodal
        displacement field U = Xcurr - Xref. The MLS fit is performed in query-
        centered coordinates using the 10-term quadratic basis:
            [1, x, y, z, x2, y2, z2, xy, xz, yz].
        Evaluating at the query origin (the atom) returns the constant coefficient
        of the fit, which is the predicted displacement at that atom.

        Strategy (high level):
        1) Build a cell list over FE nodes (Xref) to cull candidates quickly
           (GPU) or a k-d tree over all nodes (CPU).
        2) For each atom chunk (and mini-batches inside the chunk):
            - kNN on the nodes.
            - MLS quadratic fit per atom using weighted normal equations:
                (P^T W P) a = P^T W U_neighbors,  with W from inverse-distance weights.
              Rows whose normal matrix is not positive definite fall back to an
              inverse-distance average of the neighbour displacements.
            - Atoms whose nearest node is farther than `outside_factor` times
              the median node spacing are outside the mesh: they are left
              undisplaced and counted.
            - Add the predicted displacement to the atom positions.
        3) Write the deformed chunk, update the sample AABB, write the sample
           metadata and record the operation so it is not applied twice.

        Chunk positions are read raw (without thermal displacement).

        Args:
            sample (object): Provides chunked IO for positions and cell-list helpers.
            use_gpu (bool): If True and CuPy is available, prefer GPU path.
            outside_factor (float or None): Multiple of the median node spacing
                beyond which an atom counts as outside the mesh. None disables
                the test.
            force (bool): Apply even if this field was already applied to the
                sample.
            reg (float): Tikhonov regularization scale, A += reg * sum(w) * I.

        Notes:
            The MLS fit uses k = 48 nearest FE nodes per atom.

        Returns:
            None

        Raises:
            ValueError: If FE nodal field or sample metadata is not initialized.
            RuntimeError: If the sample is in streaming mode, or the field was
                already applied and `force` is False.
        """
        from scipy.spatial import cKDTree

        # Validate inputs
        if self._Xref is None or self._Xcurr is None:
            raise ValueError("FE nodal field is not initialized. Call import_fe_nodal_field(...) first.")
        if sample is None or sample.chunk_total is None:
            raise ValueError("Sample is not initialized. Ensure sample metadata is loaded.")
        if getattr(sample, "_streaming_mode", False):
            raise RuntimeError("apply_fe_nodal_field: the sample is in streaming mode; chunks are "
                               "regenerated on demand and stored positions are never read, so a "
                               "deformation cannot be applied.")

        # Parameters
        k = 48
        power = 2.0
        eps = 1e-12
        reg = float(reg)
        cell_pad_cells = 2

        gpu_ok = (use_gpu and (cp is not None))
        dtype = np.dtype(self._Xref.dtype if hasattr(self._Xref, "dtype") else np.float32)

        n_nodes = int(self._Xref.shape[0])
        if k > n_nodes:
            k = int(max(1, n_nodes))

        params = self._fe_modification_params(k)
        if (not force) and hasattr(sample, "has_modification") and sample.has_modification("fe_nodal_field", params):
            raise RuntimeError("apply_fe_nodal_field: this FE field was already applied to the sample "
                               "(see sample_metadata.json); pass force=True to apply it again.")

        # Outside-mesh threshold from the node spacing
        spacing = self._fe_node_spacing()
        r_out2 = None if outside_factor is None else float(outside_factor * spacing) ** 2

        counts = {"outside": 0, "fallback": 0}
        counts_lock = threading.Lock()

        def _guard_batch(xp_, Uadd, d2):
            # Zero rows with a non-finite prediction or with the nearest node
            # beyond the outside threshold. Returns (Uadd, n_outside).
            bad = ~xp_.isfinite(Uadd).all(axis=1)
            n_out = 0
            if r_out2 is not None:
                outside = d2[:, 0] > d2.dtype.type(r_out2)
                n_out = int(xp_.count_nonzero(outside))
                bad = bad | outside
            if bool(xp_.any(bad)):
                Uadd = Uadd.copy()
                Uadd[bad, :] = 0
            return Uadd, n_out

        # Global AABB on host
        gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        # GPU path (Multi-GPU + Multi-Stream)
        if gpu_ok:
            import queue
            from concurrent.futures import as_completed

            with cp.cuda.Device(0):
                Xr = cp.asarray(self._Xref, dtype=dtype)
                Xc = cp.asarray(self._Xcurr, dtype=dtype)
                U = (Xc - Xr).astype(dtype, copy=False)
                P = Xr

            n_gpus = cp.cuda.runtime.getDeviceCount()
            streams_per_gpu = 8
            total_workers = n_gpus * streams_per_gpu

            # Create streams for each GPU
            gpu_streams = []
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    gpu_streams.append([
                        cp.cuda.Stream(non_blocking=True) for _ in range(streams_per_gpu)
                    ])

            aabb_lock = threading.Lock()
            abort = threading.Event()

            # Build a cell list on nodes for culling (on GPU 0). Cells of three
            # node spacings with a two-cell halo cover the k = 48 stencil.
            with cp.cuda.Device(0):
                r_cut = max(1.0, 3.0 * spacing)
                P32 = P.astype(cp.float32, copy=False)
                (sortedP32, sortedIdx, cell_start, cell_end,
                 bb_min32, cell_size, nx, ny, nz) = sample.build_cell_list_gpu(P32, float(r_cut))
                bb_min = bb_min32.astype(dtype, copy=False)

                opt_kerns = self._get_fe_nodal_cuda_kernels(dtype=dtype, k=k)
                sizeof_T = 8 if dtype == np.float64 else 4
                block = 256
                max_batch_size = 65536 * 2

            # Copy node data and the cell list to every GPU
            cell_list_data = {}
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    cell_list_data[gpu_id] = {
                        'P': cp.array(P) if gpu_id != 0 else P,
                        'U': cp.array(U) if gpu_id != 0 else U,
                        'sortedIdx': cp.array(sortedIdx) if gpu_id != 0 else sortedIdx,
                        'cell_start': cp.array(cell_start) if gpu_id != 0 else cell_start,
                        'cell_end': cp.array(cell_end) if gpu_id != 0 else cell_end,
                        'bb_min': cp.array(bb_min) if gpu_id != 0 else bb_min,
                    }

            # Scratch buffers, allocated once per (gpu, stream) slot. A slot is
            # held by one worker at a time so the buffers are never shared.
            slot_buffers = {}
            slots = queue.Queue()
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    n_nodes_gpu = int(cell_list_data[gpu_id]['P'].shape[0])
                    for s in range(streams_per_gpu):
                        slot_buffers[(gpu_id, s)] = {
                            "idx": cp.empty((max_batch_size, k), dtype=cp.int32),
                            "d2": cp.empty((max_batch_size, k), dtype=dtype),
                            "A": cp.empty((max_batch_size, 100), dtype=dtype),
                            "b": cp.empty((max_batch_size, 30), dtype=dtype),
                            "coef": cp.empty((max_batch_size, 30), dtype=dtype),
                            "status": cp.empty((max_batch_size,), dtype=cp.int32),
                            "cand": cp.empty((max(1, n_nodes_gpu),), dtype=cp.int32),
                            "count": cp.zeros((1,), dtype=cp.int32),
                        }
                        slots.put((gpu_id, s))

            def process_chunk_worker(chunk_i):
                """Deform one chunk on the first free (gpu, stream) slot."""
                nonlocal gmin, gmax
                if abort.is_set():
                    return
                gpu_id, stream_idx = slots.get()
                try:
                    device = cp.cuda.Device(gpu_id)
                    stream = gpu_streams[gpu_id][stream_idx]
                    buf = slot_buffers[(gpu_id, stream_idx)]
                    with device, stream:
                        P_gpu = cell_list_data[gpu_id]['P']
                        U_gpu = cell_list_data[gpu_id]['U']
                        sortedIdx_gpu = cell_list_data[gpu_id]['sortedIdx']
                        cell_start_gpu = cell_list_data[gpu_id]['cell_start']
                        cell_end_gpu = cell_list_data[gpu_id]['cell_end']
                        bb_min_gpu = cell_list_data[gpu_id]['bb_min']

                        def _knn_gpu(X, P_sub):
                            """kNN over P_sub for the rows of X (sorted ascending by d2)."""
                            M = int(X.shape[0]); N = int(P_sub.shape[0])
                            idx = buf["idx"][:M]
                            d2 = buf["d2"][:M]
                            grid = (M + block - 1) // block
                            smem = 3 * block * sizeof_T
                            opt_kerns["knn"](
                                (grid,), (block,),
                                (P_sub.ravel(), np.int32(N),
                                 X.ravel(), np.int32(M),
                                 idx.ravel(), d2.ravel()),
                                shared_mem=smem
                            )
                            return idx, d2

                        def _candidate_indices_gpu(X_chunk):
                            """Node indices of the cells covering the chunk AABB plus halo."""
                            xmn = cp.min(X_chunk, axis=0)
                            xmx = cp.max(X_chunk, axis=0)
                            halo = float(cell_size * max(0, int(cell_pad_cells)))
                            xmn = xmn - halo
                            xmx = xmx + halo
                            cs = float(cell_size)
                            nxv, nyv, nzv = int(nx), int(ny), int(nz)

                            lo = cp.floor((xmn - bb_min_gpu) / cs)
                            hi = cp.floor((xmx - bb_min_gpu) / cs)
                            cell_bounds = cp.concatenate([lo, hi]).get()
                            cx0 = max(0, min(nxv - 1, int(cell_bounds[0])))
                            cy0 = max(0, min(nyv - 1, int(cell_bounds[1])))
                            cz0 = max(0, min(nzv - 1, int(cell_bounds[2])))
                            cx1 = max(0, min(nxv - 1, int(cell_bounds[3])))
                            cy1 = max(0, min(nyv - 1, int(cell_bounds[4])))
                            cz1 = max(0, min(nzv - 1, int(cell_bounds[5])))

                            total_cells = (cx1 - cx0 + 1) * (cy1 - cy0 + 1) * (cz1 - cz0 + 1)
                            if total_cells <= 0:
                                return cp.zeros((0,), dtype=cp.int32)

                            buf["count"][0] = 0
                            grid_cells = (total_cells + block - 1) // block
                            n_cells_total = int(cell_start_gpu.shape[0])
                            opt_kerns["cell_cull"](
                                (grid_cells,), (block,),
                                (sortedIdx_gpu, cell_start_gpu, cell_end_gpu,
                                 np.int32(cx0), np.int32(cy0), np.int32(cz0),
                                 np.int32(cx1), np.int32(cy1), np.int32(cz1),
                                 np.int32(nxv), np.int32(nyv), np.int32(nzv),
                                 np.int32(n_cells_total),
                                 buf["cand"], buf["count"])
                            )
                            count = int(buf["count"].get())
                            if count == 0:
                                return cp.zeros((0,), dtype=cp.int32)
                            return buf["cand"][:count].copy()

                        def _mls_displacement(Xe, P_all, U_all, idx_glob, d2):
                            """MLS displacement per row; IDW fallback where the
                            Cholesky factorization fails. Returns (U_pred, n_fallback)."""
                            M = int(Xe.shape[0])
                            N = int(P_all.shape[0])
                            A = buf["A"][:M]
                            b = buf["b"][:M]
                            coef = buf["coef"][:M]
                            status = buf["status"][:M]

                            grid_mls = (M + block - 1) // block
                            opt_kerns["mls_fused"](
                                (grid_mls,), (block,),
                                (Xe.ravel(), P_all.ravel(), U_all.ravel(),
                                 idx_glob.ravel(), d2.ravel(),
                                 dtype.type(power), dtype.type(eps), dtype.type(reg),
                                 np.int32(M), np.int32(N),
                                 A.ravel(), b.ravel())
                            )
                            opt_kerns["cholesky_solve"](
                                (grid_mls,), (block,),
                                (A.ravel(), b.ravel(), np.int32(M), coef.ravel(), status)
                            )

                            # Constant term of the fit is the displacement at the atom
                            U_pred = coef.reshape(M, 10, 3)[:, 0, :].copy()

                            # IDW fallback for rows the solver rejected (same formula as the CPU path)
                            bad = (status == 0) | (~cp.isfinite(U_pred).all(axis=1))
                            n_fb = int(cp.count_nonzero(bad))
                            if n_fb > 0:
                                rows = cp.nonzero(bad)[0]
                                d_b = cp.sqrt(d2[rows] + dtype.type(eps))
                                w = dtype.type(1.0) / cp.power(d_b + dtype.type(eps), dtype.type(power))
                                Un_b = U_all[idx_glob[rows]]
                                wsum = cp.maximum(w.sum(axis=1, keepdims=True), dtype.type(1e-20))
                                U_pred[rows, :] = (Un_b * w[:, :, None]).sum(axis=1) / wsum

                            # Clamp nonphysical predictions to a multiple of the local envelope
                            Un = U_all[idx_glob]
                            norms = cp.linalg.norm(Un, axis=2)
                            u_cap = dtype.type(8.0) * cp.maximum(norms.max(axis=1), dtype.type(1e-9))
                            up_norm = cp.linalg.norm(U_pred, axis=1)
                            mask = up_norm > u_cap
                            if bool(cp.any(mask)):
                                scale = (u_cap[mask] / (up_norm[mask] + dtype.type(1e-20)))[:, None]
                                U_pred[mask, :] *= scale

                            return U_pred, n_fb

                        # Load the stored chunk (no thermal displacement)
                        X = sample.load_chunk_positions(chunk_i, use_gpu=True, raw=True)
                        X = cp.asarray(X).astype(dtype, copy=False)
                        if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
                            return

                        cand_idx = _candidate_indices_gpu(X)
                        P_sub = P_gpu if int(cand_idx.size) < k else P_gpu[cand_idx]

                        rows = int(X.shape[0])
                        bs = int(min(rows, max_batch_size))
                        out = cp.empty_like(X)
                        n_out_chunk = 0
                        n_fb_chunk = 0

                        for s0 in range(0, rows, bs):
                            Xe = X[s0:s0+bs]
                            idx_loc, d2 = _knn_gpu(Xe, P_sub)
                            if P_sub is not P_gpu:
                                idx_glob = cp.take(cand_idx, idx_loc)
                            else:
                                idx_glob = idx_loc
                            Uadd, n_fb = _mls_displacement(Xe, P_gpu, U_gpu, idx_glob, d2)
                            Uadd, n_out = _guard_batch(cp, Uadd, d2)
                            out[s0:s0+bs] = Xe + Uadd
                            n_out_chunk += n_out
                            n_fb_chunk += n_fb

                        stream.synchronize()
                        if abort.is_set():
                            return

                        cmin = cp.min(out, axis=0).get()
                        cmax = cp.max(out, axis=0).get()
                        with aabb_lock:
                            gmin = np.minimum(gmin, cmin)
                            gmax = np.maximum(gmax, cmax)
                        with counts_lock:
                            counts["outside"] += n_out_chunk
                            counts["fallback"] += n_fb_chunk

                        sample.write_chunk_positions(out.get(), chunk_i)
                        del X, out
                finally:
                    slots.put((gpu_id, stream_idx))

            # Run chunks on the worker pool; the first failure stops further writes.
            chunk_indices = list(range(1, int(sample.chunk_total) + 1))
            executor = ThreadPoolExecutor(max_workers=total_workers)
            futures = [executor.submit(process_chunk_worker, c) for c in chunk_indices]
            try:
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        abort.set()
                        for f in futures:
                            f.cancel()
                        raise exc
            finally:
                executor.shutdown(wait=True)

            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    for stream in gpu_streams[gpu_id]:
                        stream.synchronize()

            # Release node data and scratch buffers
            slot_buffers.clear()
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    del cell_list_data[gpu_id]
                    cp.get_default_memory_pool().free_all_blocks()

        # CPU path
        else:
            P_np = self._Xref
            U_np = self._Xcurr
            if (cp is not None) and isinstance(P_np, cp.ndarray):
                P_np = cp.asnumpy(P_np)
            if (cp is not None) and isinstance(U_np, cp.ndarray):
                U_np = cp.asnumpy(U_np)
            P_np = np.ascontiguousarray(np.asarray(P_np, dtype=dtype))
            U_np = (np.asarray(U_np, dtype=dtype) - P_np).astype(dtype, copy=False)
            tree = cKDTree(P_np)

            def _mls_batch_size_cpu(n_rows):
                # Rows per MLS mini-batch from a byte budget covering the basis,
                # gathered neighbours, normal matrices and einsum temporaries.
                bytes_per = 8 if dtype == np.float64 else 4
                budget = 256 * 1024 * 1024
                per_row = max(1, bytes_per * (k * 10 * 3 + k * 3 * 5 + 100 * 2 + 30 * 2 + k * 4))
                return max(4096, min(n_rows, budget // per_row))

            for chunk_i in range(1, int(sample.chunk_total) + 1):
                X = sample.load_chunk_positions(chunk_i, use_gpu=False, raw=True)
                X = np.asarray(X, dtype=dtype)
                if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
                    continue

                rows = int(X.shape[0])
                bs = int(_mls_batch_size_cpu(rows))
                out = np.empty_like(X)
                n_out_chunk = 0

                for s0 in range(0, rows, bs):
                    Xe = X[s0:s0+bs]
                    d, idx = tree.query(Xe, k=k)
                    if k == 1:
                        d = d[:, None]
                        idx = idx[:, None]
                    d2 = (d * d).astype(dtype, copy=False)
                    idx = idx.astype(np.int64, copy=False)
                    Uadd = self._mls_quadratic_displacement(
                        Xe, P_np, U_np, idx, d2,
                        power=power, eps=eps, reg=reg,
                        use_gpu=False, dtype=dtype
                    )
                    Uadd, n_out = _guard_batch(np, Uadd, d2)
                    out[s0:s0+bs] = Xe + Uadd
                    n_out_chunk += n_out
                    del d, idx, d2, Uadd

                cmin = out.min(axis=0).astype(np.float64, copy=False)
                cmax = out.max(axis=0).astype(np.float64, copy=False)
                gmin = np.minimum(gmin, cmin)
                gmax = np.maximum(gmax, cmax)
                counts["outside"] += n_out_chunk
                sample.write_chunk_positions(out, chunk_i)
                del X, out

        # Finalize sample metadata and record the operation
        self._finalize_sample_aabb(sample, gmin, gmax)
        if hasattr(sample, "record_modification"):
            sample.record_modification("fe_nodal_field", params)

        if counts["outside"] > 0:
            self._log("normal", "apply_fe_nodal_field: WARNING {} atoms lie outside the mesh "
                                "(nearest node > {:.3g} x node spacing {:.4g}); left undisplaced".format(
                                    counts["outside"], float(outside_factor), spacing))
        else:
            self._log("normal", "apply_fe_nodal_field: 0 atoms outside the mesh (node spacing {:.4g})".format(spacing))
        if gpu_ok:
            level = "normal" if counts["fallback"] > 0 else "verbose"
            self._log(level, "apply_fe_nodal_field: {} rows used the IDW fallback "
                             "(Cholesky factorization failed)".format(counts["fallback"]))
        return

    # FE plotting
    def plot_mesh_and_sample_edges_3d(self,
                                      sample,
                                      Xnodes=None,
                                      elev=20,
                                      azim=35,
                                      show_projections=True,
                                      projection_plane="min",
                                      sample_color="C0",
                                      mesh_color="C2",
                                      linewidth=1.8,
                                      proj_linewidth=1.2,
                                      proj_alpha=0.5,
                                      figsize=(7, 7),
                                      use_gpu=True):
        """
        Plot sample edges and an AABB of FE nodal points in 3D.

        Mirrors style of plot_field_and_sample_edges_3d.

        Args:
            sample (object): Exposes an 8x3 `corners` array.
            Xnodes (ndarray or None): Nodal points (defaults to self.Xref).
            elev (float): Matplotlib 3D elevation (degrees).
            azim (float): Matplotlib 3D azimuth (degrees).
            show_projections (bool): Draw XY, YZ, XZ rectangles on outer planes.
            projection_plane (str): "min" or "max" plane selection.
            sample_color (str): Color for sample edges.
            mesh_color (str): Color for mesh edges.
            linewidth (float): Line width for wireframes.
            proj_linewidth (float): Line width for projections.
            proj_alpha (float): Alpha for projections.
            figsize (tuple): Figure size.
            use_gpu (bool): If True and CuPy available, accept CuPy arrays.

        Returns:
            tuple: (fig, ax)

        Raises:
            ValueError: If Xnodes shape is invalid.
        """
        import matplotlib.pyplot as plt

        if Xnodes is None:
            Xnodes = self.Xref

        if use_gpu and (cp is not None) and isinstance(Xnodes, cp.ndarray):
            P = cp.asnumpy(Xnodes)
        else:
            P = np.asarray(Xnodes)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("Xnodes must have shape (N, 3)")

        C_sample = np.asarray(sample.corners, dtype=np.float64)
        mn = P.min(axis=0)
        mx = P.max(axis=0)
        C_mesh = np.array([
            [mn[0], mn[1], mn[2]],
            [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mx[1], mx[2]],
        ], dtype=np.float64)

        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass

        def _draw_edges(ax, corners, color, lw):
            for i, j in edges:
                ax.plot([corners[i,0], corners[j,0]],
                        [corners[i,1], corners[j,1]],
                        [corners[i,2], corners[j,2]],
                        color=color, linewidth=lw)

        _draw_edges(ax, C_sample, sample_color, linewidth)
        _draw_edges(ax, C_mesh, mesh_color, linewidth)

        all_pts = np.vstack([C_sample, C_mesh])
        mn_all = all_pts.min(axis=0)
        mx_all = all_pts.max(axis=0)
        span = mx_all - mn_all
        pad = 0.03 * (np.linalg.norm(span) if np.all(span > 0) else 1.0)
        ax.set_xlim(mn_all[0] - pad, mx_all[0] + pad)
        ax.set_ylim(mn_all[1] - pad, mx_all[1] + pad)
        ax.set_zlim(mn_all[2] - pad, mx_all[2] + pad)
        try:
            ax.set_box_aspect((span[0] + 1e-12, span[1] + 1e-12, span[2] + 1e-12))
        except Exception:
            pass

        if show_projections:
            sel = projection_plane.strip().lower() if isinstance(projection_plane, str) else "min"
            which = (mn_all[0], mn_all[1], mn_all[2]) if sel == "min" else (mx_all[0], mx_all[1], mx_all[2])
            def _rect(ax, corners, plane_vals, color):
                xmin, ymin, zmin = corners.min(axis=0)
                xmax, ymax, zmax = corners.max(axis=0)
                x_plane, y_plane, z_plane = plane_vals
                # XY at z_plane
                pts = np.array([[xmin, ymin, z_plane],
                                [xmax, ymin, z_plane],
                                [xmax, ymax, z_plane],
                                [xmin, ymax, z_plane]])
                for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                    ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]], [pts[a,2], pts[b,2]],
                            color=color, linestyle="--", linewidth=proj_linewidth, alpha=proj_alpha)
                # YZ at x_plane
                pts = np.array([[x_plane, ymin, zmin],
                                [x_plane, ymax, zmin],
                                [x_plane, ymax, zmax],
                                [x_plane, ymin, zmax]])
                for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                    ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]], [pts[a,2], pts[b,2]],
                            color=color, linestyle="--", linewidth=proj_linewidth, alpha=proj_alpha)
                # XZ at y_plane
                pts = np.array([[xmin, y_plane, zmin],
                                [xmax, y_plane, zmin],
                                [xmax, y_plane, zmax],
                                [xmin, y_plane, zmax]])
                for a,b in [(0,1),(1,2),(2,3),(3,0)]:
                    ax.plot([pts[a,0], pts[b,0]], [pts[a,1], pts[b,1]], [pts[a,2], pts[b,2]],
                            color=color, linestyle="--", linewidth=proj_linewidth, alpha=proj_alpha)
            _rect(ax, C_sample, which, sample_color)
            _rect(ax, C_mesh, which, mesh_color)

        import matplotlib.pyplot as plt
        plt.tight_layout()
        return fig, ax

    # Properties
    @property
    def Xref(self):
        """
        Reference nodal coordinates.

        Returns:
            ndarray or None: If not initialized, prints a message and returns None.
        """
        if self._Xref is None:
            print("self._Xref has not been initialized yet")
        return self._Xref

    @property
    def Xcurr(self):
        """
        Current nodal coordinates.

        Returns:
            ndarray or None: If not initialized, prints a message and returns None.
        """
        if self._Xcurr is None:
            print("self._Xcurr has not been initialized yet")
        return self._Xcurr

    @property
    def elem_nodes(self):
        """
        Element connectivity, 0-based node indices.

        Returns:
            ndarray or None: If not initialized, prints a message and returns None.
        """
        if self._elem_nodes is None:
            print("self._elem_nodes has not been initialized yet")
        return self._elem_nodes
