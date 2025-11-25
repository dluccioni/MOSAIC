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
        """Initialize the deformation helper.

        Args:
            directory (str or None): Optional directory. If provided and it
                does not exist, it is created.
        """
        super().__init__(log_name="deformation")
        self.directory = directory
        self._Xref = None         # shape (N, 3) reference nodal coordinates
        self._Xcurr = None        # shape (N, 3) current nodal coordinates
        self._elem_nodes = None   # shape (E, k) element connectivity (0-based)
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
        """Import a deformation gradient tensor field from a text file.

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
        """Select numpy or cupy module.

        Args:
            use_gpu (bool): If True and CuPy is available, return CuPy; else NumPy.

        Returns:
            module: numpy or cupy.
        """
        return cp if (use_gpu and (cp is not None)) else np

    def _infer_dtype(self, positions, F, dtype):
        """Infer a floating dtype from inputs or fallback to float32.

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
        """Build a 3x3 rotation matrix.

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
        """Scale positions isotropically.

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
        """Rotate positions with row-vector convention: (p - origin) @ R + origin.

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
        """Translate positions by a vector.

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
        """Scale deformation gradients about identity: F' = I + alpha * (F - I).

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
        """Rotate second-order tensors with row-vector convention: F' = R.T @ F @ R.

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
        """Normalize clip bounds to (xmin, xmax, ymin, ymax, zmin, zmax).

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
        """Clip or clamp a field to a bounding box.

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

    # ---------------------- high-level orchestrator -----------------------------
    def transform_field(
        self,
        scale_factor=1.0,
        translate=None,
        rotate_axis=None,
        rotate_angle=None,
        rotate_matrix=None,
        degrees=True,
        origin=(0.0, 0.0, 0.0),
        clip_bounds=None,
        return_mask=False,
        use_gpu=True):
        """Transform a deformation field (placeholder API).

        The intended operation order is:
            1) Position scaling
            2) Rotation about origin
            3) Translation
            4) Optional F scaling about identity
            5) Rotation of F tensors

        Notes:
            This method is intentionally left unimplemented here. It documents
            the intended pipeline and parameters for a potential orchestrator.
        """
        # Intentionally left as a placeholder for pipeline orchestration.
        return None

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
        """Clip a deformation field to the sample axis-aligned bounding box.

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
        """Compile and cache CUDA RawKernels for field kNN and application.

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
        """Retrieve compiled CUDA kernels for field operations.

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

    def _ensure_fe_nodal_cuda_kernels(self, dtype=np.float32, k=48):
        """Compile and cache optimized CUDA kernels for FE nodal field operations.

        Creates highly optimized kernels targeting:
        - Bitonic sort-based kNN
        - GPU-native cell culling
        - Fused MLS basis + weighted normal equations
        - Custom batched 10x10 Cholesky solver

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

        # Bitonic sort-based kNN kernel
        src_knn_bitonic = r'''
        extern "C" __global__
        void knn_bitonic_sqdist(const %(T)s* __restrict__ P, const int N,
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

                        // Bitonic insertion: find position and shift
                        if (d2 < best_d[%(K)d-1]) {
                            best_d[%(K)d-1] = d2;
                            best_i[%(K)d-1] = id;

                            // Bitonic sort network for k elements
                            #pragma unroll
                            for (int stage = 1; stage < %(K)d; stage *= 2) {
                                #pragma unroll
                                for (int stride = stage; stride > 0; stride /= 2) {
                                    #pragma unroll
                                    for (int i = 0; i < %(K)d; i++) {
                                        int j = i ^ stride;
                                        if (j > i) {
                                            bool ascending = ((i & stage) == 0);
                                            bool swap = (best_d[i] > best_d[j]) == ascending;
                                            if (swap) {
                                                %(T)s tmp_d = best_d[i];
                                                int tmp_i = best_i[i];
                                                best_d[i] = best_d[j];
                                                best_i[i] = best_i[j];
                                                best_d[j] = tmp_d;
                                                best_i[j] = tmp_i;
                                            }
                                        }
                                    }
                                }
                            }
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

        # GPU cell culling kernel
        src_cell_cull = r'''
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
            %(T)s* __restrict__ x_out)    // (M, 10, 3)
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
                // Fallback: return zeros
                #pragma unroll
                for (int a = 0; a < 30; a++) x[a] = 0;
            }

            // Write output
            #pragma unroll
            for (int a = 0; a < 30; a++) {
                x_out[i*30 + a] = x[a];
            }
        }
        ''' % {"T": T, "SQRT": SQRT}

        # Compile all kernels
        kernels = {
            "knn_bitonic": cp.RawKernel(src_knn_bitonic, "knn_bitonic_sqdist"),
            "cell_cull": cp.RawKernel(src_cell_cull, "gather_cell_candidates"),
            "mls_fused": cp.RawKernel(src_mls_fused, "mls_fused_weighted_neq"),
            "cholesky_solve": cp.RawKernel(src_cholesky_10x10, "batched_cholesky_solve_10x10"),
            "dtype": dt,
            "k": K,
        }
        self._fe_nodal_kernels[key] = kernels

    def _get_fe_nodal_cuda_kernels(self, dtype, k):
        """Retrieve compiled optimized CUDA kernels for FE nodal operations.

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
        tile_size=None,        # kept for API compatibility; unused on CUDA path, need to update CPU
        yield_chunks=False,
        dtype=None,
        clip_to_sample=True,
        clip_margin=10.0,
        use_cell_list=True,
        cell_r_cut=None,
        cell_pad_cells=1,
    ):
        """Apply a deformation field to sample points in chunks.

        This accelerated path uses custom CUDA kernels for:
        - kNN over field nodes,
        - inverse-distance weighting of F,
        - row-vector affine application.

        A CPU fallback preserves the previous behavior (tiled kNN using NumPy).

        Args:
            field_positions (ndarray): Field node positions, shape (Nf, 3).
            field_F (ndarray): Field F tensors, shape (Nf, 9), row-major.
            sample (ndarray or iterable): Either a single (M, 3) array or an
                iterable of (Mi, 3) chunks. If an ndarray and yield_chunks=False,
                returns a single concatenated array (GPU: CuPy; CPU: NumPy).
            chunk_size (int): Chunk size used when `sample` is a single array.
            k (int): Number of neighbors for kNN.
            origin (sequence): 3-vector origin for affine application.
            use_gpu (bool): Use CUDA path if True and CuPy is available.
            power (float): IDW power for weighting distances.
            threads (int or None): Reserved; not used.
            tile_size (int or None): CPU-only tile size override.
            yield_chunks (bool): If True, return a generator of output chunks.
            dtype (numpy dtype or None): Output dtype; inferred if None.
            clip_to_sample (bool): If True, clip field to sample AABB first.
            clip_margin (float): Margin for clipping AABB.
            use_cell_list (bool): If True (GPU), build a cell list over field.
            cell_r_cut (float or None): Optional cell list cutoff radius.
            cell_pad_cells (int): Halo cells around each chunk AABB for culling.

        Returns:
            ndarray or list or generator:
                - If `sample` is an ndarray and yield_chunks=False:
                    returns a single array of deformed positions.
                - If `sample` is an ndarray and yield_chunks=True:
                    returns a generator of chunk outputs (backend-specific arrays).
                - If `sample` is an iterable of chunks:
                    returns a list of outputs (or a generator if yield_chunks=True).

        Raises:
            ValueError: On invalid shapes or parameters.

        Notes:
            The CUDA path can allocate temporary buffers per chunk. The CPU
            path uses a tiled distance computation to bound memory.
        """
        # Backend and dtype resolution.
        xp = cp if (use_gpu and (cp is not None)) else np
        if (use_gpu and cp is None):
            use_gpu = False
            xp = np
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

        # Optional one-time field clip to sample AABB.
        if clip_to_sample:
            P_all, F_all = self.clip_field_to_sample(
                P_all, F_all, sample, margin=float(clip_margin), use_gpu=use_gpu, dtype=dtype, copy=False
            )  # raises if empty

        origin = xp.asarray(origin, dtype=dtype).reshape(3)
        eps = dtype.type(1e-12)

        # CUDA path
        if use_gpu:
            kern = self._get_field_cuda_kernels(dtype=dtype, k=k)
            sizeof_T = 8 if dtype == np.float64 else 4
            block = 128

            # Local helpers for GPU kNN, weighting, and apply.
            def _knn_gpu(X, P_sub):
                M = int(X.shape[0]); N = int(P_sub.shape[0])
                idx = cp.empty((M, k), dtype=cp.int32)
                d2  = cp.empty((M, k), dtype=dtype)
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

            # Optional GPU cell list over field for candidate culling.
            field_cells = None
            if use_cell_list:
                if cell_r_cut is None:
                    bb = cp.max(P_all, axis=0) - cp.min(P_all, axis=0)
                    vol = float((bb[0] * bb[1] * bb[2]).get())
                    Nf = int(P_all.shape[0])
                    mean_spacing = (vol / max(1, Nf)) ** (1.0 / 3.0)
                    cell_r_cut = 3.0 * mean_spacing
                cell_r_cut = float(cell_r_cut)
                (sortedP, sortedIdx, cell_start, cell_end,
                 bb_min, cell_size, nx, ny, nz) = sample.build_cell_list_gpu(P_all, cell_r_cut)
                field_cells = {
                    "sortedP": sortedP.astype(dtype, copy=False),
                    "sortedIdx": sortedIdx,
                    "cell_start": cell_start,
                    "cell_end": cell_end,
                    "bb_min": bb_min.astype(dtype, copy=False),
                    "cell_size": float(cell_size),
                    "nx": int(nx), "ny": int(ny), "nz": int(nz),
                }

                def _candidate_indices_from_chunk_AABB(X_chunk):
                    # Compute integer cell ranges covering the chunk AABB plus halo.
                    xmn = cp.min(X_chunk, axis=0)
                    xmx = cp.max(X_chunk, axis=0)
                    halo = float(field_cells["cell_size"] * max(0, int(cell_pad_cells)))
                    xmn = xmn - halo
                    xmx = xmx + halo
                    bb_min = field_cells["bb_min"]
                    cs = field_cells["cell_size"]
                    nxv, nyv, nzv = field_cells["nx"], field_cells["ny"], field_cells["nz"]

                    def _clamp_int(a, lo, hi):
                        return max(lo, min(hi, int(a)))

                    cx0 = _clamp_int(cp.floor((xmn[0] - bb_min[0]) / cs).get(), 0, nxv - 1)
                    cy0 = _clamp_int(cp.floor((xmn[1] - bb_min[1]) / cs).get(), 0, nyv - 1)
                    cz0 = _clamp_int(cp.floor((xmn[2] - bb_min[2]) / cs).get(), 0, nzv - 1)
                    cx1 = _clamp_int(cp.floor((xmx[0] - bb_min[0]) / cs).get(), 0, nxv - 1)
                    cy1 = _clamp_int(cp.floor((xmx[1] - bb_min[1]) / cs).get(), 0, nyv - 1)
                    cz1 = _clamp_int(cp.floor((xmx[2] - bb_min[2]) / cs).get(), 0, nzv - 1)

                    segs = []
                    start_arr = field_cells["cell_start"]
                    end_arr = field_cells["cell_end"]
                    idx_all = field_cells["sortedIdx"]
                    for cz in range(cz0, cz1 + 1):
                        base_z = cz * (nxv * nyv)
                        for cy in range(cy0, cy1 + 1):
                            base_y = base_z + cy * nxv
                            for cx in range(cx0, cx1 + 1):
                                int_cid = base_y + cx
                                s = int(start_arr[int_cid].get())
                                e = int(end_arr[int_cid].get())
                                if e > s:
                                    segs.append(idx_all[s:e])
                    if len(segs) == 0:
                        return cp.zeros((0,), dtype=cp.int32)
                    return cp.concatenate(segs, axis=0)

            # Normalize sample input to chunks.
            def _as_iter():
                if hasattr(sample, "shape") and sample.ndim == 2 and sample.shape[1] == 3:
                    Xall = sample
                    total = Xall.shape[0]
                    for s in range(0, total, int(chunk_size)):
                        yield Xall[s:s+int(chunk_size)]
                else:
                    for blk in sample:
                        yield blk

            def _process_chunk_gpu(Xchunk):
                # Core GPU processing for one chunk.
                X = cp.asarray(Xchunk, dtype=dtype)
                if X.ndim != 2 or X.shape[1] != 3:
                    raise ValueError("Each sample chunk must have shape (?,3)")
                rows = int(X.shape[0])
                if rows == 0:
                    return cp.empty((0, 3), dtype=dtype)

                if field_cells is not None:
                    cand_idx = _candidate_indices_from_chunk_AABB(X)
                    if cand_idx.size >= k:
                        P_sub = P_all[cand_idx]
                        idx_sub, d2_sub = _knn_gpu(X, P_sub)
                        idx = cp.take(cand_idx, idx_sub)
                        d2 = d2_sub
                    else:
                        idx, d2 = _knn_gpu(X, P_all)
                else:
                    idx, d2 = _knn_gpu(X, P_all)

                F9 = _weight_F_gpu(idx, d2)
                return _apply_gpu(F9, X)

            iterator = _as_iter()
            if yield_chunks:
                def _gen():
                    for Xc in iterator:
                        yield _process_chunk_gpu(Xc)
                return _gen()

            outs = []
            for Xc in iterator:
                outs.append(_process_chunk_gpu(Xc))
            if len(outs) == 0:
                return cp.empty((0, 3), dtype=dtype)
            if isinstance(sample, cp.ndarray):
                return cp.concatenate(outs, axis=0)
            return outs

        # CPU fallback
        # Heuristic tile size for distance tiles (rows * tile controls temp memory).
        def _auto_tile(rows):
            bytes_per = 8 if dtype == np.float64 else 4
            if rows <= 0:
                return min(P_all.shape[0], 65536)
            cap_bytes = 800000000  # approx 0.8 GB
            t = max(1, int(cap_bytes / max(1, rows * bytes_per)))
            return int(min(P_all.shape[0], max(2048, t)))

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

        def _knn_tiled(P_sub, X, k, tile):
            # Compute kNN using distance tiles to bound memory usage.
            M = X.shape[0]
            best_d2 = np.full((M, k), np.inf, dtype=dtype)
            best_idx = np.full((M, k), -1, dtype=np.int32)
            row_idx = None
            N = P_sub.shape[0]
            for j in range(0, N, tile):
                Pj = P_sub[j:j+tile]
                diff = X[:, None, :] - Pj[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                part = np.argpartition(d2, kth=min(k-1, d2.shape[1]-1), axis=1)[:, :k]
                d2k = d2[np.arange(M)[:, None], part]
                idxk = part + j
                if row_idx is None:
                    row_idx = np.arange(M)[:, None]
                all_d2 = np.concatenate([best_d2, d2k], axis=1)
                all_idx = np.concatenate([best_idx, idxk], axis=1)
                part2 = np.argpartition(all_d2, kth=k-1, axis=1)[:, :k]
                best_d2 = all_d2[row_idx, part2]
                best_idx = all_idx[row_idx, part2]
            return best_idx, np.sqrt(best_d2)

        def _weighted_F_cpu(idx, dists, F_field):
            # Inverse-distance weighting of F9. Handle zero-distance rows.
            M = idx.shape[0]
            w = 1.0 / (np.power(dists, power, dtype=np.float64).astype(dtype, copy=False) + eps)
            zero = dists <= eps
            if zero.any():
                row_has_zero = zero.any(axis=1)
                w[row_has_zero, :] = 0
                zpos = zero[row_has_zero, :].argmax(axis=1)
                w[row_has_zero, :] = 0
                for r, c in enumerate(zpos):
                    w[np.where(row_has_zero)[0][r], int(c)] = 1.0
            if getattr(self, "_cffi_lib", None) is not None and f32:
                # Use CFFI micro-kernel for float32.
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

        # Normalize sample input to chunks.
        def _as_iter():
            if hasattr(sample, "shape") and sample.ndim == 2 and sample.shape[1] == 3:
                Xall = sample
                total = Xall.shape[0]
                for s in range(0, total, int(chunk_size)):
                    yield Xall[s:s+int(chunk_size)]
            else:
                for blk in sample:
                    yield blk

        # CPU processing loop.
        _ensure_cffi()
        iterator = _as_iter()
        if yield_chunks:
            def _gen():
                for Xchunk in iterator:
                    X = np.asarray(Xchunk, dtype=dtype)
                    if X.ndim != 2 or X.shape[1] != 3:
                        raise ValueError("Each sample chunk must have shape (?,3)")
                    rows = X.shape[0]
                    if rows == 0:
                        yield np.empty((0, 3), dtype=dtype)
                        continue
                    tsize = tile_size if tile_size is not None else _auto_tile(rows)
                    idx, d = _knn_tiled(np.asarray(P_all, dtype=dtype), X, k, int(tsize))
                    F9 = _weighted_F_cpu(idx.astype(np.int32), d.astype(dtype), np.asarray(F_all, dtype=dtype))
                    yield _apply_F_cpu(F9, X)
            return _gen()

        outs = []
        for Xchunk in iterator:
            X = np.asarray(Xchunk, dtype=dtype)
            if X.ndim != 2 or X.shape[1] != 3:
                raise ValueError("Each sample chunk must have shape (?,3)")
            rows = X.shape[0]
            if rows == 0:
                outs.append(np.empty((0, 3), dtype=dtype))
                continue
            tsize = tile_size if tile_size is not None else _auto_tile(rows)
            idx, d = _knn_tiled(np.asarray(P_all, dtype=dtype), X, k, int(tsize))
            F9 = _weighted_F_cpu(idx.astype(np.int32), d.astype(dtype), np.asarray(F_all, dtype=dtype))
            outs.append(_apply_F_cpu(F9, X))

        if len(outs) == 0:
            return np.empty((0, 3), dtype=dtype)
        if hasattr(sample, "shape") and isinstance(sample, np.ndarray):
            return np.concatenate(outs, axis=0)
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
        """Plot sample edges and field AABB edges in 3D.

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
        """Compile and cache CuPy RawKernels for FE tet mapping.

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
        """Retrieve compiled CUDA kernels for FE tet mapping.

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
        """Import FE nodal field (reference nodes plus displacement or current nodes).

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
            return Xref, Xcurr

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
        return Xref, Xcurr

    def import_fe_connectivity(self,
                            filepath,
                            columns=None,
                            preset=None,
                            delimiter=None,
                            header_lines=None,
                            one_based=True,
                            use_gpu=True,
                            dtype=np.int32):
        """Import FE element connectivity for 4-node elements.

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
        return elem_nodes

    # FE transforms --------------------------------------------------------------
    def zero_fe_nodal_field(self, center_mode="bbox", return_shift=False):
        """Translate the FE nodal field so that its center is at the origin.

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
                                 copy=True):
        """Transform FE nodal reference/current positions consistently.

        Steps:
          1) optional isotropic scale of both Xref and Xcur
          2) optional displacement amplitude scaling: Xcur := Xref + alpha*(Xcur - Xref)
          3) optional rigid rotation about origin (both)
          4) optional translation (both)

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

        return Xr, Xc

    # FE clipping ----------------------------------------------------------------
    def clip_fe_mesh_to_sample(self,
                            sample,
                            Xref_nodes=None,
                            elem_nodes=None,
                            margin=0.0,
                            use_gpu=True):
        """Clip FE mesh to elements intersecting the sample AABB (+margin).

        The test is performed in REFERENCE space and keeps entire elements. An
        element is kept if:
            - any of its nodes lie inside the sample AABB (+margin), or
            - the element AABB intersects the sample AABB (+margin).

        Args:
            sample (object): Exposes `corners` (8x3) in same frame as Xref.
            Xref_nodes (ndarray or None): Reference nodal coords; defaults to self.Xref.
            elem_nodes (ndarray or None): Element connectivity; defaults to self.elem_nodes.
            margin (float): Non-negative expansion added to the sample AABB on all sides.
            use_gpu (bool): If True and CuPy available, do computations on GPU.

        Returns:
            tuple: (Xref_out, Xcurr_out_or_None, elem_out).

        Raises:
            ValueError: If inputs are missing/invalid or the clip removes all elements.
        """
        # Select backend.
        xp = cp if (use_gpu and (cp is not None)) else np

        # Resolve inputs and validate.
        Xref_nodes = self.Xref if Xref_nodes is None else Xref_nodes
        elem_nodes = self.elem_nodes if elem_nodes is None else elem_nodes
        if Xref_nodes is None or elem_nodes is None:
            raise ValueError("Xref_nodes and elem_nodes must be provided or initialized in the class.")

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

        # Update class properties.
        self._Xref = Xref_new
        if Xcurr_new is not None:
            self._Xcurr = Xcurr_new
        self._elem_nodes = En_new

        return self._Xref, (self._Xcurr if self._Xcurr is not None else None), self._elem_nodes
    
    # FE apply (atoms) -----------------------------------------------------------
    def _estimate_mls_batch_bytes(self, n_rows, k, dtype):
        """Heuristic helper for sizing MLS mini-batches.

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
        """Compute MLS-quadratic displacement per query from nodal data.

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
                        A_i += (reg * sum_j w_ij) * I.
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

    def apply_fe_nodal_field(self, sample, use_gpu=True):
        """Apply FE nodal mapping to atom positions using an MLS quadratic fit.

        Uses a moving-least-squares (MLS) quadratic fit of the nodal
        displacement field U = Xcurr - Xref. The MLS fit is performed in query-
        centered coordinates using the 10-term quadratic basis:
            [1, x, y, z, x2, y2, z2, xy, xz, yz].
        Evaluating at the query origin (the atom) returns the constant coefficient
        of the fit, which is the predicted displacement at that atom.

        Strategy (high level):
        1) Build a cell list over FE nodes (Xref) to cull candidates quickly.
        2) For each atom chunk (and further mini-batches inside the chunk):
            - kNN on culled nodes (GPU kernel if available).
            - MLS quadratic fit per atom using weighted normal equations:
                (P^T W P) a = P^T W U_neighbors,  with W from inverse-distance weights.
            The predicted displacement at the query is the constant term a[0].
            - Add the predicted displacement to the atom positions.
        3) Write deformed chunk and update the global AABB and sample metadata.

        Args:
            sample (object): Provides chunked IO for positions and cell-list helpers.
            use_gpu (bool): If True and CuPy is available, prefer GPU path.

        Returns:
            None

        Raises:
            ValueError: If FE nodal field or sample metadata is not initialized.
        """
        # Validate inputs
        if self._Xref is None or self._Xcurr is None:
            raise ValueError("FE nodal field is not initialized. Call import_fe_nodal_field(...) first.")
        if sample is None or sample.chunk_total is None:
            raise ValueError("Sample is not initialized. Ensure sample metadata is loaded.")

        # Parameters
        k = 48
        power = 2.0
        eps = 1e-12
        reg = 1e-6
        cell_pad_cells = 2

        gpu_ok = (use_gpu and (cp is not None))
        xp = cp if gpu_ok else np
        dtype = np.dtype(self._Xref.dtype if hasattr(self._Xref, "dtype") else np.float32)

        # Nodes and displacements
        Xr = xp.asarray(self._Xref, dtype=dtype)
        Xc = xp.asarray(self._Xcurr, dtype=dtype)
        U  = (Xc - Xr).astype(dtype, copy=False)
        P  = Xr

        if k <= 0 or k > int(P.shape[0]):
            k = int(min(max(1, P.shape[0]), 64))

        # Global AABB on host
        gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        # GPU path (Multi-GPU + Multi-Stream)
        if gpu_ok:
            # Detect available GPUs
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

            # Thread-safe AABB updates
            aabb_lock = threading.Lock()

            # Build a cell list on nodes for culling (on GPU 0)
            with cp.cuda.Device(0):
                if P.shape[0] > 0:
                    bb = xp.max(P, axis=0) - xp.min(P, axis=0)
                    vol = float((bb[0] * bb[1] * bb[2]).get())
                    mean_spacing = (vol / max(1, int(P.shape[0]))) ** (1.0 / 3.0)
                    r_cut = max(1.0, 3.0 * mean_spacing)
                else:
                    r_cut = 1.0

                P32 = P.astype(cp.float32, copy=False)
                (sortedP32, sortedIdx, cell_start, cell_end,
                bb_min32, cell_size, nx, ny, nz) = sample.build_cell_list_gpu(P32, float(r_cut))

                bb_min = bb_min32.astype(dtype, copy=False)

                # Get optimized kernels
                opt_kerns = self._get_fe_nodal_cuda_kernels(dtype=dtype, k=k)
                sizeof_T = 8 if dtype == np.float64 else 4
                block = 256
                max_batch_size = 65536*2  # Conservative for RTX 4090

            # Copy cell list data to all GPUs
            cell_list_data = {}
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    cell_list_data[gpu_id] = {
                        'P': cp.array(P) if gpu_id != 0 else P,
                        'U': cp.array(U) if gpu_id != 0 else U,
                        'sortedP32': cp.array(sortedP32) if gpu_id != 0 else sortedP32,
                        'sortedIdx': cp.array(sortedIdx) if gpu_id != 0 else sortedIdx,
                        'cell_start': cp.array(cell_start) if gpu_id != 0 else cell_start,
                        'cell_end': cp.array(cell_end) if gpu_id != 0 else cell_end,
                        'bb_min': cp.array(bb_min) if gpu_id != 0 else bb_min,
                    }

            def process_chunk_worker(chunk_i, gpu_id, stream_idx):
                """Worker function to process a single chunk on a specific GPU and stream."""
                nonlocal gmin, gmax

                # Set device and stream
                device = cp.cuda.Device(gpu_id)
                stream = gpu_streams[gpu_id][stream_idx]

                with device:
                    with stream:
                        # Get cell list data for this GPU
                        P_gpu = cell_list_data[gpu_id]['P']
                        U_gpu = cell_list_data[gpu_id]['U']
                        sortedIdx_gpu = cell_list_data[gpu_id]['sortedIdx']
                        cell_start_gpu = cell_list_data[gpu_id]['cell_start']
                        cell_end_gpu = cell_list_data[gpu_id]['cell_end']
                        bb_min_gpu = cell_list_data[gpu_id]['bb_min']

                        # Pre-allocate buffers for this stream
                        buf_idx = cp.empty((max_batch_size, k), dtype=cp.int32)
                        buf_d2 = cp.empty((max_batch_size, k), dtype=dtype)
                        buf_A = cp.empty((max_batch_size, 100), dtype=dtype)
                        buf_b = cp.empty((max_batch_size, 30), dtype=dtype)
                        buf_coef = cp.empty((max_batch_size, 30), dtype=dtype)
                        buf_cand = cp.empty((int(P_gpu.shape[0]),), dtype=cp.int32)
                        buf_count = cp.zeros((1,), dtype=cp.int32)

                        def _knn_gpu_optimized(X, P_sub):
                            """Optimized kNN using bitonic sort kernel."""
                            M = int(X.shape[0]); N = int(P_sub.shape[0])
                            idx = buf_idx[:M] if M <= max_batch_size else cp.empty((M, k), dtype=cp.int32)
                            d2 = buf_d2[:M] if M <= max_batch_size else cp.empty((M, k), dtype=dtype)
                            grid = (M + block - 1) // block
                            smem = 3 * block * sizeof_T
                            opt_kerns["knn_bitonic"](
                                (grid,), (block,),
                                (P_sub.ravel(), np.int32(N),
                                X.ravel(), np.int32(M),
                                idx.ravel(), d2.ravel()),
                                shared_mem=smem
                            )
                            return idx, d2

                        def _candidate_indices_gpu(X_chunk):
                            """GPU-native cell culling using optimized kernel."""
                            xmn = cp.min(X_chunk, axis=0)
                            xmx = cp.max(X_chunk, axis=0)
                            halo = float(cell_size * max(0, int(cell_pad_cells)))
                            xmn = xmn - halo
                            xmx = xmx + halo
                            cs = float(cell_size)
                            nxv, nyv, nzv = int(nx), int(ny), int(nz)

                            # Compute cell bounds on GPU
                            cx0_f = cp.floor((xmn[0] - bb_min_gpu[0]) / cs)
                            cy0_f = cp.floor((xmn[1] - bb_min_gpu[1]) / cs)
                            cz0_f = cp.floor((xmn[2] - bb_min_gpu[2]) / cs)
                            cx1_f = cp.floor((xmx[0] - bb_min_gpu[0]) / cs)
                            cy1_f = cp.floor((xmx[1] - bb_min_gpu[1]) / cs)
                            cz1_f = cp.floor((xmx[2] - bb_min_gpu[2]) / cs)

                            # Single GPU->CPU transfer
                            cell_bounds = cp.stack([cx0_f, cy0_f, cz0_f, cx1_f, cy1_f, cz1_f]).get()
                            cx0 = max(0, min(nxv - 1, int(cell_bounds[0])))
                            cy0 = max(0, min(nyv - 1, int(cell_bounds[1])))
                            cz0 = max(0, min(nzv - 1, int(cell_bounds[2])))
                            cx1 = max(0, min(nxv - 1, int(cell_bounds[3])))
                            cy1 = max(0, min(nyv - 1, int(cell_bounds[4])))
                            cz1 = max(0, min(nzv - 1, int(cell_bounds[5])))

                            ncx = cx1 - cx0 + 1
                            ncy = cy1 - cy0 + 1
                            ncz = cz1 - cz0 + 1
                            total_cells = ncx * ncy * ncz

                            if total_cells == 0:
                                return cp.zeros((0,), dtype=cp.int32)

                            # Reset count
                            buf_count[0] = 0

                            # Launch GPU cell culling kernel
                            grid_cells = (total_cells + block - 1) // block
                            n_cells_total = int(cell_start_gpu.shape[0])

                            opt_kerns["cell_cull"](
                                (grid_cells,), (block,),
                                (sortedIdx_gpu, cell_start_gpu, cell_end_gpu,
                                 np.int32(cx0), np.int32(cy0), np.int32(cz0),
                                 np.int32(cx1), np.int32(cy1), np.int32(cz1),
                                 np.int32(nxv), np.int32(nyv), np.int32(nzv),
                                 np.int32(n_cells_total),
                                 buf_cand, buf_count)
                            )

                            count = int(buf_count.get())
                            if count == 0:
                                return cp.zeros((0,), dtype=cp.int32)
                            return buf_cand[:count].copy()

                        def _mls_batch_size_optimized(n_rows):
                            """Optimized batch size for fused kernels."""
                            return min(n_rows, max_batch_size)

                        def _mls_displacement_optimized(Xe, P_all, U_all, idx_glob, d2):
                            """Compute MLS displacement using fused kernel + custom solver."""
                            M = int(Xe.shape[0])
                            N = int(P_all.shape[0])

                            # Use pre-allocated buffers
                            A = buf_A[:M] if M <= max_batch_size else cp.empty((M, 100), dtype=dtype)
                            b = buf_b[:M] if M <= max_batch_size else cp.empty((M, 30), dtype=dtype)
                            coef = buf_coef[:M] if M <= max_batch_size else cp.empty((M, 30), dtype=dtype)

                            # Fused MLS basis + weighted normal equations kernel
                            grid_mls = (M + block - 1) // block
                            opt_kerns["mls_fused"](
                                (grid_mls,), (block,),
                                (Xe.ravel(), P_all.ravel(), U_all.ravel(),
                                 idx_glob.ravel(), d2.ravel(),
                                 dtype.type(power), dtype.type(eps), dtype.type(reg),
                                 np.int32(M), np.int32(N),
                                 A.ravel(), b.ravel())
                            )

                            # Custom batched 10x10 Cholesky solver
                            opt_kerns["cholesky_solve"](
                                (grid_mls,), (block,),
                                (A.ravel(), b.ravel(), np.int32(M), coef.ravel())
                            )

                            # Extract constant term (first row of coefficient matrix)
                            U_pred = coef.reshape(M, 10, 3)[:, 0, :]

                            # Clamp nonphysical predictions
                            Un = U_all[idx_glob]
                            norms = cp.linalg.norm(Un, axis=2)
                            u_cap = dtype.type(8.0) * cp.maximum(norms.max(axis=1), dtype.type(1e-9))
                            up_norm = cp.linalg.norm(U_pred, axis=1)
                            mask = up_norm > u_cap
                            if bool(cp.any(mask)):
                                scale = (u_cap[mask] / (up_norm[mask] + dtype.type(1e-20)))[:, None]
                                U_pred = U_pred.copy()
                                U_pred[mask, :] *= scale

                            return U_pred

                        # Load chunk on this GPU
                        X = sample.load_chunk_positions(chunk_i, use_gpu=True).astype(dtype, copy=False)
                        X = cp.asarray(X)

                        if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
                            sample.write_chunk_positions(X.get(), chunk_i)
                            return

                        # GPU-accelerated candidate culling
                        cand_idx = _candidate_indices_gpu(X)
                        P_sub = P_gpu if cand_idx.size < k else P_gpu[cand_idx]

                        rows = int(X.shape[0])
                        bs = int(_mls_batch_size_optimized(rows))
                        out = cp.empty_like(X)

                        for s0 in range(0, rows, bs):
                            Xe = X[s0:s0+bs]
                            idx_loc, d2 = _knn_gpu_optimized(Xe, P_sub)

                            # Map local idx to global if needed
                            if P_sub is not P_gpu:
                                idx_glob = cp.take(cand_idx, idx_loc)
                            else:
                                idx_glob = idx_loc

                            # Optimized MLS computation
                            Uadd = _mls_displacement_optimized(Xe, P_gpu, U_gpu, idx_glob, d2)

                            # Nonfinite guard
                            bad = ~cp.isfinite(Uadd).all(axis=1)
                            if bool(cp.any(bad)):
                                Uadd = Uadd.copy()
                                Uadd[bad, :] = 0
                            out[s0:s0+bs] = Xe + Uadd

                        # Synchronize stream before AABB update and write
                        stream.synchronize()

                        # Update global AABB (thread-safe)
                        cmin = cp.min(out, axis=0).get()
                        cmax = cp.max(out, axis=0).get()
                        with aabb_lock:
                            gmin = np.minimum(gmin, cmin)
                            gmax = np.maximum(gmax, cmax)

                        # Save positions to disk
                        sample.write_chunk_positions(out.get(), chunk_i)

                        # Clean up chunk-specific buffers
                        del X, out, buf_idx, buf_d2, buf_A, buf_b, buf_coef, buf_cand, buf_count

            # Distribute chunks across GPUs and streams using ThreadPoolExecutor
            chunk_indices = list(range(1, int(sample.chunk_total) + 1))

            with ThreadPoolExecutor(max_workers=total_workers) as executor:
                futures = []
                for idx, chunk_i in enumerate(chunk_indices):
                    # Round-robin distribution across GPUs and streams
                    worker_id = idx % total_workers
                    gpu_id = worker_id // streams_per_gpu
                    stream_idx = worker_id % streams_per_gpu

                    future = executor.submit(process_chunk_worker, chunk_i, gpu_id, stream_idx)
                    futures.append(future)

                # Wait for all chunks to complete
                wait(futures, return_when=ALL_COMPLETED)

                # Check for exceptions
                for future in futures:
                    if future.exception() is not None:
                        raise future.exception()

            # Synchronize all streams before cleanup
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    for stream in gpu_streams[gpu_id]:
                        stream.synchronize()

            # Clean up cell list data
            for gpu_id in range(n_gpus):
                with cp.cuda.Device(gpu_id):
                    del cell_list_data[gpu_id]
                    cp.get_default_memory_pool().free_all_blocks()

            # Finalize metadata
            if np.all(np.isfinite(gmin)) and np.all(np.isfinite(gmax)):
                new_dims = (gmax - gmin).astype(np.float32)
                new_offs = ((gmin + gmax) * 0.5).astype(np.float32)
                sample._dimensions = new_dims
                sample._offset = new_offs
                sample._matrix = np.diag(sample._dimensions.astype(np.float32))
                sample._corners = (sample.get_unit_corners() @ sample._matrix) - (sample._dimensions * 0.5) + sample._offset
            return

        # CPU path
        P_np = np.asarray(P, dtype=dtype)
        U_np = np.asarray(U, dtype=dtype)

        def _knn_tiled(P_sub, X, k, cap_bytes=800_000_000):
            M = X.shape[0]
            best_d2 = np.full((M, k), np.inf, dtype=dtype)
            best_idx = np.full((M, k), -1, dtype=np.int32)
            row_idx = np.arange(M)[:, None]
            N = P_sub.shape[0]
            bytes_per = 8 if dtype == np.float64 else 4
            tile = int(max(2048, min(N, cap_bytes // max(1, (M * bytes_per)))))
            for j0 in range(0, N, tile):
                Pj = P_sub[j0:j0+tile]
                diff = X[:, None, :] - Pj[None, :, :]
                d2 = np.sum(diff * diff, axis=2)
                part = np.argpartition(d2, kth=min(k-1, d2.shape[1]-1), axis=1)[:, :k]
                d2k = d2[row_idx, part]
                idxk = part + j0
                all_d2 = np.concatenate([best_d2, d2k], axis=1)
                all_idx = np.concatenate([best_idx, idxk], axis=1)
                sel = np.argpartition(all_d2, kth=k-1, axis=1)[:, :k]
                best_d2 = all_d2[row_idx, sel]
                best_idx = all_idx[row_idx, sel]
            return best_idx, best_d2

        def _mls_batch_size_cpu(n_rows, dtype, k):
            bytes_per = 8 if dtype == np.float64 else 4
            budget = 640 * 1024 * 1024
            per_row = max(1, bytes_per * (k * 10 + 10 * 10 + k * 3 + k))
            return max(4096, min(n_rows, budget // per_row))

        for chunk_i in range(1, int(sample.chunk_total) + 1):
            X = sample.load_chunk_positions(chunk_i, use_gpu=False).astype(dtype, copy=False)
            if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] == 0:
                sample.write_chunk_positions(X, chunk_i)
                continue

            # AABB-based node culling similar to GPU path
            mn = X.min(axis=0); mx = X.max(axis=0)
            if P_np.shape[0] > 0:
                bb = P_np.max(axis=0) - P_np.min(axis=0)
                vol = float(bb[0] * bb[1] * bb[2])
                mean_spacing = (vol / max(1, int(P_np.shape[0]))) ** (1.0 / 3.0)
                halo = 3.0 * mean_spacing
            else:
                halo = 1.0
            mn_h = mn - halo; mx_h = mx + halo
            mask = ((P_np[:, 0] >= mn_h[0]) & (P_np[:, 0] <= mx_h[0]) &
                    (P_np[:, 1] >= mn_h[1]) & (P_np[:, 1] <= mx_h[1]) &
                    (P_np[:, 2] >= mn_h[2]) & (P_np[:, 2] <= mx_h[2]))
            P_sub = P_np[mask] if int(np.count_nonzero(mask)) >= k else P_np

            rows = int(X.shape[0])
            bs = int(_mls_batch_size_cpu(rows, dtype, k))
            out = np.empty_like(X)

            for s0 in range(0, rows, bs):
                Xe = X[s0:s0+bs]
                idx_local, d2 = _knn_tiled(P_sub, Xe, int(min(k, max(1, P_sub.shape[0]))))
                if P_sub is not P_np:
                    global_ids = np.flatnonzero(mask)
                    idx_glob = global_ids[idx_local]
                else:
                    idx_glob = idx_local
                d2 = d2.astype(dtype, copy=False)
                Uadd = self._mls_quadratic_displacement(
                    Xe, P_np, U_np, idx_glob, d2,
                    power=power, eps=eps, reg=reg,
                    use_gpu=False, dtype=dtype
                )
                # Nonfinite guard
                bad = ~np.isfinite(Uadd).all(axis=1)
                if bad.any():
                    Uadd = Uadd.copy()
                    Uadd[bad, :] = 0
                out[s0:s0+bs] = Xe + Uadd

            # Update global AABB and write
            cmin = out.min(axis=0).astype(np.float64, copy=False)
            cmax = out.max(axis=0).astype(np.float64, copy=False)
            gmin = np.minimum(gmin, cmin)
            gmax = np.maximum(gmax, cmax)
            sample.write_chunk_positions(out, chunk_i)

        # Finalize metadata
        if np.all(np.isfinite(gmin)) and np.all(np.isfinite(gmax)):
            new_dims = (gmax - gmin).astype(np.float32)
            new_offs = ((gmin + gmax) * 0.5).astype(np.float32)
            sample._dimensions = new_dims
            sample._offset = new_offs
            sample._matrix = np.diag(sample._dimensions.astype(np.float32))
            sample._corners = (sample.get_unit_corners() @ sample._matrix) - (sample._dimensions * 0.5) + sample._offset  # :contentReference[oaicite:6]{index=6}
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
        """Plot sample edges and an AABB of FE nodal points in 3D.

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
        """Reference nodal coordinates.

        Returns:
            ndarray or None: If not initialized, prints a message and returns None.
        """
        if self._Xref is None:
            print("self._Xref has not been initialized yet")
        return self._Xref

    @property
    def Xcurr(self):
        """Current nodal coordinates.

        Returns:
            ndarray or None: If not initialized, prints a message and returns None.
        """
        if self._Xcurr is None:
            print("self._Xcurr has not been initialized yet")
        return self._Xcurr

    @property
    def elem_nodes(self):
        """Element connectivity, 0-based node indices.

        Returns:
            ndarray or None: If not initialized, prints a message and returns None.
        """
        if self._elem_nodes is None:
            print("self._elem_nodes has not been initialized yet")
        return self._elem_nodes
