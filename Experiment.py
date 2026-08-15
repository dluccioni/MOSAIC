# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class experiment:
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self,directory=os.getcwd()):
        """
        Initialize an experiment instance with a specified working directory.

        Args:
            directory (str): Path to the working directory for the experiment.
                Defaults to the current working directory. If the directory
                does not exist, it will be created.
        """
        self.directory = directory
        if not os.path.isdir(self.directory):
            os.makedirs(self.directory)

    ## Main Functions
    def scan_nD(
        self,
        sample,
        beam,
        detector,
        stage,
        ranges,
        stepsizes,
        motors,
        degrees=True,
        scan_mode="absolute",
        optics=None,
        couplings=None,
        per_step_outputs=("Intensity",),
        plot_in_angle_space=False,
        show_plots=True,
        save_dir=None,
        adi_kwargs=None,
        prop_kwargs=None,
    ):
        """
        Perform a general n-dimensional scan over stage motors and/or detector axes.

        Executes a scan with n <= 3 dimensions, moving specified motors through
        their ranges while collecting intensity data at each step. In both absolute
        and relative modes, the per-step delta is computed as (target - last_applied)
        in the same user units used to build axis_vals, keeping the real motor
        positions identical to the labels shown in plots.

        Args:
            sample (Sample): The sample object to interact with.
            beam (Beam): The X-ray beam object.
            detector (Detector): The detector object for collecting data.
            stage (Stage): The stage object controlling sample position.
            ranges (list): List of (start, stop) tuples for each scan axis.
            stepsizes (list): Step size for each scan axis.
            motors (list): Names of motors/axes to scan (e.g., ["theta", "two_theta"]).
            degrees (bool): If True, interpret angles in degrees. Defaults to True.
            scan_mode (str): Either "absolute" or "relative". Defaults to "absolute".
            optics (Optics, optional): Optics object for wavefield propagation.
            couplings (dict, optional): Motor coupling definitions mapping source
                motor to list of (target, ratio) tuples.
            per_step_outputs (tuple): Output types to plot at each step.
                Defaults to ("Intensity",).
            plot_in_angle_space (bool): If True, plot in angle space. Defaults to False.
            show_plots (bool): If True, display plots interactively. Defaults to True.
            save_dir (str, optional): Directory to save output images.
            adi_kwargs (dict, optional): Additional kwargs for atomic_direct_interaction.
            prop_kwargs (dict, optional): Additional kwargs for wavefield_propagation.

        Returns:
            dict: A dictionary containing:
                - axes (list): List of axis value arrays.
                - motor_names (list): Names of scanned motors.
                - sum_intensity (ndarray): Summed intensity at each scan position.
                - step_count (int): Total number of scan steps.

        Raises:
            ValueError: If number of axes is not 1, 2, or 3, or if ranges/stepsizes/motors
                have mismatched lengths, or if step size is zero or wrong sign.
        """
        import matplotlib.pyplot as plt

        # ------------------------ helpers ------------------------
        def _canon_det_name(name):
            n = str(name).strip().lower()
            if n in ("2theta", "tth"):
                return "two_theta"
            return n

        def _is_detector_axis(name):
            return _canon_det_name(name) in ("two_theta", "eta", "distance")

        def _parse_ratio(r):
            if isinstance(r, (int, float)):
                return float(r)
            s = str(r)
            if ":" in s:
                a, b = s.split(":")
                a = float(a.strip())
                b = float(b.strip())
                if a == 0.0:
                    raise ValueError("Invalid coupling ratio a:b with a=0.")
                return b / a
            return float(s)

        def _current_value(motor_name):
            # Return current value of the named axis in the user unit (deg vs rad for rotations)
            if _is_detector_axis(motor_name):
                cname = _canon_det_name(motor_name)
                if cname == "two_theta":
                    val = detector.two_theta  # radians
                    return np.degrees(val) if degrees else float(val)
                elif cname == "eta":
                    val = detector.eta  # radians
                    return np.degrees(val) if degrees else float(val)
                elif cname == "distance":
                    return float(detector.distance)
                else:
                    raise ValueError("Unsupported detector axis '{}'".format(motor_name))
            else:
                # stage motor: read by name
                if not hasattr(_current_value, "_stage_name_to_index"):
                    try:
                        names = np.array(stage._motor_name).astype(str)
                    except Exception:
                        raise RuntimeError("Cannot access stage motor names. Ensure stage.create_stage(...) was called.")
                    _current_value._stage_name_to_index = {str(n): i for i, n in enumerate(names)}
                    _current_value._stage_type = np.array(stage._motor_type)
                idx = _current_value._stage_name_to_index.get(motor_name)
                if idx is None:
                    raise ValueError("Unknown stage motor '{}'".format(motor_name))
                val = float(stage.motor_value[idx])
                if _current_value._stage_type[idx] == "R" and degrees:
                    val = np.degrees(val)
                return val

        def _apply_stage_relative(name, delta):
            stage.set_single_motor_value_relative(name, float(delta), degrees=degrees)

        def _apply_detector_relative(name, delta):
            cname = _canon_det_name(name)
            if cname == "two_theta":
                detector.position_detector_relative(distance=0.0, two_theta=float(delta), eta=0.0, degrees=degrees)
            elif cname == "eta":
                detector.position_detector_relative(distance=0.0, two_theta=0.0, eta=float(delta), degrees=degrees)
            elif cname == "distance":
                detector.position_detector_relative(distance=float(delta), two_theta=0.0, eta=0.0, degrees=degrees)
            else:
                raise ValueError("Unsupported detector axis '{}'".format(name))

        def _move_primary_relative(name, delta):
            if _is_detector_axis(name):
                _apply_detector_relative(name, delta)
            else:
                _apply_stage_relative(name, delta)

        # Build fast lookup for "primary-takes-precedence" on coupling targets
        primary_set = { _canon_det_name(m) if _is_detector_axis(m) else str(m) for m in motors }

        # Coupling map: normalize names and ratios once
        coup_map = {}
        if couplings:
            for src, lst in couplings.items():
                src_key = _canon_det_name(src) if _is_detector_axis(src) else str(src)
                out = []
                for (tgt, ratio) in lst:
                    tgt_key = _canon_det_name(tgt) if _is_detector_axis(tgt) else str(tgt)
                    out.append((tgt_key, _parse_ratio(ratio)))
                coup_map[src_key] = out

        # ------------------------ prepare axes ------------------------
        n = int(len(motors))
        if n < 1 or n > 3:
            raise ValueError("scan_nD supports 1, 2, or 3 axes. Got {}.".format(n))
        if not (len(stepsizes) == len(ranges) == len(motors)):
            raise ValueError("ranges, stepsizes, and motors must have same length.")

        # Build axis arrays (absolute values in user units for labeling)
        axis_vals = []
        for i in range(n):
            start, stop = float(ranges[i][0]), float(ranges[i][1])
            step = float(stepsizes[i])
            if step == 0.0:
                raise ValueError("Step size for axis {} is 0.".format(motors[i]))
            if scan_mode.lower() == "relative":
                cur = _current_value(motors[i])
                a0, a1 = cur + start, cur + stop
            else:
                a0, a1 = start, stop
            if (a1 - a0) * step < 0.0:
                raise ValueError("Step sign for '{}' does not move from start to stop.".format(motors[i]))
            count = int(np.floor(abs((a1 - a0) / step) + 1e-12)) + 1
            vals = a0 + step * np.arange(count, dtype=float)
            if count == 0 or (step > 0 and vals[-1] < a1 - 1e-10) or (step < 0 and vals[-1] > a1 + 1e-10):
                vals = np.append(vals, a1)
            axis_vals.append(vals)

        # Allocate summed intensity grid
        grid_shape = tuple(len(a) for a in axis_vals)
        sumI = np.zeros(grid_shape, dtype=float)

        # Track last applied value for each primary axis in user units
        last_applied = [ _current_value(m) for m in motors ]

        # Utility: format a position string from labels (what we show on plots)
        def _pos_str(idx_tuple):
            parts = []
            for j, mj in enumerate(motors):
                parts.append("{}={:.6g}".format(mj, axis_vals[j][idx_tuple[j]]))
            return ", ".join(parts)

        # Step enumeration
        total_steps = int(np.prod([len(a) for a in axis_vals], dtype=int))
        flat_index = 0

        # Ensure save_dir exists if requested
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # ------------------------ scan loops ------------------------
        for idx in np.ndindex(*grid_shape):
            # Move each primary axis to its target for this index by relative delta = target - last_applied
            for j in range(n):
                target_val = float(axis_vals[j][idx[j]])
                step_delta = target_val - float(last_applied[j])

                if step_delta != 0.0:
                    _move_primary_relative(motors[j], step_delta)

                    # Apply couplings tied to this primary motor (relative to this step)
                    src_key = _canon_det_name(motors[j]) if _is_detector_axis(motors[j]) else str(motors[j])
                    if src_key in coup_map:
                        for (tgt_key, ratio) in coup_map[src_key]:
                            if tgt_key in primary_set:
                                continue
                            coupled_delta = float(ratio) * step_delta
                            _move_primary_relative(tgt_key, coupled_delta)

                    # Update last applied for this axis
                    last_applied[j] = target_val

            # ---- physics: atomic + optics ----
            beam.atomic_direct_interaction(
                sample=sample,
                detector=detector,
                stage=stage,
                **(adi_kwargs or {})
            )
            if optics is not None:
                beam.wavefield_propagation(
                    detector=detector,
                    optics=optics,
                    **(prop_kwargs or {})
                )

            # sum intensity
            cur_sum = float(np.sum(detector.pixel_intensity))
            sumI[idx] = cur_sum
            # detector.input_pixel_values(np.flip(detector.pixel_values))

            # ---- per-step plots ----
            if per_step_outputs:
                for out_name in per_step_outputs:
                    title = "Step {}/{} : {}".format(flat_index + 1, total_steps, _pos_str(idx))
                    if plot_in_angle_space:
                        fig, ax = detector.plot_detector_angles(type=str(out_name), title=title, cmap="viridis")
                    else:
                        fig, ax = detector.plot_detector(type=str(out_name), title=title, cmap="viridis")
                    if save_dir:
                        safe_out = str(out_name).lower()
                        fig.savefig(os.path.join(
                            save_dir,
                            "step_{:05d}_{}.png".format(flat_index + 1, safe_out)
                        ), dpi=300)
                    if show_plots:
                        plt.show()
                    else:
                        plt.close(fig)

            flat_index += 1

        # ------------------------ summary plots ------------------------
        if n == 1:
            fig, ax = plt.subplots()
            ax.plot(axis_vals[0], sumI)
            ax.set_xlabel(motors[0])
            ax.set_ylabel("sum(Intensity)")
            ax.set_title("Summed intensity vs {}".format(motors[0]))
            if save_dir:
                fig.savefig(os.path.join(save_dir, "summary_1d.png"), dpi=300)
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        elif n == 2:
            X, Y = np.meshgrid(axis_vals[0], axis_vals[1], indexing="xy")
            fig, ax = plt.subplots()
            pcm = ax.pcolormesh(X, Y, sumI.T, shading="auto")
            fig.colorbar(pcm, ax=ax, label="sum(Intensity)")
            ax.set_xlabel(motors[0])
            ax.set_ylabel(motors[1])
            ax.set_title("Summed intensity vs {} and {}".format(motors[0], motors[1]))
            if save_dir:
                fig.savefig(os.path.join(save_dir, "summary_2d.png"), dpi=300)
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        else:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            A0, A1, A2 = np.meshgrid(axis_vals[0], axis_vals[1], axis_vals[2], indexing="ij")
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            sc = ax.scatter(A0.ravel(), A1.ravel(), A2.ravel(), c=sumI.ravel(), s=10)
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("sum(Intensity)")
            ax.set_xlabel(motors[0]); ax.set_ylabel(motors[1]); ax.set_zlabel(motors[2])
            ax.set_title("Summed intensity vs {}, {}, {}".format(motors[0], motors[1], motors[2]))
            if save_dir:
                fig.savefig(os.path.join(save_dir, "summary_3d.png"), dpi=300)
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        return {
            "axes": axis_vals,
            "motor_names": list(motors),
            "sum_intensity": sumI,
            "step_count": total_steps,
        }
        
    def plot_geometry_3d(self,
                        beam,
                        sample,
                        detector,
                        stage=None,
                        optics=None,
                        unit="mm",
                        beam_length=None,
                        show=True,
                        elev=20,
                        azim=-60,
                        figsize=(9, 7)):
        """
        Plot the experimental geometry in 3D.

        Draws the experimental setup including beam, sample, detector, and optics
        in a 3D visualization.

        Draw order:
            1. Beam as a light pink cuboid with the same transverse dimensions as the beam.
            2. Sample as a box defined by Sample dimensions (after stage rotation/translation).
            3. Black line from origin to detector center.
            4. Optics drawn along +x (if provided).
            5. Rectangle outlining the bounding box of the detector pixels in 3D.

        Args:
            beam (Beam): Beam object. Must provide _beam_size (angstrom) for
                transverse size.
            sample (Sample): Sample object. Must provide corners (8x3, angstrom).
            detector (Detector): Detector object. Must be positioned; uses center,
                direction, and pixel_coordinates.
            stage (Stage, optional): Stage object. If provided, its rotation and
                translation are applied to the sample.
            optics (Optics, optional): Optics object. If provided, drawn via
                optics.plot_stack_3d into the same axes.
            unit (str): Display unit for all geometry. One of "m", "cm", "mm",
                "um", "nm". Defaults to "mm".
            beam_length (float, optional): Length of the beam cuboid along +x in
                angstrom. If None, uses sample.dimensions[0] as a reasonable default.
            show (bool): If True, calls plt.show(). Defaults to True.
            elev (float): Matplotlib elevation view angle. Defaults to 20.
            azim (float): Matplotlib azimuth view angle. Defaults to -60.
            figsize (tuple): Figure size in inches. Defaults to (9, 7).

        Returns:
            tuple: A tuple (fig, ax) containing the matplotlib figure and 3D axes.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # ---------- unit handling ----------
        unit = str(unit).lower()
        # meters to unit
        m_to_unit = {"m": 1.0, "cm": 1e2, "mm": 1e3, "um": 1e6, "nm": 1e9}
        if unit not in m_to_unit:
            raise ValueError("unit must be one of: m, cm, mm, um, nm")
        # angstrom to unit
        A_to_unit = {
            "m": 1e-10,
            "cm": 1e-8,
            "mm": 1e-7,
            "um": 1e-4,
            "nm": 1e-1,
        }
        toU = lambda arrA: np.asarray(arrA, dtype=float) * A_to_unit[unit]
        m2U = lambda x_m: float(x_m) * m_to_unit[unit]

        # ---------- helpers ----------
        def _set_axes_equal(ax):
            # Make 3D axes have equal scale
            import numpy as np
            xlims = ax.get_xlim3d()
            ylims = ax.get_ylim3d()
            zlims = ax.get_zlim3d()
            xsize = abs(xlims[1] - xlims[0])
            ysize = abs(ylims[1] - ylims[0])
            zsize = abs(zlims[1] - zlims[0])
            maxsize = max(xsize, ysize, zsize)
            xmid = sum(xlims) * 0.5
            ymid = sum(ylims) * 0.5
            zmid = sum(zlims) * 0.5
            ax.set_xlim3d([xmid - maxsize/2, xmid + maxsize/2])
            ax.set_ylim3d([ymid - maxsize/2, ymid + maxsize/2])
            ax.set_zlim3d([zmid - maxsize/2, zmid + maxsize/2])

        def _box_faces_from_corners(C8):
            # Faces for cube with the same index pattern used by sample.corners
            idx = [
                [0, 1, 4, 2],  # z-min
                [3, 5, 7, 6],  # z-max
                [0, 1, 5, 3],  # y-min
                [2, 4, 7, 6],  # y-max
                [0, 2, 6, 3],  # x-min
                [1, 4, 7, 5],  # x-max
            ]
            return [[C8[i] for i in f] for f in idx]

        def _add_poly(ax, verts, fc, ec, alpha=0.25, lw=0.8):
            poly = Poly3DCollection(verts, facecolors=fc, edgecolors=ec, linewidths=lw, alpha=alpha)
            ax.add_collection3d(poly)

        # ---------- setup figure ----------
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(f"X [{unit}]")
        ax.set_ylabel(f"Y [{unit}]")
        ax.set_zlabel(f"Z [{unit}]")
        ax.set_proj_type("ortho")

        # Keep track of point cloud limits to autoscale later
        all_pts = []

        # ---------- 4) optics (draw first so later objects overlay nicely) ----------
        if optics is not None:
            # pick a cross section similar to beam size, but in meters for the optics API
            Sy, Sz = getattr(beam, "_beam_size", (1000.0, 1000.0))  # angstrom
            cross_section_m = max(float(Sy), float(Sz)) * 1e-10
            try:
                # uses optics.plot_stack_3d(unit=..., ax=ax) into the same axes
                # so all geometry shares the same unit and axes
                optics.plot_stack_3d(unit=unit, cross_section_m=cross_section_m, show=False, ax=ax)
            except Exception:
                # Non-fatal if user did not add any components yet
                pass

        # ---------- 1) beam cuboid ----------
        Sy, Sz = getattr(beam, "_beam_size", (1000.0, 1000.0))  # angstrom; rectangular cross section
        if beam_length is None:
            # fall back to sample thickness along x
            try:
                Lx = float(sample.dimensions[0])
            except Exception:
                Lx = 1000.0
        else:
            Lx = float(beam_length)
        # Build 8 corners for an axis-aligned box spanning x=[0, Lx], y,z by beam size
        x0, x1 = -detector.distance, float(sample.dimensions[0])
        y0, y1 = -0.5 * float(Sy), 0.5 * float(Sy)
        z0, z1 = -0.5 * float(Sz), 0.5 * float(Sz)
        beam_corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x0, y0, z1],
            [x1, y1, z0], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1]
        ], dtype=float)
        beam_faces = _box_faces_from_corners(toU(beam_corners))
        _add_poly(ax, beam_faces, fc="#f8bbd0", ec="#b0003a", alpha=0.35, lw=0.6)
        all_pts.append(beam_corners)

        # ---------- 2) sample box (with stage pose) ----------
        try:
            C8 = np.asarray(sample.corners, dtype=float)  # angstrom
        except Exception as e:
            raise RuntimeError("sample.corners is required to draw the sample") from e
        # Apply stage rotation and translation if provided
        if stage is not None:
            Rg = np.asarray(getattr(stage, "rotation", np.eye(3)), dtype=float)
            Tg = np.asarray(getattr(stage, "translation", np.zeros(3)), dtype=float)
            C8w = (C8 @ Rg.T) + Tg
        else:
            C8w = C8
        sample_faces = _box_faces_from_corners(toU(C8w))
        _add_poly(ax, sample_faces, fc="#90caf9", ec="#0d47a1", alpha=0.15, lw=0.8)
        all_pts.append(C8w)

        # ---------- 3) line from origin to detector center ----------
        try:
            det_center_A, det_dir = detector.get_detector_position_cartesian()
        except Exception:
            det_center_A = np.asarray(getattr(detector, "center"), dtype=float)
        p0 = np.zeros(3, dtype=float)
        p1 = np.asarray(det_center_A, dtype=float)
        ax.plot(toU([p0[0], p1[0]]), toU([p0[1], p1[1]]), toU([p0[2], p1[2]]),
                color="k", linewidth=1.2)
        all_pts.append(np.vstack([p0, p1]))

        # ---------- 5) detector pixel bounding rectangle ----------
        # Build an orthonormal in-plane basis (u, v) from detector normal
        try:
            _, nvec = detector.get_detector_position_cartesian()
            nvec = np.asarray(nvec, dtype=float)
        except Exception:
            nvec = np.asarray(getattr(detector, "direction"), dtype=float)
        n = nvec / (np.linalg.norm(nvec) + 1e-20)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, tmp); u /= (np.linalg.norm(u) + 1e-20)
        v = np.cross(n, u);   v /= (np.linalg.norm(v) + 1e-20)

        center = np.asarray(getattr(detector, "center", p1), dtype=float)
        Pc = getattr(detector, "pixel_coordinates", None)
        if Pc is not None:
            Pc = np.asarray(Pc, dtype=float)  # shape (3, N)
            Pc_rel = (Pc.T - center[None, :])  # (N, 3)
            pu = Pc_rel @ u
            pv = Pc_rel @ v
            umin, umax = float(np.min(pu)), float(np.max(pu))
            vmin, vmax = float(np.min(pv)), float(np.max(pv))
        else:
            # Fallback: rectangular detector geometry
            sh = getattr(detector, "shape")
            ps = getattr(detector, "pixel_size")
            half_u = 0.5 * float(sh[0]) * float(ps[0])
            half_v = 0.5 * float(sh[1]) * float(ps[1])
            umin, umax = -half_u, half_u
            vmin, vmax = -half_v, half_v

        rect_pts = np.array([
            center + umin * u + vmin * v,
            center + umax * u + vmin * v,
            center + umax * u + vmax * v,
            center + umin * u + vmax * v
        ], dtype=float)
        rect_verts = [toU(rect_pts)]
        _add_poly(ax, rect_verts, fc=(0, 0, 0, 0.02), ec="k", alpha=0.08, lw=1.2)
        # also draw the edges explicitly to ensure visibility
        R = rect_pts
        for seg in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            ax.plot(toU([R[seg[0], 0], R[seg[1], 0]]),
                    toU([R[seg[0], 1], R[seg[1], 1]]),
                    toU([R[seg[0], 2], R[seg[1], 2]]),
                    color="k", linewidth=1.2)
        all_pts.append(rect_pts)

        # ---------- autoscale and finish ----------
        if all_pts:
            P = toU(np.vstack(all_pts))
            xmin, ymin, zmin = np.min(P, axis=0)
            xmax, ymax, zmax = np.max(P, axis=0)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
            _set_axes_equal(ax)

        ax.set_title("Experiment geometry (unit: {})".format(unit))
        if show:
            plt.show()
        return fig, ax
