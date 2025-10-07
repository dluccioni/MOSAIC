# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import os
import gc
from Logging import logging
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class optics(logging):
    
    # -------------------------------------------------------------------------
    # Logging configuration
    # -------------------------------------------------------------------------
    __log_top__ = (
        "add_free_space",
        "add_CRL_box",
        "add_bragg_magnifier_2b",
        "add_aperture",
        "add_custom_component",
        "plot_stack_3d",
        "read_optics_metadata",
        "write_optics_metadata",
    )
    
    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=None):
        super().__init__(log_name="optics")
        self.directory = directory
        if self.directory is not None and not os.path.isdir(self.directory):
            os.makedirs(self.directory)

        # List of optical components
        self._components = []
        self._direction  = None
        self._origin     = None

    def _apply_thin_lens_box(self, field, dx, dy, lens_data, wavelength=None, use_gpu=True):
        """
        Apply a thin-lens phase and optional uniform absorption.

        Multiplies the field by exp(-i * k/(2f) * r^2). If `absorption_sigma` is
        provided, a uniform attenuation factor is applied.

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            lens_data (dict): Lens parameters:
                - 'focal_length' (float, mm)
                - 'thickness' (float, mm)
                - 'number' (int): number of identical lens elements
                - 'absorption_sigma' (float, meters) optional
            wavelength (float or None): Wavelength in meters. If None, tries
                self._wavelength.
            use_gpu (bool): If True and CuPy is available, use a GPU path.

        Returns:
            np.ndarray: Complex64 array with the lens applied (NumPy array).
        """
        if wavelength is None:
            wavelength = getattr(self, "_wavelength", None)
        if wavelength is None:
            raise ValueError("wavelength must be provided to _apply_thin_lens_box")

        k_val = 2.0 * np.pi / float(wavelength)

        # Convert mm -> m
        f  = float(lens_data['focal_length']) * 1e-3
        t  = float(lens_data['thickness']) * 1e-3
        nsigma = float(lens_data.get('absorption_sigma', np.inf))
        N_lenses = int(lens_data['number'])

        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        x_arr = np.arange(Nx, dtype=np.float32)
        y_arr = np.arange(Ny, dtype=np.float32)
        cx = (Nx - 1) / 2.0
        cy = (Ny - 1) / 2.0

        if use_gpu and (cp is not None):
            # Coordinate grids on GPU (meters from pixel indices)
            x_gpu = cp.asarray((x_arr - cx) * dx, dtype=cp.float32)
            y_gpu = cp.asarray((y_arr - cy) * dy, dtype=cp.float32)
            Xgpu = x_gpu[None, :].repeat(Ny, axis=0)
            Ygpu = y_gpu[:, None].repeat(Nx, axis=1)
            R2 = Xgpu * Xgpu + Ygpu * Ygpu

            # Thin-lens phase
            phase_lens = -0.5 * (k_val / f) * R2
            cph = cp.cos(phase_lens)
            sph = cp.sin(phase_lens)

            F_gpu = cp.asarray(field, dtype=cp.complex64)
            real_part = F_gpu.real * cph - F_gpu.imag * sph
            imag_part = F_gpu.real * sph + F_gpu.imag * cph
            out = real_part + 1j * imag_part

            # Optional uniform absorption
            if not cp.isinf(nsigma):
                out *= cp.exp(- N_lenses * t / nsigma)

            return out.get()

        # CPU path
        xx = (x_arr - cx) * dx
        yy = (y_arr - cy) * dy
        E_out = np.empty_like(field, dtype=np.complex64)

        for iy in range(Ny):
            r_y = yy[iy]
            for ix in range(Nx):
                r_x = xx[ix]
                r2 = r_x * r_x + r_y * r_y
                phase = -0.5 * (k_val / f) * r2
                cph = np.cos(phase)
                sph = np.sin(phase)
                val = field[iy, ix]
                re2 = val.real * cph - val.imag * sph
                im2 = val.real * sph + val.imag * cph
                E_out[iy, ix] = re2 + 1j * im2

        if not np.isinf(nsigma):
            E_out *= np.exp(- N_lenses * t / nsigma)

        return E_out
    
    def _apply_bragg_magnifier_2b(self, field, dx, dy, mag_data, use_gpu=True):
        """
        Apply a two-bounce 2D Bragg magnifier to a complex field by anisotropic
        scaling of the transverse coordinates with optional amplitude and phase.
        This is a geometric resampling model (no dynamical diffraction).

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters (unused, for API symmetry).
            dy (float): Pixel size along y in meters (unused, for API symmetry).
            mag_data (dict): Keys:
                - 'magnification_x' (float)
                - 'magnification_y' (float)
                - 'reflectivity' (float, intensity 0..1)
                - 'phase_shift' (float, radians)
                - 'order' (0 or 1): nearest or bilinear
                - 'pad_mode' ('zeros' or 'edge')
                - 'conserve_energy' (bool)
            use_gpu (bool): If True and CuPy is available, use GPU path.

        Returns:
            np.ndarray: Complex64 array with magnifier applied (NumPy array).
        """
        xp = None
        if use_gpu and (cp is not None):
            xp = cp

        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        Mx = float(mag_data.get('magnification_x', 1.0))
        My = float(mag_data.get('magnification_y', 1.0))
        refl = float(mag_data.get('reflectivity', 1.0))
        phi = float(mag_data.get('phase_shift', 0.0))
        order = int(mag_data.get('order', 1))
        pad_mode = str(mag_data.get('pad_mode', 'zeros')).lower()
        conserve = bool(mag_data.get('conserve_energy', True))

        if xp is not None:
            F = cp.asarray(field, dtype=cp.complex64)
            xx = cp.arange(Nx, dtype=cp.float32)
            yy = cp.arange(Ny, dtype=cp.float32)
        else:
            F = np.asarray(field, dtype=np.complex64)
            xx = np.arange(Nx, dtype=np.float32)
            yy = np.arange(Ny, dtype=np.float32)

        cx = (Nx - 1) / 2.0
        cy = (Ny - 1) / 2.0

        # Output grid
        if xp is not None:
            Xo = xx[None, :].repeat(Ny, axis=0)
            Yo = yy[:, None].repeat(Nx, axis=1)
        else:
            Xo = np.repeat(xx[None, :], Ny, axis=0)
            Yo = np.repeat(yy[:, None], Nx, axis=1)

        # Map output -> input coords (centered anisotropic scaling)
        Xi = cx + (Xo - cx) / Mx
        Yi = cy + (Yo - cy) / My

        # Interpolation
        if order == 0:
            # Nearest
            if xp is not None:
                Xi_n = cp.rint(Xi).astype(cp.int32)
                Yi_n = cp.rint(Yi).astype(cp.int32)
                valid = (Xi_n >= 0) & (Xi_n < Nx) & (Yi_n >= 0) & (Yi_n < Ny)
                out = cp.zeros((Ny, Nx), dtype=cp.complex64)
                out[valid] = F[Yi_n[valid], Xi_n[valid]]
            else:
                Xi_n = np.rint(Xi).astype(np.int32)
                Yi_n = np.rint(Yi).astype(np.int32)
                valid = (Xi_n >= 0) & (Xi_n < Nx) & (Yi_n >= 0) & (Yi_n < Ny)
                out = np.zeros((Ny, Nx), dtype=np.complex64)
                out[valid] = F[Yi_n[valid], Xi_n[valid]]
        else:
            # Bilinear
            if xp is not None:
                floor = cp.floor; clip = cp.clip
                x0 = floor(Xi).astype(cp.int32)
                y0 = floor(Yi).astype(cp.int32)
                x1 = x0 + 1
                y1 = y0 + 1

                if pad_mode == 'edge':
                    x0 = clip(x0, 0, Nx - 1); x1 = clip(x1, 0, Nx - 1)
                    y0 = clip(y0, 0, Ny - 1); y1 = clip(y1, 0, Ny - 1)
                    wx = Xi - x0; wy = Yi - y0
                    f00 = F[y0, x0]; f10 = F[y0, x1]
                    f01 = F[y1, x0]; f11 = F[y1, x1]
                    out = (f00 * (1 - wx) * (1 - wy) +
                        f10 * wx * (1 - wy) +
                        f01 * (1 - wx) * wy +
                        f11 * wx * wy).astype(cp.complex64)
                else:
                    valid = (Xi >= 0) & (Xi < Nx - 1) & (Yi >= 0) & (Yi < Ny - 1)
                    x0c = clip(x0, 0, Nx - 2)
                    y0c = clip(y0, 0, Ny - 2)
                    x1c = x0c + 1
                    y1c = y0c + 1
                    wx = Xi - x0; wy = Yi - y0
                    f00 = F[y0c, x0c]; f10 = F[y0c, x1c]
                    f01 = F[y1c, x0c]; f11 = F[y1c, x1c]
                    out = (f00 * (1 - wx) * (1 - wy) +
                        f10 * wx * (1 - wy) +
                        f01 * (1 - wx) * wy +
                        f11 * wx * wy).astype(cp.complex64)
                    out = cp.where(valid, out, 0.0 + 0.0j)
            else:
                floor = np.floor; clip = np.clip
                x0 = floor(Xi).astype(np.int32)
                y0 = floor(Yi).astype(np.int32)
                x1 = x0 + 1
                y1 = y0 + 1

                if pad_mode == 'edge':
                    x0 = clip(x0, 0, Nx - 1); x1 = clip(x1, 0, Nx - 1)
                    y0 = clip(y0, 0, Ny - 1); y1 = clip(y1, 0, Ny - 1)
                    wx = Xi - x0; wy = Yi - y0
                    f00 = F[y0, x0]; f10 = F[y0, x1]
                    f01 = F[y1, x0]; f11 = F[y1, x1]
                    out = (f00 * (1 - wx) * (1 - wy) +
                        f10 * wx * (1 - wy) +
                        f01 * (1 - wx) * wy +
                        f11 * wx * wy).astype(np.complex64)
                else:
                    valid = (Xi >= 0) & (Xi < Nx - 1) & (Yi >= 0) & (Yi < Ny - 1)
                    x0c = clip(x0, 0, Nx - 2)
                    y0c = clip(y0, 0, Ny - 2)
                    x1c = x0c + 1
                    y1c = y0c + 1
                    wx = Xi - x0; wy = Yi - y0
                    f00 = F[y0c, x0c]; f10 = F[y0c, x1c]
                    f01 = F[y1c, x0c]; f11 = F[y1c, x1c]
                    out = (f00 * (1 - wx) * (1 - wy) +
                        f10 * wx * (1 - wy) +
                        f01 * (1 - wx) * wy +
                        f11 * wx * wy).astype(np.complex64)
                    out = np.where(valid, out, 0.0 + 0.0j)

        amp = np.sqrt(max(refl, 0.0))
        if conserve:
            jac = abs(Mx * My)
            if jac > 0:
                amp *= (1.0 / np.sqrt(jac))

        if xp is not None:
            cph = cp.cos(phi, dtype=cp.float32)
            sph = cp.sin(phi, dtype=cp.float32)
            re = out.real * cph - out.imag * sph
            im = out.real * sph + out.imag * cph
            out = amp * (re + 1j * im)
            return out.get()
        else:
            cph = np.cos(phi, dtype=np.float32)
            sph = np.sin(phi, dtype=np.float32)
            re = out.real * cph - out.imag * sph
            im = out.real * sph + out.imag * cph
            out = amp * (re + 1j * im)
            return out.astype(np.complex64)
        
    def _apply_angular_filter_kspace(self, field, dx, dy, filt, wavelength, use_gpu=True):
        """
        Analyzer-like angular pupil in k-space.

        This matches beam.analyser_mode semantics:
        - rolloff='tophat'  : amplitude = 1 inside half-angle, else 0
        - rolloff='darwin'  : amplitude = 1 / (1 + (delta / halfwidth)^2)

        Notes
        -----
        - delta is the small-angle deviation (radians) in the paraxial limit:
            delta = sqrt((theta_x - cx)^2 + (theta_y - cy)^2) for circular 2D
        For 'elliptical' we use the normalized radius r^2 = (tpar/hx)^2 + (tperp/hy)^2
        and apply the same laws with r in place of delta/hw.
        - 'mode' can be '2d' (default) or '1d' (slit). In '1d', acceptance is along the
        rolled pass axis only: delta = |tpar|.
        - half_angle_x_mrad is interpreted as:
            * top-hat: acceptance half-angle (radians)
            * darwin : Darwin halfwidth (radians)
        which matches the GPU analyser's accept_angle_rad and darwin_halfwidth_rad.
        """
        if wavelength is None:
            raise ValueError("wavelength must be provided to _apply_angular_filter_kspace")

        on_gpu = bool(use_gpu and (cp is not None))
        xp = cp if on_gpu else np

        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        k0 = (2.0 * np.pi) / float(wavelength)

        # Spatial frequency axes -> small angles theta = k_perp / k0 (dimensionless)
        kx = (2.0 * np.pi) * xp.fft.fftfreq(Nx, d=float(dx)).astype(xp.float32)
        ky = (2.0 * np.pi) * xp.fft.fftfreq(Ny, d=float(dy)).astype(xp.float32)
        tx = (kx / k0)[xp.newaxis, :]            # (1, Nx)
        ty = (ky / k0)[:, xp.newaxis]            # (Ny, 1)

        # Parameters
        cx = float(filt.get('center_x_mrad', 0.0)) * 1e-3
        cy = float(filt.get('center_y_mrad', 0.0)) * 1e-3
        hx = float(filt.get('half_angle_x_mrad', 0.0)) * 1e-3
        hy = float(filt.get('half_angle_y_mrad', hx * 1e3)) * 1e-3  # default hy=hx
        mode = str(filt.get('mode', filt.get('dimension', '2d'))).lower()
        shape = str(filt.get('shape', 'circular')).lower()
        rolloff = str(filt.get('rolloff', 'tophat')).lower()
        roll_deg = float(filt.get('roll_deg', 0.0))
        phi = float(np.deg2rad(roll_deg))

        # Shift to center and rotate axes so tpar aligns with acceptance axis
        tX = tx - cx
        tY = ty - cy
        cph = float(np.cos(phi))
        sph = float(np.sin(phi))
        tpar  = cph * tX + sph * tY    # along rolled pass axis
        tperp = -sph * tX + cph * tY   # orthogonal to it

        # Build amplitude mask H consistent with analyser_mode
        eps = 1e-30
        if mode == '1d':
            # Slit: unlimited in tperp, accept along tpar
            s = tpar / max(hx, eps)  # dimensionless
            if rolloff == 'darwin':
                H = 1.0 / (1.0 + (s * s))   # amplitude
            elif rolloff == 'tophat':
                H = (xp.abs(s) <= 1.0).astype(xp.float32)
            else:
                # Keep Butterworth as a soft 1D slit (not part of analyser, but useful)
                order = max(1, int(filt.get('order', 4)))
                H = 1.0 / xp.sqrt(1.0 + xp.power(xp.abs(s), 2 * order))
        else:
            # 2D acceptance
            if shape == 'elliptical':
                # Normalized radius
                u = tpar  / max(hx, eps)
                v = tperp / max(hy, eps)
                r2 = u * u + v * v
                if rolloff == 'darwin':
                    H = 1.0 / (1.0 + r2)     # amplitude
                elif rolloff == 'tophat':
                    H = (r2 <= 1.0).astype(xp.float32)
                else:
                    order = max(1, int(filt.get('order', 4)))
                    H = 1.0 / xp.sqrt(1.0 + xp.power(r2, order))
            else:
                # Circular: delta is the isotropic small-angle deviation
                delta = xp.sqrt(tX * tX + tY * tY)
                if rolloff == 'darwin':
                    hw = max(hx, eps)
                    H = 1.0 / (1.0 + (delta / hw) * (delta / hw))   # amplitude
                elif rolloff == 'tophat':
                    H = (delta <= max(hx, eps)).astype(xp.float32)
                else:
                    order = max(1, int(filt.get('order', 4)))
                    H = 1.0 / xp.sqrt(1.0 + xp.power(delta / max(hx, eps), 2 * order))

        # Peak transmission and uniform phase (amplitude model, like the analyser)
        amp_peak = float(filt.get('transmission', 1.0))
        amp = float(np.sqrt(max(amp_peak, 0.0)))
        phase = float(filt.get('phase_shift', 0.0))
        cph0 = float(np.cos(phase))
        sph0 = float(np.sin(phase))
        phasor = (cph0 + 1j * sph0)

        # FFT -> mask -> iFFT
        E = xp.asarray(field, dtype=xp.complex64)
        F = xp.fft.fft2(E)
        F = F * H.astype(xp.complex64)
        Eo = xp.fft.ifft2(F) * amp * phasor

        out = Eo.get() if on_gpu else np.asarray(Eo)
        return out.astype(np.complex64)

    def _apply_aperture(self, field, dx, dy, aperture_data, use_gpu=True):
        """
        Apply a real-space aperture (square or circular) centered on the field.

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            aperture_data (dict): Aperture specification:
                - 'shape' or 'type': 'square' or 'circular'
                - 'width': float in millimeters
            use_gpu (bool): If True and CuPy is available, use GPU path.

        Returns:
            np.ndarray: Complex64 field with the aperture applied (NumPy array).
        """
        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        shape_key = aperture_data.get('shape', aperture_data.get('type', 'square'))
        shape_type = str(shape_key).lower()
        width_mm = float(aperture_data['width'])
        width_m  = width_mm * 1e-3

        # Build coordinate arrays centered at field center
        x_arr = (np.arange(Nx, dtype=np.float32) - (Nx - 1) / 2.0) * float(dx)
        y_arr = (np.arange(Ny, dtype=np.float32) - (Ny - 1) / 2.0) * float(dy)
        half = 0.5 * width_m

        if use_gpu and (cp is not None):
            x_gpu = cp.asarray(x_arr)
            y_gpu = cp.asarray(y_arr)
            Xgpu = x_gpu[None, :].repeat(Ny, axis=0)
            Ygpu = y_gpu[:, None].repeat(Nx, axis=1)

            if shape_type == 'square':
                mask = (cp.abs(Xgpu) <= half) & (cp.abs(Ygpu) <= half)
            elif shape_type == 'circular':
                R2 = Xgpu * Xgpu + Ygpu * Ygpu
                r0 = half
                mask = (R2 <= (r0 * r0))
            else:
                mask = (cp.abs(Xgpu) <= half) & (cp.abs(Ygpu) <= half)

            F_gpu = cp.asarray(field, dtype=cp.complex64)
            F_gpu[~mask] = 0.0 + 0.0j
            return F_gpu.get()

        # CPU path
        E_out = np.array(field, copy=True)
        for iy in range(Ny):
            yy = y_arr[iy]
            for ix in range(Nx):
                xx = x_arr[ix]
                if shape_type == 'square':
                    if (abs(xx) > half) or (abs(yy) > half):
                        E_out[iy, ix] = 0.0
                elif shape_type == 'circular':
                    if (xx * xx + yy * yy) > (half * half):
                        E_out[iy, ix] = 0.0
                else:
                    if (abs(xx) > half) or (abs(yy) > half):
                        E_out[iy, ix] = 0.0
        return E_out
    
    def apply_stack(
        self,
        field,
        dx,
        dy,
        wavelength,
        propagate_free_space,
        use_gpu=True
    ):
        """
        Apply all components in this optics stack to 'field' in-order.

        Parameters
        ----------
        field : array-like (Ny, Nx), complex64
            Input complex field (NumPy). If the free-space propagator returns a
            CuPy array, this function will fetch it back to host automatically.
        dx, dy : float
            Pixel sizes along x and y in meters.
        wavelength : float
            Wavelength in meters. Required by some components (e.g., angular filter, lenses).
        propagate_free_space : callable
            Function with signature:
                out = propagate_free_space(field, dx, dy, z)
            It should apply free-space propagation over distance 'z' (meters)
            and return a NumPy complex64 array (or a CuPy array, which will be
            converted to NumPy here).
        use_gpu : bool
            Whether to request GPU-accelerated paths for optics-internal
            operations when available.

        Returns
        -------
        np.ndarray (Ny, Nx), complex64
            Field after applying all components.
        """
        if wavelength is None:
            raise ValueError("wavelength must be provided to optics.apply_stack")

        E = np.asarray(field, dtype=np.complex64)
        dx = float(dx)
        dy = float(dy)

        for elem in self.components:
            kind = str(elem.get("kind", "")).lower()

            if kind == "free space":
                z = float(elem.get("length", 0.0)) * 1e-3
                E = propagate_free_space(E, dx, dy, z)
                # Ensure NumPy on return so downstream ops are consistent
                if cp is not None and hasattr(cp, "ndarray") and isinstance(E, cp.ndarray):
                    E = E.get()
                E = np.asarray(E, dtype=np.complex64)

            elif kind == "lens box":
                E = self._apply_thin_lens_box(
                    E, dx, dy, elem, wavelength=wavelength, use_gpu=use_gpu
                )

            elif kind == "bragg magnifier 2b":
                E = self._apply_bragg_magnifier_2b(
                    E, dx, dy, elem, use_gpu=use_gpu
                )

            elif kind == "angular filter":
                E = self._apply_angular_filter_kspace(
                    E, dx, dy, elem, wavelength=wavelength, use_gpu=use_gpu
                )

            elif kind == "aperture":
                E = self._apply_aperture(
                    E, dx, dy, elem, use_gpu=use_gpu
                )

            else:
                raise ValueError(f'Unknown optics element "{kind}"')

        return np.asarray(E, dtype=np.complex64)

    def read_optics_metadata(self):
        """
        Stub for reading an optics JSON or other meta file from self.directory
        """
        pass

    def write_optics_metadata(self):
        """
        Stub for writing an optics JSON or other meta file to self.directory
        """
        pass

    def add_free_space(self, length_mm):
        """
        Add a free-space propagation segment of length in millimeters.
        """
        self._components.append({
            'kind'   : 'free space',
            'length' : float(length_mm)
        })

    def add_CRL_box(self, number, focal_length_mm, thickness_mm,
                    absorption_sigma=np.inf):
        """
        A simplified compound refractive lens (CRL) "box" specification.
        For example, 'number' CRLs in series, each of focal_length_mm in
        thin-lens approximation, thickness_mm for absorption, etc.
        """
        self._components.append({
            'kind'           : 'lens box',
            'number'         : int(number),
            'focal_length'   : float(focal_length_mm),
            'thickness'      : float(thickness_mm),
            'absorption_sigma': float(absorption_sigma)
        })
        
    def add_bragg_magnifier_2b(self, magnification_x, magnification_y,
                            reflectivity=1.0, phase_shift=0.0,
                            order=1, pad_mode='zeros', conserve_energy=True):
        """
        Two-bounce 2D Bragg Magnifier component (geometric resampling model).

        Appends a component dict describing an anisotropic magnification that
        mimics a two-bounce asymmetric Bragg magnifier pair: the first bounce
        magnifies one axis, the second bounce the orthogonal axis.

        Args:
            magnification_x (float): Net magnification along x.
            magnification_y (float): Net magnification along y.
            reflectivity (float): Intensity reflectivity (0..1). Amplitude factor
                sqrt(reflectivity) is applied. Default 1.0.
            phase_shift (float): Uniform phase added in radians. Default 0.0.
            order (int): 0 for nearest, 1 for bilinear interpolation. Default 1.
            pad_mode (str): 'zeros' or 'edge' padding when sampling outside field.
                Default 'zeros'.
            conserve_energy (bool): If True, multiply by 1/sqrt(|Mx*My|) to
                approximately conserve L2 norm after resampling. Default True.
        """
        self._components.append({
            'kind': 'bragg magnifier 2b',
            'magnification_x': float(magnification_x),
            'magnification_y': float(magnification_y),
            'reflectivity': float(reflectivity),
            'phase_shift': float(phase_shift),
            'order': int(order),
            'pad_mode': str(pad_mode),
            'conserve_energy': bool(conserve_energy),
        })
        
    def add_angular_filter(self,
                        half_angle_mrad,
                        center_x_mrad=0.0,
                        center_y_mrad=0.0,
                        shape='circular',
                        half_angle_y_mrad=None,
                        rolloff='tophat',
                        order=4,
                        transmission=1.0,
                        phase_shift=0.0,
                        mode='2d',
                        roll_deg=0.0):
        """
        Analyzer-like k-space angular filter.

        Semantics aligned with beam.analyser_mode:
        - rolloff='tophat' : half_angle_mrad is the acceptance half-angle (mrad).
        - rolloff='darwin' : half_angle_mrad is the Darwin halfwidth (mrad).
        The mask is applied to the field's Fourier spectrum as an amplitude filter.

        Parameters
        ----------
        center_x_mrad, center_y_mrad : float
            Analyzer axis offset in milliradians relative to the optical axis.
        mode : '2d' or '1d'
            '2d' circular/elliptical acceptance; '1d' slit along the rolled pass axis.
        shape : 'circular' or 'elliptical'
            Only used for '2d'. 'circular' matches the isotropic analyser in beam.
        roll_deg : float
            Roll angle of the acceptance axes (degrees). 0 aligns pass axis with +x.
        transmission : float
            Peak intensity transmission (0..1). Amplitude factor sqrt(transmission) is applied.
        phase_shift : float
            Uniform phase (radians) applied after the mask.
        order : int
            Only used for 'butterworth' soft edges (not part of analyser), defaults to 4.
        """
        if shape.lower() not in ('circular', 'elliptical'):
            shape = 'circular'
        if half_angle_y_mrad is None:
            half_angle_y_mrad = half_angle_mrad

        self._components.append({
            'kind'              : 'angular filter',
            'shape'             : shape.lower(),
            'center_x_mrad'     : float(center_x_mrad),
            'center_y_mrad'     : float(center_y_mrad),
            'half_angle_x_mrad' : float(half_angle_mrad),
            'half_angle_y_mrad' : float(half_angle_y_mrad),
            'rolloff'           : str(rolloff).lower(),
            'order'             : int(order),
            'transmission'      : float(transmission),
            'phase_shift'       : float(phase_shift),
            'mode'              : str(mode).lower(),
            'roll_deg'          : float(roll_deg),
        })

    def add_aperture(self, width_mm, shape='square'):
        """
        Hard aperture (default: square of given width in mm).
        """
        self._components.append({
            'kind'  : 'aperture',
            'type'  : shape.lower(),
            'width' : float(width_mm)
        })

    def add_custom_component(self, component):
        """
        Add any arbitrary custom component (dict) to the optics stack.
        """
        self._components.append(component)

    def plot_stack_3d(self,
                    unit='m',
                    cross_section_m=0.02,
                    thin_element_thickness_m=1e-3,
                    bragg_thickness_m=5e-3,
                    colors=None,
                    annotate=True,
                    show=True,
                    savepath=None,
                    ax=None):
        """
        Plot the optical stack in 3D with components lying along the x axis.

        Parameters
        ----------
        unit : str
            Display unit for the axes and labels. One of:
            'm', 'cm', 'mm', 'um', 'nm'. Default 'm'.
        cross_section_m : float
            Square cross section size (y and z extents) in meters for all boxes.
            Default 0.02 (2 cm).
        thin_element_thickness_m : float
            Fallback thickness in meters for elements without a defined axial length.
            Default 1e-3 (1 mm).
        bragg_thickness_m : float
            Fallback thickness in meters for the bragg magnifier element.
            Default 5e-3 (5 mm).
        colors : dict or None
            Optional map from component kind -> color hex, for example:
            {'free space':'#cfd8dc', 'lens box':'#ffcc80', 'aperture':'#90caf9'}
            If None, sensible defaults are used.
        annotate : bool
            If True, place text labels above each element.
        show : bool
            If True, call plt.show() at the end.
        savepath : str or None
            If provided, save the figure to this path.
        ax : mpl_toolkits.mplot3d.Axes3D or None
            If provided, draw into this axes; otherwise a new figure and axes are created.

        Returns
        -------
        (fig, ax) : matplotlib Figure and 3D Axes
        """
        import math
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.patches import Patch

        if not hasattr(self, "_components") or len(self._components) == 0:
            raise ValueError("No components to plot. Add components first.")

        # Display unit scale (meters -> display unit)
        unit = str(unit).lower()
        unit_scale = {
            'm': 1.0,
            'cm': 1e2,
            'mm': 1e3,
            'um': 1e6,
            'nm': 1e9
        }
        if unit not in unit_scale:
            raise ValueError("unit must be one of: m, cm, mm, um, nm")

        S = unit_scale[unit]

        # Default colors per kind
        default_colors = {
            'free space': '#cfd8dc',        # light gray
            'lens box': '#ffcc80',          # orange-ish
            'aperture': '#90caf9',          # light blue
            'bragg magnifier 2b': '#a5d6a7',# light green
            'custom': "#e93600"             # blue-gray
        }
        if colors is None:
            colors = {}
        # Merge user colors over defaults
        merged_colors = dict(default_colors)
        merged_colors.update(colors)

        # Helper to compute axial length in meters from a component dict
        def comp_length_m(comp):
            k = str(comp.get('kind', 'custom')).lower()
            # Preferred explicit meter fields
            if 'length_m' in comp:
                return float(comp['length_m'])
            if 'thickness_m' in comp:
                return float(comp['thickness_m'])

            # Known kinds using mm in your current code base
            if k == 'free space':
                # 'length' is in mm
                mm = float(comp.get('length', 0.0))
                return mm * 1e-3
            if k == 'lens box':
                # number * thickness (mm) -> meters
                t_mm = float(comp.get('thickness', 0.0))
                n = int(comp.get('number', 1))
                L = n * t_mm * 1e-3
                return L if L > 0 else thin_element_thickness_m
            if k == 'aperture':
                # optional thickness_mm
                if 'thickness_mm' in comp:
                    return float(comp['thickness_mm']) * 1e-3
                return thin_element_thickness_m
            if k == 'bragg magnifier 2b':
                return bragg_thickness_m

            # Generic fallbacks using mm fields if present
            if 'length' in comp:
                return float(comp['length']) * 1e-3
            if 'thickness' in comp:
                return float(comp['thickness']) * 1e-3

            return thin_element_thickness_m

        # Helper to format a label
        def comp_label(comp):
            k = str(comp.get('kind', 'custom'))
            kl = k.lower()
            if kl == 'free space':
                Lm = comp_length_m(comp)
                return f"free space, L={Lm:.4g} m"
            if kl == 'lens box':
                n = comp.get('number', None)
                fmm = comp.get('focal_length', None)
                if n is not None and fmm is not None:
                    return f"lens box, N={int(n)}, f={float(fmm):.4g} mm"
                return "lens box"
            if kl == 'aperture':
                wmm = comp.get('width', None)
                shape = comp.get('type', comp.get('shape', 'square'))
                if wmm is not None:
                    return f"aperture, {shape}, w={float(wmm):.4g} mm"
                return f"aperture, {shape}"
            if kl == 'bragg magnifier 2b':
                mx = comp.get('magnification_x', None)
                my = comp.get('magnification_y', None)
                if (mx is not None) and (my is not None):
                    return f"bragg magnifier 2b, Mx={float(mx):.3g}, My={float(my):.3g}"
                return "bragg magnifier 2b"
            # Custom
            name = comp.get('name', comp.get('kind', 'custom'))
            return str(name)

        # Build drawable segments with positions in meters
        segments = []  # each: dict(x0_m, x1_m, kind, color, label)
        x_cursor = 0.0
        for comp in self._components:
            kind = str(comp.get('kind', 'custom'))
            k_lower = kind.lower()
            Lm = comp_length_m(comp)
            if Lm < 0:
                raise ValueError(f"Negative length encountered for component {kind}.")
            x0 = x_cursor
            x1 = x_cursor + Lm
            color = merged_colors.get(k_lower, merged_colors['custom'])
            label = comp_label(comp)
            segments.append({
                'x0_m': x0,
                'x1_m': x1,
                'kind': k_lower,
                'color': color,
                'label': label
            })
            x_cursor = x1

        total_length_m = x_cursor
        if total_length_m <= 0:
            raise ValueError("Total stack length is zero.")

        # Prepare axes
        new_fig = False
        if ax is None:
            fig = plt.figure(figsize=(10, 3.5))
            ax = fig.add_subplot(111, projection='3d')
            new_fig = True
        else:
            fig = ax.figure

        # Convert cross section and coordinates to display units
        cs_u = cross_section_m * S

        def draw_box(ax, x0_m, x1_m, color, edgecolor='k', alpha=0.9):
            x0 = x0_m * S
            x1 = x1_m * S
            y0 = -0.5 * cs_u
            y1 = +0.5 * cs_u
            z0 = -0.5 * cs_u
            z1 = +0.5 * cs_u

            verts = [
                # bottom (z0)
                [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
                # top (z1)
                [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
                # front (y0)
                [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
                # back (y1)
                [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
                # left (x0)
                [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
                # right (x1)
                [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
            ]
            box = Poly3DCollection(verts, facecolors=color, edgecolors=edgecolor, linewidths=0.6, alpha=alpha)
            ax.add_collection3d(box)

        # Draw all boxes
        legend_kinds = []
        for seg in segments:
            draw_box(ax, seg['x0_m'], seg['x1_m'], seg['color'])
            if annotate:
                xc = 0.5 * (seg['x0_m'] + seg['x1_m']) * S
                ax.text(xc, 0.0, 0.6 * cs_u, seg['label'],
                        ha='center', va='bottom', fontsize=8, zdir=None)
            if seg['kind'] not in legend_kinds:
                legend_kinds.append(seg['kind'])

        # Axes limits and labels
        ax.set_xlabel(f"x [{unit}]")
        ax.set_ylabel(f"y [{unit}]")
        ax.set_zlabel(f"z [{unit}]")

        Lx_u = total_length_m * S
        pad_x = max(0.02 * Lx_u, 0.1 * cs_u)
        ax.set_xlim(-0.02 * Lx_u, Lx_u + pad_x)
        ax.set_ylim(-0.6 * cs_u, 0.6 * cs_u)
        ax.set_zlim(-0.6 * cs_u, 0.8 * cs_u)

        # Make proportions reasonable: emphasize axial length
        ax.set_box_aspect((max(Lx_u, cs_u), cs_u, cs_u))
        ax.view_init(elev=15, azim=-60)
        ax.grid(True, which='both', alpha=0.3)

        # Legend
        legend_handles = []
        for k in legend_kinds:
            color = merged_colors.get(k, merged_colors['custom'])
            label = k
            legend_handles.append(Patch(facecolor=color, edgecolor='k', label=label))
        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', framealpha=0.9)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches='tight')

        if show and new_fig:
            plt.show()

        return fig, ax

    @property
    def components(self):
        """
        Return the internal list of components.
        """
        return self._components