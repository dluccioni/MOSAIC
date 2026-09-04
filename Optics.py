# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import os
import gc
import json
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
        "add_fresnel_zone_plate",
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
        provided, a uniform attenuation factor is applied. If `mu_per_m` and
        `radius_of_curvature_m` are provided, a parabolic (r^2-dependent) CRL
        absorption factor is applied in addition.

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            lens_data (dict): Lens parameters:
                - 'focal_length' (float, mm)
                - 'thickness' (float, mm)
                - 'number' (int): number of identical lens elements
                - 'absorption_sigma' (float, meters) optional
                - 'mu_per_m' (float, 1/m) optional: intensity linear attenuation
                  coefficient of the lens material
                - 'radius_of_curvature_m' (float, meters) optional: per-surface
                  parabolic apex radius R
            wavelength (float or None): Wavelength in meters. If None, tries
                self._wavelength.
            use_gpu (bool): If True and CuPy is available, use a GPU path.

        Returns:
            array: Complex64 array with the lens applied. A CuPy array on the
            GPU path, otherwise NumPy.
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

        # Parabolic (r^2-dependent) CRL absorption parameters (SI units).
        # mu_per_m: intensity linear attenuation coefficient of the lens
        # material (1/m). radius_of_curvature_m: per-surface parabolic apex
        # radius R (m). Both default to no-ops (0.0 and inf respectively).
        mu_per_m = float(lens_data.get('mu_per_m', 0.0))
        R_apex_m = float(lens_data.get('radius_of_curvature_m', np.inf))

        # Each parabolic surface has profile r^2/(2R); a bi-parabolic lens has
        # two surfaces, so the r^2-dependent material thickness per lens is
        # t(r) = r^2/R. Intensity transmission through N lenses is therefore
        # T(r) = exp(-mu * N * r^2 / R), and the AMPLITUDE gets
        # sqrt(T) = exp(-mu * N * r^2 / (2R)).
        # r^2 grids below are in m^2 (dx, dy are in meters), so no unit
        # conversion is needed for the m^2 * (1/m) / m exponent.
        apply_parabolic = (mu_per_m > 0.0) and np.isfinite(R_apex_m)
        parab_coef = (mu_per_m * N_lenses / (2.0 * R_apex_m)) if apply_parabolic else 0.0

        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        xp = cp if (use_gpu and cp is not None) else np

        # Centred coordinate axes in metres; the r^2 grid is built by broadcasting
        x = (xp.arange(Nx, dtype=xp.float32) - (Nx - 1) / 2.0) * float(dx)
        y = (xp.arange(Ny, dtype=xp.float32) - (Ny - 1) / 2.0) * float(dy)
        R2 = x[None, :] ** 2 + y[:, None] ** 2

        # Thin-lens phase exp(-i k r^2 / (2f))
        phase_lens = (-0.5 * (k_val / f)) * R2
        E_out = xp.asarray(field, dtype=xp.complex64) * xp.exp(1j * phase_lens).astype(xp.complex64)

        # Optional parabolic (r^2-dependent) CRL absorption:
        # amplitude factor exp(-mu * N * r^2 / (2R)), R2 in m^2.
        if apply_parabolic:
            E_out *= xp.exp(- parab_coef * R2).astype(xp.float32)

        # Optional uniform absorption
        if np.isfinite(nsigma):
            E_out *= np.float32(np.exp(- N_lenses * t / nsigma))

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
            array: Complex64 array with the magnifier applied, on the same
            (Ny, Nx) grid and pixel size as the input. A CuPy array on the
            GPU path, otherwise NumPy.
        """
        xp = cp if (use_gpu and cp is not None) else np

        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        Mx = float(mag_data.get('magnification_x', 1.0))
        My = float(mag_data.get('magnification_y', 1.0))
        refl = float(mag_data.get('reflectivity', 1.0))
        phi = float(mag_data.get('phase_shift', 0.0))
        order = int(mag_data.get('order', 1))
        pad_mode = str(mag_data.get('pad_mode', 'zeros')).lower()
        conserve = bool(mag_data.get('conserve_energy', True))

        F = xp.asarray(field, dtype=xp.complex64)
        cx = (Nx - 1) / 2.0
        cy = (Ny - 1) / 2.0

        # Map output -> input coords (centered anisotropic scaling). The 1-D
        # axes are kept as (1, Nx) and (Ny, 1) and broadcast in the indexing.
        Xi = cx + (xp.arange(Nx, dtype=xp.float32)[None, :] - cx) / Mx
        Yi = cy + (xp.arange(Ny, dtype=xp.float32)[:, None] - cy) / My

        # Interpolation
        if order == 0:
            # Nearest
            Xi_n = xp.rint(Xi).astype(xp.int32)
            Yi_n = xp.rint(Yi).astype(xp.int32)
            valid = (Xi_n >= 0) & (Xi_n < Nx) & (Yi_n >= 0) & (Yi_n < Ny)
            out = F[xp.clip(Yi_n, 0, Ny - 1), xp.clip(Xi_n, 0, Nx - 1)]
            out = xp.where(valid, out, xp.complex64(0.0))
        else:
            # Bilinear
            x0 = xp.floor(Xi).astype(xp.int32)
            y0 = xp.floor(Yi).astype(xp.int32)
            wx = (Xi - x0).astype(xp.float32)
            wy = (Yi - y0).astype(xp.float32)

            if pad_mode == 'edge':
                x0c = xp.clip(x0, 0, Nx - 1); x1c = xp.clip(x0 + 1, 0, Nx - 1)
                y0c = xp.clip(y0, 0, Ny - 1); y1c = xp.clip(y0 + 1, 0, Ny - 1)
                wx = Xi - x0c; wy = Yi - y0c
                valid = None
            else:
                valid = (Xi >= 0) & (Xi < Nx - 1) & (Yi >= 0) & (Yi < Ny - 1)
                x0c = xp.clip(x0, 0, Nx - 2); x1c = x0c + 1
                y0c = xp.clip(y0, 0, Ny - 2); y1c = y0c + 1

            f00 = F[y0c, x0c]; f10 = F[y0c, x1c]
            f01 = F[y1c, x0c]; f11 = F[y1c, x1c]
            out = (f00 * (1 - wx) * (1 - wy) +
                   f10 * wx * (1 - wy) +
                   f01 * (1 - wx) * wy +
                   f11 * wx * wy).astype(xp.complex64)
            if valid is not None:
                out = xp.where(valid, out, xp.complex64(0.0))

        amp = np.sqrt(max(refl, 0.0))
        if conserve:
            jac = abs(Mx * My)
            if jac > 0:
                amp *= (1.0 / np.sqrt(jac))

        # Uniform reflectivity and phase as one complex scalar
        out *= np.complex64(amp * np.exp(1j * phi))
        return out
        
    def _apply_angular_filter_kspace(self, field, dx, dy, filt, wavelength, use_gpu=True):
        """
        Analyzer-like angular pupil in k-space.

        This matches beam.analyser_mode semantics:
        - rolloff='tophat'  : amplitude = 1 inside half-angle, else 0
        - rolloff='darwin'  : amplitude = 1 / (1 + (delta / halfwidth)^2)

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            filt (dict): Filter parameters including 'center_x_mrad', 'center_y_mrad',
                'half_angle_x_mrad', 'half_angle_y_mrad', 'mode', 'shape', 'rolloff',
                'roll_deg', 'transmission', 'phase_shift', and 'order'.
            wavelength (float): Wavelength in meters.
            use_gpu (bool): If True and CuPy is available, use GPU path.

        Returns:
            array: Complex64 field with the angular filter applied. A CuPy
            array on the GPU path, otherwise NumPy.

        Note:
            - delta is the small-angle deviation (radians) in the paraxial limit:
                delta = sqrt((theta_x - cx)^2 + (theta_y - cy)^2) for circular 2D.
              For 'elliptical' we use the normalized radius r^2 = (tpar/hx)^2 + (tperp/hy)^2
              and apply the same laws with r in place of delta/hw.
            - 'mode' can be '2d' (default) or '1d' (slit). In '1d', acceptance is along
              the rolled pass axis only: delta = |tpar|.
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
        phasor = np.complex64(amp * np.exp(1j * phase))

        # FFT -> mask -> iFFT
        E = xp.asarray(field, dtype=xp.complex64)
        F = xp.fft.fft2(E)
        F *= H.astype(xp.complex64)
        Eo = xp.fft.ifft2(F).astype(xp.complex64)
        Eo *= phasor
        return Eo

    def _apply_fresnel_zone_plate(self, field, dx, dy, zp_data,
                                  wavelength=None, use_gpu=True):
        """
        Apply a Fresnel zone plate as a real-space binary transmission mask.

        Zone boundaries are at r_n = sqrt(n * lambda * f1) with the first-order
        focal length f1 = D * dr_N / lambda, so the plate geometry is fixed by
        D and dr_N alone. For an amplitude FZP, even zones transmit and odd
        zones are opaque. For a phase FZP, even zones transmit with phase 0 and
        odd zones with phase pi. All diffraction orders are naturally produced;
        'order' only selects the working focus f1 / m of the 'ideal' lens model.

        Args:
            field (array-like): Complex field, shape (Ny, Nx).
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            zp_data (dict): Zone plate parameters from add_fresnel_zone_plate.
            wavelength (float or None): Wavelength in meters.
            use_gpu (bool): If True and CuPy is available, use GPU path.

        Returns:
            array: Complex64 field after FZP transmission. A CuPy array on the
            GPU path, otherwise NumPy.
        """
        if wavelength is None:
            raise ValueError("wavelength must be provided to _apply_fresnel_zone_plate")

        dr_N_m  = float(zp_data['outermost_zone_width_nm']) * 1e-9
        D_m     = float(zp_data['diameter_um']) * 1e-6
        order   = max(1, int(zp_data.get('order', 1)))
        eff     = float(zp_data.get('efficiency', 1.0))
        cs      = float(zp_data.get('central_stop_fraction', 0.0))
        zp_type = str(zp_data.get('zone_plate_type', 'amplitude')).lower()

        # First-order focal length sets the zone geometry; the m-th order
        # focuses at f1 / m.
        f   = D_m * dr_N_m / float(wavelength)
        f_work = f / order
        R   = D_m / 2.0
        amp = float(np.sqrt(max(eff, 0.0)))

        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        on_gpu = bool(use_gpu and (cp is not None))
        xp = cp if on_gpu else np

        # Real-space coordinate grids (meters, centered)
        cx = (Nx - 1) / 2.0
        cy = (Ny - 1) / 2.0
        x = (xp.arange(Nx, dtype=xp.float32) - cx) * float(dx)
        y = (xp.arange(Ny, dtype=xp.float32) - cy) * float(dy)
        R2 = x[None, :] ** 2 + y[:, None] ** 2

        # Zone index: n(r) = r^2 / (lambda * f)
        n_zone = R2 / (float(wavelength) * f)
        zone_parity = xp.floor(n_zone).astype(xp.int32) % 2  # 0=even, 1=odd

        # Circular aperture
        inside = R2 <= (R * R)

        # Central stop
        if cs > 0.0:
            R_stop = R * cs
            inside = inside & (R2 >= (R_stop * R_stop))

        # Build transmission
        T = xp.zeros((Ny, Nx), dtype=xp.complex64)
        if zp_type == 'ideal':
            # Ideal thin-lens phase within circular aperture, focus at f1 / m
            k_val = 2.0 * np.pi / float(wavelength)
            phase = (-0.5 * (k_val / f_work)) * R2
            T[inside] = amp * xp.exp(1j * phase[inside]).astype(xp.complex64)
        elif zp_type == 'phase':
            # Phase FZP: even zones +amp, odd zones -amp (pi phase shift)
            T[inside & (zone_parity == 0)] = amp
            T[inside & (zone_parity == 1)] = -amp
        else:
            # Amplitude FZP: even zones transmit, odd zones blocked
            T[inside & (zone_parity == 0)] = amp

        # Edge apodization (Tukey/cosine taper)
        taper = float(zp_data.get('edge_taper', 0.0))
        if taper > 0.0:
            r = xp.sqrt(R2)
            r_inner = R * (1.0 - taper)
            taper_zone = (r > r_inner) & (r <= R)
            window = 0.5 * (1.0 + xp.cos(xp.pi * (r[taper_zone] - r_inner) / (R * taper)))
            T[taper_zone] *= window.astype(xp.float32)

        # Apply transmission
        E = xp.asarray(field, dtype=xp.complex64) * T
        return E.astype(xp.complex64, copy=False)

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
            array: Complex64 field with the aperture applied. A CuPy array on
            the GPU path, otherwise NumPy.
        """
        Ny, Nx = int(field.shape[0]), int(field.shape[1])
        shape_key = aperture_data.get('shape', aperture_data.get('type', 'square'))
        shape_type = str(shape_key).lower()
        width_mm = float(aperture_data['width'])
        width_m  = width_mm * 1e-3
        half = 0.5 * width_m

        xp = cp if (use_gpu and cp is not None) else np

        # Centred coordinate axes in metres, broadcast to the (Ny, Nx) mask
        x = (xp.arange(Nx, dtype=xp.float32) - (Nx - 1) / 2.0) * float(dx)
        y = (xp.arange(Ny, dtype=xp.float32) - (Ny - 1) / 2.0) * float(dy)
        if shape_type == 'circular':
            mask = (x[None, :] ** 2 + y[:, None] ** 2) <= (half * half)
        else:
            # 'square' and any unrecognised shape
            mask = (xp.abs(x) <= half)[None, :] & (xp.abs(y) <= half)[:, None]

        E_out = xp.array(field, dtype=xp.complex64, copy=True)
        E_out[~mask] = 0.0
        return E_out
    
    def _check_quadratic_phase_sampling(self, index, label, k_val, f, dx, dy,
                                        shape, r_limit=None):
        """
        Warn when a quadratic phase exp(-i k r^2 / 2f) is under-sampled.

        The phase step between neighbouring pixels along x is k dx x / f, so
        the largest step occurs at the field edge. A step above pi aliases.

        Args:
            index (int): 1-based component index, for the log message.
            label (str): Component description, for the log message.
            k_val (float): Wavenumber 2 pi / lambda in 1/m.
            f (float): Focal length of the phase term in meters.
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            shape (tuple[int, int]): Field shape (Ny, Nx).
            r_limit (float or None): Aperture radius in meters; the edge is
                taken as min(field half-extent, r_limit) when given.

        Returns:
            float: Largest per-pixel phase step at the field edge in radians.
        """
        Ny, Nx = int(shape[0]), int(shape[1])
        x_max = (Nx - 1) / 2.0 * dx
        y_max = (Ny - 1) / 2.0 * dy
        if r_limit is not None:
            x_max = min(x_max, float(r_limit))
            y_max = min(y_max, float(r_limit))
        f = abs(float(f))
        if f <= 0.0:
            return float('inf')
        step = max(k_val * dx * x_max / f, k_val * dy * y_max / f)
        if step > np.pi:
            self._log("normal",
                      f"Component {index}: {label} phase is under-sampled at the "
                      f"field edge: {step:.2f} rad/pixel > pi. The grid resolves "
                      f"angles up to lambda/(2 dx) = {np.pi / (k_val * max(dx, dy)) * 1e3:.3f} mrad; "
                      f"the edge ray angle is {max(x_max, y_max) / f * 1e3:.3f} mrad.")
        return float(step)

    def apply_stack(
        self,
        field,
        dx,
        dy,
        wavelength,
        propagate_free_space,
        use_gpu=True,
        diagnostics=False
    ):
        """
        Apply all components in this optics stack to 'field' in-order.

        The field is moved to the device once at entry (when use_gpu and CuPy
        are available) and back to the host once at exit; components pass
        device arrays between them. The pixel size is the same for every
        component: the Bragg magnifier resamples the magnified field onto the
        input grid instead of changing the grid spacing.

        Args:
            field (array-like): Input complex field, shape (Ny, Nx), complex64.
            dx (float): Pixel size along x in meters.
            dy (float): Pixel size along y in meters.
            wavelength (float): Wavelength in meters. Required by some components
                (e.g., angular filter, lenses).
            propagate_free_space (callable): Function with signature
                ``out = propagate_free_space(field, dx, dy, z)``. It should apply
                free-space propagation over distance 'z' (meters) and return a
                NumPy or CuPy complex64 array. On the GPU path it receives a
                CuPy array.
            use_gpu (bool): Whether to request GPU-accelerated paths for optics-internal
                operations when available.
            diagnostics (bool): If True, log the mean field amplitude before and
                after each component. Off by default: each value is a full
                reduction over the field.

        Returns:
            np.ndarray: Field after applying all components, shape (Ny, Nx), complex64.
        """
        if wavelength is None:
            raise ValueError("wavelength must be provided to optics.apply_stack")

        on_gpu = bool(use_gpu and cp is not None)
        xp = cp if on_gpu else np
        E = xp.asarray(field, dtype=xp.complex64)
        dx = float(dx)
        dy = float(dy)
        lam = float(wavelength)
        k_val = 2.0 * np.pi / lam

        self._log("normal", f"apply_stack: processing {len(self.components)} component(s)")

        for i, elem in enumerate(self.components):
            kind = str(elem.get("kind", "")).lower()
            if diagnostics:
                pre_amp = float(xp.abs(E).mean())

            if kind == "free space":
                z = float(elem.get("length", 0.0)) * 1e-3
                self._log("normal", f"Component {i+1}: free space, z={z:.6e} m ({elem.get('length', 0.0):.2f} mm)")
                E = propagate_free_space(E, dx, dy, z)
                if (not on_gpu) and cp is not None and isinstance(E, cp.ndarray):
                    E = E.get()
                E = xp.asarray(E, dtype=xp.complex64)

            elif kind == "lens box":
                self._log("normal", f"Component {i+1}: lens box, N={elem.get('number')}, f={elem.get('focal_length')} mm")
                self._check_quadratic_phase_sampling(
                    i + 1, "lens box", k_val, float(elem['focal_length']) * 1e-3,
                    dx, dy, E.shape
                )
                E = self._apply_thin_lens_box(
                    E, dx, dy, elem, wavelength=lam, use_gpu=on_gpu
                )

            elif kind == "bragg magnifier 2b":
                self._log("normal", f"Component {i+1}: bragg magnifier, Mx={elem.get('magnification_x')}, My={elem.get('magnification_y')}")
                E = self._apply_bragg_magnifier_2b(
                    E, dx, dy, elem, use_gpu=on_gpu
                )

            elif kind == "angular filter":
                self._log("normal", f"Component {i+1}: angular filter, half_angle={elem.get('half_angle_x_mrad')} mrad")
                E = self._apply_angular_filter_kspace(
                    E, dx, dy, elem, wavelength=lam, use_gpu=on_gpu
                )

            elif kind == "aperture":
                self._log("normal", f"Component {i+1}: aperture, width={elem.get('width')} mm, shape={elem.get('type', elem.get('shape'))}")
                E = self._apply_aperture(
                    E, dx, dy, elem, use_gpu=on_gpu
                )

            elif kind == "zone plate":
                dr_nm = float(elem.get('outermost_zone_width_nm'))
                dr_m  = dr_nm * 1e-9
                D_um  = float(elem.get('diameter_um'))
                D_m   = D_um * 1e-6
                m_ord = max(1, int(elem.get('order', 1)))
                # Plate geometry is that of the first order; order m works at f1 / m
                f1_m = D_m * dr_m / lam
                f_m = f1_m / m_ord
                NA_zp = min(m_ord * lam / (2.0 * dr_m), 1.0)
                res_nm = 1.22 * dr_nm / m_ord
                N_zones = int(round(D_m / (4.0 * dr_m)))
                zp_type = elem.get('zone_plate_type', 'amplitude')
                self._log("normal", f"Component {i+1}: zone plate ({zp_type}), dr_N={dr_nm} nm, "
                          f"D={D_um} um, order={m_ord}, NA={NA_zp:.4f}, res={res_nm:.2f} nm, "
                          f"f1={f1_m * 1e3:.4e} mm, f={f_m * 1e3:.4e} mm, N_zones={N_zones}")
                self._check_quadratic_phase_sampling(
                    i + 1, f"zone plate (order {m_ord})", k_val, f_m,
                    dx, dy, E.shape, r_limit=0.5 * D_m
                )
                E = self._apply_fresnel_zone_plate(
                    E, dx, dy, elem, wavelength=lam, use_gpu=on_gpu
                )

            else:
                raise ValueError(f'Unknown optics element "{kind}"')

            if diagnostics:
                post_amp = float(xp.abs(E).mean())
                self._log("normal", f"  -> amplitude: {pre_amp:.6e} -> {post_amp:.6e}")

        if on_gpu:
            E = E.get()
        return np.asarray(E, dtype=np.complex64)

    def read_optics_metadata(self):
        """
        Read optics component stack from JSON metadata file in self.directory.

        Loads the component list from 'optics_metadata.json' if it exists.
        Each component is stored as a dictionary with 'kind' and component-specific
        parameters.

        Returns:
            bool: True if metadata was loaded successfully, False otherwise.
        """
        if self.directory is None:
            return False

        metadata_path = os.path.join(self.directory, "optics_metadata.json")
        if not os.path.exists(metadata_path):
            return False

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load components list
            self._components = metadata.get("components", [])

            # Validate and ensure proper types for each component
            for comp in self._components:
                kind = comp.get("kind", "").lower()
                if kind == "free space":
                    comp["length"] = float(comp.get("length", 0.0))
                elif kind == "lens box":
                    comp["number"] = int(comp.get("number", 1))
                    comp["focal_length"] = float(comp.get("focal_length", 0.0))
                    comp["thickness"] = float(comp.get("thickness", 0.0))
                    comp["absorption_sigma"] = float(comp.get("absorption_sigma", np.inf))
                    comp["mu_per_m"] = float(comp.get("mu_per_m", 0.0))
                    comp["radius_of_curvature_m"] = float(comp.get("radius_of_curvature_m", np.inf))
                elif kind == "bragg magnifier 2b":
                    comp["magnification_x"] = float(comp.get("magnification_x", 1.0))
                    comp["magnification_y"] = float(comp.get("magnification_y", 1.0))
                    comp["reflectivity"] = float(comp.get("reflectivity", 1.0))
                    comp["phase_shift"] = float(comp.get("phase_shift", 0.0))
                elif kind == "aperture":
                    comp["width"] = float(comp.get("width", 0.0))
                    comp["type"] = str(comp.get("type", "square"))
                elif kind == "angular filter":
                    comp["half_angle_x_mrad"] = float(comp.get("half_angle_x_mrad", 0.0))
                    comp["half_angle_y_mrad"] = float(comp.get("half_angle_y_mrad", 0.0))
                    comp["shape"] = str(comp.get("shape", "circular"))
                    comp["rolloff"] = str(comp.get("rolloff", "tophat"))

            return True
        except Exception as e:
            print(f"[Optics] Failed to read metadata: {e}")
            return False

    def write_optics_metadata(self):
        """
        Write optics component stack to JSON metadata file in self.directory.

        Saves the component list to 'optics_metadata.json'. Each component is
        stored as a dictionary with 'kind' and component-specific parameters.

        Returns:
            bool: True if metadata was saved successfully, False otherwise.
        """
        if self.directory is None:
            return False

        metadata_path = os.path.join(self.directory, "optics_metadata.json")

        try:
            # Build metadata dict
            metadata = {
                "components": self._components,
                "num_components": len(self._components),
            }

            # Write to JSON file with nice formatting
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=self._json_serializer)

            return True
        except Exception as e:
            print(f"[Optics] Failed to write metadata: {e}")
            return False

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj == np.inf:
            return "inf"
        elif obj == -np.inf:
            return "-inf"
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def add_free_space(self, length_mm):
        """
        Add a free-space propagation segment to the optics stack.

        Args:
            length_mm (float): Propagation length in millimeters.
        """
        self._components.append({
            'kind'   : 'free space',
            'length' : float(length_mm)
        })

    def add_CRL_box(self, number, focal_length_mm, thickness_mm,
                    absorption_sigma=np.inf, mu_per_m=0.0,
                    radius_of_curvature_m=float('inf')):
        """
        Add a compound refractive lens (CRL) box to the optics stack.

        A simplified CRL "box" specification using thin-lens approximation.

        Args:
            number (int): Number of identical lens elements in series. Enters
                only the absorption factors; the phase uses focal_length_mm
                directly.
            focal_length_mm (float): Focal length of the WHOLE stack in
                millimeters (the thin-lens phase applied is
                exp(-i k r^2 / (2 f)) with this f). For N bi-parabolic lenses
                of apex radius R and refractive decrement delta,
                f = R / (2 N delta).
            thickness_mm (float): Thickness of each lens element in millimeters
                (used for absorption calculation).
            absorption_sigma (float): Absorption length in meters. Default is np.inf
                (no absorption).
            mu_per_m (float): Intensity linear attenuation coefficient of the
                lens material in 1/m. Used for the parabolic (r^2-dependent)
                CRL absorption factor. Default is 0.0 (no parabolic absorption).
            radius_of_curvature_m (float): Per-surface parabolic apex radius R
                in meters. With two parabolic surfaces per lens, the r^2-dependent
                material thickness per lens is r^2/R, giving an intensity
                transmission exp(-mu * N * r^2 / R). Default is inf (no
                parabolic absorption).
        """
        self._components.append({
            'kind'           : 'lens box',
            'number'         : int(number),
            'focal_length'   : float(focal_length_mm),
            'thickness'      : float(thickness_mm),
            'absorption_sigma': float(absorption_sigma),
            'mu_per_m'       : float(mu_per_m),
            'radius_of_curvature_m': float(radius_of_curvature_m)
        })
        
    def add_bragg_magnifier_2b(self, magnification_x, magnification_y,
                            reflectivity=1.0, phase_shift=0.0,
                            order=1, pad_mode='zeros', conserve_energy=True):
        """
        Two-bounce 2D Bragg Magnifier component (geometric resampling model).

        Appends a component dict describing an anisotropic magnification that
        mimics a two-bounce asymmetric Bragg magnifier pair: the first bounce
        magnifies one axis, the second bounce the orthogonal axis. The
        magnified field is resampled onto the input pixel grid, so the pixel
        size seen by later components (and written back to the detector) is
        unchanged; features simply cover Mx (My) times more pixels.

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
        Add an analyzer-like k-space angular filter to the optics stack.

        Semantics aligned with beam.analyser_mode:
        - rolloff='tophat' : half_angle_mrad is the acceptance half-angle (mrad).
        - rolloff='darwin' : half_angle_mrad is the Darwin halfwidth (mrad).
        The mask is applied to the field's Fourier spectrum as an amplitude filter.

        Args:
            half_angle_mrad (float): Acceptance half-angle or Darwin halfwidth in
                milliradians, depending on rolloff mode.
            center_x_mrad (float): Analyzer axis offset along x in milliradians
                relative to the optical axis. Default 0.0.
            center_y_mrad (float): Analyzer axis offset along y in milliradians
                relative to the optical axis. Default 0.0.
            shape (str): 'circular' or 'elliptical'. Only used for '2d' mode.
                'circular' matches the isotropic analyser in beam. Default 'circular'.
            half_angle_y_mrad (float or None): Half-angle along y for elliptical shape.
                If None, defaults to half_angle_mrad.
            rolloff (str): 'tophat' or 'darwin'. Default 'tophat'.
            order (int): Only used for 'butterworth' soft edges (not part of analyser).
                Default 4.
            transmission (float): Peak intensity transmission (0..1). Amplitude factor
                sqrt(transmission) is applied. Default 1.0.
            phase_shift (float): Uniform phase in radians applied after the mask.
                Default 0.0.
            mode (str): '2d' for circular/elliptical acceptance, '1d' for slit along
                the rolled pass axis. Default '2d'.
            roll_deg (float): Roll angle of the acceptance axes in degrees.
                0 aligns pass axis with +x. Default 0.0.
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
        Add a hard aperture to the optics stack.

        Args:
            width_mm (float): Aperture width (or diameter for circular) in millimeters.
            shape (str): Aperture shape, either 'square' or 'circular'. Default 'square'.
        """
        self._components.append({
            'kind'  : 'aperture',
            'type'  : shape.lower(),
            'width' : float(width_mm)
        })

    def add_fresnel_zone_plate(self, outermost_zone_width_nm, diameter_um,
                               efficiency=1.0, order=1,
                               central_stop_fraction=0.0,
                               zone_plate_type='amplitude',
                               edge_taper=0.0):
        """
        Add a Fresnel zone plate to the optics stack.

        The FZP is modeled as a real-space binary transmission mask. Zone
        boundaries are defined by r_n = sqrt(n * lambda * f1), where
        f1 = D * dr_N / lambda is the first-order focal length. Even zones
        transmit and odd zones are either opaque (amplitude FZP) or
        pi-shifted (phase FZP).

        Args:
            outermost_zone_width_nm (float): Width of the outermost zone in
                nanometers. Determines resolution (1.22 * dr_N / m) and NA
                (m * lambda / (2 * dr_N)) for order m.
            diameter_um (float): Zone plate diameter in micrometers.
                Determines the focal length f1 = D * dr_N / lambda and
                the number of zones N = D / (4 * dr_N).
            efficiency (float): Diffraction efficiency into the imaging order
                (0-1). Amplitude is scaled by sqrt(efficiency). Default 1.0.
            order (int): Diffraction order used for imaging. The plate
                geometry is that of order 1; the working focus is f1 / m.
                For the binary types all orders are present in the mask, so
                'order' only affects the reported focus and the 'ideal' lens
                phase. Default 1.
            central_stop_fraction (float): Fraction of the zone plate radius
                blocked by a central stop (0-1). Default 0.0.
            zone_plate_type (str): 'ideal' (thin-lens phase, 1st order only),
                'amplitude' (binary open/blocked zones, all orders), or
                'phase' (binary 0/pi phase shift, all orders). Default 'amplitude'.
            edge_taper (float): Fraction of the radius over which transmission
                tapers smoothly to zero (0-1). Uses a cosine taper (Tukey window).
                0.0 = hard edge (strong Airy rings), 0.5 = taper over outer half.
                Default 0.0.
        """
        self._components.append({
            'kind'                   : 'zone plate',
            'outermost_zone_width_nm': float(outermost_zone_width_nm),
            'diameter_um'            : float(diameter_um),
            'efficiency'             : float(efficiency),
            'order'                  : int(order),
            'central_stop_fraction'  : float(central_stop_fraction),
            'zone_plate_type'        : str(zone_plate_type).lower(),
            'edge_taper'             : float(edge_taper),
        })

    def add_custom_component(self, component):
        """
        Add an arbitrary custom component to the optics stack.

        Args:
            component (dict): Component specification dictionary. Must include a 'kind'
                key to identify the component type during apply_stack processing.
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

        Args:
            unit (str): Display unit for the axes and labels. One of 'm', 'cm', 'mm',
                'um', 'nm'. Default 'm'.
            cross_section_m (float): Square cross section size (y and z extents) in
                meters for all boxes. Default 0.02 (2 cm).
            thin_element_thickness_m (float): Fallback thickness in meters for elements
                without a defined axial length. Default 1e-3 (1 mm).
            bragg_thickness_m (float): Fallback thickness in meters for the bragg
                magnifier element. Default 5e-3 (5 mm).
            colors (dict or None): Optional map from component kind to color hex,
                for example: {'free space':'#cfd8dc', 'lens box':'#ffcc80',
                'aperture':'#90caf9'}. If None, sensible defaults are used.
            annotate (bool): If True, place text labels above each element. Default True.
            show (bool): If True, call plt.show() at the end. Default True.
            savepath (str or None): If provided, save the figure to this path.
            ax (mpl_toolkits.mplot3d.Axes3D or None): If provided, draw into this axes;
                otherwise a new figure and axes are created.

        Returns:
            tuple: (fig, ax) matplotlib Figure and 3D Axes.
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
        Return the internal list of optical components.

        Returns:
            list: List of component dictionaries in the optics stack.
        """
        return self._components