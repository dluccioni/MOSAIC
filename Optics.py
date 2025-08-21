# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
import os
import gc
try:
    import cupy as cp
except ImportError:
    cp = None

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class optics:

    # -----------------------------------------------------------------------------
    # Functions
    # -----------------------------------------------------------------------------
    ## Initialization
    def __init__(self, directory=None):
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

    @property
    def components(self):
        """
        Return the internal list of components.
        """
        return self._components