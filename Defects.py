# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
try:
    from scipy.special import k0 as _scipy_k0, k1 as _scipy_k1
except ImportError:
    _scipy_k0 = None
    _scipy_k1 = None
try:
    from cupyx.scipy.special import k0 as _cupy_k0, k1 as _cupy_k1
except Exception:
    _cupy_k0 = None
    _cupy_k1 = None
import json
import re
import os
from Logging import logging
import hardware


# -----------------------------------------------------------------------------
# Module helpers
# -----------------------------------------------------------------------------
def _same_directory(a, b):
    """True if two directory paths point at the same location."""
    if a is None or b is None:
        return a is None and b is None
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def _writes_in_place(sample, directory):
    """
    True if writing chunk files to `directory` overwrites the files the sample
    object reads back (None or the sample's own directory). Only then does an
    applied-modification record on the sample make sense.
    """
    return directory is None or _same_directory(directory, getattr(sample, "directory", None))


def _jsonable(v):
    """Convert NumPy scalars and arrays inside a params dict to plain Python."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _check_modification(sample, directory, kind, params, force):
    """
    Raise RuntimeError if the sample already carries an in-place record of
    `kind` with the same `params` and `force` is False.
    """
    if force or not _writes_in_place(sample, directory):
        return
    if sample.has_modification(kind, _jsonable(params)):
        raise RuntimeError(
            f"{kind} with these parameters was already applied to the sample in "
            f"'{sample.directory}'. Pass force=True to apply it again.")


def _record_modification(sample, directory, kind, params):
    """Record an in-place modification on the sample; no-op for other directories."""
    if _writes_in_place(sample, directory):
        sample.record_modification(kind, _jsonable(params))


def _convex_hull_inside_mask(positions, equations, tol=1e-12):
    """
    Mask of positions inside a convex hull from its facet equations
    [a, b, c, d] (a*x + b*y + c*z + d <= tol inside). Facets are tested one
    at a time in float32, so no (facets x N) matrix is formed. Accepts NumPy
    or CuPy positions and returns the mask on the same device.
    """
    xp = cp if (cp is not None and isinstance(positions, cp.ndarray)) else np
    pos = positions if positions.dtype == xp.float32 else positions.astype(xp.float32)
    eq = xp.asarray(equations, dtype=xp.float32)
    inside = xp.ones(pos.shape[0], dtype=bool)
    for k in range(int(eq.shape[0])):
        inside &= (pos @ eq[k, :3] + eq[k, 3]) <= tol
    return inside


# -----------------------------------------------------------------------------
# Spectral solver helpers (Bertin 2019 framework, displacement output)
# -----------------------------------------------------------------------------
def _cai_kernel_fourier(K, a, xp=np):
    """
    Fourier transform of the Cai et al. (2006) spreading function
    w(r, a) = 15 / (8 pi a^3) (1 + r^2/a^2)^(-7/2), closed form
    w_hat(k, a) = ((a k)^2 / 2) K_2(a k), K_2(x) = K_0(x) + (2/x) K_1(x);
    series 1 - x^2/4 below x = 1e-3, value 1 at K = 0. One convolution with
    w turns |r| into R_a = sqrt(r^2 + a^2) and 1/r into 1/R_a + a^2/(2 R_a^3),
    so fields spread once with w are the R_a closed forms of the analytic
    segment terms (Bertin 2019).

    Args:
        K: Array of |k| on the spectral grid.
        a: Spreading radius.
        xp: numpy or cupy.

    Returns:
        Array of K.shape, float64, values in [0, 1].
    """
    if xp is np:
        if _scipy_k0 is None or _scipy_k1 is None:
            raise ImportError("scipy.special.k0/k1 are required for the Cai kernel")
        k0_fn, k1_fn = _scipy_k0, _scipy_k1
    else:
        if _cupy_k0 is None or _cupy_k1 is None:
            raise ImportError("cupyx.scipy.special.k0/k1 are required for the Cai kernel on the GPU")
        k0_fn, k1_fn = _cupy_k0, _cupy_k1

    x = xp.asarray(K, dtype=xp.float64) * float(a)
    small = x < 1.0e-3
    xb = xp.where(small, xp.ones_like(x), x)
    big = 0.5 * xb * xb * k0_fn(xb) + xb * k1_fn(xb)
    return xp.where(small, 1.0 - 0.25 * x * x, big)


def _isotropic_stiffness_tensor(mu, nu):
    """
    (3,3,3,3) isotropic stiffness
    C_ijkl = lam d_ij d_kl + mu (d_ik d_jl + d_il d_jk), lam = 2 mu nu / (1 - 2 nu).
    """
    mu_ = float(mu)
    nu_ = float(nu)
    if not (0.0 < nu_ < 0.5):
        raise ValueError("nu must be in (0, 0.5)")
    lam = 2.0 * mu_ * nu_ / (1.0 - 2.0 * nu_)
    I3 = np.eye(3, dtype=np.float64)
    return (lam * np.einsum("ij,kl->ijkl", I3, I3)
            + mu_ * np.einsum("ik,jl->ijkl", I3, I3)
            + mu_ * np.einsum("il,jk->ijkl", I3, I3))


def _voigt_to_tensor(C_voigt):
    """(6,6) Voigt matrix -> (3,3,3,3) tensor; 1<->11, 2<->22, 3<->33, 4<->23, 5<->13, 6<->12."""
    Cv = np.asarray(C_voigt, dtype=np.float64)
    if Cv.shape != (6, 6):
        raise ValueError("Voigt stiffness must be (6, 6)")
    pairs = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    C = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for I in range(6):
        i, j = pairs[I]
        for J in range(6):
            k, l = pairs[J]
            v = Cv[I, J]
            C[i, j, k, l] = v
            C[j, i, k, l] = v
            C[i, j, l, k] = v
            C[j, i, l, k] = v
    return C


def _cubic_stiffness_tensor(c11, c12, c44):
    """(3,3,3,3) cubic stiffness from (c11, c12, c44)."""
    Cv = np.zeros((6, 6), dtype=np.float64)
    Cv[0, 0] = Cv[1, 1] = Cv[2, 2] = float(c11)
    Cv[0, 1] = Cv[0, 2] = Cv[1, 0] = Cv[1, 2] = Cv[2, 0] = Cv[2, 1] = float(c12)
    Cv[3, 3] = Cv[4, 4] = Cv[5, 5] = float(c44)
    return _voigt_to_tensor(Cv)


def _resolve_stiffness(stiffness, mu, nu):
    """
    `stiffness` argument of generate_nodal_field -> (3,3,3,3) tensor.
    None -> isotropic (mu, nu); {"isotropic": (mu, nu)}; {"cubic": (c11, c12, c44)};
    (6,6) Voigt; (3,3,3,3) tensor (symmetries checked).
    """
    if stiffness is None:
        if mu is None:
            raise ValueError("mu is required when stiffness is None")
        return _isotropic_stiffness_tensor(mu, nu)
    if isinstance(stiffness, dict):
        if "isotropic" in stiffness:
            mu_iso, nu_iso = stiffness["isotropic"]
            return _isotropic_stiffness_tensor(mu_iso, nu_iso)
        if "cubic" in stiffness:
            c11, c12, c44 = stiffness["cubic"]
            return _cubic_stiffness_tensor(c11, c12, c44)
        raise ValueError("stiffness dict must have key 'isotropic' or 'cubic'")
    arr = np.asarray(stiffness, dtype=np.float64)
    if arr.shape == (6, 6):
        return _voigt_to_tensor(arr)
    if arr.shape == (3, 3, 3, 3):
        if not np.allclose(arr, arr.transpose(1, 0, 2, 3), atol=1e-6):
            raise ValueError("stiffness violates minor symmetry C_ijkl = C_jikl")
        if not np.allclose(arr, arr.transpose(0, 1, 3, 2), atol=1e-6):
            raise ValueError("stiffness violates minor symmetry C_ijkl = C_ijlk")
        if not np.allclose(arr, arr.transpose(2, 3, 0, 1), atol=1e-6):
            raise ValueError("stiffness violates major symmetry C_ijkl = C_klij")
        return arr.copy()
    raise ValueError("stiffness must be None, a dict, a (6,6) Voigt matrix, or a (3,3,3,3) tensor")


def _cut_subdivisions(S0, S1, C, spacing, max_subdiv=1024):
    """Per-triangle subdivision count n = ceil(longest edge / spacing), clipped to [1, max_subdiv]."""
    e = np.maximum.reduce([np.linalg.norm(S1 - S0, axis=1),
                           np.linalg.norm(C - S0, axis=1),
                           np.linalg.norm(C - S1, axis=1)])
    n = np.ceil(e / float(spacing)).astype(np.int64)
    return np.clip(n, 1, int(max_subdiv))


def _cut_surface_samples(S0, S1, C, B, n_sub, xp=np, chunk_samples=4_000_000):
    """
    Yield (points (m, 3), weights (m, 9)) sampling the fan triangles
    (S0, S1, C) of the cut surface. Triangle t is split into n_sub[t]^2
    congruent sub-triangles; each contributes its centroid carrying the
    plastic distortion -b (x) N / n^2, N = (S1 - S0) x (C - S0) / 2 the area
    vector, so the weights of a triangle sum to -b (x) N. Arrays are on `xp`.
    """
    S0 = xp.asarray(S0, dtype=xp.float64)
    S1 = xp.asarray(S1, dtype=xp.float64)
    C = xp.asarray(C, dtype=xp.float64)
    B = xp.asarray(B, dtype=xp.float64)
    E1 = S1 - S0
    E2 = C - S0
    Nvec = 0.5 * xp.cross(E1, E2)
    W = -(B[:, :, None] * Nvec[:, None, :]).reshape(-1, 9)
    n_sub = np.asarray(n_sub, dtype=np.int64)
    for n in np.unique(n_sub):
        n = int(n)
        idx_all = np.nonzero(n_sub == n)[0]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        up = (i + j) <= n - 1
        dn = (i + j) <= n - 2
        u = xp.asarray(np.concatenate([i[up] + 1.0 / 3.0, i[dn] + 2.0 / 3.0]) / n)
        v = xp.asarray(np.concatenate([j[up] + 1.0 / 3.0, j[dn] + 2.0 / 3.0]) / n)
        per = n * n
        step = max(1, int(chunk_samples) // per)
        for c0 in range(0, idx_all.size, step):
            idx = xp.asarray(idx_all[c0:c0 + step])
            m = int(idx.shape[0])
            pts = (S0[idx][:, None, :] + u[None, :, None] * E1[idx][:, None, :]
                   + v[None, :, None] * E2[idx][:, None, :])
            w = xp.broadcast_to((W[idx] / per)[:, None, :], (m, per, 9))
            yield pts.reshape(-1, 3), xp.ascontiguousarray(w.reshape(-1, 9))


def _deposit_cic(grid, points, weights, origin, spacing, shape, xp=np, gpu_kernel=None):
    """
    Trilinear (cloud-in-cell) deposition of weighted points onto a periodic,
    cell-centred grid. `grid` is (nx*ny*nz, 9) with flat index
    (ix*ny + iy)*nz + iz; node ix sits at origin_x + (ix + 0.5) dx. Adds the
    weights in place; divide by the cell volume afterwards for a density.
    """
    nx, ny, nz = [int(s) for s in shape]
    Np = int(points.shape[0])
    if Np == 0:
        return
    if xp is not np:
        if gpu_kernel is None:
            raise RuntimeError("deposit_cic kernel is required on the GPU")
        threads = 256
        gpu_kernel(((Np + threads - 1) // threads,), (threads,),
                   (xp.ascontiguousarray(xp.asarray(points, dtype=xp.float32)),
                    xp.ascontiguousarray(xp.asarray(weights, dtype=xp.float32)), np.int32(Np),
                    np.float32(origin[0]), np.float32(origin[1]), np.float32(origin[2]),
                    np.float32(1.0 / spacing[0]), np.float32(1.0 / spacing[1]), np.float32(1.0 / spacing[2]),
                    np.int32(nx), np.int32(ny), np.int32(nz), grid))
        return
    P = np.asarray(points, dtype=np.float64)
    Wt = np.asarray(weights, dtype=np.float64)
    g = (P - np.asarray(origin, dtype=np.float64)) / np.asarray(spacing, dtype=np.float64) - 0.5
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    n_arr = np.array([nx, ny, nz], dtype=np.int64)
    ntot = nx * ny * nz
    for c in range(8):
        sel = np.array([c & 1, (c >> 1) & 1, (c >> 2) & 1], dtype=np.int64)
        idx3 = np.mod(i0 + sel, n_arr)
        w = np.prod(np.where(sel[None, :] == 1, f, 1.0 - f), axis=1)
        flat = (idx3[:, 0] * ny + idx3[:, 1]) * nz + idx3[:, 2]
        for m in range(9):
            grid[:, m] += np.bincount(flat, weights=w * Wt[:, m], minlength=ntot).astype(np.float32)


def _spectral_displacement(beta_p, spacing, C_stiff, a_grid, xp=np, deconvolve_cic=False,
                           slab_bytes=192 << 20):
    """
    Periodic displacement of a plastic distortion field (eigenstrain problem)
    div C:(grad u - beta_p) = 0, solved per Fourier mode as
        A_mk(k) u_k = -i k_j C_mjkl beta_p_kl,   A_mk = C_mjkl k_j k_l,
    after spreading beta_p with the Cai kernel w(k, a_grid). The Nye tensor
    is the curl of the spread beta_p, i.e. the Bertin (2019) non-singular
    alpha, and grad u - beta_p its elastic distortion (incompatible plus
    compatible part). The k = 0 mode is zero. C is normalised by its largest
    entry (the displacement is invariant) and the 3x3 solve uses the adjugate.

    Args:
        beta_p: (nx, ny, nz, 3, 3) real array on `xp`.
        spacing: (dx, dy, dz).
        C_stiff: (3, 3, 3, 3) stiffness.
        a_grid: Spreading radius.
        xp: numpy or cupy.
        deconvolve_cic: Divide by the trilinear transfer function
            prod sinc^2(k_i d_i / 2) so only the Cai kernel spreads the field.
        slab_bytes: Approximate working set per kx slab.

    Returns:
        (nx, ny, nz, 3) float32 displacement on `xp`.
    """
    nx, ny, nz = [int(s) for s in beta_p.shape[:3]]
    dx, dy, dz = [float(s) for s in spacing]
    Bk = xp.fft.fftn(xp.asarray(beta_p, dtype=xp.float32), axes=(0, 1, 2)).astype(xp.complex64)
    Cn = np.asarray(C_stiff, dtype=np.float64)
    Cx = xp.asarray(Cn / np.abs(Cn).max(), dtype=xp.float32)
    kx = (2.0 * np.pi * xp.fft.fftfreq(nx, d=dx)).astype(xp.float32)
    ky = (2.0 * np.pi * xp.fft.fftfreq(ny, d=dy)).astype(xp.float32)
    kz = (2.0 * np.pi * xp.fft.fftfreq(nz, d=dz)).astype(xp.float32)
    Uhat = xp.empty((nx, ny, nz, 3), dtype=xp.complex64)
    per_plane = ny * nz * (9 * 8 + 9 * 4 + 3 * 8 + 3 * 4 + 12 * 4 + 16)
    slab = max(1, int(slab_bytes) // max(1, per_plane))
    for x0 in range(0, nx, slab):
        x1 = min(nx, x0 + slab)
        KX, KY, KZ = xp.meshgrid(kx[x0:x1], ky, kz, indexing="ij")
        kvec = xp.stack([KX, KY, KZ], axis=-1)
        K = xp.sqrt(KX * KX + KY * KY + KZ * KZ)
        Phi = _cai_kernel_fourier(K, a_grid, xp=xp)
        if deconvolve_cic:
            for Kc, d in ((KX, dx), (KY, dy), (KZ, dz)):
                arg = 0.5 * Kc.astype(xp.float64) * d
                s = xp.where(xp.abs(arg) < 1.0e-12, xp.ones_like(arg), xp.sin(arg) / xp.where(arg == 0, 1.0, arg))
                Phi = Phi / (s * s)
        Phi = Phi.astype(xp.float32)
        Bs = Bk[x0:x1] * Phi[..., None, None]
        rhs = -1j * xp.einsum("mjkl,xyzj,xyzkl->xyzm", Cx, kvec, Bs)
        A = xp.einsum("mjkl,xyzj,xyzl->xyzmk", Cx, kvec, kvec)
        a00, a01, a02 = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
        a10, a11, a12 = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
        a20, a21, a22 = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
        m00 = a11 * a22 - a12 * a21; m01 = a02 * a21 - a01 * a22; m02 = a01 * a12 - a02 * a11
        m10 = a12 * a20 - a10 * a22; m11 = a00 * a22 - a02 * a20; m12 = a02 * a10 - a00 * a12
        m20 = a10 * a21 - a11 * a20; m21 = a01 * a20 - a00 * a21; m22 = a00 * a11 - a01 * a10
        det = a00 * m00 + a01 * m10 + a02 * m20
        if x0 == 0:
            det[0, 0, 0] = 1.0
        inv_det = 1.0 / det
        r0, r1, r2 = rhs[..., 0], rhs[..., 1], rhs[..., 2]
        Uhat[x0:x1, ..., 0] = (m00 * r0 + m01 * r1 + m02 * r2) * inv_det
        Uhat[x0:x1, ..., 1] = (m10 * r0 + m11 * r1 + m12 * r2) * inv_det
        Uhat[x0:x1, ..., 2] = (m20 * r0 + m21 * r1 + m22 * r2) * inv_det
        del KX, KY, KZ, kvec, K, Phi, Bs, rhs, A
    Uhat[0, 0, 0, :] = 0.0
    del Bk
    return xp.real(xp.fft.ifftn(Uhat, axes=(0, 1, 2))).astype(xp.float32)


# CUDA module for the dislocation fields. Segment records are 12 doubles:
# S0, S1, C, b.
#   dislocation_displacement: isotropic Volterra displacement per point. Each
#     segment contributes the solid angle of the triangle (S0, S1, C) times
#     b/(4 pi) and the Burgers formula elastic terms along S0->S1 with
#     |r| -> sqrt(r^2 + a^2). The solid angle is evaluated in double because
#     its atan2 arguments cancel near the plane of a large triangle; the
#     elastic terms use float32 on vertex differences formed in double.
#   dislocation_near_field_delta: Bertin 2019 near-field split. Elastic terms
#     at a_phys minus at a_grid for segments with |P - mid| <= rcut + L/2, and
#     the sharp solid angle minus the a_grid-spread one for cut triangles
#     within rcut of P. The spread double layer follows from
#     w~ * (1/r) = 1/R_a + a^2 / (2 R_a^3), R_a = sqrt(r^2 + a^2), giving
#     Omega_a = |h| [G(H) - a^2 G'(H) / (2H)], G(s) = Omega(s)/s, where h is the
#     height of P over the triangle plane, H = sqrt(h^2 + a^2) and Omega(s) the
#     solid angle seen from height s above the foot point; G' is a central
#     difference.
#   deposit_cic: trilinear deposition of weighted points onto a periodic
#     cell-centred (nx, ny, nz, 9) grid with atomics.
_DISLOCATION_DISPLACEMENT_KERNEL = r'''
#define SEG_TILE 128

// Signed solid angle of the triangle with vertex vectors R1, R2, R3 from the
// observation point (Van Oosterom and Strackee).
__device__ __forceinline__ double solid_angle(
    double R1x, double R1y, double R1z, double R2x, double R2y, double R2z,
    double R3x, double R3y, double R3z)
{
    double n1 = sqrt(R1x*R1x + R1y*R1y + R1z*R1z);
    double n2 = sqrt(R2x*R2x + R2y*R2y + R2z*R2z);
    double n3 = sqrt(R3x*R3x + R3y*R3y + R3z*R3z);
    double cx = R2y*R3z - R2z*R3y;
    double cy = R2z*R3x - R2x*R3z;
    double cz = R2x*R3y - R2y*R3x;
    double num = R1x*cx + R1y*cy + R1z*cz;
    double den = n1*n2*n3
               + (R1x*R2x + R1y*R2y + R1z*R2z) * n3
               + (R1x*R3x + R1y*R3y + R1z*R3z) * n2
               + (R2x*R3x + R2y*R3y + R2z*R3z) * n1;
    return 2.0 * atan2(num, den);
}

__device__ __forceinline__ double seg_dist2(
    double fx, double fy, double fz, const double* a, const double* b)
{
    double ex = b[0]-a[0], ey = b[1]-a[1], ez = b[2]-a[2];
    double L2 = ex*ex + ey*ey + ez*ez;
    double t = 0.0;
    if (L2 > 1.0e-30) {
        t = ((fx-a[0])*ex + (fy-a[1])*ey + (fz-a[2])*ez) / L2;
        t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    }
    double dx = fx - (a[0] + t*ex), dy = fy - (a[1] + t*ey), dz = fz - (a[2] + t*ez);
    return dx*dx + dy*dy + dz*dz;
}

// Squared in-plane distance from the foot point f (in the triangle plane with
// unit normal n) to the triangle (v0, v1, v2); zero inside.
__device__ __forceinline__ double tri_dist2_inplane(
    double fx, double fy, double fz, const double* v0, const double* v1, const double* v2,
    double nx, double ny, double nz)
{
    double c[3];
    const double* va[3] = {v0, v1, v2};
    const double* vb[3] = {v1, v2, v0};
    for (int e = 0; e < 3; ++e) {
        double ex = vb[e][0]-va[e][0], ey = vb[e][1]-va[e][1], ez = vb[e][2]-va[e][2];
        double wx = fx-va[e][0], wy = fy-va[e][1], wz = fz-va[e][2];
        c[e] = (ey*wz - ez*wy)*nx + (ez*wx - ex*wz)*ny + (ex*wy - ey*wx)*nz;
    }
    if ((c[0] >= 0.0 && c[1] >= 0.0 && c[2] >= 0.0) || (c[0] <= 0.0 && c[1] <= 0.0 && c[2] <= 0.0))
        return 0.0;
    double d = seg_dist2(fx, fy, fz, v0, v1);
    double d1 = seg_dist2(fx, fy, fz, v1, v2); if (d1 < d) d = d1;
    double d2 = seg_dist2(fx, fy, fz, v2, v0); if (d2 < d) d = d2;
    return d;
}

__device__ __forceinline__ void elastic_terms(
    float r1x, float r1y, float r1z, float tx, float ty, float tz,
    float bx, float by, float bz, float a2, float c1, float c2,
    float &ux, float &uy, float &uz)
{
    float L2 = tx*tx + ty*ty + tz*tz;
    if (L2 > 1.0e-20f) {
        float L = sqrtf(L2);
        float invL = 1.0f / L;
        tx *= invL; ty *= invL; tz *= invL;
        float lam = -(r1x*tx + r1y*ty + r1z*tz);
        float dx = -r1x - lam*tx, dy = -r1y - lam*ty, dz = -r1z - lam*tz;
        float rho2 = dx*dx + dy*dy + dz*dz + a2;
        float s1 = -lam, s2 = L - lam;
        float Ra1 = sqrtf(s1*s1 + rho2);
        float Ra2 = sqrtf(s2*s2 + rho2);
        float I1, I3;
        if (s1 >= 0.0f)      I1 = logf((s2 + Ra2) / (s1 + Ra1));
        else if (s2 <= 0.0f) I1 = logf((Ra1 - s1) / (Ra2 - s2));
        else                 I1 = logf((s2 + Ra2) * (Ra1 - s1) / rho2);
        float ssum = s1 + s2;
        if (s1 * s2 > 0.0f) I3 = L * ssum / (Ra1 * Ra2 * (s2*Ra1 + s1*Ra2));
        else                I3 = (s2 / Ra2 - s1 / Ra1) / rho2;
        float J3 = L * ssum / (Ra1 * Ra2 * (Ra1 + Ra2));
        float qx = by*tz - bz*ty, qy = bz*tx - bx*tz, qz = bx*ty - by*tx;
        float bd = qx*dx + qy*dy + qz*dz;
        float g = (c1 + c2) * I1;
        float h = c2 * bd;
        ux += g*qx - h*(dx*I3 - tx*J3);
        uy += g*qy - h*(dy*I3 - ty*J3);
        uz += g*qz - h*(dz*I3 - tz*J3);
    }
}

extern "C" __global__
void dislocation_displacement(const double* __restrict__ P, const int Np,
                              const double* __restrict__ seg, const int Ns,
                              const float a2, const float c1, const float c2,
                              float* __restrict__ U)
{
    __shared__ double sh[SEG_TILE * 12];
    const float inv4pi = 0.07957747154594767f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double px = 0.0, py = 0.0, pz = 0.0;
    if (i < Np) { px = P[3*i]; py = P[3*i+1]; pz = P[3*i+2]; }
    float ux = 0.f, uy = 0.f, uz = 0.f;

    for (int base = 0; base < Ns; base += SEG_TILE) {
        int n = Ns - base; if (n > SEG_TILE) n = SEG_TILE;
        for (int k = threadIdx.x; k < n * 12; k += blockDim.x) sh[k] = seg[base * 12 + k];
        __syncthreads();
        if (i < Np) {
            for (int s = 0; s < n; ++s) {
                const double* q = sh + 12 * s;
                double R1x = q[0] - px, R1y = q[1] - py, R1z = q[2] - pz;
                double R2x = q[3] - px, R2y = q[4] - py, R2z = q[5] - pz;
                double R3x = q[6] - px, R3y = q[7] - py, R3z = q[8] - pz;
                float bx = (float)q[9], by = (float)q[10], bz = (float)q[11];

                // Solid angle of the triangle (Van Oosterom and Strackee), in double.
                float w = (float)solid_angle(R1x, R1y, R1z, R2x, R2y, R2z, R3x, R3y, R3z) * inv4pi;
                ux += w * bx; uy += w * by; uz += w * bz;
                float r1x = (float)R1x, r1y = (float)R1y, r1z = (float)R1z;

                // Elastic terms along the real segment.
                float tx = (float)(q[3] - q[0]), ty = (float)(q[4] - q[1]), tz = (float)(q[5] - q[2]);
                elastic_terms(r1x, r1y, r1z, tx, ty, tz, bx, by, bz, a2, c1, c2, ux, uy, uz);
            }
        }
        __syncthreads();
    }
    if (i < Np) { U[3*i] = ux; U[3*i+1] = uy; U[3*i+2] = uz; }
}

extern "C" __global__
void dislocation_near_field_delta(const double* __restrict__ P, const int Np,
                                  const double* __restrict__ seg, const int Ns,
                                  const float a2_phys, const float a2_grid, const float rcut,
                                  const float c1, const float c2,
                                  float* __restrict__ U)
{
    __shared__ double sh[SEG_TILE * 12];
    const float inv4pi = 0.07957747154594767f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double px = 0.0, py = 0.0, pz = 0.0;
    if (i < Np) { px = P[3*i]; py = P[3*i+1]; pz = P[3*i+2]; }
    float ux = 0.f, uy = 0.f, uz = 0.f;
    const double rc2 = (double)rcut * (double)rcut;
    const double ag2 = (double)a2_grid;

    for (int base = 0; base < Ns; base += SEG_TILE) {
        int n = Ns - base; if (n > SEG_TILE) n = SEG_TILE;
        for (int k = threadIdx.x; k < n * 12; k += blockDim.x) sh[k] = seg[base * 12 + k];
        __syncthreads();
        if (i < Np) {
            for (int s = 0; s < n; ++s) {
                const double* q = sh + 12 * s;
                float bx = (float)q[9], by = (float)q[10], bz = (float)q[11];

                // Elastic terms: a_phys minus a_grid within rcut + L/2 of the midpoint.
                float tx = (float)(q[3] - q[0]), ty = (float)(q[4] - q[1]), tz = (float)(q[5] - q[2]);
                float L2 = tx*tx + ty*ty + tz*tz;
                float mx = (float)(0.5 * (q[0] + q[3]) - px);
                float my = (float)(0.5 * (q[1] + q[4]) - py);
                float mz = (float)(0.5 * (q[2] + q[5]) - pz);
                float rad = rcut + 0.5f * sqrtf(L2);
                if (mx*mx + my*my + mz*mz <= rad*rad) {
                    float r1x = (float)(q[0] - px), r1y = (float)(q[1] - py), r1z = (float)(q[2] - pz);
                    float vx = 0.f, vy = 0.f, vz = 0.f;
                    elastic_terms(r1x, r1y, r1z, tx, ty, tz, bx, by, bz, a2_phys, c1, c2, ux, uy, uz);
                    elastic_terms(r1x, r1y, r1z, tx, ty, tz, bx, by, bz, a2_grid, c1, c2, vx, vy, vz);
                    ux -= vx; uy -= vy; uz -= vz;
                }

                // Solid angle: sharp minus a_grid-spread within rcut of the triangle.
                double e1x = q[3]-q[0], e1y = q[4]-q[1], e1z = q[5]-q[2];
                double e2x = q[6]-q[0], e2y = q[7]-q[1], e2z = q[8]-q[2];
                double nx = e1y*e2z - e1z*e2y, ny = e1z*e2x - e1x*e2z, nz = e1x*e2y - e1y*e2x;
                double A2 = sqrt(nx*nx + ny*ny + nz*nz);
                if (A2 <= 1.0e-20) continue;
                nx /= A2; ny /= A2; nz /= A2;
                double h = (px-q[0])*nx + (py-q[1])*ny + (pz-q[2])*nz;
                if (h*h > rc2) continue;
                double fx = px - h*nx, fy = py - h*ny, fz = pz - h*nz;
                if (tri_dist2_inplane(fx, fy, fz, q, q+3, q+6, nx, ny, nz) + h*h > rc2) continue;
                double om_sharp = solid_angle(q[0]-px, q[1]-py, q[2]-pz,
                                              q[3]-px, q[4]-py, q[5]-pz,
                                              q[6]-px, q[7]-py, q[8]-pz);
                double om_a = 0.0;
                if (h != 0.0) {
                    double sgn = h > 0.0 ? 1.0 : -1.0;
                    double H = sqrt(h*h + ag2);
                    double eps = 0.1 * H;
                    double G[3];
                    const double sv[3] = {H, H + eps, H - eps};
                    for (int m = 0; m < 3; ++m) {
                        double yx = fx + sgn*sv[m]*nx, yy = fy + sgn*sv[m]*ny, yz = fz + sgn*sv[m]*nz;
                        G[m] = solid_angle(q[0]-yx, q[1]-yy, q[2]-yz,
                                           q[3]-yx, q[4]-yy, q[5]-yz,
                                           q[6]-yx, q[7]-yy, q[8]-yz) / sv[m];
                    }
                    om_a = fabs(h) * (G[0] - ag2 * (G[1] - G[2]) / (2.0*eps) / (2.0*H));
                }
                float w = (float)(om_sharp - om_a) * inv4pi;
                ux += w * bx; uy += w * by; uz += w * bz;
            }
        }
        __syncthreads();
    }
    if (i < Np) { U[3*i] = ux; U[3*i+1] = uy; U[3*i+2] = uz; }
}

extern "C" __global__
void deposit_cic(const float* __restrict__ pts, const float* __restrict__ w, const int Np,
                 const float ox, const float oy, const float oz,
                 const float inv_dx, const float inv_dy, const float inv_dz,
                 const int nx, const int ny, const int nz,
                 float* __restrict__ grid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Np) return;
    float gx = (pts[3*i]   - ox) * inv_dx - 0.5f;
    float gy = (pts[3*i+1] - oy) * inv_dy - 0.5f;
    float gz = (pts[3*i+2] - oz) * inv_dz - 0.5f;
    int ix = (int)floorf(gx), iy = (int)floorf(gy), iz = (int)floorf(gz);
    float fx = gx - ix, fy = gy - iy, fz = gz - iz;
    for (int c = 0; c < 8; ++c) {
        int sx = c & 1, sy = (c >> 1) & 1, sz = (c >> 2) & 1;
        int jx = (ix + sx) % nx; if (jx < 0) jx += nx;
        int jy = (iy + sy) % ny; if (jy < 0) jy += ny;
        int jz = (iz + sz) % nz; if (jz < 0) jz += nz;
        float wt = (sx ? fx : 1.f - fx) * (sy ? fy : 1.f - fy) * (sz ? fz : 1.f - fz);
        size_t cell = ((size_t)jx * ny + jy) * nz + jz;
        for (int m = 0; m < 9; ++m) atomicAdd(&grid[cell * 9 + m], wt * w[9*i + m]);
    }
}
'''


def _solid_angle_numpy(R1, R2, R3):
    """Signed solid angle (Van Oosterom and Strackee) of triangles from vertex vectors (..., 3)."""
    n1 = np.sqrt(np.einsum("...k,...k->...", R1, R1))
    n2 = np.sqrt(np.einsum("...k,...k->...", R2, R2))
    n3 = np.sqrt(np.einsum("...k,...k->...", R3, R3))
    num = np.einsum("...k,...k->...", R1, np.cross(R2, R3))
    den = (n1 * n2 * n3
           + np.einsum("...k,...k->...", R1, R2) * n3
           + np.einsum("...k,...k->...", R1, R3) * n2
           + np.einsum("...k,...k->...", R2, R3) * n1)
    return 2.0 * np.arctan2(num, den)


def _segment_dist2_numpy(F, a, b):
    """Squared distance from points F (P, 3) to segment a-b."""
    e = b - a
    L2 = float(e @ e)
    t = np.zeros(F.shape[0]) if L2 <= 1.0e-30 else np.clip(((F - a) @ e) / L2, 0.0, 1.0)
    d = F - (a + t[:, None] * e)
    return np.einsum("pk,pk->p", d, d)


def _solid_angle_delta_numpy(P, v0, v1, v2, a_grid, r_cut):
    """
    Sharp minus a_grid-spread solid angle of triangle (v0, v1, v2) at points
    P (N, 3) lying within r_cut of the triangle (others get 0). See the CUDA
    module comment for the spread formula.
    """
    out = np.zeros(P.shape[0], dtype=np.float64)
    n = np.cross(v1 - v0, v2 - v0)
    A2 = np.linalg.norm(n)
    if A2 <= 1.0e-20:
        return out
    n = n / A2
    h = (P - v0) @ n
    near = np.abs(h) <= r_cut
    if not np.any(near):
        return out
    idx = np.nonzero(near)[0]
    Ph = P[idx]; hh = h[idx]
    F = Ph - hh[:, None] * n
    # in-plane distance to the triangle
    c = np.stack([np.cross(v1 - v0, F - v0) @ n, np.cross(v2 - v1, F - v1) @ n,
                  np.cross(v0 - v2, F - v2) @ n], axis=1)
    inside = np.all(c >= 0.0, axis=1) | np.all(c <= 0.0, axis=1)
    d2 = np.minimum.reduce([_segment_dist2_numpy(F, v0, v1), _segment_dist2_numpy(F, v1, v2),
                            _segment_dist2_numpy(F, v2, v0)])
    d2[inside] = 0.0
    keep = d2 + hh * hh <= r_cut * r_cut
    if not np.any(keep):
        return out
    idx = idx[keep]; Ph = Ph[keep]; hh = hh[keep]; F = F[keep]
    om_sharp = _solid_angle_numpy(v0 - Ph, v1 - Ph, v2 - Ph)
    sgn = np.where(hh >= 0.0, 1.0, -1.0)
    H = np.sqrt(hh * hh + a_grid * a_grid)
    eps = 0.1 * H
    G = []
    for s in (H, H + eps, H - eps):
        Y = F + (sgn * s)[:, None] * n
        G.append(_solid_angle_numpy(v0 - Y, v1 - Y, v2 - Y) / s)
    om_a = np.abs(hh) * (G[0] - a_grid * a_grid * (G[1] - G[2]) / (2.0 * eps) / (2.0 * H))
    om_a[hh == 0.0] = 0.0
    out[idx] = om_sharp - om_a
    return out


def _elastic_terms_numpy(rel, t, Ls, ok, bxt, a2, c1, c2):
    """
    Burgers-formula elastic terms of straight segments, |r| -> sqrt(r^2 + a^2)
    (same formula as the CUDA `elastic_terms`). `rel` (P, S, 3) = point - S0,
    `t` (S, 3) unit tangents, `Ls` (S,) lengths, `ok` (S,) valid segments,
    `bxt` (S, 3) = b x t. Returns (P, 3) float64.
    """
    lam = np.einsum("psk,sk->ps", rel, t)
    d = rel - lam[:, :, None] * t[None]
    rho2 = np.einsum("psk,psk->ps", d, d) + a2
    sa = -lam
    sb = Ls[None, :] - lam
    Ra1 = np.sqrt(sa * sa + rho2)
    Ra2 = np.sqrt(sb * sb + rho2)
    with np.errstate(divide="ignore", invalid="ignore"):
        I1 = np.where(sa >= 0.0, np.log((sb + Ra2) / (sa + Ra1)),
                      np.where(sb <= 0.0, np.log((Ra1 - sa) / (Ra2 - sb)),
                               np.log((sb + Ra2) * (Ra1 - sa) / rho2)))
        ssum = sa + sb
        I3 = np.where(sa * sb > 0.0,
                      Ls[None, :] * ssum / (Ra1 * Ra2 * (sb * Ra1 + sa * Ra2)),
                      (sb / Ra2 - sa / Ra1) / rho2)
    J3 = Ls[None, :] * ssum / (Ra1 * Ra2 * (Ra1 + Ra2))
    bad = ~ok
    if np.any(bad):
        I1[:, bad] = 0.0; I3[:, bad] = 0.0; J3[:, bad] = 0.0
    bd = np.einsum("psk,sk->ps", d, bxt)
    acc = (c1 + c2) * (I1 @ bxt)
    acc -= c2 * np.einsum("ps,psk->pk", bd * I3, d)
    acc += c2 * ((bd * J3) @ t)
    return acc


def _dislocation_displacement_numpy(P, seg, a, nu, tile_points=2048, tile_segments=256):
    """
    NumPy evaluation of the segment displacement (see the CUDA kernel for the
    formula). `P` is (N, 3), `seg` is (M, 12) [S0, S1, C, b]; float64 tiles of
    points x segments are formed and summed. Returns (N, 3) float64.
    """
    P = np.asarray(P, dtype=np.float64)
    seg = np.asarray(seg, dtype=np.float64)
    N = P.shape[0]
    U = np.zeros((N, 3), dtype=np.float64)
    a2 = float(a) * float(a)
    c1 = -1.0 / (4.0 * np.pi)
    c2 = 1.0 / (8.0 * np.pi * (1.0 - float(nu)))
    inv4pi = 1.0 / (4.0 * np.pi)

    S0 = seg[:, 0:3]; S1 = seg[:, 3:6]; C = seg[:, 6:9]; B = seg[:, 9:12]
    Lvec = S1 - S0
    L = np.linalg.norm(Lvec, axis=1)
    ok = L > 1.0e-10
    T = np.zeros_like(Lvec)
    T[ok] = Lvec[ok] / L[ok, None]
    BxT = np.cross(B, T)

    for p0 in range(0, N, int(tile_points)):
        p1 = min(N, p0 + int(tile_points))
        Pt = P[p0:p1]
        acc = np.zeros((p1 - p0, 3), dtype=np.float64)
        for s0 in range(0, seg.shape[0], int(tile_segments)):
            s1 = min(seg.shape[0], s0 + int(tile_segments))
            r1 = S0[None, s0:s1] - Pt[:, None]
            r2 = S1[None, s0:s1] - Pt[:, None]
            r3 = C[None, s0:s1] - Pt[:, None]
            n1 = np.sqrt(np.einsum("psk,psk->ps", r1, r1))
            n2 = np.sqrt(np.einsum("psk,psk->ps", r2, r2))
            n3 = np.sqrt(np.einsum("psk,psk->ps", r3, r3))
            num = np.einsum("psk,psk->ps", r1, np.cross(r2, r3))
            den = (n1 * n2 * n3
                   + np.einsum("psk,psk->ps", r1, r2) * n3
                   + np.einsum("psk,psk->ps", r1, r3) * n2
                   + np.einsum("psk,psk->ps", r2, r3) * n1)
            omega = 2.0 * np.arctan2(num, den)
            acc += inv4pi * (omega @ B[s0:s1])
            del r2, r3, n1, n2, n3, num, den

            acc += _elastic_terms_numpy(-r1, T[s0:s1], L[s0:s1], ok[s0:s1], BxT[s0:s1], a2, c1, c2)
            del r1
        U[p0:p1] = acc
    return U


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
        "add_amorphous_band",
        "add_point_defects",
        "import_dislocation_network",
        "generate_nodal_field",
        "dislocation_displacement",
        "apply_dislocation_displacement",
        "finalize_dislocation_sample",
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
        self._amorphous_bands = None
        
    def read_defect_metadata(self, override_directory=None):
        """
        Read defect metadata from a JSON file and restore object state.

        Reads the defect metadata JSON file from disk and restores this defect
        object's state: stacking faults, cracks, amorphous bands, point
        defects (with their applied-position arrays) and the dislocation
        network stored in the `.npz` sidecar next to the JSON.

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
            # Restore applied positions from the per-chunk .npy files
            applied = pd_data.get("applied", {}) or {}
            if applied.get("chunks"):
                self._point_defects._load_applied_arrays(
                    applied.get("directory") or os.path.dirname(metadata_filename),
                    int(applied["chunks"]))

        # Restore amorphous band if present
        ab_data = defect_metadata.get("amorphous_bands", None)
        if ab_data is not None:
            self._amorphous_bands = self.amorphous_band(
                ab_data.get("directory", None),
                band_points=ab_data.get("band_points", None),
                center=ab_data.get("center", None),
                length=ab_data.get("length", None),
                width=ab_data.get("width", None),
                thickness=ab_data.get("thickness", None),
                orientation=ab_data.get("orientation", None),
                period=ab_data.get("period", None),
                n_stripes=ab_data.get("n_stripes", None),
                density_ratio=ab_data.get("density_ratio", 1.0),
                number_density=ab_data.get("number_density", None),
                seed=ab_data.get("seed", None),
            )

        # Restore the dislocation network from its .npz sidecar
        net = defect_metadata.get("dislocation_network", None)
        if net is not None:
            net_path = net.get("path", None)
            if net_path is None or not os.path.isfile(net_path):
                net_path = os.path.join(os.path.dirname(metadata_filename), net.get("filename", ""))
            if os.path.isfile(net_path):
                with np.load(net_path) as z:
                    self._opendis_nodes_xyz = z["nodes_xyz"]
                    self._opendis_segments = z["segments"]
                    self._opendis_S0 = z["S0"]
                    self._opendis_S1 = z["S1"]
                    self._opendis_bvec = z["bvec"]
                    self._opendis_tvec = z["tvec"]
                    self._opendis_mids = z["mids"]
                    self._opendis_halfL = z["halfL"]
                    self._opendis_bounds = {"min": z["bounds_min"], "max": z["bounds_max"]}
                self._opendis_source = net.get("source", None)
            else:
                self._log("normal", f"dislocation network sidecar not found: {net_path}")

        self._log("normal", f"Defect metadata read from {metadata_filename}.")
        
    ## Data Handling Functions
    def write_defect_metadata(self, override_directory=None):
        """
        Serialize the defect object's state to a JSON file on disk.

        Writes the defect history, stacking faults, cracks, amorphous bands
        and point defects to a human-readable JSON file so that the state can
        be restored later. The dislocation network arrays go to a
        `defects_dislocation_network.npz` sidecar next to the JSON (its path
        is recorded in the JSON), and applied point-defect positions are kept
        in per-chunk `.npy` files with only their counts in the JSON.

        Args:
            override_directory: Optional directory path to write to instead
                of the default self.directory.
        """
        if override_directory is not None:
            metadata_filename = os.path.join(override_directory, "defects_metadata.json")
        else:
            metadata_filename = os.path.join(self.directory, "defects_metadata.json")
        meta_dir = os.path.dirname(os.path.abspath(metadata_filename))

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
                    "directory": pd._applied_directory,
                    "chunks": len(pd._applied_vacancies),
                    "vacancies": int(sum(int(v.shape[0]) for v in pd._applied_vacancies)),
                    "substitutions": int(sum(int(v.shape[0]) for v in pd._applied_substitutions)),
                    "interstitials": int(sum(int(v.shape[0]) for v in pd._applied_interstitials)),
                }
            }
        else:
            defect_metadata["point_defects"] = None

        # Amorphous band spec
        if self._amorphous_bands is not None:
            ab = self._amorphous_bands
            defect_metadata["amorphous_bands"] = {
                "directory": ab.directory,
                "band_points": ab.band_points.tolist() if ab.band_points is not None else None,
                "center": ab.center.tolist() if ab.center is not None else None,
                "length": ab.length,
                "width": ab.width,
                "thickness": ab.thickness,
                "orientation": ab.orientation.tolist() if ab.orientation is not None else None,
                "period": ab.period,
                "n_stripes": ab.n_stripes if ab.period is not None else None,
                "density_ratio": ab.density_ratio,
                "number_density": ab.number_density,
                "seed": ab.seed,
            }
        else:
            defect_metadata["amorphous_bands"] = None

        # Dislocation network arrays as an .npz sidecar
        if getattr(self, "_opendis_S0", None) is not None:
            net_name = "defects_dislocation_network.npz"
            net_path = os.path.join(meta_dir, net_name)
            np.savez(net_path,
                     nodes_xyz=np.asarray(self._opendis_nodes_xyz),
                     segments=np.asarray(self._opendis_segments),
                     S0=np.asarray(self._opendis_S0), S1=np.asarray(self._opendis_S1),
                     bvec=np.asarray(self._opendis_bvec), tvec=np.asarray(self._opendis_tvec),
                     mids=np.asarray(self._opendis_mids), halfL=np.asarray(self._opendis_halfL),
                     bounds_min=np.asarray(self._opendis_bounds["min"]),
                     bounds_max=np.asarray(self._opendis_bounds["max"]))
            defect_metadata["dislocation_network"] = {
                "path": net_path,
                "filename": net_name,
                "source": getattr(self, "_opendis_source", None),
                "node_count": int(np.asarray(self._opendis_nodes_xyz).shape[0]),
                "segment_count": int(np.asarray(self._opendis_S0).shape[0]),
            }
        else:
            defect_metadata["dislocation_network"] = None

        # Write as nicely formatted JSON
        with open(metadata_filename, "w") as f:
            json.dump(defect_metadata, f, indent=4)
        self._log("normal", f"Defect metadata written to {metadata_filename} in JSON format.")

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

    def add_amorphous_band(self,
                           band_points=None,
                           center=None, length=None, width=None,
                           thickness=None, orientation=None,
                           period=None, n_stripes=None,
                           density_ratio=1.0,
                           number_density=None,
                           seed=None):
        """
        Add an amorphous band defect to the defect object.

        Defines a 3D region as either a convex hull from explicit boundary
        points or as an oriented slab (center, length, width, thickness,
        orientation). When applied to a sample, the crystalline atoms inside
        the region are removed and replaced with a uniform random distribution
        whose total count is set by either a relative density multiplier or
        an absolute number density.

        In periodic mode (slab only), passing `period` and `n_stripes` turns
        the single slab into a stack of `n_stripes` parallel stripes spaced
        `period` Angstroms apart along the slab normal. Each stripe has
        thickness `thickness`. The total stack span along the normal is
        `period * n_stripes`.

        Args:
            band_points: Array-like of shape (N, 3) defining the vertices of
                the band's convex hull. Mutually exclusive with the slab
                parameters.
            center: (3,) center of the slab in Angstroms. Required for slab
                mode.
            length: Slab extent along its primary in-plane axis.
            width: Slab extent along its secondary in-plane axis.
            thickness: Slab extent along its normal direction. In periodic
                mode this is the per-stripe thickness.
            orientation: (3,) slab normal vector. Defaults to [0, 0, 1].
            period: Optional centre-to-centre stripe spacing along the slab
                normal (Angstroms). Slab mode only.
            n_stripes: Number of parallel stripes when `period` is set.
            density_ratio: Multiplier on the local crystal density of atoms
                originally in the region. 1.0 preserves count, <1 removes
                atoms, >1 adds atoms. Defaults to 1.0.
            number_density: Absolute number density (atoms / Angstrom^3). If
                given, overrides density_ratio. Defaults to None.
            seed: Optional RNG seed for reproducible amorphization.

        Note:
            Exactly one of (band_points) or (center+length+width+thickness)
            must be provided. `orientation`, `period`, and `n_stripes` are
            only used in slab mode.
        """
        self._amorphous_bands = self.amorphous_band(
            self.directory, band_points=band_points,
            center=center, length=length, width=width,
            thickness=thickness, orientation=orientation,
            period=period, n_stripes=n_stripes,
            density_ratio=density_ratio,
            number_density=number_density,
            seed=seed,
        )

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

        Node tags (domain, id) are mapped to contiguous indices, so files
        with several domains or non-contiguous ids parse correctly. Burgers
        vectors are stored in the file as multiples of `burgers_magnitude`
        and are converted without renormalisation, so junction arms keep
        their larger magnitude and Frank's rule holds at every node.

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
                - _opendis_source: absolute path of the parsed file.
        """

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

        # Crystal-space -> real-space mapping (rows are the unit vectors a, b, c)
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
            node_key = (int(dom), int(node_id))
            nodes_xyz[node_key] = (float(x), float(y), float(z))
            i += 1
            arms = []
            for _ in range(int(n_arms)):
                m1 = arm_l1.match(lines[i]); m2 = arm_l2.match(lines[i+1]) if (i+1) < len(lines) else None
                if not (m1 and m2):
                    raise ValueError("Malformed arm block after node {}".format(node_key))
                dom2, nbr_id, bx, by, bz = m1.groups()
                nx, ny, nz = m2.groups()
                arms.append(((int(dom2), int(nbr_id)),
                            float(bx), float(by), float(bz),
                            float(nx), float(ny), float(nz)))
                i += 2
            arms_by_node[node_key] = arms

        # Contiguous node indices in sorted (domain, id) order
        node_keys = sorted(nodes_xyz.keys())
        node_index = {key: idx for idx, key in enumerate(node_keys)}

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
                if ni == nbr or nbr not in nodes_xyz:
                    continue
                key = (ni, nbr) if ni < nbr else (nbr, ni)
                if key in seen:
                    continue
                seen.add(key)
                pj = np.asarray(nodes_xyz[nbr], dtype=np.float64)

                # File components are multiples of |b|; keep their magnitude.
                b_crys = np.array([bx, by, bz], dtype=np.float64)
                if not np.any(b_crys):
                    continue
                b_vec = (Bmap.T @ b_crys) * bmag

                seg_keys.append((node_index[ni], node_index[nbr]))
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
            [nodes_xyz[k] for k in node_keys], dtype=dtype
        )
        self._opendis_segments = segs
        self._opendis_source = os.path.abspath(filepath)
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
        bounding box (AABB) plus an optional margin. The per-segment arrays
        are filtered directly, so minimum-image unwrapped endpoints are kept,
        and a compact node list is rebuilt from the kept endpoints.

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

        # Filter the per-segment arrays directly (keeps the unwrapped endpoints)
        seg_keep = seg_idx[keep, :]
        Bv_keep = Bv[keep, :]
        S0_new = S0[keep, :]
        S1_new = S1[keep, :]

        # Compact node list from the kept endpoints; S0 (the stored node
        # position) wins over an unwrapped S1 image at the same node.
        used_nodes = np.unique(seg_keep.ravel())
        remap = -np.ones((max(int(nodes.shape[0]), int(used_nodes.max()) + 1),), dtype=np.int64)
        remap[used_nodes] = np.arange(used_nodes.size, dtype=np.int64)
        seg_new = remap[seg_keep]
        nodes_new = np.zeros((used_nodes.size, 3), dtype=np.float64)
        nodes_new[seg_new[:, 1]] = S1_new
        nodes_new[seg_new[:, 0]] = S0_new

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
            rotate_matrix: Explicit 3x3 rotation matrix applied as v -> R v.
                If provided, rotate_axis and rotate_angle are ignored.
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

        # Transform positions: scale, rotate (v -> R v), then translate
        def _xform_pos(P):
            out = P * s
            if R is not None:
                out = out @ R.T
            if t is not None:
                out = out + t
            return out

        # Transform vectors: scale and rotate (no translation)
        def _xform_vec(V):
            out = V * s
            if R is not None:
                out = out @ R.T
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
                             mu,
                             nu,
                             grid_shape=(64, 64, 64),
                             bounds=None,
                             padding=0.0,
                             core_radius=5.0,
                             r_cut=None,
                             stiffness=None,
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
                             mode="direct",
                             a_grid=None,
                             reference_point=None,
                             surface_sampling=0.5,
                             deconvolve_cic=True):
        """
        Evaluate the dislocation displacement field on a regular grid and
        write it as an FE-style nodal field.

        Two solvers share the same cut surface (the fan of triangles from
        each segment to the closure point of its connected component, see
        `dislocation_displacement`), so their fields agree up to the grid
        resolution:

        * ``"direct"`` (alias ``"SR"``): the isotropic Volterra field of
          `dislocation_displacement` summed over every segment at every node
          with core radius `core_radius`. Exact; O(nodes x segments).
        * ``"LR"``: spectral solve on the periodic grid (Bertin 2019
          framework). The plastic distortion -b (x) n of the cut surface is
          deposited with trilinear cloud-in-cell weights, spread in Fourier
          space by the Cai et al. (2006) kernel of radius `a_grid`, and the
          anisotropic equilibrium equation div C:(grad u - beta_p) = 0 is
          solved per mode through the Christoffel matrix A_mk = C_mjkl k_j k_l.
          The Nye tensor is the curl of the deposited beta_p, so the elastic
          distortion grad u - beta_p is the Bertin non-singular field with
          its incompatible and compatible parts both kept, and the
          displacement carries the slip because the plastic part is supplied
          explicitly. Core radius `a_grid`; O(grid log grid).
        * ``"LR+SR"``: ``"LR"`` plus the analytic near-field correction of
          the Bertin 2019 splitting within `r_cut` of a node: the
          Burgers-formula elastic terms at `core_radius` minus the same terms
          at `a_grid` for nearby segments, and the sharp solid angle minus
          the `a_grid`-spread one for nearby cut triangles. This restores the
          sharp slip jump of the direct mode at the nodes. Anisotropic far
          field, isotropic near field.

        The spectral modes are periodic in the box, so network images
        contribute; pad the bounds by several `r_cut`. In "LR" the slip jump
        is spread over about `a_grid` around the cut; "LR+SR" and "direct"
        keep it sharp. Interpolation onto atoms smears any nodal field over
        one cell; `apply_dislocation_displacement` evaluates the direct field
        at the atoms instead.

        Args:
            mu: Shear modulus; with `nu` it builds the default isotropic
                stiffness of the spectral modes. Unused by the direct mode
                (the displacement depends on `nu` only).
            nu: Poisson's ratio in (0, 0.5).
            grid_shape: (nx, ny, nz) number of cells per axis.
            bounds: Explicit ((xmin, xmax), (ymin, ymax), (zmin, zmax)). If
                None, uses the imported network bounds plus `padding`.
            padding: Padding around the network bounds when `bounds` is None.
            core_radius: Physical core radius a (Angstrom) of the elastic
                terms, |r| -> sqrt(r^2 + a^2). Defaults to 5.0.
            r_cut: Near-core cutoff of "LR+SR". Defaults to 4 * a_grid
                (Bertin 2019 Fig. 10, about 5 % splitting error).
            stiffness: Spectral-mode elasticity: None (isotropic from mu,
                nu), ``{"isotropic": (mu, nu)}``, ``{"cubic": (c11, c12,
                c44)}``, a (6, 6) Voigt matrix or a (3, 3, 3, 3) tensor.
                Ignored by the direct mode.
            scale: Multiplier applied to the displacement field.
            write_directory: Directory for output files. If None, uses
                self.directory.
            nodes_filename: Output filename for the nodal positions.
            conn_filename: Output filename for the Tet4 connectivity.
            use_gpu: If True and CuPy is available, evaluate on the GPU.
            one_based_connectivity: If True, Tet4 indices in the returned
                `conn` start at 1; the written file is always 0-based.
            file_format: "txt", "npy", or "npz".
            float_fmt: Float format string for text output.
            chunk_rows: Number of rows per text-write chunk.
            dtype: NumPy dtype for output arrays.
            mode: "direct" (default), "SR", "LR" or "LR+SR". None means
                "direct".
            a_grid: Spectral spreading radius. Defaults to
                2 * min(dx, dy, dz).
            reference_point: Optional (3,) closure point of the cut surface,
                see `dislocation_displacement`.
            surface_sampling: Spacing of the cut-surface samples deposited on
                the grid, as a fraction of the smallest cell. Defaults to 0.5.
            deconvolve_cic: Divide the deposited field by the trilinear
                transfer function so only the Cai kernel spreads it, which
                keeps the spectral regularisation equal to `a_grid` for the
                near-field correction. Defaults to True.

        Returns:
            dict: ``Xref`` (N, 3) reference positions, ``U`` (N, 3)
            displacements, ``conn`` Tet4 connectivity, ``nodes_path``,
            ``conn_path``, ``mode``, ``a_grid`` and ``r_cut`` (the last two
            None for the direct mode).

        Notes:
            Validated on 20 A shear and prismatic loops at dx = 2 A (a_grid =
            4 A, r_cut = 16 A) against the direct mode: "LR+SR" agrees to
            about 2 % of the peak displacement beyond 2 a_grid from the cut
            and reproduces the jump at +-1 A from the cut to within 0.03 b;
            within one cell of the dislocation lines the deviation reaches
            10-15 % of the peak. Periodic images add about 1 % per 48 A of
            box half-width at this loop size. The near-field loops are
            brute force over segments per node, so "LR+SR" costs a fraction
            of the direct mode for small networks and pays off for large
            grids and networks.

        Raises:
            RuntimeError: If the dislocation network has not been imported.
            ValueError: If `mode`, `nu`, `grid_shape` or `stiffness` is
                invalid.

        References:
            Bertin, N. Int. J. Plasticity 122, 268-284 (2019).
            Cai, W. et al. J. Mech. Phys. Solids 54, 561-587 (2006).
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Call import_dislocation_network(...) first.")
        if not (0.0 < float(nu) < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        mode_key = "direct" if mode is None else str(mode).strip().lower()
        if mode_key == "sr":
            mode_key = "direct"
        if mode_key not in ("direct", "lr", "lr+sr"):
            raise ValueError('mode must be "direct" ("SR"), "LR" or "LR+SR"')
        mode_used = {"direct": "direct", "lr": "LR", "lr+sr": "LR+SR"}[mode_key]
        spectral = mode_used != "direct"
        if spectral:
            C_stiff = _resolve_stiffness(stiffness, mu, nu)
        elif stiffness is not None:
            self._log("verbose", "generate_nodal_field: stiffness is ignored by the direct mode.")

        out_dir = write_directory if write_directory is not None else (self.directory if self.directory else ".")
        os.makedirs(out_dir, exist_ok=True)

        # Bounds and cell-centred grid
        bounds_from_network = bounds is None
        if bounds is None:
            bmin = np.asarray(self._opendis_bounds["min"], dtype=np.float64)
            bmax = np.asarray(self._opendis_bounds["max"], dtype=np.float64)
            if float(padding) != 0.0:
                bmin = bmin - float(padding)
                bmax = bmax + float(padding)
            bounds = ((float(bmin[0]), float(bmax[0])),
                      (float(bmin[1]), float(bmax[1])),
                      (float(bmin[2]), float(bmax[2])))
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
        nx, ny, nz = [int(v) for v in grid_shape]
        if min(nx, ny, nz) < 2:
            raise ValueError("grid_shape must be >= 2")
        dx = float(xmax - xmin) / nx
        dy = float(ymax - ymin) / ny
        dz = float(zmax - zmin) / nz
        xs = np.linspace(xmin + 0.5*dx, xmax - 0.5*dx, nx, dtype=dtype)
        ys = np.linspace(ymin + 0.5*dy, ymax - 0.5*dy, ny, dtype=dtype)
        zs = np.linspace(zmin + 0.5*dz, zmax - 0.5*dz, nz, dtype=dtype)
        Xg, Yg, Zg = np.meshgrid(xs, ys, zs, indexing="ij")
        Xref = np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], axis=1).astype(dtype, copy=False)

        conn = self._grid_tet4(nx, ny, nz, one_based=bool(one_based_connectivity))

        # Displacement at the grid nodes
        a_grid_used = None
        r_cut_used = None
        if not spectral:
            U = self.dislocation_displacement(Xref, use_gpu=use_gpu, core_radius=core_radius,
                                              nu=nu, reference_point=reference_point)
        else:
            a_grid_used = 2.0 * min(dx, dy, dz) if a_grid is None else float(a_grid)
            r_cut_used = 4.0 * a_grid_used if r_cut is None else float(r_cut)
            if a_grid_used <= 0.0 or r_cut_used < 0.0:
                raise ValueError("a_grid must be positive and r_cut non-negative")
            if bounds_from_network and float(padding) < r_cut_used:
                self._log("normal", f"generate_nodal_field: padding {float(padding):.1f} A is below "
                                    f"r_cut {r_cut_used:.1f} A; periodic images of the network "
                                    "will affect the spectral field near the box faces.")
            U = self._spectral_grid_displacement((xmin, ymin, zmin), (dx, dy, dz), (nx, ny, nz),
                                                 C_stiff, a_grid_used, use_gpu=use_gpu,
                                                 reference_point=reference_point,
                                                 surface_sampling=surface_sampling,
                                                 deconvolve_cic=deconvolve_cic)
            if mode_used == "LR+SR":
                U = U + self._dislocation_near_field_delta(Xref, core_radius, a_grid_used, r_cut_used,
                                                           nu=nu, use_gpu=use_gpu,
                                                           reference_point=reference_point)
            self._log("normal", f"generate_nodal_field: mode {mode_used}, grid {nx}x{ny}x{nz}, "
                                f"a_grid = {a_grid_used:.3f} A, r_cut = {r_cut_used:.3f} A")
        U = np.asarray(U, dtype=dtype)
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
            "conn_path": conn_path,
            "mode": mode_used,
            "a_grid": a_grid_used,
            "r_cut": r_cut_used,
        }

    def _spectral_grid_displacement(self, origin, spacing, shape, C_stiff, a_grid, use_gpu=True,
                                    reference_point=None, surface_sampling=0.5, deconvolve_cic=True):
        """
        Spectral ("LR") displacement of the network on a periodic cell-centred
        grid: deposit the plastic distortion of the cut surface (see
        `_cut_surface_samples`, `_deposit_cic`) and solve the anisotropic
        equilibrium equation with `_spectral_displacement`.

        Args:
            origin: (xmin, ymin, zmin) of the box.
            spacing: (dx, dy, dz).
            shape: (nx, ny, nz).
            C_stiff: (3, 3, 3, 3) stiffness.
            a_grid: Spreading radius of the Cai kernel.
            use_gpu: If True and CuPy is available, run on the GPU.
            reference_point: Optional closure point of the cut surface.
            surface_sampling: Sample spacing as a fraction of the smallest cell.
            deconvolve_cic: See `_spectral_displacement`.

        Returns:
            (nx*ny*nz, 3) float32 NumPy array in the grid's "ij" order.
        """
        nx, ny, nz = [int(s) for s in shape]
        dx, dy, dz = [float(s) for s in spacing]
        seg = self._dislocation_segment_records(reference_point)
        S0, S1, C, B = seg[:, 0:3], seg[:, 3:6], seg[:, 6:9], seg[:, 9:12]

        on_gpu = bool(use_gpu) and (cp is not None)
        if on_gpu:
            try:
                on_gpu = cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                on_gpu = False
        xp = cp if on_gpu else np

        h = max(1.0e-6, float(surface_sampling)) * min(dx, dy, dz)
        n_sub = _cut_subdivisions(S0, S1, C, h)
        if np.any(n_sub >= 1024):
            self._log("normal", "generate_nodal_field: some cut triangles reach the 1024^2 sample "
                                "cap; their deposition is coarser than surface_sampling.")
        n_samples = int(np.sum(n_sub.astype(np.int64) ** 2))
        self._log("verbose", f"generate_nodal_field: depositing {n_samples} cut-surface samples "
                             f"from {seg.shape[0]} triangles")

        grid = xp.zeros((nx * ny * nz, 9), dtype=xp.float32)
        kern = self._dislocation_kernels().get_function("deposit_cic") if on_gpu else None
        for pts, w in _cut_surface_samples(S0, S1, C, B, n_sub, xp=xp):
            _deposit_cic(grid, pts, w, origin, spacing, shape, xp=xp, gpu_kernel=kern)
        grid *= np.float32(1.0 / (dx * dy * dz))
        beta = grid.reshape(nx, ny, nz, 3, 3)
        u = _spectral_displacement(beta, spacing, C_stiff, a_grid, xp=xp, deconvolve_cic=bool(deconvolve_cic))
        del beta, grid
        U = u.reshape(-1, 3)
        return cp.asnumpy(U) if on_gpu else np.asarray(U)

    @staticmethod
    def _grid_tet4(nx, ny, nz, one_based=True):
        """
        Structured Tet4 connectivity of an (nx, ny, nz) node grid: six
        tetrahedra per cell, node index i*ny*nz + j*nz + k, cells in
        C order.

        Args:
            nx, ny, nz: Number of nodes per axis.
            one_based: If True, indices start at 1.

        Returns:
            (6 * (nx-1) * (ny-1) * (nz-1), 4) int64 array.
        """
        ii, jj, kk = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
        base = (ii * ny * nz + jj * nz + kk).ravel().astype(np.int64)
        v000, v100, v010, v110 = 0, ny * nz, nz, ny * nz + nz
        v001, v101, v011, v111 = 1, ny * nz + 1, nz + 1, ny * nz + nz + 1
        offsets = np.array([
            [v000, v100, v110, v111],
            [v000, v110, v010, v111],
            [v000, v010, v011, v111],
            [v000, v011, v001, v111],
            [v000, v001, v101, v111],
            [v000, v101, v100, v111],
        ], dtype=np.int64)
        conn = (base[:, None, None] + offsets[None, :, :]).reshape(-1, 4)
        if one_based:
            conn = conn + 1
        return conn

    def _dislocation_segment_records(self, reference_point=None):
        """
        Pack the network into (M, 12) float64 records [S0, S1, C, b] where C
        is the closure point of each segment's triangle.

        Segments are grouped into connected components through the node
        connectivity in `_opendis_segments` (or, if that is unavailable, by
        coincident endpoints), and C is the centroid of the component's
        segment endpoints unless `reference_point` overrides it.

        Args:
            reference_point: Optional (3,) point used as C for every segment.

        Returns:
            (M, 12) float64 array.
        """
        S0 = np.asarray(self._opendis_S0, dtype=np.float64)
        S1 = np.asarray(self._opendis_S1, dtype=np.float64)
        Bv = np.asarray(self._opendis_bvec, dtype=np.float64)
        M = int(S0.shape[0])
        if M == 0:
            raise ValueError("No dislocation segments loaded.")

        if reference_point is not None:
            C = np.tile(np.asarray(reference_point, dtype=np.float64).reshape(1, 3), (M, 1))
        else:
            segs = getattr(self, "_opendis_segments", None)
            if segs is None or np.asarray(segs).shape != (M, 2):
                # Fall back to endpoint coincidence when node ids are missing.
                pts = np.round(np.vstack([S0, S1]), 4)
                _, inv = np.unique(pts, axis=0, return_inverse=True)
                inv = np.asarray(inv).ravel()
                segs = np.stack([inv[:M], inv[M:]], axis=1)
            segs = np.asarray(segs, dtype=np.int64)
            n_nodes = int(segs.max()) + 1
            from scipy.sparse import coo_matrix
            from scipy.sparse.csgraph import connected_components
            graph = coo_matrix((np.ones(M), (segs[:, 0], segs[:, 1])), shape=(n_nodes, n_nodes))
            n_comp, node_comp = connected_components(graph, directed=False)
            seg_comp = node_comp[segs[:, 0]]
            csum = np.zeros((n_comp, 3), dtype=np.float64)
            cnt = np.zeros(n_comp, dtype=np.float64)
            np.add.at(csum, seg_comp, S0 + S1)
            np.add.at(cnt, seg_comp, 2.0)
            C = (csum / cnt[:, None])[seg_comp]
            self._log("verbose", f"dislocation network: {n_comp} connected component(s) "
                                 f"over {M} segments")
        return np.concatenate([S0, S1, C, Bv], axis=1)

    def dislocation_displacement(self, points, use_gpu=True, core_radius=5.0, nu=0.3,
                                 reference_point=None, tile_points=None):
        """
        Isotropic Volterra displacement of the dislocation network at points.

        Each segment S0->S1 is closed into the triangle (S0, S1, C) with the
        reference point C of its connected component (Barnett 1985). Its
        contribution is b*Omega/(4 pi), with Omega the solid angle of the
        triangle, plus the Burgers-formula elastic terms integrated along the
        real segment with |r| replaced by sqrt(r^2 + a^2). Summed over a
        closed loop the fictitious edges to C cancel, and the field jumps by
        b across the fan of triangles, which is the cut surface. Components
        that are not closed (chains ending at the sample or box boundary) are
        closed through C: their cut is the fan from C and the two edges
        joining the chain ends to C carry an unscreened slip winding, so keep
        such ends away from the region of interest or pass a
        `reference_point` outside it.

        Args:
            points: (N, 3) positions in Angstrom, NumPy or CuPy.
            use_gpu: If True and CuPy is available, evaluate on the GPU
                (float32 terms from double-precision differences).
                Defaults to True.
            core_radius: Core radius a in Angstrom. Defaults to 5.0.
            nu: Poisson's ratio in (0, 0.5). Defaults to 0.3.
            reference_point: Optional (3,) closure point used for every
                component instead of the component centroids.
            tile_points: Points per evaluation tile. Defaults to 2**20 on
                the GPU and 2048 on the CPU.

        Returns:
            (N, 3) float32 displacements in Angstrom on the same array
            module as `points`.

        Raises:
            RuntimeError: If the dislocation network has not been imported.
            ValueError: If `nu` or `core_radius` is out of range or no
                segments are loaded.
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Call import_dislocation_network(...) first.")
        nu = float(nu)
        if not (0.0 < nu < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        a = float(core_radius)
        if not (a > 0.0):
            raise ValueError("core_radius must be positive")
        seg = self._dislocation_segment_records(reference_point)

        on_gpu = bool(use_gpu) and (cp is not None)
        if on_gpu:
            try:
                on_gpu = cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                on_gpu = False
        input_is_cupy = (cp is not None) and isinstance(points, cp.ndarray)

        if not on_gpu:
            P = cp.asnumpy(points) if input_is_cupy else np.asarray(points)
            tp = 2048 if tile_points is None else int(tile_points)
            U = _dislocation_displacement_numpy(P.reshape(-1, 3), seg, a, nu, tile_points=tp)
            U = U.astype(np.float32)
            return cp.asarray(U) if input_is_cupy else U

        kernel = self._dislocation_displacement_kernel()
        P = points if input_is_cupy else cp.asarray(points)
        P = P.reshape(-1, 3)
        N = int(P.shape[0])
        seg64 = cp.ascontiguousarray(cp.asarray(seg, dtype=cp.float64))
        M = int(seg64.shape[0])
        out = cp.empty((N, 3), dtype=cp.float32)
        tp = hardware.ddd_tile_points() if tile_points is None else int(tile_points)
        threads = 128
        a2 = np.float32(a * a)
        c1 = np.float32(-1.0 / (4.0 * np.pi))
        c2 = np.float32(1.0 / (8.0 * np.pi * (1.0 - nu)))
        for p0 in range(0, N, tp):
            p1 = min(N, p0 + tp)
            P64 = cp.ascontiguousarray(P[p0:p1].astype(cp.float64))
            n = p1 - p0
            kernel(((n + threads - 1) // threads,), (threads,),
                   (P64, np.int32(n), seg64, np.int32(M), a2, c1, c2, out[p0:p1]))
        return out if input_is_cupy else cp.asnumpy(out)

    def _dislocation_kernels(self):
        """Compile and cache the CUDA module holding the dislocation kernels."""
        if cp is None:
            return None
        mod = getattr(self, "_dislocation_module", None)
        if mod is None:
            mod = cp.RawModule(code=_DISLOCATION_DISPLACEMENT_KERNEL)
            self._dislocation_module = mod
        return mod

    def _dislocation_displacement_kernel(self):
        """Compiled CUDA segment-displacement kernel, or None without CuPy."""
        mod = self._dislocation_kernels()
        return None if mod is None else mod.get_function("dislocation_displacement")

    def _dislocation_near_field_delta(self, points, core_radius, a_grid, r_cut, nu=0.3,
                                      use_gpu=True, reference_point=None, tile_points=None):
        """
        Near-field correction of the spectral field (Bertin 2019 splitting).
        For every segment whose midpoint lies within r_cut + L/2 of a point,
        the Burgers-formula elastic terms with core `core_radius` minus the
        same terms with core `a_grid`; for every cut triangle within r_cut of
        a point, the sharp solid angle minus the one spread by the Cai kernel
        of radius `a_grid` (closed form through R_a, see the CUDA module
        comment), times b / 4 pi. Added to the spectral field this restores
        the sharp slip jump of `dislocation_displacement`.

        Args:
            points: (N, 3) positions, NumPy or CuPy.
            core_radius: Physical core radius a (Angstrom).
            a_grid: Spectral spreading radius (Angstrom).
            r_cut: Neighbour cutoff (Angstrom).
            nu: Poisson's ratio in (0, 0.5). Defaults to 0.3.
            use_gpu: If True and CuPy is available, evaluate on the GPU.
            reference_point: Optional closure point, see
                `dislocation_displacement`.
            tile_points: Points per GPU launch. Defaults to 2**20.

        Returns:
            (N, 3) float32 on the array module of `points`.
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Call import_dislocation_network(...) first.")
        nu = float(nu)
        if not (0.0 < nu < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        a_p = float(core_radius)
        a_g = float(a_grid)
        if not (a_p > 0.0 and a_g > 0.0):
            raise ValueError("core_radius and a_grid must be positive")
        seg = self._dislocation_segment_records(reference_point)
        c1 = -1.0 / (4.0 * np.pi)
        c2 = 1.0 / (8.0 * np.pi * (1.0 - nu))

        on_gpu = bool(use_gpu) and (cp is not None)
        if on_gpu:
            try:
                on_gpu = cp.cuda.runtime.getDeviceCount() > 0
            except Exception:
                on_gpu = False
        input_is_cupy = (cp is not None) and isinstance(points, cp.ndarray)

        if not on_gpu:
            from scipy.spatial import cKDTree
            P = (cp.asnumpy(points) if input_is_cupy else np.asarray(points))
            P = P.reshape(-1, 3).astype(np.float64)
            S0 = seg[:, 0:3]; S1 = seg[:, 3:6]; B = seg[:, 9:12]
            Lvec = S1 - S0
            L = np.linalg.norm(Lvec, axis=1)
            ok = L > 1.0e-10
            T = np.zeros_like(Lvec)
            T[ok] = Lvec[ok] / L[ok, None]
            BxT = np.cross(B, T)
            Cc = seg[:, 6:9]
            U = np.zeros((P.shape[0], 3), dtype=np.float64)
            if P.shape[0]:
                tree = cKDTree(P)
                near = tree.query_ball_point(0.5 * (S0 + S1), r=float(r_cut) + 0.5 * L)
                for s in range(seg.shape[0]):
                    idx = near[s]
                    if not ok[s] or len(idx) == 0:
                        continue
                    idx = np.asarray(idx, dtype=np.int64)
                    rel = P[idx][:, None, :] - S0[s][None, None, :]
                    args = (rel, T[s:s + 1], L[s:s + 1], ok[s:s + 1], BxT[s:s + 1])
                    U[idx] += (_elastic_terms_numpy(*args, a_p * a_p, c1, c2)
                               - _elastic_terms_numpy(*args, a_g * a_g, c1, c2))
                # solid-angle part: candidates within r_cut of the triangle's bounding sphere
                cen = (S0 + S1 + Cc) / 3.0
                rad = np.maximum.reduce([np.linalg.norm(S0 - cen, axis=1), np.linalg.norm(S1 - cen, axis=1),
                                         np.linalg.norm(Cc - cen, axis=1)]) + float(r_cut)
                near = tree.query_ball_point(cen, r=rad)
                for s in range(seg.shape[0]):
                    idx = near[s]
                    if len(idx) == 0:
                        continue
                    idx = np.asarray(idx, dtype=np.int64)
                    dom = _solid_angle_delta_numpy(P[idx], S0[s], S1[s], Cc[s], a_g, float(r_cut))
                    U[idx] += (dom / (4.0 * np.pi))[:, None] * B[s][None, :]
            U = U.astype(np.float32)
            return cp.asarray(U) if input_is_cupy else U

        kernel = self._dislocation_kernels().get_function("dislocation_near_field_delta")
        P = points if input_is_cupy else cp.asarray(points)
        P = P.reshape(-1, 3)
        N = int(P.shape[0])
        seg64 = cp.ascontiguousarray(cp.asarray(seg, dtype=cp.float64))
        M = int(seg64.shape[0])
        out = cp.empty((N, 3), dtype=cp.float32)
        tp = hardware.ddd_tile_points() if tile_points is None else int(tile_points)
        threads = 128
        for p0 in range(0, N, tp):
            p1 = min(N, p0 + tp)
            P64 = cp.ascontiguousarray(P[p0:p1].astype(cp.float64))
            n = p1 - p0
            kernel(((n + threads - 1) // threads,), (threads,),
                   (P64, np.int32(n), seg64, np.int32(M), np.float32(a_p * a_p), np.float32(a_g * a_g),
                    np.float32(r_cut), np.float32(c1), np.float32(c2), out[p0:p1]))
        return out if input_is_cupy else cp.asnumpy(out)

    def apply_dislocation_displacement(self, sample, use_gpu=True, core_radius=5.0, nu=0.3,
                                       force=False, reference_point=None):
        """
        Displace every atom of a sample by the dislocation network field.

        Evaluates `dislocation_displacement` at the stored (thermal-free)
        positions of each chunk, adds it, rewrites the chunk, updates the
        sample bounding box and metadata, and records the operation on the
        sample so it is not applied twice.

        Args:
            sample: Sample object providing chunk loading/writing methods.
            use_gpu: If True and CuPy is available, evaluate on the GPU.
            core_radius: Core radius a in Angstrom. Defaults to 5.0.
            nu: Poisson's ratio. Defaults to 0.3.
            force: If True, apply even if the same operation was already
                recorded on the sample. Defaults to False.
            reference_point: Optional (3,) closure point, see
                `dislocation_displacement`.

        Returns:
            dict: ``atoms`` displaced, ``max_displacement`` (Angstrom),
            ``segment_count``.

        Raises:
            RuntimeError: If the network is missing or the operation was
                already applied and `force` is False.
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Call import_dislocation_network(...) first.")
        M = int(np.asarray(self._opendis_S0).shape[0])
        params = {
            "source": getattr(self, "_opendis_source", None),
            "segment_count": M,
            "core_radius": float(core_radius),
            "nu": float(nu),
            "reference_point": None if reference_point is None
                               else np.asarray(reference_point, dtype=np.float64).reshape(3).tolist(),
        }
        if not force and sample.has_modification("dislocation_displacement", params):
            raise RuntimeError("dislocation_displacement with these parameters was already "
                               f"applied to the sample in '{sample.directory}'. "
                               "Pass force=True to apply it again.")

        on_gpu = bool(use_gpu) and (cp is not None)
        gmin = np.full(3, np.inf); gmax = np.full(3, -np.inf)
        n_atoms = 0
        umax = 0.0
        for k in range(int(sample.chunk_total)):
            X = sample.load_chunk_positions(k + 1, use_gpu=on_gpu, raw=True)
            U = self.dislocation_displacement(X, use_gpu=on_gpu, core_radius=core_radius,
                                              nu=nu, reference_point=reference_point)
            xp = cp if (on_gpu and isinstance(X, cp.ndarray)) else np
            Xn = X.astype(xp.float32) + U
            umax = max(umax, float(xp.abs(U).max()) if U.shape[0] else 0.0)
            if Xn.shape[0]:
                gmin = np.minimum(gmin, np.asarray(cp.asnumpy(Xn.min(axis=0)) if xp is cp else Xn.min(axis=0), dtype=np.float64))
                gmax = np.maximum(gmax, np.asarray(cp.asnumpy(Xn.max(axis=0)) if xp is cp else Xn.max(axis=0), dtype=np.float64))
            n_atoms += int(Xn.shape[0])
            sample.write_chunk_positions(cp.asnumpy(Xn) if xp is cp else Xn, k + 1)
            del X, U, Xn

        if np.all(np.isfinite(gmin)) and np.all(np.isfinite(gmax)):
            sample._dimensions = (gmax - gmin).astype(np.float32)
            sample._offset = ((gmin + gmax) * 0.5).astype(np.float32)
            sample._matrix = np.diag(sample._dimensions.astype(np.float32))
            sample._corners = (sample.get_unit_corners() @ sample._matrix) - (sample._dimensions * 0.5) + sample._offset
        sample.write_sample_metadata()
        sample.record_modification("dislocation_displacement", params)
        self._log("normal", f"apply_dislocation_displacement: {n_atoms} atoms, "
                            f"{M} segments, max |u| = {umax:.4f} A")
        return {"atoms": n_atoms, "max_displacement": umax, "segment_count": M}

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
            dmin = float(np.linalg.norm(R[0, :]))  # fallback
        return float(dmin)

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
        mod_flag  = hardware.raw_module(src_flag)
        mod_relax = hardware.raw_module(src_relax)
        self._cleanup_cuda = {
            "flag_near_segments": mod_flag.get_function("flag_near_segments"),
            "relax_repulsive_sorted": mod_relax.get_function("relax_repulsive_sorted"),
        }
        return self._cleanup_cuda

    def finalize_dislocation_sample(self,
                                    crystal,
                                    deformed_sample,
                                    pristine_sample=None,
                                    output_directory=None,
                                    mu=1.0,
                                    nu=0.33,
                                    core_radius=5.0,
                                    near_factor=1.5,
                                    relax_steps=3,
                                    relax_dt=0.125,
                                    relax_k=0.5,
                                    use_gpu=True,
                                    dtype=np.float32,
                                    force=False):
        """
        Finalize a dislocation sample: reset near-core atoms to the direct
        segment displacement and relax overlaps.

        Per chunk:
            1. Flag atoms within `near_factor * core_radius` of any segment
               (midpoint sphere test).
            2. If `pristine_sample` is given, reset the flagged atoms to
               X = Xref + u with u from `dislocation_displacement`, which
               removes the interpolation error of a grid field near the cores.
            3. Run `relax_steps` iterations of a short-range repulsive
               relaxation towards the nearest-neighbour distance of `crystal`.

        Args:
            crystal: Crystal object used for the nearest-neighbour distance.
            deformed_sample: Sample object with deformed atomic positions.
            pristine_sample: Optional pristine sample for the near-core reset.
                If None, near-core positions are not recomputed.
            output_directory: Directory for output files. If None, uses
                self.directory.
            mu: Accepted for API compatibility; the displacement depends on
                `nu` only.
            nu: Poisson's ratio (must be in (0, 0.5)). Defaults to 0.33.
            core_radius: Core radius a in Angstrom. Defaults to 5.0.
            near_factor: Factor multiplied by core_radius for the near-core
                cutoff. Defaults to 1.5.
            relax_steps: Number of repulsive relaxation iterations. Defaults to 3.
            relax_dt: Time step for relaxation. Defaults to 0.125.
            relax_k: Spring constant for repulsive relaxation. Defaults to 0.5.
            use_gpu: If True and CuPy is available, use GPU acceleration.
                Defaults to True.
            dtype: NumPy dtype for output arrays. Defaults to np.float32.
            force: If True, run even if the same finalisation is already
                recorded on the sample. Defaults to False.

        Raises:
            RuntimeError: If the network has not been imported, or the same
                finalisation was already applied in place and `force` is False.
            ValueError: If nu is out of range (0, 0.5).
        """
        if not hasattr(self, "_opendis_S0") or self._opendis_S0 is None:
            raise RuntimeError("Dislocation network must be imported first (import_dislocation_network).")

        out_dir = output_directory if output_directory is not None else (self.directory if self.directory else ".")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        nu = float(nu)
        if not (0.0 < nu < 0.5):
            raise ValueError("nu must be in (0, 0.5)")
        a = float(core_radius)
        near_rcut = float(near_factor) * a

        # Dislocation arrays (kept in float32 for GPU)
        MID = np.asarray(self._opendis_mids, dtype=np.float32)
        HL  = np.asarray(self._opendis_halfL, dtype=np.float32)
        Ns = int(MID.shape[0])
        if Ns == 0:
            raise ValueError("No dislocation segments loaded.")

        params = {
            "segment_count": Ns,
            "core_radius": a,
            "nu": nu,
            "near_factor": float(near_factor),
            "near_core_reset": pristine_sample is not None,
            "relax_steps": int(relax_steps),
            "relax_dt": float(relax_dt),
            "relax_k": float(relax_k),
        }
        _check_modification(deformed_sample, out_dir, "dislocation_finalize", params, force)

        # Relaxation target spacing from crystal
        d0 = self._nearest_neighbor_distance_from_crystal(crystal)

        gpu_ok = bool(use_gpu and (cp is not None))
        if gpu_ok:
            try:
                _ = cp.cuda.runtime.getDeviceCount()
            except Exception:
                gpu_ok = False

        if gpu_ok:
            cuda_helpers = self._ensure_cuda_helpers_for_cleanup()
            MIDg = cp.asarray(MID); HLg = cp.asarray(HL)

        # Adaptive GPU batch size to prevent OOM
        def _gpu_batch_cap(default_cap=500000):
            if not gpu_ok:
                return 0
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                bytes_per_pt = 3*4 + 3*4 + 64
                cap = int(0.6 * free_b / max(bytes_per_pt, 1))
                cap = max(32768, min(cap, default_cap))
                return cap
            except Exception:
                return default_cap

        CPU_BATCH = 200000

        K = int(deformed_sample.chunk_total)
        for k in range(K):
            if gpu_ok:
                Xg = deformed_sample.load_chunk_positions(k+1, use_gpu=True, raw=True)
                spcnp = deformed_sample.load_chunk_species(k+1, use_gpu=False)
                M = int(Xg.shape[0])

                # Flag near-core atoms
                flags = cp.empty((M,), dtype=cp.uint8)
                threads = 256
                blocks = (M + threads - 1)//threads
                cuda_helpers["flag_near_segments"](
                    (blocks,), (threads,),
                    (Xg.astype(cp.float32).ravel(),
                    MIDg.ravel(), HLg.ravel(),
                    np.int32(Ns),
                    cp.float32(near_rcut),
                    flags, np.int32(M))
                )
                near_idx = cp.where(flags != 0)[0]

                # Reset flagged atoms to Xref + u(Xref) from the segment formula
                if pristine_sample is not None and near_idx.size > 0:
                    pos_refnp = pristine_sample.load_chunk_positions(k+1, use_gpu=False, raw=True)
                    near_idxnp = near_idx.get()
                    BATCH = _gpu_batch_cap()
                    for s0 in range(0, near_idxnp.size, BATCH):
                        s1 = min(near_idxnp.size, s0 + BATCH)
                        idx_slice = near_idxnp[s0:s1]
                        Xref_sel = cp.asarray(pos_refnp[idx_slice, :], dtype=cp.float32)
                        Uout = self.dislocation_displacement(Xref_sel, use_gpu=True,
                                                             core_radius=a, nu=nu)
                        Xg[cp.asarray(idx_slice), :] = Xref_sel + Uout

                # GPU relaxation via cell list
                sorted_pos, sorted_idx, cell_start, cell_end, bbmin, cell_size, nx, ny, nz = \
                    deformed_sample.build_cell_list_gpu(Xg.astype(cp.float32), r_cut=float(d0))

                out_sorted = cp.empty_like(sorted_pos)
                threads3 = 256
                blocks3 = (int(sorted_pos.shape[0]) + threads3 - 1)//threads3
                relax_fn = cuda_helpers["relax_repulsive_sorted"]
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
                    sorted_pos, out_sorted = out_sorted, sorted_pos

                pos_out = cp.empty_like(Xg)
                pos_out[sorted_idx, :] = sorted_pos
                posnp = pos_out.get()

            else:
                posnp = np.asarray(deformed_sample.load_chunk_positions(k+1, use_gpu=False, raw=True), dtype=np.float32)
                spcnp = deformed_sample.load_chunk_species(k+1, use_gpu=False)
                M = int(posnp.shape[0])

                # Streaming midpoint test, one segment at a time
                near_mask = np.zeros(M, dtype=bool)
                for s in range(Ns):
                    rad = near_rcut + HL[s]
                    d = posnp - MID[s]
                    dist2 = (d[:, 0]*d[:, 0] + d[:, 1]*d[:, 1] + d[:, 2]*d[:, 2])
                    near_mask |= (dist2 <= rad*rad)

                if pristine_sample is not None and np.any(near_mask):
                    Xref_all = np.asarray(pristine_sample.load_chunk_positions(k+1, use_gpu=False, raw=True), dtype=np.float32)
                    idx_all = np.flatnonzero(near_mask)
                    for s0 in range(0, idx_all.size, CPU_BATCH):
                        s1 = min(idx_all.size, s0 + CPU_BATCH)
                        idx_slice = idx_all[s0:s1]
                        Xsel = Xref_all[idx_slice, :]
                        Usel = self.dislocation_displacement(Xsel, use_gpu=False,
                                                             core_radius=a, nu=nu)
                        posnp[idx_slice, :] = Xsel + Usel

                # CPU relaxation on a cell hash
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

            deformed_sample.write_chunk_positions(posnp.astype(dtype, copy=False), k+1, override_directory=out_dir)
            deformed_sample.write_chunk_species(spcnp, k+1, override_directory=out_dir)

        _record_modification(deformed_sample, out_dir, "dislocation_finalize", params)
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
            # The conventional lattice matrix holds a, b, c as its rows. The
            # Burgers vector [uvw] is a direct-lattice direction, so its
            # Cartesian form is the transpose of the normalised rows acting on
            # the indices. The fault normal (hkl) is a reciprocal-lattice
            # direction, n = h a* + k b* + l c* = inv(L) @ (h, k, l), which
            # only coincides with [hkl] for orthogonal cells.
            L = np.asarray(crystal.lattice_matrix_conventional, dtype=np.float64)
            Lens = np.asarray(crystal.lattice_lengths_conventional, dtype=np.float64)
            n = np.linalg.inv(L) @ np.asarray(self.fault_normal, dtype=np.float64)
            self.rotated_fault_normal = n / np.linalg.norm(n)
            self.rotated_burgers_vector = (L/Lens[:,None]).T@self.burgers_vector
            sample_center = sample.offset
            sample_center_proj = np.dot(sample_center,self.rotated_fault_normal)
            fault_offest_proj = np.dot(self.fault_offset,self.rotated_fault_normal)
            self.global_fault_positions = sample_center_proj + fault_offest_proj - (self.fault_number - 1)*(self.interfault_spacing+self.fault_gap)/2 + np.arange(self.fault_number, dtype=np.float32)*(self.interfault_spacing+self.fault_gap)

            self._prepare_fault_tables(use_gpu=use_gpu)

            if plotting:
                self.plot_global_positions(sample)

        def _prepare_fault_tables(self, use_gpu=True):
            """
            Build the sorted fault positions and the prefix sum of their
            orientations used by `apply_stacking_fault_chunk`, plus GPU copies
            when requested. Requires `global_fault_positions`,
            `rotated_fault_normal` and `rotated_burgers_vector` to be set.

            Args:
                use_gpu: If True and CuPy is available, also stage the
                    tables on the GPU. Defaults to True.
            """
            order = np.argsort(self.global_fault_positions, kind="stable")
            self._fault_positions_sorted = np.asarray(self.global_fault_positions)[order]
            orient = np.asarray(self.fault_orientation, dtype=np.int64)[order]
            self._fault_orientation_prefix = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(orient)])
            if cp is not None and use_gpu:
                self._global_fault_positions_cp = cp.asarray(self.global_fault_positions)
                self._fault_positions_sorted_cp = cp.asarray(self._fault_positions_sorted)
                self._fault_orientation_prefix_cp = cp.asarray(self._fault_orientation_prefix)
                self._rotated_burgers_vector_cp = cp.asarray(self.rotated_burgers_vector)
                self._fault_gap_cp = cp.float32(self.fault_gap)
                self._fault_normal_cp = cp.asarray(self.rotated_fault_normal)
                self._fault_orientation_cp = cp.asarray(self.fault_orientation, dtype=cp.int8)
            else:
                self._global_fault_positions_cp = None
                self._fault_positions_sorted_cp = None
                self._fault_orientation_prefix_cp = None
                self._rotated_burgers_vector_cp = None
                self._fault_gap_cp = None
                self._fault_normal_cp = None
                self._fault_orientation_cp = None

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

        def apply_to_sample(self, sample, use_gpu=True, force=False):
            """
            Apply stacking faults to all chunks of a sample.

            Iterates through all sample chunks, applies stacking fault
            displacements to the stored (thermal-free) atomic positions, and
            writes the modified data. When writing into the sample's own
            directory the operation is recorded on the sample and refused on
            a repeat call unless `force` is True.

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True and CuPy is available, use GPU acceleration.
                    Defaults to True.
                force: If True, apply even if the same faults were already
                    recorded on the sample. Defaults to False.

            Raises:
                RuntimeError: If the same faults were already applied in place
                    and `force` is False.
            """
            params = {
                "fault_number": int(self.fault_number),
                "fault_offset": np.asarray(self.fault_offset, dtype=np.float64).tolist(),
                "fault_normal": np.asarray(self.fault_normal, dtype=np.float64).tolist(),
                "interfault_spacing": float(self.interfault_spacing),
                "burgers_vector": np.asarray(self.burgers_vector, dtype=np.float64).tolist(),
                "fault_orientation": np.asarray(self.fault_orientation).tolist(),
                "fault_gap": float(self.fault_gap),
            }
            _check_modification(sample, self.directory, "stacking_fault", params, force)
            # apply_stacking_fault_chunk reads the sorted-position and
            # orientation-prefix tables, not global_fault_positions itself.
            # generate_global_positions builds them, so any caller that sets
            # the positions afterwards would otherwise be silently ignored.
            if self.global_fault_positions is not None:
                self._prepare_fault_tables(use_gpu=(cp is not None and use_gpu))
            for i in range(sample.chunk_total):
                if cp is not None and use_gpu:
                    positions_chunk_cp = sample.load_chunk_positions(i+1,use_gpu=True, raw=True)
                    positions_chunk_cp = self.apply_stacking_fault_chunk(positions_chunk_cp,use_gpu=True)
                    positions_chunknp = cp.asnumpy(positions_chunk_cp)
                else:
                    positions_chunknp = sample.load_chunk_positions(i+1,use_gpu=False, raw=True)
                    positions_chunknp = self.apply_stacking_fault_chunk(positions_chunknp,use_gpu=False)

                sample.write_chunk_positions(positions_chunknp,i+1,override_directory=self.directory)
                if self.directory is not None:
                    species_chunknp = sample.load_chunk_species(i + 1, use_gpu=False)
                    sample.write_chunk_species(species_chunknp,i+1,override_directory=self.directory)
            _record_modification(sample, self.directory, "stacking_fault", params)
            sample.write_sample_metadata(override_directory=self.directory)
            
        def apply_stacking_fault_chunk(self, positions_chunk, use_gpu=True):
            """
            Apply stacking fault displacements to a single chunk of positions.

            Shifts atoms that lie beyond each fault plane by the Burgers vector,
            accounting for the fault orientation. Adds a small gap at each fault
            plane crossing. The number of planes below each atom comes from a
            `searchsorted` on the sorted plane positions and the signed count
            from a prefix sum of the orientations, so no (N x faults) array is
            formed.

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
                # planes strictly below the atom, and the signed count of them
                count_faults_abs = cp.searchsorted(self._fault_positions_sorted_cp, position_projection, side="left")
                count_faults = self._fault_orientation_prefix_cp[count_faults_abs]
                positions_chunk = positions_chunk \
                    + count_faults[:, None] * self._rotated_burgers_vector_cp \
                    + count_faults_abs[:, None] * self._fault_normal_cp * self._fault_gap_cp \
                    - self._fault_normal_cp * self._fault_gap_cp * self.fault_number/2
                return positions_chunk
            else:
                position_projection = np.dot(positions_chunk, self.rotated_fault_normal)
                count_faults_abs = np.searchsorted(self._fault_positions_sorted, position_projection, side="left")
                count_faults = self._fault_orientation_prefix[count_faults_abs]
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
        def apply_to_sample(self, sample, use_gpu=True, force=False):
            """
            Apply the crack to all chunks of a sample.

            Iterates through all sample chunks and removes atoms whose stored
            (thermal-free) positions fall inside the crack's convex hull. When
            writing into the sample's own directory the operation is recorded
            on the sample and refused on a repeat call unless `force` is True.

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True and CuPy is available, use GPU acceleration.
                    Defaults to True.
                force: If True, apply even if the same crack was already
                    recorded on the sample. Defaults to False.

            Raises:
                RuntimeError: If the same crack was already applied in place
                    and `force` is False.
            """
            params = {"crack_points": np.asarray(self.crack_points, dtype=np.float64).tolist()}
            _check_modification(sample, self.directory, "crack", params, force)
            for i in range(sample.chunk_total):
                if cp is not None and use_gpu:
                    positions_chunk_cp = sample.load_chunk_positions(i + 1, use_gpu=True, raw=True)
                    species_chunknp = sample.load_chunk_species(i + 1, use_gpu=False)
                    positions_chunk_cp, species_chunknp = self.apply_crack_chunk(positions_chunk_cp,species_chunknp,use_gpu=True)
                    positions_chunknp = cp.asnumpy(positions_chunk_cp)
                else:
                    positions_chunknp = sample.load_chunk_positions(i + 1, use_gpu=False, raw=True)
                    species_chunknp = sample.load_chunk_species(i + 1, use_gpu=False)
                    positions_chunknp, species_chunknp = self.apply_crack_chunk(positions_chunknp,species_chunknp,use_gpu=False)

                sample.write_chunk_positions(positions_chunknp,i+1,override_directory=self.directory)
                sample.write_chunk_species(species_chunknp,i+1,override_directory=self.directory)
            _record_modification(sample, self.directory, "crack", params)
            sample.write_sample_metadata(override_directory=self.directory)

        def apply_crack_chunk(self, positions_chunk, species_chunknp, use_gpu=True):
            """
            Remove atoms inside the crack's convex hull from a single chunk.

            Tests positions against the half-space inequalities defined by the
            convex hull facets, one facet at a time in float32, and removes
            atoms that satisfy all inequalities (i.e., are inside the hull).

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
                    self._hull_equations_cp = cp.asarray(self.hull_equations, dtype=cp.float32)
                inside_mask = _convex_hull_inside_mask(positions_chunk, self._hull_equations_cp)
                positions_chunk = positions_chunk[~inside_mask]
                species_chunknp = species_chunknp[~(inside_mask.get())]
                return positions_chunk, species_chunknp
            else:
                inside_mask = _convex_hull_inside_mask(positions_chunk, self.hull_equations)
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

    class amorphous_band(logging):
        """
        Represents an amorphous band defect: a 3D region in which crystalline
        atoms are replaced by a uniform random distribution.

        The region is defined either as a convex hull built from explicit
        boundary points (mirrors `crack`) or as an oriented slab specified by
        a center, length, width, thickness, and orientation vector. The total
        number of replacement atoms is set by either a relative density
        multiplier or an absolute number density.

        Attributes:
            directory: Output directory for modified sample data.
            density_ratio: Multiplier on the original in-region atom count.
            number_density: Absolute target number density (atoms / A^3) or None.
            seed: Optional RNG seed for reproducible amorphization.
            band_points: (N, 3) hull vertices in hull mode, else None.
            hull, hull_equations: scipy ConvexHull and its plane equations,
                hull mode only.
            center, length, width, thickness, orientation: slab parameters,
                slab mode only.
        """

        # -------------------------------------------------------------------------
        # Logging configuration
        # -------------------------------------------------------------------------
        __log_top__ = (
            "apply_to_sample",
            "apply_amorphous_band_chunk",
            "plot_band_geometry",
        )

        # -----------------------------------------------------------------------------
        # Functions
        # -----------------------------------------------------------------------------
        ## Initialization
        def __init__(self, directory,
                     band_points=None,
                     center=None, length=None, width=None,
                     thickness=None, orientation=None,
                     period=None, n_stripes=None,
                     density_ratio=1.0,
                     number_density=None,
                     seed=None):
            """
            Initialize an amorphous band defect.

            Args:
                directory: Output directory for storing modified sample data.
                band_points: Array-like of shape (N, 3) defining the vertices
                    of a convex hull in 3D. Mutually exclusive with the slab
                    parameters.
                center: (3,) slab center in Angstroms. Required for slab mode.
                length: Slab extent along its primary in-plane axis.
                width: Slab extent along its secondary in-plane axis.
                thickness: Slab extent along the slab normal. In periodic
                    mode this is the per-stripe thickness.
                orientation: (3,) slab normal vector. Defaults to [0, 0, 1].
                period: Optional centre-to-centre stripe spacing along the
                    slab normal (Angstroms). When set, the band becomes a
                    stack of `n_stripes` parallel stripes. Slab mode only.
                n_stripes: Number of parallel stripes when `period` is set.
                    Stripes are placed symmetrically about `center`, spanning
                    `period * n_stripes` along the normal.
                density_ratio: Multiplier on the original in-region atom count.
                    Defaults to 1.0.
                number_density: Absolute number density (atoms / A^3). When
                    provided, overrides density_ratio. Defaults to None.
                seed: Optional RNG seed for reproducibility.

            Raises:
                ValueError: If both hull and slab parameters are provided, if
                    neither is provided, if slab mode is chosen but any of
                    center/length/width/thickness is missing, or if periodic
                    parameters are passed in hull mode.
            """
            super().__init__(log_name="amorphous_band")
            self.directory = directory

            # Determine mode and validate exclusivity of hull vs slab params
            slab_any = any(x is not None for x in (center, length, width, thickness))
            hull_given = band_points is not None
            if hull_given and slab_any:
                raise ValueError(
                    "amorphous_band: pass either `band_points` (hull mode) OR "
                    "slab parameters (center, length, width, thickness), not both."
                )
            if not hull_given and not slab_any:
                raise ValueError(
                    "amorphous_band: must provide either `band_points` or the "
                    "slab parameters (center, length, width, thickness)."
                )

            self.density_ratio = float(density_ratio)
            self.number_density = None if number_density is None else float(number_density)
            self.seed = seed

            # Periodic-stripe state. Defaults degenerate to a single band
            # at offset 0 along the normal so the same code paths handle both
            # the single-band and periodic-stack cases.
            self.period = None
            self.n_stripes = 1
            self._stripe_offsets = np.array([0.0])
            self._stack_span = None
            self._stripe_offsets_cp = None  # lazy GPU cache

            # Hull mode
            if hull_given:
                if period is not None or n_stripes is not None:
                    raise ValueError(
                        "amorphous_band: periodic stripes are only supported in slab mode."
                    )
                self._mode = "hull"
                from scipy.spatial import ConvexHull
                self.band_points = np.asarray(band_points, dtype=float)
                self.hull = ConvexHull(self.band_points)
                # Each row [a, b, c, d] => a*x + b*y + c*z + d <= 0 for points inside
                self.hull_equations = self.hull.equations
                # Slab fields kept None for serialization symmetry
                self.center = None
                self.length = None
                self.width = None
                self.thickness = None
                self.orientation = None
                self._slab_origin = None
                self._slab_axes = None
                self._slab_extents = None
                # Lazy GPU caches
                self._hull_equations_cp = None
                self._slab_origin_cp = None
                self._slab_axes_cp = None
                self._slab_extents_cp = None
            # Slab mode
            else:
                missing = [n for n, v in (("center", center), ("length", length),
                                          ("width", width), ("thickness", thickness))
                           if v is None]
                if missing:
                    raise ValueError(
                        f"amorphous_band slab mode requires {missing} to be provided."
                    )
                self._mode = "slab"
                self.band_points = None
                self.hull = None
                self.hull_equations = None
                self.center = np.asarray(center, dtype=float).reshape(3,)
                self.length = float(length)
                self.width = float(width)
                self.thickness = float(thickness)
                self.orientation = (np.asarray(orientation, dtype=float).reshape(3,)
                                    if orientation is not None
                                    else np.array([0.0, 0.0, 1.0]))
                self._build_slab_geometry()
                # Lazy GPU caches
                self._hull_equations_cp = None
                self._slab_origin_cp = None
                self._slab_axes_cp = None
                self._slab_extents_cp = None

                # Periodic stripe stack
                if period is not None:
                    if n_stripes is None or int(n_stripes) < 1:
                        raise ValueError(
                            "amorphous_band: periodic mode requires n_stripes >= 1."
                        )
                    self.period = float(period)
                    self.n_stripes = int(n_stripes)
                    i = np.arange(self.n_stripes, dtype=float)
                    # Stripe centres along the slab normal in slab-local
                    # coordinates, placed symmetrically about 0.
                    self._stripe_offsets = (i - (self.n_stripes - 1) / 2.0) * self.period
                self._stack_span = (self.period * self.n_stripes
                                    if self.period is not None
                                    else self.thickness)

        ## Geometry helpers
        def _build_slab_geometry(self):
            """
            Build an orthonormal slab basis (u, v, n) and store extents.

            The slab normal `n` is `orientation / ||orientation||`. In-plane
            axes (u, v) are produced by Gram-Schmidt against a seed axis chosen
            to be the global axis least aligned with `n`. Together (u, v, n)
            form a right-handed orthonormal basis. The basis is stored as a
            (3, 3) matrix whose rows are u, v, n; the half-extents are
            (length/2, width/2, thickness/2) along (u, v, n).
            """
            n = self.orientation
            n_norm = np.linalg.norm(n)
            if n_norm <= 0.0:
                raise ValueError("amorphous_band: orientation must be a non-zero vector.")
            n = n / n_norm

            # Seed axis: pick the global axis least aligned with n
            seed_options = np.eye(3, dtype=float)
            dots = np.abs(seed_options @ n)
            seed = seed_options[int(np.argmin(dots))]

            # Gram-Schmidt: u = (seed - (seed.n) n), normalized
            u = seed - np.dot(seed, n) * n
            u = u / np.linalg.norm(u)
            # v = n x u so that (u, v, n) is right-handed
            v = np.cross(n, u)

            self._slab_origin = self.center.astype(float)
            self._slab_axes = np.stack([u, v, n], axis=0)  # rows: u, v, n
            self._slab_extents = np.array([self.length / 2.0,
                                           self.width / 2.0,
                                           self.thickness / 2.0], dtype=float)

        def _in_region_mask(self, positions, use_gpu=False):
            """
            Boolean mask selecting positions inside the band region.

            In hull mode, applies the convex-hull half-space inequalities one
            facet at a time in float32 (mirrors `crack.apply_crack_chunk`). In
            slab mode, projects each position onto the slab basis and tests
            against the half-extents.

            Args:
                positions: (N, 3) array of positions. Numpy or cupy.
                use_gpu: If True and CuPy is available, the mask is computed
                    on the device and returned as a cupy bool array.

            Returns:
                (N,) boolean mask, on the same device as the input when
                `use_gpu=True`, otherwise on CPU.
            """
            on_gpu = (cp is not None) and bool(use_gpu)
            xp = cp if on_gpu else np

            if self._mode == "hull":
                if on_gpu:
                    if self._hull_equations_cp is None:
                        self._hull_equations_cp = cp.asarray(self.hull_equations, dtype=cp.float32)
                    eq = self._hull_equations_cp
                else:
                    eq = self.hull_equations
                return _convex_hull_inside_mask(xp.asarray(positions), eq)
            else:
                if on_gpu:
                    if self._slab_origin_cp is None:
                        self._slab_origin_cp = cp.asarray(self._slab_origin)
                        self._slab_axes_cp = cp.asarray(self._slab_axes)
                        self._slab_extents_cp = cp.asarray(self._slab_extents)
                    origin = self._slab_origin_cp
                    axes = self._slab_axes_cp
                    extents = self._slab_extents_cp
                else:
                    origin = self._slab_origin
                    axes = self._slab_axes
                    extents = self._slab_extents
                rel = positions - origin
                proj = rel @ axes.T  # (N, 3): components along (u, v, n)

                # In-plane (u, v) test is identical for single-band and
                # periodic stacks.
                in_uv = (xp.abs(proj[:, 0]) <= extents[0] + 1e-12) & \
                        (xp.abs(proj[:, 1]) <= extents[1] + 1e-12)

                if self.period is None:
                    in_n = xp.abs(proj[:, 2]) <= extents[2] + 1e-12
                else:
                    # Periodic stripe stack: stripes of half-thickness
                    # extents[2] centred at multiples of `period`, the whole
                    # stack spanning `n_stripes * period` along the normal.
                    span = float(self._stack_span)
                    P = float(self.period)
                    in_stack = xp.abs(proj[:, 2]) <= span / 2.0 + 1e-12
                    # Distance from each point to the nearest stripe centre.
                    # Use floor() rather than `%` for cupy-portability of
                    # floating-point modulo.
                    shifted = proj[:, 2] + span / 2.0
                    mod_q = shifted - xp.floor(shifted / P) * P
                    d_mid = xp.abs(mod_q - P / 2.0)
                    in_stripe = d_mid <= extents[2] + 1e-12
                    in_n = in_stack & in_stripe

                return in_uv & in_n

        def _region_volume(self):
            """
            Return the geometric volume of the band region (Angstrom^3).

            Hull mode uses scipy ConvexHull's reported volume. Slab mode
            returns length * width * thickness, multiplied by `n_stripes`
            in periodic mode (sum over the stripe stack).
            """
            if self._mode == "hull":
                return float(self.hull.volume)
            return float(self.length * self.width * self.thickness * self.n_stripes)

        def _region_aabb(self):
            """
            Return the axis-aligned bounding box (lo, hi) of the band region.
            For periodic stacks the AABB extends along the slab normal to
            cover all stripes.
            """
            if self._mode == "slab":
                # Half-extent along the normal axis: full stack span / 2 in
                # periodic mode, otherwise the single-band thickness / 2.
                half_n = (self._stack_span / 2.0
                          if self.period is not None
                          else self._slab_extents[2])
                stack_extents = np.array([self._slab_extents[0],
                                          self._slab_extents[1],
                                          half_n], dtype=float)
                signs = np.array([[sx, sy, sz]
                                  for sx in (-1.0, 1.0)
                                  for sy in (-1.0, 1.0)
                                  for sz in (-1.0, 1.0)], dtype=float)
                local = signs * stack_extents
                slab_corners = local @ self._slab_axes + self._slab_origin
                return slab_corners.min(axis=0), slab_corners.max(axis=0)
            return self.band_points.min(axis=0), self.band_points.max(axis=0)

        def _sample_uniform_in_region(self, n, rng,
                                      sample_min=None, sample_max=None):
            """
            Draw `n` positions uniformly at random from the band region,
            optionally further restricted to a sample axis-aligned bounding
            box [sample_min, sample_max].

            When no sample bounds are provided, slab mode uses the fast direct
            sampler (uniform in slab-local coordinates) and hull mode uses
            rejection sampling from the hull's AABB. When sample bounds are
            provided, both modes use rejection sampling from the intersection
            of the region AABB and the sample AABB, accepting only points
            that satisfy `_in_region_mask`.

            Args:
                n: Number of samples to draw.
                rng: numpy.random.RandomState (or compatible) instance.
                sample_min: Optional (3,) array, lower corner of the sample
                    AABB. New atoms are guaranteed to lie above this bound.
                sample_max: Optional (3,) array, upper corner of the sample
                    AABB. New atoms are guaranteed to lie below this bound.

            Returns:
                (n, 3) float32 array of positions.
            """
            if n <= 0:
                return np.zeros((0, 3), dtype=np.float32)

            clip = (sample_min is not None) and (sample_max is not None)

            # Fast path: slab mode without sample clipping
            if self._mode == "slab" and not clip:
                local = rng.uniform(low=-self._slab_extents,
                                    high=self._slab_extents,
                                    size=(n, 3))
                world = local @ self._slab_axes + self._slab_origin
                return world.astype(np.float32)

            # Rejection-sample within the AABB of (region) intersected with
            # (sample box, if given).
            region_lo, region_hi = self._region_aabb()
            if clip:
                lo = np.maximum(region_lo, np.asarray(sample_min, dtype=float))
                hi = np.minimum(region_hi, np.asarray(sample_max, dtype=float))
            else:
                lo, hi = region_lo, region_hi

            if np.any(hi <= lo):
                return np.zeros((0, 3), dtype=np.float32)

            out = np.empty((n, 3), dtype=np.float32)
            filled = 0
            # Heuristic batch size: oversample to limit loop iterations.
            aabb_vol = float(np.prod(hi - lo))
            region_vol = max(self._region_volume(), 1e-30)
            accept = max(min(region_vol / max(aabb_vol, 1e-30), 1.0), 1e-3)
            batch = max(int(np.ceil((n - filled) * 4.0 / accept)), 1024)
            while filled < n:
                cand = rng.uniform(low=lo, high=hi, size=(batch, 3))
                mask = self._in_region_mask(cand, use_gpu=False)
                cand = cand[mask]
                take = min(cand.shape[0], n - filled)
                out[filled:filled + take] = cand[:take].astype(np.float32)
                filled += take
                if filled < n and cand.shape[0] > 0:
                    realized = max(cand.shape[0] / float(batch), 1e-3)
                    batch = max(int(np.ceil((n - filled) * 1.5 / realized)), 1024)
            return out

        def _enumerate_polytope_vertices(self, planes):
            """
            Run the standard 'pick 3 hyperplanes -> solve linear system ->
            test against all inequalities' vertex-enumeration algorithm on
            a list of half-space planes (a, b) such that a . x <= b.

            Args:
                planes: list of (a, b) tuples; a is (3,), b is scalar.

            Returns:
                (M, 3) array of feasible vertices (deduplicated), or None
                if the polytope is empty or degenerate.
            """
            tol = 1e-6
            verts = []
            n_planes = len(planes)
            from itertools import combinations
            for trip in combinations(range(n_planes), 3):
                A = np.array([planes[t][0] for t in trip])
                b = np.array([planes[t][1] for t in trip])
                if abs(np.linalg.det(A)) < 1e-9:
                    continue
                try:
                    p = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue
                ok = True
                for aq, bq in planes:
                    if np.dot(aq, p) > bq + tol:
                        ok = False
                        break
                if ok:
                    verts.append(p)
            if not verts:
                return None
            verts = np.asarray(verts)
            keep = np.ones(len(verts), dtype=bool)
            for i in range(len(verts)):
                if not keep[i]:
                    continue
                for j in range(i + 1, len(verts)):
                    if keep[j] and np.linalg.norm(verts[i] - verts[j]) < 1e-6:
                        keep[j] = False
            return verts[keep]

        def _band_sample_intersection_polytopes(self, sample_min, sample_max):
            """
            Return the band-region / sample-AABB intersection as a list of
            convex polytope vertex sets.

            For hull mode and single-band slab mode this is always a list
            of length 1 (the band region is itself convex). For periodic
            slab mode each stripe is its own convex polytope, so the list
            has up to `n_stripes` entries (stripes that fall fully outside
            the sample are skipped).

            Args:
                sample_min: (3,) lower corner of the sample AABB.
                sample_max: (3,) upper corner of the sample AABB.

            Returns:
                List of (M_k, 3) arrays of feasible vertices, one per
                non-empty stripe. May be empty.
            """
            sample_min = np.asarray(sample_min, dtype=float)
            sample_max = np.asarray(sample_max, dtype=float)

            # Sample AABB half-spaces (shared by all stripes / hull)
            sample_planes = []
            for axis in range(3):
                e = np.zeros(3); e[axis] = 1.0
                sample_planes.append((e, sample_max[axis]))
                sample_planes.append((-e, -sample_min[axis]))

            polytopes = []
            if self._mode == "hull":
                planes = list(sample_planes)
                for eq in self.hull_equations:
                    planes.append((eq[:3], -eq[3]))
                verts = self._enumerate_polytope_vertices(planes)
                if verts is not None and verts.shape[0] >= 4:
                    polytopes.append(verts)
            else:
                u, v, n_axis = self._slab_axes  # rows
                ext_u, ext_v, ext_n = self._slab_extents
                base_origin = self._slab_origin
                for offset in self._stripe_offsets:
                    stripe_origin = base_origin + float(offset) * n_axis
                    planes = list(sample_planes)
                    # u, v faces use the original origin (in-plane extents
                    # are constant across stripes); the n faces are
                    # recentred on `stripe_origin`.
                    for axis_vec, extent, ref_origin in (
                        (u,      ext_u, base_origin),
                        (v,      ext_v, base_origin),
                        (n_axis, ext_n, stripe_origin),
                    ):
                        c = np.dot(axis_vec, ref_origin)
                        planes.append((axis_vec, c + extent))
                        planes.append((-axis_vec, -c + extent))
                    verts = self._enumerate_polytope_vertices(planes)
                    if verts is not None and verts.shape[0] >= 4:
                        polytopes.append(verts)

            return polytopes

        def _intersection_volume(self, sample_min, sample_max):
            """
            Return the volume of the band-region / sample-AABB intersection.

            Sums scipy ConvexHull volumes across the polytope list returned
            by `_band_sample_intersection_polytopes`. Returns 0.0 when the
            intersection is empty.
            """
            polytopes = self._band_sample_intersection_polytopes(sample_min, sample_max)
            if not polytopes:
                return 0.0
            from scipy.spatial import ConvexHull
            total = 0.0
            for verts in polytopes:
                try:
                    total += float(ConvexHull(verts).volume)
                except Exception:
                    pass
            return total

        ## Main Functions
        def apply_to_sample(self, sample, use_gpu=True, force=False):
            """
            Apply the amorphous band to all chunks of a sample.

            Two-pass over `sample.chunk_total`:
              Pass 1 (count + species histogram): for each chunk, computes
                the in-region mask on GPU (if available), counts in-region
                atoms, and accumulates a per-species histogram via
                `np.unique` on the in-region slice. Memory stays O(unique
                species) rather than O(in-region atoms). Species loading is
                skipped for chunks contributing zero in-region atoms.
              Pass 2 (rewrite): for each chunk, recomputes the mask on GPU,
                filters in-region atoms on the device, transfers only the
                kept slice back to host, then **generates this chunk's share
                of the new atoms in place** and appends them before writing
                back to disk. Generating per-chunk (rather than allocating a
                full (n_target, 3) buffer up front) keeps peak host memory
                bounded by the per-chunk share, which is essential for very
                large bands.

            The total number of replacement atoms is `number_density *
            volume(band ∩ sample_AABB)` when `number_density` is set, else
            `density_ratio * n_in_region`. New atoms are clipped to the
            sample's bounding box. Positions are read without thermal
            displacements; when writing into the sample's own directory the
            operation is recorded on the sample and refused on a repeat call
            unless `force` is True.

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True and CuPy is available, the region inside-test
                    and the in-region filter run on the GPU per chunk.
                    Defaults to True.
                force: If True, apply even if the same band was already
                    recorded on the sample. Defaults to False.

            Raises:
                RuntimeError: If the same band was already applied in place
                    and `force` is False.
            """
            params = {
                "mode": self._mode,
                "band_points": self.band_points.tolist() if self.band_points is not None else None,
                "center": self.center.tolist() if self.center is not None else None,
                "length": self.length, "width": self.width, "thickness": self.thickness,
                "orientation": self.orientation.tolist() if self.orientation is not None else None,
                "period": self.period, "n_stripes": self.n_stripes,
                "density_ratio": self.density_ratio,
                "number_density": self.number_density,
                "seed": self.seed,
            }
            _check_modification(sample, self.directory, "amorphous_band", params, force)
            rng = (np.random.RandomState(self.seed)
                   if self.seed is not None else np.random.RandomState())
            n_chunks = int(sample.chunk_total)
            on_gpu = (cp is not None) and bool(use_gpu)

            # Sample axis-aligned bounding box, used to clip new atoms so they
            # cannot land outside the physical sample even when the band
            # extends beyond it.
            sample_corners = np.asarray(sample.corners, dtype=float)
            sample_min = sample_corners.min(axis=0)
            sample_max = sample_corners.max(axis=0)

            # ---- Pass 1: count + per-chunk species histogram (no pool) ----
            n_in_region = 0
            hist_keys_parts = []     # list of (unique_per_chunk,) species arrays
            hist_counts_parts = []   # list of (unique_per_chunk,) int arrays
            species_dtype = None
            for i in range(n_chunks):
                if on_gpu:
                    pos_cp = sample.load_chunk_positions(i + 1, use_gpu=True, raw=True)
                    mask_cp = self._in_region_mask(pos_cp, use_gpu=True)
                    n_in = int(cp.count_nonzero(mask_cp))
                    n_in_region += n_in
                    if n_in > 0:
                        spc_np = sample.load_chunk_species(i + 1, use_gpu=False)
                        in_spc = spc_np[cp.asnumpy(mask_cp)]
                        if species_dtype is None:
                            species_dtype = spc_np.dtype
                        u, c = np.unique(in_spc, return_counts=True)
                        hist_keys_parts.append(u)
                        hist_counts_parts.append(c)
                    del pos_cp, mask_cp
                else:
                    pos_np = sample.load_chunk_positions(i + 1, use_gpu=False, raw=True)
                    mask_np = self._in_region_mask(pos_np, use_gpu=False)
                    n_in = int(np.count_nonzero(mask_np))
                    n_in_region += n_in
                    if n_in > 0:
                        spc_np = sample.load_chunk_species(i + 1, use_gpu=False)
                        in_spc = spc_np[mask_np]
                        if species_dtype is None:
                            species_dtype = spc_np.dtype
                        u, c = np.unique(in_spc, return_counts=True)
                        hist_keys_parts.append(u)
                        hist_counts_parts.append(c)

            # Aggregate per-chunk histograms into a single (keys, counts) pair.
            if hist_keys_parts:
                all_keys = np.concatenate(hist_keys_parts)
                all_counts = np.concatenate(hist_counts_parts)
                hist_keys, inverse = np.unique(all_keys, return_inverse=True)
                hist_counts = np.zeros(len(hist_keys), dtype=np.int64)
                np.add.at(hist_counts, inverse, all_counts)
            else:
                hist_keys = np.array([], dtype=(species_dtype if species_dtype is not None else object))
                hist_counts = np.array([], dtype=np.int64)

            # ---- Target count (uses the band-sample intersection volume so
            # ---- absolute number_density matches the physical region) ----
            if self.number_density is not None:
                vol = self._intersection_volume(sample_min, sample_max)
                n_target = int(round(self.number_density * vol))
            else:
                n_target = int(round(self.density_ratio * n_in_region))
            if n_target < 0:
                n_target = 0
            # No species histogram means there are no atoms to derive
            # stoichiometry from; fall back to pure atom removal.
            if hist_keys.size == 0:
                n_target = 0

            # Pre-compute species sampling probabilities once. New atoms are
            # generated per-chunk inside the Pass 2 loop to keep peak memory
            # bounded by the per-chunk share rather than the full n_target,
            # which is essential for very large bands.
            if n_target > 0:
                probs = hist_counts.astype(np.float64)
                probs /= probs.sum()
            else:
                probs = None

            # ---- Pass 2: rewrite chunks (filter on device, generate + append on host) ----
            remaining = n_target
            for i in range(n_chunks):
                if on_gpu:
                    pos_cp = sample.load_chunk_positions(i + 1, use_gpu=True, raw=True)
                    spc_np = sample.load_chunk_species(i + 1, use_gpu=False)
                    keep_cp = ~self._in_region_mask(pos_cp, use_gpu=True)
                    # Filter on device, then transfer only the kept slice.
                    pos_np = cp.asnumpy(pos_cp[keep_cp])
                    keep_np = cp.asnumpy(keep_cp)
                    spc_np = spc_np[keep_np]
                    del pos_cp, keep_cp
                else:
                    pos_np = sample.load_chunk_positions(i + 1, use_gpu=False, raw=True)
                    spc_np = sample.load_chunk_species(i + 1, use_gpu=False)
                    keep_np = ~self._in_region_mask(pos_np, use_gpu=False)
                    pos_np = pos_np[keep_np]
                    spc_np = spc_np[keep_np]

                # This chunk's share of newly generated atoms (round-robin
                # mirrors point-defect interstitial distribution).
                chunks_left = n_chunks - i
                to_take = int(np.ceil(remaining / float(chunks_left))) if chunks_left > 0 else 0
                if to_take > 0:
                    add_pos = self._sample_uniform_in_region(
                        to_take, rng,
                        sample_min=sample_min,
                        sample_max=sample_max,
                    )
                    add_spc = rng.choice(hist_keys, size=to_take, replace=True, p=probs)
                    pos_np = np.concatenate([pos_np.astype(np.float32, copy=False),
                                             add_pos.astype(np.float32, copy=False)],
                                            axis=0)
                    spc_np = np.concatenate([spc_np, add_spc], axis=0)
                    del add_pos, add_spc
                    remaining -= to_take

                sample.write_chunk_positions(pos_np, i + 1, override_directory=self.directory)
                sample.write_chunk_species(spc_np, i + 1, override_directory=self.directory)

            _record_modification(sample, self.directory, "amorphous_band", params)
            sample.write_sample_metadata(override_directory=self.directory)

        def plot_band_geometry(self, sample, color='c', alpha=0.5,
                               elev=20, azim=-60):
            """
            Plot the band-sample intersection inside the sample bounding box.

            The sample is drawn as a gray wireframe and the band's intersection
            with the sample's AABB is drawn as a filled convex polytope. Both
            slab and hull band modes are supported. The plot's axis limits are
            set to the sample bounds so the visualization shows only the
            physical region where atoms are amorphized.

            Args:
                sample: Sample object with a `corners` attribute of shape (8, 3).
                color: Matplotlib color for the intersection surface. Defaults
                    to 'c'.
                alpha: Face alpha for the intersection surface. Defaults to 0.5.
                elev: Elevation angle for 3D view. Defaults to 20.
                azim: Azimuth angle for 3D view. Defaults to -60.

            Returns:
                tuple: (fig, ax) matplotlib figure and axes objects.
            """
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')

            # 1) Sample wireframe
            sample_corners = np.asarray(sample.corners)
            sample_min = sample_corners.min(axis=0)
            sample_max = sample_corners.max(axis=0)
            edges = [
                (0,1), (0,2), (0,3),
                (1,4), (1,5),
                (2,4), (2,6),
                (3,5), (3,6),
                (4,7), (5,7), (6,7)
            ]
            segs = [(sample_corners[i], sample_corners[j]) for i, j in edges]
            ax.add_collection3d(Line3DCollection(segs, colors='gray', lw=1))

            # 2) Band-sample intersection as one or more filled convex
            # polytopes (one per stripe in periodic mode).
            polytopes = self._band_sample_intersection_polytopes(sample_min, sample_max)
            if polytopes:
                from scipy.spatial import ConvexHull as _CH
                all_faces = []
                for verts in polytopes:
                    try:
                        hull = _CH(verts)
                        for s in hull.simplices:
                            all_faces.append(verts[s])
                    except Exception:
                        continue
                if all_faces:
                    poly = Poly3DCollection(all_faces, facecolors=color,
                                            edgecolors='k', alpha=alpha,
                                            linewidths=0.3)
                    ax.add_collection3d(poly)

            # 3) Axis limits exactly the sample bounds
            ax.set_xlim(sample_min[0], sample_max[0])
            ax.set_ylim(sample_min[1], sample_max[1])
            ax.set_zlim(sample_min[2], sample_max[2])
            ax.set_xlabel("X (A)")
            ax.set_ylabel("Y (A)")
            ax.set_zlabel("Z (A)")
            ax.view_init(elev=elev, azim=azim)
            ax.set_title("Amorphous Band \u2229 Sample")
            plt.tight_layout()
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

            # Applied positions, one (M, 3) float32 array per chunk. Species
            # changes are the object's substitution_to / interstitial_species.
            self._applied_vacancies = []
            self._applied_substitutions = []
            self._applied_interstitials = []
            self._applied_directory = None

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

        def apply_to_sample(self, sample, use_gpu=False, tol_match=1e-4, force=False):
            """
            Apply all configured point defects to a sample.

            Streams over chunks and applies:
                - Random and specific vacancy deletions
                - Random and specific substitutions (species swap)
                - Random and specific interstitial insertions

            Then optionally performs local relaxation and writes back modified
            positions and species arrays. Positions are read without thermal
            displacements; when writing into the sample's own directory the
            operation is recorded on the sample and refused on a repeat call
            unless `force` is True. Applied positions are stored per chunk
            (see `_write_applied_arrays`).

            Args:
                sample: Sample object providing chunk loading/writing methods.
                use_gpu: If True, use GPU acceleration where available.
                    Defaults to False.
                tol_match: Tolerance for position matching when applying
                    specific defects by position. Defaults to 1e-4.
                force: If True, apply even if the same defects were already
                    recorded on the sample. Defaults to False.

            Raises:
                RuntimeError: If the same defects were already applied in
                    place and `force` is False.

            Note:
                Uses sample.load_chunk_positions/species and write_* counterparts,
                preserving the chunked .npy layout.
            """
            params = self._spec_params()
            _check_modification(sample, self.directory, "point_defects", params, force)
            rng = np.random.RandomState(self.seed) if self.seed is not None else np.random.RandomState()
            n_chunks = int(sample.chunk_total)

            # First pass: for random "count" (no fraction) compute candidate totals across all chunks.
            vac_need_fraction = (self.vacancy_count is not None and self.vacancy_fraction is None)
            sub_need_fraction = (self.substitution_count is not None and self.substitution_fraction is None)
            if vac_need_fraction or sub_need_fraction:
                vac_total_cand = 0
                sub_total_cand = 0
                for i in range(n_chunks):
                    pos_i = sample.load_chunk_positions(i+1, use_gpu=False, raw=True)
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
            r_min = float(self.interstitial_min_separation)

            self._applied_vacancies = []
            self._applied_substitutions = []
            self._applied_interstitials = []
            empty = np.zeros((0, 3), dtype=np.float32)

            for i in range(n_chunks):
                # Load
                pos = sample.load_chunk_positions(i+1, use_gpu=False, raw=True)
                spc = sample.load_chunk_species(i+1, use_gpu=False)
                if spc.dtype.kind != "U":
                    spc = spc.astype(str)

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
                sub_pos = empty
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

                    # Commit substitution: change species at subs_mask_local.
                    # Widen the string dtype if the new label is longer.
                    if np.any(subs_mask_local):
                        new_dtype = np.result_type(spc.dtype, np.asarray(str(self.substitution_to)).dtype)
                        spc = spc.astype(new_dtype, copy=True)
                        from_mask = (spc == str(self.substitution_from))
                        apply_mask = subs_mask_local & from_mask & (~delete_mask)
                        if np.any(apply_mask):
                            sub_pos = pos[apply_mask].astype(np.float32, copy=True)
                            spc[apply_mask] = str(self.substitution_to)

                # Positions of deleted atoms (for plotting/relax)
                vac_pos = pos[delete_mask].astype(np.float32, copy=True) if np.any(delete_mask) else empty

                # Remove vacancies
                keep_mask = ~delete_mask
                pos = pos[keep_mask]
                spc = spc[keep_mask]

                # Interstitials: candidates must keep `interstitial_min_separation`
                # from the chunk atoms and from interstitials already accepted
                # in this chunk.
                new_pos = empty
                tree = None
                if r_min > 0.0 and pos.shape[0] > 0:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(pos)

                # Interstitials: specific
                if interstitials_specific_left is not None and len(interstitials_specific_left) > 0:
                    # round-robin allocation of specific positions across chunks
                    slice_len = max(0, int(np.ceil(len(interstitials_specific_left) / float(n_chunks - i))))
                    if slice_len > 0:
                        sel_idx = interstitials_specific_left[:slice_len]
                        interstitials_specific_left = interstitials_specific_left[slice_len:]
                        P = self.interstitial_positions[sel_idx, :]
                        if self.region_min is not None and self.region_max is not None:
                            P = P[self._in_region_mask(P)]
                        new_pos = self._accept_min_sep(P, tree, r_min, new_pos)

                # Interstitials: random
                if interstitials_remaining_random > 0 and self.interstitial_species is not None:
                    to_take = int(np.ceil(interstitials_remaining_random / float(n_chunks - i)))
                    if to_take > 0:
                        box_min, box_max = self._region_or_sample_box(sample)
                        n_before = new_pos.shape[0]
                        tries = 0
                        while (new_pos.shape[0] - n_before) < to_take and tries < 50 * to_take:
                            need = to_take - (new_pos.shape[0] - n_before)
                            batch = int(min(max(2 * need, 64), 50 * to_take - tries))
                            cand = rng.uniform(low=box_min, high=box_max, size=(batch, 3)).astype(np.float32)
                            tries += batch
                            if self.region_min is not None and self.region_max is not None:
                                cand = cand[self._in_region_mask(cand)]
                            new_pos = self._accept_min_sep(cand, tree, r_min, new_pos, limit=need)
                        interstitials_remaining_random -= int(new_pos.shape[0] - n_before)

                # Append interstitials to chunk arrays
                if new_pos.shape[0] > 0:
                    pos = np.concatenate([pos, new_pos], axis=0)
                    spc = np.concatenate([spc, np.full(new_pos.shape[0], str(self.interstitial_species))], axis=0)
                    if spc.dtype.kind != "U":
                        spc = spc.astype(str)

                self._applied_vacancies.append(vac_pos)
                self._applied_substitutions.append(sub_pos)
                self._applied_interstitials.append(new_pos.astype(np.float32, copy=False))

                # Write updated arrays
                sample.write_chunk_positions(pos, i+1, override_directory=self.directory)
                sample.write_chunk_species(spc, i+1, override_directory=self.directory)

                # Advance global index window
                global_start += N

            # Applied positions and sample metadata in the chosen directory
            self._write_applied_arrays(self.directory if self.directory is not None else sample.directory)
            _record_modification(sample, self.directory, "point_defects", params)
            sample.write_sample_metadata(override_directory=self.directory)

            # Optional relaxation now
            if self._relax_after:
                params = self._relax_params if self._relax_params else {}
                self.relax_local_atoms(sample, **params)

        def _spec_params(self):
            """Defining parameters of this point-defect set, JSON-serialisable."""
            def _lst(v):
                return None if v is None else np.asarray(v).tolist()
            return {
                "seed": self.seed,
                "region_min": _lst(self.region_min), "region_max": _lst(self.region_max),
                "vacancy_fraction": self.vacancy_fraction, "vacancy_count": self.vacancy_count,
                "vacancy_global_indices": _lst(self.vacancy_global_indices),
                "vacancy_positions": _lst(self.vacancy_positions),
                "vacancy_species_filter": self.vacancy_species_filter,
                "substitution_fraction": self.substitution_fraction,
                "substitution_count": self.substitution_count,
                "substitution_from": self.substitution_from, "substitution_to": self.substitution_to,
                "substitution_positions": _lst(self.substitution_positions),
                "substitution_global_indices": _lst(self.substitution_global_indices),
                "interstitial_count": self.interstitial_count,
                "interstitial_positions": _lst(self.interstitial_positions),
                "interstitial_species": self.interstitial_species,
                "interstitial_min_separation": self.interstitial_min_separation,
            }

        @staticmethod
        def _applied_filename(kind, chunk_num):
            return f"point_defects_applied_{kind}_{chunk_num}.npy"

        def _write_applied_arrays(self, directory):
            """
            Save the applied vacancy/substitution/interstitial positions as
            one (M, 3) float32 `.npy` per chunk and kind in `directory`.
            """
            if directory is None:
                return
            os.makedirs(directory, exist_ok=True)
            self._applied_directory = os.path.abspath(directory)
            for kind, arrs in (("vacancies", self._applied_vacancies),
                               ("substitutions", self._applied_substitutions),
                               ("interstitials", self._applied_interstitials)):
                for k, arr in enumerate(arrs):
                    np.save(os.path.join(directory, self._applied_filename(kind, k + 1)),
                            np.asarray(arr, dtype=np.float32).reshape(-1, 3))

        def _load_applied_arrays(self, directory, n_chunks):
            """Load the per-chunk applied-position arrays written by `_write_applied_arrays`."""
            self._applied_directory = os.path.abspath(directory)
            for kind, attr in (("vacancies", "_applied_vacancies"),
                               ("substitutions", "_applied_substitutions"),
                               ("interstitials", "_applied_interstitials")):
                arrs = []
                for k in range(int(n_chunks)):
                    path = os.path.join(directory, self._applied_filename(kind, k + 1))
                    arrs.append(np.load(path).reshape(-1, 3) if os.path.isfile(path)
                                else np.zeros((0, 3), dtype=np.float32))
                setattr(self, attr, arrs)

        def _applied_centers(self):
            """Concatenated (vacancy, substitution, interstitial) positions, each (M, 3) float32."""
            def _cat(arrs):
                arrs = [np.asarray(a, dtype=np.float32).reshape(-1, 3) for a in arrs]
                return np.concatenate(arrs, axis=0) if arrs else np.zeros((0, 3), dtype=np.float32)
            return (_cat(self._applied_vacancies), _cat(self._applied_substitutions),
                    _cat(self._applied_interstitials))

        # ------------------------
        # Relaxation

        def relax_local_atoms(self,
                              sample,
                              r_cut=2.0,
                              strength=0.05,
                              iterations=2,
                              decay=0.8,
                              use_gpu=False,
                              force=False):
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
                force: If True, run even if the same relaxation was already
                    recorded on the sample. Defaults to False.

            Raises:
                RuntimeError: If the same relaxation was already applied in
                    place and `force` is False.

            Note:
                The update rule per center c and neighbor x within r_cut is:
                    w = exp(-(r/r_cut)^2)
                    delta = sgn * strength * w * (x - c) / (r + 1e-12)
                where sgn is +1 for interstitials (push), -1 for vacancies (pull),
                and -0.5 for substitutions. Strength decays each iteration by decay.
                Neighbours are found with a k-d tree on the chunk positions.
                Operates chunk-by-chunk; atoms are clamped to sample AABB.
            """
            V, S, I = self._applied_centers()

            box_min, box_max = self._region_or_sample_box(sample)

            # Early out if nothing to do
            if V.shape[0] == 0 and S.shape[0] == 0 and I.shape[0] == 0:
                return

            params = {"r_cut": float(r_cut), "strength": float(strength),
                      "iterations": int(iterations), "decay": float(decay),
                      "centers": [int(V.shape[0]), int(S.shape[0]), int(I.shape[0])]}
            _check_modification(sample, self.directory, "point_defect_relax", params, force)

            n_chunks = int(sample.chunk_total)
            for it in range(int(iterations)):
                step = float(strength) * (float(decay) ** it)
                for i in range(n_chunks):
                    pos = sample.load_chunk_positions(i+1, use_gpu=False, raw=True)
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

            _record_modification(sample, self.directory, "point_defect_relax", params)
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
            V, S, I = self._applied_centers()
            if V.shape[0]:
                ax.scatter(V[:,0], V[:,1], V[:,2], s=12, c="r", marker="x", label="vacancy")
            if S.shape[0]:
                ax.scatter(S[:,0], S[:,1], S[:,2], s=10, c="g", marker="o", label="substitution")
            if I.shape[0]:
                ax.scatter(I[:,0], I[:,1], I[:,2], s=10, c="m", marker="^", label="interstitial")

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
                allowed = np.zeros(spc.shape[0], dtype=bool)
                for s in self.vacancy_species_filter:
                    allowed |= (spc == str(s))
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
                mask &= (spc == str(self.substitution_from))
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

        def _accept_min_sep(self, cand, tree, r_min, accepted, limit=None):
            """
            Append candidate positions that keep a minimum separation from
            the chunk atoms and from each other.

            Args:
                cand: (K, 3) candidate positions.
                tree: scipy.spatial.cKDTree over the chunk atoms, or None.
                r_min: Minimum separation; <= 0 accepts every candidate.
                accepted: (M, 3) positions already accepted in this chunk.
                limit: Optional maximum number of candidates to add.

            Returns:
                (M + m, 3) float32 array of accepted positions.
            """
            cand = np.asarray(cand, dtype=np.float32).reshape(-1, 3)
            accepted = np.asarray(accepted, dtype=np.float32).reshape(-1, 3)
            if cand.shape[0] == 0:
                return accepted
            if r_min <= 0.0:
                take = cand if limit is None else cand[:int(limit)]
                return np.concatenate([accepted, take], axis=0)
            if tree is not None:
                ok = tree.query_ball_point(cand, float(r_min), return_length=True) == 0
                cand = cand[ok]
            r2 = float(r_min) * float(r_min)
            out = [accepted]
            acc = accepted
            n_added = 0
            for p in cand:
                if limit is not None and n_added >= int(limit):
                    break
                if acc.shape[0] > 0:
                    d = acc - p[None, :]
                    if np.min(np.einsum("ij,ij->i", d, d)) < r2:
                        continue
                out.append(p[None, :])
                acc = np.concatenate([acc, p[None, :]], axis=0)
                n_added += 1
            return acc

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
            Accumulate radial displacements from defect centers. Neighbours
            within `r_cut` of each centre are found with a k-d tree on `pos`.

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
            from scipy.spatial import cKDTree
            acc = np.zeros((pos.shape[0], 3), dtype=np.float64)
            rc = float(r_cut)
            tree = cKDTree(pos)
            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
            batch = 4096
            for c0 in range(0, centers.shape[0], batch):
                cs = centers[c0:c0 + batch]
                lists = tree.query_ball_point(cs, rc)
                lens = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
                if lens.sum() == 0:
                    continue
                j = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists if len(l)])
                ci = np.repeat(np.arange(len(lists)), lens)
                v = pos[j].astype(np.float64) - cs[ci].astype(np.float64)
                r = np.linalg.norm(v, axis=1)
                m = r > 1e-12
                if not np.any(m):
                    continue
                w = np.exp(-(r[m] / rc) ** 2)
                np.add.at(acc, j[m], (step_signed * w)[:, None] * v[m] / r[m][:, None])
            return acc.astype(np.float32)
