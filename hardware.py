# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
import os
import sys
import json
import time
import platform
import threading
import warnings

import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
try:
    import cupy as cp
except ImportError:
    cp = None

# -----------------------------------------------------------------------------
# Purpose
# -----------------------------------------------------------------------------
# One place that knows what machine the code is running on and how much of it
# each path may use.  Everything that used to be a fixed constant (streams per
# GPU, chunks in flight, batch rows, thread counts, memory fractions) is derived
# from the Profile returned by probe(), through the two governors below, with
# the old environment variables and call arguments kept as overrides.
#
# Overrides (environment):
#   MOSAIC_HOST_MEM_LIMIT   pretend the machine has this much RAM ("16G")
#   MOSAIC_GPU_MEM_LIMIT    cap the CuPy pool on every device ("4G", "50%")
#   MOSAIC_RESERVE          host reserve kept free ("2G"), default max(2G, 12 %)
#   MOSAIC_DEVICE_RESERVE   device reserve ("512M"), default max(512M, 10 %)
#   MOSAIC_DEVICES          comma-separated CUDA device indices to use
#   MOSAIC_CPU_THREADS      worker threads for CPU paths
#   MOSAIC_LAUNCH_CAP_S     seconds one kernel launch may take
#   MOSAIC_CUDA_BACKEND     "nvrtc" (default) or "nvcc"
#   MOSAIC_HOME             directory for calibration files (~/.mosaic)
#   MOSAIC_MONITOR=0        no background host-memory monitor
#   MOSAIC_CHUNK_CACHE=0    no host chunk cache (Sample.py)
#   MOSAIC_CPU_SLICE_ATOMS, MOSAIC_DEVICE_SLICE_ATOMS   pin slice sizes (tests)
#   SAMPLE_STREAMS_PER_GPU, BEAM_STREAMS_PER_GPU, BEAM_EIN_STREAMS_PER_GPU,
#   BEAM_EIN_SAVE_THREADS   the historical stream and thread pins still win

_GIB = 1024 ** 3
_MIB = 1024 ** 2


def _log(level, msg):
    """Console line in the project's [name|LEVEL] style."""
    print(f"[hardware|{str(level).upper()}] {msg}")


# -----------------------------------------------------------------------------
# Runtime compilation
# -----------------------------------------------------------------------------
# Every CUDA kernel in the code base is compiled at run time from a source
# string.  NVRTC ships inside the CuPy wheel, so it is the default; nvcc needs
# the full CUDA toolkit on the machine and is kept only as a debugging aid
# (MOSAIC_CUDA_BACKEND=nvcc).  Both produce bit-identical kernels here; the
# nvcc-only flags (-O3 is host-side, --gpu-architecture=native is picked by
# CuPy for NVRTC) are added only when nvcc is selected.

_CUDA_FLAGS_COMMON = ('--ftz=true', '--fmad=true')
_CUDA_FLAGS_NVCC = ('--gpu-architecture=native', '-O3')


def cuda_backend():
    """Compile backend for cupy.RawModule: 'nvrtc' unless overridden."""
    name = os.environ.get("MOSAIC_CUDA_BACKEND", "nvrtc").strip().lower()
    return "nvcc" if name == "nvcc" else "nvrtc"


def cuda_options(*extra):
    """Compile options for cupy.RawModule under the selected backend.

    Args:
        *extra: Further options, typically ``-D`` macro definitions.

    Returns:
        tuple[str, ...]: Options accepted by the backend from
        :func:`cuda_backend`.
    """
    base = _CUDA_FLAGS_COMMON
    if cuda_backend() == "nvcc":
        base = _CUDA_FLAGS_NVCC + base
    return tuple(base) + tuple(extra)


def raw_module(code, *extra, **kwargs):
    """cupy.RawModule with the backend and options from this module.

    Kernel sources may include host headers behind ``#ifndef __CUDACC_RTC__``;
    NVRTC defines that macro and carries no headers of its own.
    """
    if cp is None:
        raise RuntimeError("CuPy is required to compile CUDA kernels")
    return cp.RawModule(code=code, backend=cuda_backend(),
                        options=cuda_options(*extra), **kwargs)


class CompilerUnavailable(RuntimeError):
    """Raised when a CPU kernel cannot be compiled on this machine."""


def cffi_compile_args():
    """Flags for the C compiler cffi drives.

    MSVC: /O2, plus /arch:AVX2 when the CPU has it; gcc and clang: -O3
    -march=native.  The kernels are compiled on the machine that runs them,
    so tuning for its own instruction set is safe, and cffi keys its build
    cache on these flags, so a change rebuilds the module.
    """
    if sys.platform == "win32" and "GCC" not in sys.version:
        args = ["/O2"]
        try:
            if "AVX2" in _simd_features():
                args.append("/arch:AVX2")
        except Exception:
            pass
        return args
    return ["-O3", "-march=native"]


def _compiler_hint():
    """How to get a C compiler on this platform, for the error message."""
    if sys.platform == "win32":
        return ("install the Microsoft C++ Build Tools (Visual Studio "
                "Installer, 'Desktop development with C++')")
    if sys.platform == "darwin":
        return "install the Xcode command line tools (xcode-select --install)"
    return "install gcc (for example 'apt install build-essential')"


def cffi_verify(ffi, source, what, **kwargs):
    """Compile a cffi C source, or raise :class:`CompilerUnavailable`.

    cffi's ``verify`` needs a working C compiler on the machine.  Its own
    failure surfaces as a distutils error deep in a traceback, so the message
    here names the kernel that needed it and how to get a compiler.

    Args:
        ffi: cffi.FFI instance with the cdef already applied.
        source (str): C source.
        what (str): Name of the kernel, for the message.
        **kwargs: Passed to ``ffi.verify`` (extra_compile_args, libraries...).
    """
    try:
        return ffi.verify(source, **kwargs)
    except Exception as exc:
        first = str(exc).splitlines()[0] if str(exc) else repr(exc)
        raise CompilerUnavailable(
            f"The CPU kernel '{what}' could not be compiled on this machine "
            f"({type(exc).__name__}: {first}). A C compiler is required for "
            f"the CPU path; {_compiler_hint()}, or run on a machine with a "
            f"CUDA GPU.") from exc


# -----------------------------------------------------------------------------
# Sizes and overrides
# -----------------------------------------------------------------------------
def parse_bytes(text, total=None):
    """Parse a size such as '16G', '512MiB' or '50%' into bytes.

    Args:
        text (str | int | float | None): Size with an optional unit (K, M,
            G, T, with or without B/iB) or a percentage of ``total``.
            Numbers pass through; None returns None.
        total (int, optional): Reference for percentages.

    Returns:
        int or None: The size in bytes.
    """
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return int(text)
    s = str(text).strip().replace(" ", "")
    if not s:
        return None
    if s.endswith("%"):
        if total is None:
            raise ValueError("percentage needs a total")
        return int(float(s[:-1]) / 100.0 * total)
    units = {"k": 1024, "m": _MIB, "g": _GIB, "t": 1024 ** 4}
    low = s.lower()
    for suffix in ("ib", "b"):
        if low.endswith(suffix) and len(low) > len(suffix) and low[-len(suffix) - 1] in units:
            low = low[:-len(suffix)]
            break
    if low[-1] in units:
        return int(float(low[:-1]) * units[low[-1]])
    return int(float(low))


def _env(name, default=None):
    """Environment variable ``name``, or ``default`` when unset or empty."""
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v


def _env_int(name, default=None):
    """Environment variable ``name`` as an int, or ``default``."""
    v = _env(name)
    return default if v is None else int(v)


def _env_float(name, default=None):
    """Environment variable ``name`` as a float, or ``default``."""
    v = _env(name)
    return default if v is None else float(v)


def mosaic_home():
    """Per-user directory for calibration and pinned settings."""
    d = _env("MOSAIC_HOME") or os.path.join(os.path.expanduser("~"), ".mosaic")
    os.makedirs(d, exist_ok=True)
    return d


def _pinned_settings():
    """User pins from MOSAIC_HOME/hardware.json (same keys as the env vars)."""
    path = os.path.join(mosaic_home(), "hardware.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        warnings.warn(f"hardware.json ignored: {exc}")
        return {}


def setting(name, default=None):
    """Override lookup: the environment first, then hardware.json, then ``default``.

    Args:
        name (str): Variable name such as ``MOSAIC_DEVICES``.
        default: Value when neither source defines it.

    Returns:
        The value as a string, or ``default``.
    """
    v = _env(name)
    if v is not None:
        return v
    return _pinned_settings().get(name, default)


# -----------------------------------------------------------------------------
# Profile
# -----------------------------------------------------------------------------
class HostInfo:
    """What the host offers: RAM (under the MOSAIC_HOST_MEM_LIMIT pretence),
    cores, SIMD features and whether a C compiler is available.
    """
    def __init__(self):
        self.os = platform.system()
        self.machine = platform.machine()
        self.python = platform.python_version()
        self.psutil_ok = psutil is not None
        if psutil is not None:
            vm = psutil.virtual_memory()
            self.ram_total = int(vm.total)
            self.cores_physical = int(psutil.cpu_count(logical=False) or 1)
            self.cores_logical = int(psutil.cpu_count(logical=True) or 1)
        else:
            warnings.warn("psutil is not installed; host memory is assumed to "
                          "be 16 GB. Install psutil for machine-aware budgets.")
            self.ram_total = 16 * _GIB
            self.cores_physical = int(os.cpu_count() or 1)
            self.cores_logical = int(os.cpu_count() or 1)
        limit = parse_bytes(setting("MOSAIC_HOST_MEM_LIMIT"), self.ram_total)
        self.ram_limit = int(min(self.ram_total, limit)) if limit else self.ram_total
        self.simd = _simd_features()
        self._c_compiler = None

    def available(self):
        """Bytes this process may still allocate on the host, honouring MOSAIC_HOST_MEM_LIMIT."""
        if psutil is None:
            return max(0, self.ram_limit // 2)
        avail = int(psutil.virtual_memory().available)
        if self.ram_limit < self.ram_total:
            avail = min(avail, self.ram_limit - self.rss())
        return max(0, avail)

    @staticmethod
    def rss():
        """Resident set size of this process in bytes (0 without psutil)."""
        if psutil is None:
            return 0
        try:
            return int(psutil.Process().memory_info().rss)
        except Exception:
            return 0

    @property
    def c_compiler(self):
        """True when cffi can build a C extension here; probed once, lazily."""
        if self._c_compiler is None:
            self._c_compiler = _probe_c_compiler()
        return self._c_compiler


def _simd_features():
    """Names of the SIMD features NumPy reports as available on this CPU."""
    feats = set()
    for modname in ("numpy._core._multiarray_umath", "numpy.core._multiarray_umath"):
        try:
            mod = __import__(modname, fromlist=["__cpu_features__"])
            table = getattr(mod, "__cpu_features__", {})
            feats = {k for k, v in table.items() if v}
            break
        except Exception:
            continue
    return feats


def _probe_c_compiler():
    """True when cffi can build and load a trivial C extension here."""
    try:
        from cffi import FFI
        ffi = FFI()
        ffi.cdef("int mosaic_probe(int x);")
        lib = ffi.verify("int mosaic_probe(int x) { return x + 1; }")
        return int(lib.mosaic_probe(1)) == 2
    except Exception:
        return False


class GpuInfo:
    """One CUDA device: architecture, memory, watchdog and pool state.

    ``watchdog`` is the ``kernelExecTimeoutEnabled`` property, true on
    Windows display devices and Linux devices driving X, where a launch
    longer than about two seconds is killed by the driver.
    """
    def __init__(self, index):
        self.index = int(index)
        props = cp.cuda.runtime.getDeviceProperties(self.index)
        name = props.get("name", b"")
        self.name = name.decode() if isinstance(name, bytes) else str(name)
        self.cc = (int(props.get("major", 0)), int(props.get("minor", 0)))
        self.sm_count = int(props.get("multiProcessorCount", 1))
        self.regs_per_block = int(props.get("regsPerBlock", 65536))
        self.shared_per_block = int(props.get("sharedMemPerBlock", 49152))
        self.mem_total = int(props.get("totalGlobalMem", 0))
        self.watchdog = bool(props.get("kernelExecTimeoutEnabled", 0))
        self.tcc = bool(props.get("tccDriver", 0))
        self.integrated = bool(props.get("integrated", 0))
        self.clock_khz = int(props.get("clockRate", 0))
        # A limit already on the pool (CUPY_GPU_MEMORY_LIMIT, or an earlier
        # probe in this process) is part of what this process may use.
        try:
            with cp.cuda.Device(self.index):
                self.pool_limit = int(cp.get_default_memory_pool().get_limit())
        except Exception:
            self.pool_limit = 0

    def mem_effective(self):
        """Memory this process may use on the device.

        The pool cap when one is set, else the whole card.
        """
        return int(self.pool_limit) if self.pool_limit else int(self.mem_total)

    def mem_free(self):
        """Bytes allocatable on this device now.

        Driver free memory plus the pool's idle blocks, capped by the pool
        limit when one is set.
        """
        with cp.cuda.Device(self.index):
            free, _total = cp.cuda.runtime.memGetInfo()
            pool = cp.get_default_memory_pool()
            free = int(free) + int(pool.free_bytes())
            limit = int(pool.get_limit())
            if limit > 0:
                free = min(free, limit - int(pool.used_bytes()))
        return max(0, int(free))

    def fingerprint(self):
        """Identity for calibration files: card, architecture and software stack."""
        cuda = CudaInfo()
        raw = f"{self.name}|sm{self.cc[0]}{self.cc[1]}|drv{cuda.driver}|rt{cuda.runtime}|nvrtc{cuda.nvrtc}|cupy{cuda.cupy}"
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)


class CudaInfo:
    """Versions of the driver, runtime, NVRTC and CuPy in use (None without CuPy)."""
    def __init__(self):
        self.driver = self.runtime = self.nvrtc = None
        self.cupy = None
        if cp is None:
            return
        try:
            self.cupy = cp.__version__
            self.driver = int(cp.cuda.runtime.driverGetVersion())
            self.runtime = int(cp.cuda.runtime.runtimeGetVersion())
            major, minor = cp.cuda.nvrtc.getVersion()
            self.nvrtc = f"{major}.{minor}"
        except Exception:
            pass


class Profile:
    """The machine as this process sees it: host, CUDA stack and usable devices.

    Built once by :func:`probe`.  MOSAIC_DEVICES restricts the devices,
    MOSAIC_GPU_MEM_LIMIT caps every device's CuPy pool.
    """
    def __init__(self):
        self.host = HostInfo()
        self.cuda = CudaInfo()
        self.gpus = []
        self.gpu_error = None
        if cp is not None:
            try:
                n = int(cp.cuda.runtime.getDeviceCount())
            except Exception as exc:
                n = 0
                self.gpu_error = str(exc).splitlines()[0] if str(exc) else repr(exc)
            wanted = setting("MOSAIC_DEVICES")
            if wanted is not None and str(wanted).strip() != "":
                indices = [int(x) for x in str(wanted).split(",") if x.strip() != ""]
                indices = [i for i in indices if 0 <= i < n]
            else:
                indices = list(range(n))
            for i in indices:
                try:
                    self.gpus.append(GpuInfo(i))
                except Exception as exc:
                    self.gpu_error = f"device {i}: {exc}"
            self._apply_pool_limits()
        self.backend = cuda_backend() if self.gpus else None

    # ---- devices
    @property
    def n_gpus(self):
        """Number of usable devices."""
        return len(self.gpus)

    def gpu_indices(self):
        """CUDA indices of the usable devices, in profile order."""
        return [g.index for g in self.gpus]

    def gpu(self, index):
        """GpuInfo for a CUDA device index.

        Args:
            index (int): CUDA device index.

        Returns:
            GpuInfo: The device.

        Raises:
            KeyError: If the device is not in the profile.
        """
        for g in self.gpus:
            if g.index == int(index):
                return g
        raise KeyError(f"device {index} is not in the profile (MOSAIC_DEVICES={setting('MOSAIC_DEVICES')})")

    def _apply_pool_limits(self):
        """MOSAIC_GPU_MEM_LIMIT caps the CuPy pool on every device; "0" lifts the cap."""
        spec = setting("MOSAIC_GPU_MEM_LIMIT")
        if spec is None:
            return
        for g in self.gpus:
            try:
                limit = 0 if str(spec).strip().lower() in ("0", "none", "off") else parse_bytes(spec, g.mem_total)
                with cp.cuda.Device(g.index):
                    cp.get_default_memory_pool().set_limit(size=int(limit))
                g.pool_limit = int(limit)
            except Exception as exc:
                warnings.warn(f"could not apply MOSAIC_GPU_MEM_LIMIT on device {g.index}: {exc}")

    # ---- summary
    def to_dict(self):
        """JSON-serialisable copy of the profile (sets become sorted lists)."""
        return {
            "host": {k: (sorted(v) if isinstance(v, set) else v)
                     for k, v in vars(self.host).items() if not k.startswith("_")},
            "cuda": vars(self.cuda),
            "gpus": [{k: v for k, v in vars(g).items()} for g in self.gpus],
            "backend": self.backend,
        }

    def report(self):
        """Multi-line text summary for the console, the GUI panel and bug reports."""
        h = self.host
        lines = [
            f"Host: {h.os} {h.machine}, Python {h.python}, "
            f"{h.cores_physical} cores / {h.cores_logical} threads, "
            f"RAM {h.ram_total / _GIB:.1f} GB total, {h.available() / _GIB:.1f} GB available"
            + (f" (limited to {h.ram_limit / _GIB:.1f} GB)" if h.ram_limit < h.ram_total else ""),
            f"      SIMD: {', '.join(sorted(f for f in h.simd if f.startswith(('AVX', 'NEON', 'SSE4')))) or 'none reported'}; "
            f"psutil: {'yes' if h.psutil_ok else 'MISSING'}",
        ]
        if cp is None:
            lines.append("GPU:  CuPy not installed; CPU paths only")
        elif not self.gpus:
            lines.append(f"GPU:  none usable ({self.gpu_error or 'no CUDA device'}); CPU paths only")
        else:
            lines.append(f"CUDA: driver {self.cuda.driver}, runtime {self.cuda.runtime}, "
                         f"NVRTC {self.cuda.nvrtc}, cupy {self.cuda.cupy}, kernels via {self.backend}")
            for g in self.gpus:
                lines.append(
                    f"GPU {g.index}: {g.name}, sm_{g.cc[0]}{g.cc[1]}, {g.sm_count} SMs, "
                    f"{g.mem_total / _GIB:.1f} GB ({g.mem_free() / _GIB:.1f} GB free"
                    + (f", pool capped at {g.pool_limit / _GIB:.1f} GB" if g.pool_limit else "") + "), "
                    + ("display watchdog ON" if g.watchdog else "no watchdog")
                    + (", TCC" if g.tcc else "") + (", integrated" if g.integrated else ""))
                hg = host_governor()
                lines.append(f"      launch cap {launch_cap_s(g.index):.2f} s, device reserve "
                             f"{device_governor(g.index).reserve / _GIB:.2f} GB, host reserve {hg.reserve / _GIB:.2f} GB")
        return "\n".join(lines)


_PROFILE = None
_PROFILE_LOCK = threading.Lock()


def probe(refresh=False):
    """The machine profile, built once per process.

    Args:
        refresh (bool, optional): Rebuild it (after changing MOSAIC_*
            overrides in the environment). Defaults to False.

    Returns:
        Profile: The cached profile.
    """
    global _PROFILE
    with _PROFILE_LOCK:
        if _PROFILE is None or refresh:
            _PROFILE = Profile()
        return _PROFILE


def report():
    """Text summary of the profile (see :meth:`Profile.report`)."""
    return probe().report()


# -----------------------------------------------------------------------------
# Governors
# -----------------------------------------------------------------------------
class Reservation:
    """A working set claimed against a governor; commit() once it is allocated."""

    def __init__(self, governor, name, nbytes):
        self.governor = governor
        self.name = name
        self.nbytes = int(nbytes)
        self.committed = False
        self.released = False

    def commit(self):
        """The buffers now exist, so free-memory readings already include them."""
        self.committed = True
        return self

    def release(self):
        """Drop the claim; the ledger no longer counts it."""
        self.governor._release(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class _Governor:
    """Budget = free memory now - reserve - working sets planned but not yet allocated.

    The reserve is what stays free for the OS, the display driver and any other
    process.  A working set is reserved before its buffers are allocated and
    committed after, so the ledger never double-counts memory that the free
    reading already reflects.  Evictors (the chunk cache) are called when a
    plan does not fit.
    """

    def __init__(self, reserve):
        self.reserve = int(reserve)
        self._lock = threading.Lock()
        self._ledger = {}
        self._evictors = []
        self._seq = 0

    def free_now(self):
        """Free memory of this kind right now, in bytes."""
        raise NotImplementedError

    def budget(self):
        """Free memory minus the reserve, in bytes."""
        return self.free_now() - self.reserve

    def pending(self):
        """Bytes of working sets claimed but not yet allocated."""
        with self._lock:
            return sum(r.nbytes for r in self._ledger.values() if not r.committed)

    def allowance(self):
        """Bytes a new working set may take right now."""
        return self.budget() - self.pending()

    def reserve_bytes(self, name, nbytes):
        """Claim a working set before allocating it.

        Args:
            name (str): Label for logs.
            nbytes (int): Bytes the working set will take.

        Returns:
            Reservation: Call ``commit()`` once the buffers exist and
            ``release()`` when they are gone; usable as a context manager.
        """
        r = Reservation(self, name, nbytes)
        with self._lock:
            self._seq += 1
            self._ledger[self._seq] = r
            r._key = self._seq
        return r

    def _release(self, r):
        """Remove a reservation from the ledger."""
        with self._lock:
            self._ledger.pop(getattr(r, "_key", None), None)
            r.released = True

    def register_evictor(self, fn):
        """fn(nbytes) -> bytes freed; called when a plan needs room."""
        self._evictors.append(fn)

    def make_room(self, nbytes):
        """Ask evictors for ``nbytes``; returns the allowance afterwards."""
        need = int(nbytes) - self.allowance()
        for fn in self._evictors:
            if need <= 0:
                break
            try:
                need -= int(fn(need))
            except Exception:
                continue
        return self.allowance()

    def plan(self, bytes_per_item, want, floor=1, fraction=1.0):
        """How many items of ``bytes_per_item`` fit the allowance right now.

        Evictors are asked for room first when the full request does not
        fit.

        Args:
            bytes_per_item (int): Bytes one item takes.
            want (int): Items wanted.
            floor (int, optional): Never fewer than this. Defaults to 1.
            fraction (float, optional): Share of the allowance the items may
                take. Defaults to 1.0.

        Returns:
            int: Items that fit, between ``floor`` and ``want``.
        """
        want = int(max(floor, want))
        per = max(1, int(bytes_per_item))
        allow = self.allowance()
        if allow < per * want:
            allow = self.make_room(per * want)
        n = int((allow * float(fraction)) // per)
        return int(min(want, max(floor, n)))

    def check(self):
        """False when free memory has dropped into the reserve."""
        return self.free_now() > self.reserve


class HostGovernor(_Governor):
    """Governor of host RAM.

    The reserve keeps the OS, the display driver and other processes
    running; MOSAIC_RESERVE overrides it.  Starts the background monitor
    unless MOSAIC_MONITOR=0.
    """
    def __init__(self):
        host = probe().host
        spec = setting("MOSAIC_RESERVE")
        reserve = parse_bytes(spec, host.ram_limit) if spec else max(2 * _GIB, int(0.12 * host.ram_limit))
        super().__init__(reserve)
        self.host = host
        # The monitor only acts once free memory falls into the reserve;
        # MOSAIC_MONITOR=0 leaves it off.
        if str(setting("MOSAIC_MONITOR", "1")).strip().lower() not in ("0", "off", "false"):
            try:
                start_monitor()
            except Exception:
                pass

    def free_now(self):
        """Bytes this process may still allocate on the host."""
        return self.host.available()

    def rss(self):
        """Resident set size of this process in bytes."""
        return self.host.rss()


class DeviceGovernor(_Governor):
    """Governor of one CUDA device's memory.

    Free memory is pool-aware (see :meth:`GpuInfo.mem_free`); the
    reserve is 10 % of the device, or of the pool cap when one is set,
    and at least 512 MB.  MOSAIC_DEVICE_RESERVE overrides it.
    """
    def __init__(self, index):
        gpu = probe().gpu(index)
        spec = setting("MOSAIC_DEVICE_RESERVE")
        total = gpu.mem_effective()
        reserve = parse_bytes(spec, total) if spec else max(512 * _MIB, int(0.10 * total))
        super().__init__(reserve)
        self.gpu = gpu

    def free_now(self):
        """Bytes allocatable on the device now, pool-aware."""
        return self.gpu.mem_free()


_HOST_GOV = None
_DEV_GOVS = {}
_GOV_LOCK = threading.Lock()


def host_governor():
    """The process-wide host governor, created on first use."""
    global _HOST_GOV
    with _GOV_LOCK:
        if _HOST_GOV is None:
            _HOST_GOV = HostGovernor()
        return _HOST_GOV


def device_governor(index=None):
    """The governor of a CUDA device, created on first use.

    Args:
        index (int, optional): CUDA device index; None means the current
            device.

    Returns:
        DeviceGovernor: The governor.
    """
    if index is None:
        index = int(cp.cuda.Device().id)
    with _GOV_LOCK:
        g = _DEV_GOVS.get(int(index))
        if g is None:
            g = _DEV_GOVS[int(index)] = DeviceGovernor(index)
        return g


def reset_governors():
    """Forget budgets and ledgers (tests, or after MOSAIC_* overrides change)."""
    global _HOST_GOV, _DEV_GOVS
    with _GOV_LOCK:
        _HOST_GOV = None
        _DEV_GOVS = {}


# -----------------------------------------------------------------------------
# Derived defaults
# -----------------------------------------------------------------------------
def cpu_slice_atoms(threads, npix, floor=20_000, cap=2_000_000):
    """Atoms per CPU scatter task, so all threads' tables fit half the host allowance.

    Each task holds its slice's per-atom tables plus double accumulators and
    a float output over the pixels.  MOSAIC_CPU_SLICE_ATOMS overrides.
    """
    env = _env_int("MOSAIC_CPU_SLICE_ATOMS")
    if env is not None:
        return max(1, env)
    per_thread = 0.5 * host_governor().allowance() / max(1, int(threads))
    per_thread -= 24.0 * float(npix)
    n = per_thread / bytes_per("cpu_scatter_host_per_atom")
    return int(min(cap, max(floor, n)))


def cpu_threads(default=None):
    """Worker threads for CPU paths.

    Args:
        default (int, optional): Used when MOSAIC_CPU_THREADS is unset;
            None means every logical core.

    Returns:
        int: Threads, at least 1.
    """
    n = _env_int("MOSAIC_CPU_THREADS")
    if n is None:
        n = default if default is not None else probe().host.cores_logical
    return max(1, int(n))


def has_watchdog(index=None):
    """Whether the display driver kills long launches on a device.

    Args:
        index (int, optional): CUDA device index; None means the current
            device.

    Returns:
        bool: True under a watchdog (also when the device is unknown).
    """
    if cp is None:
        return False
    try:
        return probe().gpu(int(cp.cuda.Device().id) if index is None else index).watchdog
    except Exception:
        return True


def launch_cap_s(index=None):
    """Seconds one kernel launch may run: 0.5 under a display watchdog, 4 otherwise.

    MOSAIC_LAUNCH_CAP_S=0 (or "inf"/"off") disables the cap, which pins the
    launch sizes to the memory bound alone for bit-for-bit comparisons.
    """
    spec = _env("MOSAIC_LAUNCH_CAP_S")
    if spec is not None:
        if str(spec).strip().lower() in ("0", "inf", "off", "none"):
            return float("inf")
        return float(spec)
    return 0.5 if has_watchdog(index) else 4.0


DEFAULT_THROUGHPUT = 5e10   #: atom.px/s assumed for a GPU before calibration


def kernel_throughput(index=None, kernel="fast"):
    """atom.px/s of a kernel on a device: the live estimate, the calibration
    file, or the conservative default, in that order.

    Args:
        index (int, optional): CUDA device index; None means the current
            device.
        kernel (str, optional): "fast" or "general". Defaults to "fast".

    Returns:
        float: Atom-pixels per second.
    """
    t = _LAUNCH_TIMERS.get((_dev_index(index), kernel))
    if t is not None and t.throughput is not None:
        return float(t.throughput)
    data = load_calibration(index)
    for key in ("live", "best"):
        try:
            return float(data["throughput"][kernel][key])
        except Exception:
            continue
    return DEFAULT_THROUGHPUT


def _dev_index(index=None):
    """CUDA device index: ``index`` when given, else the current device (0 without CuPy)."""
    if index is not None:
        return int(index)
    try:
        return int(cp.cuda.Device().id)
    except Exception:
        return 0


class LaunchTimer:
    """Keeps one kernel's launches under the device's time cap.

    Every launch is bracketed by CUDA events; completed ones are harvested on
    the next call without blocking, and the atom.px/s estimate is updated as
    a moving average.  :meth:`atoms` then returns how many atoms the next
    launch may carry so it finishes within half the cap.  The estimate starts
    from the calibration file or the conservative default, so the first
    launches are small and grow as measurements arrive.
    """

    ALPHA = 0.5          # weight of the newest measurement
    TARGET = 0.5         # fraction of the cap a launch aims at

    def __init__(self, kernel, index=None):
        self.kernel = kernel
        self.index = _dev_index(index)
        self.cap_s = launch_cap_s(self.index)
        self.throughput = None
        seed = kernel_throughput(self.index, kernel)
        self._seed = float(seed)
        self._pending = []
        self._lock = threading.Lock()
        self.samples = 0
        self._persisted = 0

    def estimate(self):
        """Current atom.px/s: the live average, else the seed."""
        return float(self.throughput if self.throughput is not None else self._seed)

    def atoms(self, npix, chunk=128, floor=None):
        """Atoms the next launch may carry over ``npix`` pixels.

        Completed launches are harvested first.  The count is a power of
        two multiple of ``chunk``, so run-to-run timing noise does not
        change the partition.

        Args:
            npix (int): Pixels the launch covers.
            chunk (int, optional): Kernel chunk size the count is a multiple
                of. Defaults to 128.
            floor (int, optional): Never fewer than this. Defaults to
                ``chunk``.

        Returns:
            int or None: Atoms per launch, or None when the cap is off.
        """
        self.harvest()
        if not (self.cap_s > 0) or self.cap_s == float("inf"):
            return None
        n = self.estimate() * self.cap_s * self.TARGET / max(1.0, float(npix))
        floor = int(floor if floor is not None else chunk)
        # Quantised to a power of two multiple of the chunk, so run-to-run
        # timing noise does not change the launch partition.
        units = max(1, int(n // chunk))
        units = 1 << (units.bit_length() - 1)
        return int(max(floor, units * chunk))

    def record(self, ev0, ev1, atom_px):
        """Register a launch bracketed by two recorded events.

        Args:
            ev0 (cupy.cuda.Event): Recorded before the launch.
            ev1 (cupy.cuda.Event): Recorded after it.
            atom_px (float): Atom-pixels the launch computed.
        """
        with self._lock:
            self._pending.append((ev0, ev1, float(atom_px)))

    def harvest(self):
        """Fold every completed launch into the estimate; never waits."""
        if cp is None:
            return
        with self._lock:
            keep = []
            for ev0, ev1, atom_px in self._pending:
                try:
                    if not ev1.done:
                        keep.append((ev0, ev1, atom_px))
                        continue
                    ms = float(cp.cuda.get_elapsed_time(ev0, ev1))
                except Exception:
                    continue
                if ms <= 0.05:
                    continue                       # too short to measure
                rate = atom_px / (ms * 1e-3)
                self.throughput = rate if self.throughput is None else (
                    (1 - self.ALPHA) * self.throughput + self.ALPHA * rate)
                self.samples += 1
                if ms > self.cap_s * 1e3:
                    _log("verbose", f"{self.kernel} launch took {ms / 1e3:.2f} s, above the "
                                    f"{self.cap_s:.2f} s cap; next launches shrink")
            self._pending = keep
        if self.samples - self._persisted >= 4:
            self.persist()

    def persist(self):
        """Write the live estimate into the calibration file (best effort)."""
        if self.throughput is None or self.samples < 3:
            return
        self._persisted = self.samples
        try:
            data = load_calibration(self.index)
            data.setdefault("throughput", {}).setdefault(self.kernel, {})["live"] = float(self.throughput)
            save_calibration(data, self.index)
        except Exception:
            pass


_LAUNCH_TIMERS = {}
_TIMER_LOCK = threading.Lock()


def _persist_all_timers():
    """atexit hook: write every timer's live estimate to the calibration file."""
    for t in list(_LAUNCH_TIMERS.values()):
        try:
            t.harvest()
            if t.samples >= 3 and t.samples != t._persisted:
                t.persist()
        except Exception:
            pass


import atexit as _atexit
_atexit.register(_persist_all_timers)


def launch_timer(kernel, index=None):
    """The per-process timer of a kernel on a device, created on first use.

    Args:
        kernel (str): "fast" or "general".
        index (int, optional): CUDA device index; None means the current
            device.

    Returns:
        LaunchTimer: The timer.
    """
    key = (_dev_index(index), str(kernel))
    with _TIMER_LOCK:
        t = _LAUNCH_TIMERS.get(key)
        if t is None:
            t = _LAUNCH_TIMERS[key] = LaunchTimer(kernel, key[0])
        return t


def timed_launch(kernel, atom_px, fn, *args, stream=None, **kwargs):
    """Run one kernel launch between two events and register its time.

    Args:
        kernel (str): Timer key, "fast" or "general".
        atom_px (float): Atom-pixels the launch computes.
        fn (callable): The launch; ``fn(*args, **kwargs)``.
        *args: Passed to ``fn``.
        stream (cupy.cuda.Stream, optional): Stream the launch goes to,
            passed on as ``stream=``; None means the current stream.
        **kwargs: Passed to ``fn``.

    Returns:
        Whatever ``fn`` returns.
    """
    if cp is None:
        return fn(*args, **kwargs) if stream is None else fn(*args, stream=stream, **kwargs)
    t = launch_timer(kernel)
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
    ev0.record(stream)
    result = fn(*args, **kwargs) if stream is None else fn(*args, stream=stream, **kwargs)
    ev1.record(stream)
    t.record(ev0, ev1, atom_px)
    return result


def tune_entry(key_str, index=None):
    """Persisted autotune winner for a key string.

    Args:
        key_str (str): Key from the autotuner (shape, config, source hash).
        index (int, optional): CUDA device index.

    Returns:
        tuple or None: (pix, unroll, bx, by), or None when unknown.
    """
    try:
        v = load_calibration(index).get("tune", {}).get(key_str)
        return tuple(v) if v else None
    except Exception:
        return None


def save_tune_entry(key_str, value, index=None):
    """Store an autotune winner in the calibration file (best effort).

    Args:
        key_str (str): Key from the autotuner.
        value (tuple): (pix, unroll, bx, by).
        index (int, optional): CUDA device index.
    """
    try:
        data = load_calibration(index)
        data.setdefault("tune", {})[key_str] = list(value)
        save_calibration(data, index)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Calibration store
# -----------------------------------------------------------------------------
def calibration_path(index=None):
    """Path of a device's calibration file (``cpu.json`` without CuPy).

    Args:
        index (int, optional): CUDA device index; None means the current
            device.

    Returns:
        str: The path under MOSAIC_HOME/calibration.
    """
    if cp is None:
        return os.path.join(mosaic_home(), "calibration", "cpu.json")
    idx = int(cp.cuda.Device().id) if index is None else int(index)
    return os.path.join(mosaic_home(), "calibration", probe().gpu(idx).fingerprint() + ".json")


def load_calibration(index=None):
    """Calibration record of a device.

    Args:
        index (int, optional): CUDA device index; None means the current
            device.

    Returns:
        dict: The record, or {} when none has been written.
    """
    try:
        with open(calibration_path(index), "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_calibration(data, index=None):
    """Write a calibration record (creating the directory).

    Args:
        data (dict): The record; ``written`` is added when missing.
        index (int, optional): CUDA device index.

    Returns:
        str: The path written.
    """
    path = calibration_path(index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = dict(data)
    data.setdefault("written", time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w") as f:
        json.dump(data, f, indent=1, sort_keys=True, default=str)
    return path


# -----------------------------------------------------------------------------
# Memory model
# -----------------------------------------------------------------------------
# Bytes each path holds per atom (or per candidate site, batch row, tile point)
# for every resident unit.  These are the figures from reading the code, used
# until calibrate() stores measured ones.
BYTES = {
    "gen_device_per_site": 48,       # per atom in flight: sites, mask, copies (one chunk per stream)
    "gen_host_per_atom": 13,         # positions (12) + mask (~1) per chunk in the results window
    "scatter_host_per_atom": 12,     # one chunk of positions passing through the host
    "scatter_device_resident": 20,   # pos_m + incident field held for the whole chunk
    "scatter_device_staging": 150,   # per sub-chunk transient: sorted positions, Morton scratch, tables
    "ein_pinned_per_atom": 8,        # complex64 entrance field per cache slot
    "deform_host_per_atom": 24,      # float64 output positions per worker
    "cpu_scatter_host_per_atom": 108,  # f0 table 44, f64 + f32 positions 36, amplitudes, anomalous
    "slice_host_per_atom": 20,       # positions + fr/fi in the transmission chunk cache
    "slice_device_per_atom": 220,    # projections + TSC deposition per atom batch
    "ddd_device_per_point": 32,      # dislocation displacement tile
}


def bytes_per(key, index=None):
    """Bytes per unit for a memory-model key.

    Args:
        key (str): Key of :data:`BYTES`.
        index (int, optional): CUDA device index for the calibration file.

    Returns:
        float: The calibrated value when there is one, else the table's.
    """
    try:
        v = load_calibration(index).get("bytes", {}).get(key)
        if v:
            return float(v)
    except Exception:
        pass
    return float(BYTES[key])


def _atoms(chunk_atoms):
    """Chunk atom count as a float (12.5M when it cannot be read)."""
    try:
        return max(1.0, float(np.asarray(chunk_atoms).ravel()[0]))
    except Exception:
        return 12_500_000.0


def _min_over_gpus(fn):
    """Apply ``fn(index)`` to every profile device and return the smallest answer."""
    p = probe()
    if not p.gpus:
        return None
    return min(fn(g.index) for g in p.gpus)


# -----------------------------------------------------------------------------
# Host cache and monitor
# -----------------------------------------------------------------------------
class HostCache:
    """Byte-budgeted cache of host arrays that yields to working sets.

    This is where spare RAM goes.  An entry is kept only while the host
    governor's allowance would still cover another entry of the same size, so
    a scan whose sample does not fit keeps its first chunks cached and streams
    the rest instead of thrashing.  The governor calls :meth:`evict` (least
    recently used first) when a working set needs room or free memory falls
    into the reserve.
    """

    def __init__(self, name="cache", governor=None):
        from collections import OrderedDict
        self.name = name
        self._gov = governor
        self._lock = threading.Lock()
        self._items = OrderedDict()      # key -> (array, nbytes)
        self.nbytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        (governor or host_governor()).register_evictor(self.evict)

    @property
    def governor(self):
        """The governor this cache yields to."""
        return self._gov or host_governor()

    def get(self, key):
        """Cached array for ``key``, now the most recently used, or None."""
        with self._lock:
            item = self._items.get(key)
            if item is None:
                self.misses += 1
                return None
            self._items.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key, array):
        """Keep ``array`` if the budget allows.

        Args:
            key: Cache key (a file path for chunks).
            array (np.ndarray): The array to keep.

        Returns:
            bool: True when it was kept.
        """
        try:
            nbytes = int(array.nbytes)
        except AttributeError:
            return False
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self.nbytes -= old[1]
        if self.governor.allowance() < nbytes:
            return False
        with self._lock:
            self._items[key] = (array, nbytes)
            self.nbytes += nbytes
        return True

    def invalidate(self, key=None, prefix=None):
        """Drop ``key``, or every key starting with ``prefix``."""
        with self._lock:
            keys = [k for k in self._items if (key is not None and k == key) or
                    (prefix is not None and str(k).startswith(prefix))]
            for k in keys:
                self.nbytes -= self._items.pop(k)[1]

    def evict(self, need):
        """Drop least recently used entries until ``need`` bytes are freed.

        Args:
            need (int): Bytes to free.

        Returns:
            int: Bytes freed.
        """
        freed = 0
        with self._lock:
            while self._items and freed < need:
                _k, (_arr, nb) = self._items.popitem(last=False)
                freed += nb
                self.nbytes -= nb
                self.evictions += 1
        return freed

    def clear(self):
        """Drop every entry."""
        with self._lock:
            self._items.clear()
            self.nbytes = 0

    def stats(self):
        """Counters for the GUI panel and the tests."""
        return {"name": self.name, "entries": len(self._items), "bytes": self.nbytes,
                "hits": self.hits, "misses": self.misses, "evictions": self.evictions}


_MONITOR = {"thread": None, "stop": None}


def start_monitor(interval=1.0):
    """Background check that free host memory stays above the reserve.

    When it does not, the governor's evictors run and one line is logged; at
    most one line per 30 s so a tight machine does not flood the console.
    """
    if _MONITOR["thread"] is not None and _MONITOR["thread"].is_alive():
        return _MONITOR["thread"]
    stop = threading.Event()

    def loop():
        gov = host_governor()
        last_log = 0.0
        while not stop.wait(interval):
            try:
                if gov.check():
                    continue
                short = gov.reserve - gov.free_now()
                freed = gov.make_room(short + gov.reserve // 10)
                now = time.time()
                if now - last_log > 30:
                    last_log = now
                    _log("normal", f"host memory inside the reserve by {short / _MIB:.0f} MB; "
                                   f"caches released {max(0, freed) / _MIB:.0f} MB")
            except Exception:
                continue
    t = threading.Thread(target=loop, name="mosaic-memory-monitor", daemon=True)
    t.start()
    _MONITOR["thread"], _MONITOR["stop"] = t, stop
    return t


def stop_monitor():
    """Stop the background monitor if one is running."""
    if _MONITOR["stop"] is not None:
        _MONITOR["stop"].set()
    _MONITOR["thread"], _MONITOR["stop"] = None, None


# -----------------------------------------------------------------------------
# Derived defaults
# -----------------------------------------------------------------------------
# Each returns what a fixed constant used to be, from the governors.  An
# explicit argument or the historical environment variable always wins.

#: Most chunks any path keeps in host memory at once (generation window plus writer).
RESIDENT_CHUNKS_MAX = 9
CHUNK_VOLUME_DEFAULT = 12_500_000


def auto_chunk_volume(default=CHUNK_VOLUME_DEFAULT, floor=1_000_000):
    """Atoms per chunk for ``create_sample(chunk_volume="auto")``.

    The on-disk chunk size normally stays at the default so that seeded
    samples are the same on every machine (thermal and alloy random
    streams are keyed on the chunk index).  "auto" lowers it only where
    the widest host working set, RESIDENT_CHUNKS_MAX chunks of float64
    output, would not fit half the host budget.

    Args:
        default (int, optional): Upper bound and usual answer.
        floor (int, optional): Lower bound.

    Returns:
        int: Atoms per chunk.
    """
    per_atom = RESIDENT_CHUNKS_MAX * bytes_per("deform_host_per_atom")
    fit = 0.5 * host_governor().budget() / per_atom
    return int(min(default, max(floor, fit)))


def host_chunk_slots(bytes_per_atom, chunk_atoms, want, floor=1, fraction=0.5):
    """Chunks a path may keep in host memory at once.

    Args:
        bytes_per_atom (float): Host bytes per atom of one chunk.
        chunk_atoms (int): Atoms per chunk.
        want (int): Slots wanted.
        floor (int, optional): Never fewer than this. Defaults to 1.
        fraction (float, optional): Share of the allowance the slots may
            take. Defaults to 0.5.

    Returns:
        int: Slots that fit.
    """
    return host_governor().plan(bytes_per_atom * _atoms(chunk_atoms), want, floor=floor, fraction=fraction)


def gen_streams_per_gpu(chunk_atoms, want=None, index=None):
    """Chunks generated concurrently per GPU.

    Each stream holds one geometric chunk's candidate sites on the
    device.

    Args:
        chunk_atoms (int): Atoms per geometric chunk.
        want (int, optional): An explicit count wins outright.
        index (int, optional): CUDA device index; None takes the smallest
            answer over the profile's devices.

    Returns:
        int: Streams, 1 to 4 (SAMPLE_STREAMS_PER_GPU overrides).
    """
    if want is not None:
        return max(1, int(want))
    env = _env_int("SAMPLE_STREAMS_PER_GPU")
    if env is not None:
        return max(1, env)
    if cp is None or not probe().gpus:
        return 4
    per_chunk = bytes_per("gen_device_per_site") * _atoms(chunk_atoms) * 1.5   # rotated boxes enumerate more sites
    fn = lambda i: device_governor(i).plan(per_chunk, 4, floor=1, fraction=0.8)
    return int(_min_over_gpus(fn) if index is None else fn(index))


def scatter_streams_per_gpu(Ny, Nz, chunk_atoms=None, index=None):
    """Concurrent chunks per GPU on the kinematic path.

    Three streams (today's default) keep chunk loading and staging
    overlapped with the running kernel; a fourth is added when one launch
    cannot fill the card (fewer than two blocks per SM).  The count is
    then clipped so that every stream's resident chunk plus one minimal
    staging sub-chunk fits the device budget.

    Args:
        Ny (int): Detector pixels along the first axis.
        Nz (int): Detector pixels along the second axis.
        chunk_atoms (int, optional): Atoms per file chunk; None assumes
            12.5M.
        index (int, optional): CUDA device index; None takes the smallest
            answer over the profile's devices.

    Returns:
        int: Streams (BEAM_STREAMS_PER_GPU overrides).
    """
    env = _env_int("BEAM_STREAMS_PER_GPU")
    if env is not None:
        return max(1, env)
    if cp is None or not probe().gpus:
        return 3
    blocks = max(1, -(-int(Ny) // 32)) * max(1, -(-int(Nz) // 16))
    per_stream = (bytes_per("scatter_device_resident") * _atoms(chunk_atoms if chunk_atoms is not None else 12_500_000)
                  + 500_000 * bytes_per("scatter_device_staging"))

    def fn(i):
        want = max(3, min(4, -(-2 * probe().gpu(i).sm_count // blocks)))
        return device_governor(i).plan(per_stream, want, floor=1, fraction=0.8)
    return int(_min_over_gpus(fn) if index is None else fn(index))


def scatter_subchunk_atoms(M=1, resident_bytes=0, concurrency=1, index=None,
                           cap=50_000_000, floor=500_000):
    """Atoms per scatter launch from the device budget.

    The budget is shared equally between concurrent streams; the
    staging cost per atom comes from the memory model.

    Args:
        M (int, optional): Beam channels (amplitude columns). Defaults
            to 1.
        resident_bytes (int, optional): What the caller already holds for
            the whole chunk. Defaults to 0.
        concurrency (int, optional): Streams staging at once. Defaults
            to 1.
        index (int, optional): CUDA device index.
        cap (int, optional): Upper bound. Defaults to 50M.
        floor (int, optional): Lower bound. Defaults to 500k.

    Returns:
        int: Atoms per launch.
    """
    per_atom = bytes_per("scatter_device_staging", index) + 8 * int(M)
    try:
        budget = 0.8 * device_governor(index).allowance() / max(1, int(concurrency)) - float(resident_bytes)
    except Exception:
        return int(floor)
    return int(min(cap, max(floor, budget // per_atom)))


def ein_cache_slots(chunk_atoms, want_streams=4, want_savers=6):
    """Streams per GPU and save threads for the entrance-field cache.

    Each slot pins one chunk's complex64 field in host memory.

    Args:
        chunk_atoms (int): Atoms per file chunk.
        want_streams (int, optional): Streams wanted. Defaults to 4.
        want_savers (int, optional): Save threads wanted. Defaults to 6.

    Returns:
        tuple[int, int]: (streams, save threads); the BEAM_EIN_* variables
        override the wants.
    """
    streams = _env_int("BEAM_EIN_STREAMS_PER_GPU", want_streams)
    savers = _env_int("BEAM_EIN_SAVE_THREADS", want_savers)
    per_slot = bytes_per("ein_pinned_per_atom") * _atoms(chunk_atoms)
    slots = host_governor().plan(per_slot, streams + savers, floor=2, fraction=0.25)
    if slots < streams + savers:
        streams = max(1, min(streams, slots // 2))
        savers = max(1, slots - streams)
    return int(streams), int(savers)


def deform_gpu_workers(chunk_atoms, want=8, index=None):
    """Workers (one stream each) per GPU for the FE nodal field.

    Each holds a chunk's float64 output on the host and the chunk's
    input and output on the device (48 B/atom), so both governors bound
    the count.

    Args:
        chunk_atoms (int): Atoms per file chunk.
        want (int, optional): Workers wanted. Defaults to 8.
        index (int, optional): CUDA device index.

    Returns:
        int: Workers, at least 1.
    """
    w = int(host_chunk_slots(bytes_per("deform_host_per_atom"), chunk_atoms, want, floor=1))
    if cp is not None and probe().gpus:
        try:
            w = min(w, int(device_governor(index).plan(48.0 * _atoms(chunk_atoms), w, floor=1, fraction=0.8)))
        except Exception:
            pass
    return max(1, w)


def scatter_host_slice_atoms(concurrency=1, index=None, floor=500_000, cap=200_000_000):
    """Atoms of one file chunk handed to the device at a time on the kinematic path.

    A slice holds its positions twice while the stage transform runs, plus
    the incident field (about 40 B/atom); the scatter staging on top is
    sized separately.  MOSAIC_DEVICE_SLICE_ATOMS overrides (tests).
    """
    env = _env_int("MOSAIC_DEVICE_SLICE_ATOMS")
    if env is not None:
        return max(1, env)
    try:
        budget = 0.8 * device_governor(index).allowance() / max(1, int(concurrency))
    except Exception:
        return int(floor)
    return int(min(cap, max(floor, budget // 40)))


def deform_batch_rows(per_row_bytes, streams, want=131072, floor=8192, index=None):
    """Rows per MLS batch so that every stream's scratch fits half the device budget.

    Args:
        per_row_bytes (int): Scratch bytes per row (kNN indices and
            distances, normal matrix, right-hand sides, status).
        streams (int): Streams sharing the device.
        want (int, optional): Upper bound. Defaults to 131072.
        floor (int, optional): Lower bound. Defaults to 8192.
        index (int, optional): CUDA device index.

    Returns:
        int: Rows per batch.
    """
    return int(device_governor(index).plan(per_row_bytes * max(1, int(streams)), want, floor=floor, fraction=0.5))


def deform_cpu_batch_bytes():
    """Byte budget of one CPU MLS mini-batch: today's 256 MB, less on tight machines.

    Larger batches do not speed the einsum up and would raise the peak the
    audit tests bound, so the budget only shrinks (to 64 MB at the least).
    """
    return int(min(256 * _MIB, max(64 * _MIB, 0.25 * host_governor().budget())))


def candidate_batch_bytes():
    """Bytes for one batch of candidate chunks in Sample.create_sample.

    Half the host budget, between 256 MB and 8 GB.
    """
    return int(max(256 * _MIB, min(8 * _GIB, host_governor().budget() // 2)))


def voronoi_budget(fraction=0.5, index=None):
    """Device bytes for Voronoi distance tiles.

    Args:
        fraction (float, optional): Share of the device budget. Defaults
            to 0.5.
        index (int, optional): CUDA device index.

    Returns:
        int: Bytes, at least 256 MB (2 GB without a device).
    """
    try:
        return int(max(256 * _MIB, fraction * device_governor(index).budget()))
    except Exception:
        return 2 * _GIB


def slice_accum_budget(index=None):
    """Device bytes for the transmission path's slice accumulators.

    Args:
        index (int, optional): CUDA device index.

    Returns:
        int: Half the device budget, at least 256 MB.
    """
    try:
        return int(max(256 * _MIB, 0.5 * device_governor(index).budget()))
    except Exception:
        return 2 * _GIB


def slice_atom_batch(index=None):
    """Atoms per transmission-path batch.

    Args:
        index (int, optional): CUDA device index.

    Returns:
        int: Atoms from 40 % of the device budget, at least 32768.
    """
    try:
        return int(max(32768, 0.4 * device_governor(index).budget() / bytes_per("slice_device_per_atom", index)))
    except Exception:
        return 2_000_000


def slice_cache_bytes():
    """Host bytes the transmission path may cache: everything the governor still allows."""
    return int(max(256 * _MIB, host_governor().allowance()))


def ddd_tile_points(index=None):
    """Points per dislocation-displacement tile.

    Today's 1M, less on small devices.  Larger tiles would lengthen
    each launch (points x segments), so the tile is not grown on big
    cards until the launch controller covers this kernel.

    Args:
        index (int, optional): CUDA device index.

    Returns:
        int: Points per tile, 64k to 1M.
    """
    try:
        return int(min(1 << 20, max(1 << 16, device_governor(index).budget() // (4 * bytes_per("ddd_device_per_point", index)))))
    except Exception:
        return 1 << 20


# -----------------------------------------------------------------------------
# Devices and work split
# -----------------------------------------------------------------------------
def gpu_devices():
    """CUDA device indices this process may use (MOSAIC_DEVICES), in profile order."""
    return list(probe().gpu_indices())


def device_weights(indices, kernel="fast"):
    """Relative speed of each device.

    Args:
        indices (list[int]): CUDA device indices.
        kernel (str, optional): Throughput entry to read. Defaults to
            "fast".

    Returns:
        list[float]: Weights summing to 1: live or calibrated throughput,
        else SMs x clock.
    """
    w = []
    for i in indices:
        t = None
        try:
            data = load_calibration(i)
            for key in ("live", "best"):
                v = data.get("throughput", {}).get(kernel, {}).get(key)
                if v:
                    t = float(v)
                    break
        except Exception:
            t = None
        if t is None:
            g = probe().gpu(i)
            t = float(max(1, g.sm_count) * max(1, g.clock_khz))
        w.append(t)
    s = sum(w) or 1.0
    return [x / s for x in w]


def split_ranges(n_items, indices, first=1):
    """Contiguous ranges of ``n_items`` per device, sized by :func:`device_weights`.

    Deterministic, so a run is reproducible on the same machine; with
    equal weights the first devices take the remainder, as the old split
    did.

    Args:
        n_items (int): Items to split.
        indices (list[int]): CUDA device indices.
        first (int, optional): Index of the first item. Defaults to 1.

    Returns:
        list[tuple[int, int]]: [start, stop) per device, in order.
    """
    if not indices:
        return []
    w = device_weights(indices)
    bounds = [int(first)]
    acc = 0.0
    for k, wk in enumerate(w):
        acc += wk
        stop = first + (n_items if k == len(w) - 1 else int(acc * n_items + 0.5))
        bounds.append(max(stop, bounds[-1]))
    return [(bounds[k], bounds[k + 1]) for k in range(len(w))]


def split_round_robin(n_items, indices):
    """Item positions per device, dealt so each device's share follows its weight.

    Args:
        n_items (int): Items to deal.
        indices (list[int]): CUDA device indices.

    Returns:
        list[list[int]]: Positions 0..n_items-1 per device, deterministic.
    """
    if not indices:
        return []
    w = device_weights(indices)
    shards = [[] for _ in indices]
    counts = [0] * len(indices)
    for i in range(int(n_items)):
        k = min(range(len(indices)), key=lambda j: (counts[j] / w[j], j))
        shards[k].append(i)
        counts[k] += 1
    return shards


# -----------------------------------------------------------------------------
# Calibration and run-time estimate
# -----------------------------------------------------------------------------
def _gpu_busy_percent(index=0):
    """GPU utilisation from nvidia-smi, or None when it cannot be read."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                              "--format=csv,noheader,nounits", "-i", str(index)],
                             capture_output=True, text=True, timeout=5)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _bandwidth_gbs(nbytes, seconds):
    """Gigabytes per second for ``nbytes`` moved in ``seconds``."""
    return float(nbytes) / max(seconds, 1e-9) / 1e9


class _DevicePeak:
    """Peak device memory taken during a block, from polling free memory."""

    def __init__(self, index, interval=0.002):
        self.index = int(index)
        self.interval = interval
        self._stop = threading.Event()
        self._min_free = None
        self._free0 = None
        self._thread = None

    def _read(self):
        """Free bytes on the device."""
        with cp.cuda.Device(self.index):
            return int(cp.cuda.runtime.memGetInfo()[0])

    def start(self):
        """Begin polling, after releasing the pool's idle blocks."""
        with cp.cuda.Device(self.index):
            cp.get_default_memory_pool().free_all_blocks()
        self._free0 = self._min_free = self._read()

        def loop():
            while not self._stop.is_set():
                try:
                    self._min_free = min(self._min_free, self._read())
                except Exception:
                    pass
                time.sleep(self.interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        """End polling."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_bytes(self):
        """Largest drop in free memory seen since ``start``."""
        return max(0, int(self._free0 - self._min_free)) if self._free0 is not None else 0


def calibrate(quick=False, sample_dir=None, index=None, save=True, log=print, force=False):
    """Measure this machine and store what the governors and the estimator use.

    Runs the production kinematic path on a 170 A silicon cube (~250k
    atoms, enough to fill any card) for the fast and general kernels at
    several detector sizes, records the end-to-end and kernel-only rates,
    measures host and device bytes per atom during generation (two sample
    sizes), host-to-device bandwidth, chunk-file read bandwidth (from
    ``sample_dir`` when given, else the temporary sample) and the CPU
    kernel.  Two to three minutes on a 4090; ``quick`` keeps one detector
    size.

    Args:
        quick (bool, optional): One detector size instead of three.
        sample_dir (str, optional): Directory with chunk files to time
            disk reads on.
        index (int, optional): CUDA device index; None means the current
            device.
        save (bool, optional): Write the record. Defaults to True.
        log (callable, optional): Line printer. Defaults to print.
        force (bool, optional): Calibrate even if the GPU looks busy.

    Returns:
        dict: The calibration record.

    Raises:
        RuntimeError: If the GPU is more than 40 % busy and ``force`` is
            False.
    """
    import shutil
    import tempfile
    from Crystal import crystal            # local imports: Beam imports this module
    from Sample import sample
    from Detector import detector
    from Beam import beam
    from Stage import stage

    root = os.path.dirname(os.path.abspath(__file__))
    idx = _dev_index(index)
    gpu_ok = cp is not None and bool(probe().gpus)
    if gpu_ok:
        # nvidia-smi counts any engine (video decode, the compositor), so a
        # desktop with a browser open reads 20-30 % while the compute engine
        # is free; refuse only when a real load is likely.
        busy = _gpu_busy_percent(idx)
        if busy is not None and busy > 40 and not force:
            raise RuntimeError(f"GPU {idx} is {busy:.0f}% busy; calibrate on an idle device "
                               f"(or pass force=True / --force)")
        if busy is not None and busy > 10:
            log(f"note: GPU {idx} reports {busy:.0f}% utilisation from other processes; "
                f"throughput may read a little low")
    data = load_calibration(idx) if gpu_ok else {}
    data["profile"] = probe().to_dict()
    data.setdefault("bytes", {})
    data.setdefault("throughput", {})
    work = tempfile.mkdtemp(prefix="mosaic_calib_") + os.sep
    saved_env = os.environ.get("MOSAIC_LAUNCH_CAP_S")
    os.environ["MOSAIC_LAUNCH_CAP_S"] = "0"        # steady-state launch sizes
    _LAUNCH_TIMERS.clear()
    try:
        xtal = crystal(os.path.join(root, "databases", "lattice", "Si.cif"))
        xtal.get_lattice_from_cif()
        xtal.align_axes(np.array([[1, 1, -2], [1, -1, 0]]).T)

        # ---- generation: bytes per atom on host and device
        # Warm CuPy and the generation kernels on a small sample first, so
        # the host delta below is the sample's and not the context's.
        warm_dir = tempfile.mkdtemp(prefix="mosaic_calib_warm_") + os.sep
        warm = sample(warm_dir)
        warm.create_sample([40.0] * 3, chunk_volume=12_500_000)
        warm.generate_sample_single(xtal, use_gpu=gpu_ok)
        shutil.rmtree(warm_dir, ignore_errors=True)
        # Two sample sizes: bytes per atom is the slope between them, so the
        # fixed overheads of a run (contexts, tables, the beam grid) drop out.
        measured = []
        for cube in (170.0, 340.0):
            wdir = work if cube == 170.0 else tempfile.mkdtemp(prefix="mosaic_calib_b_") + os.sep
            rss0 = HostInfo.rss()
            peak = _DevicePeak(idx) if gpu_ok else None
            smp = sample(wdir)
            smp.create_sample([cube] * 3, chunk_volume=12_500_000)
            t0 = time.perf_counter()
            if peak is not None:
                peak.start()
            smp.generate_sample_single(xtal, use_gpu=gpu_ok)
            if peak is not None:
                peak.stop()
            t_gen = time.perf_counter() - t0
            n_at = int(np.load(os.path.join(wdir, "atomic_positions_1.npy"), mmap_mode="r").shape[0])
            measured.append((n_at, HostInfo.rss() - rss0, peak.peak_bytes if peak is not None else 0, t_gen))
            if cube == 170.0:
                samp, natoms = smp, n_at
            else:
                shutil.rmtree(wdir, ignore_errors=True)
        (n1, h1, d1, t1), (n2, h2, d2, t2) = measured
        # Host: the chunk cache keeps every atom (13 B); the slope between the
        # sizes removes fixed overheads.  Device: the generators hold one
        # geometric chunk per stream, so the peak is per atom in flight.
        data["bytes"]["gen_host_per_atom"] = float(min(64.0, max(13.0, (h2 - h1) / max(1, n2 - n1))))
        if gpu_ok:
            geom = samp.geometric_chunk_atoms(xtal)
            in_flight = min(n2, gen_streams_per_gpu(geom, None, idx) * geom)
            data["bytes"]["gen_device_per_site"] = float(min(128.0, max(16.0, d2 / max(1, in_flight))))
        data["generation"] = {"atoms": n2, "seconds": t2, "atoms_per_s": n2 / max(t2, 1e-6)}
        log(f"generation: {n2:,} atoms in {t2:.2f} s; host {data['bytes']['gen_host_per_atom']:.0f} B/atom"
            + (f", device {data['bytes']['gen_device_per_site']:.0f} B per atom in flight" if gpu_ok else ""))

        # ---- chunk file read bandwidth
        path = None
        if sample_dir:
            cands = [f for f in os.listdir(sample_dir) if f.startswith("atomic_positions_") and f.endswith(".npy")]
            if cands:
                path = os.path.join(sample_dir, sorted(cands)[0])
        if path is None:
            path = os.path.join(work, "atomic_positions_1.npy")
        t0 = time.perf_counter()
        arr = np.load(path)
        t_read = time.perf_counter() - t0
        data["chunk_read_gbs"] = _bandwidth_gbs(arr.nbytes, t_read)
        data["chunk_read_from"] = os.path.dirname(path)
        log(f"chunk read: {data['chunk_read_gbs']:.2f} GB/s from {data['chunk_read_from']}")
        del arr

        if gpu_ok:
            # ---- host to device bandwidth, pageable and pinned
            with cp.cuda.Device(idx):
                host = np.ones(64 * _MIB // 4, dtype=np.float32)
                cp.asarray(host); cp.cuda.Device().synchronize()
                t0 = time.perf_counter(); cp.asarray(host); cp.cuda.Device().synchronize()
                data["h2d_pageable_gbs"] = _bandwidth_gbs(host.nbytes, time.perf_counter() - t0)
                try:
                    pin = beam.allocate_pinned_array(host)
                    cp.asarray(pin); cp.cuda.Device().synchronize()
                    t0 = time.perf_counter(); cp.asarray(pin); cp.cuda.Device().synchronize()
                    data["h2d_pinned_gbs"] = _bandwidth_gbs(host.nbytes, time.perf_counter() - t0)
                    del pin
                except Exception:
                    data["h2d_pinned_gbs"] = None
                del host
            log(f"host->device: pageable {data['h2d_pageable_gbs']:.1f} GB/s, pinned "
                f"{data.get('h2d_pinned_gbs') or float('nan'):.1f} GB/s")

            # ---- kernel throughput on the production path
            stg = stage(work); stg.create_stage(); stg.set_motor_value_relative([-44.1384, 0, 0, 0, 0, 0, 0])
            bx = beam(work); bx.create_beam(10000, beam_shape="rectangular", beam_size=(3000.0, 3000.0))
            sizes = [512] if quick else [256, 512, 1024]
            for kernel in ("fast", "general"):
                rec = data["throughput"].setdefault(kernel, {})
                per = rec.setdefault("per_npix", {})
                for npix in sizes:
                    det = detector(work)
                    det.create_detector(np.array([npix, npix]), np.array([1e5, 1e5]))
                    det.position_detector_absolute(1e9, 88.2769, 0)
                    bx._use_fast_kernel = (kernel == "fast")
                    times = []
                    for rep in range(4):                      # first pass compiles and tunes
                        cp.cuda.Device().synchronize()
                        t0 = time.perf_counter()
                        bx.atomic_direct_interaction(samp, det, stg, scattering=True, use_gpu=True)
                        cp.cuda.Device().synchronize()
                        times.append(time.perf_counter() - t0)
                        if rep == 0:
                            _LAUNCH_TIMERS.clear()            # drop the compile-time launch
                    best = min(times[1:])
                    per[str(npix)] = natoms * npix * npix / best
                    # Kernel-only rate from the CUDA events around every launch
                    # of the timed passes (staging and transfers excluded).
                    t = _LAUNCH_TIMERS.get((idx, kernel))
                    if t is not None:
                        t.harvest()
                        if t.throughput:
                            rec.setdefault("kernel_per_npix", {})[str(npix)] = float(t.throughput)
                    kern_txt = (f", kernel-only {rec['kernel_per_npix'][str(npix)]:.3e}"
                                if str(npix) in rec.get("kernel_per_npix", {}) else "")
                    log(f"{kernel:8s} {npix:4d}^2: {per[str(npix)]:.3e} atom.px/s end to end (pass {best:.3f} s){kern_txt}")
                rec["best"] = max(per.values())
                rec["e2e"] = rec["best"]
                if rec.get("kernel_per_npix"):
                    rec["kernel"] = max(rec["kernel_per_npix"].values())
            # per-pass overhead from a tiny detector (kernel time negligible)
            det = detector(work)
            det.create_detector(np.array([32, 32]), np.array([1e5, 1e5]))
            det.position_detector_absolute(1e9, 88.2769, 0)
            bx._use_fast_kernel = True
            times = []
            for rep in range(3):
                cp.cuda.Device().synchronize(); t0 = time.perf_counter()
                bx.atomic_direct_interaction(samp, det, stg, scattering=True, use_gpu=True)
                cp.cuda.Device().synchronize(); times.append(time.perf_counter() - t0)
            data["overhead_s"] = float(min(times[1:]))
            log(f"per-pass overhead: {data['overhead_s']:.3f} s")

        # ---- CPU kernel on a small problem (one chunk -> one thread today)
        if probe().host.c_compiler:
            work_cpu = tempfile.mkdtemp(prefix="mosaic_calib_cpu_") + os.sep
            s2 = sample(work_cpu); s2.create_sample([60.0] * 3, chunk_volume=12_500_000)
            s2.generate_sample_single(xtal, use_gpu=False)
            n2 = int(np.load(os.path.join(work_cpu, "atomic_positions_1.npy"), mmap_mode="r").shape[0])
            stg2 = stage(work_cpu); stg2.create_stage(); stg2.set_motor_value_relative([-44.1384, 0, 0, 0, 0, 0, 0])
            det2 = detector(work_cpu); det2.create_detector(np.array([64, 64]), np.array([1e5, 1e5]))
            det2.position_detector_absolute(1e9, 88.2769, 0)
            bx2 = beam(work_cpu); bx2.create_beam(10000, beam_shape="rectangular", beam_size=(3000.0, 3000.0))
            cpu_kw = {"use_gpu": False}
            bx2.atomic_direct_interaction(s2, det2, stg2, scattering=True, sc_kwargs=cpu_kw)   # compiles
            t0 = time.perf_counter()
            bx2.atomic_direct_interaction(s2, det2, stg2, scattering=True, sc_kwargs=cpu_kw)
            t_cpu = time.perf_counter() - t0
            thr = int(min(cpu_threads(), max(1, -(-n2 // 1000))))
            data["cpu"] = {"atoms": n2, "npix": 64 * 64, "seconds": t_cpu,
                           "atom_px_per_s": n2 * 64 * 64 / max(t_cpu, 1e-6), "threads": thr}
            log(f"cpu kernel: {data['cpu']['atom_px_per_s']:.3e} atom.px/s on {thr} thread(s)")
            shutil.rmtree(work_cpu, ignore_errors=True)
        else:
            data["cpu"] = {"available": False}
    finally:
        if saved_env is None:
            os.environ.pop("MOSAIC_LAUNCH_CAP_S", None)
        else:
            os.environ["MOSAIC_LAUNCH_CAP_S"] = saved_env
        _LAUNCH_TIMERS.clear()
        shutil.rmtree(work, ignore_errors=True)
    data["written"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if save and gpu_ok:
        save_calibration(data, idx)
        log(f"written {calibration_path(idx)}")
    return data


def estimate_runtime(natoms, npix, steps=1, kernel="fast", index=None):
    """Seconds a scan should take on this machine.

    Args:
        natoms (float): Atoms in the sample.
        npix (int): Detector pixels.
        steps (int, optional): Scan steps. Defaults to 1.
        kernel (str, optional): "fast" or "general". Defaults to "fast".
        index (int, optional): CUDA device index.

    Returns:
        float: Seconds, from the kernel throughput and the per-pass
        overhead in the calibration file.
    """
    rate = kernel_throughput(index, kernel)
    overhead = float(load_calibration(index).get("overhead_s", 0.1)) if cp is not None else 0.0
    return float(steps) * (float(natoms) * float(npix) / rate + overhead)


def format_seconds(s):
    """Seconds as a short string: '45.0 s', '3.2 min' or '2.00 h'."""
    s = float(s)
    if s < 90:
        return f"{s:.1f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


if __name__ == "__main__":
    import argparse
    # Run as a script this file is the module ``__main__`` while Beam imports
    # it as ``hardware``; the launch timers and governors must be the ones
    # Beam writes to, so everything below goes through that instance.
    import hardware as _hw
    calibrate, report = _hw.calibrate, _hw.report
    ap = argparse.ArgumentParser(description="MOSAIC hardware profile, calibration and self-test")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("report", help="print the machine profile (default)")
    c = sub.add_parser("calibrate", help="measure this machine and store the results")
    c.add_argument("--quick", action="store_true", help="one detector size instead of three")
    c.add_argument("--sample-dir", help="directory with chunk files to time disk reads on")
    c.add_argument("--device", type=int, help="CUDA device index")
    c.add_argument("--force", action="store_true", help="calibrate even if the GPU looks busy")
    sub.add_parser("selftest", help="quick calibration run that is not stored")
    args = ap.parse_args()
    if args.cmd in (None, "report"):
        print(report())
    else:
        d = calibrate(quick=(args.cmd == "selftest" or args.quick),
                      sample_dir=getattr(args, "sample_dir", None),
                      index=getattr(args, "device", None),
                      save=(args.cmd == "calibrate"), force=getattr(args, "force", False))
        print(report())
        for k in ("fast", "general"):
            if k in d.get("throughput", {}):
                r = d["throughput"][k]
                print(f"{k:8s} end to end {r['best']:.3e} atom.px/s"
                      + (f", kernel-only {r['kernel']:.3e}" if r.get("kernel") else ""))
        if "cpu" in d and d["cpu"].get("atom_px_per_s"):
            print(f"cpu      {d['cpu']['atom_px_per_s']:.3e} atom.px/s ({d['cpu'].get('threads', 1)} thread(s))")
