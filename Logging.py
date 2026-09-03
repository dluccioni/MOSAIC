# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
from __future__ import annotations
import functools, inspect, time, sys, types
try:
    import numpy as np
except Exception:
    np = None
try:
    import cupy as cp
except Exception:
    cp = None
try:
    import psutil
    _PSUTIL_PROC = psutil.Process()
except Exception:
    psutil = None
    _PSUTIL_PROC = None
_RESOURCE_MOD = None
if psutil is None:
    try:
        import resource as _RESOURCE_MOD
    except Exception:
        _RESOURCE_MOD = None

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class logging:
    _LOG_LEVELS = {"silent": 0, "normal": 1, "verbose": 2, "debug": 3}

    def __init__(self, log_name: str | None = None, *args, **kwargs):
        # For multiple inheritance, this keeps MRO happy:
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass

        self._log_name = (log_name or self.__class__.__name__).lower()
        self._log_level = self._LOG_LEVELS["normal"]
        self._logging_wrapped = False
        self._wrapped_methods = set()

        # Profiling state
        self._profile_enabled = False
        self._profile_stats = {}
        self._call_depth = 0
        self._mem_threshold = 100 * 1024 * 1024  # 100 MB

    # ---------- public API ----------
    def set_logging(self, level: str = "normal", auto_instrument: bool = True, profile: bool = False):
        """
        Set logging verbosity and (optionally) wrap methods to emit start/stop
        messages and durations. Levels: "silent", "normal", "verbose", "debug".

        At "silent" any installed wrappers are removed so calls run at native
        cost. With profile=True the wrappers also collect timing, RSS and GPU
        memory statistics for print_profile_report(); otherwise those
        measurements are skipped.
        """
        lvl = str(level).strip().lower()
        if lvl not in self._LOG_LEVELS:
            raise ValueError(f"Unknown level: {level}")
        prev = self._log_level
        self._log_level = self._LOG_LEVELS[lvl]
        self._profile_enabled = bool(profile)
        if self._log_level == self._LOG_LEVELS["silent"]:
            if self._logging_wrapped:
                self._uninstall_logging_wrappers()
        elif auto_instrument and not self._logging_wrapped:
            try:
                self._install_logging_wrappers()
                self._logging_wrapped = True
            except Exception:
                # Instrumentation should never crash flow
                pass
        self._log("normal", f"set_logging: level set to {lvl} (was {self._level_name(prev)})")
        if self._log_level >= self._LOG_LEVELS.get("verbose", 2):
            try:
                self.log_memory_snapshot()
            except Exception:
                pass

    def _level_name(self, num: int | None = None) -> str:
        num = self._log_level if num is None else num
        for k, v in self._LOG_LEVELS.items():
            if v == num:
                return k
        return "normal"

    def _log(self, level: str, msg: str):
        """Instance-safe printer gated by _LOG_LEVELS."""
        try:
            lvl = self._LOG_LEVELS.get(level, 1)
            cur = getattr(self, "_log_level", 1)
            if cur >= lvl:
                # Prefix with the class-derived name for clarity across instances
                print(f"[{self._log_name}|{level.upper()}] {msg}")
        except Exception:
            print(f"[{self._log_name}|LOG] {msg}")

    # ---------- profiling: memory helpers ----------
    @staticmethod
    def _get_rss_bytes():
        """Return current process RSS in bytes, or None if unavailable."""
        try:
            if _PSUTIL_PROC is not None:
                return _PSUTIL_PROC.memory_info().rss
        except Exception:
            pass
        try:
            if _RESOURCE_MOD is not None:
                return _RESOURCE_MOD.getrusage(_RESOURCE_MOD.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:
            pass
        return None

    @staticmethod
    def _get_gpu_mem_used():
        """Return GPU memory currently used in bytes, or None if unavailable."""
        try:
            if cp is not None:
                free, total = cp.cuda.runtime.memGetInfo()
                return total - free
        except Exception:
            pass
        return None

    @staticmethod
    def _fmt_bytes(b):
        """Format byte count as human-readable string with sign."""
        if b is None:
            return "N/A"
        sign = "+" if b >= 0 else ""
        ab = abs(b)
        if ab < 1024:
            return f"{sign}{b} B"
        elif ab < 1024 ** 2:
            return f"{sign}{b / 1024:.1f} KB"
        elif ab < 1024 ** 3:
            return f"{sign}{b / (1024**2):.1f} MB"
        else:
            return f"{sign}{b / (1024**3):.2f} GB"

    # ---------- profiling: stats accumulator ----------
    def _update_profile_stats(self, method_name, dt, mem_delta, gpu_delta):
        """Update cumulative profiling statistics for a method."""
        stats = self._profile_stats
        if method_name not in stats:
            stats[method_name] = {
                "calls": 0,
                "total_time": 0.0,
                "min_time": float("inf"),
                "max_time": 0.0,
                "total_mem_delta": 0,
                "peak_mem_delta": 0,
                "total_gpu_delta": 0,
                "peak_gpu_delta": 0,
            }
        s = stats[method_name]
        s["calls"] += 1
        s["total_time"] += dt
        if dt < s["min_time"]:
            s["min_time"] = dt
        if dt > s["max_time"]:
            s["max_time"] = dt
        if mem_delta is not None:
            s["total_mem_delta"] += mem_delta
            if abs(mem_delta) > abs(s["peak_mem_delta"]):
                s["peak_mem_delta"] = mem_delta
        if gpu_delta is not None:
            s["total_gpu_delta"] += gpu_delta
            if abs(gpu_delta) > abs(s["peak_gpu_delta"]):
                s["peak_gpu_delta"] = gpu_delta

    # ---------- profiling: public reporting ----------
    def print_profile_report(self, sort_by="total_time", top_n=20):
        """
        Print a formatted table of the most expensive methods.

        Args:
            sort_by: Key to sort by. One of "total_time", "calls", "max_time",
                     "avg_time", "peak_mem_delta", "peak_gpu_delta".
            top_n:   Number of rows to show.
        """
        stats = getattr(self, "_profile_stats", {})
        if not stats:
            self._log("normal", "print_profile_report: no profiling data collected")
            return

        rows = []
        for name, s in stats.items():
            avg = s["total_time"] / s["calls"] if s["calls"] else 0.0
            rows.append({
                "method": name,
                "calls": s["calls"],
                "total_time": s["total_time"],
                "min_time": s["min_time"] if s["min_time"] != float("inf") else 0.0,
                "max_time": s["max_time"],
                "avg_time": avg,
                "peak_mem_delta": s["peak_mem_delta"],
                "peak_gpu_delta": s["peak_gpu_delta"],
            })

        if sort_by == "avg_time":
            rows.sort(key=lambda r: r["avg_time"], reverse=True)
        elif rows and sort_by in rows[0]:
            rows.sort(key=lambda r: abs(r.get(sort_by, 0)), reverse=True)
        else:
            rows.sort(key=lambda r: r["total_time"], reverse=True)

        rows = rows[:top_n]

        hdr = (f"{'Method':<40} {'Calls':>6} {'Total(s)':>10} {'Avg(s)':>10} "
               f"{'Min(s)':>10} {'Max(s)':>10} {'PeakMem':>12} {'PeakGPU':>12}")
        sep = "-" * len(hdr)
        self._log("normal", f"=== Profile Report for {self._log_name} (top {top_n}, sort={sort_by}) ===")
        self._log("normal", hdr)
        self._log("normal", sep)
        for r in rows:
            line = (f"{r['method']:<40} {r['calls']:>6} {r['total_time']:>10.3f} "
                    f"{r['avg_time']:>10.4f} {r['min_time']:>10.4f} {r['max_time']:>10.3f} "
                    f"{self._fmt_bytes(r['peak_mem_delta']):>12} "
                    f"{self._fmt_bytes(r['peak_gpu_delta']):>12}")
            self._log("normal", line)
        self._log("normal", sep)

    def log_memory_snapshot(self):
        """Print current process and GPU memory state."""
        rss = self._get_rss_bytes()
        gpu = self._get_gpu_mem_used()
        parts = []
        if rss is not None:
            parts.append(f"RSS={self._fmt_bytes(rss)}")
        else:
            parts.append("RSS=unavailable")
        if gpu is not None:
            parts.append(f"GPU_used={self._fmt_bytes(gpu)}")
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                parts.append(f"GPU_free={self._fmt_bytes(free)}")
                parts.append(f"GPU_total={self._fmt_bytes(total)}")
            except Exception:
                pass
        else:
            parts.append("GPU=unavailable")
        self._log("normal", f"memory_snapshot: {', '.join(parts)}")

    def reset_profile_stats(self):
        """Clear all collected profiling statistics."""
        self._profile_stats.clear()
        self._log("normal", "reset_profile_stats: profiling data cleared")

    # ---------- internal: wrapper installation (general-purpose) ----------
    def _uninstall_logging_wrappers(self):
        """Remove the per-instance wrappers so class methods are called directly."""
        for name in list(self._wrapped_methods):
            self.__dict__.pop(name, None)
        self._wrapped_methods.clear()
        self._logging_wrapped = False

    def _install_logging_wrappers(self):
        """
        Install logging wrappers around callable methods of this instance.

        Behavior by log level:
        - normal : start/end with duration for __log_top__ methods
        - verbose: start/end for all methods (+ RSS memory deltas above
                    threshold when profiling)
        - debug  : all methods with call-depth indentation, detailed bound
                    parameters and return summaries (+ memory/GPU deltas when
                    profiling)

        Cumulative profiling statistics are collected only when set_logging
        was called with profile=True. A wrapped method whose level is not
        enabled, with profiling off, is forwarded directly.

        Notes:
        - Skips __dunder__ methods, properties, and items declared in __log_exclude__.
        - Honors optional per-class list __log_top__ to mark high-level entry points.
        - Idempotent per-instance via _wrapped_methods.
        """

        # ----- configuration per-class -----
        top_names = tuple(getattr(self, "__log_top__", ()))
        exclude = set(getattr(self, "__log_exclude__", ())) | {
            "_install_logging_wrappers", "_uninstall_logging_wrappers",
            "_log", "_level_name", "set_logging",
            "_get_rss_bytes", "_get_gpu_mem_used", "_fmt_bytes",
            "_update_profile_stats",
            "print_profile_report", "log_memory_snapshot", "reset_profile_stats",
        }

        # Heuristic: decide the minimum level at which a method should print
        def _infer_min_level(name: str) -> str | None:
            if name in exclude:
                return None
            if name.startswith("__") and name.endswith("__"):
                return None  # never wrap dunders
            if top_names and name in top_names:
                return "normal"
            heavy_prefixes = ("build_", "compile_", "compute_", "parse_", "make_",
                              "allocate_", "write_", "load_", "get_", "_compute_", "_ein_")
            if name.startswith(heavy_prefixes):
                return "verbose"
            if name.startswith("_"):
                return "debug"
            return "verbose"

        # Compact value summary for debug mode
        def _summ(v, maxlen=64):
            try:
                if np is not None and isinstance(v, (np.ndarray,)):
                    return f"ndarray(shape={v.shape}, dtype={v.dtype})"
                if np is not None and hasattr(np, "generic") and isinstance(v, np.generic):
                    return f"{v.dtype}"
            except Exception:
                pass
            if cp is not None:
                try:
                    if isinstance(v, cp.ndarray):
                        return f"cupy(shape={v.shape}, dtype={v.dtype})"
                except Exception:
                    pass
            try:
                if hasattr(v, "shape"):
                    return f"obj(shape={getattr(v, 'shape', None)})"
                if isinstance(v, (list, tuple, set, dict)):
                    return f"{type(v).__name__}(len={len(v)})"
                if isinstance(v, (str, bytes)):
                    s = v if isinstance(v, str) else v.decode("utf-8", "ignore")
                    s = s if len(s) <= maxlen else (s[:maxlen] + "...")
                    return f"str({s})"
                return f"{v}"
            except Exception:
                return "<unrepr>"

        def _make_wrapper(method_name: str, unbound_attr, min_level: str):
            lvlmap = getattr(self, "_LOG_LEVELS",
                             {"silent": 0, "normal": 1, "verbose": 2, "debug": 3})
            min_lvl_num = lvlmap.get(min_level, 1)
            verbose_num = lvlmap.get("verbose", 2)
            debug_num = lvlmap.get("debug", 3)

            # Resolve the original method once; the raw class attribute keeps
            # staticmethod/classmethod binding correct
            try:
                orig_attr = inspect.getattr_static(type(self), method_name)
                target = (orig_attr.__get__(self, type(self))
                          if hasattr(orig_attr, "__get__") else orig_attr)
            except Exception:
                target = (unbound_attr.__get__(self, type(self))
                          if hasattr(unbound_attr, "__get__") else unbound_attr)

            @functools.wraps(unbound_attr)
            def _wrapper(self_ref, *args, **kwargs):
                cur = getattr(self_ref, "_log_level", 1)
                profiling = getattr(self_ref, "_profile_enabled", False)
                should_log = (cur >= min_lvl_num)
                if not should_log and not profiling:
                    return target(*args, **kwargs)

                is_verbose = (cur >= verbose_num)
                is_debug = (cur >= debug_num)

                track_mem = profiling and is_verbose
                track_gpu = profiling and is_debug

                # Call depth
                depth = getattr(self_ref, "_call_depth", 0)
                indent = ("  " * depth) if is_debug else ""

                # Memory before
                rss_before = None
                gpu_before = None
                if track_mem:
                    try:
                        rss_before = logging._get_rss_bytes()
                    except Exception:
                        pass
                if track_gpu:
                    try:
                        gpu_before = logging._get_gpu_mem_used()
                    except Exception:
                        pass

                # Start message
                t0 = time.perf_counter()
                try:
                    if should_log:
                        if is_debug:
                            try:
                                target_for_sig = (unbound_attr.__get__(self_ref, type(self_ref))
                                                  if hasattr(unbound_attr, "__get__")
                                                  else unbound_attr)
                                sig = inspect.signature(target_for_sig)
                                ba = sig.bind_partial(*args, **kwargs)
                                items = [f"{k}={_summ(v)}" for k, v in ba.arguments.items()]
                                self_ref._log(min_level,
                                              f"{indent}{method_name}() start: {', '.join(items)}")
                            except Exception:
                                self_ref._log(min_level, f"{indent}{method_name}() start")
                        else:
                            self_ref._log(min_level, f"{method_name}() start")
                except Exception:
                    pass

                # Increment depth
                try:
                    self_ref._call_depth = depth + 1
                except Exception:
                    pass

                # Body
                try:
                    result = target(*args, **kwargs)
                except Exception as ex:
                    try:
                        self_ref._call_depth = depth
                    except Exception:
                        pass
                    dt = time.perf_counter() - t0
                    try:
                        if is_debug:
                            self_ref._log("debug", f"{indent}{method_name}() exception: {ex}")
                    except Exception:
                        pass
                    if profiling:
                        try:
                            self_ref._update_profile_stats(method_name, dt, None, None)
                        except Exception:
                            pass
                    raise

                # Restore depth
                try:
                    self_ref._call_depth = depth
                except Exception:
                    pass

                # Timing
                dt = time.perf_counter() - t0

                # Memory after
                mem_delta = None
                gpu_delta = None
                if track_mem and rss_before is not None:
                    try:
                        rss_after = logging._get_rss_bytes()
                        if rss_after is not None:
                            mem_delta = rss_after - rss_before
                    except Exception:
                        pass
                if track_gpu and gpu_before is not None:
                    try:
                        gpu_after = logging._get_gpu_mem_used()
                        if gpu_after is not None:
                            gpu_delta = gpu_after - gpu_before
                    except Exception:
                        pass

                # Update cumulative stats when profiling
                if profiling:
                    try:
                        self_ref._update_profile_stats(method_name, dt, mem_delta, gpu_delta)
                    except Exception:
                        pass

                # End message
                try:
                    if should_log:
                        parts = [f"{dt:.3f}s"]
                        if is_debug:
                            if mem_delta is not None:
                                parts.append(f"mem={logging._fmt_bytes(mem_delta)}")
                            if gpu_delta is not None:
                                parts.append(f"gpu={logging._fmt_bytes(gpu_delta)}")
                        elif is_verbose:
                            threshold = getattr(self_ref, "_mem_threshold",
                                                100 * 1024 * 1024)
                            if mem_delta is not None and abs(mem_delta) >= threshold:
                                parts.append(f"mem={logging._fmt_bytes(mem_delta)}")

                        detail = ", ".join(parts)
                        self_ref._log(min_level, f"{indent}{method_name}() done in {detail}")

                        if is_debug:
                            try:
                                self_ref._log("debug",
                                              f"{indent}{method_name}() return: {_summ(result)}")
                            except Exception:
                                pass
                except Exception:
                    pass

                return result

            return _wrapper

        # Iterate over the class dictionary to avoid triggering properties
        for name in dir(type(self)):
            if name in exclude:
                continue
            # Skip wrapped attributes
            if name in getattr(self, "_wrapped_methods", set()):
                continue
            # Pull the attribute from the class (not instance) and test callability
            try:
                attr_cls = getattr(type(self), name)
            except Exception:
                continue
            if not callable(attr_cls):
                continue

            min_level = _infer_min_level(name)
            if min_level is None:
                continue

            try:
                wrapper = _make_wrapper(name, attr_cls, min_level)
                setattr(self, name, types.MethodType(wrapper, self))
                self._wrapped_methods.add(name)
            except Exception:
                continue
