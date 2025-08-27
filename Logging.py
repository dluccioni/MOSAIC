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

# -----------------------------------------------------------------------------
# Class
# -----------------------------------------------------------------------------
class logging:
    _LOG_LEVELS = {"silent": 0, "normal": 1, "verbose": 2, "debug": 3}

    def __init__(self, log_name: str | None = None, *args, **kwargs):
        # If you use multiple inheritance, this keeps MRO happy:
        try:
            super().__init__(*args, **kwargs)
        except Exception:
            pass

        self._log_name = (log_name or self.__class__.__name__).lower()
        self._log_level = self._LOG_LEVELS["normal"]
        self._logging_wrapped = False
        self._wrapped_methods = set()

    # ---------- public API (kept compatible with your current calls) ----------
    def set_logging(self, level: str = "normal", auto_instrument: bool = True):
        """
        Set logging verbosity and (optionally) wrap methods to emit start/stop
        messages and durations. Levels: "silent", "normal", "verbose", "debug".
        """
        lvl = str(level).strip().lower()
        if lvl not in self._LOG_LEVELS:
            raise ValueError(f"Unknown level: {level}")
        prev = self._log_level
        self._log_level = self._LOG_LEVELS[lvl]
        if auto_instrument and not self._logging_wrapped:
            try:
                self._install_logging_wrappers()
                self._logging_wrapped = True
            except Exception:
                # Instrumentation should never crash your flow
                pass
        self._log("normal", f"set_logging: level set to {lvl} (was {self._level_name(prev)})")

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

    # ---------- internal: wrapper installation (general-purpose) ----------
    def _install_logging_wrappers(self):
        """
        Install logging wrappers around callable methods of this instance.

        Behavior by log level:
        - normal : start/end with duration
        - verbose: same as normal (kept concise)
        - debug  : also prints detailed bound parameters and a short return summary

        Notes:
        - Skips __dunder__ methods, properties, and items declared in __log_exclude__.
        - Honors optional per-class list __log_top__ to mark high-level entry points.
        - Idempotent per-instance via _wrapped_methods.
        """

        # ----- configuration per-class (optional) -----
        top_names = tuple(getattr(self, "__log_top__", ()))
        exclude = set(getattr(self, "__log_exclude__", ())) | {
            "_install_logging_wrappers", "_log", "_level_name", "set_logging"
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
            @functools.wraps(unbound_attr)
            def _wrapper(self_ref, *args, **kwargs):
                # current level map (read from the instance)
                lvlmap = getattr(self_ref, "_LOG_LEVELS", {"silent": 0, "normal": 1, "verbose": 2, "debug": 3})
                cur = getattr(self_ref, "_log_level", lvlmap.get("normal", 1))
                t0 = time.perf_counter()

                # Rebind the original attr correctly (honor staticmethod/classmethod)
                try:
                    orig_attr = inspect.getattr_static(type(self_ref), method_name)
                    target = orig_attr.__get__(self_ref, type(self_ref)) if hasattr(orig_attr, "__get__") else orig_attr
                except Exception:
                    target = unbound_attr.__get__(self_ref, type(self_ref)) if hasattr(unbound_attr, "__get__") else unbound_attr

                # Header
                try:
                    if cur >= lvlmap.get(min_level, 1):
                        if cur >= lvlmap.get("debug", 3):
                            try:
                                sig = inspect.signature(target)
                                ba = sig.bind_partial(*args, **kwargs)
                                items = [f"{k}={_summ(v)}" for k, v in ba.arguments.items()]
                                self_ref._log(min_level, f"{method_name}() start: {', '.join(items)}")
                            except Exception:
                                self_ref._log(min_level, f"{method_name}() start")
                        else:
                            self_ref._log(min_level, f"{method_name}() start")
                except Exception:
                    pass

                # Body
                try:
                    result = target(*args, **kwargs)
                except Exception as ex:
                    try:
                        if cur >= lvlmap.get("debug", 3):
                            self_ref._log("debug", f"{method_name}() exception: {ex}")
                    except Exception:
                        pass
                    raise

                # Footer
                dt = time.perf_counter() - t0
                try:
                    if cur >= lvlmap.get(min_level, 1):
                        self_ref._log(min_level, f"{method_name}() done in {dt:.3f}s")
                        if cur >= lvlmap.get("debug", 3):
                            try:
                                self_ref._log("debug", f"{method_name}() return: {_summ(result)}")
                            except Exception:
                                pass
                except Exception:
                    pass
                return result

            return _wrapper

        # Iterate over the *class* dictionary to avoid triggering properties
        for name in dir(type(self)):
            if name in exclude:
                continue
            # Skip attributes we already wrapped
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
                # Never let logging instrumentation break your object
                continue