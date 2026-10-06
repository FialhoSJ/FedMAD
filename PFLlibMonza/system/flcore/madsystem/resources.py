"""Process memory in bytes, with no optional dependency or external service."""

import os


def process_memory():
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                    (key, ctypes.c_size_t) for key in ("peak", "current", "qpp", "qpc", "qnp", "qnc", "pagefile", "peak_pagefile")
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            if not memory_info(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError("process memory unavailable")
            return {"rss_bytes": int(counters.current), "peak_memory_bytes": int(counters.peak), "memory_source": "Windows process working set"}
        import resource
        import sys
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak *= 1 if sys.platform == "darwin" else 1024
        return {"rss_bytes": None, "peak_memory_bytes": peak, "memory_source": "process ru_maxrss"}
    except (ImportError, OSError, AttributeError):
        return {"rss_bytes": None, "peak_memory_bytes": None, "memory_source": "unavailable"}
