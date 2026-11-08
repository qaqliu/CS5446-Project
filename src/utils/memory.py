"""Small process-memory diagnostics for long Linux/macOS training jobs."""

import sys
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None


def memory_usage() -> dict[str, float | None]:
    """Return available process RSS in MiB, without sampling any RNG."""
    peak = None
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak /= 2**20 if sys.platform == "darwin" else 1024
    current = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) / 1024
                break
    return {"rss_mib": current, "peak_rss_mib": peak}
