#!/usr/bin/env python3
"""translate.py (beside this file or in ../flan-nl) with the peak HBM per chip after its run, for the length trial
(bt_length.py): the same arguments; after translate.main a line

    MEM {"peak_gib_max": ..., "peak_gib": [...], "limit_gib": ...}

from jax's memory_stats of the process's chips (also after a failure, then with "error"). translate.py itself stays as
it is; like its own __main__ this ends the process with os._exit, whatever threads are left.
"""

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _path in (HERE.parent / "flan-nl", HERE):
    sys.path.insert(0, str(_path))

GIB = 1024 ** 3


def memory_line(devices, error=None):
    """The MEM line of the chips' memory_stats (peak_bytes_in_use, bytes_limit); None where a chip has no stats"""
    peaks, limits = [], []
    for d in devices:
        stats = d.memory_stats() or {}
        peaks.append(round(stats["peak_bytes_in_use"] / GIB, 3) if "peak_bytes_in_use" in stats else None)
        if "bytes_limit" in stats:
            limits.append(stats["bytes_limit"] / GIB)
    known = [p for p in peaks if p is not None]
    out = {"peak_gib_max": max(known) if known else None, "peak_gib": peaks,
           "limit_gib": round(min(limits), 3) if limits else None}
    if error:
        out["error"] = error
    return "MEM " + json.dumps(out)


def main():
    import translate

    code, error = 0, None
    try:
        translate.main(sys.argv[1:])
    except SystemExit as stop:
        code = stop.code if isinstance(stop.code, int) else (0 if stop.code is None else 1)
        if not isinstance(stop.code, int) and stop.code is not None:
            print(stop.code, file=sys.stderr)
    except BaseException as e:  # noqa: BLE001 - the process ends here with the error's traceback and code
        traceback.print_exc()
        code, error = 1, f"{type(e).__name__}: {str(e)[:300]}"
    try:
        import jax

        print(memory_line(jax.local_devices(), error), flush=True)
    except Exception as e:  # noqa: BLE001 - the line is a measurement; its failure must not hide translate's code
        print(f"MEM {json.dumps({'peak_gib_max': None, 'unavailable': str(e)[:200]})}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
