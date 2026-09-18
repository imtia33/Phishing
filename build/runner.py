#!/usr/bin/env python3
"""Execute the Rev 6 notebook with nbclient; record errors, save partial results."""
import os, sys, time, json, traceback

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
SRC = sys.argv[2] if len(sys.argv) > 2 else "trac-phish-revision6.ipynb"
DST = sys.argv[3] if len(sys.argv) > 3 else f"run/trac-phish-revision6_{MODE}.ipynb"

BASE = "/home/z/my-project/rev6"
os.environ["TRAC_RUN_MODE"] = MODE
os.environ["TRAC_INPUT_ROOT"] = f"{BASE}/run/input"
os.environ["TRAC_WORK_ROOT"] = f"{BASE}/run/work"
os.environ["TRAC_N_JOBS"] = "2"
os.environ.pop("TRAC_MAX_ROWS", None)

import nbformat
from nbclient import NotebookClient

t0 = time.time()
nb = nbformat.read(SRC, as_version=4)
print(f"[runner] mode={MODE} src={SRC} -> {DST}, cells={len(nb.cells)}", flush=True)

client = NotebookClient(nb, timeout=7200, kernel_name="python3",
                        resources={"metadata": {"path": f"{BASE}/run"}},
                        allow_errors=True, store_widget_state=False)

errors = []
try:
    client.execute()
    status = "completed"
except Exception as e:
    status = "halted"
    errors.append({"type": type(e).__name__, "msg": str(e)[:2000]})
    traceback.print_exc()

err_cells = []
for i, c in enumerate(nb.cells):
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                err_cells.append({"cell": i, "ename": o.get("ename"), "evalue": (o.get("evalue") or "")[:500]})
                break

nbformat.write(nb, DST)
summary = {"status": status, "elapsed_min": round((time.time() - t0) / 60, 1),
           "n_cells": len(nb.cells), "error_cells": err_cells}
with open(f"{BASE}/run/runner_{MODE}_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2), flush=True)
