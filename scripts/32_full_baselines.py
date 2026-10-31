"""Phase 3: frozen baseline configurations on all test + pilot pairs. Resumable.

Usage: python scripts/32_full_baselines.py [full_rows_dev]
  (no arg)       all 6 methods, 1,000-row sample (package default), test + pilot pairs
  full_rows_dev  3 instance methods on the dev pairs with all rows (sensitivity check)
"""
import json
import sys

from jevmatch.baselines import FAMILY, done_keys, run_many
from jevmatch.datasets import load_index
from jevmatch.run import ROOT

F = json.loads((ROOT / "config" / "frozen.json").read_text())["baselines"]
idx = load_index()
if len(sys.argv) > 1 and sys.argv[1] == "full_rows_dev":
    pairs, methods, iss, log, timeout = (idx[idx.split == "dev"], [m for m, f in FAMILY.items() if f == "instance"],
                                         None, ROOT / "runs" / "dev" / "baselines_full_rows.jsonl", 1800)
else:
    pairs, methods, iss, log, timeout = (idx[idx.split.isin(["test", "pilot"])], list(FAMILY), 1000,
                                         ROOT / "runs" / "full" / "baselines.jsonl", 900)
pairs = pairs.sort_values("n_cells")
tasks = [{"pair_id": p.pair_id, "src_path": p.src_path, "tgt_path": p.tgt_path, "method": m,
          "params": F["params"][m], "instance_sample_size": iss} for _, p in pairs.iterrows() for m in methods]
print(len(tasks), "tasks", flush=True)
run_many(tasks, log, workers=8, timeout_s=timeout, done=done_keys(log))
