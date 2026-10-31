"""Phase 2: grid search for each baseline on the 24 dev pairs (paper Table II ranges)."""
from jevmatch.baselines import GRIDS, done_keys, run_many
from jevmatch.datasets import load_index
from jevmatch.run import ROOT

LOG = ROOT / "runs" / "dev" / "baselines_grid.jsonl"
idx = load_index()
dev = idx[idx.split == "dev"]
tasks = [{"pair_id": p.pair_id, "src_path": p.src_path, "tgt_path": p.tgt_path, "method": m,
          "params": params, "instance_sample_size": 1000}
         for m, grid in GRIDS.items() for params in grid for _, p in dev.iterrows()]
print(len(tasks), "tasks")
run_many(tasks, LOG, workers=8, timeout_s=600, done=done_keys(LOG))
