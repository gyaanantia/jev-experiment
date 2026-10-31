"""Phase 1 pilot for the Valentine baselines: runtime per pair on this Mac.

All 6 matchers with package defaults on the 6 pilot pairs (1,000-row
instance sample, the package default). The 3 instance matchers also run
on all rows (instance_sample_size=None) for the TPC-DI pilot pair.
"""
from jevmatch.baselines import FAMILY, run_many
from jevmatch.datasets import load_index
from jevmatch.run import ROOT

idx = load_index()
pilot = idx[idx.split == "pilot"].sort_values("n_cells")
tasks = [{"pair_id": p.pair_id, "src_path": p.src_path, "tgt_path": p.tgt_path, "method": m,
          "params": {}, "instance_sample_size": 1000} for _, p in pilot.iterrows() for m in FAMILY]
tpc = pilot[pilot.source == "TPC-DI"].iloc[0]
tasks += [{"pair_id": tpc.pair_id, "src_path": tpc.src_path, "tgt_path": tpc.tgt_path, "method": m,
           "params": {}, "instance_sample_size": None} for m, f in FAMILY.items() if f == "instance"]
run_many(tasks, ROOT / "runs" / "pilot" / "baselines.jsonl", workers=8, timeout_s=1800)
