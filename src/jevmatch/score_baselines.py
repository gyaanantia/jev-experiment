"""Turn baseline run logs into per-pair metric rows."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .baselines import FAMILY
from .datasets import gt_set, load_index
from .metrics import all_metrics
from .run import columns


def score_log(log_path: Path, phase: str) -> pd.DataFrame:
    idx = load_index().set_index("pair_id")
    rows = []
    for line in log_path.read_text().splitlines():
        r = json.loads(line)
        pair = idx.loc[r["pair_id"]]
        base = {"pair_id": r["pair_id"], "method": r["method"], "family": FAMILY[r["method"]], "phase": phase,
                "params": json.dumps(r["params"], sort_keys=True), "instance_sample_size": r["instance_sample_size"],
                "status": r["status"], "wall_s": r["wall_s"], "match_s": r.get("match_s"), "load_s": r.get("load_s"),
                "cbc_orbit_off": bool(r.get("cbc_orbit_off", False))}
        if r["status"] == "ok":
            scores = {(s, t): v for s, t, v in r["scores"]}
            base.update(all_metrics(scores, gt_set(pair), columns(pair["src_path"]), columns(pair["tgt_path"]),
                                    probabilistic=False))
            base["scores"] = json.dumps(r["scores"])
        rows.append(base)
    return pd.DataFrame(rows)
