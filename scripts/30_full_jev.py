"""Phase 3: Jev on all test pairs (+ pilot pairs, re-run frozen), frozen config.

Order: hybrid, then schema, then instance (spec: protect the headline if the
budget runs out). Resumable: (pair, mode) rows already in results/full_jev.jsonl
are skipped.
"""
import asyncio
import json
import sys

from jevmatch.datasets import load_index
from jevmatch.jev import JevConfig
from jevmatch.ledger import BudgetExceeded, Ledger
from jevmatch.run import ROOT, read_jsonl, run_jev

OUT = ROOT / "results" / "full_jev.jsonl"
FROZEN = json.loads((ROOT / "config" / "frozen.json").read_text())["jev"]
ORDER = ["TPC-DI", "ChEMBL", "WikiData", "Magellan", "NYU", "OpenData"]
modes = sys.argv[1:] or ["hybrid", "schema", "instance"]


async def main():
    idx = load_index()
    pairs = idx[idx.split.isin(["test", "pilot"])].copy()
    pairs["o"] = pairs.source.map(ORDER.index)
    pairs = pairs.sort_values(["o", "n_cells"])
    done = read_jsonl(OUT)
    ledger = Ledger()
    for mode in modes:
        have = set(done[done["mode"] == mode].pair_id) if len(done) else set()
        todo = pairs[~pairs.pair_id.isin(have)]
        print(f"== {mode}: {len(todo)} pairs to run ({len(have)} done)", flush=True)
        cfg = JevConfig(mode=mode, n_values=FROZEN["n_values"], max_len=FROZEN["max_len"],
                        wording=FROZEN["wording"], criteria=FROZEN["criteria"])
        try:
            await run_jev(todo, cfg, concurrency=FROZEN["concurrency"], phase="full", tag=f"full-{mode}",
                          ledger=ledger, log_path=ROOT / "runs" / "full" / f"jev_{mode}.jsonl",
                          extra={"wording_id": FROZEN["wording_id"]}, out_jsonl=OUT)
        except BudgetExceeded as e:
            print(f"STOPPED: {e}", flush=True)
            return
    print(f"done. typesafe spend ${ledger.totals['typesafe']:.4f}")


asyncio.run(main())
