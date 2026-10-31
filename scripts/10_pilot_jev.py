"""Phase 1 pilot for Jev (spec section 7, Phase 1).

1. 3 modes x 6 pilot pairs, fan-out.
2. Hybrid singles on all pilot pairs (packing comparison).
3. Hybrid reverse direction on 2 pilot pairs (symmetry check).
4. Concurrency sweep (1, 8, 32) on the OpenData pilot pair, hybrid fan-out.
"""
import asyncio
import sys
from pathlib import Path

from jevmatch.datasets import load_index
from jevmatch.jev import JevConfig
from jevmatch.ledger import Ledger
from jevmatch.run import ROOT, append_results, run_jev

OUT = ROOT / "results" / "pilot_jev.parquet"
LOG = ROOT / "runs" / "pilot"
steps = set(sys.argv[1:]) or {"modes", "singles", "reverse", "concurrency"}


async def main():
    idx = load_index()
    pilot = idx[idx.split == "pilot"].sort_values("n_cells")
    ledger = Ledger()
    if "modes" in steps:
        for mode in ("schema", "instance", "hybrid"):
            rows = await run_jev(pilot, JevConfig(mode=mode), concurrency=16, phase="pilot",
                                 tag=f"modes-{mode}", ledger=ledger, log_path=LOG / f"jev_{mode}_fanout.jsonl")
            append_results(rows, OUT)
    if "singles" in steps:
        rows = await run_jev(pilot, JevConfig(mode="hybrid", packing="singles"), concurrency=16, phase="pilot",
                             tag="singles", ledger=ledger, log_path=LOG / "jev_hybrid_singles.jsonl")
        append_results(rows, OUT)
    if "reverse" in steps:
        two = pilot[pilot.source.isin(["TPC-DI", "ChEMBL"])]
        rows = await run_jev(two, JevConfig(mode="hybrid", reverse=True), concurrency=16, phase="pilot",
                             tag="reverse", ledger=ledger, log_path=LOG / "jev_hybrid_reverse.jsonl")
        append_results(rows, OUT)
    if "concurrency" in steps:
        od = pilot[pilot.source == "OpenData"]
        for c in (1, 8, 32):
            rows = await run_jev(od, JevConfig(mode="hybrid"), concurrency=c, phase="pilot",
                                 tag=f"conc-{c}", ledger=ledger, log_path=LOG / "jev_hybrid_conc.jsonl")
            append_results(rows, OUT)
    print(f"\nTypeSafe spend so far: ${ledger.totals['typesafe']:.4f}")


asyncio.run(main())
