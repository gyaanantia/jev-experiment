"""Phase 1 pilot for Haiku 5.5: hybrid, 6 pilot pairs, thinking off and adaptive."""
import asyncio
import sys

from jevmatch.datasets import load_index
from jevmatch.haiku import HaikuConfig
from jevmatch.jev import DEFAULT_WORDING
from jevmatch.ledger import Ledger
from jevmatch.run import ROOT, append_results, run_haiku

thinking = sys.argv[1] if len(sys.argv) > 1 else "disabled"


async def main():
    idx = load_index()
    pilot = idx[idx.split == "pilot"].sort_values("n_cells")
    ledger = Ledger()
    cfg = HaikuConfig(wording=DEFAULT_WORDING, thinking=thinking, effort="low")
    rows = await run_haiku(pilot, cfg, concurrency=16, phase="pilot", tag=f"haiku-{thinking}",
                           ledger=ledger, log_path=ROOT / "runs" / "pilot" / f"haiku_{thinking}.jsonl")
    append_results(rows, ROOT / "results" / "pilot_haiku.parquet")


asyncio.run(main())
