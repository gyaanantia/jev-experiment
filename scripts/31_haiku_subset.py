"""Phase 3: Haiku 5.5 hybrid on the Haiku subset (frozen config). Resumable."""
import asyncio
import json

from jevmatch.datasets import load_index
from jevmatch.haiku import HaikuConfig
from jevmatch.ledger import BudgetExceeded, Ledger
from jevmatch.run import ROOT, read_jsonl, run_haiku

OUT = ROOT / "results" / "full_haiku.jsonl"
F = json.loads((ROOT / "config" / "frozen.json").read_text())
J, H = F["jev"], F["haiku"]


async def main():
    idx = load_index()
    pairs = idx[idx.haiku_subset].sort_values("n_cells")
    done = read_jsonl(OUT)
    if len(done):
        pairs = pairs[~pairs.pair_id.isin(set(done.pair_id))]
    ledger = Ledger()
    cfg = HaikuConfig(wording=J["wording"], n_values=H["n_values"], max_len=H["max_len"],
                      thinking=H["thinking"], effort=H["effort"])
    try:
        await run_haiku(pairs, cfg, concurrency=16, phase="full", tag="full-haiku", ledger=ledger,
                        log_path=ROOT / "runs" / "full" / "haiku.jsonl", extra={"wording_id": J["wording_id"]},
                        out_jsonl=OUT)
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
    print(f"anthropic spend ${ledger.totals['anthropic']:.4f}")


asyncio.run(main())
