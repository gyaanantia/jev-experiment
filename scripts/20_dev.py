"""Phase 2 on the 24 dev pairs (spec section 7, Phase 2).

  wording   - 3 question wordings, hybrid mode, N=10
  sweep     - N in {5, 25} (and 50 if asked), hybrid only, given wording
  modes     - schema + instance modes with the given wording and N (stability run 1)
  stability - 2 more runs of the frozen setup, hybrid only (run 1 = wording step)
  haiku     - Haiku hybrid, thinking disabled vs adaptive, given wording and N
Usage: python scripts/20_dev.py <step> [wording_id] [N ...]
"""
import asyncio
import sys

from jevmatch.datasets import load_index
from jevmatch.haiku import HaikuConfig
from jevmatch.jev import JevConfig
from jevmatch.ledger import Ledger
from jevmatch.run import ROOT, append_results, run_haiku, run_jev

W1 = "Does `candidate_column` hold the same attribute as `source_column`?"
WORDINGS = {
    "W0": ("Do `source_column` and `candidate_column` hold the same attribute of the same kind of entity, "
           "so that their values could be stored in one column?", None),
    "W1": (W1, None),
    "W2": (W1, {"true": "Both columns record the same property of the same kind of entity; names, "
                        "abbreviations and value formats may differ.",
                "false": "The columns record different properties, even if their names or values look similar."}),
    "W3": (W1, {"true": "Same property; names and formats may differ.",
                "false": "Different properties, even if they look alike."}),
}
OUT = ROOT / "results" / "dev_jev.parquet"
LOG = ROOT / "runs" / "dev"


async def main(step: str, wid: str, ns: list[int]):
    idx = load_index()
    dev = idx[idx.split == "dev"].sort_values("n_cells")
    ledger = Ledger()
    wording, criteria = WORDINGS[wid]

    async def jev_run(mode, n, tag):
        cfg = JevConfig(mode=mode, n_values=n, wording=wording, criteria=criteria)
        rows = await run_jev(dev, cfg, concurrency=32, phase="dev", tag=tag, ledger=ledger,
                             log_path=LOG / f"jev_{tag}.jsonl", extra={"wording_id": wid})
        append_results(rows, OUT)

    if step == "wording":
        for w in ns_w or WORDINGS:
            wording, criteria = WORDINGS[w]
            wid = w
            await jev_run("hybrid", 10, f"wording-{w}-hybrid-n10")
    elif step == "sweep":
        for n in ns:  # hybrid only (budget; see config/frozen.json)
            await jev_run("hybrid", n, f"sweep-{wid}-hybrid-n{n}")
    elif step == "modes":  # stability run 1 for schema and instance (hybrid run 1 = wording step)
        for mode in ("schema", "instance"):
            await jev_run(mode, ns[0], f"stab1-{wid}-{mode}-n{ns[0]}")
    elif step == "stability":
        n = ns[0]
        for rep in (2, 3):  # hybrid only (budget; see config/frozen.json)
            await jev_run("hybrid", n, f"stab{rep}-{wid}-hybrid-n{n}")
    elif step == "haiku":
        for th in ("disabled", "adaptive"):
            cfg = HaikuConfig(wording=wording, n_values=ns[0], thinking=th, effort="low")
            rows = await run_haiku(dev, cfg, concurrency=16, phase="dev", tag=f"haiku-{th}", ledger=ledger,
                                   log_path=LOG / f"haiku_{th}.jsonl", extra={"wording_id": wid})
            append_results(rows, ROOT / "results" / "dev_haiku.parquet")
    print(f"\nspend: typesafe ${ledger.totals['typesafe']:.4f}  anthropic ${ledger.totals['anthropic']:.4f}")


if __name__ == "__main__":
    step = sys.argv[1]
    wid = sys.argv[2] if len(sys.argv) > 2 else "W1"
    ns = [int(x) for x in sys.argv[3:] if x.isdigit()] or [10]
    ns_w = [x for x in sys.argv[2:] if x in WORDINGS] if step == "wording" else []
    asyncio.run(main(step, wid, ns))
