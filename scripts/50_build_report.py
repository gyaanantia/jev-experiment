"""Phase 4: build report/index.html from report/template.html + report/data.json + drill-down data.

The drill-down data holds, per non-dev pair: column names, ground truth, each
method's score matrix (noul x 100, ints), and the hybrid column profiles,
rebuilt with the frozen profile settings (deterministic, so identical to the
text Jev received; checked against the request log below).
"""
import json
from pathlib import Path

import pandas as pd

from jevmatch.datasets import gt_set, load_index
from jevmatch.profiles import read_table, table_profiles
from jevmatch.run import ROOT, read_jsonl

REPORT = ROOT / "report"
F = json.loads((ROOT / "config" / "frozen.json").read_text())
J = F["jev"]
MODES = ["jev_hybrid", "jev_schema", "jev_instance", "haiku_hybrid"]


def matrix(scores_json: str, src, tgt):
    sc = {(s, t): v for s, t, v in json.loads(scores_json)}
    return [[int(round(100 * sc.get((s, t), 0.0))) for t in tgt] for s in src]


def drill_data(idx: pd.DataFrame) -> dict:
    res = pd.concat([read_jsonl(ROOT / "results" / "full_jev.jsonl"), read_jsonl(ROOT / "results" / "full_haiku.jsonl")],
                    ignore_index=True).drop_duplicates(["pair_id", "method"], keep="last")
    pairs = {}
    for _, p in idx[idx.split.isin(["test", "pilot"])].iterrows():
        rows = res[res.pair_id == p.pair_id]
        if rows.empty:
            continue
        src_df, tgt_df = read_table(p.src_path), read_table(p.tgt_path)
        src, tgt = list(src_df.columns), list(tgt_df.columns)
        kw = dict(n_values=J["n_values"], max_len=J["max_len"], seed=J["seed"])
        ps, pt = table_profiles(src_df, "hybrid", **kw), table_profiles(tgt_df, "hybrid", **kw)
        pairs[p.pair_id] = {
            "scenario": p.scenario, "source": p.source, "src": src, "tgt": tgt,
            "gt": sorted(list(x) for x in gt_set(p)),
            "sprof": [{"name": c, "type": ps[c]["inferred_type"], "values": ps[c]["sample_values"]} for c in src],
            "tprof": [{"name": c, "type": pt[c]["inferred_type"], "values": pt[c]["sample_values"]} for c in tgt],
            "scores": {r.method: matrix(r.scores, src, tgt) for r in rows.itertuples() if r.method in MODES},
            "recall": {r.method: round(float(r.recall_gt_expected), 4) for r in rows.itertuples()},
        }
    return pairs


def check_profiles(pairs: dict) -> int:
    """Compare rebuilt profiles with the state actually sent in the hybrid request log."""
    log = ROOT / "runs" / "full" / "jev_hybrid.jsonl"
    checked = 0
    with log.open() as f:
        for line in f:
            r = json.loads(line)
            if r["status"] != "ok" or r["pair_id"] not in pairs or r["req_idx"] != 0:
                continue
            P = pairs[r["pair_id"]]
            sent = r["state"]["source_column"]
            mine = P["sprof"][0]
            assert sent == {"name": mine["name"], "inferred_type": mine["type"], "sample_values": mine["values"]}, r["pair_id"]
            checked += 1
            if checked >= 60:
                break
    return checked


def example() -> dict:
    """One real hybrid request from the log: a noisy-schema TPC-DI pair, the AgencyID-like first column."""
    log = ROOT / "runs" / "full" / "jev_hybrid.jsonl"
    want = "TPC-DI/prospect_vertical_50_ac4_ev"
    idx = load_index().set_index("pair_id")
    with log.open() as f:
        for line in f:
            r = json.loads(line)
            if r["pair_id"] == want and r["status"] == "ok" and r["req_idx"] == 1:
                break
        else:
            return {}
    gt = gt_set(idx.loc[want])
    src_col = r["state"]["source_column"]["name"]
    answers = sorted(({"target": r["keymap"][q][1], "noul": v, "gt": (src_col, r["keymap"][q][1]) in gt}
                      for q, v in r["answers"].items()), key=lambda a: -a["noul"])
    qs = list(r["questions"].items())
    excerpt = {"model": r["model"], "state": r["state"],
               "questions": {qid: q for qid, q in qs[:2]} | {"…": f"{len(qs) - 2} more Nouls, one per target column"}}
    text = json.dumps(excerpt, indent=2, ensure_ascii=False)
    return {"pair_id": want, "source": src_col, "answers": answers[:8], "request_text": text,
            "input_tokens": r["input_tokens"], "latency_s": r["latency_s"]}


def main():
    idx = load_index()
    data = json.loads((REPORT / "data.json").read_text())
    # Drill-down data and the example request depend only on the result files; cache them so that
    # template-only rebuilds take seconds. The cache is rebuilt when any input is newer.
    cache = ROOT / "runs" / "drill_cache.json"
    inputs = [ROOT / "results" / "full_jev.jsonl", ROOT / "results" / "full_haiku.jsonl", ROOT / "data" / "pairs.parquet",
              ROOT / "config" / "frozen.json", Path(__file__)]
    if cache.exists() and cache.stat().st_mtime > max(f.stat().st_mtime for f in inputs):
        drill, ex = (c := json.loads(cache.read_text()))["drill"], c["example"]
        print(f"drill-down: {len(drill['pairs'])} pairs from {cache.relative_to(ROOT)}")
    else:
        pairs = drill_data(idx)
        n_checked = check_profiles(pairs)
        print(f"drill-down: {len(pairs)} pairs; rebuilt profiles match the request log for {n_checked} checked requests")
        default = "TPC-DI/prospect_vertical_50_ac4_ev" if "TPC-DI/prospect_vertical_50_ac4_ev" in pairs else next(iter(pairs))
        drill = {"pairs": pairs, "modes": [m for m in MODES if any(m in p["scores"] for p in pairs.values())],
                 "default_pair": default}
        ex = example()
        cache.write_text(json.dumps({"drill": drill, "example": ex}, ensure_ascii=False))
    data["example"] = ex
    data["nyu_commit"] = (ROOT / "data" / "raw" / "nyu" / "COMMIT_SHA").read_text().strip()
    data["deviations"] = json.loads((ROOT / "config" / "deviations.json").read_text())
    data["repro"] = (REPORT / "REPRODUCE.txt").read_text()
    blob = "const DATA = " + json.dumps(data, ensure_ascii=False) + ";\nconst DRILL = " + \
           json.dumps(drill, ensure_ascii=False, separators=(",", ":")) + ";"
    blob = blob.replace("</", "<\\/")
    html = (REPORT / "template.html").read_text().replace("/*__DATA__*/", blob)
    (REPORT / "index.html").write_text(html)
    print(f"wrote report/index.html ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
