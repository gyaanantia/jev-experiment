"""Phase 4: compute every number the report shows and write report/data.json.

Each number carries a citation key; report/data.json["cite"] maps keys to
their source (paper table/figure + page, or result file + filter + script).
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from jevmatch.baselines import FAMILY
from jevmatch.datasets import load_index
from jevmatch.run import ROOT, read_jsonl
from jevmatch.score_baselines import score_log

RNG = np.random.default_rng(42)
OUT = ROOT / "report" / "data.json"
FAB = ["TPC-DI", "OpenData", "ChEMBL"]
SCEN = ["Unionable", "View-Unionable", "Joinable", "Semantically-Joinable"]
LABEL = {
    "jev_hybrid": "Jev · hybrid", "jev_schema": "Jev · schema-only", "jev_instance": "Jev · instance-only",
    "haiku_hybrid": "Haiku 5.5 · hybrid", "cupid": "Cupid", "similarity_flooding": "Similarity Flooding",
    "coma_schema": "COMA (schema)", "coma_instance": "COMA (instance)", "distribution_based": "Distribution-based",
    "jaccard_levenshtein": "Jaccard-Levenshtein", "embdi": "EmbDI", "semprop": "SemProp",
}
FAMILY_OF = {**FAMILY, "jev_schema": "schema", "jev_instance": "instance", "jev_hybrid": "hybrid",
             "haiku_hybrid": "hybrid", "embdi": "hybrid", "semprop": "hybrid"}
PAPER = json.loads((ROOT / "config" / "paper_numbers.json").read_text())
N_GT = load_index().set_index("pair_id").n_gt
FROZEN = json.loads((ROOT / "config" / "frozen.json").read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


SCRIPTS = {p.name: sha(p) for p in (ROOT / "scripts").glob("*.py")}
SCRIPTS.update({f"jevmatch/{p.name}": sha(p) for p in (ROOT / "src" / "jevmatch").glob("*.py")})
CITE: dict[str, dict] = {}


def cite(key: str, **kw) -> str:
    CITE[key] = kw
    return key


def npairs(n: int) -> str:
    return f"{n} pair" if n == 1 else f"{n} pairs"


def r4(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


def boot_ci(x, stat=np.mean, n=10000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return [r4(stat(x)) if len(x) else None] * 2
    b = stat(x[RNG.integers(0, len(x), (n, len(x)))], axis=1)
    return [r4(np.percentile(b, 2.5)), r4(np.percentile(b, 97.5))]


def box(x):
    x = np.sort(np.asarray(x, float))
    if not len(x):
        return None
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo = x[x >= q1 - 1.5 * iqr].min()
    hi = x[x <= q3 + 1.5 * iqr].max()
    return {"n": int(len(x)), "mean": r4(x.mean()), "median": r4(med), "q1": r4(q1), "q3": r4(q3),
            "lo": r4(lo), "hi": r4(hi), "min": r4(x.min()), "max": r4(x.max()),
            "outliers": [r4(v) for v in x[(x < lo) | (x > hi)]][:60],
            "mean_ci": boot_ci(x), "median_ci": boot_ci(x, np.median)}


# ---------------------------------------------------------------- load
def load_all():
    idx = load_index()
    meta_cols = ["pair_id", "source", "scenario", "schema_noise", "instance_noise", "row_overlap", "col_overlap",
                 "n_src_cols", "n_tgt_cols", "n_cells", "n_gt", "n_src_rows", "n_tgt_rows", "split", "haiku_subset"]
    meta = idx[meta_cols]
    frames = []
    jev = read_jsonl(ROOT / "results" / "full_jev.jsonl")
    if len(jev):
        jev = jev.drop_duplicates(["pair_id", "method"], keep="last")
        jev["status"] = np.where(jev["complete"], "ok", "incomplete")
        frames.append(jev)
    hk = read_jsonl(ROOT / "results" / "full_haiku.jsonl")
    if len(hk):
        hk = hk.drop_duplicates(["pair_id"], keep="last")
        hk["status"] = np.where(hk["complete"], "ok", "incomplete")
        frames.append(hk)
    blog = ROOT / "runs" / "full" / "baselines.jsonl"
    if blog.exists():
        b = score_log(blog, "full").drop_duplicates(["pair_id", "method"], keep="last")
        b["cost_usd"] = 0.0
        frames.append(b)
    res = pd.concat(frames, ignore_index=True)
    res = res.merge(meta, on="pair_id", how="left")
    return idx, res


def ok(res):
    return res[res.status == "ok"]


# ---------------------------------------------------------------- sections
def datasets_table(idx):
    g = idx.groupby("source").agg(pairs=("pair_id", "size"), cells=("n_cells", "sum"),
                                  cols_min=("n_src_cols", "min"), cols_max=("n_src_cols", "max"),
                                  rows_min=("n_src_rows", "min"), rows_max=("n_src_rows", "max"),
                                  dev=("split", lambda s: int((s == "dev").sum()))).reset_index()
    return g.to_dict("records")


def accuracy_boxes(res):
    """Fig. 4/5/6-style boxes on fabricated test pairs, per scenario and noise split."""
    d = ok(res[res.source.isin(FAB) & (res.split != "dev")])
    out = {}
    for sc in SCEN:
        ds = d[d.scenario == sc]
        out[sc] = {}
        for m in ds.method.unique():
            dm = ds[ds.method == m]
            entry = {"all": box(dm.recall_gt_expected)}
            for col in ("schema_noise", "instance_noise"):
                for lvl in ("noisy", "verbatim"):
                    x = dm[dm[col] == lvl].recall_gt_expected
                    if len(x):
                        entry[f"{col}:{lvl}"] = box(x)
            for s in FAB:
                x = dm[dm.source == s].recall_gt_expected
                if len(x):
                    entry[f"source:{s}"] = box(x)
                for col in ("schema_noise", "instance_noise"):
                    for lvl in ("noisy", "verbatim"):
                        x = dm[(dm.source == s) & (dm[col] == lvl)].recall_gt_expected
                        if len(x):
                            entry[f"source:{s}|{col}:{lvl}"] = box(x)
            out[sc][m] = entry
            cite(f"box:{sc}:{m}", kind="run", text=f"{LABEL[m]}, {sc}: Recall@GT (expected ties) over fabricated test pairs",
                 file=_file_for(m), filter=f"source in TPC-DI/OpenData/ChEMBL, scenario={sc}, split!=dev, status=ok",
                 script="scripts/40_analyze.py:accuracy_boxes")
    return out


def _file_for(m):
    if m.startswith("jev"):
        return "results/full_jev.jsonl (raw requests: runs/full/jev_<mode>.jsonl)"
    if m.startswith("haiku"):
        return "results/full_haiku.jsonl (raw requests: runs/full/haiku.jsonl)"
    return "runs/full/baselines.jsonl"


def curated(res):
    out = {}
    d = ok(res[res.source.isin(["Magellan", "WikiData", "NYU"])])
    for s in ["Magellan", "NYU"]:
        ds = d[d.source == s]
        out[s] = {m: {"mean": r4(g.recall_gt_expected.mean()), "n": int(len(g)), "ci": boot_ci(g.recall_gt_expected),
                      "per_pair": {r.pair_id: r4(r.recall_gt_expected) for r in g.itertuples()}}
                  for m, g in ds.groupby("method")}
        for m in out[s]:
            cite(f"cur:{s}:{m}", kind="run", text=f"{LABEL[m]} mean Recall@GT on {s} ({npairs(out[s][m]['n'])})",
                 file=_file_for(m), filter=f"source={s}, status=ok", script="scripts/40_analyze.py:curated")
    ds = d[d.source == "WikiData"]
    out["WikiData"] = {sc: {m: r4(g.recall_gt_expected.iloc[0]) for m, g in ds[ds.scenario == sc].groupby("method")}
                       for sc in SCEN}
    return out


def cost_latency(res):
    d = ok(res[res.split != "dev"])
    out = {}
    for m, g in d.groupby("method"):
        row = {"n": int(len(g)), "recall_mean": r4(g.recall_gt_expected.mean()),
               "recall_ci": boot_ci(g.recall_gt_expected)}
        if m.startswith(("jev", "haiku")):
            row.update(cost_mean=r4(g.cost_usd.mean()), cost_total=r4(g.cost_usd.sum()), cost_ci=boot_ci(g.cost_usd),
                       cost_per_1k_cells=r4(1000 * g.cost_usd.sum() / g.n_cells.sum()),
                       tokens_per_cell=r4(g.input_tokens.sum() / g.n_cells.sum()),
                       wall_mean=r4(g.wall_s.mean()), wall_median=r4(g.wall_s.median()), wall_ci=boot_ci(g.wall_s),
                       api_p50=r4(g.api_p50_s.median()), api_p95=r4(g.api_p95_s.median()))
        else:
            t = g.match_s + g.load_s
            row.update(cost_mean=0.0, wall_mean=r4(t.mean()), wall_median=r4(t.median()), wall_ci=boot_ci(t),
                       wall_with_startup_mean=r4(g.wall_s.mean()))
        out[m] = row
        cite(f"cl:{m}", kind="run", text=f"{LABEL[m]}: mean Recall@GT, API cost and wall-clock per pair over {row['n']} non-dev pairs",
             file=_file_for(m), filter="split!=dev, status=ok", script="scripts/40_analyze.py:cost_latency")
    # per-source breakdown for the explorer
    per_src = {}
    for (m, s), g in d.groupby(["method", "source"]):
        w = g.wall_s if m.startswith(("jev", "haiku")) else g.match_s + g.load_s
        per_src.setdefault(m, {})[s] = {"n": int(len(g)), "recall": r4(g.recall_gt_expected.mean()),
                                         "cost": r4(g.cost_usd.mean()), "wall": r4(w.mean())}
    return out, per_src


def stats_tests(res):
    d = ok(res[(res.split != "dev") & res.source.isin(FAB)])
    jev_modes = [m for m in ("jev_hybrid", "jev_schema", "jev_instance") if m in d.method.unique()]
    if not jev_modes:
        return {}
    best = max(jev_modes, key=lambda m: d[d.method == m].recall_gt_expected.mean())
    piv = d.pivot_table(index="pair_id", columns="method", values="recall_gt_expected")
    sc_of = d.drop_duplicates("pair_id").set_index("pair_id").scenario
    rows = []
    for b in [m for m in piv.columns if m != best and not m.startswith("jev")]:
        for sc in SCEN + ["All"]:
            sub = piv[[best, b]].dropna()
            if sc != "All":
                sub = sub[sc_of.reindex(sub.index) == sc]
            diff = sub[best] - sub[b]
            win, tie, loss = int((diff > 1e-9).sum()), int((diff.abs() <= 1e-9).sum()), int((diff < -1e-9).sum())
            p = None
            if (diff.abs() > 1e-9).sum() >= 6:
                p = float(wilcoxon(sub[best], sub[b], zero_method="wilcox").pvalue)
            rows.append({"jev": best, "baseline": b, "scenario": sc, "n": int(len(sub)), "win": win, "tie": tie,
                         "loss": loss, "mean_diff": r4(diff.mean()), "diff_ci": boot_ci(diff), "p": p})
    # Holm correction over all tests with a p-value
    ps = sorted([(r["p"], i) for i, r in enumerate(rows) if r["p"] is not None])
    m = len(ps)
    running = 0.0
    for rank, (p, i) in enumerate(ps):
        running = max(running, min(1.0, (m - rank) * p))
        rows[i]["p_holm"] = running
    cite("stats", kind="run", text=f"Paired Wilcoxon signed-rank ({best} vs each baseline), Holm-corrected; fabricated non-dev pairs",
         file="results/full_jev.jsonl + runs/full/baselines.jsonl", filter="paired on pair_id, both status=ok",
         script="scripts/40_analyze.py:stats_tests")
    return {"best_jev": best, "rows": rows}


def difficulty(res):
    d = ok(res[(res.split != "dev") & res.source.isin(FAB)])
    rows = []
    keys = ["scenario", "schema_noise", "instance_noise"]
    for k, g in d.groupby(keys + ["method"]):
        rows.append({**dict(zip(keys + ["method"], k)), "n": int(len(g)), "mean": r4(g.recall_gt_expected.mean())})
    by_overlap = []
    for k, g in d.groupby(["col_overlap", "method"]):
        by_overlap.append({"col_overlap": int(k[0]), "method": k[1], "n": int(len(g)),
                           "mean": r4(g.recall_gt_expected.mean())})
    by_k = []
    d = d.assign(k_bin=pd.cut(d.n_gt, [0, 1, 5, 15, 100], labels=["1", "2–5", "6–15", "16+"]))
    for k, g in d.groupby(["k_bin", "method"], observed=True):
        by_k.append({"k_bin": str(k[0]), "method": k[1], "n": int(len(g)), "mean": r4(g.recall_gt_expected.mean())})
    return {"cells": rows, "col_overlap": by_overlap, "gt_size": by_k}


def worst_pairs(res, method="jev_hybrid", n=15):
    d = ok(res[(res.method == method) & (res.split != "dev")]).sort_values("recall_gt_expected")
    return [{"pair_id": r.pair_id, "recall": r4(r.recall_gt_expected), "scenario": r.scenario, "source": r.source,
             "n_gt": int(r.n_gt)} for r in d.head(n).itertuples()]


def thresholded(res):
    d = ok(res[(res.split != "dev") & res.method.isin(["jev_hybrid", "jev_schema", "jev_instance", "haiku_hybrid"])])
    out = {}
    for m, g in d.groupby("method"):
        out[m] = {k: r4(g[k].mean()) for k in ["t5_precision", "t5_recall", "t5_f1", "t7_precision", "t7_recall",
                                               "t7_f1", "mrr", "p_at_1", "h1_f1"]}
    for m, g in ok(res[(res.split != "dev") & ~res.method.str.startswith(("jev", "haiku"))]).groupby("method"):
        out[m] = {k: r4(g[k].mean()) for k in ["mrr", "p_at_1", "h1_f1"]}
    cite("thr", kind="run", text="Secondary metrics, mean over non-dev pairs per method",
         file="results/full_*.jsonl, runs/full/baselines.jsonl", filter="split!=dev, status=ok",
         script="scripts/40_analyze.py:thresholded")
    return out


def haiku_compare(res):
    """Like-for-like on the Haiku subset: every method on the same pairs."""
    d = ok(res[res.haiku_subset & (res.split != "dev")])
    pairs = set(d[d.method == "haiku_hybrid"].pair_id)
    d = d[d.pair_id.isin(pairs)]
    out = {}
    for m, g in d.groupby("method"):
        w = g.wall_s if m.startswith(("jev", "haiku")) else g.match_s + g.load_s
        out[m] = {"n": int(len(g)), "recall": r4(g.recall_gt_expected.mean()), "recall_ci": boot_ci(g.recall_gt_expected),
                  "cost_mean": r4(g.cost_usd.mean()), "wall_mean": r4(w.mean())}
    cite("haiku_subset", kind="run", text=f"Haiku subset ({len(pairs)} pairs): every method on the same pairs",
         file="results/full_haiku.jsonl + results/full_jev.jsonl + runs/full/baselines.jsonl",
         filter="haiku_subset=True, split!=dev, status=ok", script="scripts/40_analyze.py:haiku_compare")
    return out


# ---------------------------------------------------------------- dev / pilot
def dev_section():
    d = pd.read_parquet(ROOT / "results" / "dev_jev.parquet")
    wording = d[d.tag.str.startswith("wording")].groupby("wording_id").agg(
        recall=("recall_gt_expected", "mean"), rmin=("recall_gt_expected", "min"), cost=("cost_usd", "sum"),
        tok=("input_tokens", "sum")).reset_index()
    wording["tok_per_cell"] = wording.tok / 17683
    sweep_tags = {"sweep-W2-hybrid-n5": 5, "wording-W2-hybrid-n10": 10, "sweep-W2-hybrid-n25": 25}
    sweep = (d[d.tag.isin(sweep_tags)].assign(N=lambda x: x.tag.map(sweep_tags)).groupby("N")
             .agg(recall=("recall_gt_expected", "mean"), cost=("cost_usd", "sum"), tok=("input_tokens", "sum"))
             .reset_index())
    sweep["tok_per_cell"] = sweep.tok / 17683
    modes = d[d.tag.isin(["wording-W2-hybrid-n10", "stab1-W2-schema-n10", "stab1-W2-instance-n10"])].groupby("mode") \
        .agg(recall=("recall_gt_expected", "mean"), cost=("cost_usd", "sum")).reset_index()
    # stability
    tags = ["wording-W2-hybrid-n10", "stab2-W2-hybrid-n10", "stab3-W2-hybrid-n10"]
    runs = {t: d[d.tag == t].set_index("pair_id") for t in tags}
    sds, ident, ranges, per_run = [], [], [], {t: r4(runs[t].recall_gt_expected.mean()) for t in tags}
    for pid in runs[tags[0]].index:
        S = [{(s, t): v for s, t, v in json.loads(runs[t].loc[pid, "scores"])} for t in tags]
        keys = sorted(S[0])
        M = np.array([[s[k] for k in keys] for s in S])
        sds.append(M.std(axis=0, ddof=1))
        ident.append((M.max(0) == M.min(0)).mean())
        rr = [runs[t].loc[pid, "recall_gt_expected"] for t in tags]
        ranges.append({"pair_id": pid, "range": r4(max(rr) - min(rr)), "runs": [r4(x) for x in rr],
                       "n_gt": int(N_GT[pid])})
    sds = np.concatenate(sds)
    stab = {"per_run_mean": per_run, "sd_mean": r4(sds.mean()), "sd_median": r4(np.median(sds)),
            "sd_p95": r4(np.percentile(sds, 95)), "sd_max": r4(sds.max()), "identical_share": r4(np.mean(ident)),
            "pairs_changed": int(sum(r["range"] > 0 for r in ranges)), "pairs": len(ranges),
            "changed": [r for r in ranges if r["range"] > 0]}
    hk = pd.read_parquet(ROOT / "results" / "dev_haiku.parquet").groupby("thinking").agg(
        recall=("recall_gt_expected", "mean"), cost=("cost_usd", "sum"), out_tok=("output_tokens", "sum")).reset_index()
    for k, txt in [("dev:wording", "Question wordings W0–W3 on the 24 dev pairs (hybrid, N=10)"),
                   ("dev:sweep", "Sample-size sweep N∈{5,10,25}, W2, hybrid, 24 dev pairs"),
                   ("dev:stability", "3 identical runs of the frozen hybrid setup on the 24 dev pairs"),
                   ("dev:modes", "Frozen setup in each mode on the 24 dev pairs")]:
        cite(k, kind="run", text=txt, file="results/dev_jev.parquet (raw: runs/dev/jev_*.jsonl)",
             filter="split=dev", script="scripts/20_dev.py, scripts/40_analyze.py:dev_section")
    cite("dev:haiku", kind="run", text="Haiku thinking disabled vs adaptive on the dev pairs",
         file="results/dev_haiku.parquet", filter="split=dev", script="scripts/20_dev.py haiku")
    return {"wording": wording.round(4).to_dict("records"), "sweep": sweep.round(4).to_dict("records"),
            "modes": modes.round(4).to_dict("records"), "stability": stab, "haiku": hk.round(4).to_dict("records")}


def baseline_grid():
    p = ROOT / "results" / "dev_baselines_grid.parquet"
    if not p.exists():
        return {}
    g = pd.read_parquet(p)
    g = g[g.status == "ok"].groupby(["method", "params"]).recall_gt_expected.mean().reset_index()
    return {m: {"configs": int(len(x)), "best": r4(x.recall_gt_expected.max()), "worst": r4(x.recall_gt_expected.min())}
            for m, x in g.groupby("method")}


def pilot_section():
    p = pd.read_parquet(ROOT / "results" / "pilot_jev.parquet")
    base = p[p.tag == "modes-hybrid"].set_index("pair_id")

    def sc(r):
        return {(s, t): v for s, t, v in json.loads(r.scores)}

    packing = []
    for _, r in p[p.tag == "singles"].iterrows():
        a, b = sc(base.loc[r.pair_id]), sc(r)
        k = sorted(a)
        x, y = np.array([a[i] for i in k]), np.array([b[i] for i in k])
        packing.append({"pair_id": r.pair_id, "source": r.pair_id.split("/")[0], "cells": len(k),
                        "fanout_tokens": int(base.loc[r.pair_id].input_tokens), "singles_tokens": int(r.input_tokens),
                        "fanout_wall": r4(base.loc[r.pair_id].wall_s), "singles_wall": r4(r.wall_s),
                        "mean_abs_diff": r4(np.abs(x - y).mean()), "fanout_recall": r4(base.loc[r.pair_id].recall_gt_expected),
                        "singles_recall": r4(r.recall_gt_expected)})
    conc = [{"concurrency": int(r.concurrency), "wall_s": r4(r.wall_s), "api_p50": r4(r.api_p50_s),
             "api_p95": r4(r.api_p95_s), "requests": int(r.n_requests)} for _, r in p[p.tag.str.startswith("conc")].iterrows()]
    rev = []
    for _, r in p[p.tag == "reverse"].iterrows():
        a, b = sc(base.loc[r.pair_id]), sc(r)
        k = sorted(a)
        rev.append({"pair_id": r.pair_id, "mean_abs_diff": r4(np.mean([abs(a[i] - b[i]) for i in k]))})
    cite("pilot:packing", kind="run", text="Fan-out vs one-request-per-pair on the 6 pilot pairs (hybrid, wording W0)",
         file="results/pilot_jev.parquet (raw: runs/pilot/jev_hybrid_singles.jsonl)", filter="split=pilot",
         script="scripts/10_pilot_jev.py singles")
    cite("pilot:conc", kind="run", text="Concurrency sweep on the OpenData pilot pair (38 requests, hybrid, wording W0)",
         file="results/pilot_jev.parquet (raw: runs/pilot/jev_hybrid_conc.jsonl)", filter="tag=conc-*",
         script="scripts/10_pilot_jev.py concurrency")
    return {"packing": packing, "concurrency": conc, "reverse": rev}


def paper_cites():
    cite("paper:t4", kind="paper", text=PAPER["table_iv"]["source"], exact=True)
    cite("paper:t5", kind="paper", text=PAPER["table_v"]["source"] + ". " + PAPER["table_v"]["hardware"], exact=True)
    for f in ("fig4", "fig5", "fig6", "fig7"):
        cite(f"paper:{f}", kind="paper", text=PAPER[f]["source"], exact=False, note=PAPER[f]["note"])
    cite("price:jev", kind="price", text="Jev 1.13/1.14: $0.042 per million input tokens; output tokens free",
         url="https://docs.typesafe.ai/models.md", read="2026-10-30")
    cite("price:haiku", kind="price", text="Claude Haiku 5.5: $0.10 / $0.50 per million input / output tokens (prompts ≤100K)",
         url="https://claude.com/pricing", read="claude-api model table cached 2026-10-06")


def ledger_totals():
    tot = {}
    for l in (ROOT / "runs" / "ledger.jsonl").read_text().splitlines():
        r = json.loads(l)
        k = (r["provider"], r["phase"])
        tot[k] = tot.get(k, 0) + r["cost_usd"]
    out = {f"{p}:{ph}": r4(v) for (p, ph), v in tot.items()}
    cite("ledger", kind="run", text="Every billed API call, summed by provider and phase", file="runs/ledger.jsonl",
         script="src/jevmatch/ledger.py")
    return out


def tie_bounds(res):
    d = ok(res[res.split != "dev"])
    return {m: {k: r4(g[f"recall_gt_{k}"].mean()) for k in ("pessimistic", "expected", "optimistic")}
            for m, g in d.groupby("method")}


def full_rows_dev():
    """Sensitivity: instance matchers on dev pairs with all rows vs the 1,000-row sample."""
    p = ROOT / "runs" / "dev" / "baselines_full_rows.jsonl"
    g = ROOT / "results" / "dev_baselines_grid.parquet"
    if not (p.exists() and g.exists()):
        return {}
    full = score_log(p, "dev")
    grid = pd.read_parquet(g)
    out = {}
    for m, prm in FROZEN["baselines"]["params"].items():
        a = grid[(grid.method == m) & (grid.params == json.dumps(prm, sort_keys=True)) & (grid.status == "ok")]
        b = full[(full.method == m)]
        if b.empty:
            continue
        # Paired: compare only pairs where both the sampled and the all-row run finished.
        both = a.merge(b[b.status == "ok"], on="pair_id", suffixes=("_s", "_a"))
        out[m] = {"n_paired": int(len(both)),
                  "sample_1000": r4(both.recall_gt_expected_s.mean()), "all_rows": r4(both.recall_gt_expected_a.mean()),
                  "sample_1000_all": r4(a.recall_gt_expected.mean()), "n_sample_all": int(len(a)),
                  "all_rows_ok": int((b.status == "ok").sum()), "all_rows_timeout": int((b.status == "timeout").sum()),
                  "time_1000": r4((both.match_s_s + both.load_s_s).mean()), "time_all": r4((both.match_s_a + both.load_s_a).mean())}
    cite("full_rows_dev", kind="run", text="Instance matchers on the dev pairs: 1,000-row sample vs all rows, "
         "on the pairs where both runs finished",
         file="runs/dev/baselines_full_rows.jsonl + results/dev_baselines_grid.parquet", script="scripts/40_analyze.py:full_rows_dev")
    return out


def main():
    idx, res = load_all()
    paper_cites()
    cl, per_src = cost_latency(res)
    data = {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "labels": LABEL, "family": FAMILY_OF, "frozen": FROZEN, "paper": PAPER,
        "datasets": datasets_table(idx),
        "accuracy": accuracy_boxes(res),
        "curated": curated(res),
        "cost_latency": cl, "cost_latency_by_source": per_src,
        "stats": stats_tests(res),
        "difficulty": difficulty(res),
        "worst": worst_pairs(res),
        "secondary": thresholded(res),
        "haiku_subset": haiku_compare(res) if "haiku_hybrid" in set(res.method) else {},
        "dev": dev_section(), "baseline_grid": baseline_grid(), "pilot": pilot_section(),
        "ledger": ledger_totals(), "tie_bounds": tie_bounds(res), "full_rows_dev": full_rows_dev(),
        "coverage": {m: {"ok": int((g.status == "ok").sum()), "total": int(len(g))} for m, g in res.groupby("method")},
        "baseline_limit_s": 900,  # timeout_s in scripts/32_full_baselines.py
        "scripts": SCRIPTS,
    }
    data["cite"] = CITE
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    res.drop(columns=["scores"], errors="ignore").to_parquet(ROOT / "results" / "all_methods.parquet", index=False)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB); methods: {sorted(res.method.unique())}")


if __name__ == "__main__":
    main()
