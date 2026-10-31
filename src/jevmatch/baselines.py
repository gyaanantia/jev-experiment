"""Valentine baselines (spec section 4).

Each (pair, method, params) run happens in its own subprocess so that a
per-run timeout can kill it. `python -m jevmatch.baselines '<json task>'`
runs one task and prints one JSON line.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

FAMILY = {
    "cupid": "schema",
    "similarity_flooding": "schema",
    "coma_schema": "schema",
    "coma_instance": "instance",
    "distribution_based": "instance",
    "jaccard_levenshtein": "instance",
}


def _rng(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


# Paper Table II ranges (p. 475). Methods with fixed parameters in the
# paper get a single configuration.
GRIDS = {
    "cupid": [
        {"leaf_w_struct": a, "w_struct": b, "th_accept": c}
        for a, b, c in itertools.product(_rng(0, 0.6, 0.2), _rng(0, 0.6, 0.2), _rng(0.3, 0.8, 0.1))
    ],
    "similarity_flooding": [{}],
    "coma_schema": [{}],
    "coma_instance": [{}],
    # Two runs in the paper (Dist.#1, Dist.#2), each a phase-1 x phase-2 grid.
    "distribution_based": [
        {"threshold1": a, "threshold2": b}
        for rng in (_rng(0.1, 0.2, 0.05), _rng(0.3, 0.5, 0.1))
        for a, b in itertools.product(rng, rng)
    ],
    "jaccard_levenshtein": [{"threshold_dist": t} for t in _rng(0.4, 0.8, 0.1)],
}


def make_matcher(method: str, params: dict):
    from valentine.algorithms import Coma, Cupid, DistributionBased, JaccardDistanceMatcher, SimilarityFlooding

    if method == "cupid":
        return Cupid(**params)
    if method == "similarity_flooding":
        return SimilarityFlooding(**params)
    if method == "coma_schema":
        return Coma(max_n=0, use_schema=True, use_instances=False, threshold=0.0, **params)
    if method == "coma_instance":
        # The paper's COMA 3.0 instance strategy adds the instance matcher to the schema matchers.
        return Coma(max_n=0, use_schema=True, use_instances=True, threshold=0.0, **params)
    if method == "distribution_based":
        return DistributionBased(**params)
    if method == "jaccard_levenshtein":
        return JaccardDistanceMatcher(**params)
    raise ValueError(method)


def _worker(task: dict) -> dict:
    import pandas as pd
    from valentine import valentine_match

    t0 = time.perf_counter()
    src = pd.read_csv(task["src_path"], low_memory=False)
    tgt = pd.read_csv(task["tgt_path"], low_memory=False)
    t_load = time.perf_counter() - t0
    n = task.get("instance_sample_size", 1000)
    t1 = time.perf_counter()
    try:
        res = valentine_match([src, tgt], make_matcher(task["method"], task["params"]), instance_sample_size=n)
        fallback = False
    except Exception as e:
        # The CBC build bundled with PuLP 4.2 aborts inside its nauty symmetry step on some
        # DistributionBased problems. Retry once with orbital branching off; flag the run.
        if task["method"] != "distribution_based" or type(e).__name__ != "PulpSolverError":
            raise
        import valentine.algorithms.distribution_based.discovery as disc
        coin = disc.COIN_CMD
        disc.COIN_CMD = lambda *a, **k: coin(*a, **{**k, "options": [*k.get("options", []), "Orbit off"]})
        res = valentine_match([src, tgt], make_matcher(task["method"], task["params"]), instance_sample_size=n)
        fallback = True
    t_match = time.perf_counter() - t1
    scores = [[p.source_column, p.target_column, float(v)] for p, v in res.items()]
    return {"load_s": t_load, "match_s": t_match, "scores": scores, "cbc_orbit_off": fallback}


def run_task(task: dict, timeout_s: float) -> dict:
    """Run one task in a fresh interpreter; returns a result record."""
    p0 = time.perf_counter()
    base = {k: task[k] for k in ("pair_id", "method", "params", "instance_sample_size")}
    try:
        out = subprocess.run(
            [sys.executable, "-I", "-m", "jevmatch.baselines", json.dumps(task)],
            capture_output=True, text=True, timeout=timeout_s, cwd=ROOT,
            env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "1",
                 "HOME": str(Path.home())},
        )
        wall = time.perf_counter() - p0
        if out.returncode != 0:
            return {**base, "status": "error", "wall_s": wall, "error": out.stderr[-1500:]}
        return {**base, "status": "ok", "wall_s": wall, **json.loads(out.stdout.strip().splitlines()[-1])}
    except subprocess.TimeoutExpired:
        return {**base, "status": "timeout", "wall_s": time.perf_counter() - p0}


def run_many(tasks: list[dict], log_path: Path, workers: int, timeout_s: float, done: set | None = None):
    """Run tasks on a thread pool of subprocesses; append each result to log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    done = done or set()
    todo = [t for t in tasks if _key(t) not in done]
    with ThreadPoolExecutor(workers) as ex, log_path.open("a") as f:
        for rec in ex.map(lambda t: run_task(t, timeout_s), todo):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{rec['status']:7s} {rec['wall_s']:8.1f}s {rec['method']:20s} {rec['pair_id']}", flush=True)


def _key(t: dict) -> tuple:
    return (t["pair_id"], t["method"], json.dumps(t["params"], sort_keys=True), t.get("instance_sample_size", 1000))


def done_keys(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    recs = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    return {_key(r) for r in recs if r["status"] in ("ok", "timeout")}  # errors are retried


if __name__ == "__main__":
    print(json.dumps(_worker(json.loads(sys.argv[1]))))
