"""Jev runner: ask one Noul per (source column, target column) pair.

Packing (spec section 5.3/5.4):
  fanout  - one request per source column; state = source column (+ table
            context), one Noul per target column.
  singles - one request per column pair; state = both columns, one Noul.
Pairs run one after another, so each pair's wall-clock time is clean;
requests inside a pair run concurrently up to `concurrency`.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Noul, NoulCriteria, RetryPolicy

from .ledger import Ledger
from .profiles import read_table, table_context, table_profiles

MODEL = "jev-1.14.0"
DEFAULT_WORDING = (
    "Do `source_column` and `candidate_column` hold the same attribute of the same "
    "kind of entity, so that their values could be stored in one column?"
)


@dataclass
class JevConfig:
    mode: str
    packing: str = "fanout"
    n_values: int = 10
    max_len: int = 60
    wording: str = DEFAULT_WORDING
    criteria: dict | None = None  # {"true": ..., "false": ...}
    reverse: bool = False  # ask with the target column as state
    seed: int = 42


@dataclass
class PairResult:
    pair_id: str
    scores: dict = field(default_factory=dict)
    wall_s: float = 0.0
    api_s: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    n_requests: int = 0
    n_errors: int = 0


def make_client(concurrency: int) -> AsyncTypeSafeClient:
    import httpx2

    limits = httpx2.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    return AsyncTypeSafeClient(
        model=MODEL,
        http_client=httpx2.AsyncClient(limits=limits),
        retry=RetryPolicy(max_retries=5, timeout=60.0),
    )


def _noul(instructions, cfg: JevConfig) -> tuple[Noul, dict]:
    crit = NoulCriteria(**cfg.criteria) if cfg.criteria else None
    q = Noul(instructions=instructions, criteria=crit) if crit else Noul(instructions=instructions)
    raw = {"type": "noul", "instructions": instructions}
    if cfg.criteria:
        raw["criteria"] = cfg.criteria
    return q, raw


def build_requests(src, tgt, cfg: JevConfig) -> list[dict]:
    """Return request specs: {state, questions(raw), sdk_questions, keymap}."""
    kw = dict(n_values=cfg.n_values, max_len=cfg.max_len, seed=cfg.seed)
    a, b = (tgt, src) if cfg.reverse else (src, tgt)
    pa, pb = table_profiles(a, cfg.mode, **kw), table_profiles(b, cfg.mode, **kw)
    ctx_a, ctx_b = table_context(a, cfg.mode), table_context(b, cfg.mode)

    def key(x, y):  # always (source_col, target_col)
        return (y, x) if cfg.reverse else (x, y)

    reqs = []
    if cfg.packing == "fanout":
        for x in a.columns:
            state = {"source_column": pa[x]}
            if ctx_a is not None:
                state["source_table_columns"] = ctx_a
                state["candidate_table_columns"] = ctx_b
            sdk_q, raw_q, keymap = {}, {}, {}
            for j, y in enumerate(b.columns):
                qid = f"t_{j}"
                sdk_q[qid], raw_q[qid] = _noul(
                    {"candidate_column": pb[y], "question": cfg.wording}, cfg
                )
                keymap[qid] = key(x, y)
            reqs.append({"state": state, "questions": raw_q, "sdk": sdk_q, "keymap": keymap})
    elif cfg.packing == "singles":
        for x in a.columns:
            for y in b.columns:
                state = {"source_column": pa[x], "candidate_column": pb[y]}
                if ctx_a is not None:
                    state["source_table_columns"] = ctx_a
                    state["candidate_table_columns"] = ctx_b
                sdk_q, raw_q = {}, {}
                sdk_q["q"], raw_q["q"] = _noul(cfg.wording, cfg)
                reqs.append(
                    {"state": state, "questions": raw_q, "sdk": sdk_q, "keymap": {"q": key(x, y)}}
                )
    else:
        raise ValueError(cfg.packing)
    return reqs


def estimate_tokens(reqs: list[dict]) -> int:
    """Conservative token estimate: 1 token per 3 JSON chars + 400 per request."""
    n = 0
    for r in reqs:
        n += len(json.dumps(r["state"])) // 3 + 400
        n += sum(len(json.dumps(q)) // 3 for q in r["questions"].values())
    return n


async def _send(client, req, sem, ledger: Ledger, phase: str, log_f, meta: dict):
    async with sem:
        t0 = time.time()
        p0 = time.perf_counter()
        try:
            resp = await client.system_one(state=req["state"], questions=req["sdk"])
            dt = time.perf_counter() - p0
            usage = resp.usage
            cost = ledger.record("typesafe", phase, usage.input_tokens, usage.output_tokens)
            answers = {k: v.noul for k, v in resp.nouls.items()}
            rec = {
                **meta,
                "status": "ok",
                "t_start": t0,
                "latency_s": dt,
                "request_id": resp.request_id,
                "model": resp.model,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": cost,
                "answers": answers,
                "keymap": {k: list(v) for k, v in req["keymap"].items()},
                "state": req["state"],
                "questions": req["questions"],
            }
        except Exception as e:  # recorded; the pair is marked incomplete
            dt = time.perf_counter() - p0
            rec = {**meta, "status": "error", "t_start": t0, "latency_s": dt,
                   "error": f"{type(e).__name__}: {e}"[:500],
                   "keymap": {k: list(v) for k, v in req["keymap"].items()},
                   "state": req["state"], "questions": req["questions"]}
            if type(e).__name__ == "BudgetExceeded":
                log_f.write(json.dumps(rec) + "\n")
                raise
        log_f.write(json.dumps(rec) + "\n")
        return rec


async def run_pair(client, pair, cfg: JevConfig, *, sem, ledger: Ledger, phase: str, log_path: Path,
                   run_tag: str = "") -> PairResult:
    p0 = time.perf_counter()
    src, tgt = read_table(pair["src_path"]), read_table(pair["tgt_path"])
    reqs = build_requests(src, tgt, cfg)
    ledger.check("typesafe", estimate_tokens(reqs) * 0.042 / 1e6)
    meta = {"phase": phase, "run_tag": run_tag, "pair_id": pair["pair_id"], "mode": cfg.mode,
            "packing": cfg.packing, "n_values": cfg.n_values, "max_len": cfg.max_len,
            "reverse": cfg.reverse, "wording": cfg.wording, "has_criteria": bool(cfg.criteria)}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_f:
        recs = await asyncio.gather(
            *(_send(client, r, sem, ledger, phase, log_f, {**meta, "req_idx": i}) for i, r in enumerate(reqs))
        )
    res = PairResult(pair_id=pair["pair_id"], wall_s=time.perf_counter() - p0, n_requests=len(recs))
    for rec in recs:
        if rec["status"] != "ok":
            res.n_errors += 1
            continue
        res.api_s.append(rec["latency_s"])
        res.input_tokens += rec["input_tokens"]
        res.output_tokens += rec["output_tokens"]
        res.cost_usd += rec["cost_usd"]
        for qid, v in rec["answers"].items():
            res.scores[tuple(rec["keymap"][qid])] = v
    return res
