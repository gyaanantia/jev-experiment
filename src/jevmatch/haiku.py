"""Haiku 5.5 baseline (spec section 6).

One Messages API request per source column. The prompt lists every target
column as a candidate; structured output returns one probability per
candidate. Same profiles, sample size and question wording as Jev hybrid.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import anthropic

from .jev import PairResult
from .ledger import Ledger
from .profiles import read_table, table_context, table_profiles

MODEL = "claude-haiku-5-5"
SYSTEM = (
    "You compare columns from two database tables. For each candidate column, "
    "answer the question with the probability (a number from 0 to 1) that the answer is yes. "
    "Judge every candidate independently."
)


@dataclass
class HaikuConfig:
    mode: str = "hybrid"
    n_values: int = 10
    max_len: int = 60
    wording: str = ""
    thinking: str = "disabled"  # "disabled" or "adaptive"
    effort: str = "low"
    seed: int = 42


def build_requests(src, tgt, cfg: HaikuConfig) -> list[dict]:
    kw = dict(n_values=cfg.n_values, max_len=cfg.max_len, seed=cfg.seed)
    ps, pt = table_profiles(src, cfg.mode, **kw), table_profiles(tgt, cfg.mode, **kw)
    ctx_s, ctx_t = table_context(src, cfg.mode), table_context(tgt, cfg.mode)
    reqs = []
    for s in src.columns:
        cands = {f"t_{j}": pt[t] for j, t in enumerate(tgt.columns)}
        payload = {"source_column": ps[s]}
        if ctx_s is not None:
            payload["source_table_columns"] = ctx_s
            payload["candidate_table_columns"] = ctx_t
        payload["candidate_columns"] = cands
        payload["question"] = cfg.wording.replace("`candidate_column`", "each candidate column")
        schema = {
            "type": "object",
            "properties": {k: {"type": "number"} for k in cands},
            "required": list(cands),
            "additionalProperties": False,
        }
        reqs.append({
            "user": json.dumps(payload, ensure_ascii=False),
            "schema": schema,
            "keymap": {f"t_{j}": (s, t) for j, t in enumerate(tgt.columns)},
        })
    return reqs


async def _send(client, req, cfg: HaikuConfig, sem, ledger: Ledger, phase, log_f, meta):
    kwargs = dict(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM,
        messages=[{"role": "user", "content": req["user"]}],
        output_config={"effort": cfg.effort, "format": {"type": "json_schema", "schema": req["schema"]}},
        thinking={"type": cfg.thinking},
    )
    async with sem:
        t0, p0 = time.time(), time.perf_counter()
        try:
            resp = await client.messages.create(**kwargs)
            dt = time.perf_counter() - p0
            u = resp.usage
            in_tok = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            cost = ledger.record("anthropic", phase, in_tok, u.output_tokens)
            text = next(b.text for b in resp.content if b.type == "text")
            answers = json.loads(text)
            rec = {**meta, "status": "ok", "t_start": t0, "latency_s": dt, "request_id": resp.id,
                   "model": resp.model, "stop_reason": resp.stop_reason, "input_tokens": in_tok,
                   "output_tokens": u.output_tokens, "cost_usd": cost, "answers": answers,
                   "keymap": {k: list(v) for k, v in req["keymap"].items()}, "user": req["user"]}
        except Exception as e:
            dt = time.perf_counter() - p0
            rec = {**meta, "status": "error", "t_start": t0, "latency_s": dt,
                   "error": f"{type(e).__name__}: {e}"[:500],
                   "keymap": {k: list(v) for k, v in req["keymap"].items()}, "user": req["user"]}
            if type(e).__name__ == "BudgetExceeded":
                log_f.write(json.dumps(rec) + "\n")
                raise
        log_f.write(json.dumps(rec) + "\n")
        return rec


async def run_pair(client, pair, cfg: HaikuConfig, *, sem, ledger: Ledger, phase, log_path: Path,
                   run_tag: str = "") -> PairResult:
    p0 = time.perf_counter()
    src, tgt = read_table(pair["src_path"]), read_table(pair["tgt_path"])
    reqs = build_requests(src, tgt, cfg)
    est_in = sum(len(r["user"]) // 3 + 300 for r in reqs)
    est_out = sum(len(r["keymap"]) * 12 + 50 for r in reqs)
    ledger.check("anthropic", ledger.cost("anthropic", est_in, est_out))
    meta = {"phase": phase, "run_tag": run_tag, "pair_id": pair["pair_id"], "mode": cfg.mode,
            "n_values": cfg.n_values, "thinking": cfg.thinking, "effort": cfg.effort}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log_f:
        recs = await asyncio.gather(*(
            _send(client, r, cfg, sem, ledger, phase, log_f, {**meta, "req_idx": i})
            for i, r in enumerate(reqs)
        ))
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
            if qid in rec["keymap"]:
                res.scores[tuple(rec["keymap"][qid])] = float(v)
    return res


def make_client(concurrency: int) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(max_retries=5, timeout=120.0)
