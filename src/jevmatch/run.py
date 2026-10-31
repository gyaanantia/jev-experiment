"""Shared helpers to run Jev or Haiku over a set of pairs and score them."""

from __future__ import annotations

import asyncio
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from . import haiku, jev
from .datasets import gt_set
from .ledger import Ledger
from .metrics import all_metrics


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")  # keys stay in the process environment; never printed


def columns(path) -> list[str]:
    """Column names exactly as pandas reads them (handles duplicate headers)."""
    return list(pd.read_csv(path, dtype=str, nrows=0).columns)


def _row(pair, res, cfg, method: str, extra: dict) -> dict:
    src_cols, tgt_cols = columns(pair["src_path"]), columns(pair["tgt_path"])
    m = all_metrics(res.scores, gt_set(pair), src_cols, tgt_cols, probabilistic=True)
    api = res.api_s
    return {
        "pair_id": pair["pair_id"], "method": method, **extra,
        "wall_s": res.wall_s, "n_requests": res.n_requests, "n_errors": res.n_errors,
        "api_p50_s": statistics.median(api) if api else None,
        "api_p95_s": sorted(api)[int(0.95 * (len(api) - 1))] if api else None,
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens, "cost_usd": res.cost_usd,
        "n_scored": len(res.scores), "complete": res.n_errors == 0 and len(res.scores) == len(src_cols) * len(tgt_cols),
        "scores": json.dumps([[s, t, v] for (s, t), v in res.scores.items()]),
        **m,
    }


async def run_jev(pairs: pd.DataFrame, cfg: jev.JevConfig, *, concurrency: int, phase: str, tag: str,
                  ledger: Ledger, log_path: Path, method: str | None = None, extra: dict | None = None,
                  out_jsonl: Path | None = None) -> list[dict]:
    rows = []
    sem = asyncio.Semaphore(concurrency)
    async with jev.make_client(concurrency) as client:
        for _, pair in pairs.iterrows():
            res = await jev.run_pair(client, pair, cfg, sem=sem, ledger=ledger, phase=phase,
                                     log_path=log_path, run_tag=tag)
            row = _row(pair, res, cfg, method or f"jev_{cfg.mode}",
                       {"phase": phase, "tag": tag, "mode": cfg.mode, "packing": cfg.packing,
                        "n_values": cfg.n_values, "max_len": cfg.max_len, "reverse": cfg.reverse,
                        "concurrency": concurrency, **(extra or {})})
            rows.append(row)
            if out_jsonl:
                write_jsonl(row, out_jsonl)
            print(f"[{tag}] {pair['pair_id']:55s} R@GT={row['recall_gt_expected']:.3f} "
                  f"wall={row['wall_s']:.1f}s tok={row['input_tokens']} ${row['cost_usd']:.4f} "
                  f"err={row['n_errors']} total=${ledger.totals['typesafe']:.3f}", flush=True)
    return rows


async def run_haiku(pairs: pd.DataFrame, cfg: haiku.HaikuConfig, *, concurrency: int, phase: str, tag: str,
                    ledger: Ledger, log_path: Path, extra: dict | None = None,
                    out_jsonl: Path | None = None) -> list[dict]:
    rows = []
    sem = asyncio.Semaphore(concurrency)
    async with haiku.make_client(concurrency) as client:
        for _, pair in pairs.iterrows():
            res = await haiku.run_pair(client, pair, cfg, sem=sem, ledger=ledger, phase=phase,
                                       log_path=log_path, run_tag=tag)
            row = _row(pair, res, cfg, "haiku_hybrid",
                       {"phase": phase, "tag": tag, "mode": cfg.mode, "n_values": cfg.n_values,
                        "max_len": cfg.max_len, "thinking": cfg.thinking, "effort": cfg.effort,
                        "concurrency": concurrency, **(extra or {})})
            rows.append(row)
            if out_jsonl:
                write_jsonl(row, out_jsonl)
            print(f"[{tag}] {pair['pair_id']:55s} R@GT={row['recall_gt_expected']:.3f} "
                  f"wall={row['wall_s']:.1f}s in={row['input_tokens']} out={row['output_tokens']} "
                  f"${row['cost_usd']:.4f} err={row['n_errors']} total=${ledger.totals['anthropic']:.3f}",
                  flush=True)
    return rows


def append_results(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    new.to_parquet(path, index=False)


def write_jsonl(row: dict, path: Path) -> None:
    """Append one result row; used by long runs so an interruption loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=float) + "\n")


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
