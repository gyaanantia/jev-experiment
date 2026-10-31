"""Spend ledger with hard caps.

Every billed API call appends one line to runs/ledger.jsonl. Before a batch
of calls, the caller reserves the batch's estimated cost; the reservation
fails if the running total plus the estimate would pass the provider's cap.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = ROOT / "runs" / "ledger.jsonl"

# USD per token. Sources: docs.typesafe.ai/models.md (read 2026-10-30),
# claude-api model table (cached 2026-10-06; re-check before publishing).
PRICES = {
    "typesafe": {"input": 0.042 / 1e6, "output": 0.0},
    "anthropic": {"input": 0.10 / 1e6, "output": 0.50 / 1e6},
}

# Caps from the spec (section 9). The TypeSafe cap is raised to 9.50 only
# after the user tops up the account; set JEV_CAP_USD to change it.
CAPS = {"typesafe": 4.00, "anthropic": 4.50}


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path = LEDGER_PATH, caps: dict[str, float] | None = None):
        import os

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.caps = dict(caps or CAPS)
        if os.environ.get("JEV_CAP_USD"):
            self.caps["typesafe"] = float(os.environ["JEV_CAP_USD"])
        self._lock = threading.Lock()
        self.totals = {p: 0.0 for p in PRICES}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                rec = json.loads(line)
                self.totals[rec["provider"]] += rec["cost_usd"]

    def cost(self, provider: str, input_tokens: int, output_tokens: int) -> float:
        p = PRICES[provider]
        return input_tokens * p["input"] + output_tokens * p["output"]

    def check(self, provider: str, estimate_usd: float) -> None:
        total = self.totals[provider]
        if total + estimate_usd > self.caps[provider]:
            raise BudgetExceeded(
                f"{provider}: spent ${total:.4f} + estimate ${estimate_usd:.4f} "
                f"> cap ${self.caps[provider]:.2f}"
            )

    def record(self, provider: str, phase: str, input_tokens: int, output_tokens: int) -> float:
        c = self.cost(provider, input_tokens, output_tokens)
        rec = {
            "ts": time.time(),
            "provider": provider,
            "phase": phase,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": c,
        }
        with self._lock:
            self.totals[provider] += c
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            if self.totals[provider] > self.caps[provider]:
                raise BudgetExceeded(f"{provider}: cap ${self.caps[provider]:.2f} reached")
        return c
