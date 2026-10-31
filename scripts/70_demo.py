"""Live demo: ask Jev about one table pair and stream its answers to a browser page.

Serves demo/index.html and a small JSON + Server-Sent Events API on
127.0.0.1:8770 (loopback only). The TypeSafe key stays in this process.

  uv run python -I scripts/70_demo.py               # live Jev calls (spends money)
  DEMO_FAKE=1 uv run python -I scripts/70_demo.py   # random answers, nothing billed
  DEMO_FAKE=fail | DEMO_FAKE=slow                   # fake network failure / no answer

Live requests go through jev._send: ledger phase "demo", cap JEV_CAP_USD (else
$9.90, never above $10), request log runs/demo/jev_demo.jsonl. Replay streams the
recorded requests in runs/full/jev_{mode}.jsonl with their original timing.
"""
import asyncio
import io
import json
import math
import os
import queue
import random
import re
import threading
import time
import traceback
from collections import OrderedDict, defaultdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlsplit

import pandas as pd

from jevmatch import jev
from jevmatch.datasets import gt_set, load_index
from jevmatch.jev import JevConfig, build_requests, estimate_tokens
from jevmatch.ledger import CAPS, PRICES, BudgetExceeded, Ledger
from jevmatch.metrics import full_matrix, recall_at_gt
from jevmatch.profiles import read_table
from jevmatch.run import ROOT  # loads .env into this process; the key is never printed or sent

HOST, PORT = "127.0.0.1", 8770
PAGE = ROOT / "demo" / "index.html"
LOG = ROOT / "runs" / "demo" / "jev_demo.jsonl"
FROZEN = json.loads((ROOT / "config" / "frozen.json").read_text())["jev"]
FAKE = os.environ.get("DEMO_FAKE", "").strip().lower()  # "", "1", "fail", "slow"
ABS_CAP = 10.00  # the user's absolute TypeSafe budget
CAP = min(float(os.environ.get("JEV_CAP_USD") or 9.90), ABS_CAP)
PRICE = PRICES["typesafe"]["input"]
MAX_BYTES, MAX_COLS = 20 * 2**20, 60
MODES = ("hybrid", "schema", "instance")
SLOW_S = 15.0  # no answer after this long: the page offers replay
RECORDED = "2026-10-30–31"
ALLOWED_HOSTS = {f"{HOST}:{PORT}", f"localhost:{PORT}"}

LABELS = {"jev_hybrid": "Jev · hybrid", "jev_schema": "Jev · schema-only", "jev_instance": "Jev · instance-only",
          "haiku_hybrid": "Haiku 5.5 · hybrid", "coma_instance": "COMA (instance)", "coma_schema": "COMA (schema)",
          "cupid": "Cupid", "similarity_flooding": "Similarity Flooding", "distribution_based": "Distribution-based",
          "jaccard_levenshtein": "Jaccard-Levenshtein"}
BASELINES = ["coma_instance", "coma_schema", "cupid", "similarity_flooding", "distribution_based", "jaccard_levenshtein"]

# Talk picks; numbers in {} are filled from results/all_methods.parquet at startup.
PICKS = [
    ("TPC-DI/prospect_both_0_50_ac5_ev", "Noisy names",
     "Names lose their vowels (NumberChildren → prospect_nmbrchldrn). Jev hybrid {jev}; best Valentine matcher: {best}."),
    ("OpenData/miller2_vertical_50_ac5_av", "Semantically-joinable",
     "38 × 38 columns, cryptic names, noisy values. Jev hybrid {jev}; best Valentine matcher: {best}."),
    ("ChEMBL/assays_vertical_1_ac3_ev", "Where Jev misses",
     "The join key assay_id became ss_d and holds plain integers. Jev {jev}; COMA (instance) {coma_instance} from value overlap."),
    ("WikiData/Musicians_viewunion", "WikiData",
     "Curated Wikidata musicians with changed names (givenNameLabel → forename). Jev hybrid {jev}; best Valentine matcher: {best}."),
]


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def _num(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def f2(v) -> str:
    return "–" if v is None else f"{v:.2f}"


class DryLedger(Ledger):
    """Checks the real caps but records nothing (fake client)."""

    def record(self, provider: str, phase: str, input_tokens: int, output_tokens: int) -> float:
        return self.cost(provider, input_tokens, output_tokens)


class FakeClient:
    """Stands in for AsyncTypeSafeClient (DEMO_FAKE): random nouls after a short random delay."""

    def __init__(self, kind: str):
        self.kind = kind

    async def system_one(self, state, questions):
        if self.kind == "fail":
            await asyncio.sleep(random.uniform(0.1, 0.4))
            raise ConnectionError("fake network failure (DEMO_FAKE=fail)")
        await asyncio.sleep(random.uniform(0.2, 1.5) + (SLOW_S + 5 if self.kind == "slow" else 0))
        usage = SimpleNamespace(input_tokens=len(json.dumps(state)) // 3 + 250 * len(questions), output_tokens=0)
        nouls = {q: SimpleNamespace(noul=round(random.random() ** 3, 2)) for q in questions}
        return SimpleNamespace(request_id="fake", model="fake", usage=usage, nouls=nouls)


def make_ledger() -> Ledger:
    led = (DryLedger if FAKE else Ledger)(caps={**CAPS, "typesafe": CAP})
    led.caps["typesafe"] = CAP  # Ledger also reads JEV_CAP_USD; keep the $10 clamp
    return led


def recorded_results() -> dict[str, list[dict]]:
    cols = ["pair_id", "method", "recall_gt_expected", "wall_s", "status", "cost_usd"]
    df = pd.read_parquet(ROOT / "results" / "all_methods.parquet", columns=cols)
    out = defaultdict(list)
    for r in df.itertuples():
        out[r.pair_id].append({"method": r.method, "label": LABELS.get(r.method, r.method),
                               "recall": _num(r.recall_gt_expected), "wall_s": _num(r.wall_s),
                               "status": r.status if isinstance(r.status, str) else "ok", "cost_usd": _num(r.cost_usd)})
    return out


class ReplayIndex:
    """Byte offsets of every recorded request, per mode and pair; built once in the background."""

    PAT = re.compile(rb'"pair_id": ("(?:[^"\\]|\\.)*")')

    def __init__(self):
        self.offsets = {m: defaultdict(list) for m in MODES}
        self.ready = threading.Event()
        threading.Thread(target=self._build, daemon=True, name="replay-index").start()

    @staticmethod
    def path(mode: str):
        return ROOT / "runs" / "full" / f"jev_{mode}.jsonl"

    def _build(self):
        t = time.perf_counter()
        for m in MODES:
            off = 0
            with self.path(m).open("rb") as f:
                for line in f:
                    hit = self.PAT.search(line, 0, 2000)
                    if hit:
                        self.offsets[m][json.loads(hit.group(1))].append(off)
                    off += len(line)
        self.ready.set()
        log(f"replay index ready ({time.perf_counter() - t:.1f} s)")

    def records(self, pair_id: str, mode: str) -> list[dict]:
        self.ready.wait()
        by_idx = {}
        with self.path(mode).open("rb") as f:
            for off in self.offsets[mode].get(pair_id, []):
                f.seek(off)
                r = json.loads(f.readline())
                by_idx[r["req_idx"]] = r  # keep the last record per request
        return list(by_idx.values())


class Cache:
    """Small LRU cache; builds each key once even when several threads ask at the same time."""

    def __init__(self, size: int = 48):
        self.size, self.data, self.lock, self.key_locks = size, OrderedDict(), threading.Lock(), defaultdict(threading.Lock)

    def get(self, key, build):
        with self.lock:
            if key in self.data:
                self.data.move_to_end(key)
                return self.data[key]
            klock = self.key_locks[key]
        with klock:
            with self.lock:
                if key in self.data:
                    return self.data[key]
            val = build()
            with self.lock:
                self.data[key] = val
                while len(self.data) > self.size:
                    self.data.popitem(last=False)
            return val


IDX = load_index()
PAIRS = {pid: row for pid, row in IDX[IDX.split != "dev"].set_index("pair_id", drop=False).iterrows()}
RESULTS = recorded_results()
LEDGER = make_ledger()
REPLAY = ReplayIndex()
CACHE = Cache()
UPLOADS: dict[str, tuple[str, pd.DataFrame]] = {}  # slot -> (file name, table)
UPLOAD_VER = [0]


def config(mode: str) -> JevConfig:
    """Frozen Jev settings, built as scripts/30_full_jev.py does."""
    return JevConfig(mode=mode, n_values=FROZEN["n_values"], max_len=FROZEN["max_len"],
                     wording=FROZEN["wording"], criteria=FROZEN["criteria"])


def pkey(pid: str) -> str:
    return f"upload#{UPLOAD_VER[0]}" if pid == "upload" else pid


def tables(pid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pid == "upload":
        if "src" not in UPLOADS or "tgt" not in UPLOADS:
            raise KeyError("upload")
        return UPLOADS["src"][1], UPLOADS["tgt"][1]
    p = PAIRS[pid]
    return CACHE.get(("tables", pid), lambda: (read_table(p["src_path"]), read_table(p["tgt_path"])))


def requests_for(pid: str, mode: str) -> tuple[list[dict], int]:
    def build():
        src, tgt = tables(pid)
        reqs = build_requests(src, tgt, config(mode))
        return reqs, estimate_tokens(reqs)
    return CACHE.get(("reqs", pkey(pid), mode), build)


def preview(df: pd.DataFrame, name: str, n: int = 3) -> dict:
    rows = [[None if pd.isna(v) else str(v)[:200] for v in row] for row in df.head(n).itertuples(index=False)]
    return {"name": name, "columns": [str(c) for c in df.columns], "rows": rows, "n_rows": len(df)}


def budget() -> dict:
    spent = LEDGER.totals["typesafe"]
    return {"spent": spent, "cap": CAP, "left": CAP - spent - LIVE.reserved, "fake": bool(FAKE)}


def picks() -> list[dict]:
    out = []
    for pid, title, text in PICKS:
        if pid not in PAIRS:
            continue
        rec = {r["method"]: r for r in RESULTS.get(pid, [])}
        base = [rec[m] for m in BASELINES if m in rec and rec[m]["recall"] is not None]
        best = max(base, key=lambda r: r["recall"])
        p = PAIRS[pid]
        reason = text.format(jev=f2(rec["jev_hybrid"]["recall"]), best=f"{best['label']} {f2(best['recall'])}",
                             coma_instance=f2(rec.get("coma_instance", {}).get("recall")))
        out.append({"pair_id": pid, "title": title, "reason": reason, "source": p["source"], "scenario": p["scenario"],
                    "size": f"{p['n_src_cols']} × {p['n_tgt_cols']}", "k": int(p["n_gt"])})
    return out


def pair_info(pid: str) -> dict:
    src, tgt = tables(pid)
    sc, tc = [str(c) for c in src.columns], [str(c) for c in tgt.columns]
    if pid == "upload":
        return {"pair_id": "upload", "upload": True, "src": preview(src, UPLOADS["src"][0]),
                "tgt": preview(tgt, UPLOADS["tgt"][0]), "gt": None, "k": None, "recorded": [], "replay": False}
    p = PAIRS[pid]
    gt = sorted(gt_set(p))
    txt = {k: p[k] if isinstance(p[k], str) else None for k in ("source", "scenario", "schema_noise", "instance_noise", "split")}
    return {"pair_id": pid, "upload": False, **txt,
            "src": preview(src, "source"), "tgt": preview(tgt, "target"), "k": len(gt),
            "gt": [[sc.index(s), tc.index(t)] for s, t in gt if s in sc and t in tc],
            "recorded": RESULTS.get(pid, []), "recorded_on": RECORDED, "replay": True}


def req_event(rec: dict, si: dict, ti: dict, t_rel: float) -> dict:
    """One finished request as the page needs it: row i, scores by target index, the exact request."""
    n = len(ti)
    scores, qids, questions = [None] * n, [None] * n, [None] * n
    i = None
    for qid, (s, t) in rec["keymap"].items():
        i, j = si[s], ti[t]
        qids[j], questions[j] = qid, rec["questions"][qid]
        if rec["status"] == "ok":
            scores[j] = rec["answers"][qid]
    return {"i": i, "req_idx": rec["req_idx"], "status": rec["status"], "t": t_rel, "latency_s": rec["latency_s"],
            "input_tokens": rec.get("input_tokens", 0), "cost_usd": rec.get("cost_usd", 0.0),
            "state": rec["state"], "qids": qids, "questions": questions, "scores": scores}


def summary(pid: str, sc: list, tc: list, recs: list[dict], wall_s: float, kind: str) -> dict:
    """Totals, Recall@GT (expected over ties) and the cells above / tied at the top-k cut-off."""
    scores = {}
    for r in recs:
        if r["status"] == "ok":
            for qid, v in r["answers"].items():
                scores[tuple(r["keymap"][qid])] = v
    ok = [r for r in recs if r["status"] == "ok"]
    out = {"kind": kind, "wall_s": wall_s, "n_ok": len(ok), "n_err": len(recs) - len(ok),
           "input_tokens": sum(r.get("input_tokens", 0) for r in ok), "cost_usd": sum(r.get("cost_usd", 0.0) for r in ok),
           "budget": budget()}
    si, ti = {c: i for i, c in enumerate(sc)}, {c: j for j, c in enumerate(tc)}
    if pid == "upload" or not scores:
        best = defaultdict(lambda: (None, -1.0))
        for (s, t), v in scores.items():
            if v > best[s][1]:
                best[s] = (t, v)
        out["picks"] = [[si[s], ti[t]] for s, (t, v) in best.items() if v >= 0.5]
        return out
    gt = gt_set(PAIRS[pid])
    full = full_matrix(scores, sc, tc)
    out["recall"] = recall_at_gt(full, gt)
    k = len(gt)
    cut = sorted(full.values(), reverse=True)[k - 1]
    above = [[si[s], ti[t]] for (s, t), v in full.items() if v > cut]
    tied = [[si[s], ti[t]] for (s, t), v in full.items() if v == cut] if math.isfinite(cut) else []
    slots = k - len(above)
    if len(tied) <= slots:  # the whole tie group fits in the top k
        above, tied = above + tied, []
    out.update(above=above, tied=tied, slots=slots)
    return out


class Live:
    """One asyncio loop in a background thread; live requests go through jev._send."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True, name="jev-loop").start()
        self.client = None
        self.reserved = 0.0
        self.lock = threading.Lock()

    def reserve(self, usd: float) -> None:
        """Raise BudgetExceeded if this run's estimate plus runs in flight would pass the cap."""
        with self.lock:
            LEDGER.check("typesafe", usd + self.reserved)
            self.reserved += usd

    def start(self, reqs: list[dict], meta: dict, est_usd: float) -> queue.Queue:
        q = queue.Queue()
        asyncio.run_coroutine_threadsafe(self._run(reqs, meta, est_usd, q), self.loop)
        return q

    async def _run(self, reqs, meta, est_usd, q):
        try:
            if self.client is None:
                self.client = FakeClient(FAKE) if FAKE else jev.make_client(FROZEN["concurrency"])
            sem = asyncio.Semaphore(FROZEN["concurrency"])
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with (open(os.devnull, "a") if FAKE else LOG.open("a", buffering=1)) as log_f:
                tasks = [asyncio.ensure_future(jev._send(self.client, r, sem, LEDGER, "demo", log_f, {**meta, "req_idx": i}))
                         for i, r in enumerate(reqs)]
                for fut in asyncio.as_completed(tasks):
                    try:
                        rec = await fut
                    except BudgetExceeded as e:
                        log(f"budget: {e}")
                        q.put(("budget", None))
                        continue
                    if rec["status"] != "ok":
                        log(f"request {rec['req_idx']} failed: {rec.get('error')}")
                    q.put(("rec", rec))
        except Exception:
            traceback.print_exc()
            q.put(("fail", None))
        finally:
            with self.lock:
                self.reserved -= est_usd
            q.put(("end", None))


LIVE = Live()


class Handler(BaseHTTPRequestHandler):
    server_version = "jevdemo"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def _dispatch(self, method: str):
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in ALLOWED_HOSTS or (origin and origin.split("//")[-1] not in ALLOWED_HOSTS):
            self.send_error(403)
            return
        url = urlsplit(self.path)
        qs = {k: v[0] for k, v in parse_qs(url.query).items()}
        route = ROUTES.get((method, url.path))
        self.streaming = False
        if route is None:
            self.send_json({"error": "Not found."}, 404)
            return
        try:
            route(self, qs)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            if self.streaming:
                self.event("error", {"kind": "server", "message": "Server error. Details are in the server console."})
            else:
                self.send_json({"error": "Server error. Details are in the server console."}, 500)

    def send_json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.streaming = True

    def event(self, name: str, data) -> bool:
        """Write one SSE event; False once the page has gone away."""
        try:
            self.wfile.write(f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False, allow_nan=False)}\n\n".encode())
            return True
        except (BrokenPipeError, ConnectionResetError):
            return False

    # ---- routes ----
    def page(self, qs):
        body = PAGE.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; "
                         "style-src 'unsafe-inline'; connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'")
        self.end_headers()
        self.wfile.write(body)

    def init(self, qs):
        pairs = [{"id": pid, "source": p["source"], "scenario": p["scenario"], "size": f"{p['n_src_cols']} × {p['n_tgt_cols']}",
                  "k": int(p["n_gt"])} for pid, p in sorted(PAIRS.items())]
        self.send_json({"picks": picks(), "pairs": pairs, "budget": budget(), "fake": FAKE or None,
                        "model": FROZEN["model"], "concurrency": FROZEN["concurrency"], "n_values": FROZEN["n_values"],
                        "max_mb": MAX_BYTES // 2**20, "max_cols": MAX_COLS, "slow_s": SLOW_S, "recorded_on": RECORDED})

    def budget(self, qs):
        self.send_json(budget())

    def pair(self, qs):
        pid = qs.get("id", "")
        if pid != "upload" and pid not in PAIRS:
            self.send_json({"error": "Unknown pair."}, 404)
            return
        try:
            self.send_json(pair_info(pid))
        except KeyError:
            self.send_json({"error": "Upload both CSV files first."}, 400)

    def estimate(self, qs):
        pid, mode = qs.get("id", ""), qs.get("mode", "hybrid")
        if (pid != "upload" and pid not in PAIRS) or mode not in MODES:
            self.send_json({"error": "Unknown pair or mode."}, 400)
            return
        if pid == "upload" and not ("src" in UPLOADS and "tgt" in UPLOADS):
            self.send_json({"error": "Upload both CSV files first."}, 400)
            return
        reqs, tok = requests_for(pid, mode)
        b = budget()
        self.send_json({"n_requests": len(reqs), "tokens": tok, "usd": tok * PRICE, "fits": tok * PRICE <= b["left"], "budget": b})

    def upload(self, qs):
        slot, name = qs.get("slot"), (unquote(self.headers.get("X-File-Name") or "") or "file.csv")[:120]
        n = int(self.headers.get("Content-Length") or 0)
        if slot not in ("src", "tgt"):
            self.send_json({"error": "Bad upload slot."}, 400)
            return
        if n > MAX_BYTES:
            left = n if n <= 10 * MAX_BYTES else 0
            while left > 0:  # drain so the browser can read the reply
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                left -= len(chunk)
            self.close_connection = True
            self.send_json({"error": f"{name} is {n / 2**20:.1f} MB. The limit is {MAX_BYTES // 2**20} MB per file."}, 413)
            return
        body = self.rfile.read(n) if n else b""
        try:
            try:
                text = body.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = body.decode("latin-1")
            df = read_table(io.StringIO(text))
        except Exception as e:
            log(f"upload {slot} {name!r}: {type(e).__name__}: {e}"[:300])
            self.send_json({"error": f"Could not read {name} as a CSV file with a header row."}, 400)
            return
        if df.shape[1] > MAX_COLS:
            self.send_json({"error": f"{name} has {df.shape[1]} columns. The limit is {MAX_COLS}."}, 400)
            return
        if df.shape[0] == 0 or df.shape[1] == 0:
            self.send_json({"error": f"{name} has no data rows."}, 400)
            return
        df.columns = [str(c) for c in df.columns]
        UPLOADS[slot] = (name, df)
        UPLOAD_VER[0] += 1
        log(f"upload {slot}: {name!r} {df.shape[0]} rows × {df.shape[1]} columns")
        self.send_json({"ok": True, "slot": slot, "name": name, "n_rows": len(df), "n_cols": df.shape[1],
                        "ready": "src" in UPLOADS and "tgt" in UPLOADS})

    def run(self, qs):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            pid, mode, replay = body["id"], body["mode"], bool(body.get("replay"))
        except (ValueError, KeyError, TypeError):
            self.send_json({"error": "Bad request."}, 400)
            return
        self.start_stream()
        if (pid != "upload" and pid not in PAIRS) or mode not in MODES or (pid == "upload" and not ("src" in UPLOADS and "tgt" in UPLOADS)):
            self.event("error", {"kind": "bad_request", "message": "Pick a pair first."})
            return
        if replay:
            self.run_replay(pid, mode)
        else:
            self.run_live(pid, mode)

    def run_replay(self, pid: str, mode: str):
        if pid == "upload":
            self.event("error", {"kind": "bad_request", "message": "Uploaded tables have no recorded run."})
            return
        recs = [r for r in REPLAY.records(pid, mode) if r["status"] == "ok"]
        if not recs:
            self.event("error", {"kind": "bad_request", "message": "No recorded run for this pair and mode."})
            return
        src, tgt = tables(pid)
        sc, tc = [str(c) for c in src.columns], [str(c) for c in tgt.columns]
        si, ti = {c: i for i, c in enumerate(sc)}, {c: j for j, c in enumerate(tc)}
        t0 = min(r["t_start"] for r in recs)
        recs.sort(key=lambda r: r["t_start"] + r["latency_s"])
        span = recs[-1]["t_start"] + recs[-1]["latency_s"] - t0
        log(f"replay {pid} {mode}: {len(recs)} requests over {span:.2f} s")
        self.event("start", {"kind": "replay", "n": len(recs), "file": f"runs/full/jev_{mode}.jsonl",
                             "recorded_at": datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M"),
                             "tokens": sum(r["input_tokens"] for r in recs), "usd": sum(r["cost_usd"] for r in recs)})
        p0 = time.perf_counter()
        for r in recs:
            t_rel = r["t_start"] + r["latency_s"] - t0
            delay = t_rel - (time.perf_counter() - p0)
            if delay > 0:
                time.sleep(delay)
            if not self.event("req", req_event(r, si, ti, t_rel)):
                return
        self.event("done", summary(pid, sc, tc, recs, span, "replay"))

    def run_live(self, pid: str, mode: str):
        reqs, tok = requests_for(pid, mode)
        est = tok * PRICE
        try:
            LIVE.reserve(est)
        except BudgetExceeded as e:
            log(f"refused {pid} {mode}: {e}")
            left = max(budget()["left"], 0.0)
            self.event("error", {"kind": "budget", "message": f"Refused: this run is estimated at ${est:.4f}, but only "
                                 f"${left:.4f} is left under the demo cap (${CAP:.2f})."})
            return
        src, tgt = tables(pid)
        sc, tc = [str(c) for c in src.columns], [str(c) for c in tgt.columns]
        si, ti = {c: i for i, c in enumerate(sc)}, {c: j for j, c in enumerate(tc)}
        log_id = f"upload:{UPLOADS['src'][0]}|{UPLOADS['tgt'][0]}" if pid == "upload" else pid
        cfg = config(mode)
        meta = {"phase": "demo", "run_tag": "demo-fake" if FAKE else "demo-live", "pair_id": log_id, "mode": mode,
                "packing": cfg.packing, "n_values": cfg.n_values, "max_len": cfg.max_len, "reverse": cfg.reverse,
                "wording": cfg.wording, "has_criteria": bool(cfg.criteria)}
        kind = "fake" if FAKE else "live"
        log(f"{kind} {log_id} {mode}: {len(reqs)} requests, estimate {tok} tokens ${est:.4f}")
        alive = self.event("start", {"kind": kind, "n": len(reqs), "tokens": tok, "usd": est})
        p0 = time.perf_counter()
        q = LIVE.start(reqs, meta, est)
        recs, slow_sent, failed, over, last = [], False, False, False, p0
        while True:
            try:
                typ, rec = q.get(timeout=0.25)
            except queue.Empty:
                if alive and not slow_sent and time.perf_counter() - last > SLOW_S:
                    slow_sent = True
                    alive = self.event("slow", {"after_s": SLOW_S, "done": len(recs)})
                continue
            if typ == "end":
                break
            if typ == "rec":
                recs.append(rec)
                last = time.perf_counter()
                if alive:
                    alive = self.event("req", req_event(rec, si, ti, time.perf_counter() - p0))
            failed |= typ == "fail"
            over |= typ == "budget"
        wall = time.perf_counter() - p0
        out = summary(pid, sc, tc, recs, wall, kind)
        log(f"{kind} {log_id} {mode}: {out['n_ok']} ok, {out['n_err']} failed, {out['input_tokens']} tokens, "
            f"${out['cost_usd']:.4f}, {wall:.2f} s, recall {out.get('recall', {}).get('expected', '-')}; "
            f"budget left ${out['budget']['left']:.4f}")
        if not alive:
            return
        if failed or over or out["n_ok"] < len(reqs):
            msg = ("Stopped: the demo budget cap was reached." if over else
                   "Live run failed: no answer from Jev (network or service error)." if out["n_ok"] == 0 else
                   f"{len(reqs) - out['n_ok']} of {len(reqs)} requests failed. Their cells stay empty.")
            self.event("error", {"kind": "budget" if over else "network", "message": msg, "partial": out["n_ok"] > 0})
        if out["n_ok"]:
            self.event("done", out)


ROUTES = {("GET", "/"): Handler.page, ("GET", "/index.html"): Handler.page, ("GET", "/api/init"): Handler.init,
          ("GET", "/api/budget"): Handler.budget, ("GET", "/api/pair"): Handler.pair,
          ("GET", "/api/estimate"): Handler.estimate, ("POST", "/api/upload"): Handler.upload,
          ("POST", "/api/run"): Handler.run}


def warm():
    """Build the talk picks' requests ahead of time (profiles of large tables take a few seconds)."""
    t = time.perf_counter()
    for mode in MODES:
        for pid, *_ in PICKS:
            try:
                requests_for(pid, mode)
            except Exception:
                traceback.print_exc()
    log(f"talk picks ready in all modes ({time.perf_counter() - t:.1f} s)")


def main():
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    b = budget()
    log(f"Jev demo at http://{HOST}:{PORT}/  ({'FAKE client (' + FAKE + '), nothing billed' if FAKE else 'LIVE Jev calls'}; "
        f"spent ${b['spent']:.4f}, cap ${b['cap']:.2f}, left ${b['left']:.4f}). Ctrl-C stops.")
    threading.Thread(target=warm, daemon=True, name="warm").start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        log("stopped")


if __name__ == "__main__":
    main()
