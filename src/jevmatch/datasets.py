"""Pair index over the Valentine (Zenodo) and NYU data (spec section 3)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ZENODO = ROOT / "data" / "raw" / "zenodo" / "extracted" / "Valentine-datasets"
NYU = ROOT / "data" / "raw" / "nyu"
INDEX = ROOT / "data" / "pairs.parquet"

SCENARIOS = ["Unionable", "View-Unionable", "Joinable", "Semantically-Joinable"]
_WIKI = {"unionable": "Unionable", "viewunion": "View-Unionable", "joinable": "Joinable",
         "semjoinable": "Semantically-Joinable"}
# Ground-truth typo in two WikiData mapping files: the source column is
# `givenNameLabel`; `givenName` does not exist in the table. Applied to all methods.
GT_FIXES = {"givenName": "givenNameLabel"}


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return next(csv.reader(f))


def _rows(path: Path) -> int:
    with path.open("rb") as f:
        return sum(buf.count(b"\n") for buf in iter(lambda: f.read(1 << 20), b"")) - 1


def _parse_name(name: str) -> dict:
    """prospect_both_0_50_ac1_av -> split/row/col overlap and noise flags."""
    m = re.match(r".*_(horizontal|vertical|both)_(\d+)(?:_(\d+))?_(ec|ac\d)_(ev|av)$", name)
    if not m:
        return {}
    split, a, b, cols, vals = m.groups()
    if split == "horizontal":
        row, col = int(a), 100
    elif split == "vertical":
        row, col = 100, int(a)
    else:
        row, col = int(a), int(b)
    return {"split_kind": split, "row_overlap": row, "col_overlap": col,
            "schema_noise": "verbatim" if cols == "ec" else "noisy",
            "schema_variant": cols, "instance_noise": "verbatim" if vals == "ev" else "noisy"}


def _zenodo_pairs():
    for src_dir in sorted(ZENODO.iterdir()):
        if not src_dir.is_dir():
            continue
        source = {"Wikidata": "WikiData"}.get(src_dir.name, src_dir.name)
        for mapping in sorted(src_dir.rglob("*_mapping.json")):
            d = mapping.parent
            stem = mapping.name[: -len("_mapping.json")]
            rec = {"source": source, "pair_id": f"{source}/{d.name}",
                   "src_path": str(d / f"{stem}_source.csv"), "tgt_path": str(d / f"{stem}_target.csv")}
            gt = [(GT_FIXES.get(m["source_column"], m["source_column"]) if source == "WikiData" else m["source_column"],
                   m["target_column"]) for m in json.loads(mapping.read_text())["matches"]]
            if source in ("TPC-DI", "OpenData", "ChEMBL"):
                rec["scenario"] = d.parent.name.replace(" ", "")
                rec.update(_parse_name(d.name))
            elif source == "WikiData":
                rec["scenario"] = _WIKI[d.name.split("_", 1)[1]]
                rec.update(schema_noise="curated", instance_noise="curated")
            else:  # Magellan: curated unionable pairs (paper p. 473)
                rec["scenario"] = "Unionable"
                rec.update(schema_noise="curated", instance_noise="curated")
            yield rec, gt


def _nyu_pairs():
    sha = (NYU / "COMMIT_SHA").read_text().strip()
    for d in sorted(p for p in NYU.iterdir() if p.is_dir()):
        g = json.loads((d / "ground_truth.json").read_text())
        gt = [(m["source_column"], m["target_column"]) for m in g["matches"]]
        rec = {"source": "NYU", "pair_id": f"NYU/{d.name}", "src_path": str(d / "source_table.csv"),
               "tgt_path": str(d / "target_table.csv"), "scenario": "NYU",
               "schema_noise": "curated", "instance_noise": "curated", "nyu_commit": sha}
        yield rec, gt


def build_index() -> pd.DataFrame:
    rows = []
    for rec, gt in list(_zenodo_pairs()) + list(_nyu_pairs()):
        sc, tc = _header(Path(rec["src_path"])), _header(Path(rec["tgt_path"]))
        bad = [(s, t) for s, t in gt if s not in sc or t not in tc]
        rec.update(n_src_cols=len(sc), n_tgt_cols=len(tc), n_cells=len(sc) * len(tc),
                   n_src_rows=_rows(Path(rec["src_path"])), n_tgt_rows=_rows(Path(rec["tgt_path"])),
                   n_gt=len(gt), gt=json.dumps(gt), gt_missing_cols=json.dumps(bad))
        rows.append(rec)
    df = pd.DataFrame(rows)
    return assign_splits(df)


def assign_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = "test"
    fab = df["source"].isin(["TPC-DI", "OpenData", "ChEMBL"])

    # Pilot: the hardest view-unionable variant per fabricated source, plus the
    # smallest WikiData, Magellan and NYU pair.
    pilot = []
    for s, prefix in [("TPC-DI", "prospect"), ("OpenData", "miller2"), ("ChEMBL", "assays")]:
        pilot.append(f"{s}/{prefix}_both_0_50_ac1_av")
    for s in ["WikiData", "Magellan", "NYU"]:
        pilot.append(df[df.source == s].sort_values("n_cells").iloc[0]["pair_id"])
    df.loc[df.pair_id.isin(pilot), "split"] = "pilot"

    # Dev: one pair per source x scenario x schema-noise cell (24 pairs).
    for (s, sc, sn), g in df[fab & (df.split == "test")].groupby(["source", "scenario", "schema_noise"]):
        df.loc[g.index[rng.integers(len(g))], "split"] = "dev"

    # Haiku subset: 2 test pairs per fabricated cell + all WikiData + Magellan + 1 NYU.
    df["haiku_subset"] = False
    for _, g in df[fab & (df.split == "test")].groupby(["source", "scenario", "schema_noise"]):
        df.loc[rng.choice(g.index, size=min(2, len(g)), replace=False), "haiku_subset"] = True
    df.loc[df.source.isin(["WikiData", "Magellan"]) & (df.split == "test"), "haiku_subset"] = True
    nyu = df[(df.source == "NYU") & (df.split == "test")]
    df.loc[nyu.sort_values("n_cells").index[0], "haiku_subset"] = True
    return df


def load_index() -> pd.DataFrame:
    return pd.read_parquet(INDEX)


def gt_set(row) -> set:
    return {tuple(x) for x in json.loads(row["gt"])}
