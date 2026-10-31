"""Pick each baseline's configuration by mean dev Recall@GT (expected) and write it to config/frozen.json."""
import json

from jevmatch.run import ROOT
from jevmatch.score_baselines import score_log

df = score_log(ROOT / "runs" / "dev" / "baselines_grid.jsonl", "dev")
df.drop(columns=["scores"], errors="ignore").to_parquet(ROOT / "results" / "dev_baselines_grid.parquet", index=False)
ok = df[df.status == "ok"]
g = ok.groupby(["method", "params"]).agg(R=("recall_gt_expected", "mean"), n=("pair_id", "size"),
                                         wall=("wall_s", "median")).reset_index()
full = g[g.n == df.pair_id.nunique()]  # configs that finished on every dev pair
best = full.sort_values(["method", "R", "wall"], ascending=[True, False, True]).groupby("method").head(1)
print(best.to_string(index=False))
print("\nconfigs per method:", g.groupby("method").size().to_dict())
F = json.loads((ROOT / "config" / "frozen.json").read_text())
F["baselines"]["params"] = {r.method: json.loads(r.params) for r in best.itertuples()}
F["baselines"]["dev_recall_gt"] = {r.method: round(r.R, 4) for r in best.itertuples()}
F["baselines"]["selection"] = ("Grid over paper Table II ranges on the 24 dev pairs (1,000-row sample); "
                               "highest mean expected Recall@GT, median wall time as tie-breaker.")
(ROOT / "config" / "frozen.json").write_text(json.dumps(F, indent=2))
