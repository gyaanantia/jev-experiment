"""Build data/pairs.parquet (pair metadata, ground truth, splits)."""
from jevmatch.datasets import INDEX, build_index

df = build_index()
df.to_parquet(INDEX, index=False)
print(df.groupby(["source", "scenario"]).size().to_string())
print("\nsplits:", df.split.value_counts().to_dict(), "| haiku subset:", int(df.haiku_subset.sum()))
print("GT columns missing from tables:", int((df.gt_missing_cols != "[]").sum()), "pairs")
print("\ncolumn pairs (cells) by source:")
print(df.groupby("source").agg(pairs=("pair_id", "size"), cells=("n_cells", "sum"),
      mean_src_cols=("n_src_cols", "mean"), mean_rows=("n_src_rows", "mean")).round(1).to_string())
print("\npilot:", df[df.split == "pilot"][["pair_id", "n_src_cols", "n_tgt_cols", "n_gt"]].to_string(index=False))
