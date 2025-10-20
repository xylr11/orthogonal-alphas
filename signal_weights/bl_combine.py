import polars as pl
import glob
import os

weights_dir = "bl_weights/"
files = glob.glob(os.path.join(weights_dir, "*.parquet"))
if not files:
    raise FileNotFoundError(f"No parquet files found in {weights_dir}")

dfs = []
for f in files:
    basename = os.path.basename(f)
    
    # Key change: Select the columns and ADD a 'signal' column.
    # We keep the 'weight' column name consistent for now.
    df = (
        pl.read_parquet(f)
    )
    dfs.append(df)

combined = pl.concat(dfs).sort(['date', 'barrid'])

out_file = os.path.join(weights_dir, "bl_weights_pivot.parquet")
combined.write_parquet(out_file)

print(f"[INFO] Combined dataframe written to {out_file}")
print(combined.head())