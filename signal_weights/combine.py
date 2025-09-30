import polars as pl
import glob
import os

weights_dir = "weights/"
files = glob.glob(os.path.join(weights_dir, "*.parquet"))
if not files:
    raise FileNotFoundError(f"No parquet files found in {weights_dir}")

dfs = []
for f in files:
    basename = os.path.basename(f)
    signal = basename.split("_")[0]  # e.g. momentum, meanrev, bab
    
    # Key change: Select the columns and ADD a 'signal' column.
    # We keep the 'weight' column name consistent for now.
    df = (
        pl.read_parquet(f)
        .select(["date", "barrid", "weight"])
        .with_columns(
            pl.lit(signal).alias("signal") # Add the signal name as a column
        )
    )
    dfs.append(df)

# 1. Concatenate all DataFrames vertically (stacking them up)
all_weights_stacked = pl.concat(dfs, how="vertical")

# 2. Pivot the resulting long DataFrame to the desired wide format
combined = (
    all_weights_stacked
    .pivot(
        index=["date", "barrid"],       # The join keys become the index
        columns="signal",               # The signal column values become new column headers
        values="weight",                # The weight values populate the new columns
        aggregate_function=None,        # Use None since there's one weight per (date, barrid, signal)
    )
    .sort(["date", "barrid"])
)

# Rename the weight columns to include the suffix "_weight"
# The columns are currently named after the signal (e.g., 'momentum', 'meanrev').
# The 'cols' argument should be a list of columns to rename.
weight_cols = [col for col in combined.columns if col not in ("date", "barrid")]
combined = combined.rename({col: f"{col}_weight" for col in weight_cols})


out_file = os.path.join(weights_dir, "all_weights_pivot.parquet")
combined.write_parquet(out_file)

print(f"[INFO] Combined dataframe written to {out_file}")
print(combined.head())