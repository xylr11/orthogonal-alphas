# combine_weights.py
import polars as pl
import glob

def main():
    files = glob.glob("weights/*.parquet")
    if not files:
        raise FileNotFoundError("No parquet files found in weights/")

    print(f"Found {len(files)} weight files")

    dfs = [pl.read_parquet(f) for f in files]
    combined = pl.concat(dfs)

    print(f"Combined shape: {combined.shape}")
    combined.write_parquet("weights/all_weights.parquet")

if __name__ == "__main__":
    main()
