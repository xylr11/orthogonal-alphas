import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import polars as pl
import datetime as dt
import tqdm
import argparse
import os

def get_bl_weights(df:pl.LazyFrame, start, end, n_cpus=8, write=False, gamma=2):
    filtered = (
        df.filter(
        (pl.col('date') >= start) &
        (pl.col('date') <= end) &
        (pl.col('bl_return').is_not_null())
        )
        .select(['date', 'barrid', 'bl_return', 'predicted_beta'
        ])
        .collect()
    )
    if filtered.is_empty(): 
        print("[WARNING] After filtering, input df was empty.")
        return None

    constraints = [
        sfo.FullInvestment(),
        sfo.LongOnly(),
        sfo.NoBuyingOnMargin(),
        sfo.UnitBeta()
    ]

    weights = sfb.backtest_parallel(filtered.rename({f'bl_return': 'alpha'}), constraints, gamma=gamma, n_cpus=n_cpus)

    # Check nothing terrible has happened before writing

    if weights.is_empty():
        print(f"[WARNING] bl {start}–{end}: weights output is EMPTY")
    
    else:
        n_dates = weights.select(pl.col("date")).n_unique()
        total_weight = weights.select(pl.col("weight")).sum().item()
        print(f"[INFO] bl {start}–{end}: {n_dates} dates, total weight sum = {total_weight:.6f}")
    

    if write: weights.write_parquet(f'weights/bl_weights_{start}_{end}.parquet')

    return weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run signal weighting on a parquet dataset."
    )

    parser.add_argument("parquet", help="Path to parquet file containing the data")
    parser.add_argument("start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the output parquet with weights to disk",
    )

    args = parser.parse_args()

    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    print(f"Ray: Slurm allocated {n_cpus} CPUs...")

    # Parse dates into datetime.date objects
    start = dt.date.fromisoformat(args.start)
    print(f"Starting at {start}...")
    end = dt.date.fromisoformat(args.end)
    print(f"Ending at {end}...")

    # Load parquet into polars DataFrame
    print(f"Loading data from {args.parquet}...")
    df = pl.scan_parquet(args.parquet)
    
    gamma = 2

    print(f'Set gamma to {gamma}...')

    # Run the signal weights calculation
    print(f"Starting MVO...")
    weights = get_bl_weights(df, start, end, n_cpus=min(8, n_cpus), write=args.write, gamma=gamma)
    print("Done!")