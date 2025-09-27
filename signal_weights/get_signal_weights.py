import sf_quant.data as sfd
import sf_quant.optimizer as sfo
import sf_quant.backtester as sfb
import sf_quant.performance as sfp
import polars as pl
import datetime as dt
import tqdm
import argparse

def get_signal_weights(df: pl.DataFrame, signal: str, start, end, write=False):
    filtered = (
        df.filter(
        (pl.col('date') >= start) & (pl.col('date') <= end)
        )
        .select(['date', 'barrid', f'{signal}_alpha', 'predicted_beta'
        ])
    )
    constraints = [
        sfo.FullInvestment(),
        sfo.LongOnly(),
        sfo.NoBuyingOnMargin(),
        sfo.UnitBeta()
    ]

    weights = sfb.backtest_parallel(filtered.rename({f'{signal}_alpha': 'alpha'}), constraints, 2)

    if write: weights.write_parquet(f'{signal}_weights_{start}_{end}.parquet')

    return weights

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run signal weighting on a parquet dataset."
    )

    parser.add_argument("parquet", help="Path to parquet file containing the data")
    parser.add_argument("signal", help="Signal name (without _alpha suffix)")
    parser.add_argument("start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the output parquet with weights to disk",
    )

    args = parser.parse_args()

    # Parse dates into datetime.date objects
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    # Load parquet into polars DataFrame
    df = pl.read_parquet(args.parquet)

    # Run the signal weights calculation
    weights = get_signal_weights(df, args.signal, start, end, write=args.write)