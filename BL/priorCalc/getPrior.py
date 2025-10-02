import sf_quant.data as sfd
import polars as pl
import numpy as np
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# ------------------------
# CONFIG
# ------------------------
start = dt.date(1995, 6, 30)
end = dt.date(2024, 12, 31)
lookback_months = 3
risk_free_rate = 0
frequency = 252
output_file = "priors.parquet"


# ------------------------
# LOAD DATA
# ------------------------
print("Loading benchmark and returns...")
benchmark = sfd.load_benchmark(start=start, end=end)

columns = ["date", "barrid", "fwd_return"]
returns = sfd.load_assets(start=start, end=end, in_universe=True, columns=columns)

benchmark_returns = (
    benchmark.join(returns, on=["date", "barrid"], how="left")
    .with_columns((pl.col("fwd_return") / 100))
    .group_by("date")
    .agg((pl.col("fwd_return") * pl.col("weight")).sum().alias("benchmark_return"))
    .sort("date")
)


# ------------------------
# FUNCTIONS
# ------------------------
def compute_daily_deltas(benchmark_returns: pl.DataFrame, 
                         lookback_months: int = 3, 
                         risk_free_rate: float = 0.0, 
                         frequency: int = 252) -> pl.DataFrame:
    """
    Compute daily market-implied risk aversion (delta) using rolling windows of benchmark returns.

    Args:
        benchmark_returns (pl.DataFrame): columns ['date', 'benchmark_return'] in decimal space.
        lookback_months (int): how many months of data to use for rolling estimation.
        risk_free_rate (float): annualized risk-free rate (decimal).
        frequency (int): trading periods per year (252 for daily).

    Returns:
        pl.DataFrame: ['date', 'delta'] with one delta per date.
    """
    window = lookback_months * 22 
    
    df = (
        benchmark_returns
        .sort("date")
        # Convert back into percent space for calculations
        .with_columns(
            pl.col("benchmark_return").mul(100)
        )
        .with_columns([
            # rolling mean and var over lookback window
            pl.col("benchmark_return").rolling_mean(window_size=window).alias("mean_daily"),
            pl.col("benchmark_return").rolling_var(window_size=window).alias("var_daily")
        ])
        .with_columns([
            # annualize
            (pl.col("mean_daily") * frequency).alias("mean_ann"),
            (pl.col("var_daily") * frequency).alias("var_ann")
        ])
        .with_columns([
            ((pl.col("mean_ann") - risk_free_rate) / pl.col("var_ann")).alias("delta")
        ])
        .select(["date", "delta"])
    )
    return df

def compute_daily_priors(benchmark: pl.DataFrame, 
                         deltas: pl.DataFrame, 
                         cov_matrix_func) -> pl.DataFrame:
    """
    Compute the Black–Litterman prior (Π) for each date in the sample period.

    For each day, this function:
      1. Extracts the benchmark holdings (barrids and weights).
      2. Retrieves the corresponding market-implied risk aversion (delta).
      3. Builds the covariance matrix Σ for that day's universe of barrids.
      4. Computes the market-implied prior returns using:
            Π = δ Σ w_mkt
         where w_mkt are the benchmark weights for that day.

    Args:
        benchmark (pl.DataFrame): benchmark holdings with columns ['date', 'barrid', 'weight'].
        deltas (pl.DataFrame): daily market-implied risk aversion with ['date', 'delta'].
        cov_matrix_func (Callable): function(date, barrids) -> covariance matrix (NxN numpy array).

    Returns:
        pl.DataFrame: concatenated DataFrame across dates with columns ['date', 'barrid', 'pi'].
    """
    results = []
        
    for date_, subset in benchmark.group_by("date"):
        date_scalar = date_[0]  

        # Extract securities and their benchmark weights for this date
        barrids = subset["barrid"].to_list()
        weights = subset["weight"].to_numpy()
        
        # Match risk aversion parameter for this date
        delta = deltas.filter(pl.col("date") == date_scalar)["delta"][0]

        # If delta is missing or invalid, skip this date
        if delta is None or np.isnan(delta):
            continue

        # Construct covariance matrix for this day's universe
        cov_mat = cov_matrix_func(date_scalar, barrids)
        
        # Compute market-implied prior returns: Π = δ Σ w
        pi = delta * cov_mat.dot(weights)

        results.append(
            pl.DataFrame({
                "date": [date_scalar] * len(barrids),
                "barrid": barrids,
                "pi": pi
            })
        )
        
    return pl.concat(results)

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    print("Computing daily deltas...")
    deltas = compute_daily_deltas(
        benchmark_returns,
        lookback_months=lookback_months,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
    )

    print("Computing daily priors")
    priors = compute_daily_priors(
        benchmark=benchmark,
        deltas=deltas,
        cov_matrix_func=lambda date_, barrids: sfd.construct_covariance_matrix(date_, barrids).drop("barrid").to_numpy()
    )

    print(f"Saving priors to {output_file}...")
    priors.write_parquet(output_file)

    print("Done.")
