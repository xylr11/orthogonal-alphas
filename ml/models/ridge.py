import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm

def make_lagged_features(
        df: pl.DataFrame, 
        feature_cols: list[str], 
        lags: list[int]
) -> pl.DataFrame:
    out = df.clone()
    new_features = []
    for lag in lags:
        out = out.with_columns([
            pl.col(c).shift(lag).alias(f"{c}_lag{lag}") for c in feature_cols
        ])
        new_features.extend([f"{c}_lag{lag}" for c in feature_cols])
    return new_features, out.drop_nulls()

def rolling_ridge_with_val(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_return",
    lags: list[int] = [],
    alpha: float = 1e-6,
):
    """
    Trains ridge regression day by day with optional lags.
    Uses previous day's data as training, current day as validation.
    """
    if lags:
        feature_cols, df = make_lagged_features(df, feature_cols, lags)

    dates = df["date"].unique().sort().to_list()
    results = []

    for i in tqdm(range(1, len(dates)), desc="Rolling Ridge"):
        train_date = dates[i - 1]
        val_date = dates[i]

        train_df = df.filter(pl.col("date") == train_date)
        val_df = df.filter(pl.col("date") == val_date)

        if train_df.height == 0 or val_df.height == 0:
            continue

        X_train = train_df.select(feature_cols).to_numpy()
        y_train = train_df[target_col].to_numpy()

        X_val = val_df.select(feature_cols).to_numpy()
        y_val = val_df[target_col].to_numpy()

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_train, y_train)

        y_val_hat = model.predict(X_val)

        val_r2 = r2_score(y_val, y_val_hat)
        avg_pearson = np.corrcoef(y_val, y_val_hat)[0, 1] if len(y_val) > 1 else np.nan

        results.append({
            "train_date": train_date,
            "val_date": val_date,
            "val_r2": val_r2,
            "val_corr": avg_pearson,
        })

    return pl.DataFrame(results)

if __name__ == "__main__":
    df = pl.read_parquet("../../signal_weights/signal_data.parquet")
    df = df.with_columns(pl.col('return').shift(-1).alias('fwd_return'))
    results = rolling_ridge_with_val(
        df,
        feature_cols=["meanrev_alpha", "bab_alpha", "momentum_alpha"],
        target_col="fwd_return",
        lags=[10, 20, 41],
        alpha=1e-6,
    )
    print(results)
    print(results['val_corr'].mean())
    