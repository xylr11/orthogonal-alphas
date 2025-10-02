import polars as pl
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
import datetime as dt

def make_lagged_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    lags: list[int]
) -> tuple[list[str], pl.DataFrame]:
    out = df.clone()
    new_features = []
    for lag in lags:
        out = out.with_columns([
            pl.col(c).shift(lag).alias(f"{c}_lag{lag}") for c in feature_cols
        ])
        new_features.extend([f"{c}_lag{lag}" for c in feature_cols])
    return feature_cols + new_features, out

def rolling_ridge_with_val(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_return",
    lags: list[int] = [],
    alpha: float = 1e-6,
    train_window: int = 21,
) -> pl.DataFrame:
    """
    Trains ridge regression day by day with optional lags.
    Uses the past `train_window` days as training, current day as validation.
    """
    if lags:
        feature_cols, df = make_lagged_features(df, feature_cols, lags)

    df = df.filter(pl.col('date').gt(dt.date(2015, 1, 1)))
    
    df = df.drop_nulls()

    dates = df["date"].unique().sort().to_list()
    results = []

    for i in tqdm(range(train_window, len(dates)), desc="Rolling Ridge"):
        train_dates = dates[i - train_window:i]
        val_date = dates[i]

        train_df = df.filter(pl.col("date").is_in(train_dates))
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
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_date": val_date,
            "val_r2": val_r2,
            "val_corr": avg_pearson,
        })

    return pl.DataFrame(results)


if __name__ == "__main__":
    df = pl.read_parquet("../../signal_weights/signal_data.parquet")
    df = ( 
        df.with_columns(
            pl.col("return").shift(-1).alias("fwd_return")
        )
        .with_columns(
            pl.col('market_cap').log().alias('log_cap')
        )
    )

    results = rolling_ridge_with_val(
        df,
        feature_cols=["meanrev_z", "bab_z", "momentum_z", 'price', 'specific_risk', 'log_cap'],
        target_col="fwd_return",
        lags=[22, 252],
        alpha=1e-1,
        train_window=1,
    )
    print(f"alpha=1e-1")
    print(f"train_window=5")
    print(results)
    print("mean corr:", results["val_corr"].mean())