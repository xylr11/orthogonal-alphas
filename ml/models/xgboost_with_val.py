import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
import polars as pl
import numpy as np
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

def XGBoost_with_val_and_deciles(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_return",
    lags: list[int] = [],
    alpha: float = 1e-6,
    train_window: int = 21,
):
    """
    Trains XGBoost day by day with optional lags.
    Uses the past `train_window` days as training, current day as validation.
    Forms decile portfolios from predictions.
    """
    if lags:
        feature_cols, df = make_lagged_features(df, feature_cols, lags)

    # df = df.filter(pl.col('date').gt(dt.date(2024, 1, 1)))
    # df = df.filter(pl.col('date').lt(dt.date(2024, 6, 30)))
    df = df.drop_nulls()
    
    dates = df["date"].unique().sort().to_list()
    results = []
    portfolio_returns = []

    for i in tqdm(range(train_window, len(dates)), desc="Rolling XGBoost"):
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

        # model = XGBRegressor(
        #     n_estimators=200,
        #     learning_rate=0.1,
        #     max_depth=10,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     reg_lambda=alpha,
        #     tree_method="hist",
        #     n_jobs=-1,
        #     random_state=0
        # )
        model = Ridge(alpha=alpha, fit_intercept=True)

        model.fit(X_train, y_train)
        y_val_hat = model.predict(X_val)

        # Save predictions into val_df
        val_df = val_df.with_columns(pl.Series(name="y_hat", values=y_val_hat))

        # Form deciles by sorting on predicted returns
        val_df = val_df.with_columns(
            pl.col("y_hat").rank("ordinal", descending=True).over("date").alias("rank")
        )
        n = val_df.height
        val_df = val_df.with_columns(((pl.col("rank") * 10) / n).cast(int).alias("decile"))

        # Compute mean realized return per decile
        decile_returns = val_df.group_by("decile").agg(pl.mean(target_col).alias("mean_return"))
        decile_dict = dict(zip(decile_returns["decile"].to_list(), decile_returns["mean_return"].to_list()))

        first_dec = decile_dict.get(0, 0.0)
        tenth_dec = decile_dict.get(9, 0.0)
        spread = tenth_dec - first_dec

        portfolio_returns.append({
            "date": val_date,
            "decile1": first_dec,
            "decile10": tenth_dec,
            "spread": spread
        })

        # Track metrics
        val_r2 = r2_score(y_val, y_val_hat)
        avg_pearson = np.corrcoef(y_val, y_val_hat)[0, 1] if len(y_val) > 1 else np.nan

        results.append({
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_date": val_date,
            "val_r2": val_r2,
            "val_corr": avg_pearson,
        })

    return pl.DataFrame(results), pl.DataFrame(portfolio_returns)


if __name__ == "__main__":
    df = pl.read_parquet("../../signal_weights/signal_data.parquet")
    df = ( 
        df.with_columns(
            pl.col("return").shift(-1).over("barrid").alias("fwd_return")
        )
        .with_columns(
            pl.col('market_cap').log().alias('log_cap')
        )
    )   
     
    alpha = 1e-1
    train_window = 20
    lags = []
    feature_cols = ["meanrev_alpha"]
    target_col = "fwd_return"

    results, port_rets = XGBoost_with_val_and_deciles(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        lags=lags,
        alpha=alpha,
        train_window=train_window,
    )

    print("mean corr:", results["val_corr"].mean())

    # Plot cumulative returns of decile1, decile10, and spread
    port_rets = port_rets.sort("date")
    port_rets = port_rets.with_columns([
        pl.col("decile1").cum_sum().alias("cum_decile1"),
        pl.col("decile10").cum_sum().alias("cum_decile10"),
        pl.col("spread").cum_sum().alias("cum_spread"),
    ])

    plt.figure(figsize=(10,6))
    plt.plot(port_rets["date"], port_rets["cum_decile1"], label="Decile 1 (lowest)")
    plt.plot(port_rets["date"], port_rets["cum_decile10"], label="Decile 10 (highest)")
    plt.plot(port_rets["date"], port_rets["cum_spread"], label="Spread (D10 - D1)")
    plt.legend()
    plt.title("Cumulative Returns by Decile")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.savefig(f'XGBoost_{target_col}_from_{feature_cols}.png')
