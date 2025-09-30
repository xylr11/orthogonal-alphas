from sklearn.linear_model import Ridge
import polars as pl
import datetime as dt

def main():
    df = pl.read_parquet('../../signal_weights/signal_data.parquet')
    df = df.filter(pl.col("date") == dt.date(2014, 7, 30))
    df = df.with_columns(pl.col('return').shift(-1).over('barrid').alias('fwd_return'))
    df = df.drop_nulls()
    print(df.head(30))
    features = ['momentum_alpha', 'meanrev_alpha', 'bab_alpha']
    X = df.select(features).to_numpy()
    y = df.select('fwd_return').to_numpy()
    alpha = .000001
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    df = df.with_columns(pl.Series('r_hat', model.predict(X)))
    print(df.head(30))

        # Pearson correlation per date
    df_corr = (
        df.group_by("date")
        .agg(pl.corr("r_hat", "fwd_return").alias("pearson_corr"))
    )

    # Average correlation over all dates
    avg_corr = df_corr.select(
        pl.mean("pearson_corr").alias("avg_pearson"),
    )
    print(avg_corr)

    print(df.select('r_hat').std())
    print(df.select('fwd_return').std())

if __name__ == "__main__":
    main()
    