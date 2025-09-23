import sf_quant as sf
import polars as pl
import datetime as dt
import tqdm
import json

def fetch_from_sf(start=dt.date(1996, 1, 1), end=dt.date(2024, 12, 31), columns=[
    'date',
    'barrid',
    'price',
    'return',
    'predicted_beta',
    'specific_risk'
    ], get_cov=False, noisy=True):
    """
    Don't run this if you're not on the supercomputer!

    For more columns, message the creator.
    """
    
    df = sf.data.load_assets(
    start=start,
    end=end,
    in_universe=True,
    columns=columns
    )
    df = df.with_columns((pl.col('return') / 100).alias('return')).with_columns((pl.col('specific_risk') / 100).alias('specific_risk'))
    df.write_parquet('asset_data.parquet')

    if get_cov:

        dates = df.select(pl.col("date").unique().sort()).to_series().to_list()

        cov_dict = {}

        for date in tqdm.tqdm(dates, disable=not noisy):

            daily_data = df.filter(pl.col("date") == date)
            barrids = daily_data.select("barrid").unique().sort("barrid").to_series().to_list()

            covariance_matrix = sf.data.construct_covariance_matrix(date, barrids).select(pl.col(pl.Float64)).to_numpy()
            cov_dict[date] = covariance_matrix

        with open('covariances.json', 'w') as out:
            json.dump(cov_dict, out, indent=4)

    if noisy:
        print('Data loaded successfully!')

if __name__ == '__main__':
    fetch_from_sf()