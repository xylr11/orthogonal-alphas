import xgboost 
import datetime
import polars as pl

def get_data():
    signal_data = pl.read_parquet('../../signal_weights/signal_data.parquet')
    market_data = pl.read_parquet('../../russell_3000_daily.parquet')
    df = market_data.join(signal_data, on=['barrid', 'date'], how='left').select('date', 'barrid', 'momentum_alpha', 'meanrev_alpha', 'bab_alpha', 'return')
    df = df.with_columns(
        pl.col('return').truediv(100)
    )
    df = df.with_columns(pl.col('return').shift(1).over('barrid').alias('fwd_return')).drop_nulls('fwd_return')
    

if __name__ == "__main__":
    print(get_data())