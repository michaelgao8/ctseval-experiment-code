import pandas as pd 
import polars as pl 
from pathlib import Path
import logging 
import os
import gc
import numpy as np


DATA_PATH = Path("../data/mimic/processed/features/")


def process_binary_columns(df, binary_cols):
    exclude_cols =[]
    cumsum_df = df.copy()
    cumsum_df.update(cumsum_df.groupby('ID')[binary_cols].cumsum())
    cumsum_df.update(cumsum_df.groupby('ID')[binary_cols].ffill())

    # Rename step

    cumsum_df = cumsum_df.loc[:, ['ID', 't'] + binary_cols]
    new_cols = []
    for col in cumsum_df.columns:
        if col in ['ID', 't']:
            new_cols.append(col)
        elif col in binary_cols:
            new_cols.append(f"{col}_cumcount")
    cumsum_df.columns = new_cols

    time_since_df = df.copy()
    time_since_df = time_since_df.loc[:, ['ID', 't'] + binary_cols]
    for col in binary_cols:
        time_since_df[f'{col}_timesince'] = np.where(time_since_df[col].notnull(), time_since_df['t'], np.nan)

    feature_cols = [f"{col}_timesince" for col in binary_cols]
    time_since_df.update(time_since_df.groupby('ID')[feature_cols].ffill())
    for col in feature_cols:
        time_since_df[col] = time_since_df['t'] - time_since_df[col]

    combined_binary_df = pd.merge(cumsum_df, time_since_df, how='outer', on=['ID', 't'])
    return combined_binary_df

def process_non_binary_columns(df, non_binary_cols):
    pl_df = pl.DataFrame(df)
    del df
    gc.collect()
    pl_df = pl_df.with_columns(pl.col('t').cast(pl.Int32).alias('t'))
    print(pl_df.head())
    results = {}
    for period in ['12i']:
        print(period)
        result = (pl_df
                    .sort(['ID', 't'])
                    .groupby_rolling(by='ID', index_column='t', period=period)
                    .agg([
                         pl.col(col).mean().alias(f"{period}_mean_{col}") for col in non_binary_cols] +
                         [
                            pl.col(col).std().alias(f"{period}_std_{col}") for col in non_binary_cols] +
                         [
                            pl.col(col).max().alias(f"{period}_max_{col}") for col in non_binary_cols] 
                    ))
        results[period] = result
        print(result.head())
    del pl_df
    gc.collect()
    return results['12i'].to_pandas()

def process_file(df):

    include_cols = [col for col in df.columns if col not in ['ID', 't']] 
    print(include_cols)
    binary_cols = [col for col in include_cols if sorted(df[col].dropna().unique()) == [0,1]]
    print(binary_cols)
    non_binary_cols = [col for col in include_cols if col not in binary_cols]

    if non_binary_cols:
        combined_non_binary_df = process_non_binary_columns(df, non_binary_cols)
    
    if binary_cols:
        combined_binary_df = process_binary_columns(df, binary_cols)

    if binary_cols and non_binary_cols:
        combined_df = pd.merge(combined_binary_df, combined_non_binary_df, how='inner', on=['ID', 't'])
        return combined_df
    
    elif non_binary_cols:
        return combined_non_binary_df
    
    elif binary_cols:
        return combined_binary_df

for f in DATA_PATH.iterdir():
    if 'base' in f.name and 'INPUT' not in f.name:
        print(f"Processing {f.name}")
        # Get the name 
        output_name = f.name.split("_base")[0]
        if f.is_dir():
            df = pd.read_parquet(f)
            df = df.drop('chunk', axis=1)
        elif f.is_file():
            df = pd.read_feather(f)
        feature_df = process_file(df)
        feature_df.to_feather(DATA_PATH / f"{output_name}_window.feather")
        del feature_df
        del df
        gc.collect()



