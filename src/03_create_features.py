import math
import gc
import pandas as pd 
import pickle
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


# Define helpers
def process_cols(x, first_cols):
    if isinstance(first_cols, str):
        first_cols = [first_cols]
    if x[0] in first_cols:
        return x[0]
    else:
        return x[1]

# Define Constants
FILTERED_STACKED_DATA_PATH = Path("../data/mimic/processed/formatted/filtered_data.stacked.p")
STATIC_FEATURE_PATH = Path("../data/mimic/processed/features/static_features.feather")
FEATURE_PATH = Path("../data/mimic/processed/features/") 
BASE_SUFFIX = "base"
PROCESS_STATIC = True
PROCESS_DYNAMIC = True

logging.info("Reading in data")
with open(FILTERED_STACKED_DATA_PATH, "rb") as f:
    stacked_data = pickle.load(f)

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast dataframe columns to the smallest possible datatype.
    
    :param df: Input dataframe.
    :return: Downcasted dataframe.
    """
    # Downcast numeric types
    df_downcasted = df.apply(pd.to_numeric, downcast='integer').apply(pd.to_numeric, downcast='float')
    return df_downcasted


if PROCESS_STATIC:
    logging.info("Processing static features")
    static_data = (pd.pivot_table(stacked_data['TIME_INVARIANT'].drop('t', axis=1),
                    index=['ID'], 
                    columns=['variable_name'],
                    aggfunc='first',
                    dropna=False))

    static_data_to_keep = [
                            'ID',
                            'ADMISSION_LOCATION',
                            'ADMISSION_TYPE',
                            'AGE',
                            'ETHNICITY',
                            'GENDER',
                            'INSURANCE',
                            'LANGUAGE',
                            'MARITAL_STATUS',
                            'RELIGION'
                        ]

    static_data = static_data.reset_index()
    static_data.columns = [process_cols(col, "ID") for col in static_data.columns]
    static_data = static_data.loc[:, static_data_to_keep].copy()
    logging.info("Creating dummy variables")
    static_data = pd.get_dummies(static_data, columns = [col for col in static_data_to_keep if col not in  ["ID","AGE"]])

    logging.info(f"Writing out static features to {STATIC_FEATURE_PATH}")
    static_data.to_feather(STATIC_FEATURE_PATH)

if PROCESS_DYNAMIC:
    del stacked_data['TIME_INVARIANT']
    logging.info("Processing dynamic data")
    for data_type in stacked_data.keys():
        if data_type not in ['TIME_INVARIANT', 'INPUTEVENTS_MV', 'PROCEDUREEVENTS_MV', 'CHARTEVENTS']: # TODO: CHANGE BACK  to != 'TIME_INVARIANT'
            logging.info(f"Now Processing {data_type}")
            df = stacked_data[data_type]
            try:
                df['t'] = [math.ceil(x) for x in df['t']]
            except:
                df['t'] = df['t_start']
                df = df.drop(['t_start', 't_end'], axis=1)
            df['variable_value'] = pd.to_numeric(df['variable_value'], errors='coerce')
            df = df.dropna()
            df_features= df.groupby(["ID", 'variable_name', 't'], as_index=False)['variable_value'].mean()
            logging.info("Pivoting...")
            df_features = pd.pivot_table(df_features, index=['ID', 't'], columns='variable_name', dropna=False)
            df_features = df_features.reset_index()
            df_features.columns = [process_cols(col, ["ID", "t"]) for col in df_features.columns]
            del df
            gc.collect()
            intermedate_path = FEATURE_PATH / f"{data_type}_{BASE_SUFFIX}.feather"
            logging.info(f"Writing out intermedate base file to {intermedate_path}")
            df_features = downcast_dataframe(df_features)
            df_features.to_feather(intermedate_path)
            logging.info("Forward Filling")
            df_features.update(df_features.groupby("ID").ffill())
            final_path = FEATURE_PATH / f"{data_type}_ffill.feather"
            logging.info(f"Writing out ffill features to {final_path}")
            df_features.to_feather(final_path)
            del df_features
            gc.collect()

