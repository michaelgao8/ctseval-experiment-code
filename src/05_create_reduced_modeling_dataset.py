import pandas as pd 
import logging
from pathlib import Path
import math
import numpy as np

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

FEATURE_PATH = Path("../data/mimic/processed/features/") 
OUTCOME_PATH = Path("../data/mimic/processed/labels/deathtime.csv")

logging.info("Reading in and creating outcome")
outcome_df = pd.read_csv(OUTCOME_PATH)
outcome_df = outcome_df.loc[(outcome_df['HOUR_OF_DEATH'] > 0) | (outcome_df['HOUR_OF_DEATH'].isnull())].copy()

outcome_df['INTIME'] = pd.to_datetime(outcome_df['INTIME'])
outcome_df['OUTTIME'] = pd.to_datetime(outcome_df['OUTTIME'])
outcome_df['DEATHTIME'] = pd.to_datetime(outcome_df['DEATHTIME'])
outcome_df['los'] = (outcome_df['OUTTIME'] - outcome_df['INTIME']).dt.total_seconds()/60/60
outcome_df['cutoff_time'] = outcome_df[['HOUR_OF_DEATH', 'los']].apply(np.nanmin, axis=1)

logging.info("Create the skeleton")
outcome_df['cutoff_time'] = outcome_df['cutoff_time'].apply(math.floor)
skeleton_df = outcome_df.loc[:, ['ICUSTAY_ID', 'cutoff_time']].reindex(outcome_df.index.repeat(outcome_df['cutoff_time']))
skeleton_df['hour'] = skeleton_df.groupby('ICUSTAY_ID').cumcount() + 1

skeleton_df = skeleton_df.rename({'hour': 't', 'ICUSTAY_ID': 'ID'}, axis=1)


logging.info("Adding in features")

FEAT_TO_ADD = [ 
    "static_features",
    "CHARTEVENTS_window",
    "DATETIMEEVENTS_window",
    "LABEVENTS_window",
    "MICROBIOLOGYEVENTS_window",
    "OUTPUTEVENTS_window",
]

for feat in FEAT_TO_ADD:
    logging.info(f"Adding {feat}")
    feat_df = pd.read_feather(FEATURE_PATH / f"{feat}.feather")
    print(feat_df.columns)
    if 'static' in feat:
        skeleton_df = pd.merge(skeleton_df, feat_df, how='left', left_on='ID', right_on='ID')
        skeleton_df.info()
        
    else:
        skeleton_df = pd.merge(skeleton_df, feat_df, how='left', left_on=['ID', 't'], right_on=['ID', 't'])
        skeleton_df.info()

skeleton_df = skeleton_df.drop(['cutoff_time'], axis=1)
skeleton_df = skeleton_df.sort_values(['ID', 't'])

outcome_df.to_csv("../data/mimic/processed/modeling/outcome.csv", index=False)
skeleton_df.to_feather("../data/mimic/processed/modeling/model_matrix_no_inputevents.feather")