import pandas as pd 
import logging
from pathlib import Path
import math
import json
import numpy as np

np.random.seed(1234)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)


logging.info("Read in the outcome file")
outcome_df = pd.read_csv("../data/mimic/processed/modeling/outcome.csv")
outcome_df = outcome_df.rename({'ICUSTAY_ID': "ID"}, axis=1)

logging.info("Read in the model matrix files")
model_matrix_no_input = pd.read_feather("../data/mimic/processed/modeling/model_matrix_no_inputevents.feather")
logging.info("Create the outcome of interest")
MORTALITY_LOOKAHEAD_HOURS = 24

logging.info("Add in the cutoff hour and the flag")
model_matrix_no_input = (pd.merge(model_matrix_no_input, 
        outcome_df.loc[:, ['ID', 'HOUR_OF_DEATH']].drop_duplicates(),
        how='left',
        on='ID'))


logging.info("Create the label")

model_matrix_no_input['24hour_mortality'] = 0
model_matrix_no_input.loc[((model_matrix_no_input['HOUR_OF_DEATH'] - model_matrix_no_input['t']) <= MORTALITY_LOOKAHEAD_HOURS)
                         &(model_matrix_no_input['t'] < model_matrix_no_input['HOUR_OF_DEATH']), '24hour_mortality'] = 1

all_ids = model_matrix_no_input['ID'].unique()
TRAIN_PROP = 0.70
VAL_PROP = 0.15
TEST_PROP = 0.15

shuffled_ids = np.random.permutation(all_ids)

train_ids = shuffled_ids[:int(TRAIN_PROP * len(shuffled_ids))]
val_ids = shuffled_ids[int(TRAIN_PROP * len(shuffled_ids)):int(TRAIN_PROP * len(shuffled_ids)) + int(VAL_PROP * len(shuffled_ids))]
test_ids = np.setdiff1d(all_ids, np.concatenate([train_ids, val_ids]))

outcome_df = pd.read_csv("../data/mimic/processed/modeling/outcome.csv")
with open("../data/mimic/processed/modeling/ids.json", "w") as f:
        id_dict = {'train': train_ids.tolist(), 'val_ids': val_ids.tolist(), 'test_ids': test_ids.tolist()}
        json.dump(id_dict, f, indent=2)


train_df_no_input = model_matrix_no_input.loc[model_matrix_no_input['ID'].isin(train_ids)].drop(['HOUR_OF_DEATH'], axis=1)
val_df_no_input = model_matrix_no_input.loc[model_matrix_no_input['ID'].isin(val_ids)].drop(['HOUR_OF_DEATH'], axis=1)
test_df_no_input = model_matrix_no_input.loc[model_matrix_no_input['ID'].isin(test_ids)].drop(['HOUR_OF_DEATH'], axis=1)

train_df_no_input.reset_index(drop=True).to_feather("../data/mimic/processed/modeling/train_df_noinput.feather")
val_df_no_input.reset_index(drop=True).to_feather("../data/mimic/processed/modeling/val_df_noinput.feather")
test_df_no_input.reset_index(drop=True).to_feather("../data/mimic/processed/modeling/test_df_noinput.feather")
