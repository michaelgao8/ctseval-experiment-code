
from ray import tune, train
from ray.air import session, RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback
import lightgbm as lgb
import numpy as np
import pandas as pd 
import json
import yaml
from src.config import paths
import logging 
import sklearn
import argparse
from pathlib import Path
from functools import partial

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Train model with hyperparameter tuning")
parser.add_argument('outcome', type=str, help='the outcome to model')

args = parser.parse_args()

logging.info("Reading in configuration")

with open(paths.MODEL_MATRIX_CONFIG_PATH, "r") as f:
    model_matrix_config = yaml.load(f, yaml.CLoader)

outcomes_to_model = model_matrix_config['outcome_columns']
if args.outcome not in outcomes_to_model:
    raise
else:
    logging.info(f"Modeling {args.outcome}")
encounter_id_col = model_matrix_config['outcome_encounter_id_column']

def drop_nonoutcome_cols(outcome):
    cols = ['encounter_id'] + [col for col in outcomes_to_model if col != outcome]
    return ["".join (c if c.isalnum() else "_" for c in str(x)) for x in cols]

logging.info("Reading in data")
all_non_feature_cols = [encounter_id_col] + outcomes_to_model
train_data = pd.read_feather("./data/mimic/processed/modeling/train_df_input.feather")
test_data = pd.read_feather("./data/mimic/processed/modeling/test_df_input.feather")
valid_data = pd.read_feather("./data/mimic/processed/modeling/val_df_input.feather")
train_data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train_data.columns]
test_data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in test_data.columns]
valid_data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in valid_data.columns]
train_dataset = train_data.drop(columns=all_non_feature_cols)
dataset_cols = list(train_dataset.columns)
train_numpy = train_dataset.values
train_label_numpy = train_data[args.outcome].values
valid_numpy = valid_data.drop(columns=all_non_feature_cols).values
valid_label_numpy = valid_data[args.outcome].values


def train_model(config, train_numpy=None, 
                train_label_numpy=None, valid_numpy=None, 
                valid_label_numpy=None, test_data=None,
                dataset_cols=None):
    train_set = lgb.Dataset(train_numpy, label = train_label_numpy, feature_name=dataset_cols)
    valid_set = lgb.Dataset(valid_numpy, label = valid_label_numpy, feature_name=dataset_cols)
    

    def dummy_metric(preds, eval_dataset):
        metric_name = "dummy"
        higher_better=True
        return metric_name, 1, higher_better


    gbm = lgb.train(
        config,
        train_set,
        valid_sets=[valid_set],
        valid_names=["eval"],
        callbacks=[lgb.log_evaluation(period=0),
        TuneReportCheckpointCallback(
                {"binary_logloss": "eval-binary_logloss"}
        )
        ]
    )
    preds = gbm.predict(test_data.drop(columns=all_non_feature_cols))
    session.report({
        "binary_logloss": 1,
        "auc": sklearn.metrics.roc_auc_score(test_data[args.outcome], preds),
        "aucpr": sklearn.metrics.average_precision_score(test_data[args.outcome], preds),
        "done": True,
        "dummy_metric": 1
    })
    return {"test_auc": sklearn.metrics.roc_auc_score(test_data[args.outcome], preds),
            "test_aucpr": sklearn.metrics.average_precision_score(test_data[args.outcome], preds),
            "dummy_metric": 1}

if __name__ == '__main__':

    config = {
        "objective": "binary",
        "metric": ["binary_error", "binary_logloss", "auc", "average_precision"],
        "verbose": -1,
        "num_iteration": tune.randint(50, 1000),
        "boosting_type": tune.grid_search(["gbdt", "dart"]),
        "num_leaves": tune.randint(10, 1000),
        "learning_rate": tune.loguniform(1e-8, 1e-1),
        "min_data_in_leaf": tune.randint(10, 10000),
        "max_depth": tune.grid_search([-1, 2, 4, 6, 8, 10, 12, 14]),
        "feature_fraction": tune.uniform(0.01, 1)
    }
    
    tuner = tune.Tuner(
        tune.with_parameters(train_model, 
                                train_numpy=train_numpy, 
                                train_label_numpy=train_label_numpy, 
                                valid_numpy=valid_numpy, 
                                valid_label_numpy=valid_label_numpy, 
                                test_data=test_data,
                                dataset_cols=dataset_cols),
        tune_config=tune.TuneConfig(
            metric="binary_logloss",
            mode="min",
            #scheduler=ASHAScheduler(),
            num_samples=50
            #,time_budget_s=20
        ),
        param_space=config,
        run_config=RunConfig(local_dir="./models", name=f"{args.outcome}_2"))
    
    results = tuner.fit()
    print(dir(results.get_best_result()))
    print(type(results.get_best_result()))

    print("Best hyperparameters found were: ", results.get_best_result().config)
