from ray.tune import Tuner
import argparse
from pathlib import Path
import logging
import yaml
from src.config import paths
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

parser = argparse.ArgumentParser(description="Get the results of runs")
parser.add_argument('outcome', type=str)
parser.add_argument('resultpath', type=str)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

with open(paths.MODEL_MATRIX_CONFIG_PATH, "r") as f:
    model_matrix_config = yaml.load(f, yaml.CLoader)

args = parser.parse_args()

outcomes_to_model = model_matrix_config['outcome_columns']
if args.outcome not in outcomes_to_model:
    raise
else:
    logging.info(f"Modeling {args.outcome}")
encounter_id_col = model_matrix_config['outcome_encounter_id_column']


logging.info(f"Restoring experiment from ./models/{args.outcome}")
tuner = Tuner.restore(f"./models/{args.outcome}")
results = tuner.get_results()
print(results.get_best_result())

best_result = results.get_best_result()

booster = lgb.Booster(model_file=Path(best_result.checkpoint._local_path) / "checkpoint")

all_non_feature_cols = [encounter_id_col] + outcomes_to_model

# Read in the test data
test_data = pd.read_feather("./data/modeling/test.feather")
test_data.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in test_data.columns]

predictions = booster.predict(test_data.drop(columns=all_non_feature_cols))

logging.info(f"Test AUROC: {roc_auc_score(test_data[args.outcome], predictions)}")
logging.info(f"Test AUPRC: {average_precision_score(test_data[args.outcome], predictions)}")

results_dir = Path(args.resultpath) / args.outcome
results_dir.mkdir(parents=True, exist_ok=True)

logging.info("Reassmembling dataset")
test_data['predictions'] = predictions
test_data.loc[:, [encounter_id_col, args.outcome, 'predictions']].to_csv(results_dir / "prediction_df.csv", index=False)

logging.info("Saving booster file")
booster.save_model(results_dir / "booster.model")
