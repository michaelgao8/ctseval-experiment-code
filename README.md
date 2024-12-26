# Data preprocessing and Experiment Code

## Data PreprocessingSteps
1. All data is preprocessed using FIDDLE and put into the all_data.stacked.p file.

This is done by following the instructions in this [FIDDLE repository](https://github.com/MLD3/FIDDLE-experiments/tree/master/mimic3_experiments), in the `1_data_extraction` directory.
This should create a file called `all_data.stacked.p` which contains the raw data for downstream processing. For the purposes of this code, we will assume that this file is in the `data/mimic/processed/` directory (default for FIDDLE).

2. Run the `01_create_deathtime.py` script to create the `deathtime.csv` file. This file contains the death time for each patient.

3. Run the `02_create_dense.py` script to create the `filtered_data.stacked.p` file. This file contains the data for downstream processing.

4. Run the `03_create_features.py` script to create the `static_features.feather` file and the `dynamic_features` directory.

5. Run the `04_create_window_features.py` script to create intermediate files for window features

6. Run the `05_create_reduced_modeling_dataset.py` script to create the `outcome.csv` file and the `model_matrix_no_inputevents.feather` file.

7. Run the `06_create_train_test.py` script to create the `train_df_noinput.feather`, `val_df_noinput.feather`, and `test_df_noinput.feather` files. These files contain the data for the train, validation, and test sets.

## Modeling

In order to train models and get results, run the `train_models.py` script. You may need to modify the config/model_matrix_config.yaml file to include the outcome you want to model.

the `get_results.py` script will get the results of the best model for a given outcome and save the results in the `results` directory.

All of these script should be run from the top level directory using

`python -m src.modeling.train_models --outcome <outcome> --resultpath <resultpath>`

Be sure to create any results directories that are needed prior to running scripts. 