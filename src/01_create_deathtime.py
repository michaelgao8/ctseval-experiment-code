import pandas as pd 
from datetime import datetime, timedelta
import yaml
import pathlib

config = yaml.safe_load(open('../config.yaml'))
data_path = config['data_path']
mimic3_path = config['mimic3_path']
pathlib.Path(data_path, 'population').mkdir(parents=True, exist_ok=True)

# Read in data
patients = pd.read_csv(mimic3_path + 'PATIENTS.csv', parse_dates=['DOB', 'DOD'], usecols=['SUBJECT_ID', 'DOB', 'DOD'])
admissions = pd.read_csv(mimic3_path + 'ADMISSIONS.csv', parse_dates=['DEATHTIME'], usecols=['SUBJECT_ID', 'HADM_ID', 'DEATHTIME', 'HOSPITAL_EXPIRE_FLAG'])
examples = pd.read_csv(data_path + 'prep/icustays_MV.csv', parse_dates=['INTIME', 'OUTTIME']).sort_values(by='ICUSTAY_ID') # Only Metavision

examples = pd.merge(examples, patients, on='SUBJECT_ID', how='left')
examples = pd.merge(examples, admissions, on=['SUBJECT_ID', 'HADM_ID'], how='left')
examples['AGE'] = examples.apply(lambda x: (x['INTIME'].to_pydatetime() - x['DOB'].to_pydatetime()).total_seconds(), axis=1) / 3600 / 24 / 365.25
examples['HOUR_OF_DEATH'] = (examples['DEATHTIME'] - examples['INTIME']).dt.total_seconds()/60/60
examples['DOD_within_time'] = 0

# Remove patients that have no death time, but there is a DOD listed that is within the admisision and 48 hours of discharge and no associated DEATHTIME
(examples.loc[
    ((examples['DOD'] >= examples['INTIME']) &
    (examples['DOD'] <= examples['OUTTIME'] + timedelta(hours=48)))
    &
    (examples['DEATHTIME'].isnull())
    ,
    "DOD_within_time"]) = 1

examples = examples.loc[examples['DOD_within_time'] == 0].copy()

examples = examples.loc[:, [
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "INTIME",
    "OUTTIME",
    "partition",
    "DOB", 
    "DEATHTIME", 
    "HOSPITAL_EXPIRE_FLAG",
    "AGE",
    "HOUR_OF_DEATH"
]].copy()

examples.to_csv(f"{data_path}/labels/deathtime.csv", index=False)