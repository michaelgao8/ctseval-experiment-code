from pathlib import Path

RAW_PARQUET_ROOT = Path("../cohere/data/raw/parquet")

# Specify Raw Paths
ANALYTE_RAW_PARQUET = RAW_PARQUET_ROOT / "analyte"
FLOWSHEET_RAW_PARQUET = RAW_PARQUET_ROOT / "flowsheet"
MED_ADMIN_RAW_PARQUET = RAW_PARQUET_ROOT / "med_admin"
ORDER_RAW_PARQUET = RAW_PARQUET_ROOT / "order"
NOTES_RAW_PARQUET = RAW_PARQUET_ROOT / "note"
IP_ENCOUNTER_RAW_PARQUET = RAW_PARQUET_ROOT / "ip_encounter"
DIAGNOSIS_RAW_PARQUET = RAW_PARQUET_ROOT / "diagnosis"
PROCEDURE_RAW_PARQUET = RAW_PARQUET_ROOT / "procedure"
DEMOGRAPHIC_RAW_PATH = RAW_PARQUET_ROOT / "demographic.feather"
PATIENT_RAW_PARQUET = RAW_PARQUET_ROOT / "patient"
VISIT_REASON_PARQUET = RAW_PARQUET_ROOT / "visit_reason"

FEATURE_ROOT = Path("../cohere/data/feature_store/parquet/")

# Generic Feature Store Paths
ADMISSION_FEATURES_PATH = FEATURE_ROOT / "admission"
DIAGNOSIS_FEATURES_PATH_1yr =FEATURE_ROOT/ "comorbidities_1yr"
PRIOR_PROCEDURE_PATH_1yr = FEATURE_ROOT / "procedure_1yr"
CURRENT_PROCEDURE_PATH_1yr = FEATURE_ROOT / "procedure_current"
DEMOGRAPHICS_FEATURE_PATH = FEATURE_ROOT / "demographic"
AGE_AT_ADMISSION_PATH = FEATURE_ROOT / "age_at_admission"
VISIT_REASON_FEATURE_PATH = FEATURE_ROOT / "visit_reason"

# Paths for configurations
FEATURES_TO_INCLUDE_PATH = "./src/config/features.yaml"
MODEL_MATRIX_CONFIG_PATH = "./src/config/model_matrix_config.yaml"