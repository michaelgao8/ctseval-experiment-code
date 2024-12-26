import pandas as pd 
import pickle

# set constants
STACKED_DATA_PATH = "../data/mimic/processed/formatted/all_data.stacked.p"
PREVALENCE_THRESHOLD = 0.05
OUTPUT_STACKED_DATA_PATH = "../data/mimic/processed/formatted/filtered_data.stacked.p"


with open(STACKED_DATA_PATH, "rb") as f:
    stacked_data = pickle.load(f)

def calculate_prevalence(var_name, domain):
    total_n_ids = stacked_data[domain]['ID'].nunique()
    temp_n = stacked_data[domain].loc[stacked_data[domain]['variable_name'] == var_name, 'ID'].nunique()
    return temp_n/total_n_ids

domains_to_filter = [
                    'LABEVENTS',
                    'MICROBIOLOGYEVENTS',
                    'DATETIMEEVENTS', 
                    'OUTPUTEVENTS',
                    'CHARTEVENTS', 
                    'INPUTEVENTS_MV',
                    'PROCEDUREEVENTS_MV'
                    ]
# Estimate the total number of icu stays
N_TOTAL_ENCS = stacked_data['LABEVENTS']['ID'].nunique()

filter_dict = {dom:[] for dom in domains_to_filter}
for dom in domains_to_filter:
    filter_dict[dom].append(( \
        stacked_data[dom] \
            .loc[:, ['ID', 'variable_name']] \
            .drop_duplicates()['variable_name'] \
            .value_counts()/N_TOTAL_ENCS) \
            .where(lambda x: x>=PREVALENCE_THRESHOLD) \
            .dropna().index)


# Process stacked data
# * Remove uncommon features
# * Get values only where the timestamp is past 1
# * Re-save the data for downstream processing
for dom in domains_to_filter:
    print(f"Processing {dom}")
    try:
        stacked_data[dom] = (stacked_data[dom]
                             .loc[((stacked_data[dom]['t'] >= 0) 
                                   & (stacked_data[dom]['variable_name'].isin(filter_dict[dom][0]))
                                  )])
    except:
        stacked_data[dom] = (stacked_data[dom]
                             .loc[((stacked_data[dom]['t_start'] >= 0) 
                                   & (stacked_data[dom]['variable_name'].isin(filter_dict[dom][0]))
                                  )])

with open(OUTPUT_STACKED_DATA_PATH, "wb") as f:
    pickle.dump(stacked_data, f)