from ast import literal_eval
import json

with open('data/processed/feature_map.json', 'r') as f:
    feature_map = json.load(f)

def format_table(df, task: str, supervised: bool =False, baseline: bool =False):
    # Extract the diagnostics labels
    diagnostics_raw_list = df[task].deepcopy().tolist()
    diagnostics_labels_list = list(map(int, diagnostics_raw_list))

    # Extract the laboratory groups ids
    group_idx_list = df["lab_group_idx"].deepcopy().tolist()
    group_label_list = list(map(lambda x: literal_eval(x), group_idx_list))

    if supervised:
        if baseline:
            header = list(feature_map.values())
            table_data = df[header].deepcopy().values.tolist()
        else:
            header = []
            table_data = []

            for index, row in df.iterrows():