#!/usr/bin/env python3
"""
clean_csvs.py
Load parser outputs from clean_output/ and produce deduplicated, normalized cleaned_* CSVs
"""
import pandas as pd, re, os
from fuzzywuzzy import process

IN_DIR = "clean_output"
OUT_PREFIX = "cleaned"

def normalize_disease(name):
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9\s\-]", " ", name)
    name = re.sub(r"\s{2,}", " ", name)
    return name.strip()

def is_noise(name):
    if not name or len(name) < 4: return True
    if re.fullmatch(r"[a-z]\d+|\d+[a-z]*", name): return True
    return False

drugs = pd.read_csv(os.path.join(IN_DIR, "drugs.csv"))
proteins = pd.read_csv(os.path.join(IN_DIR, "proteins.csv"))
edges_dt = pd.read_csv(os.path.join(IN_DIR, "edges_drug_targets.csv"))
edges_dd = pd.read_csv(os.path.join(IN_DIR, "edges_drug_treats_disease.csv"))
diseases = pd.read_csv(os.path.join(IN_DIR, "diseases.csv"))

drugs = drugs.drop_duplicates(subset=["id"]).reset_index(drop=True)
proteins = proteins.drop_duplicates(subset=["id"]).reset_index(drop=True)
edges_dt = edges_dt.drop_duplicates().reset_index(drop=True)
edges_dd['disease_name'] = edges_dd['disease_name'].astype(str).map(normalize_disease)
edges_dd = edges_dd.drop_duplicates().reset_index(drop=True)
diseases['name'] = diseases['name'].astype(str).map(normalize_disease)
diseases = diseases[~diseases['name'].map(is_noise)].drop_duplicates().reset_index(drop=True)

# Save cleaned
drugs.to_csv(f"data_cleaning/{OUT_PREFIX}_drugs.csv", index=False)
proteins.to_csv(f"data_cleaning/{OUT_PREFIX}_proteins.csv", index=False)
diseases.to_csv(f"data_cleaning/{OUT_PREFIX}_diseases.csv", index=False)
edges_dt.to_csv(f"data_cleaning/{OUT_PREFIX}_edges_drug_targets.csv", index=False)
edges_dd.to_csv(f"data_cleaning/{OUT_PREFIX}_edges_drug_treats_disease.csv", index=False)

print("Cleaned files written: cleaned_*.csv")
