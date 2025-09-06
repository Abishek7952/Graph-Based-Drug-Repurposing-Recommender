#!/usr/bin/env python3
"""
diagnose_and_finalize.py
Load cleaned_* CSVs, dedupe, conservative fuzzy merge, write final_* CSVs + final_diagnostics.json
"""
import os, json, re
from collections import Counter
import pandas as pd
from fuzzywuzzy import process, fuzz

CLEAN_PREFIX = "cleaned"
FINAL_PREFIX = "final"
FUZZY_THRESHOLD = 92
MIN_DISEASE_LEN = 4

def normalize(s):
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# Load cleaned files
drugs = pd.read_csv(f"data_cleaning/{CLEAN_PREFIX}_drugs.csv").drop_duplicates(subset=["id"]).reset_index(drop=True)
proteins = pd.read_csv(f"data_cleaning/{CLEAN_PREFIX}_proteins.csv").drop_duplicates(subset=["id"]).reset_index(drop=True)
edges_dt = pd.read_csv(f"data_cleaning/{CLEAN_PREFIX}_edges_drug_targets.csv").drop_duplicates().reset_index(drop=True)
edges_dd = pd.read_csv(f"data_cleaning/{CLEAN_PREFIX}_edges_drug_treats_disease.csv")
diseases = pd.read_csv(f"data_cleaning/{CLEAN_PREFIX}_diseases.csv")

# normalize disease strings
edges_dd['disease_name'] = edges_dd['disease_name'].astype(str).map(normalize)
diseases['name'] = diseases['name'].astype(str).map(normalize)

# build canonical set from most frequent names
freq = edges_dd['disease_name'].value_counts()
canon_list = list(freq.nlargest(1500).index) + list(diseases['name'].unique())
canon_list = sorted(list(dict.fromkeys(canon_list)))

# fuzzy merge conservative
unique_names = sorted(set(edges_dd['disease_name'].dropna().tolist()))
mapping = {}
for name in unique_names:
    if name in canon_list:
        mapping[name]=name; continue
    best = process.extractOne(name, canon_list, scorer=fuzz.token_sort_ratio)
    if best and best[1] >= FUZZY_THRESHOLD:
        mapping[name]=best[0]
    else:
        mapping[name]=name
edges_dd['disease_name'] = edges_dd['disease_name'].map(lambda x: mapping.get(x,x))

# filter diseases with tiny length or numeric tokens
def is_noise(n):
    if not n or len(n) < MIN_DISEASE_LEN: return True
    if re.fullmatch(r"[a-z]\d+|\d+[a-z]*", n): return True
    return False

final_disease_names = sorted(set([n for n in edges_dd['disease_name'].unique() if not is_noise(n)]))
final_diseases = pd.DataFrame(final_disease_names, columns=['name'])

# dedupe edges
edges_dt = edges_dt.drop_duplicates(subset=['drug_id','target_id']).reset_index(drop=True)
edges_dd = edges_dd.drop_duplicates(subset=['drug_id','disease_name']).reset_index(drop=True)

# compute diagnostics
diagnostics = {
    "final_counts": {
        "drugs": int(drugs.shape[0]),
        "proteins": int(proteins.shape[0]),
        "edges_drug_targets": int(edges_dt.shape[0]),
        "edges_drug_diseases": int(edges_dd.shape[0]),
        "diseases": int(final_diseases.shape[0])
    },
    "top_20_diseases": Counter(edges_dd['disease_name'].tolist()).most_common(20)
}

# save final files
drugs.to_csv(f"data_cleaning/{FINAL_PREFIX}_drugs.csv", index=False)
proteins.to_csv(f"data_cleaning/{FINAL_PREFIX}_proteins.csv", index=False)
edges_dt.to_csv(f"data_cleaning/{FINAL_PREFIX}_edges_drug_targets.csv", index=False)
edges_dd.to_csv(f"data_cleaning/{FINAL_PREFIX}_edges_drug_treats_disease.csv", index=False)
final_diseases.to_csv(f"data_cleaning/{FINAL_PREFIX}_diseases.csv", index=False)
with open("final_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)

print("Finalization complete. See final_diagnostics.json")
