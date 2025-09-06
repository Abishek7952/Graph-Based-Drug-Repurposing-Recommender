#!/usr/bin/env python3
"""
quick_fix_diseases.py
- Load final_diseases.csv and final_edges_drug_treats_disease.csv
- Apply a small stoplist, normalization, targeted manual corrections,
  conservative fuzzy corrections for rare variants mapped to frequent canonical names.
- Overwrites:
    final_diseases.csv
    final_edges_drug_treats_disease.csv
- Conservative: keeps names unless confident to map.
"""
import pandas as pd
import re
from fuzzywuzzy import process

# --------- Config ----------
DISEASE_CSV = "data_cleaning/final_diseases.csv"
EDGES_DD = "data_cleaning/final_edges_drug_treats_disease.csv"
MIN_LEN = 4
FUZZY_THRESHOLD = 92   # conservative
STOPLIST = {
    "fatty acid", "pneumoniae type 7f capsular polysaccharide antigen",
    "nephro-", "polysaccharide antigen", "fatty acids"
}
MANUAL_CORRECTIONS = {
    "psorasis": "psoriasis",
    "osteoperosis": "osteoporosis",
    "alzheimer s": "alzheimer",
    "alzheimer s disease": "alzheimer",
    "nsclc": "non-small-cell lung cancer",
    "ctcl": "cutaneous t cell lymphoma",
    # add any project-specific mappings here
}

# --------- Helpers ----------
def normalize(s):
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-\']", " ", s)   # allow hyphen/apostrophe
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def is_noise(x):
    if not x or len(x) < MIN_LEN:
        return True
    if x in STOPLIST:
        return True
    # short alphanumeric tokens with digits (e.g., a244710)
    if re.fullmatch(r"[a-z]\d+|\d+[a-z]*", x):
        return True
    if x.endswith("-") or x.startswith("-"):
        return True
    return False

# --------- Load ----------
print("Loading files...")
df_d = pd.read_csv(DISEASE_CSV)
df_e = pd.read_csv(EDGES_DD)

# --------- Normalize ----------
df_d['name'] = df_d['name'].astype(str).map(normalize)
df_e['disease_name'] = df_e['disease_name'].astype(str).map(normalize)

# --------- Manual corrections ----------
df_d['name'] = df_d['name'].map(lambda x: MANUAL_CORRECTIONS.get(x, x))
df_e['disease_name'] = df_e['disease_name'].map(lambda x: MANUAL_CORRECTIONS.get(x, x))

# --------- Remove obvious noise from disease list ----------
df_d = df_d[~df_d['name'].map(is_noise)].drop_duplicates().reset_index(drop=True)

# --------- Conservative fuzzy merge: map rare names onto frequent canonicals ----------
# Build candidate canonicals from high-frequency disease names in edges
freq = df_e['disease_name'].value_counts()
canon = list(freq.nlargest(2000).index)  # top frequent names as canonical base
canon_set = set(canon)

unique_names = sorted(set(df_e['disease_name'].unique()))
mapping = {}
for name in unique_names:
    if name in canon_set:
        mapping[name] = name
        continue
    match = process.extractOne(name, canon, scorer=process.fuzz.token_sort_ratio)
    if match and match[1] >= FUZZY_THRESHOLD:
        mapping[name] = match[0]
    else:
        mapping[name] = name

df_e['disease_name'] = df_e['disease_name'].map(lambda x: mapping.get(x, x))

# --------- Final dedupe and ensure disease list matches edges ----------
df_e = df_e.drop_duplicates(subset=['drug_id','disease_name']).reset_index(drop=True)
final_names = sorted(df_e['disease_name'].unique())
df_d_final = pd.DataFrame(final_names, columns=['name'])

# --------- Save (overwrite final files in place) ----------
df_d_final.to_csv(DISEASE_CSV, index=False)
df_e.to_csv(EDGES_DD, index=False)

print("Quick-fix complete.")
print("Diseases count:", len(df_d_final))
print("Edges (drug->disease) count:", len(df_e))
