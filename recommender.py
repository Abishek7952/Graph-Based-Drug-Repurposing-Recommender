#!/usr/bin/env python3
"""
recommender.py — improved, robust bias-correcting recommender

Improvements vs earlier version:
 - Robust mappings & name loading (maps id <-> index correctly)
 - Existing-edges exclusion tolerant to id/index formats
 - Safer predictor state loading with flexible key mapping (tries several common patterns)
 - Uses torch.no_grad wherever appropriate
 - Vectorized disease-bias computation for dot-product/cosine fallback (much faster)
 - Better printing / saving and small diagnostics
 - Hard-coded TOP_K as requested (TOP_K variable)
"""

import os
import time
import heapq
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# optional pretty print
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except Exception:
    HAS_TABULATE = False

# ---------------- Config (edit if you want) ----------------
TOP_K = 20                 # global top-K (hard-coded)
TOP_PER_DRUG = 5           # per-drug top-N
MODEL_DIR = "model_out"
MAPPINGS_PATHS = [os.path.join(MODEL_DIR, "mappings.pkl"), "mappings.pkl"]
DRUG_EMB_PATHS = [os.path.join(MODEL_DIR, "embeddings_drug.pt"), "embeddings_drug.pt"]
DIS_EMB_PATHS = [os.path.join(MODEL_DIR, "embeddings_disease.pt"), "embeddings_disease.pt"]
MODEL_PATHS = [os.path.join(MODEL_DIR, "model_full.pt"),
               os.path.join(MODEL_DIR, "trained_model.pt"),
               "model_full.pt", "trained_model.pt"]

TRAIN_EDGES_CSV = os.path.join("data_cleaning", "final_edges_drug_treats_disease.csv")
EXISTING_EDGES_CSV = os.path.join("data_cleaning", "final_edges_drug_treats_disease.csv")  # reuse cleaned edges by default

DRUGS_CSV_CANDIDATES = ["final_drugs.csv", os.path.join("data_cleaning", "final_drugs.csv")]
DISEASES_CSV_CANDIDATES = ["final_diseases.csv", os.path.join("data_cleaning", "final_diseases.csv")]

MIN_PROB = 0.50   # filtering threshold for per-drug adjusted results output
EXCLUDE_PATTERNS = ['salt', 'hydrate', 'acid', 'monohydrate', 'sodium']  # sample blacklist

# ---------------- Predictor (attempt to match your training classifier) ----------------
class Predictor(nn.Module):
    def __init__(self, embedding_dim=128, hidden=128, dropout=0.6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, z_src, z_dst):
        # expects z_src, z_dst shapes (batch, emb_dim)
        x = torch.cat([z_src, z_dst], dim=-1)
        return self.net(x).view(-1)

# ---------------- Helpers ----------------
def find_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def try_load_names():
    drug_names = {}
    disease_names = {}
    for p in DRUGS_CSV_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str).fillna("")
                # common columns: 'id','name' or 'drug_id','drug_name'
                if 'id' in df.columns and 'name' in df.columns:
                    drug_names = df.set_index('id')['name'].to_dict()
                elif 'drug_id' in df.columns and 'drug_name' in df.columns:
                    drug_names = df.set_index('drug_id')['drug_name'].to_dict()
                elif df.shape[1] >= 2:
                    drug_names = df.set_index(df.columns[0])[df.columns[1]].to_dict()
                else:
                    # fallback map each value to itself
                    drug_names = {str(r): str(r) for r in df[df.columns[0]].tolist()}
                print(f"Loaded drug names from {p} ({len(drug_names)} rows)")
                break
            except Exception as e:
                print("Warning loading drug names:", e)
    for p in DISEASES_CSV_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype=str).fillna("")
                if 'id' in df.columns and 'name' in df.columns:
                    disease_names = df.set_index('id')['name'].to_dict()
                elif 'disease_id' in df.columns and 'disease_name' in df.columns:
                    disease_names = df.set_index('disease_id')['disease_name'].to_dict()
                elif 'name' in df.columns:
                    disease_names = {row['name']: row['name'] for _, row in df.iterrows()}
                elif df.shape[1] >= 2:
                    disease_names = df.set_index(df.columns[0])[df.columns[1]].to_dict()
                else:
                    disease_names = {str(r): str(r) for r in df[df.columns[0]].tolist()}
                print(f"Loaded disease names from {p} ({len(disease_names)} rows)")
                break
            except Exception as e:
                print("Warning loading disease names:", e)
    return drug_names, disease_names

def pretty_print(df, maxrows=20):
    if HAS_TABULATE:
        print(tabulate(df.head(maxrows), headers=df.columns, tablefmt="psql", showindex=False))
    else:
        with pd.option_context('display.max_rows', maxrows, 'display.max_columns', None, 'display.width', 200):
            print(df.head(maxrows).to_string(index=False))

def is_excluded_name(n):
    if not isinstance(n, str):
        return False
    n = n.lower()
    return any(p in n for p in EXCLUDE_PATTERNS)

def safe_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

# Flexible mapping loader: try to load classifier/predictor weights into Predictor
def try_load_predictor_from_state(state, emb_dim):
    # state may be a dict (state_dict) or a checkpoint containing 'model_state_dict'
    sd = None
    if state is None:
        return None
    if isinstance(state, dict) and 'model_state_dict' in state:
        sd = state['model_state_dict']
    elif isinstance(state, dict) and any(isinstance(v, (np.ndarray, torch.Tensor)) for v in state.values()):
        sd = state
    else:
        sd = state if isinstance(state, dict) else None

    if sd is None:
        return None

    pred = Predictor(embedding_dim=emb_dim, hidden=emb_dim, dropout=0.6)
    # attempt strict load
    try:
        pred.load_state_dict(sd, strict=False)
        return pred
    except Exception:
        # try different key remapping heuristics
        remapped = {}
        for k, v in sd.items():
            newk = k
            # common names: classifier.net..., predictor.net..., model.net..., module.net...
            newk = newk.replace('classifier.', 'net.')
            newk = newk.replace('predictor.', 'net.')
            newk = newk.replace('model.', 'net.')
            newk = newk.replace('module.', '')
            remapped[newk] = v
        try:
            pred.load_state_dict(remapped, strict=False)
            return pred
        except Exception:
            # last resort: try to map any 2*emb_dim -> hidden, hidden->1 style keys automatically (dangerous)
            return None

# ---------------- Main ----------------
def main():
    print("=== Recommender (robust) starting ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # load mappings
    mpath = find_first(MAPPINGS_PATHS)
    if mpath is None:
        raise FileNotFoundError("mappings.pkl not found. Run training to produce mappings.pkl")
    with open(mpath, "rb") as f:
        maps = pickle.load(f)
    if 'drug' not in maps or 'disease' not in maps:
        raise KeyError("mappings.pkl must contain 'drug' and 'disease' keys (maps from id -> index).")
    drug_map = maps['drug']      # dict drug_id -> index
    disease_map = maps['disease']  # dict disease_id -> index
    inv_drug = {v:k for k,v in drug_map.items()}   # index -> drug_id
    inv_disease = {v:k for k,v in disease_map.items()} # index -> disease_id
    num_drugs = len(drug_map)
    num_diseases = len(disease_map)
    print("Num nodes:", {'drug': num_drugs, 'disease': num_diseases})

    # friendly names
    drug_names, disease_names = try_load_names()

    # load embeddings
    demb_p = find_first(DRUG_EMB_PATHS)
    disemb_p = find_first(DIS_EMB_PATHS)
    if demb_p is None or disemb_p is None:
        raise FileNotFoundError("Embeddings not found. Run training first (looked in model_out/ and project root).")
    print("Loading embeddings...")
    drug_embeddings = torch.load(demb_p, map_location="cpu")
    disease_embeddings = torch.load(disemb_p, map_location="cpu")
    emb_dr = safe_numpy(drug_embeddings)
    emb_di = safe_numpy(disease_embeddings)
    assert emb_dr.ndim == 2 and emb_di.ndim == 2
    emb_dim = emb_dr.shape[1]
    print("Emb shapes:", emb_dr.shape, emb_di.shape)

    # Optional: training label diagnostics
    if os.path.exists(TRAIN_EDGES_CSV):
        try:
            df_train = pd.read_csv(TRAIN_EDGES_CSV, dtype=str)
            if 'disease_name' in df_train.columns:
                print("\nTop 20 diseases by label count (training set):")
                print(df_train['disease_name'].value_counts().head(20).to_string())
        except Exception as e:
            print("Warning reading training edges:", e)

    # Try to load predictor
    predictor = None
    model_path = find_first(MODEL_PATHS)
    if model_path:
        print("Attempting to load predictor state from:", model_path)
        try:
            state = torch.load(model_path, map_location="cpu")
            pred = try_load_predictor_from_state(state, emb_dim)
            if pred is not None:
                pred.to(device).eval()
                predictor = pred
                print("Predictor loaded and ready (will use classifier for logits).")
            else:
                print("Couldn't map saved state to Predictor class; using dot-product / cosine fallback.")
        except Exception as e:
            print("Error loading predictor state, falling back to dot-product/cosine. Error:", e)
    else:
        print("No saved model state found; using dot-product/cosine fallback.")

    # Load existing edges to exclude from recommendations (support both id-based and index-based)
    existing_pairs = set()
    if os.path.exists(EXISTING_EDGES_CSV):
        try:
            df_exist = pd.read_csv(EXISTING_EDGES_CSV, dtype=str)
            # assume two cols: drug_id,disease_id OR drug_idx,disease_idx
            if df_exist.shape[1] >= 2:
                cols = df_exist.columns.tolist()
                for _, row in df_exist.iterrows():
                    a = str(row[cols[0]]).strip()
                    b = str(row[cols[1]]).strip()
                    # try to map to ids if they look numeric (index) or string id
                    if a in drug_map:
                        drug_id = a
                    elif a in inv_drug:
                        drug_id = inv_drug[int(a)]
                    else:
                        drug_id = a
                    if b in disease_map:
                        disease_id = b
                    elif b in inv_disease:
                        disease_id = inv_disease[int(b)]
                    else:
                        disease_id = b
                    existing_pairs.add((drug_id, disease_id))
            print("Loaded existing edges:", len(existing_pairs))
        except Exception as e:
            print("Warning loading existing edges:", e)

    # Prepare arrays for disease bias computation
    disease_sum = np.zeros(num_diseases, dtype=np.float64)
    disease_count = np.zeros(num_diseases, dtype=np.int64)

    # global heap for raw top-K (store probability and other fields)
    global_heap = []  # will store tuples (prob, drug_id, disease_id, raw_logit)

    # We will compute disease mean logit in a vectorized manner:
    start = time.time()

    if predictor is None:
        # Dot-product / cosine fallback. Vectorized disease mean: compute emb_dr @ emb_di.T in batches.
        # For stability using raw dot product as 'logit-like' measure, and cosine for probability mapping.
        emb_di_norm = emb_di / (np.linalg.norm(emb_di, axis=1, keepdims=True) + 1e-12)
        batch_size = 256
        drug_ids_list = list(drug_map.keys())
        disease_ids_list = list(disease_map.keys())
        for i in tqdm(range(0, num_drugs, batch_size), desc="drugs (bias pass)"):
            b_end = min(i + batch_size, num_drugs)
            # get embedding rows by map index ordering: we must use inv_drug mapping: index -> id
            idxs = list(range(i, b_end))
            dr_block = emb_dr[idxs, :]  # (B, D)
            # compute dot-products with all diseases (B, N)
            logits_block = dr_block @ emb_di.T
            disease_sum += logits_block.sum(axis=0)
            disease_count += logits_block.shape[0]
            # update global heap using probs (cosine)
            # compute cosine probs for this block
            dr_block_norm = dr_block / (np.linalg.norm(dr_block, axis=1, keepdims=True) + 1e-12)
            probs_block = (dr_block_norm @ emb_di_norm.T + 1.0) / 2.0
            # iterate rows to push top-K candidates (avoid pushing all NxM)
            for bi, idx in enumerate(idxs):
                drug_index = idx
                drug_id = inv_drug.get(drug_index, f"DRUG_{drug_index}")
                row_probs = probs_block[bi]
                row_logits = logits_block[bi]
                # to speed, pick top 200 candidates per drug to consider for global top-K
                pick_k = min(200, len(row_probs))
                top_idx = np.argpartition(-row_probs, pick_k - 1)[:pick_k]
                for j in top_idx:
                    disease_id = inv_disease.get(j, f"DIS_{j}")
                    if (drug_id, disease_id) in existing_pairs:
                        continue
                    p = float(row_probs[j])
                    raw = float(row_logits[j])
                    if len(global_heap) < TOP_K:
                        heapq.heappush(global_heap, (p, drug_id, disease_id, raw))
                    else:
                        heapq.heappushpop(global_heap, (p, drug_id, disease_id, raw))
    else:
        # Predictor exists: compute logits by batching drugs and predicting against all diseases (vectorized)
        emb_di_t = torch.from_numpy(emb_di).to(device)
        batch_size = 128
        for i in tqdm(range(0, num_drugs, batch_size), desc="drugs (bias pass)"):
            b_end = min(i + batch_size, num_drugs)
            idxs = list(range(i, b_end))
            dr_block = torch.from_numpy(emb_dr[idxs, :]).to(device)              # (B, D)
            # expand dr_block to compare to all diseases in one call: (B*N, D) approach is memory heavy.
            # Instead compute logits per drug in block iteratively but keep disease_sum accumulation vectorized per block:
            with torch.no_grad():
                # compute logits for each drug in block (vectorized by running predictor on (N, D) pairs per drug)
                # We'll compute per-drug in loop but disease_sum accumulation will be per-drug
                for bi, idx in enumerate(idxs):
                    z_dr = dr_block[bi:bi+1].expand(num_diseases, -1)  # (N, D)
                    logits_tensor = predictor(z_dr, emb_di_t)          # (N,)
                    logits = logits_tensor.cpu().numpy()
                    disease_sum += logits
                    disease_count += 1
                    probs = 1.0 / (1.0 + np.exp(-logits))
                    drug_id = inv_drug.get(idx, f"DRUG_{idx}")
                    # take top candidates to update global heap
                    pick_k = min(200, len(probs))
                    top_idx = np.argpartition(-probs, pick_k - 1)[:pick_k]
                    for j in top_idx:
                        disease_id = inv_disease.get(j, f"DIS_{j}")
                        if (drug_id, disease_id) in existing_pairs:
                            continue
                        p = float(probs[j])
                        raw = float(logits[j])
                        if len(global_heap) < TOP_K:
                            heapq.heappush(global_heap, (p, drug_id, disease_id, raw))
                        else:
                            heapq.heappushpop(global_heap, (p, drug_id, disease_id, raw))

    elapsed = time.time() - start
    print(f"Scored (approx) pairs and computed disease bias in {elapsed:.1f}s")

    # finalize disease mean
    disease_mean = disease_sum / np.maximum(disease_count, 1)
    # report top biased diseases
    top_bias_idx = np.argsort(-disease_mean)[:20]
    print("\nTop 20 diseases by mean raw logit (disease bias):")
    for idx in top_bias_idx:
        did = inv_disease.get(idx, f"DIS_{idx}")
        name = disease_names.get(did, "")
        print(f"  {did} ({name}) mean_logit={disease_mean[idx]:.6f}")

    # Build DataFrame from global heap (raw)
    top_sorted = sorted(global_heap, key=lambda x: x[0], reverse=True)
    rows = []
    for p, drug_id, disease_id, raw in top_sorted:
        rows.append({
            "drug_id": drug_id,
            "drug_name": drug_names.get(drug_id, ""),
            "disease_id": disease_id,
            "disease_name": disease_names.get(disease_id, ""),
            "probability_raw": float(p),
            "logit_raw": float(raw)
        })
    df_global_raw = pd.DataFrame(rows)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_global_raw = os.path.join(MODEL_DIR, f"top_{TOP_K}_global_raw.csv")
    df_global_raw.to_csv(out_global_raw, index=False)
    print("Saved global raw top-K to", out_global_raw)

    # Adjusted: subtract disease mean for each row (map disease_id -> index)
    # Build map disease_id -> mean
    disease_mean_map = {inv_disease[i]: float(disease_mean[i]) for i in range(len(disease_mean))}
    df_global_raw['mean_logit_disease'] = df_global_raw['disease_id'].map(lambda x: disease_mean_map.get(x, 0.0))
    df_global_raw['logit_adjusted'] = df_global_raw['logit_raw'] - df_global_raw['mean_logit_disease']
    df_global_raw['probability_adjusted'] = 1.0 / (1.0 + np.exp(-df_global_raw['logit_adjusted'].astype(float)))

    out_global_adj = os.path.join(MODEL_DIR, f"top_{TOP_K}_global_adjusted.csv")
    df_global_raw.to_csv(out_global_adj, index=False)
    print("Saved global adjusted top-K to", out_global_adj)

    # Diversified adjusted: pick entries ensuring disease diversity
    df_sorted_adj = df_global_raw.sort_values('probability_adjusted', ascending=False)
    seen = set()
    diversified = []
    for _, r in df_sorted_adj.iterrows():
        did = r['disease_id']
        if did in seen:
            continue
        diversified.append(r.to_dict())
        seen.add(did)
        if len(diversified) >= TOP_K:
            break
    df_div_adj = pd.DataFrame(diversified)
    out_div_adj = os.path.join(MODEL_DIR, f"top_{TOP_K}_diversified_adjusted.csv")
    df_div_adj.to_csv(out_div_adj, index=False)
    print("Saved diversified adjusted top-K to", out_div_adj)
    print("\nDiversified adjusted sample (up to 20):")
    pretty_print(df_div_adj[['drug_id','drug_name','disease_id','disease_name','probability_adjusted','logit_adjusted']], maxrows=20)

    # Per-drug top-N (vectorized for fallback, predictor path uses batched predictor); keep both raw & adjusted
    per_rows_raw = []
    per_rows_adj = []

    print("\nComputing per-drug top-N (raw + adjusted).")
    # Build ordered lists to index into emb arrays by map index ordering
    # inv_drug: index->id ; inv_disease: index->id
    # we will iterate over drug indices 0..num_drugs-1 (embedding order must match mapping indices)
    if predictor is None:
        # fallback vectorized per-drug using numpy
        emb_di_norm = emb_di / (np.linalg.norm(emb_di, axis=1, keepdims=True) + 1e-12)
        for d_idx in tqdm(range(num_drugs), desc="per-drug"):
            drug_id = inv_drug.get(d_idx, f"DRUG_{d_idx}")
            dr = emb_dr[d_idx]
            drn = dr / (np.linalg.norm(dr) + 1e-12)
            cosine = (emb_di_norm * drn).sum(axis=1)
            probs = (cosine + 1.0) / 2.0
            logits = cosine
            adj_logits = logits - disease_mean
            adj_probs = 1.0 / (1.0 + np.exp(-adj_logits))

            # pick top indices for raw and adjusted
            # top raw
            if len(probs) <= TOP_PER_DRUG:
                idxs = np.argsort(-probs)
            else:
                idxs = np.argpartition(-probs, TOP_PER_DRUG - 1)[:TOP_PER_DRUG]
                idxs = idxs[np.argsort(-probs[idxs])]
            for idx in idxs:
                per_rows_raw.append({
                    "drug_id": drug_id,
                    "drug_name": drug_names.get(drug_id, ""),
                    "disease_id": inv_disease.get(idx, f"DIS_{idx}"),
                    "disease_name": disease_names.get(inv_disease.get(idx, f"DIS_{idx}"), ""),
                    "probability_raw": float(probs[idx]),
                    "logit_raw": float(logits[idx])
                })
            # top adjusted
            if len(adj_probs) <= TOP_PER_DRUG:
                idxs2 = np.argsort(-adj_probs)
            else:
                idxs2 = np.argpartition(-adj_probs, TOP_PER_DRUG - 1)[:TOP_PER_DRUG]
                idxs2 = idxs2[np.argsort(-adj_probs[idxs2])]
            for idx in idxs2:
                per_rows_adj.append({
                    "drug_id": drug_id,
                    "drug_name": drug_names.get(drug_id, ""),
                    "disease_id": inv_disease.get(idx, f"DIS_{idx}"),
                    "disease_name": disease_names.get(inv_disease.get(idx, f"DIS_{idx}"), ""),
                    "probability_adjusted": float(adj_probs[idx]),
                    "logit_adjusted": float(adj_logits[idx])
                })
    else:
        # predictor path - batch per drug for performance
        emb_di_t = torch.from_numpy(emb_di).to(device)
        for d_idx in tqdm(range(num_drugs), desc="per-drug"):
            drug_id = inv_drug.get(d_idx, f"DRUG_{d_idx}")
            with torch.no_grad():
                z_dr = torch.from_numpy(emb_dr[d_idx]).to(device).unsqueeze(0).expand(num_diseases, -1)
                logits_t = predictor(z_dr, emb_di_t).cpu().numpy()
            logits = logits_t
            probs = 1.0 / (1.0 + np.exp(-logits))
            adj_logits = logits - disease_mean
            adj_probs = 1.0 / (1.0 + np.exp(-adj_logits))
            # raw top
            if len(probs) <= TOP_PER_DRUG:
                idxs = np.argsort(-probs)
            else:
                idxs = np.argpartition(-probs, TOP_PER_DRUG - 1)[:TOP_PER_DRUG]
                idxs = idxs[np.argsort(-probs[idxs])]
            for idx in idxs:
                per_rows_raw.append({
                    "drug_id": drug_id,
                    "drug_name": drug_names.get(drug_id, ""),
                    "disease_id": inv_disease.get(idx, f"DIS_{idx}"),
                    "disease_name": disease_names.get(inv_disease.get(idx, f"DIS_{idx}"), ""),
                    "probability_raw": float(probs[idx]),
                    "logit_raw": float(logits[idx])
                })
            # adjusted top
            if len(adj_probs) <= TOP_PER_DRUG:
                idxs2 = np.argsort(-adj_probs)
            else:
                idxs2 = np.argpartition(-adj_probs, TOP_PER_DRUG - 1)[:TOP_PER_DRUG]
                idxs2 = idxs2[np.argsort(-adj_probs[idxs2])]
            for idx in idxs2:
                per_rows_adj.append({
                    "drug_id": drug_id,
                    "drug_name": drug_names.get(drug_id, ""),
                    "disease_id": inv_disease.get(idx, f"DIS_{idx}"),
                    "disease_name": disease_names.get(inv_disease.get(idx, f"DIS_{idx}"), ""),
                    "probability_adjusted": float(adj_probs[idx]),
                    "logit_adjusted": float(adj_logits[idx])
                })

    df_per_raw = pd.DataFrame(per_rows_raw)
    df_per_adj = pd.DataFrame(per_rows_adj)
    os.makedirs(MODEL_DIR, exist_ok=True)
    out_per_raw = os.path.join(MODEL_DIR, f"top_{TOP_PER_DRUG}_per_drug_raw.csv")
    out_per_adj = os.path.join(MODEL_DIR, f"top_{TOP_PER_DRUG}_per_drug_adjusted.csv")
    df_per_raw.to_csv(out_per_raw, index=False)
    df_per_adj.to_csv(out_per_adj, index=False)
    print("Saved per-drug raw to", out_per_raw)
    print("Saved per-drug adjusted to", out_per_adj)
    print("Sample per-drug raw (first 10):")
    pretty_print(df_per_raw.head(10), maxrows=10)

    # Post-filtering: remove salts and apply MIN_PROB
    df_filtered = df_per_adj.copy()
    if 'drug_name' in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered['drug_name'].fillna("").map(is_excluded_name)]
    df_filtered = df_filtered[df_filtered['probability_adjusted'] >= MIN_PROB]
    out_filtered = os.path.join(MODEL_DIR, f"top_{TOP_PER_DRUG}_per_drug_filtered_prob{int(MIN_PROB*100)}_adjusted.csv")
    df_filtered.to_csv(out_filtered, index=False)
    print(f"Saved filtered per-drug adjusted to {out_filtered} (MIN_PROB={MIN_PROB})")
    print("Filtered sample (first 10):")
    pretty_print(df_filtered.head(10), maxrows=10)

    # diversified raw (unique disease) as well
    df_div_raw = df_global_raw.drop_duplicates(subset=['disease_id']).sort_values('probability_raw', ascending=False).head(TOP_K)
    out_div_raw = os.path.join(MODEL_DIR, f"top_{TOP_K}_diversified_raw.csv")
    df_div_raw.to_csv(out_div_raw, index=False)
    print("Saved diversified raw to", out_div_raw)

    # optional histogram plot saved (may fail in headless environments)
    try:
        import matplotlib.pyplot as plt
        if 'probability_raw' in df_per_raw.columns:
            plt.figure(figsize=(6,3))
            plt.hist(df_per_raw['probability_raw'].values, bins=50)
            plt.title("Histogram of raw predicted probabilities (per-drug top entries)")
            plt.tight_layout()
            plt.savefig(os.path.join(MODEL_DIR,"prob_hist_raw.png"))
            print("Saved probability histogram to", os.path.join(MODEL_DIR,"prob_hist_raw.png"))
    except Exception:
        pass

    print("\nDone. Outputs written to", MODEL_DIR)
    print("Notes:")
    print(" - Ensure your final_drugs/final_diseases CSVs align with embedding row order. If they don't, use mappings.pkl index mapping.")
    print(" - If many adjusted scores cluster on a few diseases, consider stronger bias correction (Platt / calibration, reweighting during training, or negative sampling improvements).")
    print(" - If you'd like I can wire the exact classifier (if you provide the training model definition or a known state_dict layout) to compute logits exactly instead of dot-product/cosine fallback.")

if __name__ == "__main__":
    main()
