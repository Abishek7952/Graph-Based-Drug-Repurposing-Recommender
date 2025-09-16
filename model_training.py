#!/usr/bin/env python3
"""
model_training.py (final) — embedding-based, robust hetero-GNN, with better defaults

Changes from previous:
- Removed Matthews CorrCoef and ranking metrics (Hits@K/MRR).
- Suppress per-epoch verbose evaluation output and plotting.
- Save and print ONLY final evaluation summaries + plots for TRAIN and TEST.
- Compact per-epoch line printed (loss + val ROC + val PR) while tqdm progress runs.
- All filenames/dirs preserved.
"""

import os
import math
import time
import random
import pickle
import traceback
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.nn import SAGEConv

# plotting & metrics
import matplotlib
matplotlib.use('Agg')  # for headless saving
import matplotlib.pyplot as plt
import numpy as np

# Optional Neo4j import
try:
    from neo4j import GraphDatabase
    NEO4J_OK = True
except Exception:
    NEO4J_OK = False

# ---------------- Config ----------------
NEO4J_URI = "bolt://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
CSV_DIR = "."

EMBEDDING_SIZE = 128        # embedding dimensionality used for node features
HIDDEN = 128
NUM_LAYERS = 2
NUM_EPOCHS = 100

LEARNING_RATE = 1e-3        # changed to 1e-3
WEIGHT_DECAY = 1e-5         # small weight decay to regularize
DROPOUT = 0.6               # increased dropout
SEED = 42

EARLY_STOPPING = True
PATIENCE = 8

CHECKPOINT_PATH = "best_checkpoint.pt"
MODEL_OUT_DIR = "model_out"

# Choose which metric to use for checkpointing: 'roc' (default) or 'pr'
CHECKPOINT_METRIC = 'roc'  # 'roc' or 'pr'

torch.manual_seed(SEED)
random.seed(SEED)

# ---------------- Data helpers ----------------
def fetch_from_neo4j():
    if not NEO4J_OK:
        raise RuntimeError("neo4j driver not available")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            drugs = pd.DataFrame([dict(r) for r in session.run("MATCH (d:Drug) RETURN d.id AS id, d.name AS name")])
            proteins = pd.DataFrame([dict(r) for r in session.run("MATCH (p:Protein) RETURN p.id AS id, p.name AS name")])
            diseases = pd.DataFrame([dict(r) for r in session.run("MATCH (dis:Disease) RETURN dis.name AS name")])
            targets_edges = pd.DataFrame([dict(r) for r in session.run("MATCH (d:Drug)-[:TARGETS]->(p:Protein) RETURN d.id AS source, p.id AS target")])
            treats_edges = pd.DataFrame([dict(r) for r in session.run("MATCH (d:Drug)-[:TREATS]->(dis:Disease) RETURN d.id AS source, dis.name AS target")])
    finally:
        driver.close()
    return {'drugs': drugs, 'proteins': proteins, 'diseases': diseases,
            'targets_edges': targets_edges, 'treats_edges': treats_edges}

def load_from_csv(csv_dir=CSV_DIR):
    def r(fn, cols=None):
        p = os.path.join(csv_dir, fn)
        if not os.path.exists(p):
            return pd.DataFrame(columns=cols) if cols else pd.DataFrame()
        return pd.read_csv(p, dtype=str).fillna("")
    drugs = r("final_drugs.csv", cols=["id","name"])
    proteins = r("final_proteins.csv", cols=["id","name"])
    diseases = r("final_diseases.csv", cols=["name"])
    targets_edges = r("final_edges_drug_targets.csv", cols=["drug_id","target_id"])
    treats_edges = r("final_edges_drug_treats_disease.csv", cols=["drug_id","disease_name"])
    if "drug_id" in targets_edges.columns and "target_id" in targets_edges.columns:
        targets_edges = targets_edges.rename(columns={"drug_id":"source","target_id":"target"})
    if "drug_id" in treats_edges.columns and "disease_name" in treats_edges.columns:
        treats_edges = treats_edges.rename(columns={"drug_id":"source","disease_name":"target"})
    return {'drugs': drugs, 'proteins': proteins, 'diseases': diseases,
            'targets_edges': targets_edges, 'treats_edges': treats_edges}

# ---------------- Build HeteroData ----------------
def build_heterodata(graph_dict):
    drugs_df = graph_dict.get('drugs') if graph_dict.get('drugs') is not None else pd.DataFrame(columns=['id','name'])
    proteins_df = graph_dict.get('proteins') if graph_dict.get('proteins') is not None else pd.DataFrame(columns=['id','name'])
    diseases_df = graph_dict.get('diseases') if graph_dict.get('diseases') is not None else pd.DataFrame(columns=['name'])
    targets_df = graph_dict.get('targets_edges') if graph_dict.get('targets_edges') is not None else pd.DataFrame(columns=['source','target'])
    treats_df = graph_dict.get('treats_edges') if graph_dict.get('treats_edges') is not None else pd.DataFrame(columns=['source','target'])

    if 'id' not in drugs_df.columns and drugs_df.shape[1] >= 1:
        drugs_df = drugs_df.rename(columns={drugs_df.columns[0]:"id"})
    if 'id' not in proteins_df.columns and proteins_df.shape[1] >= 1:
        proteins_df = proteins_df.rename(columns={proteins_df.columns[0]:"id"})
    if 'name' not in diseases_df.columns and diseases_df.shape[1] >= 1:
        diseases_df = diseases_df.rename(columns={diseases_df.columns[0]:"name"})

    drugs_list = list(dict.fromkeys(drugs_df['id'].astype(str).tolist()))
    proteins_list = list(dict.fromkeys(proteins_df['id'].astype(str).tolist()))
    diseases_list = list(dict.fromkeys(diseases_df['name'].astype(str).tolist()))

    drug_map = {nid:i for i,nid in enumerate(drugs_list)}
    protein_map = {nid:i for i,nid in enumerate(proteins_list)}
    disease_map = {nid:i for i,nid in enumerate(diseases_list)}

    print(f"Node counts -> drugs: {len(drug_map)}, proteins: {len(protein_map)}, diseases: {len(disease_map)}")

    data = HeteroData()
    data['drug'].num_nodes = len(drug_map)
    data['protein'].num_nodes = len(protein_map)
    data['disease'].num_nodes = len(disease_map)

    for df in (targets_df, treats_df):
        if df is None or df.empty:
            continue
        cols = df.columns.tolist()
        if 'drug_id' in cols and 'target_id' in cols:
            df.rename(columns={'drug_id':'source','target_id':'target'}, inplace=True)
        if 'drug_id' in cols and 'disease_name' in cols:
            df.rename(columns={'drug_id':'source','disease_name':'target'}, inplace=True)

    if not targets_df.empty:
        tdf = targets_df[targets_df['source'].isin(drug_map) & targets_edges['target'].isin(protein_map)] if 'targets_edges' in locals() else targets_df
        # fallback safe building
        tdf = targets_df[targets_df['source'].isin(drug_map) & targets_df['target'].isin(protein_map)].copy()
        src = [drug_map[s] for s in tdf['source'].astype(str)]
        dst = [protein_map[t] for t in tdf['target'].astype(str)]
        data['drug','targets','protein'].edge_index = torch.tensor([src,dst], dtype=torch.long) if len(src)>0 else torch.empty((2,0), dtype=torch.long)
    else:
        data['drug','targets','protein'].edge_index = torch.empty((2,0), dtype=torch.long)

    if not treats_df.empty:
        tdf = treats_df[treats_df['source'].isin(drug_map) & treats_df['target'].isin(disease_map)].copy()
        src = [drug_map[s] for s in tdf['source'].astype(str)]
        dst = [disease_map[t] for t in tdf['target'].astype(str)]
        data['drug','treats','disease'].edge_index = torch.tensor([src,dst], dtype=torch.long) if len(src)>0 else torch.empty((2,0), dtype=torch.long)
    else:
        data['drug','treats','disease'].edge_index = torch.empty((2,0), dtype=torch.long)

    data = ToUndirected()(data)
    maps = {'drug':drug_map, 'protein':protein_map, 'disease':disease_map}
    return data, maps

# ---------------- Hetero GNN (manual) ----------------
class HeteroGraphSAGE(nn.Module):
    def __init__(self, node_in_dims, hidden_channels, out_channels, edge_types, num_layers=2, dropout=0.5):
        super().__init__()
        self.edge_types = edge_types
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.node_in_dims = node_in_dims
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.convs = nn.ModuleDict()
        for layer in range(num_layers):
            for (src, rel, dst) in edge_types:
                key = f"l{layer}__{src}__{rel}__{dst}"
                if layer == 0:
                    in_src = node_in_dims[src]
                    in_dst = node_in_dims[dst]
                else:
                    in_src = hidden_channels
                    in_dst = hidden_channels
                out_ch = hidden_channels if layer < num_layers - 1 else out_channels
                self.convs[key] = SAGEConv((in_src, in_dst), out_ch)

    def forward(self, x_dict, edge_index_dict):
        h_dict = {nt: x for nt, x in x_dict.items()}
        for layer in range(self.num_layers):
            agg = {}
            for nt, h in h_dict.items():
                out_dim = self.hidden_channels if layer < self.num_layers - 1 else self.out_channels
                agg[nt] = torch.zeros((h.size(0), out_dim), device=h.device)
            for (src, rel, dst) in self.edge_types:
                key = f"l{layer}__{src}__{rel}__{dst}"
                conv = self.convs[key]
                ei = edge_index_dict.get((src, rel, dst), None)
                if ei is None or ei.numel() == 0:
                    continue
                h_src = h_dict[src]
                h_dst = h_dict[dst]
                out = conv((h_src, h_dst), ei)
                agg[dst] = agg[dst] + out
            for nt in h_dict:
                h_new = agg[nt]
                if layer < self.num_layers - 1:
                    h_new = F.relu(h_new)
                    h_new = self.dropout(h_new)
                h_dict[nt] = h_new
        return h_dict

# ---------------- Full model with embeddings ----------------
class FullModel(nn.Module):
    def __init__(self, num_nodes_map, embedding_dim, node_in_dims, hidden_channels, out_channels, edge_types, num_layers, dropout):
        """
        num_nodes_map: dict node_type -> number of nodes
        embedding_dim: embedding size used for nn.Embedding (we will use same for all types)
        node_in_dims: dict node_type -> input dim (here equals embedding_dim)
        """
        super().__init__()
        # embeddings
        self.drug_embed = nn.Embedding(num_nodes_map['drug'], embedding_dim)
        self.protein_embed = nn.Embedding(num_nodes_map['protein'], embedding_dim)
        self.disease_embed = nn.Embedding(num_nodes_map['disease'], embedding_dim)
        nn.init.xavier_uniform_(self.drug_embed.weight)
        nn.init.xavier_uniform_(self.protein_embed.weight)
        nn.init.xavier_uniform_(self.disease_embed.weight)

        # hetero GNN
        self.gnn = HeteroGraphSAGE(node_in_dims=node_in_dims, hidden_channels=hidden_channels,
                                   out_channels=out_channels, edge_types=edge_types, num_layers=num_layers, dropout=dropout)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(2 * out_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, edge_index_dict, edge_label_index):
        # build x_dict from embeddings
        device = next(self.parameters()).device
        x_dict = {
            'drug': self.drug_embed.weight.to(device),
            'protein': self.protein_embed.weight.to(device),
            'disease': self.disease_embed.weight.to(device)
        }
        # ensure edge_index tensors are on same device
        eid = {k: v.to(device) for k, v in edge_index_dict.items()}
        z_dict = self.gnn(x_dict, eid)
        src_idx = edge_label_index[0].to(device)
        dst_idx = edge_label_index[1].to(device)
        z_src = z_dict['drug'][src_idx]
        z_dst = z_dict['disease'][dst_idx]
        z_pair = torch.cat([z_src, z_dst], dim=-1)
        out = self.predictor(z_pair).view(-1)
        return out, z_dict

# ---------------- Helpers: metrics, plotting (simplified) ----------------
def _safe_sklearn_imports():
    try:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, precision_recall_curve,
            roc_curve, precision_recall_fscore_support, confusion_matrix,
            brier_score_loss, log_loss
        )
        return {
            'roc_auc_score': roc_auc_score,
            'average_precision_score': average_precision_score,
            'precision_recall_curve': precision_recall_curve,
            'roc_curve': roc_curve,
            'precision_recall_fscore_support': precision_recall_fscore_support,
            'confusion_matrix': confusion_matrix,
            'brier_score_loss': brier_score_loss,
            'log_loss': log_loss
        }
    except Exception as e:
        print("Warning: sklearn missing or failed to import. Some metrics will be skipped.", e)
        return None

def save_pr_roc_plots(labs, probs, out_path_pr, out_path_roc, title_suffix=""):
    sk = _safe_sklearn_imports()
    if sk is None:
        return
    # precision_recall_curve returns (precision, recall, thresholds)
    precision, recall, _ = sk['precision_recall_curve'](labs, probs)
    fpr, tpr, _ = sk['roc_curve'](labs, probs)

    # PR curve
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curve {title_suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_pr, dpi=150)
    plt.close()

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path_roc, dpi=150)
    plt.close()

def save_confusion_matrix(labs, preds_binary, out_path, title_suffix=""):
    sk = _safe_sklearn_imports()
    if sk is None:
        return
    cm = sk['confusion_matrix'](labs, preds_binary)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.colorbar()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['0','1'])
    plt.yticks(tick_marks, ['0','1'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def pretty_print_metrics_simple(split_name, roc, pr, prec, rec, f1, lloss, brier, cm_counts, total_samples):
    sep = "="*70
    print("\n" + sep)
    print(f" EVALUATION SUMMARY -> {split_name.upper()}")
    print(sep)
    print(f"Samples evaluated : {total_samples}")
    print(f"ROC AUC           : {roc:.4f}" if roc is not None else "ROC AUC           : N/A")
    print(f"PR AUC (AP)       : {pr:.4f}" if pr is not None else "PR AUC (AP)       : N/A")
    print(f"LogLoss           : {lloss:.6f}" if lloss is not None else "LogLoss           : N/A")
    print(f"Brier score       : {brier:.6f}" if brier is not None else "Brier score       : N/A")
    print(f"Precision (thr=0.5): {prec:.4f}")
    print(f"Recall    (thr=0.5): {rec:.4f}")
    print(f"F1        (thr=0.5): {f1:.4f}")
    if cm_counts is not None:
        tn, fp, fn, tp = cm_counts
        print("\nConfusion Matrix (counts):")
        print(f"  TN: {tn}   FP: {fp}")
        print(f"  FN: {fn}   TP: {tp}")
        pos = tp + fn
        neg = tn + fp
        print(f"\nPositives: {pos}, Negatives: {neg}")
    print(sep + "\n")

# ---------------- Helpers: training / eval ----------------
def train_epoch(model, optimizer, data_train, device):
    model.train()
    optimizer.zero_grad()
    eli = data_train['drug','treats','disease'].edge_label_index.to(device).long()
    labels = data_train['drug','treats','disease'].edge_label.to(device).float()
    edge_index_dict = {k: v.to(device) for k, v in data_train.edge_index_dict.items()}
    out, _ = model(edge_index_dict, eli)
    loss = F.binary_cross_entropy_with_logits(out, labels)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())

@torch.no_grad()
def eval_split(model, data_split, device, split_name="split", save_plots=False, out_dir=MODEL_OUT_DIR, verbose=False):
    """
    Simplified eval:
    - When verbose=False, no print/save occurs (used during epochs).
    - When verbose=True and save_plots=True, prints a single readable evaluation summary
      and saves PR/ROC + confusion matrix images to out_dir.
    Returns: (roc, pr) preserving old interface.
    """
    model.eval()
    eli = data_split['drug','treats','disease'].edge_label_index.to(device).long()
    labels = data_split['drug','treats','disease'].edge_label.to(device).float()
    edge_index_dict = {k: v.to(device) for k, v in data_split.edge_index_dict.items()}
    out, _ = model(edge_index_dict, eli)
    probs = torch.sigmoid(out).cpu().numpy()
    labs = labels.cpu().numpy()

    sk = _safe_sklearn_imports()
    roc = None
    pr = None
    prec = rec = f1 = 0.0
    cm_counts = None
    lloss = None
    brier = None

    if sk is not None:
        try:
            roc = float(sk['roc_auc_score'](labs, probs))
        except Exception:
            roc = None
        try:
            pr = float(sk['average_precision_score'](labs, probs))
        except Exception:
            pr = None
        try:
            prec, rec, f1, _ = sk['precision_recall_fscore_support'](labs, (probs >= 0.5).astype(int), average='binary', zero_division=0)
        except Exception:
            prec = rec = f1 = 0.0
        try:
            cm = sk['confusion_matrix'](labs, (probs >= 0.5).astype(int))
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                cm_counts = (int(tn), int(fp), int(fn), int(tp))
            else:
                cm_counts = tuple(int(x) for x in cm.ravel())
        except Exception:
            cm_counts = None
        try:
            eps = 1e-12
            lloss = float(sk['log_loss'](labs, np.clip(probs, eps, 1 - eps)))
        except Exception:
            lloss = None
        try:
            brier = float(sk['brier_score_loss'](labs, probs))
        except Exception:
            brier = None

    # Save + print only when requested
    if verbose:
        # Ensure dir
        os.makedirs(out_dir, exist_ok=True)
        title_suffix = f"{split_name.upper()}"
        pr_path = os.path.join(out_dir, f"pr_curve_{split_name}.png")
        roc_path = os.path.join(out_dir, f"roc_curve_{split_name}.png")
        cm_path = os.path.join(out_dir, f"confusion_matrix_{split_name}.png")
        # Save plots
        try:
            save_pr_roc_plots(labs, probs, pr_path, roc_path, title_suffix=title_suffix)
        except Exception as e:
            print("Warning saving PR/ROC plots:", e)
        try:
            save_confusion_matrix(labs, (probs >= 0.5).astype(int), cm_path, title_suffix=title_suffix)
        except Exception as e:
            print("Warning saving confusion matrix:", e)
        # Pretty print
        pretty_print_metrics_simple(split_name=split_name,
                                    roc=(roc if roc is not None else float('nan')),
                                    pr=(pr if pr is not None else float('nan')),
                                    prec=prec, rec=rec, f1=f1,
                                    lloss=(lloss if lloss is not None else float('nan')),
                                    brier=(brier if brier is not None else float('nan')),
                                    cm_counts=cm_counts,
                                    total_samples=len(labs))

    # Preserve old return signature
    if roc is not None:
        return float(roc), (float(pr) if pr is not None else float('nan'))
    else:
        try:
            pred_bin = (probs >= 0.5).astype(int)
            acc = float((pred_bin == labs).mean())
            return acc, None
        except Exception:
            return None, None

# ---------------- Main ----------------
def main():
    try:
        print("Attempting to fetch from Neo4j...")
        graph = fetch_from_neo4j()
    except Exception as e:
        print("Neo4j fetch failed (", e, "), falling back to CSVs.")
        graph = load_from_csv(CSV_DIR)

    data, maps = build_heterodata(graph)
    print("Built HeteroData (counts):", data)

    transform = RandomLinkSplit(
        is_undirected=True,
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=1.0,
        edge_types=[('drug','treats','disease')],
        rev_edge_types=[('disease','treats_rev','drug')]
    )
    train_data, val_data, test_data = transform(data)

    # Sanity check
    for split_name, d in (("train", train_data), ("val", val_data), ("test", test_data)):
        print(f"Sanity check edge_index types for {split_name}:")
        for k, v in d.edge_index_dict.items():
            print(" ", k, type(v), None if not isinstance(v, torch.Tensor) else tuple(v.shape), getattr(v, "dtype", None))

    # ensure edge_label_index long
    for d in (train_data, val_data, test_data):
        try:
            d['drug','treats','disease'].edge_label_index = d['drug','treats','disease'].edge_label_index.long()
        except Exception:
            pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Build embeddings (learnable)
    num_nodes_map = {'drug': data['drug'].num_nodes, 'protein': data['protein'].num_nodes, 'disease': data['disease'].num_nodes}
    print("Num nodes:", num_nodes_map)

    # edge types list
    edge_types = list(train_data.edge_index_dict.keys())
    print("Edge types:", edge_types)

    # node_in_dims for SAGE conv layer 0 (equals embedding dim)
    node_in_dims = {nt: EMBEDDING_SIZE for nt in num_nodes_map}

    # instantiate model
    model = FullModel(num_nodes_map=num_nodes_map,
                      embedding_dim=EMBEDDING_SIZE,
                      node_in_dims=node_in_dims,
                      hidden_channels=HIDDEN,
                      out_channels=EMBEDDING_SIZE,
                      edge_types=edge_types,
                      num_layers=NUM_LAYERS,
                      dropout=DROPOUT)
    model = model.to(device)

    # optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # move split data to device for edge_index operations
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    best_val = -math.inf
    best_epoch = -1
    no_improve = 0
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)

    print("Starting training...")
    start_time = time.time()
    # Use tqdm to show progress; keep prints minimal
    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Train epochs"):
        try:
            loss = train_epoch(model, optimizer, train_data, device)
        except Exception:
            print("Error during train_epoch; dumping traceback and edge_index types:")
            traceback.print_exc()
            try:
                for k,v in train_data.edge_index_dict.items():
                    print(" ", k, type(v), getattr(v,"dtype",None), None if not isinstance(v, torch.Tensor) else tuple(v.shape))
            except Exception:
                pass
            raise

        # Evaluate compactly without verbose prints or plots (fast)
        # We only need val metrics to decide checkpointing and a compact per-epoch line
        train_roc, train_pr = eval_split(model, train_data, device, split_name="train", save_plots=False, verbose=False)
        val_roc, val_pr = eval_split(model, val_data, device, split_name="val", save_plots=False, verbose=False)

        # compact single-line epoch summary (no big block prints)
        # print a single compact line per epoch (this will appear beneath tqdm)
        print(f"Epoch {epoch:03d} Loss {loss:.4f} | Val ROC {val_roc:.4f} | Val PR {val_pr if val_pr is not None else float('nan'):.4f}")

        # choose checkpointing metric
        if CHECKPOINT_METRIC == 'pr':
            current_metric = val_pr if val_pr is not None else -math.inf
        else:
            current_metric = val_roc if val_roc is not None else -math.inf

        if current_metric > best_val + 1e-6:
            best_val = current_metric
            best_epoch = epoch
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metric': current_metric,
                'val_roc': val_roc,
                'val_pr': val_pr,
                'mappings': maps,
                'checkpoint_metric': CHECKPOINT_METRIC
            }, CHECKPOINT_PATH)
            print(f"Checkpoint saved (epoch {epoch}, val_{CHECKPOINT_METRIC} {current_metric:.4f})")
        else:
            no_improve += 1

        if EARLY_STOPPING and no_improve >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs. Early stopping.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes. Best val {CHECKPOINT_METRIC.upper()} {best_val:.4f} at epoch {best_epoch}.")

    # load best checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        chk = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(chk['model_state_dict'])
        print("Loaded best checkpoint.")

    # FINAL: compute and print only final TRAIN and TEST summaries (with plots)
    print("Computing final TRAIN evaluation and saving plots...")
    _ = eval_split(model, train_data, device, split_name="train_final", save_plots=True, out_dir=MODEL_OUT_DIR, verbose=True)

    print("Computing final TEST evaluation and saving plots...")
    test_roc, test_pr = eval_split(model, test_data, device, split_name="test_final", save_plots=True, out_dir=MODEL_OUT_DIR, verbose=True)

    print(f"Final Test ROC AUC: {test_roc:.4f}" if test_roc is not None else "Final Test ROC AUC: N/A")
    print(f"Final Test PR AUC (avg precision): {test_pr:.4f}" if test_pr is not None else "Final Test PR AUC (avg precision): N/A")

    # Save model & embeddings & mappings (same filenames as before)
    torch.save(model.state_dict(), os.path.join(MODEL_OUT_DIR, "model_full.pt"))
    # save embedding matrices (from nn.Embedding weights)
    torch.save(model.drug_embed.weight.cpu(), os.path.join(MODEL_OUT_DIR, "embeddings_drug.pt"))
    torch.save(model.protein_embed.weight.cpu(), os.path.join(MODEL_OUT_DIR, "embeddings_protein.pt"))
    torch.save(model.disease_embed.weight.cpu(), os.path.join(MODEL_OUT_DIR, "embeddings_disease.pt"))
    with open(os.path.join(MODEL_OUT_DIR, "mappings.pkl"), "wb") as f:
        pickle.dump(maps, f)
    print("Saved model, embeddings and mappings to", MODEL_OUT_DIR)

if __name__ == "__main__":
    main()
