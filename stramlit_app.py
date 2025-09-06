#!/usr/bin/env python3
"""
streamlit_app_fixed.py

Polished Streamlit UI for the drug repurposing recommender (fixed).
Key fixes:
 - Do NOT store pandas DataFrames in widget values / session_state.
 - Avoid deprecated/absent experimental rerun/set_query_params usage.
 - Defensive reading of CSVs + deduping duplicate columns.
 - Friendly "recommend" mode that returns verbal summary + table + download.
"""

import os
import textwrap
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- Config ----------------
MODEL_OUT = "model_out"
DRUGS_CSV = os.path.join("data_cleaning", "final_drugs.csv")
DISEASES_CSV = os.path.join("data_cleaning", "final_diseases.csv")

FILES = {
    "global_adjusted": os.path.join(MODEL_OUT, "top_20_global_adjusted.csv"),
    "global_raw": os.path.join(MODEL_OUT, "top_20_global_raw.csv"),
    "diversified_adjusted": os.path.join(MODEL_OUT, "top_20_diversified_adjusted.csv"),
    "diversified_raw": os.path.join(MODEL_OUT, "top_20_diversified_raw.csv"),
    "per_drug_adjusted": os.path.join(MODEL_OUT, "top_5_per_drug_adjusted.csv"),
    "per_drug_raw": os.path.join(MODEL_OUT, "top_5_per_drug_raw.csv"),
    "per_drug_filtered_adj": os.path.join(MODEL_OUT, "top_5_per_drug_filtered_prob50_adjusted.csv"),
}

st.set_page_config(page_title="Drug Repurposing Explorer", layout="wide")

# ---------------- Helpers ----------------
@st.cache_data(ttl=300)
def safe_read_csv(path):
    """Read CSV and drop duplicated column names (narwhals issue)."""
    df = pd.read_csv(path)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    # coerce prob/logit columns to numeric where possible
    for c in df.columns:
        if 'prob' in c.lower() or 'logit' in c.lower():
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

@st.cache_data(ttl=300)
def load_name_maps():
    drugs_map = {}
    diseases_map = {}
    if os.path.exists(DRUGS_CSV):
        try:
            df = pd.read_csv(DRUGS_CSV, dtype=str)
            if 'id' in df.columns and 'name' in df.columns:
                drugs_map = df.set_index('id')['name'].to_dict()
        except Exception:
            pass
    if os.path.exists(DISEASES_CSV):
        try:
            df = pd.read_csv(DISEASES_CSV, dtype=str)
            if 'id' in df.columns and 'name' in df.columns:
                diseases_map = df.set_index('id')['name'].to_dict()
            elif 'name' in df.columns:
                diseases_map = {row['name']: row['name'] for _, row in df.iterrows()}
        except Exception:
            pass
    return drugs_map, diseases_map

def safe_get_prob_col(df):
    """Return the first column that looks like a probability column, or None."""
    for c in df.columns:
        if 'prob' in c.lower():
            return c
    return None

def verbalize_recommendations_for_drug(df_per_drug, drug_id, top_n=5):
    """Return verbal summary (list of sentences) and DataFrame of top_n recommendations for a drug."""
    if df_per_drug is None or df_per_drug.empty:
        return ["No per-drug recommendations available."], pd.DataFrame()
    # try match on drug_id exactly first
    rows = df_per_drug[df_per_drug['drug_id'].astype(str) == str(drug_id)]
    if rows.empty and 'drug_name' in df_per_drug.columns:
        # allow search by name substring
        rows = df_per_drug[df_per_drug['drug_name'].str.contains(str(drug_id), case=False, na=False)]
    if rows.empty:
        return [f"No predictions found for '{drug_id}'."], pd.DataFrame()
    prob_col = safe_get_prob_col(rows)
    if prob_col:
        rows = rows.sort_values(prob_col, ascending=False)
    top = rows.head(top_n)
    drug_name = top['drug_name'].iloc[0] if 'drug_name' in top.columns else ""
    pieces = []
    for _, r in top.iterrows():
        dn = r.get('disease_name', r.get('disease_id', 'Unknown'))
        p = r[prob_col] if prob_col else None
        if p is not None and not np.isnan(p):
            pieces.append(f"{dn} (prob {p:.3f})")
        else:
            pieces.append(str(dn))
    header = f"Top {len(top)} predictions for **{drug_name}** ({drug_id}):" if drug_name else f"Top {len(top)} predictions for {drug_id}:"
    sentence = header + " " + ", ".join(pieces) + "."
    return [sentence], top

def verbalize_recommendations_for_disease(df_per_drug, disease_id, top_n=5):
    """Return verbal summary and DataFrame of top_n drugs for a disease."""
    if df_per_drug is None or df_per_drug.empty:
        return ["No per-drug recommendations available."], pd.DataFrame()
    rows = df_per_drug[df_per_drug['disease_id'].astype(str) == str(disease_id)]
    if rows.empty and 'disease_name' in df_per_drug.columns:
        rows = df_per_drug[df_per_drug['disease_name'].str.contains(str(disease_id), case=False, na=False)]
    if rows.empty:
        return [f"No matching entries for '{disease_id}'."], pd.DataFrame()
    prob_col = safe_get_prob_col(rows)
    if prob_col:
        rows = rows.sort_values(prob_col, ascending=False)
    top = rows.head(top_n)
    disease_name = top['disease_name'].iloc[0] if 'disease_name' in top.columns else ""
    pieces = []
    for _, r in top.iterrows():
        dn = r.get('drug_name', r.get('drug_id', 'Unknown'))
        p = r[prob_col] if prob_col else None
        if p is not None and not np.isnan(p):
            pieces.append(f"{dn} (prob {p:.3f})")
        else:
            pieces.append(str(dn))
    header = f"Top {len(top)} candidate drugs for **{disease_name}** ({disease_id}):" if disease_name else f"Top {len(top)} candidate drugs for {disease_id}:"
    sentence = header + " " + ", ".join(pieces) + "."
    return [sentence], top

# ---------------- Load data once ----------------
drug_name_map, disease_name_map = load_name_maps()

per_drug_adj = safe_read_csv(FILES["per_drug_adjusted"]) if os.path.exists(FILES["per_drug_adjusted"]) else None
per_drug_raw = safe_read_csv(FILES["per_drug_raw"]) if os.path.exists(FILES["per_drug_raw"]) else None
global_adj = safe_read_csv(FILES["global_adjusted"]) if os.path.exists(FILES["global_adjusted"]) else None
global_raw = safe_read_csv(FILES["global_raw"]) if os.path.exists(FILES["global_raw"]) else None
div_adj = safe_read_csv(FILES["diversified_adjusted"]) if os.path.exists(FILES["diversified_adjusted"]) else None
div_raw = safe_read_csv(FILES["diversified_raw"]) if os.path.exists(FILES["diversified_raw"]) else None

# ---------------- UI layout ----------------
st.markdown("<div style='padding:12px;background:linear-gradient(90deg,#02263b,#085d73);border-radius:8px;color:white'>"
            "<h2 style='margin:0'>Drug Repurposing Explorer — polished</h2>"
            "<div style='font-size:13px;margin-top:4px;color:#cfe8f0'>Interactive recommendations + Top-K explorer + diagnostics</div>"
            "</div>", unsafe_allow_html=True)
st.write(" ")

# Sidebar navigation (single source of truth)
page = st.sidebar.radio("Select page", ["Home", "Recommend", "Explore Top-K", "Diagnostics"])

# ---------------- HOME ----------------
if page == "Home":
    st.header("Welcome")
    st.markdown(textwrap.dedent("""
        **Quick steps**
        1. Run your training & recommender pipeline to produce files under `model_out/`.
        2. Use the **Recommend** page for interactive verbal + table recommendations.
        3. Use **Explore Top-K** to inspect global/diversified/per-drug CSVs and download them.
        4. Use **Diagnostics** to check output distributions and bias.
    """))
    st.markdown("**Files detected in model_out/**")
    for key, p in FILES.items():
        st.write(f"- `{os.path.basename(p)}`: {'found' if os.path.exists(p) else 'missing'}")
    st.markdown("---")
    st.info("This app prefers the *per-drug adjusted* CSV. If it's missing, it gracefully falls back to raw outputs.")

# ---------------- RECOMMEND ----------------
elif page == "Recommend":
    st.header("Interactive recommendations")
    st.write("Type a drug or disease (ID or substring) and get a quick human-friendly summary + table.")
    dataset = per_drug_adj if per_drug_adj is not None else per_drug_raw if per_drug_raw is not None else None
    if dataset is None:
        st.warning("No per-drug CSV detected in model_out/. Generate `top_5_per_drug_adjusted.csv` or `top_5_per_drug_raw.csv` and reload.")
    else:
        mode = st.radio("Query mode", ["Drug → diseases", "Disease → drugs"])
        top_n = st.slider("Number of recommendations to show", 1, 20, 5)

        if mode == "Drug → diseases":
            q = st.text_input("Enter drug id or name (substring). E.g. DB00001 or aspirin")
            if st.button("Get recommendations for drug"):
                if not q:
                    st.warning("Type a drug id or name first.")
                else:
                    sentences, top_df = verbalize_recommendations_for_drug(dataset, q, top_n=top_n)
                    st.subheader("Summary")
                    for s in sentences:
                        st.success(s)
                    if not top_df.empty:
                        st.subheader("Top results")
                        st.dataframe(top_df, use_container_width=True)
                        st.download_button("Download results (.csv)", data=top_df.to_csv(index=False).encode('utf-8'),
                                           file_name=f"{q}_recommendations.csv", mime="text/csv")
                    else:
                        st.info("No table results for this query.")

        else:
            q = st.text_input("Enter disease id or name (substring). E.g. 'heart failure' or 'hypertension'")
            if st.button("Get recommendations for disease"):
                if not q:
                    st.warning("Type a disease id or name first.")
                else:
                    sentences, top_df = verbalize_recommendations_for_disease(dataset, q, top_n=top_n)
                    st.subheader("Summary")
                    for s in sentences:
                        st.success(s)
                    if not top_df.empty:
                        st.subheader("Top results")
                        st.dataframe(top_df, use_container_width=True)
                        st.download_button("Download results (.csv)", data=top_df.to_csv(index=False).encode('utf-8'),
                                           file_name=f"{q}_drug_candidates.csv", mime="text/csv")
                    else:
                        st.info("No table results for this query.")

# ---------------- EXPLORE TOP-K ----------------
elif page == "Explore Top-K":
    st.header("Explore Top-K and per-drug outputs")
    st.write("Browse the files produced by the recommender script. Select dataset and visualize.")
    choices = {
        "Global (adjusted)": global_adj,
        "Global (raw)": global_raw,
        "Diversified (adjusted)": div_adj,
        "Diversified (raw)": div_raw,
        "Per-drug (adjusted)": per_drug_adj,
        "Per-drug (raw)": per_drug_raw
    }
    choice_label = st.selectbox("Select dataset", list(choices.keys()))
    df_sel = choices[choice_label]
    top_k = st.slider("Top-K to visualize (if applicable)", 5, 200, 20)

    if df_sel is None or getattr(df_sel, "empty", True):
        st.warning(f"Dataset `{choice_label}` is not available.")
    else:
        st.subheader(choice_label)
        st.dataframe(df_sel.head(300), use_container_width=True)

        prob_col = safe_get_prob_col(df_sel)
        if prob_col:
            st.write(f"Detected probability column: `{prob_col}`")
            df_top = df_sel.sort_values(prob_col, ascending=False).head(top_k)
            # choose label for y axis
            y_label = 'drug_name' if 'drug_name' in df_top.columns else ('disease_name' if 'disease_name' in df_top.columns else 'drug_id')
            fig = px.bar(df_top, x=prob_col, y=y_label, orientation='h', height=480)
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No probability-like column found for visualization.")

        st.download_button("Download dataset (.csv)", data=df_sel.to_csv(index=False).encode('utf-8'),
                           file_name=f"{choice_label.replace(' ','_')}.csv", mime="text/csv")

# ---------------- DIAGNOSTICS ----------------
elif page == "Diagnostics":
    st.header("Diagnostics & bias checks")
    st.write("Quick diagnostics to inspect bias and distribution of scores.")

    # show global sample
    df_bias = global_adj if global_adj is not None else global_raw if global_raw is not None else None
    if df_bias is None or getattr(df_bias, "empty", True):
        st.info("No global CSV found (global_adjusted or global_raw).")
    else:
        st.subheader("Global sample")
        st.dataframe(df_bias.head(100), use_container_width=True)
        # pick a logit col
        log_cols = [c for c in df_bias.columns if 'logit' in c.lower()]
        if log_cols and 'disease_id' in df_bias.columns:
            log_col = log_cols[0]
            st.write(f"Using `{log_col}` to compute per-disease mean (bias).")
            bias = df_bias.groupby('disease_id')[log_col].mean().sort_values(ascending=False).head(30)
            bias_df = bias.reset_index().rename(columns={log_col: 'mean_logit'})
            st.table(bias_df)
            fig = px.bar(bias_df, x='mean_logit', y='disease_id', orientation='h', height=500)
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No logit column present to compute bias summary.")

    # probability histogram for per-drug top entries
    st.subheader("Probability distribution (per-drug top entries)")
    src = per_drug_adj if per_drug_adj is not None else per_drug_raw if per_drug_raw is not None else None
    if src is None or getattr(src, "empty", True):
        st.info("No per-drug CSV found to plot histogram.")
    else:
        pc = safe_get_prob_col(src)
        if pc:
            fig = px.histogram(src, x=pc, nbins=60, title="Per-drug probability distribution (top entries)", marginal="box")
            fig.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No probability column detected in per-drug CSV.")

st.markdown("<div style='color:#666;font-size:12px;margin-top:10px;'>App: fixed Streamlit UI — reads CSVs from model_out/; prefer bias-corrected per-drug outputs.</div>", unsafe_allow_html=True)
