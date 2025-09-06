#!/usr/bin/env python3
"""
xml_to_csv.py - Production DrugBank XML -> CSV parser (scispaCy-only, start/end parsing)

Usage:
  python xml_to_csv.py --xml "rawdata/full database.xml" --out "clean_output"
"""
import os, sys, re, json, argparse
from collections import Counter
import pandas as pd
from lxml import etree
import spacy

try:
    from fuzzywuzzy import fuzz
    FUZZY_OK = True
except Exception:
    FUZZY_OK = False

MODEL_NAME = "en_ner_bc5cdr_md"
REQUIRED_LABEL = "DISEASE"

DEFAULT_XML_PATH = "rawdata/full database.xml"
DEFAULT_OUT = "clean_output"
DEFAULT_MIN_SUPPORT = 1
DEFAULT_MIN_ENTITY_LEN = 4
DEFAULT_FUZZY_THRESHOLD = 88

def local_name(tag):
    return etree.QName(tag).localname if tag else ""

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9\s\-\+/]", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    s = re.sub(r"\b(disease|syndrome|disorder|condition|infection|cancer|tumor|tumour)\b$", "", s).strip()
    return s

def keep_entity(t: str, min_len: int) -> bool:
    if not t or not isinstance(t, str): return False
    t = t.strip()
    if t.isnumeric(): return False
    if len(t) < min_len: return False
    if t in {"disease","disorder","syndrome","infection","condition"}: return False
    if re.fullmatch(r"[-\s]+", t): return False
    return True

def fuzzy_cluster(names, threshold=88):
    if not FUZZY_OK: return {n: n for n in names}
    names = list(dict.fromkeys(names))
    clusters = []
    for n in names:
        placed = False
        for c in clusters:
            if fuzz.token_sort_ratio(c[0], n) >= threshold:
                c.append(n); placed = True; break
        if not placed: clusters.append([n])
    mapping = {}
    for c in clusters:
        canon = sorted(c, key=lambda x:(len(x),x))[0]
        for m in c: mapping[m] = canon
    return mapping

def parse_drugbank(xml_path, out_dir=DEFAULT_OUT,
                   min_support=DEFAULT_MIN_SUPPORT,
                   fuzzy_threshold=DEFAULT_FUZZY_THRESHOLD,
                   min_entity_len=DEFAULT_MIN_ENTITY_LEN,
                   cluster_diseases=True):
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)

    try:
        nlp = spacy.load(MODEL_NAME)
    except Exception as e:
        print(f"ERROR: Could not load model '{MODEL_NAME}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded scispaCy model '{MODEL_NAME}'. Starting parse...")

    drugs, targets, drug_target_edges, drug_disease_edges = [], {}, [], []
    disease_counter = Counter()

    context = etree.iterparse(xml_path, events=("start","end"), recover=True, huge_tree=True)

    in_drug = False
    drug = {}
    texts = []

    for event, elem in context:
        ln = local_name(elem.tag).lower()

        if event == "start" and ln == "drug":
            in_drug = True
            drug = {"id": None, "name": None, "targets": []}
            texts = []

        elif event == "end" and ln == "drug":
            if drug.get("id"):
                drugs.append({"id": drug["id"], "name": drug["name"] or drug["id"]})
                for tid,tname,uniprot in drug["targets"]:
                    if tid not in targets:
                        targets[tid] = {"id": tid, "name": tname, "uniprot": uniprot}
                    drug_target_edges.append({"drug_id": drug["id"], "target_id": tid})

                for block in texts[:3]:
                    doc = nlp(block)
                    for ent in doc.ents:
                        if ent.label_.upper() == REQUIRED_LABEL:
                            norm = normalize_text(ent.text)
                            if keep_entity(norm, min_entity_len):
                                disease_counter[norm]+=1
                                drug_disease_edges.append({"drug_id": drug["id"], "disease_name": norm})

            in_drug = False
            elem.clear()

        elif in_drug:
            if ln == "drugbank-id" and elem.get("primary") == "true":
                txt = "".join(elem.itertext()).strip()
                if txt: drug["id"] = txt
            elif ln == "drugbank-id" and not drug.get("id"):
                txt = "".join(elem.itertext()).strip()
                if txt: drug["id"] = txt
            elif ln == "name" and not drug.get("name"):
                txt = "".join(elem.itertext()).strip()
                if txt: drug["name"] = txt
            elif ln == "target":
                tid, tname, uniprot = None, None, None
                for sub in elem.iter():
                    lns = local_name(sub.tag).lower()
                    if lns == "id": tid = "".join(sub.itertext()).strip()
                    if lns == "name": tname = "".join(sub.itertext()).strip()
                    if lns in {"xref","xrefs"}:
                        res, xid = None, None
                        for xx in sub.iter():
                            lx = local_name(xx.tag).lower()
                            if lx == "resource" and xx.text: res = xx.text.lower()
                            if lx == "id" and xx.text: xid = xx.text.strip()
                        if res and "uniprot" in res and xid: uniprot = xid
                if tid:
                    drug["targets"].append((tid,tname,uniprot))
                elif tname:
                    tmpid = f"name:{tname}"
                    drug["targets"].append((tmpid,tname,uniprot))
            elif ln in {"indication","indications","description","pharmacology"}:
                txt = "".join(elem.itertext()).strip()
                if txt: texts.append(txt)

    # Build DataFrames
    df_drugs=pd.DataFrame(drugs)
    df_targets=pd.DataFrame(list(targets.values()))
    df_drug_target=pd.DataFrame(drug_target_edges)
    df_drug_disease=pd.DataFrame(drug_disease_edges)

    # cluster and filter
    if not df_drug_disease.empty:
        names=list(dict.fromkeys(df_drug_disease['disease_name'].tolist()))
        if cluster_diseases and FUZZY_OK and names:
            mapping=fuzzy_cluster(names,threshold=fuzzy_threshold)
            df_drug_disease['disease_canon']=df_drug_disease['disease_name'].map(lambda x:mapping.get(x,x))
        else:
            df_drug_disease['disease_canon']=df_drug_disease['disease_name']
        support=df_drug_disease.groupby('disease_canon').size().to_dict()
        keep={d for d,c in support.items() if c>=min_support}
        df_drug_disease=df_drug_disease[df_drug_disease['disease_canon'].isin(keep)].copy()
        df_drug_disease=df_drug_disease.drop_duplicates(subset=['drug_id','disease_canon'])
        df_diseases=pd.DataFrame(sorted(list(keep)),columns=['name'])
    else:
        df_diseases=pd.DataFrame(columns=['name'])

    os.makedirs(out_dir, exist_ok=True)
    df_drugs.to_csv(os.path.join(out_dir,'drugs.csv'),index=False)
    df_targets.to_csv(os.path.join(out_dir,'proteins.csv'),index=False)
    df_diseases.to_csv(os.path.join(out_dir,'diseases.csv'),index=False)
    df_drug_target.to_csv(os.path.join(out_dir,'edges_drug_targets.csv'),index=False)
    if not df_drug_disease.empty:
        df_drug_disease[['drug_id','disease_canon']].rename(columns={'disease_canon':'disease_name'}).to_csv(os.path.join(out_dir,'edges_drug_treats_disease.csv'),index=False)
    else:
        pd.DataFrame(columns=['drug_id','disease_name']).to_csv(os.path.join(out_dir,'edges_drug_treats_disease.csv'),index=False)

    diagnostics={
        "num_drugs":int(df_drugs.shape[0]),
        "num_targets":int(df_targets.shape[0]),
        "num_drug_target_edges":int(df_drug_target.shape[0]),
        "num_drug_disease_edges":int(df_drug_disease.shape[0]),
        "top_20_disease_mentions":disease_counter.most_common(20)
    }
    with open(os.path.join(out_dir,'diagnostics.json'),'w') as f: json.dump(diagnostics,f,indent=2)
    print("Parsing complete. Wrote CSVs to:", os.path.abspath(out_dir))
    print(json.dumps(diagnostics, indent=2))
    return diagnostics

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, default=DEFAULT_XML_PATH)
    parser.add_argument('--out', type=str, default=DEFAULT_OUT)
    parser.add_argument('--min_support', type=int, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument('--fuzzy_threshold', type=int, default=DEFAULT_FUZZY_THRESHOLD)
    parser.add_argument('--min_entity_len', type=int, default=DEFAULT_MIN_ENTITY_LEN)
    parser.add_argument('--no-cluster', action='store_true')
    args=parser.parse_args()
    parse_drugbank(args.xml, args.out, min_support=args.min_support, fuzzy_threshold=args.fuzzy_threshold, min_entity_len=args.min_entity_len, cluster_diseases=(not args.no_cluster))
