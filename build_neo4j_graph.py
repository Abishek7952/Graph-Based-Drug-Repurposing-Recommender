#!/usr/bin/env python3
"""
build_neo4j_graph.py

Upload data_cleaning/final_*.csv files into Neo4j using the official Python driver.
This version has credentials and CSV directory preconfigured,
so you can run `python build_neo4j_graph.py` without extra arguments.
"""

import os
import pandas as pd
from neo4j import GraphDatabase

# --- Neo4j Connection Details ---
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"   # <--- change if you set a different password

# --- CSV Directory (where data_cleaning/final_*.csv files live) ---
CSV_DIR = "."   # current directory, adjust if needed

# --- Config ---
BATCH_SIZE = 1000

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

class Neo4jUploader:
    def __init__(self, uri, user, password):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            with self.driver.session() as s:
                s.run("RETURN 1")
            print(f"Connected to Neo4j at {uri} as {user}")
        except Exception as e:
            raise RuntimeError(f"Could not connect to Neo4j: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def safe_run(self, cypher, params=None):
        with self.driver.session() as session:
            try:
                session.run(cypher, params or {})
            except Exception as e:
                raise RuntimeError(f"Cypher failed: {e}\nQuery: {cypher}")

    def create_constraints(self):
        statements = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (dis:Disease) REQUIRE dis.name IS UNIQUE",
        ]
        for st in statements:
            try:
                self.safe_run(st)
            except Exception as e:
                print(f"[constraint ignored] {e}")

    def upload_nodes(self, df, label, id_col, extra_cols=None):
        if df.empty:
            print(f"No {label} nodes, skipping.")
            return
        print(f"Uploading {len(df)} {label} nodes...")
        rows = df[[id_col] + (extra_cols or [])].fillna("").to_dict(orient="records")
        for chunk in chunked(rows, BATCH_SIZE):
            cypher = f"UNWIND $rows AS r MERGE (n:{label} {{ {id_col}: r['{id_col}'] }})"
            if extra_cols:
                set_parts = [f"n.{c} = r['{c}']" for c in extra_cols]
                cypher += "\nSET " + ", ".join(set_parts)
            self.safe_run(cypher, {"rows": chunk})
        print(f"Done uploading {label} nodes.")

    def upload_targets(self, df):
        if df.empty:
            print("No TARGETS edges.")
            return
        print(f"Uploading {len(df)} TARGETS relationships...")
        rows = df[["drug_id","target_id"]].dropna().astype(str).to_dict(orient="records")
        for chunk in chunked(rows, BATCH_SIZE):
            cypher = """
            UNWIND $rows AS r
            MATCH (d:Drug {id: r.drug_id})
            MATCH (p:Protein {id: r.target_id})
            MERGE (d)-[:TARGETS]->(p)
            """
            self.safe_run(cypher, {"rows": chunk})
        print("Done uploading TARGETS.")

    def upload_treats(self, df):
        if df.empty:
            print("No TREATS edges.")
            return
        print(f"Uploading {len(df)} TREATS relationships...")
        rows = df[["drug_id","disease_name"]].dropna().astype(str).to_dict(orient="records")
        for chunk in chunked(rows, BATCH_SIZE):
            cypher = """
            UNWIND $rows AS r
            MATCH (d:Drug {id: r.drug_id})
            MATCH (dis:Disease {name: r.disease_name})
            MERGE (d)-[:TREATS]->(dis)
            """
            self.safe_run(cypher, {"rows": chunk})
        print("Done uploading TREATS.")

def main():
    # load CSVs
    f_drugs   = os.path.join(CSV_DIR, "data_cleaning/final_drugs.csv")
    f_prots   = os.path.join(CSV_DIR, "data_cleaning/final_proteins.csv")
    f_dis     = os.path.join(CSV_DIR, "data_cleaning/final_diseases.csv")
    f_edges_dt= os.path.join(CSV_DIR, "data_cleaning/final_edges_drug_targets.csv")
    f_edges_dd= os.path.join(CSV_DIR, "data_cleaning/final_edges_drug_treats_disease.csv")

    drugs = pd.read_csv(f_drugs, dtype=str).fillna("")
    prots = pd.read_csv(f_prots, dtype=str).fillna("")
    dis   = pd.read_csv(f_dis, dtype=str).fillna("")
    edges_dt = pd.read_csv(f_edges_dt, dtype=str).fillna("")
    edges_dd = pd.read_csv(f_edges_dd, dtype=str).fillna("")

    uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        uploader.create_constraints()
        uploader.upload_nodes(drugs, "Drug", "id", ["name"] if "name" in drugs.columns else None)
        uploader.upload_nodes(prots, "Protein", "id", ["name"] if "name" in prots.columns else None)
        uploader.upload_nodes(dis, "Disease", "name")
        uploader.upload_targets(edges_dt)
        uploader.upload_treats(edges_dd)
        print("\nGraph upload complete!")
    finally:
        uploader.close()

if __name__ == "__main__":
    main()
