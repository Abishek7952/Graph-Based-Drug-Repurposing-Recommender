from neo4j import GraphDatabase
import os

# --- Neo4j Connection Details ---
# Replace with your Neo4j AuraDB URI or local bolt address
NEO4J_URI = "bolt://localhost:7687" 
NEO4J_USER = "neo4j"
# !!! IMPORTANT: Using the password you provided for the project !!!
NEO4J_PASSWORD = "12345678"

class Neo4jUploader:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("Attempting to connect to Neo4j database...")
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("Successfully connected to Neo4j database.")
        except Exception as e:
            print(f"ERROR: Could not connect to Neo4j. Please check your credentials and that the database is running. Error details: {e}")
            self.driver = None # Invalidate driver on connection failure

    def close(self):
        if self.driver:
            self.driver.close()

    def run_query(self, query, message):
        if not self.driver:
            print("Cannot run query, driver is not connected.")
            return

        with self.driver.session() as session:
            try:
                session.run(query)
                print(f"SUCCESS: {message}")
            except Exception as e:
                print(f"ERROR executing query for '{message}': {e}")

    def upload_graph(self):
        if not self.driver:
            print("Cannot upload graph, not connected to database.")
            return
            
        print("\n--- Step 2: Starting Neo4j Graph Construction ---")

        # 1. Create uniqueness constraints for faster loading and data integrity
        # This ensures you don't create duplicate nodes for the same entity.
        self.run_query(
            "CREATE CONSTRAINT drug_id_unique IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE",
            "Created uniqueness constraint for Drug nodes."
        )
        self.run_query(
            "CREATE CONSTRAINT protein_id_unique IF NOT EXISTS FOR (p:Protein) REQUIRE p.id IS UNIQUE",
            "Created uniqueness constraint for Protein nodes."
        )
        self.run_query(
            "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (dis:Disease) REQUIRE dis.name IS UNIQUE",
            "Created uniqueness constraint for Disease nodes."
        )

        # 2. Load Drug nodes
        self.run_query(
            "LOAD CSV WITH HEADERS FROM 'file:///drugs.csv' AS row MERGE (d:Drug {id: row.id, name: row.name})",
            "Loaded all Drug nodes."
        )

        # 3. Load Protein (Target) nodes
        self.run_query(
            "LOAD CSV WITH HEADERS FROM 'file:///proteins.csv' AS row MERGE (p:Protein {id: row.id, name: row.name})",
            "Loaded all Protein nodes."
        )

        # 4. Load Disease nodes
        self.run_query(
            "LOAD CSV WITH HEADERS FROM 'file:///diseases.csv' AS row MERGE (dis:Disease {name: row.name})",
            "Loaded all Disease nodes."
        )

        # 5. Create [:TARGETS] relationships
        self.run_query(
            """
            LOAD CSV WITH HEADERS FROM 'file:///edges_drug_targets.csv' AS row
            MATCH (d:Drug {id: row.drug_id})
            MATCH (p:Protein {id: row.target_id})
            MERGE (d)-[:TARGETS]->(p)
            """,
            "Created all Drug-TARGETS->Protein relationships."
        )

        # 6. Create [:TREATS] relationships
        self.run_query(
            """
            LOAD CSV WITH HEADERS FROM 'file:///edges_drug_treats_disease.csv' AS row
            MATCH (d:Drug {id: row.drug_id})
            MATCH (dis:Disease {name: row.disease_name})
            MERGE (d)-[:TREATS]->(dis)
            """,
            "Created all Drug-TREATS->Disease relationships."
        )
        
        print("\n--- Step 2 complete. Your biomedical knowledge graph is now in Neo4j! ---")


if __name__ == "__main__":
    uploader = Neo4jUploader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    uploader.upload_graph()
    uploader.close()

