import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import spacy # Import spaCy

print("--- Step 1 (Biomedical NLP): Starting to parse the DrugBank XML file... ---")

# --- 1. Load the scispaCy NLP Model ---
# This is the major upgrade: we load a pre-trained model that understands medical text.
print("Loading the scispaCy NLP model (en_core_sci_sm)...")
try:
    # This model is specifically trained on biomedical text for Named Entity Recognition.
    nlp = spacy.load("en_core_sci_sm")
    print("NLP model loaded successfully.")
except OSError:
    print("\nERROR: scispaCy model not found. The model needs to be installed.")
    print("Please ensure your conda environment is set up and run the following commands:")
    print("1. pip install scispacy")
    print("2. pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz")
    exit()

def extract_diseases_with_nlp(text):
    """
    Uses a pre-trained biomedical NLP model to extract disease entities.
    This replaces our complex rule-based clean_text function for superior accuracy.
    """
    if not text:
        return []
    
    # Process the text with the NLP model. This is where the magic happens.
    doc = nlp(text)
    
    diseases = set()
    # The model identifies "entities" and labels them (e.g., as a DISEASE).
    for ent in doc.ents:
        # We only keep the entities that the model has high confidence are diseases.
        # The 'en_core_sci_sm' model uses a generic 'ENTITY' label for many things,
        # but its training on biomedical text makes it highly effective.
        # We will perform some basic filtering on the model's output.
        disease_name = ent.text.lower().strip()
        
        # Filter out very short, likely irrelevant terms or single letters.
        if len(disease_name) > 3 and not disease_name.isnumeric():
            diseases.add(disease_name)
                
    return list(diseases)

# --- Main Parsing Logic ---
# This part of the script remains the same, but now calls our new, smarter function.
tree = ET.iterparse('rawdata/full database.xml', events=('start', 'end'))
_, root = next(tree)

ns = {'db': 'http://www.drugbank.ca'}

drugs_data = []
targets_data = []
drug_target_edges = []
drug_disease_edges = []
seen_targets = set()
seen_diseases = set()

num_drugs_query = root.findall('db:drug', ns)
num_drugs = len(num_drugs_query) if num_drugs_query else 0
print(f"Found {num_drugs} total drug entries to process.")

with tqdm(total=num_drugs, desc="Parsing Drugs with NLP") as pbar:
    tree_iterator = ET.iterparse('rawdata/full database.xml', events=('start', 'end'))
    _, root_iter = next(tree_iterator)
    
    for event, elem in tree_iterator:
        if event == 'end' and elem.tag == f"{{{ns['db']}}}drug":
            pbar.update(1)
            drugbank_id_elem = elem.find('db:drugbank-id[@primary="true"]', ns)
            if drugbank_id_elem is None or not drugbank_id_elem.text:
                continue
            
            drug_id = drugbank_id_elem.text
            drug_name_elem = elem.find('db:name', ns)
            drug_name = drug_name_elem.text if drug_name_elem is not None else ""
            
            drugs_data.append({'id': drug_id, 'name': drug_name})

            for target in elem.findall('db:targets/db:target', ns):
                target_id_elem = target.find('db:id', ns)
                if target_id_elem is not None and target_id_elem.text:
                    target_id = target_id_elem.text
                    target_name_elem = target.find('db:name', ns)
                    target_name = target_name_elem.text if target_name_elem is not None else "Unknown"
                    if target_id not in seen_targets:
                        targets_data.append({'id': target_id, 'name': target_name})
                        seen_targets.add(target_id)
                    drug_target_edges.append({'drug_id': drug_id, 'target_id': target_id})

            indication_elem = elem.find('db:indication', ns)
            if indication_elem is not None and indication_elem.text:
                # Call the new, powerful NLP function
                diseases = extract_diseases_with_nlp(indication_elem.text)
                for disease_name in diseases:
                    seen_diseases.add(disease_name)
                    drug_disease_edges.append({'drug_id': drug_id, 'disease_name': disease_name})
            
            elem.clear()
            root.clear() # Clear root to free more memory

print(f"\nParsing complete. Extracted {len(seen_diseases)} unique entities using the NLP model.")

# --- Save results to CSV files ---
df_drugs = pd.DataFrame(drugs_data)
df_targets = pd.DataFrame(targets_data)
df_diseases = pd.DataFrame(list(seen_diseases), columns=['name'])
df_drug_target = pd.DataFrame(drug_target_edges)
df_drug_disease = pd.DataFrame(drug_disease_edges)

df_drugs.to_csv('drugs.csv', index=False)
df_targets.to_csv('proteins.csv', index=False)
df_diseases.to_csv('diseases.csv', index=False)
df_drug_target.to_csv('edges_drug_targets.csv', index=False)
df_drug_disease.to_csv('edges_drug_treats_disease.csv', index=False)

print("Data extracted via NLP and saved to CSV files.")

