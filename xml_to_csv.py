import xml.etree.ElementTree as ET
import pandas as pd
from tqdm import tqdm
import spacy
from collections import Counter

print("--- Step 1 (Final Definitive Version): Starting NLP Parsing ---")

# --- Configuration for Final Cleaning ---
# Using the purpose-built BC5CDR model, we can now filter for the precise "DISEASE" label.
VALID_ENTITY_LABELS = {"DISEASE"}

# --- 1. Load the scispaCy BC5CDR NLP Model ---
MODEL_NAME = "en_ner_bc5cdr_md"
print(f"Loading the scispaCy NLP model ({MODEL_NAME})...")
try:
    nlp = spacy.load(MODEL_NAME)
    print("NLP model loaded successfully.")
except OSError:
    print(f"\nERROR: scispaCy model '{MODEL_NAME}' not found.")
    print("Please run the following command in your 'drug-project' conda environment:")
    print("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz")
    exit()

def normalize_entity(text):
    """Performs final cleaning and normalization on an entity name."""
    text = text.lower().strip()
    # Simple plural handling
    if text.endswith('s') and len(text) > 3 and text[-2] not in 'is':
        return text[:-1]
    return text

# --- Main Parsing Logic ---
def count_drug_tags(filename):
    """Efficiently counts the number of <drug> tags for the progress bar."""
    count = 0
    try:
        for event, elem in ET.iterparse(filename, events=('end',)):
            if elem.tag == '{http://www.drugbank.ca}drug':
                count += 1
            elem.clear()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"Error reading XML during count: {e}")
        return 0
    return count

XML_FILE_PATH = 'rawdata/full database.xml'
num_drugs = count_drug_tags(XML_FILE_PATH)
print(f"Found {num_drugs} total drug entries to process.")

# --- Single Pass Data Extraction ---
print("\n--- Starting Data Extraction ---")
drugs_data = []
targets_data = []
drug_target_edges = []
drug_disease_edges = []
seen_targets = set()
all_diseases = set()
debug_count = 0

try:
    with open(XML_FILE_PATH, 'rb') as f:
        context = ET.iterparse(f, events=('end',))
        for event, elem in tqdm(context, total=num_drugs, desc="Parsing DrugBank XML"):
            if elem.tag == '{http://www.drugbank.ca}drug':
                ns = {'db': 'http://www.drugbank.ca'}
                
                drugbank_id_elem = elem.find('db:drugbank-id[@primary="true"]', ns)
                if drugbank_id_elem is None or not drugbank_id_elem.text:
                    elem.clear()
                    continue
                
                drug_id = drugbank_id_elem.text
                drug_name_elem = elem.find('db:name', ns)
                drug_name = drug_name_elem.text if drug_name_elem is not None else "Unknown"
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
                    doc = nlp(indication_elem.text)
                    
                    # Debug the first 5 drugs with entities
                    if debug_count < 5 and doc.ents:
                        print(f"\n--- DEBUG: Entities for Drug {drug_id} ---")
                        for ent in doc.ents:
                            print(f"  - Found Entity: '{ent.text}', Label: '{ent.label_}'")
                        debug_count += 1

                    for ent in doc.ents:
                        if ent.label_ in VALID_ENTITY_LABELS:
                            normalized_entity = normalize_entity(ent.text)
                            if len(normalized_entity) > 3 and not normalized_entity.isnumeric():
                                all_diseases.add(normalized_entity)
                                drug_disease_edges.append({'drug_id': drug_id, 'disease_name': normalized_entity})
                
                elem.clear()
except FileNotFoundError:
    print(f"ERROR: XML file not found at '{XML_FILE_PATH}'")
    exit()

# --- Save results to CSV files ---
print(f"\nParsing complete. Found {len(all_diseases)} unique, high-quality disease/symptom nodes.")

df_drugs = pd.DataFrame(drugs_data)
df_targets = pd.DataFrame(targets_data)
df_diseases = pd.DataFrame(list(all_diseases), columns=['name'])
df_drug_target = pd.DataFrame(drug_target_edges)
df_drug_disease = pd.DataFrame(drug_disease_edges).drop_duplicates()

if df_diseases.empty or df_drug_disease.empty:
    print("\nWARNING: No valid disease data was extracted.")
else:
    df_drugs.to_csv('drugs.csv', index=False)
    df_targets.to_csv('proteins.csv', index=False)
    df_diseases.to_csv('diseases.csv', index=False)
    df_drug_target.to_csv('edges_drug_targets.csv', index=False)
    df_drug_disease.to_csv('edges_drug_treats_disease.csv', index=False)
    print("Definitive scientific parsing complete. High-quality data saved to CSV files.")

