import xml.etree.ElementTree as ET
import pandas as pd
import re
from tqdm import tqdm

def clean_text(text, drug_name):
    """
    An even more advanced function to clean the indication text.
    Now removes the drug's own name to prevent self-reference.
    """
    if not text:
        return ""
    
    # Remove content in brackets (e.g., [FDA Label], [L12345])
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove the drug's own name (case-insensitive, whole word only)
    drug_name_pattern = r'\b' + re.escape(drug_name) + r'\b'
    text = re.sub(drug_name_pattern, '', text, flags=re.IGNORECASE)

    # Remove common introductory phrases (case-insensitive)
    phrases_to_remove = [
        r'is indicated for the treatment of', r'is indicated for', r'are indicated for',
        r'indicated for the treatment of', r'indicated for', r'for the treatment of',
        r'also indicated for', r'may be used off-label to treat', r'use with',
        r'in combination with', r'such as', r'management of', r'treatment of'
    ]
    for phrase in phrases_to_remove:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)

    # Remove special characters, leftover HTML, and extra whitespace
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s,-]', '', text) # Keep only useful characters
    text = re.sub(r'\s+', ' ', text).strip() # Consolidate whitespace
    return text

print("--- Step 1 (v3): Starting to parse the DrugBank XML with final cleaning enhancements ---")
print("This may take a few minutes...")

NS = {'db': 'http://www.drugbank.ca'}
XML_FILE_PATH = 'rawdata/full database.xml'

context = ET.iterparse(XML_FILE_PATH, events=('start', 'end'))
_, root = next(context)

drugs_data = []
targets_data = []
drug_target_edges = []
drug_disease_edges = []
seen_targets = set()
seen_diseases = set()

for event, elem in tqdm(context, desc="Parsing XML"):
    if event == 'end' and elem.tag == f"{{{NS['db']}}}drug":
        drugbank_id_elem = elem.find('db:drugbank-id[@primary="true"]', NS)
        if drugbank_id_elem is None:
            continue
        
        drug_id = drugbank_id_elem.text
        drug_name = elem.find('db:name', NS).text
        drugs_data.append({'id': drug_id, 'name': drug_name})

        for target in elem.findall('db:targets/db:target', NS):
            target_id_elem = target.find('db:id', NS)
            if target_id_elem is None or not target_id_elem.text:
                continue
            
            target_id = target_id_elem.text
            if target_id not in seen_targets:
                target_name = target.find('db:name', NS).text
                targets_data.append({'id': target_id, 'name': target_name})
                seen_targets.add(target_id)
            drug_target_edges.append({'drug_id': drug_id, 'target_id': target_id})

        indication_elem = elem.find('db:indication', NS)
        if indication_elem is not None and indication_elem.text:
            # Pass the drug_name to the cleaning function for removal
            cleaned_indication = clean_text(indication_elem.text, drug_name)
            
            # *** USE A MORE ROBUST SPLITTING PATTERN ***
            # Split by punctuation, 'and', 'or'
            diseases = re.split(r'\s*[,;.]\s*|\s+\band\b\s+|\s+\bor\b\s+', cleaned_indication)
            for disease in diseases:
                disease_name = disease.strip().lower()
                # Final check for validity
                if disease_name and len(disease_name) > 3 and not disease_name.isnumeric():
                    seen_diseases.add(disease_name)
                    drug_disease_edges.append({'drug_id': drug_id, 'disease_name': disease_name})
        
        root.clear()

print(f"\nParsing complete. Found {len(drugs_data)} drugs.")

df_drugs = pd.DataFrame(drugs_data)
df_targets = pd.DataFrame(targets_data)
df_diseases = pd.DataFrame(list(seen_diseases), columns=['name'])
df_drug_target = pd.DataFrame(drug_target_edges)
df_drug_disease = pd.DataFrame(drug_disease_edges)

print("Saving final, cleaned data to CSV files...")
df_drugs.to_csv('drugs.csv', index=False)
df_targets.to_csv('proteins.csv', index=False)
df_diseases.to_csv('diseases.csv', index=False)
df_drug_target.to_csv('edges_drug_targets.csv', index=False)
df_drug_disease.to_csv('edges_drug_treats_disease.csv', index=False)

print("--- Step 1 complete. High-quality CSV files are now ready for Neo4j. ---")

