import torch
import pandas as pd
import pickle
from tqdm import tqdm
import heapq

# --- Import Model Architecture from training script ---
# Make sure '03_train_model.py' is in the same directory
from model_training import Model

# --- Configuration ---
TOP_K = 20
EMBEDDING_SIZE = 128
DROPOUT_RATE = 0.5

print("--- Step 4: Generating Drug Repurposing Recommendations ---")

# --- 1. Load Everything ---
print("Loading model, embeddings, and mappings...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

try:
    with open('mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    drug_map = mappings['drug']
    disease_map = mappings['disease']
except FileNotFoundError:
    print("ERROR: 'mappings.pkl' not found. Please run '03_train_model.py' first.")
    exit()

rev_drug_map = {v: k for k, v in drug_map.items()}
rev_disease_map = {v: k for k, v in disease_map.items()}

try:
    drug_embeddings = torch.load('embeddings_drug.pt', map_location=device, weights_only=True)
    disease_embeddings = torch.load('embeddings_disease.pt', map_location=device, weights_only=True)
except FileNotFoundError as e:
    print(f"ERROR: Could not find embedding files. Details: {e}")
    exit()

model = Model(hidden_channels=EMBEDDING_SIZE, dropout=DROPOUT_RATE)
try:
    # Set strict=False as the saved state_dict might have extra keys from the hetero wrapper
    model.load_state_dict(torch.load('trained_model.pt', map_location=device, weights_only=True), strict=False)
    model.to(device)
    model.eval()
    print("Model, embeddings, and mappings loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not find 'trained_model.pt'. Details: {e}")
    exit()

# --- 2. Identify Candidate Pairs for Prediction ---
print("Identifying candidate drug-disease pairs...")
try:
    existing_treats_df = pd.read_csv('edges_drug_treats_disease.csv')
    existing_pairs = set(tuple(x) for x in existing_treats_df.to_numpy())
except FileNotFoundError as e:
    print(f"ERROR: Could not find 'edges_drug_treats_disease.csv'. Details: {e}")
    exit()

all_drugs = list(drug_map.keys())
all_diseases = list(disease_map.keys())

# --- 3. Predict Scores and keep only the Top K ---
print("Predicting scores for candidate pairs... (This may take a while, but is memory-efficient)")

# We use a min-heap to keep track of the top K results with the highest scores
top_k_heap = []
batch_size = 2048 

with torch.no_grad():
    # We create a generator instead of a list to save memory
    candidate_pair_generator = (
        (drug_id, disease_name)
        for drug_id in all_drugs
        for disease_name in all_diseases
        if (drug_id, disease_name) not in existing_pairs
    )
    
    total_candidates = len(all_drugs) * len(all_diseases) - len(existing_pairs)
    print(f"Found {total_candidates} potential new drug-disease pairs to evaluate.")


    batch = []
    for pair in tqdm(candidate_pair_generator, total=total_candidates, desc="Predicting"):
        batch.append(pair)
        if len(batch) == batch_size:
            drug_indices = [drug_map[p[0]] for p in batch]
            disease_indices = [disease_map[p[1]] for p in batch]

            z_drug = drug_embeddings[drug_indices]
            z_disease = disease_embeddings[disease_indices]
            
            scores = model.classifier(z_drug, z_disease).sigmoid()

            for i, p in enumerate(batch):
                score = scores[i].item()
                if len(top_k_heap) < TOP_K:
                    heapq.heappush(top_k_heap, (score, p[0], p[1]))
                else:
                    # If the new score is higher than the smallest score in the heap, replace it
                    heapq.heappushpop(top_k_heap, (score, p[0], p[1]))
            batch = [] # Reset the batch

# Process any remaining items in the last batch
if batch:
    drug_indices = [drug_map[p[0]] for p in batch]
    disease_indices = [disease_map[p[1]] for p in batch]
    z_drug = drug_embeddings[drug_indices]
    z_disease = disease_embeddings[disease_indices]
    scores = model.classifier(z_drug, z_disease).sigmoid()
    for i, p in enumerate(batch):
        score = scores[i].item()
        if len(top_k_heap) < TOP_K:
            heapq.heappush(top_k_heap, (score, p[0], p[1]))
        else:
            heapq.heappushpop(top_k_heap, (score, p[0], p[1]))

print("\nPrediction complete.")

# --- 4. Format Top Recommendations ---
results = [{'drug_id': drug, 'disease_name': disease, 'score': score} for score, drug, disease in sorted(top_k_heap, reverse=True)]
results_df = pd.DataFrame(results)

# --- 5. Save the Top Recommendations to a File ---
output_filename = 'top_20_recommendations.csv'
results_df.to_csv(output_filename, index=False)
print(f"Top {TOP_K} drug repurposing recommendations have been saved to '{output_filename}'.")
print("You can now open this file to view the results.")

