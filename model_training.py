import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from neo4j import GraphDatabase
import pandas as pd
from tqdm import tqdm
import pickle

# --- Neo4j Connection Details ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# --- Model & Training Configuration ---
EMBEDDING_SIZE = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
DROPOUT_RATE = 0.5

# --- 1. Fetch Data from Neo4j ---
print("--- Step 3: Starting GNN Model Training ---")
print("Connecting to Neo4j to fetch graph data...")

try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        drugs_query = "MATCH (d:Drug) RETURN d.id AS id"
        proteins_query = "MATCH (p:Protein) RETURN p.id AS id"
        diseases_query = "MATCH (dis:Disease) RETURN dis.name AS id"
        
        drugs = pd.DataFrame([dict(record) for record in session.run(drugs_query)])
        proteins = pd.DataFrame([dict(record) for record in session.run(proteins_query)])
        diseases = pd.DataFrame([dict(record) for record in session.run(diseases_query)])

    drug_map = {id: i for i, id in enumerate(drugs['id'])}
    protein_map = {id: i for i, id in enumerate(proteins['id'])}
    disease_map = {id: i for i, id in enumerate(diseases['id'])}
    
    with driver.session() as session:
        targets_query = "MATCH (d:Drug)-[:TARGETS]->(p:Protein) RETURN d.id AS source, p.id AS target"
        treats_query = "MATCH (d:Drug)-[:TREATS]->(dis:Disease) RETURN d.id AS source, dis.name AS target"
        
        targets_edges_df = pd.DataFrame([dict(record) for record in session.run(targets_query)])
        treats_edges_df = pd.DataFrame([dict(record) for record in session.run(treats_query)])
    
    driver.close()
    print("Successfully fetched data from Neo4j.")

except Exception as e:
    print(f"ERROR: Could not fetch data from Neo4j. Details: {e}")
    exit()

# --- 2. Build PyTorch Geometric HeteroData Object ---
data = HeteroData()
data['drug'].x = torch.eye(len(drug_map))
data['protein'].x = torch.eye(len(protein_map))
data['disease'].x = torch.eye(len(disease_map))

source = [drug_map[id] for id in targets_edges_df['source']]
target = [protein_map[id] for id in targets_edges_df['target']]
data['drug', 'targets', 'protein'].edge_index = torch.tensor([source, target])

source = [drug_map[id] for id in treats_edges_df['source']]
target = [disease_map[id] for id in treats_edges_df['target']]
data['drug', 'treats', 'disease'].edge_index = torch.tensor([source, target])

data = ToUndirected()(data)
print("PyG HeteroData object created:")
print(data)

# --- 3. Prepare Data for Link Prediction ---
transform = RandomLinkSplit(
    is_undirected=True, num_val=0.1, num_test=0.1,
    neg_sampling_ratio=1.0,
    edge_types=[('drug', 'treats', 'disease')],
    rev_edge_types=[('disease', 'rev_treats', 'drug')]
)
train_data, val_data, test_data = transform(data)
print("\nData split for link prediction:")
print(f"Train data: {train_data['drug', 'treats', 'disease'].edge_label.shape[0]} edges")
print(f"Val data:   {val_data['drug', 'treats', 'disease'].edge_label.shape[0]} edges")
print(f"Test data:  {test_data['drug', 'treats', 'disease'].edge_label.shape[0]} edges")

# --- 4. Define the GNN Model and Link Predictor ---
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class EdgeClassifier(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, z_src, z_dst):
        edge_embedding = torch.cat([z_src, z_dst], dim=-1)
        x = self.lin1(edge_embedding).relu()
        x = self.dropout(x)
        x = self.lin2(x)
        return x.view(-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, dropout):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, dropout)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.classifier = EdgeClassifier(hidden_channels, dropout)

    def forward(self, data, edge_label_index):
        z = self.encoder(data.x_dict, data.edge_index_dict)
        z_src = z['drug'][edge_label_index[0]]
        z_dst = z['disease'][edge_label_index[1]]
        return self.classifier(z_src, z_dst)

model = Model(hidden_channels=EMBEDDING_SIZE, dropout=DROPOUT_RATE)

# --- 5. Train the Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")
model = model.to(device)
train_data = train_data.to(device)
val_data = val_data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train():
    model.train()
    optimizer.zero_grad()
    edge_label_index = train_data['drug', 'treats', 'disease'].edge_label_index
    edge_label = train_data['drug', 'treats', 'disease'].edge_label
    out = model(train_data, edge_label_index)
    loss = F.binary_cross_entropy_with_logits(out, edge_label.float())
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data_split):
    from sklearn.metrics import roc_auc_score
    model.eval()
    edge_label_index = data_split['drug', 'treats', 'disease'].edge_label_index
    edge_label = data_split['drug', 'treats', 'disease'].edge_label
    out = model(data_split, edge_label_index)
    preds = out.sigmoid()
    return roc_auc_score(edge_label.cpu().numpy(), preds.cpu().numpy())

print("Starting training...")
for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Training Epochs"):
    loss = train()
    if epoch % 10 == 0:
        train_auc = test(train_data)
        val_auc = test(val_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')

print("Training finished.")

# --- 6. Save the model and final embeddings ---
model.eval()
with torch.no_grad():
    data = data.to(device)
    final_embeddings = model.encoder(data.x_dict, data.edge_index_dict)

for node_type, embedding in final_embeddings.items():
    torch.save(embedding.cpu(), f'embeddings_{node_type}.pt')
    print(f"Saved embeddings for node type '{node_type}' to embeddings_{node_type}.pt")

torch.save(model.state_dict(), 'trained_model.pt')
print("Saved trained model to trained_model.pt")

# --- 7. Save the mappings for later use ---
with open('mappings.pkl', 'wb') as f:
    pickle.dump({'drug': drug_map, 'disease': disease_map}, f)
print("Saved ID/name mappings to mappings.pkl")

