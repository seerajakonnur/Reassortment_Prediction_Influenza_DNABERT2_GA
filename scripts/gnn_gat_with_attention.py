import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, classification_report

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

# =========================================================
# Settings
# =========================================================
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "GAT_WITH_ATTENTION"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEGMENTS = ["PB2", "PB1", "PA", "HA", "NP", "NA", "MP", "NS"]

LABEL_MAP = {
    "Non-Reassortant": 0,
    "Reassortant": 1
}
INV_LABEL_MAP = {0: "Non-Reassortant", 1: "Reassortant"}

# =========================================================
# File paths
# =========================================================
train_embeddings_file = "../TRAIN_MARCH/train_march_embeddings.npy"
train_metadata_file = "../TRAIN_MARCH/train_march_metadata.csv"

same_embeddings_file = "../TEST_VALIDATION_MARCH_SAME_CLADE/test_same_clade_embeddings.npy"
same_metadata_file = "../TEST_VALIDATION_MARCH_SAME_CLADE/test_same_clade_metadata.csv"

other_embeddings_file = "../TEST_MARCH_FINAL_CLUSTER/test_march_2026_final_embeddings.npy"
other_metadata_file = "../TEST_MARCH_FINAL_CLUSTER/test_march_2026_final_metadata.csv"

# =========================================================
# Fully connected directed graph for 8 nodes
# =========================================================
edges = []
for i in range(8):
    for j in range(8):
        if i != j:
            edges.append([i, j])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, 56)

# =========================================================
# Helper: load dataset and build graphs
# =========================================================
def load_graph_dataset(embeddings_file, metadata_file):
    embeddings = np.load(embeddings_file)              # (N, 6144)
    meta = pd.read_csv(metadata_file)

    if embeddings.shape[0] != len(meta):
        raise ValueError(f"Mismatch between {embeddings_file} and {metadata_file}")

    embeddings = embeddings.reshape(embeddings.shape[0], 8, -1)   # (N, 8, 768)
    labels = meta["label"].map(LABEL_MAP).values

    data_list = []
    for i in range(len(meta)):
        x = torch.tensor(embeddings[i], dtype=torch.float)
        y = torch.tensor(labels[i], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)

    return data_list, meta

# =========================================================
# Load datasets
# =========================================================
train_data_list, train_meta = load_graph_dataset(train_embeddings_file, train_metadata_file)
same_data_list, same_meta = load_graph_dataset(same_embeddings_file, same_metadata_file)
other_data_list, other_meta = load_graph_dataset(other_embeddings_file, other_metadata_file)

print("Train graphs:", len(train_data_list))
print("Same-clade graphs:", len(same_data_list))
print("Other-clade graphs:", len(other_data_list))

# =========================================================
# Internal validation split inside training set only
# =========================================================
train_labels = train_meta["label"].map(LABEL_MAP).values
indices = np.arange(len(train_data_list))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.2,
    stratify=train_labels,
    random_state=42
)

train_graphs = [train_data_list[i] for i in train_idx]
val_graphs = [train_data_list[i] for i in val_idx]

train_meta_split = train_meta.iloc[train_idx].reset_index(drop=True)
val_meta_split = train_meta.iloc[val_idx].reset_index(drop=True)

print("Internal train graphs:", len(train_graphs))
print("Internal val graphs:", len(val_graphs))

# =========================================================
# Dataloaders
# =========================================================
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)

same_loader = DataLoader(same_data_list, batch_size=16, shuffle=False)
other_loader = DataLoader(other_data_list, batch_size=16, shuffle=False)

# =========================================================
# GAT model
# =========================================================
class GATClassifier(nn.Module):
    def __init__(self, in_channels=768, hidden_channels=128, heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=True, dropout=dropout)

        self.lin1 = nn.Linear(hidden_channels, 64)
        self.lin2 = nn.Linear(64, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, batch, return_attention=False):
        if return_attention:
            x, attn1 = self.gat1(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat1(x, edge_index)
            attn1 = None

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attn2 = self.gat2(x, edge_index, return_attention_weights=True)
        else:
            x = self.gat2(x, edge_index)
            attn2 = None

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x)

        if return_attention:
            return out, attn1, attn2
        return out

# =========================================================
# Device, model, optimizer
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = GATClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# =========================================================
# Train / evaluate
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []
    all_prob = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        prob = torch.softmax(out, dim=1)
        pred = out.argmax(dim=1)

        all_true.extend(batch.y.cpu().numpy())
        all_pred.extend(pred.cpu().numpy())
        all_prob.extend(prob.cpu().numpy())

    acc = accuracy_score(all_true, all_pred)
    mcc = matthews_corrcoef(all_true, all_pred)
    return acc, mcc, np.array(all_true), np.array(all_pred), np.array(all_prob)

# =========================================================
# Training loop
# =========================================================
num_epochs = 50
best_val_mcc = -1.0
best_epoch = -1
best_state = None

for epoch in range(1, num_epochs + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_acc, val_mcc, _, _, _ = evaluate(model, val_loader, device)

    if val_mcc > best_val_mcc:
        best_val_mcc = val_mcc
        best_epoch = epoch
        best_state = model.state_dict()

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val MCC: {val_mcc:.4f}")

print(f"\nBest validation MCC: {best_val_mcc:.4f} at epoch {best_epoch}")

# load best model
model.load_state_dict(best_state)

# save best model
model_path = os.path.join(OUTPUT_DIR, "gat_best_model.pt")
torch.save(model.state_dict(), model_path)
print("Saved best model to:", model_path)

# =========================================================
# Reporting
# =========================================================
def report_results(name, model, loader, device):
    acc, mcc, y_true, y_pred, y_prob = evaluate(model, loader, device)

    print(f"\n=== {name} RESULTS ===")
    print("Accuracy:", round(acc, 4))
    print("MCC:", round(mcc, 4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-Reassortant", "Reassortant"],
        zero_division=0
    ))

    return acc, mcc, y_true, y_pred, y_prob

val_results = report_results("INTERNAL VALIDATION", model, val_loader, device)
same_results = report_results("SAME-CLADE TEST", model, same_loader, device)
other_results = report_results("OTHER-CLADE TEST", model, other_loader, device)

# =========================================================
# Attention extraction
# =========================================================
@torch.no_grad()
def extract_attention(model, loader, meta_df, split_name, device):
    model.eval()

    rows = []
    sample_offset = 0

    for batch in loader:
        batch = batch.to(device)

        out, attn1, attn2 = model(batch.x, batch.edge_index, batch.batch, return_attention=True)

        # attn1 = (edge_index1, alpha1)
        # alpha1 shape: (num_edges_in_batch, heads)
        edge_idx1, alpha1 = attn1
        edge_idx2, alpha2 = attn2

        edge_idx1 = edge_idx1.cpu()
        alpha1 = alpha1.cpu().numpy()
        edge_idx2 = edge_idx2.cpu()
        alpha2 = alpha2.cpu().numpy()

        batch_vec = batch.batch.cpu().numpy()

        num_graphs_in_batch = batch.num_graphs

        # -------- layer 1 --------
        for e in range(edge_idx1.shape[1]):
            src_global = int(edge_idx1[0, e])
            dst_global = int(edge_idx1[1, e])

            graph_id_in_batch = src_global // 8
            src_local = src_global % 8
            dst_local = dst_global % 8

            sample_idx = sample_offset + graph_id_in_batch
            meta_row = meta_df.iloc[sample_idx]

            rows.append({
                "split": split_name,
                "layer": "gat1",
                "sample_idx": sample_idx,
                "sample_name": meta_row.get("sample_name", f"sample_{sample_idx}"),
                "label": meta_row.get("label", "Unknown"),
                "clade": meta_row.get("clade", "Unknown"),
                "source_group": meta_row.get("source_group", "Unknown"),
                "from_node": src_local,
                "to_node": dst_local,
                "from_segment": SEGMENTS[src_local],
                "to_segment": SEGMENTS[dst_local],
                "attention_mean": float(alpha1[e].mean()),
                **{f"head_{h+1}": float(alpha1[e, h]) for h in range(alpha1.shape[1])}
            })

        # -------- layer 2 --------
        for e in range(edge_idx2.shape[1]):
            src_global = int(edge_idx2[0, e])
            dst_global = int(edge_idx2[1, e])

            graph_id_in_batch = src_global // 8
            src_local = src_global % 8
            dst_local = dst_global % 8

            sample_idx = sample_offset + graph_id_in_batch
            meta_row = meta_df.iloc[sample_idx]

            # alpha2 may be shape (E,1)
            rows.append({
                "split": split_name,
                "layer": "gat2",
                "sample_idx": sample_idx,
                "sample_name": meta_row.get("sample_name", f"sample_{sample_idx}"),
                "label": meta_row.get("label", "Unknown"),
                "clade": meta_row.get("clade", "Unknown"),
                "source_group": meta_row.get("source_group", "Unknown"),
                "from_node": src_local,
                "to_node": dst_local,
                "from_segment": SEGMENTS[src_local],
                "to_segment": SEGMENTS[dst_local],
                "attention_mean": float(alpha2[e].mean()),
                **{f"head_{h+1}": float(alpha2[e, h]) for h in range(alpha2.shape[1])}
            })

        sample_offset += num_graphs_in_batch

    return pd.DataFrame(rows)

# =========================================================
# Extract attention for all splits
# =========================================================
train_loader_noshuffle = DataLoader(train_graphs, batch_size=16, shuffle=False)

train_attn = extract_attention(model, train_loader_noshuffle, train_meta_split, "train_internal", device)
val_attn = extract_attention(model, val_loader, val_meta_split, "val_internal", device)
same_attn = extract_attention(model, same_loader, same_meta.reset_index(drop=True), "same_clade_test", device)
other_attn = extract_attention(model, other_loader, other_meta.reset_index(drop=True), "other_clade_test", device)

all_attn = pd.concat([train_attn, val_attn, same_attn, other_attn], ignore_index=True)

# save detailed attention
detailed_csv = os.path.join(OUTPUT_DIR, "gat_attention_detailed.csv")
all_attn.to_csv(detailed_csv, index=False)
print("Saved detailed attention to:", detailed_csv)

# =========================================================
# Summaries
# =========================================================
# 1. by split, layer, label, segment pair
summary_label = (
    all_attn
    .groupby(["split", "layer", "label", "from_segment", "to_segment"], as_index=False)["attention_mean"]
    .mean()
    .sort_values(["split", "layer", "label", "attention_mean"], ascending=[True, True, True, False])
)

summary_label_csv = os.path.join(OUTPUT_DIR, "gat_attention_summary_by_label.csv")
summary_label.to_csv(summary_label_csv, index=False)

# 2. by split, layer, clade, segment pair
summary_clade = (
    all_attn
    .groupby(["split", "layer", "clade", "from_segment", "to_segment"], as_index=False)["attention_mean"]
    .mean()
    .sort_values(["split", "layer", "clade", "attention_mean"], ascending=[True, True, True, False])
)

summary_clade_csv = os.path.join(OUTPUT_DIR, "gat_attention_summary_by_clade.csv")
summary_clade.to_csv(summary_clade_csv, index=False)

# 3. by split, layer, source_group, segment pair
summary_source = (
    all_attn
    .groupby(["split", "layer", "source_group", "from_segment", "to_segment"], as_index=False)["attention_mean"]
    .mean()
    .sort_values(["split", "layer", "source_group", "attention_mean"], ascending=[True, True, True, False])
)

summary_source_csv = os.path.join(OUTPUT_DIR, "gat_attention_summary_by_source_group.csv")
summary_source.to_csv(summary_source_csv, index=False)

print("Saved summaries:")
print(summary_label_csv)
print(summary_clade_csv)
print(summary_source_csv)

# =========================================================
# Print top interactions for train/val/same-clade only
# =========================================================
trusted = all_attn[all_attn["split"].isin(["train_internal", "val_internal", "same_clade_test"])]

top_trusted = (
    trusted
    .groupby(["layer", "label", "from_segment", "to_segment"], as_index=False)["attention_mean"]
    .mean()
    .sort_values(["layer", "label", "attention_mean"], ascending=[True, True, False])
)

top_trusted_csv = os.path.join(OUTPUT_DIR, "gat_attention_top_trusted_sets.csv")
top_trusted.to_csv(top_trusted_csv, index=False)

print("Saved trusted-set summary to:", top_trusted_csv)

print("\nTop 20 interactions in trusted sets:")
print(top_trusted.head(20).to_string(index=False))
