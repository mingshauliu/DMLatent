import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm

# --- Set seed ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Dataset ---
from utils import CosmicWebPk 
from models import PkMLP  

# --- Load data ---
# Cosmic Web Power Spectrum Dataset
# This dataset computes the power spectrum for CDM/WDM maps and returns it as input features
# and the corresponding labels (0 for CDM, 1 for WDM).
# Assume cdm_data and wdm_data are loaded from .npy or h5

cdm_file='/n/netscratch/iaifi_lab/Lab/msliu/CMD/data/IllustrisTNG/Maps_Mtot_IllustrisTNG_LH_z=0.00.npy'
wdm_file='/n/netscratch/iaifi_lab/Lab/ccuestalazaro/DREAMS/Images/WDM/boxes/Maps_Mtot_IllustrisTNG_WDM_z=0.00.npy'

cdm_data = np.load(cdm_file)  # shape [N, H, W]
wdm_data = np.load(wdm_file)  # shape [N, H, W]

total_samples = min(len(cdm_data), len(wdm_data))
indices = list(range(total_samples))

random.shuffle(indices)
n_train = int(0.6 * total_samples)
n_val   = int(0.2 * total_samples)

train_dataset = CosmicWebPk(cdm_data, wdm_data, indices[:n_train])
val_dataset   = CosmicWebPk(cdm_data, wdm_data, indices[n_train:n_train+n_val])
test_dataset  = CosmicWebPk(cdm_data, wdm_data, indices[n_train+n_val:])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=64)
test_loader  = DataLoader(test_dataset, batch_size=64)

# --- Model and training ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Infer Pk dimensionality from one sample
sample_pk, _ = train_dataset[0]
input_dim = len(sample_pk)

model = PkMLP(input_dim).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# --- Training loop ---
best_val_acc = 0.0
patience_counter = 0
max_patience = 20
num_epochs = 80

print("Starting training...")
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze(1)  # Squeeze to match target shape
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

    train_acc = correct / total
    train_loss /= total

    # --- Validation ---
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += x.size(0)

    val_acc = correct / total
    val_loss /= total
    scheduler.step(val_acc)

    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_pk_model.pt")
        print(f"  -> New best model saved (Val Acc: {val_acc:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print("Early stopping triggered.")
            break

# --- Final evaluation ---
model.load_state_dict(torch.load("best_pk_model.pt"))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = (torch.sigmoid(model(x)) > 0.5).float()
        correct += (preds == y).sum().item()
        total += x.size(0)

test_acc = correct / total
print(f"Final Test Accuracy: {test_acc:.4f}")
