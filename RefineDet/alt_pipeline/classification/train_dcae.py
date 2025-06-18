import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from dcae import DCAE
import logging
import pandas as pd

# Dummy dataset for illustration
class DummyClassificationDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=3):
        self.num_samples = num_samples
        self.num_classes = num_classes
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        image = torch.randn(3, 64, 64)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return image, label

# --- CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 3
EPOCHS = 20
BATCH_SIZE = 8
LOG_DIR = 'logs_dcae'
CKPT_DIR = 'checkpoints_dcae'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
METRICS_CSV = os.path.join(LOG_DIR, 'metrics_history.csv')

# --- LOGGING SETUP ---
logging.basicConfig(filename=os.path.join(LOG_DIR, 'train.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

# --- DATASET & DATALOADER ---
dataset = DummyClassificationDataset(num_samples=200, num_classes=NUM_CLASSES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- MODEL, LOSS, OPTIMIZER ---
model = DCAE(num_classes=NUM_CLASSES, feature_dim=128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
recon_criterion = nn.MSELoss()
class_criterion = nn.CrossEntropyLoss()

# --- RESUME SUPPORT ---
start_epoch = 0
metrics_history = []
resume_ckpt = None  # Set to checkpoint path to resume
if resume_ckpt and os.path.exists(resume_ckpt):
    checkpoint = torch.load(resume_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    if 'metrics_history' in checkpoint:
        metrics_history = checkpoint['metrics_history']
    logging.info(f"Resumed from checkpoint {resume_ckpt} at epoch {start_epoch}")

# --- TRAINING LOOP ---
for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        feat, recon, logits = model(images)
        recon_loss = recon_criterion(recon, images)
        class_loss = class_criterion(logits, labels)
        loss = recon_loss + class_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples
    # --- LOGGING ---
    logging.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
    # --- METRIC HISTORY ---
    metrics_history.append({'epoch': epoch+1, 'loss': avg_loss, 'accuracy': accuracy})
    pd.DataFrame(metrics_history).to_csv(METRICS_CSV, index=False)
    # --- CHECKPOINTING ---
    ckpt_path = os.path.join(CKPT_DIR, f'dcae_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics_history': metrics_history
    }, ckpt_path)
    logging.info(f"Saved checkpoint: {ckpt_path}")

print("DCAE training complete.")
logging.info("DCAE training complete.") 