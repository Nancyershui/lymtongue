import os
import copy
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, classification_report




try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
    print(">> [Mamba]")
except ImportError:
    HAS_MAMBA = False
    print("!! [Warning]")




def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> [Seed]: {seed}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

SEED = 42
seed_everything(SEED)
g = torch.Generator()
g.manual_seed(SEED)




data_root = "/path/to/tongue_dataset_split"
save_path = "checkpoints/model.pth"

num_classes = 2
img_size = 224
batch_size = 16
num_epochs = 50
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr_backbone = 1e-5
lr_head = 2e-4
weight_decay = 1e-4
label_smoothing = 0.1

print(f"Using device: {device}")
os.makedirs("checkpoints", exist_ok=True)




train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=val_test_transform)
test_dataset  = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=val_test_transform)

print(f"Classes: {train_dataset.classes}")


pos_name = "lymphoma"
if pos_name not in train_dataset.classes:
    raise ValueError(f"Positive class '{pos_name}' not found in classes: {train_dataset.classes}")
pos_idx = train_dataset.classes.index(pos_name)
print(f"Positive class = {pos_name}, pos_idx = {pos_idx} (label=={pos_idx} -> y=1 for AUC)")

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g
)




class DenseMambaTongue(nn.Module):
    def __init__(self, num_classes=2, d_model=1024):
        super().__init__()

        base_model = models.densenet121(weights='DEFAULT')
        self.features = base_model.features
        self.d_model = d_model

        if HAS_MAMBA:
            self.mamba = Mamba(
                d_model=self.d_model,
                d_state=16,
                d_conv=4,
                expand=2
            )
        else:
            self.mamba = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )

        self.pos_embedding = nn.Parameter(torch.randn(1, 49, self.d_model) * 0.02)

        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.d_model, num_classes)

    def forward(self, x):
        x = self.features(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)
        x = x + self.pos_embedding

        if HAS_MAMBA:
            x_fwd = self.mamba(x)
            x_bwd = self.mamba(x.flip([1])).flip([1])
            x = x_fwd + x_bwd
        else:
            x, _ = self.mamba(x)

        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)

model = DenseMambaTongue(num_classes=num_classes).to(device)




backbone_params = list(map(id, model.features.parameters()))
head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

optimizer = torch.optim.AdamW([
    {'params': model.features.parameters(), 'lr': lr_backbone},
    {'params': head_params, 'lr': lr_head}
], weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)

criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)




def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, leave=False, desc="Training")

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    scheduler.step()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate_auc(model, loader):
    model.eval()
    probs, gts = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)

        p = torch.softmax(outputs, dim=1)[:, pos_idx].cpu().numpy()
        y = (labels.numpy() == pos_idx).astype(np.int64)

        probs.extend(p.tolist())
        gts.extend(y.tolist())

    try:
        return roc_auc_score(gts, probs)
    except Exception:
        return 0.0

best_auc = -1.0
print("\nStart Training V3 (Differential LR + Label Smoothing) [AUC fixed]...")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
    val_auc = evaluate_auc(model, val_loader)

    current_lr_cnn = optimizer.param_groups[0]['lr']
    current_lr_head = optimizer.param_groups[1]['lr']

    print(
        f"[Epoch {epoch+1}/{num_epochs}] "
        f"Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f} | "
        f"LR(CNN): {current_lr_cnn:.1e} | LR(Head): {current_lr_head:.1e}"
    )

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(f"==> Best Model Saved (AUC: {best_auc:.4f})")

print(f"\nBest Val AUC: {best_auc:.4f}")




print("\nTesting V3 model...")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()

y_true_auc, y_probs = [], []
y_true_report, y_pred = [], []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        outputs = model(imgs)

        probs = torch.softmax(outputs, dim=1)[:, pos_idx]
        preds = torch.argmax(outputs, dim=1)

        y_auc = (labels.numpy() == pos_idx).astype(np.int64)

        y_true_auc.extend(y_auc.tolist())
        y_probs.extend(probs.cpu().numpy().tolist())

        y_true_report.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

test_auc = roc_auc_score(y_true_auc, y_probs)
print(f"\nFinal Test AUC (positive={pos_name}): {test_auc:.4f}")

print(classification_report(
    y_true_report, y_pred,
    target_names=[str(c) for c in train_dataset.classes],
    digits=4
))
