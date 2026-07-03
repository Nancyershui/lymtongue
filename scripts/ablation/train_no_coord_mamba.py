import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import math
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

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
SEED = 42
seed_everything(SEED)

def patch_densenet121_to_14x14_with_mixed_dilation(
    features: nn.Module,
    pattern=(1, 2),
):
    if hasattr(features, "transition3") and hasattr(features.transition3, "pool"):
        features.transition3.pool = nn.Identity()
    else:
        raise RuntimeError("找不到 features.transition3.pool。")
    idx = 0
    for m in features.denseblock4.modules():
        if m.__class__.__name__ == "_DenseLayer" and hasattr(m, "conv2"):
            old = m.conv2
            if isinstance(old, nn.Conv2d) and old.kernel_size == (3, 3):
                d = pattern[idx % len(pattern)]
                idx += 1
                new = nn.Conv2d(
                    in_channels=old.in_channels,
                    out_channels=old.out_channels,
                    kernel_size=3,
                    stride=old.stride,
                    padding=(d, d),
                    dilation=(d, d),
                    groups=old.groups,
                    bias=(old.bias is not None),
                    padding_mode=old.padding_mode,
                )
                new.weight.data.copy_(old.weight.data)
                if old.bias is not None:
                    new.bias.data.copy_(old.bias.data)
                m.conv2 = new
    return features
data_root = "/path/to/tongue_dataset_split"
save_path = "checkpoints/densenet_A2_14x14_mixedDil_noCoordMamba.pth"
num_classes = 2
img_size = 224
batch_size = 16
num_epochs = 30
num_workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-4
weight_decay = 1e-4
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
print("train class_to_idx:", train_dataset.class_to_idx)
print("val   class_to_idx:", val_dataset.class_to_idx)
print("test  class_to_idx:", test_dataset.class_to_idx)
class_to_idx = train_dataset.class_to_idx
class_names = train_dataset.classes
lymphoma_idx = class_to_idx["lymphoma"]
normal_idx = class_to_idx["normal"]
print("lymphoma_idx =", lymphoma_idx)
print("normal_idx   =", normal_idx)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, worker_init_fn=seed_worker, generator=g
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, worker_init_fn=seed_worker, generator=g
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, worker_init_fn=seed_worker, generator=g
)

class DenseAblationTongue(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=1024,
        img_size=224,
        mixed_dilation_pattern=(1, 2),
        use_global_fusion=False,
    ):
        super().__init__()
        self.use_global_fusion = use_global_fusion
        base_model = models.densenet121(weights="DEFAULT")
        self.features = base_model.features
        self.features = patch_densenet121_to_14x14_with_mixed_dilation(
            self.features, pattern=mixed_dilation_pattern
        )
        self.d_model = d_model
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            feat = self.features(dummy)
            _, _, H, W = feat.shape
            self.num_tokens = H * W
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.d_model) * 0.02)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.d_model, num_classes)
        print(f">> [Backbone] DenseNet 输出 token 数 = {self.num_tokens} (期望 196 对应 14x14)")
        print(f">> [Backbone] denseblock4 mixed dilation pattern = {mixed_dilation_pattern}")
        print(">> [Ablation] Coord2D-Mamba 已移除，采用直接 token pooling 分类。")

    def forward(self, x):
        feat = self.features(x)
        feat = F.relu(feat, inplace=True)
        if self.use_global_fusion:
            g = F.avg_pool2d(feat, kernel_size=2, stride=2)
            g = F.interpolate(g, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat + g
        b, c, h, w = feat.shape
        x = feat.view(b, c, h * w).permute(0, 2, 1)
        if x.size(1) != self.pos_embedding.size(1):
            L0 = self.pos_embedding.size(1)
            s0 = int(math.sqrt(L0))
            pos = self.pos_embedding.transpose(1, 2).view(1, self.d_model, s0, s0)
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
            pos = pos.view(1, self.d_model, h * w).transpose(1, 2)
            x = x + pos
        else:
            x = x + self.pos_embedding
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.classifier(x)
        return out
model = DenseAblationTongue(
    num_classes=num_classes,
    img_size=img_size,
    mixed_dilation_pattern=(1, 2),
    use_global_fusion=False,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs, eta_min=1e-6
)
criterion = nn.CrossEntropyLoss()

def compute_auc_for_lymphoma(labels, probs, lymphoma_idx):
    binary_labels = np.array([1 if y == lymphoma_idx else 0 for y in labels])
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    return float(roc_auc_score(binary_labels, probs))

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, leave=False, desc="Training")
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
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
def evaluate_auc(model, loader, positive_class_idx):
    model.eval()
    probs, gts = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        p = torch.softmax(outputs, dim=1)[:, positive_class_idx]
        probs.extend(p.cpu().numpy())
        gts.extend(labels.numpy())
    return compute_auc_for_lymphoma(gts, probs, positive_class_idx)
best_auc = 0.0
print(f"\nStart Training Ablation: DenseNet + (A2)14x14 + mixedDil, w/o Coord2D-Mamba [Seed={SEED}]...")
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
    val_auc = evaluate_auc(model, val_loader, lymphoma_idx)
    current_lr_cnn = optimizer.param_groups[0]["lr"]
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {train_loss:.4f} | Val AUC(lymphoma): {val_auc:.4f} | LR(CNN): {current_lr_cnn:.1e}")
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), save_path)
        print(f"==> Best Model Saved (AUC: {best_auc:.4f})")
print(f"\nBest Val AUC (lymphoma): {best_auc:.4f}")
print("\nTesting model...")
model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
y_true, y_pred, y_probs = [], [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, lymphoma_idx]
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_probs.extend(probs.cpu().numpy())
test_auc = compute_auc_for_lymphoma(y_true, y_probs, lymphoma_idx)
print(f"\nFinal Test AUC (lymphoma as positive class): {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
