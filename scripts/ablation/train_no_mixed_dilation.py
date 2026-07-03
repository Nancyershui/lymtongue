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
from einops import rearrange, repeat

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
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    except ImportError:
        causal_conv1d_fn, causal_conv1d_update = None, None
    HAS_MAMBA_KERNELS = True
    print(">> [Mamba]")
except ImportError:
    HAS_MAMBA_KERNELS = False
    print("!! [Warning]")

def patch_densenet121_to_14x14(features: nn.Module):
    if hasattr(features, "transition3") and hasattr(features.transition3, "pool"):
        features.transition3.pool = nn.Identity()
    else:
        raise RuntimeError("找不到 features.transition3.pool。")
    return features

class CoordAtt2D_for_Mamba(nn.Module):
    def __init__(self, inp, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        B, D, L = x.size()
        H_feat = int(math.sqrt(L))
        W_feat = int(math.sqrt(L))
        if H_feat * W_feat != L:
            return x
        x_2d = x.view(B, D, H_feat, W_feat)
        identity = x_2d
        _, _, h, w = x_2d.size()
        x_h = self.pool_h(x_2d)
        x_w = self.pool_w(x_2d).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out_2d = identity * a_h * a_w
        out = out_2d.view(B, D, L)
        return out

class CoordMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.coord_att = CoordAtt2D_for_Mamba(inp=self.d_inner, reduction=16)

    def forward(self, hidden_states, inference_params=None):
        batch, seqlen, dim = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        A = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x, dt, A, B, C, self.D.float(), z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = self.coord_att(y)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
data_root = "/path/to/tongue_dataset_split"
save_path = "checkpoints/densenet_mamba_coord_A2_14x14_noMixedDil.pth"
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

class DenseMambaTongue_NoMixedDil(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=1024,
        img_size=224,
        use_global_fusion=False,
    ):
        super().__init__()
        self.use_global_fusion = use_global_fusion
        base_model = models.densenet121(weights="DEFAULT")
        self.features = base_model.features
        self.features = patch_densenet121_to_14x14(self.features)
        self.d_model = d_model
        if HAS_MAMBA_KERNELS:
            self.mamba = CoordMamba(
                d_model=self.d_model, d_state=16, d_conv=4, expand=2, use_fast_path=False
            )
        else:
            self.mamba = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
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
        print(">> [Ablation] 保留 Coord2D-Mamba，移除 denseblock4 mixed dilation。")

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
        if HAS_MAMBA_KERNELS:
            x_fwd = self.mamba(x)
            x_bwd = self.mamba(x.flip([1])).flip([1])
            x = x_fwd + x_bwd
        else:
            x, _ = self.mamba(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        out = self.classifier(x)
        return out
model = DenseMambaTongue_NoMixedDil(
    num_classes=num_classes,
    img_size=img_size,
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
print(f"\nStart Training Coord-Mamba + A2(14x14), w/o Mixed Dilation [Seed={SEED}]...")
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
