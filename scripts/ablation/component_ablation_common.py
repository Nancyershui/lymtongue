import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None
    HAS_MAMBA_KERNELS = True
except ImportError:
    HAS_MAMBA_KERNELS = False
    causal_conv1d_fn = None
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", default="/path/to/tongue_dataset_split")
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def patch_densenet121_to_14x14(features):
    if hasattr(features, "transition3") and hasattr(features.transition3, "pool"):
        features.transition3.pool = nn.Identity()
    else:
        raise RuntimeError("Cannot find features.transition3.pool; torchvision may have changed.")
    return features

def patch_denseblock4_mixed_dilation(features, pattern=(1, 2)):
    idx = 0
    for module in features.denseblock4.modules():
        if module.__class__.__name__ == "_DenseLayer" and hasattr(module, "conv2"):
            old = module.conv2
            if isinstance(old, nn.Conv2d) and old.kernel_size == (3, 3):
                dilation = pattern[idx % len(pattern)]
                idx += 1
                new = nn.Conv2d(
                    in_channels=old.in_channels,
                    out_channels=old.out_channels,
                    kernel_size=3,
                    stride=old.stride,
                    padding=(dilation, dilation),
                    dilation=(dilation, dilation),
                    groups=old.groups,
                    bias=(old.bias is not None),
                    padding_mode=old.padding_mode,
                )
                new.weight.data.copy_(old.weight.data)
                if old.bias is not None:
                    new.bias.data.copy_(old.bias.data)
                module.conv2 = new
    return features

class CoordAtt2DForMamba(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid_channels = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1)

    def forward(self, x):
        batch, channels, length = x.size()
        side = int(math.sqrt(length))
        if side * side != length:
            return x
        x_2d = x.view(batch, channels, side, side)
        identity = x_2d
        _, _, h, w = x_2d.size()
        x_h = self.pool_h(x_2d)
        x_w = self.pool_w(x_2d).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out = identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()
        return out.view(batch, channels, length)

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
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -(self.dt_rank ** -0.5), self.dt_rank ** -0.5)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        a = repeat(torch.arange(1, self.d_state + 1, dtype=torch.float32), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(a))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.coord_att = CoordAtt2DForMamba(self.d_inner)

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")
        a = -torch.exp(self.A_log.float())
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation="silu",
            )
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, b, c = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        b = rearrange(b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        c = rearrange(c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(
            x,
            dt,
            a,
            b,
            c,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = self.coord_att(y)
        return self.out_proj(rearrange(y, "b d l -> b l d"))

class ComponentAblationDenseNet(nn.Module):
    def __init__(
        self,
        num_classes=2,
        img_size=224,
        use_14x14=False,
        use_mixed_dilation=False,
        use_coord_mamba=False,
        d_model=1024,
    ):
        super().__init__()
        self.use_coord_mamba = use_coord_mamba
        self.d_model = d_model
        base_model = models.densenet121(weights="DEFAULT")
        self.features = base_model.features
        if use_14x14:
            self.features = patch_densenet121_to_14x14(self.features)
        if use_mixed_dilation:
            self.features = patch_denseblock4_mixed_dilation(self.features, pattern=(1, 2))
        with torch.no_grad():
            feat = self.features(torch.zeros(1, 3, img_size, img_size))
            _, _, h, w = feat.shape
            self.num_tokens = h * w
        if use_coord_mamba:
            if HAS_MAMBA_KERNELS:
                self.sequence_model = CoordMamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            else:
                self.sequence_model = nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model // 2,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                )
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
            self.norm = nn.LayerNorm(d_model)
        elif use_14x14:
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, d_model) * 0.02)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.pos_embedding = None
            self.norm = None
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(d_model, num_classes)
        print(
            ">> [Ablation] "
            f"14x14={use_14x14}, mixed_dilation={use_mixed_dilation}, "
            f"coord_mamba={use_coord_mamba}, tokens={self.num_tokens}"
        )

    def _add_pos_embedding(self, tokens, h, w):
        if self.pos_embedding is None:
            return tokens
        if tokens.size(1) == self.pos_embedding.size(1):
            return tokens + self.pos_embedding
        side0 = int(math.sqrt(self.pos_embedding.size(1)))
        pos = self.pos_embedding.transpose(1, 2).view(1, self.d_model, side0, side0)
        pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
        pos = pos.view(1, self.d_model, h * w).transpose(1, 2)
        return tokens + pos

    def forward(self, x):
        feat = F.relu(self.features(x), inplace=False)
        batch, channels, h, w = feat.shape
        if self.use_coord_mamba:
            tokens = feat.view(batch, channels, h * w).permute(0, 2, 1)
            tokens = self._add_pos_embedding(tokens, h, w)
            if HAS_MAMBA_KERNELS:
                x_fwd = self.sequence_model(tokens)
                x_bwd = self.sequence_model(tokens.flip([1])).flip([1])
                tokens = x_fwd + x_bwd
            else:
                tokens, _ = self.sequence_model(tokens)
            tokens = self.norm(tokens)
            pooled = tokens.mean(dim=1)
        elif self.pos_embedding is not None:
            tokens = feat.view(batch, channels, h * w).permute(0, 2, 1)
            tokens = self.norm(self._add_pos_embedding(tokens, h, w))
            pooled = tokens.mean(dim=1)
        else:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        return self.classifier(self.dropout(pooled))

def build_dataloaders(args):
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_dataset = datasets.ImageFolder(os.path.join(args.data_root, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_root, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_root, "test"), transform=eval_transform)
    if "lymphoma" not in train_dataset.class_to_idx or "normal" not in train_dataset.class_to_idx:
        raise ValueError(f"Expected classes 'lymphoma' and 'normal', got {train_dataset.class_to_idx}")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
        "pin_memory": True,
    }
    return {
        "train_loader": DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs),
        "val_loader": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs),
        "test_loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs),
        "class_names": train_dataset.classes,
        "lymphoma_idx": train_dataset.class_to_idx["lymphoma"],
    }

def compute_auc(labels, probs, positive_idx):
    binary_labels = np.array([1 if y == positive_idx else 0 for y in labels])
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    return float(roc_auc_score(binary_labels, probs))

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate_auc(model, loader, positive_idx, device):
    model.eval()
    labels_out, probs_out = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, positive_idx]
        labels_out.extend(labels.numpy().tolist())
        probs_out.extend(probs.cpu().numpy().tolist())
    return compute_auc(labels_out, probs_out, positive_idx)

@torch.no_grad()
def evaluate_test(model, loader, positive_idx, device):
    model.eval()
    labels_out, preds_out, probs_out = [], [], []
    for images, labels in tqdm(loader, desc="Testing"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, positive_idx]
        preds = torch.argmax(logits, dim=1)
        labels_out.extend(labels.numpy().tolist())
        preds_out.extend(preds.cpu().numpy().tolist())
        probs_out.extend(probs.cpu().numpy().tolist())
    return labels_out, preds_out, probs_out

def run_component_ablation(
    variant_name,
    use_14x14,
    use_mixed_dilation,
    use_coord_mamba,
    description,
    default_save_name,
):
    args = parse_args(description)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    save_path = args.save_path or os.path.join("checkpoints", default_save_name)
    print(f"Using device: {device}")
    print(f"Variant: {variant_name}")
    print(f"Checkpoint: {save_path}")
    data = build_dataloaders(args)
    model = ComponentAblationDenseNet(
        num_classes=len(data["class_names"]),
        img_size=args.img_size,
        use_14x14=use_14x14,
        use_mixed_dilation=use_mixed_dilation,
        use_coord_mamba=use_coord_mamba,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    best_auc = 0.0
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, data["train_loader"], optimizer, scheduler, criterion, device)
        val_auc = evaluate_auc(model, data["val_loader"], data["lymphoma_idx"], device)
        print(
            f"[Epoch {epoch + 1}/{args.num_epochs}] "
            f"loss={train_loss:.4f} val_auc={val_auc:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.1e}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"==> Best model saved (AUC={best_auc:.4f})")
    print(f"\nBest Val AUC: {best_auc:.4f}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    y_true, y_pred, y_probs = evaluate_test(model, data["test_loader"], data["lymphoma_idx"], device)
    test_auc = compute_auc(y_true, y_probs, data["lymphoma_idx"])
    print(f"\nFinal Test AUC (lymphoma as positive class): {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=data["class_names"], digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
