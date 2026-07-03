import os
import csv
import math
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
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
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    try:
        from causal_conv1d import causal_conv1d_fn
    except ImportError:
        causal_conv1d_fn = None
    HAS_MAMBA_KERNELS = True
    print(">> [Mamba] 底层加速算子加载成功。")
except ImportError:
    HAS_MAMBA_KERNELS = False
    print("!! [Warning] 未找到 Mamba 环境，将退化为 Bi-LSTM。")

def patch_densenet121_to_14x14_with_mixed_dilation(features: nn.Module, pattern=(1, 2)):
    if hasattr(features, "transition3") and hasattr(features.transition3, "pool"):
        features.transition3.pool = nn.Identity()
    else:
        raise RuntimeError("找不到 features.transition3.pool，torchvision 版本可能不同。")
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
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
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

class DenseMambaTongue(nn.Module):
    def __init__(
        self,
        num_classes=2,
        d_model=1024,
        img_size=320,
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
            self.feat_h = H
            self.feat_w = W
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, self.d_model) * 0.02)
        self.norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(self.d_model, num_classes)
        print(f">> [Backbone] DenseNet 输出 token 数 = {self.num_tokens}")
        print(f">> [Backbone] denseblock4 mixed dilation pattern = {mixed_dilation_pattern}")

    def forward(self, x):
        feat = self.features(x)
        feat = F.relu(feat, inplace=False)
        if self.use_global_fusion:
            g = F.avg_pool2d(feat, kernel_size=2, stride=2)
            g = F.interpolate(g, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            feat = feat + g
        b, c, h, w = feat.shape
        tokens = feat.view(b, c, h * w).permute(0, 2, 1)
        if tokens.size(1) != self.pos_embedding.size(1):
            L0 = self.pos_embedding.size(1)
            s0 = int(math.sqrt(L0))
            pos = self.pos_embedding.transpose(1, 2).view(1, self.d_model, s0, s0)
            pos = F.interpolate(pos, size=(h, w), mode="bilinear", align_corners=False)
            pos = pos.view(1, self.d_model, h * w).transpose(1, 2)
            tokens = tokens + pos
        else:
            tokens = tokens + self.pos_embedding
        if HAS_MAMBA_KERNELS:
            x_fwd = self.mamba(tokens)
            x_bwd = self.mamba(tokens.flip([1])).flip([1])
            x = x_fwd + x_bwd
        else:
            x, _ = self.mamba(tokens)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        pooled = self.dropout(pooled)
        out = self.classifier(pooled)
        return out
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_transform(img_size=320):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def build_dataloader(data_root, img_size=320, batch_size=16, num_workers=4):
    transform = build_transform(img_size)
    dataset = datasets.ImageFolder(root=data_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataset, loader

def build_model(checkpoint_path, img_size=320, num_classes=2, device="cuda"):
    model = DenseMambaTongue(
        num_classes=num_classes,
        img_size=img_size,
        mixed_dilation_pattern=(1, 2),
        use_global_fusion=False,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    print(f"[INFO] Loaded checkpoint from: {checkpoint_path}")
    model = model.to(device)
    model.eval()
    return model

@torch.no_grad()
def evaluate_binary_classification(model, loader, device="cuda", positive_class_idx=0):
    all_labels = []
    all_probs = []
    all_preds = []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, positive_class_idx]
        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    binary_labels = np.array([1 if y == positive_class_idx else 0 for y in all_labels])
    auc = roc_auc_score(binary_labels, all_probs) if len(np.unique(binary_labels)) > 1 else 0.0
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, pos_label=positive_class_idx, average="binary", zero_division=0)
    recall = recall_score(all_labels, all_preds, pos_label=positive_class_idx, average="binary", zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=positive_class_idx, average="binary", zero_division=0)
    return {
        "AUC": auc,
        "ACC": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

def parse_corruption_info(folder_name: str):
    if "_" not in folder_name:
        return folder_name, folder_name
    corruption_type, severity = folder_name.split("_", 1)
    severity = severity.replace("p", ".")
    return corruption_type, severity

def discover_corruption_folders(corruption_root: Path):
    return sorted([p for p in corruption_root.iterdir() if p.is_dir()])

def save_results_csv(results, save_path):
    fieldnames = [
        "Dataset",
        "Corruption",
        "Severity",
        "AUC",
        "ACC",
        "Precision",
        "Recall",
        "F1",
        "AUC_Retention",
        "F1_Retention",
    ]
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"[INFO] Results saved to: {save_path}")

def print_results_table(results):
    print("\n" + "=" * 120)
    print(
        f"{'Dataset':<25} {'Corruption':<12} {'Severity':<10} "
        f"{'AUC':<10} {'ACC':<10} {'F1':<10} {'AUC_Ret':<10} {'F1_Ret':<10}"
    )
    print("-" * 120)
    for r in results:
        print(
            f"{r['Dataset']:<25} "
            f"{r['Corruption']:<12} "
            f"{str(r['Severity']):<10} "
            f"{r['AUC']:<10.4f} "
            f"{r['ACC']:<10.4f} "
            f"{r['F1']:<10.4f} "
            f"{r['AUC_Retention']:<10.4f} "
            f"{r['F1_Retention']:<10.4f}"
        )
    print("=" * 120 + "\n")

def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[INFO] Using device: {device}")
    clean_test_root = Path(args.clean_test_root)
    corruption_root = Path(args.corruption_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        num_classes=args.num_classes,
        device=device,
    )
    print(f"\n[INFO] Evaluating clean test set: {clean_test_root}")
    clean_dataset, clean_loader = build_dataloader(
        data_root=str(clean_test_root),
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"[INFO] Clean test classes: {clean_dataset.classes}")
    print(f"[INFO] Clean test size   : {len(clean_dataset)}")
    if "lymphoma" not in clean_dataset.class_to_idx:
        raise ValueError(f"'lymphoma' 不在类别名中，当前类别映射为: {clean_dataset.class_to_idx}")
    lymphoma_idx = clean_dataset.class_to_idx["lymphoma"]
    print(f"[INFO] lymphoma_idx = {lymphoma_idx}")
    clean_metrics = evaluate_binary_classification(
        model, clean_loader, device=device, positive_class_idx=lymphoma_idx
    )
    results = []
    clean_row = {
        "Dataset": clean_test_root.name,
        "Corruption": "clean",
        "Severity": "0",
        "AUC": clean_metrics["AUC"],
        "ACC": clean_metrics["ACC"],
        "Precision": clean_metrics["Precision"],
        "Recall": clean_metrics["Recall"],
        "F1": clean_metrics["F1"],
        "AUC_Retention": 1.0,
        "F1_Retention": 1.0,
    }
    results.append(clean_row)
    clean_auc = clean_metrics["AUC"]
    clean_f1 = clean_metrics["F1"]
    corruption_folders = discover_corruption_folders(corruption_root)
    print(f"\n[INFO] Found {len(corruption_folders)} corruption folders.")
    for folder in corruption_folders:
        corruption_name = folder.name
        ctype, severity = parse_corruption_info(corruption_name)
        print(f"\n[INFO] Evaluating corruption set: {folder}")
        dataset, loader = build_dataloader(
            data_root=str(folder),
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        if dataset.class_to_idx != clean_dataset.class_to_idx:
            raise ValueError(
                f"类别映射不一致！\n"
                f"clean: {clean_dataset.class_to_idx}\n"
                f"{folder.name}: {dataset.class_to_idx}"
            )
        metrics = evaluate_binary_classification(
            model, loader, device=device, positive_class_idx=lymphoma_idx
        )
        auc_ret = metrics["AUC"] / clean_auc if clean_auc > 0 else 0.0
        f1_ret = metrics["F1"] / clean_f1 if clean_f1 > 0 else 0.0
        row = {
            "Dataset": corruption_name,
            "Corruption": ctype,
            "Severity": severity,
            "AUC": metrics["AUC"],
            "ACC": metrics["ACC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "AUC_Retention": auc_ret,
            "F1_Retention": f1_ret,
        }
        results.append(row)
    csv_path = output_dir / "robustness_results.csv"
    save_results_csv(results, csv_path)
    print_results_table(results)

def get_args():
    parser = argparse.ArgumentParser("Robustness evaluation for DenseMambaTongue")
    parser.add_argument("--clean_test_root", type=str, required=True)
    parser.add_argument("--corruption_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./robustness_eval_results")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
