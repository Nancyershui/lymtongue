import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
VMAMBA_SEARCH_DIRS = (
    REPO_ROOT,
    REPO_ROOT / "VMamba",
    REPO_ROOT / "Baseline" / "VMamba",
)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODELS = (
    "densenet121",
    "resnet18",
    "vgg16",
    "inception_v3",
    "efficientnet_b0",
    "vit_base",
    "swin_tiny",
    "vmamba",
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified baseline training script for Table II models."
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--data-root", default="/path/to/tongue_dataset_split")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=3407)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--save-each-run-model", dest="save_each_run_model", action="store_true")
    parser.add_argument("--no-save-each-run-model", dest="save_each_run_model", action="store_false")
    parser.set_defaults(save_each_run_model=True)
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

def build_dataloaders(args, seed):
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, "train"),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, "val"),
        transform=eval_transform,
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_root, "test"),
        transform=eval_transform,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    class_to_idx = train_dataset.class_to_idx
    if "lymphoma" not in class_to_idx or "normal" not in class_to_idx:
        raise ValueError(
            f"Expected ImageFolder classes to include 'lymphoma' and 'normal', got {class_to_idx}."
        )
    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": train_dataset.classes,
        "class_to_idx": class_to_idx,
        "lymphoma_idx": class_to_idx["lymphoma"],
        "normal_idx": class_to_idx["normal"],
    }

class InceptionV3ForClassification(nn.Module):
    def __init__(self, pretrained, num_classes):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.inception_v3(weights=weights, aux_logits=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if self.model.AuxLogits is not None:
            self.model.AuxLogits.fc = nn.Linear(
                self.model.AuxLogits.fc.in_features,
                num_classes,
            )

    def forward(self, x):
        out = self.model(x)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, tuple):
            return out[0]
        return out

class VMambaWrapper(nn.Module):
    def __init__(self, backbone, num_classes, img_size):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_size = img_size
        self.head = self._build_head()

    def _forward_backbone(self, x):
        out = self.backbone(x)
        if isinstance(out, dict):
            out = next(v for v in out.values() if torch.is_tensor(v))
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def _build_head(self):
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.img_size, self.img_size)
            feat = self._forward_backbone(dummy)
        if feat.ndim == 2 and feat.shape[1] == self.num_classes:
            return nn.Identity()
        if feat.ndim == 2:
            return nn.Linear(feat.shape[1], self.num_classes)
        if feat.ndim == 4:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(feat.shape[1], self.num_classes),
            )
        return nn.Sequential(
            nn.Flatten(1),
            nn.Linear(math.prod(feat.shape[1:]), self.num_classes),
        )

    def forward(self, x):
        return self.head(self._forward_backbone(x))

def build_vmamba_model(num_classes, img_size):
    for candidate_dir in VMAMBA_SEARCH_DIRS:
        if candidate_dir.exists() and str(candidate_dir) not in sys.path:
            sys.path.insert(0, str(candidate_dir))
    try:
        from vmamba import VSSM
    except ImportError as exc:
        raise ImportError(
            "VMamba baseline requires a vmamba.py module in the repository root, "
            "VMamba/, Baseline/VMamba/, or an installed vmamba package."
        ) from exc
    try:
        backbone = VSSM(img_size=img_size, in_chans=3, num_classes=num_classes)
    except TypeError:
        backbone = VSSM()
    return VMambaWrapper(backbone, num_classes=num_classes, img_size=img_size)

def build_timm_model(model_name, pretrained, num_classes):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Transformer baselines require timm. Install it before running this model.") from exc
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

def build_baseline_model(model_name, num_classes=2, pretrained=True, img_size=224):
    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    if model_name == "inception_v3":
        return InceptionV3ForClassification(pretrained=pretrained, num_classes=num_classes)
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "vit_base":
        return build_timm_model("vit_base_patch16_224", pretrained, num_classes)
    if model_name == "swin_tiny":
        return build_timm_model("swin_tiny_patch4_window7_224", pretrained, num_classes)
    if model_name == "vmamba":
        return build_vmamba_model(num_classes=num_classes, img_size=img_size)
    raise ValueError(f"Unsupported model: {model_name}")

def compute_auc_for_lymphoma(labels, probs, lymphoma_idx):
    binary_labels = np.array([1 if y == lymphoma_idx else 0 for y in labels])
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    return float(roc_auc_score(binary_labels, probs))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate_metrics(model, loader, device, lymphoma_idx, normal_idx):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = F.softmax(logits, dim=1)[:, lymphoma_idx]
        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_probs.extend(probs.cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[lymphoma_idx, normal_idx],
        average=None,
        zero_division=0,
    )
    return {
        "auc": compute_auc_for_lymphoma(y_true, y_probs, lymphoma_idx),
        "acc": float(accuracy_score(y_true, y_pred)),
        "lymphoma_precision": float(precision[0]),
        "lymphoma_recall": float(recall[0]),
        "lymphoma_f1": float(f1[0]),
        "normal_precision": float(precision[1]),
        "normal_recall": float(recall[1]),
        "normal_f1": float(f1[1]),
    }

def mean_std(values):
    values = np.array(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

def run_once(args, run_id, seed, device):
    print("\n" + "=" * 100)
    print(f"{args.model} | Run {run_id + 1}/{args.num_runs} | Seed = {seed}")
    print("=" * 100)
    seed_everything(seed)
    data = build_dataloaders(args, seed)
    model = build_baseline_model(
        args.model,
        num_classes=len(data["class_names"]),
        pretrained=not args.no_pretrained,
        img_size=args.img_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6,
    )
    best_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(
            model,
            data["train_loader"],
            optimizer,
            criterion,
            device,
        )
        val_metrics = evaluate_metrics(
            model,
            data["val_loader"],
            device,
            data["lymphoma_idx"],
            data["normal_idx"],
        )
        scheduler.step()
        val_auc = val_metrics["auc"]
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        print(
            f"[Epoch {epoch + 1}/{args.num_epochs}] "
            f"loss={train_loss:.4f} val_auc={val_auc:.4f} "
            f"val_acc={val_metrics['acc']:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            if args.save_each_run_model:
                ckpt_dir = Path(args.output_dir) / args.model / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_wts, ckpt_dir / f"{args.model}_run{run_id + 1}.pth")
    model.load_state_dict(best_model_wts)
    test_metrics = evaluate_metrics(
        model,
        data["test_loader"],
        device,
        data["lymphoma_idx"],
        data["normal_idx"],
    )
    result = {
        "model": args.model,
        "run_id": run_id + 1,
        "seed": seed,
        "best_val_auc": float(best_auc),
        "test_auc": test_metrics["auc"],
        "test_acc": test_metrics["acc"],
        "lymphoma_precision": test_metrics["lymphoma_precision"],
        "lymphoma_recall": test_metrics["lymphoma_recall"],
        "lymphoma_f1": test_metrics["lymphoma_f1"],
        "normal_precision": test_metrics["normal_precision"],
        "normal_recall": test_metrics["normal_recall"],
        "normal_f1": test_metrics["normal_f1"],
        "history": history,
    }
    print(
        f"[Test] auc={result['test_auc']:.4f} acc={result['test_acc']:.4f} "
        f"lym_f1={result['lymphoma_f1']:.4f} normal_f1={result['normal_f1']:.4f}"
    )
    return result

def summarize_results(results):
    metrics = [
        "test_auc",
        "test_acc",
        "lymphoma_precision",
        "lymphoma_recall",
        "lymphoma_f1",
        "normal_precision",
        "normal_recall",
        "normal_f1",
    ]
    summary = {}
    for metric in metrics:
        mean_v, std_v = mean_std([r[metric] for r in results])
        summary[metric] = {"mean": mean_v, "std": std_v}
    return summary

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")
    print(f"Training baseline: {args.model}")
    print(f"Data root: {args.data_root}")
    results = []
    seed_list = [args.seed_start + i for i in range(args.num_runs)]
    for run_id, seed in enumerate(seed_list):
        results.append(run_once(args, run_id, seed, device))
    summary = summarize_results(results)
    print("\n" + "=" * 100)
    print(f"{args.model} summary over all {args.num_runs} independent runs")
    print("=" * 100)
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    payload = {
        "config": vars(args),
        "class_positive_for_auc": "lymphoma",
        "runs": results,
        "summary": summary,
    }
    out_path = output_dir / f"{args.model}_5runs_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {out_path}")

if __name__ == "__main__":
    main()
