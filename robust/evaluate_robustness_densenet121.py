import os
import csv
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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

def build_model(checkpoint_path, num_classes=2, device="cuda"):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
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
    precision = precision_score(
        all_labels, all_preds,
        pos_label=positive_class_idx,
        average="binary",
        zero_division=0
    )
    recall = recall_score(
        all_labels, all_preds,
        pos_label=positive_class_idx,
        average="binary",
        zero_division=0
    )
    f1 = f1_score(
        all_labels, all_preds,
        pos_label=positive_class_idx,
        average="binary",
        zero_division=0
    )
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
    csv_path = output_dir / "robustness_results_densenet121.csv"
    save_results_csv(results, csv_path)
    print_results_table(results)

def get_args():
    parser = argparse.ArgumentParser("Robustness evaluation for original DenseNet121 baseline")
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
