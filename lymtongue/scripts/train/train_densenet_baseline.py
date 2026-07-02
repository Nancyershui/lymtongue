


import os
import json
import random
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)




def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)





data_root = "/path/to/tongue_dataset_split"
num_classes = 2
img_size = 224
batch_size = 16
num_epochs = 30
lr = 1e-4
weight_decay = 1e-4
num_workers = 4

num_runs = 5
save_each_run_model = True

seed_list = [3407 + i for i in range(num_runs)]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("checkpoints", exist_ok=True)





def build_dataloaders(seed):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_root, "val"),
        transform=transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(data_root, "test"),
        transform=transform
    )

    print("Classes:", train_dataset.classes)
    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    pos_name = "lymphoma"
    if pos_name not in class_names:
        raise ValueError(f"Positive class '{pos_name}' not found in classes: {class_names}")

    lymphoma_idx = class_to_idx["lymphoma"]
    normal_idx = class_to_idx["normal"]

    print(f"Positive class = {pos_name}, pos_idx = {lymphoma_idx}")
    print(f"normal_idx = {normal_idx}")

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_names,
        class_to_idx,
        lymphoma_idx,
        normal_idx
    )





def build_model():
    try:
        model = models.densenet121(weights="DEFAULT")
    except TypeError:
        model = models.densenet121(pretrained=True)

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model = model.to(device)
    return model





def compute_auc_for_lymphoma(labels, probs, lymphoma_idx):
    binary_labels = np.array([1 if y == lymphoma_idx else 0 for y in labels])
    if len(np.unique(binary_labels)) < 2:
        return 0.0
    return float(roc_auc_score(binary_labels, probs))





def train_one_epoch(model, loader, optimizer, criterion, lymphoma_idx):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0
    all_labels = []
    all_probs = []

    for imgs, labels in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)[:, lymphoma_idx]
        preds = torch.argmax(outputs, dim=1)

        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    train_loss = total_loss / total_num
    train_acc = total_correct / total_num
    train_auc = compute_auc_for_lymphoma(all_labels, all_probs, lymphoma_idx)

    return train_loss, train_acc, train_auc





@torch.no_grad()
def evaluate_metrics(model, loader, lymphoma_idx, normal_idx):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, lymphoma_idx]
        preds = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    auc = compute_auc_for_lymphoma(all_labels, all_probs, lymphoma_idx)
    acc = float(accuracy_score(all_labels, all_preds))

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=[lymphoma_idx, normal_idx],
        average=None,
        zero_division=0
    )

    return {
        "auc": float(auc),
        "acc": float(acc),

        "lymphoma_precision": float(precision[0]),
        "lymphoma_recall": float(recall[0]),
        "lymphoma_f1": float(f1[0]),

        "normal_precision": float(precision[1]),
        "normal_recall": float(recall[1]),
        "normal_f1": float(f1[1]),
    }





def run_once(run_id, seed):
    print("\n" + "=" * 90)
    print(f"Run {run_id + 1}/{num_runs} | Seed = {seed}")
    print("=" * 90)

    set_seed(seed)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
        class_names,
        class_to_idx,
        lymphoma_idx,
        normal_idx
    ) = build_dataloaders(seed)

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_auc = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(num_epochs):
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, lymphoma_idx
        )

        val_metrics = evaluate_metrics(
            model, val_loader, lymphoma_idx, normal_idx
        )

        val_auc = val_metrics["auc"]

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train AUC: {train_auc:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "train_auc": float(train_auc),
            "val_auc": float(val_auc),
            "val_acc": float(val_metrics["acc"])
        })

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())

            if save_each_run_model:
                ckpt_path = f"checkpoints/densenet121_run{run_id + 1}.pth"
                torch.save(best_model_wts, ckpt_path)
                print(f"==> Best model saved to {ckpt_path}")
            else:
                print("==> Best model updated")

    print(f"\nBest Val AUC: {best_auc:.4f}")

    model.load_state_dict(best_model_wts)

    final_val_metrics = evaluate_metrics(
        model, val_loader, lymphoma_idx, normal_idx
    )
    test_metrics = evaluate_metrics(
        model, test_loader, lymphoma_idx, normal_idx
    )

    result = {
        "run_id": run_id + 1,
        "seed": seed,
        "val_auc": float(final_val_metrics["auc"]),

        "test_auc": float(test_metrics["auc"]),
        "test_acc": float(test_metrics["acc"]),

        "lymphoma_precision": float(test_metrics["lymphoma_precision"]),
        "lymphoma_recall": float(test_metrics["lymphoma_recall"]),
        "lymphoma_f1": float(test_metrics["lymphoma_f1"]),

        "normal_precision": float(test_metrics["normal_precision"]),
        "normal_recall": float(test_metrics["normal_recall"]),
        "normal_f1": float(test_metrics["normal_f1"]),
    }

    print("\nFinal metrics of this run:")
    print(f"val_auc            = {result['val_auc']:.4f}")
    print(f"test_auc           = {result['test_auc']:.4f}")
    print(f"test_acc           = {result['test_acc']:.4f}")
    print(f"lymphoma_precision = {result['lymphoma_precision']:.4f}")
    print(f"lymphoma_recall    = {result['lymphoma_recall']:.4f}")
    print(f"lymphoma_f1        = {result['lymphoma_f1']:.4f}")
    print(f"normal_precision   = {result['normal_precision']:.4f}")
    print(f"normal_recall      = {result['normal_recall']:.4f}")
    print(f"normal_f1          = {result['normal_f1']:.4f}")

    run_json_path = f"densenet121_run{run_id + 1}_results.json"
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "history": history,
            "result": result,
            "class_to_idx": class_to_idx
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved run results to {run_json_path}")

    del model
    torch.cuda.empty_cache()

    return result





def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return arr.mean(), arr.std()





def main():
    all_results = []

    for run_id, seed in enumerate(seed_list):
        result = run_once(run_id, seed)
        all_results.append(result)

    print("\n" + "=" * 100)
    print(f"All {num_runs} independent runs finished.")
    print("=" * 100)

    print("\nAll run results:")
    for r in all_results:
        print(
            f"Run {r['run_id']:2d} | seed={r['seed']} | "
            f"val_auc={r['val_auc']:.4f} | "
            f"test_auc={r['test_auc']:.4f} | "
            f"test_acc={r['test_acc']:.4f} | "
            f"lym_p={r['lymphoma_precision']:.4f} | "
            f"lym_r={r['lymphoma_recall']:.4f} | "
            f"lym_f1={r['lymphoma_f1']:.4f} | "
            f"nor_p={r['normal_precision']:.4f} | "
            f"nor_r={r['normal_recall']:.4f} | "
            f"nor_f1={r['normal_f1']:.4f}"
        )

    metrics_to_report = [
        "val_auc",
        "test_auc",
        "test_acc",
        "lymphoma_precision",
        "lymphoma_recall",
        "lymphoma_f1",
        "normal_precision",
        "normal_recall",
        "normal_f1",
    ]

    print("\n" + "=" * 100)
    print(f"Average over all {num_runs} independent runs")
    print("=" * 100)

    summary = {}
    for metric in metrics_to_report:
        mean_v, std_v = mean_std([r[metric] for r in all_results])
        summary[metric] = {
            "mean": float(mean_v),
            "std": float(std_v)
        }
        print(f"{metric}: {mean_v:.4f} ± {std_v:.4f}")

    save_txt_path = "densenet121_5runs_summary.txt"
    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write(f"All {num_runs} independent runs:\n")
        for r in all_results:
            f.write(
                f"Run {r['run_id']:2d} | seed={r['seed']} | "
                f"val_auc={r['val_auc']:.4f} | "
                f"test_auc={r['test_auc']:.4f} | "
                f"test_acc={r['test_acc']:.4f} | "
                f"lymphoma_precision={r['lymphoma_precision']:.4f} | "
                f"lymphoma_recall={r['lymphoma_recall']:.4f} | "
                f"lymphoma_f1={r['lymphoma_f1']:.4f} | "
                f"normal_precision={r['normal_precision']:.4f} | "
                f"normal_recall={r['normal_recall']:.4f} | "
                f"normal_f1={r['normal_f1']:.4f}\n"
            )

        f.write(f"\nAverage over all {num_runs} independent runs:\n")
        for metric in metrics_to_report:
            mean_v = summary[metric]["mean"]
            std_v = summary[metric]["std"]
            f.write(f"{metric}: {mean_v:.4f} ± {std_v:.4f}\n")

    print(f"\nResults saved to: {save_txt_path}")

    save_json_path = "densenet121_5runs_summary.json"
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "data_root": data_root,
                "num_classes": num_classes,
                "img_size": img_size,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_workers": num_workers,
                "num_runs": num_runs
            },
            "all_results": all_results,
            "summary": summary
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {save_json_path}")


if __name__ == "__main__":
    main()














































































































































































































































































































































































































































