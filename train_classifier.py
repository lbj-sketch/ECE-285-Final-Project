"""
Stage 4: Downstream Classification with Synthetic Augmentation
==============================================================
Train a ResNet-18 skin lesion classifier, comparing the following scenarios:
  (A) Baseline:     Train with real data only
  (B) Augmented:    Train with real + synthetic data (minority-only augmentation)
  (C) Balanced Aug: Train with real + synthetic data (balanced augmentation)

Metrics: Accuracy, Macro-F1, Balanced Accuracy, Per-class Recall

Usage:
  python train_classifier.py --mode baseline
  python train_classifier.py --mode augmented
  python train_classifier.py --mode balanced
  python train_classifier.py --mode all        # Run all three and generate report
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from datasets import load_dataset
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, balanced_accuracy_score
)
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import json
import argparse


# ============================================================
# Dataset Definitions
# ============================================================
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
LABEL2IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX2LABEL = {i: name for name, i in LABEL2IDX.items()}
NUM_CLASSES = len(CLASS_NAMES)

# Original labels in dataset -> our short label names (based on debug_labels.py results)
DX_MAPPING = {
    "akiec": "akiec", "actinic_keratoses": "akiec", "actinic_keratosis": "akiec",
    "bcc": "bcc", "basal_cell_carcinoma": "bcc",
    "bkl": "bkl", "benign_keratosis-like_lesions": "bkl", "benign_keratosis": "bkl",
    "df": "df", "dermatofibroma": "df",
    "mel": "mel", "melanoma": "mel",
    "nv": "nv", "melanocytic_nevi": "nv", "melanocytic_nevus": "nv",
    "vasc": "vasc", "vascular_lesions": "vasc", "vascular_lesion": "vasc",
}


def get_transforms(train=True, size=224):
    """Data augmentation (training) and normalization (testing)"""
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class HFSkinDataset(Dataset):
    """Load real HAM10000 dataset from HuggingFace"""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        raw_dx = sample["dx"].strip().lower()
        mapped = DX_MAPPING.get(raw_dx, raw_dx)
        label = LABEL2IDX[mapped]
        if self.transform:
            image = self.transform(image)
        return image, label


class SyntheticDataset(Dataset):
    """Load synthetic images from local folder (output of batch_generate)"""
    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"  Warning: Synthetic image folder not found: {class_dir}")
                continue
            
            files = sorted([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
            
            if max_per_class:
                files = files[:max_per_class]
            
            for fname in files:
                self.images.append(os.path.join(class_dir, fname))
                self.labels.append(LABEL2IDX[class_name])
        
        print(f"  Synthetic dataset: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# Training and Evaluation
# ============================================================
def create_model(num_classes=NUM_CLASSES):
    """Create ResNet-18 classifier"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(dataloader), 100.0 * correct / total


def evaluate(model, dataloader, device):
    """Evaluate on validation/test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = 100.0 * np.mean(all_preds == all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    # Per-class recall
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True
    )
    per_class_recall = {name: report[name]['recall'] for name in CLASS_NAMES}
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'balanced_accuracy': balanced_acc,
        'per_class_recall': per_class_recall,
        'classification_report': classification_report(
            all_labels, all_preds, target_names=CLASS_NAMES
        ),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
    }


def run_experiment(
    mode="baseline",
    synthetic_dir="/home/jovyan/ece285/synthetic_dataset",
    output_dir="/home/jovyan/ece285/classifier_results",
    num_epochs=15,
    batch_size=32,
    lr=1e-4,
    syn_max_per_class=None
):
    """
    Run a complete training + evaluation experiment.
    
    mode:
        'baseline'    - Real data only
        'augmented'   - Real + synthetic (supplement minority classes with synthetic images)
        'oversampled' - Real + duplicated (supplement minority classes with duplicated real images)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Stage 4: Classifier Training - Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    # 1. Load real dataset
    print("\nLoading HAM10000 dataset...")
    hf_ds = load_dataset("marmal88/skin_cancer", split="train")
    
    # 80/20 train/val split
    hf_ds_split = hf_ds.train_test_split(test_size=0.2, seed=42)
    
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    train_real = HFSkinDataset(hf_ds_split["train"], transform=train_transform)
    val_dataset = HFSkinDataset(hf_ds_split["test"], transform=val_transform)
    
    # Count per-class samples in real data
    real_class_counts = {c: 0 for c in CLASS_NAMES}
    for sample in hf_ds_split["train"]:
        raw_dx = sample["dx"].strip().lower()
        mapped = DX_MAPPING.get(raw_dx, raw_dx)
        if mapped in CLASS_NAMES:
            real_class_counts[mapped] = real_class_counts.get(mapped, 0) + 1
    
    print(f"  Real training set: {len(train_real)} images")
    print(f"  Validation set: {len(val_dataset)} images")
    print(f"  Per-class counts: {real_class_counts}")
    
    # 2. Count available synthetic images (for aligning controlled variables)
    synthetic_counts = {c: 0 for c in CLASS_NAMES}
    for c in CLASS_NAMES:
        c_dir = os.path.join(synthetic_dir, c)
        if os.path.exists(c_dir):
            synthetic_counts[c] = len([f for f in os.listdir(c_dir) if f.endswith((".png", ".jpg"))])

    # 3. 决定增强目标值（使用 75% 分位数，即“中位数靠右”，向大类看齐但避开极端的 nv 类）
    counts_list = list(real_class_counts.values())
    target_count = int(np.percentile(counts_list, 75)) # approximately around 850
    minority_classes = [c for c, n in real_class_counts.items() if n < target_count]

    if mode == "baseline":
        train_dataset = train_real
        print(f"\n  [Baseline] Training with real data only")
        
    elif mode == "augmented":
        print(f"\n  [Augmented] Minority-class synthetic augmentation (target: {target_count})")
        print(f"  Augmented classes: {minority_classes}")
        
        syn_dataset = SyntheticMinorityDataset(
            synthetic_dir, minority_classes, 
            target_count=target_count,
            real_counts=real_class_counts,
            transform=train_transform
        )
        train_dataset = ConcatDataset([train_real, syn_dataset])
        print(f"  Augmented training set total: {len(train_dataset)} images")
        
    elif mode == "oversampled":
        print(f"\n  [Oversampled] Minority-class random oversampling (target: {target_count})")
        print(f"  Augmented classes: {minority_classes}")
        print(f"  Note: Supplement count aligned with available synthetic images for fair comparison.")
        
        over_dataset = RealOversampledDataset(
            hf_ds_split["train"], 
            minority_classes,
            target_count=target_count,
            real_counts=real_class_counts,
            synthetic_counts=synthetic_counts,
            transform=train_transform
        )
        train_dataset = ConcatDataset([train_real, over_dataset])
        print(f"  Augmented training set total: {len(train_dataset)} images (aligned with Augmented data volume)")
    
    # 4. DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 5. Model
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 6. Training
    best_f1 = 0.0
    train_history = []
    
    for epoch in range(num_epochs):
        loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        
        # Validate each epoch
        val_results = evaluate(model, val_loader, device)
        
        print(f"  Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {loss:.4f} | "
              f"Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_results['accuracy']:.1f}% | "
              f"Macro-F1: {val_results['macro_f1']:.4f}")
        
        train_history.append({
            'epoch': epoch + 1,
            'loss': loss,
            'train_acc': train_acc,
            'val_acc': val_results['accuracy'],
            'macro_f1': val_results['macro_f1']
        })
        
        # Save best model
        if val_results['macro_f1'] > best_f1:
            best_f1 = val_results['macro_f1']
            model_path = os.path.join(output_dir, f"best_model_{mode}.pth")
            torch.save(model.state_dict(), model_path)
    
    # 7. Final evaluation (load best model)
    model.load_state_dict(torch.load(model_path))
    final_results = evaluate(model, val_loader, device)
    
    print(f"\n{'='*60}")
    print(f"Final Results [{mode.upper()}]")
    print(f"{'='*60}")
    print(f"  Accuracy:          {final_results['accuracy']:.2f}%")
    print(f"  Macro-F1:          {final_results['macro_f1']:.4f}")
    print(f"  Balanced Accuracy: {final_results['balanced_accuracy']:.4f}")
    print(f"\n  Per-class Recall:")
    for cls_name, recall in final_results['per_class_recall'].items():
        print(f"    {cls_name:<10}: {recall:.4f}")
    print(f"\n  Classification Report:")
    print(final_results['classification_report'])
    
    # Save results
    results_path = os.path.join(output_dir, f"results_{mode}.json")
    save_data = {
        'mode': mode,
        'accuracy': final_results['accuracy'],
        'macro_f1': final_results['macro_f1'],
        'balanced_accuracy': final_results['balanced_accuracy'],
        'per_class_recall': final_results['per_class_recall'],
        'confusion_matrix': final_results['confusion_matrix'],
        'train_history': train_history
    }
    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    
    return final_results


# ============================================================
# Synthetic Data Augmentation Strategies
# ============================================================
class SyntheticMinorityDataset(Dataset):
    """Load synthetic images for minority classes only, up to target count (limited by actual folder contents)"""
    def __init__(self, root_dir, minority_classes, target_count, real_counts, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        
        for class_name in minority_classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Take at most the number needed to reach target count
            need = max(0, target_count - real_counts.get(class_name, 0))
            
            files = sorted([
                f for f in os.listdir(class_dir) if f.endswith((".png", ".jpg"))
            ])[:need]
            
            for fname in files:
                self.images.append(os.path.join(class_dir, fname))
                self.labels.append(LABEL2IDX[class_name])
            
            print(f"    {class_name}: Added {len(files)} synthetic images (target need: {need})")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class RealOversampledDataset(Dataset):
    """Oversample real minority-class data by duplication (aligned with synthetic image limit)"""
    def __init__(self, hf_train_split, minority_classes, target_count, real_counts, synthetic_counts, transform=None):
        self.samples = []
        self.transform = transform
        
        # Group original data indices by class
        indices_by_class = {c: [] for c in minority_classes}
        for idx, sample in enumerate(hf_train_split):
            raw_dx = sample["dx"].strip().lower()
            mapped = DX_MAPPING.get(raw_dx, raw_dx)
            if mapped in minority_classes:
                indices_by_class[mapped].append(idx)

        # Oversample minority classes
        for class_name in minority_classes:
            available_indices = indices_by_class[class_name]
            if not available_indices:
                continue
            
            # Number to duplicate, capped by the actual available synthetic images for fair comparison
            need_for_median = max(0, target_count - real_counts.get(class_name, 0))
            # Align with actual synthetic supplement count
            actual_supplement = min(need_for_median, synthetic_counts.get(class_name, 0))
            
            # Randomly duplicate existing data points
            if actual_supplement > 0:
                oversampled_indices = np.random.choice(available_indices, size=actual_supplement, replace=True)
                for idx in oversampled_indices:
                    self.samples.append((hf_train_split[int(idx)], LABEL2IDX[class_name]))
                
            print(f"    {class_name}: Added {actual_supplement} duplicated images (aligned with synthetic limit, original need: {need_for_median})")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        image = sample["image"].convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# Comparison Report Generation
# ============================================================
def generate_comparison_report(output_dir="/home/jovyan/ece285/classifier_results"):
    """Read all experiment results and generate comparison report"""
    
    print(f"\n{'='*70}")
    print("Stage 4: Classifier Performance Comparison Report")
    print(f"{'='*70}")
    
    modes = ["baseline", "augmented", "oversampled"]
    all_results = {}
    
    for mode in modes:
        path = os.path.join(output_dir, f"results_{mode}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_results[mode] = json.load(f)
    
    if len(all_results) == 0:
        print("No result files found! Please train models first.")
        return
    
    # Summary table
    print(f"\n{'Metric':<25}", end="")
    for mode in all_results:
        print(f"{mode.upper():<18}", end="")
    print()
    print("-" * 70)
    
    for metric_name, metric_key in [
        ("Accuracy (%)", "accuracy"),
        ("Macro-F1", "macro_f1"),
        ("Balanced Accuracy", "balanced_accuracy"),
    ]:
        print(f"{metric_name:<25}", end="")
        for mode, res in all_results.items():
            val = res[metric_key]
            if metric_key == "accuracy":
                print(f"{val:<18.2f}", end="")
            else:
                print(f"{val:<18.4f}", end="")
        print()
    
    # Per-class recall comparison
    print(f"\n--- Per-class Recall ---")
    print(f"{'Class':<12}", end="")
    for mode in all_results:
        print(f"{mode.upper():<18}", end="")
    print()
    print("-" * 60)
    
    for cls_name in CLASS_NAMES:
        print(f"{cls_name:<12}", end="")
        for mode, res in all_results.items():
            recall = res['per_class_recall'].get(cls_name, 0)
            print(f"{recall:<18.4f}", end="")
        print()
    
    # Save report
    report_path = os.path.join(output_dir, "comparison_report.txt")
    with open(report_path, 'w') as f:
        f.write("Stage 4: Classification Performance Comparison\n")
        f.write("=" * 60 + "\n\n")
        for mode, res in all_results.items():
            f.write(f"[{mode.upper()}]\n")
            f.write(f"  Accuracy: {res['accuracy']:.2f}%\n")
            f.write(f"  Macro-F1: {res['macro_f1']:.4f}\n")
            f.write(f"  Balanced Accuracy: {res['balanced_accuracy']:.4f}\n\n")
    
    print(f"\nReport saved to: {report_path}")


# ============================================================
# Main Program
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Downstream Classification")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["baseline", "augmented", "oversampled", "all"],
                        help="baseline=real only, augmented=synthetic aug, oversampled=duplicated real, all=run all")
    parser.add_argument("--synthetic_dir", type=str, default="/home/jovyan/ece285/synthetic_dataset")
    parser.add_argument("--output_dir", type=str, default="/home/jovyan/ece285/classifier_results")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    if args.mode == "all":
        for mode in ["baseline", "augmented", "oversampled"]:
            run_experiment(
                mode=mode,
                synthetic_dir=args.synthetic_dir,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
        generate_comparison_report(args.output_dir)
    else:
        run_experiment(
            mode=args.mode,
            synthetic_dir=args.synthetic_dir,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
