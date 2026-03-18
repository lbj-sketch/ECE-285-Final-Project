"""
HAM10000 LoRA Generated Image Evaluation Script
Calculates FID (Fréchet Inception Distance) and IS (Inception Score)
Supports overall and per-class evaluation

Usage:
    python eval_metrics.py                          # Default: Prepare real images + Compute metrics
    python eval_metrics.py --skip_prepare           # Skip preparing real images (use when already existing)
    python eval_metrics.py --num_real_per_class 500  # Specify number of real images per class
"""

import os
import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import argparse

# ============================================================
# Path Configuration (consistent with batch_generate.py)
# ============================================================
GEN_DIR = "/home/jovyan/ece285/synthetic_dataset"
REAL_DIR = "/home/jovyan/ece285/real_images"
RESULTS_FILE = "/home/jovyan/ece285/fid_is_results.txt"

CLASS_IDS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "actinic keratosis",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion"
}


# ============================================================
# 1. Prepare Real Images (Extract from HuggingFace)
# ============================================================
def prepare_real_images(real_output_dir=REAL_DIR, num_per_class=500):
    """
    Extract real images from HuggingFace dataset and save them locally by class.
    """
    from datasets import load_dataset
    
    # Define robust mapping table (precisely adjusted based on debug_labels.py results)
    FULL_MAPPING = {
        "akiec": "akiec", "actinic_keratoses": "akiec",
        "bcc": "bcc", "basal_cell_carcinoma": "bcc",
        "bkl": "bkl", "benign_keratosis-like_lesions": "bkl", "benign_keratosis": "bkl",
        "df": "df", "dermatofibroma": "df",
        "mel": "mel", "melanoma": "mel",
        "nv": "nv", "melanocytic_nevi": "nv", "melanocytic_nevus": "nv", "melanocytic_nevi": "nv",
        "vasc": "vasc", "vascular_lesions": "vasc",
    }

    # Check if already prepared
    all_ready = True
    for cid in CLASS_IDS:
        cdir = os.path.join(real_output_dir, cid)
        if not os.path.exists(cdir) or len([f for f in os.listdir(cdir) if f.endswith(".png")]) < 10: # Just need some images
            all_ready = False
            break

    if all_ready:
        print(f"Real image directory already exists: {real_output_dir}")
        return real_output_dir

    print("Loading HAM10000 dataset from HuggingFace...")
    ds = load_dataset("marmal88/skin_cancer", split="train")

    os.makedirs(real_output_dir, exist_ok=True)
    class_counts = {c: 0 for c in CLASS_IDS}

    print(f"Starting extraction of real images (target {num_per_class} per class)...")
    for sample in tqdm(ds, desc="Extracting real images"):
        raw_label = str(sample["dx"]).strip().lower()
        
        # Match mapping
        class_id = FULL_MAPPING.get(raw_label)
        if class_id is None:
            # Try replacing underscores
            raw_label_alt = raw_label.replace("_", " ")
            class_id = FULL_MAPPING.get(raw_label_alt)
        
        if class_id is None or class_id not in CLASS_IDS:
            continue

        if class_counts[class_id] >= num_per_class:
            continue

        class_folder = os.path.join(real_output_dir, class_id)
        os.makedirs(class_folder, exist_ok=True)

        img = sample["image"].convert("RGB")
        save_path = os.path.join(class_folder, f"real_{class_id}_{class_counts[class_id]:04d}.png")
        img.save(save_path)
        class_counts[class_id] += 1

        # Check if all reached the target
        if all(v >= num_per_class for v in class_counts.values()):
            break

    print("\nReal image extraction statistics:")
    for c, cnt in class_counts.items():
        print(f"  {c}: {cnt} 张")

    return real_output_dir

    # === Start extracting images ===
    os.makedirs(real_output_dir, exist_ok=True)
    class_counts = {c: 0 for c in CLASS_IDS}

    print(f"\nExtracting real images (max {num_per_class} per class)...")
    for idx in tqdm(range(len(ds)), desc="Extracting real images"):
        sample = ds[idx]
        raw_label = sample[label_col]

        # Convert through mapping table
        if raw_label not in label_to_classid:
            continue
        class_id = label_to_classid[raw_label]

        if class_counts[class_id] >= num_per_class:
            continue

        class_folder = os.path.join(real_output_dir, class_id)
        os.makedirs(class_folder, exist_ok=True)

        img = sample["image"].convert("RGB")
        save_path = os.path.join(class_folder, f"real_{class_id}_{class_counts[class_id]:04d}.png")
        img.save(save_path)
        class_counts[class_id] += 1

        if all(v >= num_per_class for v in class_counts.values()):
            break

    print("\nReal image extraction complete:")
    for c, cnt in class_counts.items():
        print(f"  {c}: {cnt} images")

    return real_output_dir


# ============================================================
# 2. Image Loading Utils
# ============================================================
def load_images_as_tensor(folder, max_images=None):
    """
    Load all images from a folder (including subfolders) and return unit8 tensor [N, 3, 299, 299].
    torchmetrics FID/IS requires uint8 format.
    """
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))
    ])

    imgs = []
    count = 0

    if not os.path.exists(folder):
        print(f"  Warning: Path does not exist {folder}")
        return None

    for root, dirs, files in os.walk(folder):
        for fname in sorted(files):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                imgs.append(transform(img))
                count += 1
                if max_images and count >= max_images:
                    return torch.stack(imgs)
            except Exception as e:
                print(f"  Skipping corrupted file: {fpath} ({e})")

    if len(imgs) == 0:
        return None

    return torch.stack(imgs)


def count_images(folder):
    """Count number of images in folder"""
    if not os.path.exists(folder):
        return 0
    count = 0
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                count += 1
    return count


# ============================================================
# 3. Compute FID and IS
# ============================================================
def compute_fid_is(gen_dir=GEN_DIR, real_dir=REAL_DIR, results_file=RESULTS_FILE):
    """
    Compute FID and IS, saving results to file.

    - FID (Fréchet Inception Distance): Lower is better, 0 means identical
    - IS (Inception Score): Higher is better, measures clarity and diversity
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32

    print("=" * 60)
    print("FID & IS Evaluation")
    print("=" * 60)

    # Image statistics
    gen_total = count_images(gen_dir)
    real_total = count_images(real_dir)
    print(f"\nGenerated images directory: {gen_dir} ({gen_total} images)")
    print(f"Real images directory: {real_dir} ({real_total} images)")

    for cid in CLASS_IDS:
        gc = count_images(os.path.join(gen_dir, cid))
        rc = count_images(os.path.join(real_dir, cid))
        print(f"  {cid:6s}: Generated {gc:5d}, Real {rc:5d}")

    if real_total == 0:
        print("\nError: Real image count is 0! Run prepare_real_images first or check label mapping.")
        return None

    # ----------------------------
    # 3.1 Compute Overall IS
    # ----------------------------
    print("\n" + "-" * 40)
    print("Computing Inception Score (IS)...")
    print("-" * 40)

    gen_all = load_images_as_tensor(gen_dir)
    is_mean_val = 0.0
    is_std_val = 0.0

    if gen_all is not None and len(gen_all) > 0:
        is_metric = InceptionScore(normalize=False).to(device)
        for i in tqdm(range(0, len(gen_all), batch_size), desc="IS"):
            batch = gen_all[i:i + batch_size].to(device)
            is_metric.update(batch)
        is_mean, is_std = is_metric.compute()
        is_mean_val = is_mean.item()
        is_std_val = is_std.item()
        print(f"  Overall IS: {is_mean_val:.4f} ± {is_std_val:.4f}")
        del is_metric
        torch.cuda.empty_cache()
    else:
        print("  Error: No generated images found!")
        return

    # ----------------------------
    # 3.2 Compute Overall FID
    # ----------------------------
    print("\n" + "-" * 40)
    print("Computing Overall FID...")
    print("-" * 40)

    real_all = load_images_as_tensor(real_dir)
    overall_fid_val = -1.0

    if real_all is not None and len(real_all) > 0:
        fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

        print("  Processing real images...")
        for i in tqdm(range(0, len(real_all), batch_size), desc="FID-Real"):
            batch = real_all[i:i + batch_size].to(device)
            fid_metric.update(batch, real=True)

        print("  Processing generated images...")
        for i in tqdm(range(0, len(gen_all), batch_size), desc="FID-Fake"):
            batch = gen_all[i:i + batch_size].to(device)
            fid_metric.update(batch, real=False)

        overall_fid_val = fid_metric.compute().item()
        print(f"  Overall FID: {overall_fid_val:.4f}")
        del fid_metric
        torch.cuda.empty_cache()
    else:
        print("  Error: No real images found!")
        return

    # Free overall tensors to save memory
    del gen_all, real_all
    torch.cuda.empty_cache()

    # ----------------------------
    # 3.3 Per-class FID
    # ----------------------------
    print("\n" + "-" * 40)
    print("Computing Per-class FID...")
    print("-" * 40)

    per_class_fid = {}
    for cid in CLASS_IDS:
        real_cls_dir = os.path.join(real_dir, cid)
        gen_cls_dir = os.path.join(gen_dir, cid)

        # 1. Count real images
        n_real = count_images(real_cls_dir)
        if n_real < 10:
            print(f"  {cid:6s}: Skipping (too few real images: {n_real})")
            continue

        # 2. Load images, limiting generated count to match real count
        r_imgs = load_images_as_tensor(real_cls_dir)
        g_imgs = load_images_as_tensor(gen_cls_dir, max_images=n_real) # Align counts

        if r_imgs is not None and g_imgs is not None:
            print(f"  {cid:6s}: Computing Balanced FID ({len(r_imgs)} real vs {len(g_imgs)} fake)...")
            fid_cls = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

            for i in range(0, len(r_imgs), batch_size):
                fid_cls.update(r_imgs[i:i + batch_size].to(device), real=True)
            for i in range(0, len(g_imgs), batch_size):
                fid_cls.update(g_imgs[i:i + batch_size].to(device), real=False)

            score = fid_cls.compute().item()
            per_class_fid[cid] = score
            print(f"  {cid:6s} ({CLASS_NAMES[cid]:30s}): FID = {score:.4f}")

            del fid_cls
            torch.cuda.empty_cache()
        else:
            print(f"  {cid:6s}: Failed to load")

        del r_imgs, g_imgs
        torch.cuda.empty_cache()

    # ----------------------------
    # 4. Save Evaluation Report
    # ----------------------------
    report_lines = [
        "=" * 60,
        "HAM10000 LoRA Generation - Evaluation Report",
        "=" * 60,
        "",
        f"Generated images dir : {gen_dir}",
        f"Real images dir      : {real_dir}",
        f"Generated images     : {gen_total}",
        f"Real images          : {real_total}",
        "",
        "--- Overall Metrics ---",
        f"Inception Score (IS) : {is_mean_val:.4f} +/- {is_std_val:.4f}  (higher is better)",
        f"FID Score            : {overall_fid_val:.4f}  (lower is better)",
        "",
        "--- Per-class FID ---",
    ]

    for cid in CLASS_IDS:
        if cid in per_class_fid:
            report_lines.append(f"  {cid:6s} ({CLASS_NAMES[cid]:30s}): {per_class_fid[cid]:.4f}")
        else:
            report_lines.append(f"  {cid:6s} ({CLASS_NAMES[cid]:30s}): N/A")

    report_lines.append("")
    report_lines.append("=" * 60)

    report = "\n".join(report_lines)

    with open(results_file, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nResults saved to: {results_file}")

    return {
        "IS_mean": is_mean_val,
        "IS_std": is_std_val,
        "FID_overall": overall_fid_val,
        "FID_per_class": per_class_fid,
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAM10000 FID & IS Evaluation Script")
    parser.add_argument("--gen_dir", type=str, default=GEN_DIR,
                        help="Generated images directory")
    parser.add_argument("--real_dir", type=str, default=REAL_DIR,
                        help="Real images directory")
    parser.add_argument("--results_file", type=str, default=RESULTS_FILE,
                        help="Results output file")
    parser.add_argument("--num_real_per_class", type=int, default=500,
                        help="Number of real images to extract per class (aligned with generated count)")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip real image preparation step (use when already exists)")
    args = parser.parse_args()

    # Step 1: Prepare real images
    if not args.skip_prepare:
        prepare_real_images(
            real_output_dir=args.real_dir,
            num_per_class=args.num_real_per_class
        )

    # Step 2: Compute FID & IS
    compute_fid_is(
        gen_dir=args.gen_dir,
        real_dir=args.real_dir,
        results_file=args.results_file
    )
