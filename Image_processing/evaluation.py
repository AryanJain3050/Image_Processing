import cv2
import numpy as np
import os
import sys

# Import all preprocessing methods from the main pipeline
sys.path.insert(0, os.path.dirname(__file__))
from Image_processing import (
    adaptive_noise_filter,
    apply_clahe,
    gaussian_illumination,
    median_divide_illumination,
    polynomial_background_illumination,
    homomorphic_filter_hsi,
    quotient_method_hsi,
)

# DRIVE dataset paths
IMAGES_DIR  = 'DRIVE/training/images'
GT_DIR      = 'DRIVE/training/1st_manual'
MASK_DIR    = 'DRIVE/training/mask'


# --- QUANTITATIVE EVALUATION: COEFFICIENT OF VARIATION (CV) [cite: 179] ---
def calculate_mean_cv(channel, cell_size=31):
    """
    Calculates the mean CV across sampled cells [cite: 180].
    Formula: CV = Standard Deviation / Mean [cite: 178, 179].
    Optimal cell size: 31x31 pixels [cite: 177].
    """
    rows, cols = channel.shape[:2]
    cv_values = []

    # Sample multiple cells across the image [cite: 180]
    step = cell_size * 2
    for y in range(0, rows - cell_size, step):
        for x in range(0, cols - cell_size, step):
            cell = channel[y:y+cell_size, x:x+cell_size].astype(np.float32)
            mu = np.mean(cell)
            sigma = np.std(cell)
            if mu > 10:  # Filter out black background/empty pixels
                cv_values.append(sigma / mu)

    # Lower mean CV = more uniform illumination [cite: 180]
    return np.mean(cv_values) if cv_values else 0.0


# --- VESSEL SEGMENTATION: MATCHED FILTER (Zhang et al.) [cite: 36] ---
def matched_filter_segmentation(green_channel):
    """
    Vessel segmentation using a bank of matched filters at 12 orientations.
    Kernel: -exp(-x^2 / 2*sigma^2) along vessel axis. [cite: 36]
    Returns a binary vessel map (uint8, 0 or 255).
    """
    img = green_channel.astype(np.float32)
    sigma = 2.0
    half_len = int(3 * sigma)
    kernel_len = 2 * half_len + 1

    # Build 1D matched filter kernel along x-axis
    x = np.arange(-half_len, half_len + 1, dtype=np.float32)
    kernel_1d = -np.exp(-x**2 / (2 * sigma**2))
    kernel_1d -= kernel_1d.mean()  # zero DC component

    # Extend to 2D (vessel runs along y-axis in base orientation)
    kernel_2d = np.tile(kernel_1d.reshape(1, -1), (kernel_len, 1))

    # Apply at 12 orientations (0 to 165 degrees) and take max response
    max_response = np.zeros_like(img)
    for angle in range(0, 180, 15):
        M = cv2.getRotationMatrix2D(
            (kernel_len // 2, kernel_len // 2), float(angle), 1.0
        )
        rotated_kernel = cv2.warpAffine(
            kernel_2d, M, (kernel_len, kernel_len),
            flags=cv2.INTER_LINEAR
        )
        response = cv2.filter2D(img, cv2.CV_32F, rotated_kernel)
        max_response = np.maximum(max_response, response)

    # Otsu threshold on the max response to get binary vessel map
    response_norm = cv2.normalize(
        max_response, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, binary = cv2.threshold(
        response_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


# --- EVALUATION METRICS [cite: 36] ---
def compute_metrics(predicted, ground_truth, fov_mask):
    """
    Compute sensitivity, specificity, and accuracy against ground truth.
    Evaluation is restricted to pixels inside the FOV mask. [cite: 36]
    """
    # Binarize all inputs
    pred   = (predicted > 127).astype(np.uint8)
    gt     = (ground_truth > 127).astype(np.uint8)
    mask   = (fov_mask > 127).astype(bool)

    # Only evaluate inside FOV
    pred_fov = pred[mask]
    gt_fov   = gt[mask]

    TP = np.sum((pred_fov == 1) & (gt_fov == 1))
    TN = np.sum((pred_fov == 0) & (gt_fov == 0))
    FP = np.sum((pred_fov == 1) & (gt_fov == 0))
    FN = np.sum((pred_fov == 0) & (gt_fov == 1))

    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    accuracy    = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    return sensitivity, specificity, accuracy


# --- CV EVALUATION FOR ONE IMAGE (illumination correction) ---
def evaluate_cv_image(img_path):
    """Returns a dict of mean CV per illumination method for a single image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Skipping {img_path} — could not load.")
        return None

    b, g, r = cv2.split(img)
    denoised_g   = adaptive_noise_filter(g)
    denoised_img = cv2.merge([
        adaptive_noise_filter(b),
        denoised_g,
        adaptive_noise_filter(r)
    ])

    homo_img     = homomorphic_filter_hsi(denoised_img)
    quotient_img = quotient_method_hsi(denoised_img)

    return {
        'Original':              calculate_mean_cv(g),
        'Gaussian (A2)':         calculate_mean_cv(gaussian_illumination(denoised_g)),
        'Median Divide (A1)':    calculate_mean_cv(median_divide_illumination(denoised_g)),
        'Polynomial SVD (A3)':   calculate_mean_cv(polynomial_background_illumination(denoised_g)),
        'Homomorphic (M5)':      calculate_mean_cv(cv2.split(homo_img)[1]),
        'Quotient (M6)':         calculate_mean_cv(cv2.split(quotient_img)[1]),
    }


# --- SEGMENTATION EVALUATION FOR ONE IMAGE (contrast enhancement) ---
def evaluate_segmentation_image(img_path, gt_path, mask_path):
    """
    Runs all contrast enhancement pipelines on a single image and returns
    a dict of (sensitivity, specificity, accuracy) per method.
    """
    img  = cv2.imread(img_path)
    gt   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or gt is None or mask is None:
        print(f"  Skipping {img_path} — could not load files.")
        return None

    b, g, r = cv2.split(img)
    denoised_g   = adaptive_noise_filter(g)
    denoised_img = cv2.merge([
        adaptive_noise_filter(b),
        denoised_g,
        adaptive_noise_filter(r)
    ])

    homo_img     = homomorphic_filter_hsi(denoised_img)
    quotient_img = quotient_method_hsi(denoised_img)

    pipelines = {
        'Original (no preprocessing)': g,
        'Gaussian + CLAHE':            apply_clahe(gaussian_illumination(denoised_g)),
        'Median Divide + CLAHE (A1)':  apply_clahe(median_divide_illumination(denoised_g)),
        'Polynomial + CLAHE (A3)':     apply_clahe(polynomial_background_illumination(denoised_g)),
        'Homomorphic + CLAHE (M5)':    apply_clahe(cv2.split(homo_img)[1]),
        'Quotient + CLAHE (M6)':       apply_clahe(cv2.split(quotient_img)[1]),
    }

    results = {}
    for label, green in pipelines.items():
        binary = matched_filter_segmentation(green)
        sen, spe, acc = compute_metrics(binary, gt, mask)
        results[label] = (sen, spe, acc)

    return results


# --- MAIN: RUN BOTH EVALUATIONS OVER ALL TRAINING IMAGES ---
def run_evaluation():
    image_files = sorted([
        f for f in os.listdir(IMAGES_DIR) if f.endswith('_training.tif')
    ])

    cv_totals  = {}
    seg_totals = {}

    for img_file in image_files:
        num       = img_file.split('_')[0]
        img_path  = os.path.join(IMAGES_DIR, img_file)
        gt_path   = os.path.join(GT_DIR,   f'{num}_manual1.gif')
        mask_path = os.path.join(MASK_DIR, f'{num}_training_mask.gif')

        print(f"Processing image {num}...")

        # CV evaluation (illumination correction)
        cv_results = evaluate_cv_image(img_path)
        if cv_results:
            for label, val in cv_results.items():
                cv_totals.setdefault(label, []).append(val)

        # Segmentation evaluation (contrast enhancement)
        seg_results = evaluate_segmentation_image(img_path, gt_path, mask_path)
        if seg_results:
            for label, (sen, spe, acc) in seg_results.items():
                seg_totals.setdefault(label, {'sen': [], 'spe': [], 'acc': []})
                seg_totals[label]['sen'].append(sen)
                seg_totals[label]['spe'].append(spe)
                seg_totals[label]['acc'].append(acc)

    # --- CV COMPARISON TABLE ---
    print()
    print("-" * 45)
    print("ILLUMINATION CORRECTION — MEAN CV (Green Channel)")
    print("-" * 45)
    print(f"{'Method':<25} {'Mean CV':>10}")
    print("-" * 45)
    for label, vals in cv_totals.items():
        print(f"{label:<25} {np.mean(vals):>10.4f}")
    print("-" * 45)
    print("Lower CV = more uniform illumination.")

    # --- SEGMENTATION METRICS TABLE ---
    print()
    print("-" * 70)
    print("CONTRAST ENHANCEMENT — VESSEL SEGMENTATION METRICS")
    print("-" * 70)
    print(f"{'Method':<35} {'Sensitivity':>12} {'Specificity':>12} {'Accuracy':>10}")
    print("-" * 70)
    for label, vals in seg_totals.items():
        print(f"{label:<35} {np.mean(vals['sen']):>12.4f} "
              f"{np.mean(vals['spe']):>12.4f} {np.mean(vals['acc']):>10.4f}")
    print("-" * 70)


if __name__ == "__main__":
    run_evaluation()
