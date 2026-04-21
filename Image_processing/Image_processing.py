import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

# STAGE 1: ADAPTIVE NOISE REDUCTION [cite: 65]
def adaptive_noise_filter(img_channel):
    """Formula: f'(x,y) = g(x,y) - (v^2 / sigma^2) * (g(x,y) - mu)"""
    if not isinstance(img_channel, np.ndarray):
        return img_channel
    
    # sab is a 3x3 local neighborhood [cite: 73]
    img_float = img_channel.astype(np.float32)
    mu = cv2.blur(img_float, (3, 3))
    mu_sq = cv2.blur(img_float**2, (3, 3))
    sigma_sq = mu_sq - mu**2
    
    # v^2 is the average of all local variances [cite: 71]
    v_sq = np.mean(sigma_sq)
    
    f_prime = img_float - (v_sq / (sigma_sq + 1e-6)) * (img_float - mu)
    return np.clip(f_prime, 0, 255).astype(np.uint8)

# STAGE 2: ILLUMINATION CORRECTION (GAUSSIAN) [cite: 99]
def gaussian_illumination(img_channel):
    """Models light as a 2D Gaussian pattern to correct concave eye shape."""
    rows, cols = img_channel.shape[:2]
    # Parameter: sigma = 0.8 * row size [cite: 101]
    sigma = 0.8 * rows
    
    # Create coordinate grids
    ax = np.linspace(-(cols - 1) / 2., (cols - 1) / 2., cols)
    ay = np.linspace(-(rows - 1) / 2., (rows - 1) / 2., rows)
    xx, yy = np.meshgrid(ax, ay)
    
    # Gaussian distribution G(x,y)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    
    # Reflectance r(x,y) = f(x,y) / i(x,y) [cite: 86]
    corrected = img_channel.astype(np.float32) / (kernel + 1e-6)
    return np.clip(corrected, 0, 255).astype(np.uint8)

#STAGE 3: CONTRAST ENHANCEMENT (CLAHE) [cite: 154]
def apply_clahe(img_channel):
    """Parameters: 8x8 window and 0.8% clipping limit."""
    # OpenCV's clipLimit is scaled 0-255; 0.008 * 255 approx 2.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_channel)


# --- QUANTITATIVE EVALUATION: COEFFICIENT OF VARIATION (CV) [cite: 179] ---
def calculate_mean_cv(channel, cell_size=31):
    """
    Calculates the mean CV across sampled cells[cite: 180].
    Formula: CV = Standard Deviation / Mean[cite: 178, 179].
    Optimal cell size: 31x31 pixels[cite: 177].
    """
    rows, cols = channel.shape[:2]
    cv_values = []
    
    # Sample multiple cells across the image to find the average variation [cite: 180]
    # This mimics the study's approach of evaluating multiple regions [cite: 174]
    step = cell_size * 2
    for y in range(0, rows - cell_size, step):
        for x in range(0, cols - cell_size, step):
            cell = channel[y:y+cell_size, x:x+cell_size].astype(np.float32)
            mu = np.mean(cell)
            sigma = np.std(cell)
            
            if mu > 10: # Filter out purely black background/empty pixels
                cv_values.append(sigma / mu)
    
    # The mean value of CVs indicates the illumination variation [cite: 180]
    return np.mean(cv_values) if cv_values else 0.0


#MAIN EXECUTION PIPELINE
def process_retina(image_path):
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find {image_path}")
        return

    # 1. Split Channels (Process Green for best vessel contrast) 
    b, g, r = cv2.split(img)

    # 2. Statistical Baseline: Original Green CV
    orig_cv = calculate_mean_cv(g)

    # 3. Denoise (3x3 window) [cite: 73]
    denoised_g = adaptive_noise_filter(g)

    # 4. Correct Illumination (Gaussian Method) [cite: 202]
    illum_fixed_g = gaussian_illumination(denoised_g)

    # 5. Statistical Check: Illumination-Corrected CV
    corrected_cv = calculate_mean_cv(illum_fixed_g)

    # 6. Enhance Contrast (CLAHE) [cite: 289]
    final_g = apply_clahe(illum_fixed_g)

    # Reconstruct for visualization
    final_output = cv2.merge([b, final_g, r])

    # Save outputs
    cv2.imwrite('step1_denoised.jpg', denoised_g)
    cv2.imwrite('step2_illumination.jpg', illum_fixed_g)
    cv2.imwrite('step3_final_contrast.jpg', final_output)
    print("Success! Processed images saved to your folder.")

    # Final Report Printout
    print("-" * 30)
    print("IMAGE PROCESSING SUCCESSFUL")
    print("-" * 30)
    print(f"Original Green CV:  {orig_cv:.4f}")
    print(f"Corrected Green CV: {corrected_cv:.4f}")
    print("-" * 30)
    print("Key Insight: A lower CV indicates more uniform illumination.")
    print("If Corrected CV < Original CV, your lighting normalization worked!")

if __name__ == "__main__":
    process_retina('01_test.tif')
