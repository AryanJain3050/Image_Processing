import cv2
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift

# --- STAGE 1: ADAPTIVE NOISE REDUCTION [cite: 65] ---
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

# --- STAGE 2: ILLUMINATION CORRECTION (GAUSSIAN) [cite: 99] ---
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

# --- STAGE 3: CONTRAST ENHANCEMENT (CLAHE) [cite: 154] ---
def apply_clahe(img_channel):
    """Parameters: 8x8 window and 0.8% clipping limit."""
    # OpenCV's clipLimit is scaled 0-255; 0.008 * 255 approx 2.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_channel)

# --- MAIN EXECUTION PIPELINE ---
def process_retina(image_path):
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find {image_path}")
        return

    # 1. Split Channels (Process Green for best vessel contrast) 
    b, g, r = cv2.split(img)

    # 2. Denoise (3x3 window) [cite: 73]
    denoised_g = adaptive_noise_filter(g)

    # 3. Correct Illumination (Gaussian Method - preferred by doctors) [cite: 202]
    illum_fixed_g = gaussian_illumination(denoised_g)

    # 4. Enhance Contrast (CLAHE - preferred for computer detection) [cite: 289]
    final_g = apply_clahe(illum_fixed_g)

    # Reconstruct for visualization
    final_output = cv2.merge([b, final_g, r])

    # Save outputs for your report
    cv2.imwrite('step1_denoised.jpg', denoised_g)
    cv2.imwrite('step2_illumination.jpg', illum_fixed_g)
    cv2.imwrite('step3_final_contrast.jpg', final_output)
    print("Success! Processed images saved to your folder.")

if __name__ == "__main__":
    # Ensure you have an image named 'input.jpg' in your folder
    process_retina('input.jpg')
