import cv2
import numpy as np

# STAGE 1: ADAPTIVE NOISE REDUCTION
def adaptive_noise_filter(img_channel):
    """Formula: f'(x,y) = g(x,y) - (v^2 / sigma^2) * (g(x,y) - mu)"""
    if not isinstance(img_channel, np.ndarray):
        return img_channel
    
    #sab : 3x3 local neighborhood
    img_float = img_channel.astype(np.float32)
    mu = cv2.blur(img_float, (3, 3))
    mu_sq = cv2.blur(img_float**2, (3, 3))
    sigma_sq = mu_sq - mu**2
    
    # v^2 : avg of all local variances
    v_sq = np.mean(sigma_sq)
    
    f_prime = img_float - (v_sq / (sigma_sq + 1e-6)) * (img_float - mu)
    return np.clip(f_prime, 0, 255).astype(np.uint8)

# STAGE 2: ILLUMINATION CORRECTION
def gaussian_illumination(img_channel):
    """Models light as a 2D Gaussian pattern to correct concave eye shape."""
    rows, cols = img_channel.shape[:2]
    sigma = 0.8 * rows
    
    # Creating coordinate grids
    ax = np.linspace(-(cols - 1) / 2., (cols - 1) / 2., cols)
    ay = np.linspace(-(rows - 1) / 2., (rows - 1) / 2., rows)
    xx, yy = np.meshgrid(ax, ay)
    
    # Gaussian distribution G(x,y)
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
    
    # Reflectance r(x,y) = f(x,y) / i(x,y)
    corrected = img_channel.astype(np.float32) / (kernel + 1e-6)
    return np.clip(corrected, 0, 255).astype(np.uint8)

#STAGE 3: CONTRAST ENHANCEMENT(CLAHE)
def apply_clahe(img_channel):
    """Parameters: 8x8 window and 0.8% clipping limit."""
    # OpenCV's clipLimit is scaled 0-255; 0.008 * 255 approx 2.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_channel)


# STAGE 4: ILLUMINATION CORRECTION (MEDIAN FILTER DIVIDE)
def median_divide_illumination(img_channel):
    """Divides channel by 25x25 median-filtered background to estimate reflectance."""
    # 25x25 window is larger than the maximum vessel diameter
    background = cv2.medianBlur(img_channel, 25).astype(np.float32)
    mean_bg = np.mean(background)

    #scaled by mean to restore intensity range
    corrected = (img_channel.astype(np.float32) / (background + 1e-6)) * mean_bg
    return np.clip(corrected, 0, 255).astype(np.uint8)


# STAGE 5: ILLUMINATION CORRECTION (5TH-DEGREE POLYNOMIAL SVD)
def polynomial_background_illumination(img_channel):
    """Fits a 21-term 5th-degree 2D polynomial to control points via SVD to model background."""
    rows, cols = img_channel.shape
    img_float = img_channel.astype(np.float32)

    # Build 21-term design matrix for 5th-degree bivariate polynomial
    def poly_features(xv, yv):
        return np.column_stack([
            np.ones_like(xv),
            xv, yv,
            xv**2, xv*yv, yv**2,
            xv**3, xv**2*yv, xv*yv**2, yv**3,
            xv**4, xv**3*yv, xv**2*yv**2, xv*yv**3, yv**4,
            xv**5, xv**4*yv, xv**3*yv**2, xv**2*yv**3, xv*yv**4, yv**5
        ])

    # Sample control points on a grid, excluding the central fovea region
    cx, cy = cols // 2, rows // 2
    fovea_r_sq = (min(rows, cols) // 5) ** 2
    step = max(rows // 8, cols // 8, 1)

    pts_row, pts_col = [], []
    for pr in range(0, rows, step):
        for pc in range(0, cols, step):
            if (pr - cy)**2 + (pc - cx)**2 > fovea_r_sq:
                pts_row.append(pr)
                pts_col.append(pc)

    pts_row = np.array(pts_row)
    pts_col = np.array(pts_col)
    # Normalizing coordinates to [-1, 1] for numerical stability
    xv_pts = pts_col / cols * 2 - 1
    yv_pts = pts_row / rows * 2 - 1

    # Solving for 21 polynomial coefficients using SVD
    A_mat = poly_features(xv_pts, yv_pts)
    b_vec = img_float[pts_row, pts_col]
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    # Evaluating polynomial over full image to get the illumination map i(x,y)
    xx, yy = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    A_all = poly_features(xx.ravel(), yy.ravel())
    illumination = (A_all @ coeffs).reshape(rows, cols)
    illumination = np.clip(illumination, 1e-6, None)

    # r(x,y) = f(x,y) / i(x,y)
    corrected = (img_float / illumination) * np.mean(illumination)
    return np.clip(corrected, 0, 255).astype(np.uint8)


# STAGE 6: ILLUMINATION CORRECTION (HOMOMORPHIC FILTERING - HSI)
def homomorphic_filter_hsi(img_bgr):
    """Applies homomorphic high-pass filter on HSI intensity to suppress low-freq illumination."""
    img_float = img_bgr.astype(np.float32) / 255.0
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # Intensity of HSI: I = (R + G + B) / 3
    I = (r + g + b) / 3.0
    I_safe = np.clip(I, 1e-6, 1.0)

    # Log-transform to convert multiplicative model to additive
    log_I = np.log(I_safe)

    rows, cols = I.shape
    Fshift = np.fft.fftshift(np.fft.fft2(log_I))

    # H(u,v) high-pass filter — Eq. 8: C=0.1, D0^2=150, gamma_L=0, gamma_H=1
    u = np.arange(rows) - rows // 2
    v = np.arange(cols) - cols // 2
    V, U = np.meshgrid(v, u)
    D2 = U**2 + V**2
    gamma_L, gamma_H, C, D0_sq = 0, 1, 0.1, 150
    H = (gamma_H - gamma_L) * (1 - np.exp(-C * D2 / D0_sq)) + gamma_L

    # Inverse FFT and exp to return from log domain
    log_I_corrected = np.real(np.fft.ifft2(np.fft.ifftshift(Fshift * H)))
    I_corrected = np.exp(log_I_corrected)

    # Applying per-pixel correction ratio to all BGR channels
    ratio = np.clip(I_corrected / I_safe, 0, 10)
    r_c = np.clip(r * ratio, 0, 1)
    g_c = np.clip(g * ratio, 0, 1)
    b_c = np.clip(b * ratio, 0, 1)

    return (np.stack([b_c, g_c, r_c], axis=2) * 255).astype(np.uint8)


# STAGE 7: ILLUMINATION CORRECTION (QUOTIENT METHOD - HSI)
def quotient_method_hsi(img_bgr):
    """Applies quotient-based illumination correction on HSI intensity component (Eq. 10)."""
    img_float = img_bgr.astype(np.float32) / 255.0
    b, g, r = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]

    # Intensity component of HSI
    I = (r + g + b) / 3.0
    I_uint8 = (I * 255).astype(np.uint8)

    # Is(x,y): smoothed intensity via 25x25 median filter
    Is = cv2.medianBlur(I_uint8, 25).astype(np.float32) / 255.0

    # l0: ideal illumination = mean of intensity image
    l0 = float(np.mean(I))

    #brighten dark regions, leave bright regions unchanged
    I0 = np.where(Is < l0, l0 * I / (Is + 1e-6), I)
    I0 = np.clip(I0, 0, 1)

    # Apply correction ratio to all BGR channels
    ratio = np.clip(I0 / (I + 1e-6), 0, 10)
    r_c = np.clip(r * ratio, 0, 1)
    g_c = np.clip(g * ratio, 0, 1)
    b_c = np.clip(b * ratio, 0, 1)

    return (np.stack([b_c, g_c, r_c], axis=2) * 255).astype(np.uint8)


# MAIN EXECUTION
def process_retina(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not find {image_path}")
        return

    #Split Channels
    b, g, r = cv2.split(img)

    #Denoise green channel and full image (3x3 window)
    denoised_g = adaptive_noise_filter(g)
    denoised_img = cv2.merge([
        adaptive_noise_filter(b),
        denoised_g,
        adaptive_noise_filter(r)
    ])

    illum_gaussian_g = gaussian_illumination(denoised_g)

    illum_median_g = median_divide_illumination(denoised_g)

    illum_poly_g = polynomial_background_illumination(denoised_g)

    illum_homo_img = homomorphic_filter_hsi(denoised_img)
    illum_homo_g = cv2.split(illum_homo_img)[1]

    illum_quotient_img = quotient_method_hsi(denoised_img)
    illum_quotient_g = cv2.split(illum_quotient_img)[1]

    #Applying clahe to each illumination output

    final_gaussian_g = apply_clahe(illum_gaussian_g)

    final_median_g = apply_clahe(illum_median_g)

    final_poly_g = apply_clahe(illum_poly_g)

    final_homo_g = apply_clahe(illum_homo_g)

    final_quotient_g = apply_clahe(illum_quotient_g)

    cv2.imwrite('step1_denoised.jpg', denoised_g)

    cv2.imwrite('step2a_illum_gaussian.jpg', illum_gaussian_g)
    cv2.imwrite('step2b_illum_median.jpg', illum_median_g)
    cv2.imwrite('step2c_illum_polynomial.jpg', illum_poly_g)
    cv2.imwrite('step2d_illum_homomorphic.jpg', illum_homo_img)
    cv2.imwrite('step2e_illum_quotient.jpg', illum_quotient_img)

    cv2.imwrite('step3a_final_gaussian.jpg', cv2.merge([b, final_gaussian_g, r]))
    cv2.imwrite('step3b_final_median.jpg', cv2.merge([b, final_median_g, r]))
    cv2.imwrite('step3c_final_polynomial.jpg', cv2.merge([b, final_poly_g, r]))
    cv2.imwrite('step3d_final_homomorphic.jpg', cv2.merge([b, final_homo_g, r]))
    cv2.imwrite('step3e_final_quotient.jpg', cv2.merge([b, final_quotient_g, r]))

    print("Success! All processed images saved.")

if __name__ == "__main__":
    process_retina('01_test.tif')
