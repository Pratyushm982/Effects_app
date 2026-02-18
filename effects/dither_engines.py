import numpy as np
import cv2
# -------------------------------
# ERROR DITHER MATRICES
# -------------------------------
def error_diffusion(gray, kernel, cutoff):
    h, w = gray.shape
    out = gray.copy()

    for y in range(h):
        for x in range(w):
            old = out[y, x]

            # Per-pixel cutoff (adaptive ONLY for Ostromoukhov)
            pixel_cutoff = cutoff
            if kernel is ERROR_DIFFUSION_KERNELS["ostromoukhov"]:
                if old < 0.5:
                    pixel_cutoff = cutoff * 0.9
                else:
                    pixel_cutoff = cutoff * 1.1

            new = 1.0 if old >= pixel_cutoff else 0.0
            err = np.clip(old - new, -0.75, 0.75)

            # -------------------------------
            # Kernel personality (ONE place)
            # -------------------------------
            if kernel is ERROR_DIFFUSION_KERNELS["floyd"]:
                err *= 1.0
            elif kernel is ERROR_DIFFUSION_KERNELS["atkinson"]:
                err *= 1
            elif kernel is ERROR_DIFFUSION_KERNELS["burkes"]:
                err = err * 0.9
            elif kernel is ERROR_DIFFUSION_KERNELS["jarvis"]:
                err *= 0.85
            elif kernel is ERROR_DIFFUSION_KERNELS["stucki"]:
                err *= 0.8
            elif kernel is ERROR_DIFFUSION_KERNELS["sierra3"]:
                err *= 0.88
            elif kernel is ERROR_DIFFUSION_KERNELS["sierra_lite"]:
                err *= 1.05
            elif kernel is ERROR_DIFFUSION_KERNELS["two_row_sierra"]:
                err *= 1.
            elif kernel is ERROR_DIFFUSION_KERNELS["ostromoukhov"]:
                err *= 0.9
            elif kernel is ERROR_DIFFUSION_KERNELS["stevenson"]:
                err *= 0.75

            out[y, x] = new

            for row in kernel:
                for dx, dy, weight in row:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        out[ny, nx] += err * weight

    return np.clip(out, 0.0, 1.0)

def error_diffusion_directional(gray, kernel, cutoff, direction="horizontal",line_scale=1):
    """
    Directional error diffusion.
    direction: 'horizontal' or 'vertical'
    """
    h, w = gray.shape
    out = gray.copy()

    for y in range(h):
        # line spacing control
        if direction == "horizontal" and y % line_scale != 0:
            continue

        for x in range(w):
            if direction == "vertical" and x % line_scale != 0:
                continue

            old = out[y, x]
            new = 1.0 if old >= cutoff else 0.0
            err = np.clip(old - new, -0.75, 0.75)
            out[y, x] = new

            for row in kernel:
                for dx, dy, weight in row:
                    # Directional bias
                    if direction == "horizontal" and dy != 0:
                        continue
                    if direction == "vertical" and dx != 0:
                        continue

                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        out[ny, nx] += err * weight

    return np.clip(out, 0.0, 1.0)


# -------------------------------
# ORDERED DITHER MATRICES
# -------------------------------
BAYER_2 = (1 / 4) * np.array([
    [0, 2],
    [3, 1],
])

BAYER_4 = (1 / 16) * np.array([
    [0,  8,  2, 10],
    [12, 4, 14, 6],
    [3, 11, 1,  9],
    [15, 7, 13, 5],
])

BAYER_8 = (1 / 64) * np.array([
    [0, 48, 12, 60, 3, 51, 15, 63],
    [32, 16, 44, 28, 35, 19, 47, 31],
    [8, 56, 4, 52, 11, 59, 7, 55],
    [40, 24, 36, 20, 43, 27, 39, 23],
    [2, 50, 14, 62, 1, 49, 13, 61],
    [34, 18, 46, 30, 33, 17, 45, 29],
    [10, 58, 6, 54, 9, 57, 5, 53],
    [42, 26, 38, 22, 41, 25, 37, 21],
])

def generate_bayer(n):
    """Generate Bayer matrix of size n×n (n must be power of 2)."""
    if n == 2:
        return BAYER_2
    if n == 4:
        return BAYER_4
    if n == 8:
        return BAYER_8

    prev = generate_bayer(n // 2)
    return (1 / (n * n)) * np.block([
        [4 * prev + 0, 4 * prev + 2],
        [4 * prev + 3, 4 * prev + 1],
    ])

def ordered_dither(gray, matrix, strength=1.0):
    h, w = gray.shape
    mh, mw = matrix.shape
    out = np.zeros_like(gray)

    gmin = gray.min()
    gmax = gray.max()
    if gmax > gmin:
        gray_n = (gray - gmin) / (gmax - gmin)
    else:
        gray_n = gray.copy()

    order = mh * mw
    density_comp = np.clip(1.0 - 0.12 * np.log2(order), 0.55, 1.0)

    for y in range(h):
        for x in range(w):
            t = matrix[y % mh, x % mw]

            # Remap threshold distribution
            t = 0.5 + (t - 0.5) * density_comp

            blended_t = (1.0 - strength) * 0.5 + strength * t
            out[y, x] = 1.0 if gray[y, x] > blended_t else 0.0

    return out

def ordered_modulation(gray, matrix, modulation):
    """
    Ordered dither with per-pixel threshold modulation.
    modulation: 0–1 map (contrast or luminance driven)
    """
    h, w = gray.shape
    mh, mw = matrix.shape
    out = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            base_t = matrix[y % mh, x % mw]
            t = base_t * (1.0 - modulation[y, x])
            out[y, x] = 1.0 if gray[y, x] > t else 0.0

    return out

BITONE = np.array([
    [0.0, 0.9],
    [0.6, 0.3],
], dtype=np.float32)

# Mosaic (coarse Bayer look)
MOSAIC = np.kron(BAYER_2, np.ones((4, 4)))

# Bayer Void (blue-noise-like approximation)
def generate_bayer_void(size=8, iters=32):
    """
    Blue-noise-like threshold matrix using void-and-cluster approximation.
    Deterministic, non-structured, perceptually even.
    """
    mat = np.random.rand(size, size)

    for _ in range(iters):
        # Find densest and emptiest points
        hi = np.unravel_index(np.argmax(mat), mat.shape)
        lo = np.unravel_index(np.argmin(mat), mat.shape)

        # Push extremes toward center
        mat[hi] *= 0.95
        mat[lo] = min(mat[lo] * 1.05, 1.0)

    mat -= mat.min()
    mat /= mat.max() + 1e-6
    return mat.astype(np.float32)
BAYER_VOID = generate_bayer_void(8)

# -------------------------------
# PATTERN ENGINE (BINARY)
# -------------------------------
def pattern_checkers(gray, scale):
    h, w = gray.shape
    out = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            effective = max(2, int(scale * 0.6))
            checker = ((x // effective) + (y // effective)) % 2
            threshold = 0.5 + (checker - 0.5) * 0.4  # was 0.25/0.75 → too strong
            out[y, x] = 1.0 if gray[y, x] > threshold else 0.0

    return out

def pattern_grid(gray, scale):
    h, w = gray.shape
    out = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            if x % scale == 0 or y % scale == 0:
                out[y, x] = 1.0 if gray[y, x] > 0.5 else 0.0
            else:
                out[y, x] = 0.0

    return out

def pattern_stipple(gray, density):
    h, w = gray.shape
    out = np.zeros_like(gray)

    block = 4  # dot spacing

    # edge strength (very light)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    edges = gx + gy
    edges /= edges.max() + 1e-6

    for y in range(0, h, block):
        for x in range(0, w, block):
            g = gray[y:y+block, x:x+block].mean()
            e = edges[y:y+block, x:x+block].mean()

            # midtone bias (diamond-like)
            tone = g * (1.0 - g) * 4.0

            # edge emphasis (very subtle)
            prob = (0.7 * tone + 0.3 * e) * density

            if np.random.rand() < prob:
                cy = y + block // 2
                cx = x + block // 2
                if cy < h and cx < w:
                    out[cy, cx] = 1.0

    return out


 
def pattern_diamonds(gray, scale):
    h, w = gray.shape
    out = np.zeros_like(gray)

    half = scale // 2

    for y in range(h):
        for x in range(w):
            dx = abs((x % scale) - half)
            dy = abs((y % scale) - half)
            dist = (dx + dy) / max(half, 1)
            out[y, x] = 1.0 if gray[y, x] > dist else 0.0

    return out


def pattern_crosshatch(gray, scale):
    h, w = gray.shape
    out = np.zeros_like(gray)

    for y in range(h):
        for x in range(w):
            v = gray[y, x]
            on = False

            # light shading (midtones)
            if 0.40 < v < 0.85 and (x + y) % scale == 0:
                on = True

            # medium shading
            if 0.30 < v < 0.65 and (x - y) % scale == 0:
                on = True

            # dark shading (reduced dominance)
            if 0.20 < v < 0.45 and y % scale == 0:
                on = True

            # very dark → SPARSE verticals only
            if v < 0.25 and x % (scale * 2) == 0:
                on = True

            out[y, x] = 1.0 if on else 0.0

    return out



# -------------------------------
# GLITCH / MODULATION ENGINE
# -------------------------------
def wave_x(gray,pattern_strength,glitch_strength,wave_density,
    depth,luminance,pixel_size):
    h, w = gray.shape

    # ----------------------------------
    # FULL-RES COORDINATES (LOCKED)
    # ----------------------------------
    y = np.arange(h, dtype=np.float32).reshape(-1, 1)
    x = np.arange(w, dtype=np.float32).reshape(1, -1)

    wave_density = max(wave_density, 1.0)

    # ----------------------------------
    # DEPTH (UNCHANGED)
    # ----------------------------------
    depth = int(np.clip(depth, 2, 5))
    bands = np.linspace(0.0, 1.0, depth)

    idx = np.abs(gray[..., None] - bands).argmin(axis=-1)
    target = bands[idx]
    quant = gray + (target - gray) * 0.25

    # ----------------------------------
    # CLEAN SYMMETRICAL WAVE (UNCHANGED)
    # ----------------------------------
    freq = glitch_strength * (2.0 * np.pi / max(w, 1))
    amp = pattern_strength

    base_phase = (y / wave_density) * (2.0 * np.pi)
    phase = base_phase + np.sin(x * freq) * amp
    wave = np.sin(phase)

    # ----------------------------------
    # LUMINANCE DARKENING (UNCHANGED)
    # ----------------------------------
    wave_norm = (wave + 1.0) * 0.5
    darken = 1.0 - luminance * (1.0 - wave_norm)

    out = quant * darken
    out = np.clip(out, 0.0, 1.0)

    # ----------------------------------
    # DITHER (UNCHANGED, YOUR GOOD ONE)
    # ----------------------------------
    band_step = 1.0 / (depth - 1)
    band_dist = np.abs(out - target) / band_step
    band_dist = np.clip(band_dist, 0.0, 1.0)

    noise = (np.random.rand(h, w) - 0.5) * band_step * 0.75
    out += noise * (band_dist ** 1.2)
    out = np.clip(out, 0.0, 1.0)

    # ----------------------------------
    # PIXEL SIZE (ONLY VISUAL SAMPLING)
    # ----------------------------------
    if pixel_size > 1:
        out = cv2.resize(
            cv2.resize(
                out,
                (w // pixel_size, h // pixel_size),
                interpolation=cv2.INTER_AREA
            ),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    return out

def wave_y(
    gray,
    pattern_strength,
    glitch_strength,
    wave_density,
    depth,
    luminance,
    pixel_size,
):
    h, w = gray.shape

    y = np.arange(h, dtype=np.float32).reshape(-1, 1)
    x = np.arange(w, dtype=np.float32).reshape(1, -1)

    wave_density = max(wave_density, 1.0)

    depth = int(np.clip(depth, 2, 5))
    bands = np.linspace(0.0, 1.0, depth)

    idx = np.abs(gray[..., None] - bands).argmin(axis=-1)
    target = bands[idx]
    quant = gray + (target - gray) * 0.25

    freq = glitch_strength * (2.0 * np.pi / max(h, 1))
    amp = pattern_strength

    base_phase = (x / wave_density) * (2.0 * np.pi)
    phase = base_phase + np.sin(y * freq) * amp
    wave = np.sin(phase)

    wave_norm = (wave + 1.0) * 0.5
    darken = 1.0 - luminance * (1.0 - wave_norm)

    out = quant * darken
    out = np.clip(out, 0.0, 1.0)

    # same dither you already locked
    band_step = 1.0 / (depth - 1)
    band_dist = np.abs(out - target) / band_step
    band_dist = np.clip(band_dist, 0.0, 1.0)

    noise = (np.random.rand(h, w) - 0.5) * band_step * 0.75
    out += noise * (band_dist ** 1.2)
    out = np.clip(out, 0.0, 1.0)

    if pixel_size > 1:
        out = cv2.resize(
            cv2.resize(out, (w // pixel_size, h // pixel_size),
                       interpolation=cv2.INTER_AREA),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

    return out









def glitch_block_shift(gray, block_size, intensity):
    h, w = gray.shape
    out = gray.copy()

    for y in range(0, h, block_size):
        if np.random.rand() < intensity:
            shift = np.random.randint(-block_size, block_size)
            band = out[y:y+block_size]

            # HARD edge cut (no interpolation)
            out[y:y+block_size] = np.roll(band, shift, axis=1)

            # introduce threshold break (artifact feel)
            out[y:y+block_size] = (out[y:y+block_size] > 0.5).astype(np.float32)

    return out

def glitch_artifact(gray, block_size, intensity):
    h, w = gray.shape
    out = gray.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            if np.random.rand() < intensity:
                block = out[y:y+block_size, x:x+block_size]
                avg = block.mean()
                out[y:y+block_size, x:x+block_size] = (
                    avg > 0.5
                ).astype(np.float32)

    return out

def glitch_vhs(gray, block_size, intensity):
    h, w = gray.shape
    out = gray.copy()

    for y in range(0, h, block_size):
        jitter = int(np.random.randn() * block_size * intensity)
        band = out[y:y+block_size]

        # horizontal jitter
        band = np.roll(band, jitter, axis=1)

        # luminance degradation
        band = (band * 0.85 + 0.15 * np.random.rand(*band.shape))

        out[y:y+block_size] = (band > 0.5).astype(np.float32)

    return out

def glitch_threshold_mod(gray, strength):
    noise = np.random.randn(*gray.shape) * strength
    return np.clip(gray + noise, 0, 1)









def radial_mask(shape, power=1.5):
    h, w = shape
    cy, cx = h / 2, w / 2

    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)

    max_dist = np.sqrt(cx**2 + cy**2)
    mask = 1.0 - (dist / max_dist)

    return np.clip(mask ** power, 0, 1)

def glitch_radial_threshold(gray, strength):
    mask = radial_mask(gray.shape, power=1.6)
    noise = np.random.uniform(-strength, strength, gray.shape)
    return np.clip(gray + noise * mask, 0, 1)

def glitch_topography(gray, levels, edge_bias=0.0):
    q = np.floor(gray * levels) / levels

    # Extract contour lines
    contours = np.abs(gray - q)

    # Normalize
    contours /= contours.max() + 1e-6

    if edge_bias > 0:
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        edges = gx + gy
        edges /= edges.max() + 1e-6

        contours = contours * ((1 - edge_bias) + edge_bias * edges)

    # Binary contour lines
    return (contours > 0.5).astype(np.float32)

def glitch_displace_contour(gray, strength, edge_bias=0.0):
    h, w = gray.shape

    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Gradient magnitude & direction
    mag = np.sqrt(gx * gx + gy * gy)
    mag /= mag.max() + 1e-6

    # Normalized normals
    nx = gx / (mag + 1e-6)
    ny = gy / (mag + 1e-6)

    # Edge-aware displacement mask
    mask = (1.0 - edge_bias) + edge_bias * mag

    # Coordinate grid
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # Apply displacement
    dx = nx * strength * mask
    dy = ny * strength * mask

    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)

    displaced = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    return displaced


# -------------------------------
#  ENGINE DEFINE
# -------------------------------
ERROR_DIFFUSION_KERNELS = {
    "floyd": (
        [(1, 0, 7/16)],
        [(-1, 1, 3/16), (0, 1, 5/16), (1, 1, 1/16)],
    ),
    "atkinson": (
        [(1, 0, 1/8), (2, 0, 1/8)],
        [(-1, 1, 1/8), (0, 1, 1/8), (1, 1, 1/8)],
        [(0, 2, 1/8)],
    ),
    "jarvis": (
        [(1, 0, 7/48), (2, 0, 5/48)],
        [(-2, 1, 3/48), (-1, 1, 5/48), (0, 1, 7/48),
         (1, 1, 5/48), (2, 1, 3/48)],
        [(-2, 2, 1/48), (-1, 2, 3/48), (0, 2, 5/48),
         (1, 2, 3/48), (2, 2, 1/48)],
    ),
    "stucki": (
        [(1, 0, 8/42), (2, 0, 4/42)],
        [(-2, 1, 2/42), (-1, 1, 4/42), (0, 1, 8/42),
         (1, 1, 4/42), (2, 1, 2/42)],
        [(-2, 2, 1/42), (-1, 2, 2/42), (0, 2, 4/42),
         (1, 2, 2/42), (2, 2, 1/42)],
    ),
    "burkes": (
        [(1, 0, 8/32), (2, 0, 4/32)],
        [(-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 8/32),
         (1, 1, 4/32), (2, 1, 2/32)],
    ),
    "sierra3": (
        [(1, 0, 5/32), (2, 0, 3/32)],
        [(-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 5/32),
         (1, 1, 4/32), (2, 1, 2/32)],
        [(-1, 2, 2/32), (0, 2, 3/32), (1, 2, 2/32)],
    ),
    "two_row_sierra": (
        [(1, 0, 4/16), (2, 0, 3/16)],
        [(-2, 1, 1/16), (-1, 1, 2/16), (0, 1, 3/16),
         (1, 1, 2/16), (2, 1, 1/16)],
    ),
    "sierra_lite": (
        [(1, 0, 2/4)],
        [(-1, 1, 1/4), (0, 1, 1/4)],
    ),
    "stevenson": (
        [(2, 0, 32/200)],
        [(-3, 1, 12/200), (-1, 1, 26/200),
         (1, 1, 30/200), (3, 1, 16/200)],
        [(-2, 2, 12/200), (0, 2, 26/200), (2, 2, 12/200)],
        [(-3, 3, 5/200), (-1, 3, 12/200),
         (1, 3, 12/200), (3, 3, 5/200)],
    ),
    "ostromoukhov": (
        [(1, 0, 8/42), (2, 0, 4/42)],
        [(-2, 1, 2/42), (-1, 1, 4/42), (0, 1, 8/42),
         (1, 1, 4/42), (2, 1, 2/42)],
        [(-2, 2, 1/42), (-1, 2, 2/42), (0, 2, 4/42),
         (1, 2, 2/42), (2, 2, 1/42)],
    ),
}


ORDERED_MATRICES = {
    "bayer2": BAYER_2,
    "bayer4": BAYER_4,
    "bayer8": BAYER_8,
    "bayer16": generate_bayer(16),
    "bitone": BITONE,
    "mosaic": MOSAIC,
    "bayer_void": BAYER_VOID,
    "random": None,  # generated per frame
}

PATTERN_ENGINES = {
    "checkers": pattern_checkers,
    "grid": pattern_grid,
    "diamonds": pattern_diamonds,
    "stipple": pattern_stipple,
    "crosshatch": pattern_crosshatch,
}

GLITCH_ENGINES = {
    # "dither_wave_x": dither_wave_x,
    # "dither_wave_y": dither_wave_y,
    "block": glitch_block_shift,
    "threshold": glitch_threshold_mod,
    "radial_threshold": glitch_radial_threshold,
    "topography": glitch_topography,
    "displace_contour": glitch_displace_contour,
    # "dither_wave_alt": dither_wave_alt,
    "artifact": glitch_artifact,
    "vhs": glitch_vhs,
    "wave_x": wave_x,
    "wave_y": wave_y,
}






# def dither_wave_x(gray, pattern_strength, glitch_strength, wave_density):
#     h, w = gray.shape
#     y, x = np.meshgrid(
#         np.linspace(0, 1, h),
#         np.linspace(0, 1, w),
#         indexing="ij"
#     )

#     # --- NEW PARAMETER SEMANTICS ---
#     # wave_density = pixels per wave (vertical repetition)
#     wave_density = max(4, wave_density)
#     band_count = h / wave_density

#     # glitch_strength = wave amplitude
#     amp = glitch_strength * 0.25

#     # wavelength of horizontal wave = wave_density (intuitive)
#     wavelength = wave_density / w
#     freq = 2 * np.pi / wavelength

#     # --- continuous wave phase ---
#     phase = y + np.sin(x * freq) * amp

#     band_pos = phase * band_count
#     band_frac = band_pos % 1.0

#     # distance to wave center (light crest / dark trough)
#     dist = np.abs(band_frac - 0.5)

#     # --- smooth darkening inside the wave (NOT black gaps) ---
#     separator_width = 0.5
#     band_mask = 1.0 - np.clip(dist / separator_width, 0, 1)

#     # preserve full image, just modulate luminance
#     out = gray * (0.65 + 0.35 * band_mask)

#     # --- DITHER (pattern_strength ONLY) ---
#     luma_mask = 1.0 - np.abs(gray - 0.5) * 2.0
#     out = micro_dither(
#         out,
#         strength=(0.08 + 0.12 * pattern_strength) * luma_mask
#     )

#     return np.clip(out, 0.0, 1.0)


# def dither_wave_y(gray, pattern_strength, glitch_strength, wave_density):
#     h, w = gray.shape
#     y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

#     wavelength = max(4, wave_density)
#     amp = glitch_strength * 0.25
#     freq = 2 * np.pi / wavelength

#     wave = np.sin(x * freq)
#     out = gray + wave * amp

#     sway = np.sin(x * freq + gray * 2.0) * glitch_strength * 2.0
#     map_y = np.clip(y + sway, 0, h - 1).astype(np.float32)

#     out = cv2.remap(
#         out,
#         x.astype(np.float32),
#         map_y,
#         interpolation=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT
#     )

#     dither_strength = 0.08 + 0.12 * pattern_strength
#     out = micro_dither(out, strength=dither_strength)

#     return np.clip(out, 0.0, 1.0)

# def dither_wave_alt(gray, pattern_strength, glitch_strength, wave_density):
#     h, w = gray.shape
#     y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

#     wave_density = max(2, int(wave_density))
#     diag = (x + y) // wave_density
#     phase = ((x + y) % wave_density) / wave_density

#     amp = glitch_strength * wave_density * (0.2 + 0.6 * gray)

#     in_wave = phase < 0.5
#     phase = phase * 2.0

#     offset = np.sin(phase * 2 * np.pi + diag * 0.9) * amp

#     sample_x = np.clip(x + offset, 0, w - 1).astype(np.float32)
#     sample_y = np.clip(y + offset, 0, h - 1).astype(np.float32)

#     waved = cv2.remap(
#         gray, sample_x, sample_y,
#         interpolation=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_REFLECT
#     )

#     dark = gray * 0.12
#     out = np.where(in_wave, waved, dark)

#     edge_mask = np.abs(phase - 1.0)
#     luma_mask = 1.0 - np.abs(gray - 0.5) * 2.0
#     dither_mask = edge_mask * luma_mask

#     out = micro_dither(out, strength=0.14 * pattern_strength * dither_mask)

#     return np.clip(out, 0.0, 1.0)

# def soft_quantize(gray, levels):
#     """
#     Gentle banding, keeps grayscale.
#     levels: 4–8 is ideal for Dither Boy look
#     """
#     levels = max(2, int(levels))
#     return np.round(gray * (levels - 1)) / (levels - 1)
    
# def micro_dither(gray, strength=0.06):
#     """
#     strength can be:
#     - scalar
#     - per-pixel mask (same shape as gray)
#     """
#     noise = (np.random.rand(*gray.shape) - 0.5)
#     return np.clip(gray + noise * strength, 0.0, 1.0)
