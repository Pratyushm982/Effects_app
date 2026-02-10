import cv2
import numpy as np

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import to_gray, normalize, denormalize

from effects.dither_registry import get_style, get_style_keys
from effects.dither_engines import (
    error_diffusion,
    error_diffusion_directional,
    ordered_dither,
    ordered_modulation,
    ERROR_DIFFUSION_KERNELS,
    ORDERED_MATRICES,
    PATTERN_ENGINES,
    GLITCH_ENGINES,
    glitch_block_shift,
    glitch_artifact,
    glitch_vhs,
    dither_wave_x,
    dither_wave_y,
    dither_wave_alt,
    soft_quantize,
    micro_dither,
    dither_waveform,
)

def apply_tonal_shaping(
    gray,
    contrast=1.0,
    midtones=1.0,
    highlights=1.0,
    luminance_threshold=0.0,
    invert=False,
):
    """
    Dither-only tonal shaping.
    Operates in normalized grayscale (0–1).
    """

    # Invert first (matches Dither Boy behavior)
    if invert:
        gray = 1.0 - gray

    # Contrast (around mid-gray)
    if contrast != 1.0:
        gray = (gray - 0.5) * contrast + 0.5

    gray = np.clip(gray, 0.0, 1.0)

    # Midtones (gamma-like)
    if midtones != 1.0:
        gray = np.power(gray, 1.0 / midtones)

    # Highlights compression (prevents white blowout)
    if highlights != 1.0:
        gray = 1.0 - np.power(1.0 - gray, highlights)

    # Luminance threshold (pre-dither cutoff bias)
    if luminance_threshold > 0.0:
        t = luminance_threshold
        gray = np.where(
            gray < t,
            gray * (1.0 / max(t, 1e-6)),
            gray
        )

    return np.clip(gray, 0.0, 1.0)

class DitherEffect(Effect):
    name = "Dither+"

    params = {
        "style": ChoiceParam(
            "Style",
            "floyd",
            get_style_keys()
        ),

        "cutoff": NumericParam(
            "Cutoff",
            0.0, 0.0, 1.0, 0.01
        ),

        "pattern_strength": NumericParam(
            "Pattern Strength",
            1.0, 0.0, 1.0, 0.01
        ),

        "glitch_strength": NumericParam(
            "Glitch Strength",
            1.0, 0.0, 5.0, 0.01
        ),

        "glitch_direction": NumericParam(
            "Glitch Direction",
            0.0, -1.0, 1.0, 0.01
        ),

        "pixel_size": NumericParam(
            "Pixel Size",
            1, 1, 16, 1
        ),
        "contrast": NumericParam(
            "Contrast",
            1.0, 0.0, 2.0, 0.01
        ),

        "midtones": NumericParam(
            "Midtones",
            0.0, 0.5, 2.0, 0.01
        ),

        "highlights": NumericParam(
            "Highlights",
            1.0, 0.5, 2.0, 0.01
        ),

        "luminance_threshold": NumericParam(
            "Luminance Threshold",
            0.0, 0.0, 1.0, 0.01
        ),

        "invert": ChoiceParam(
            "Invert",
            "off",
            ["off", "on"]
        ),

        "wave_density": NumericParam(
            "Wave Density",
            1.0, 1.0, 40.0, 0.1
        ),

        "depth": NumericParam(
            "Depth",
            0.0, 0.0, 1.0, 0.01
        ),
        "luminance": NumericParam(
            "Luminance",
            0.5,0.0,1.0,0.01
        ),

    
    }

    def apply(self, image, params):
        style_key = params["style"]
        cutoff = params["cutoff"]
        wave_density = params.get("wave_density", 1.0)
        pattern_strength = params.get("pattern_strength", 1.0)
        glitch_strength = params.get("glitch_strength", 1.0)
        glitch_dir = params.get("glitch_direction", 0.0)
        contrast_amt = params.get("contrast", 1.0)

        # Nonlinear glitch scaling (artist tuned)
        glitch_power = glitch_strength ** 1.5

        # Get style definition
        style = get_style(style_key)

        # Convert image → normalized grayscale
        gray = normalize(to_gray(image))
        if style["engine"] != "glitch" or not style["config"]["type"].startswith("dither_wave"):
            gray = apply_tonal_shaping(
                gray,
                contrast=params.get("contrast", 1.0),
                midtones=params.get("midtones", 1.0),
                highlights=params.get("highlights", 1.0),
                luminance_threshold=params.get("luminance_threshold", 0.0),
                invert=(params.get("invert", "off") == "on"),
            )

        # -------------------------------
        # Directional contrast maps
        # -------------------------------
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))  # horizontal edges
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))  # vertical edges

        gx /= gx.max() + 1e-6
        gy /= gy.max() + 1e-6
        gx = gx ** 0.8
        gy = gy ** 0.8

        # -------------------------------
        # Global contrast magnitude (for luminance engines)
        # -------------------------------
        contrast = np.sqrt(gx * gx + gy * gy)
        contrast /= contrast.max() + 1e-6
        contrast = contrast ** 0.8

        # User control
        contrast_amt = params.get("contrast_aware", 0.0)

        # Direction-specific masks
        contrast_x = (1.0 - contrast_amt) + contrast_amt * gy  # for wave_x
        contrast_y = (1.0 - contrast_amt) + contrast_amt * gx  # for wave_y

        # Fallback/global mask (for block & threshold)
        glitch_mask = (contrast_x + contrast_y) * 0.5



        # ===============================
        # ERROR DIFFUSION ENGINE
        # ===============================
        if style["engine"] == "error_diffusion":
            kernel_name = style["config"]["kernel"]

            if kernel_name not in ERROR_DIFFUSION_KERNELS:
                raise ValueError(f"Unknown diffusion kernel: {kernel_name}")

            kernel = ERROR_DIFFUSION_KERNELS[kernel_name]

            # --- Luminance / directional variants ---
            if style_key == "stucki_lines":
                direction = "horizontal" if glitch_dir >= 0 else "vertical"
                out = error_diffusion_directional(
                    gray, kernel, cutoff, direction=direction
                )

            elif style_key == "smooth_diffuse":
                # smoother = slightly relaxed cutoff
                out = error_diffusion(
                    gray, kernel, cutoff * 0.9
                )
            elif style_key == "atkinson_line_modulation":
                direction = "horizontal" if glitch_dir >= 0 else "vertical"

                # Slightly relaxed cutoff for sketch-like feel
                out = error_diffusion_directional(
                    gray,
                    kernel,
                    cutoff * 0.85,
                    direction=direction,
                    line_scale=2,
                )

            else:
                out = error_diffusion(gray, kernel, cutoff)

        # ===============================
        # ORDERED DITHER ENGINE
        # ===============================
        elif style["engine"] == "ordered":
            matrix_name = style["config"]["matrix"]

            if matrix_name == "random":
                matrix = np.random.rand(4, 4)
                matrix = 0.5 + (matrix - 0.5) * 0.75
            else:
                if matrix_name not in ORDERED_MATRICES:
                    raise ValueError(f"Unknown ordered matrix: {matrix_name}")
                matrix = ORDERED_MATRICES[matrix_name]

            if style_key == "ordered_modulation":
                ordered = ordered_modulation(gray, matrix, contrast)
                out = ordered
            else:
                out = ordered_dither(
                    gray,
                    matrix,
                    strength=pattern_strength,
                )

        
        # ===============================
        # PATTERN ENGINE (CORRECTED)
        # ===============================
        elif style["engine"] == "pattern":
            cfg = style["config"]
            pattern_name = cfg["pattern"]

            func = PATTERN_ENGINES[pattern_name]

            pixel_size = max(1, int(params.get("pixel_size", 1)))
            h, w = gray.shape

            # Downscale ONLY for visibility
            gray_small = cv2.resize(
                gray,
                (w // pixel_size, h // pixel_size),
                interpolation=cv2.INTER_AREA,
            )

            # Generate binary pattern
            if pattern_name == "stipple":
                density = cfg.get("density", 1.0)
                pattern_small = func(gray_small, density * pattern_strength)
            else:
                scale = cfg.get("scale", 8)
                pattern_small = func(gray_small, scale)

            # Upscale back
            pattern = cv2.resize(
                pattern_small,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Binary mix with original image
            if pattern_strength >= 1.0:
                out = pattern
            else:
                base = (gray > cutoff).astype(np.float32)
                out = (pattern_strength * pattern + (1.0 - pattern_strength) * base) > 0.5
                out = out.astype(np.float32)


        # ===============================
        # GLITCH / MODULATION ENGINE
        # ===============================
        elif style["engine"] == "glitch":
            cfg = style["config"]
            glitch_type = cfg["type"]

            if glitch_type not in GLITCH_ENGINES:
                raise ValueError(f"Unknown glitch type: {glitch_type}")

            func = GLITCH_ENGINES[glitch_type]

            # Global contrast influence (safe scalar)
            mask_strength = np.mean(glitch_mask)
            
            if glitch_type == "waveform":
                out = dither_waveform(
                    gray,
                    pattern_strength=params["pattern_strength"],  # wave amplitude
                    glitch_strength=params["glitch_strength"],    # wave frequency
                    wave_density=params["wave_density"],          # band spacing
                    depth=params["depth"],                         # luminance bands
                    luminance=params["luminance"],                 # dark trough strength
                    pixel_size=params["pixel_size"],               # visibility scaling
                )

                out = np.clip(out, 0.0, 1.0)




            elif glitch_type == "dither_wave_x":
                out = dither_wave_x(
                    gray,
                    pattern_strength=params.get("pattern_strength", 1.0),
                    glitch_strength=glitch_strength,
                    wave_density=params.get("wave_density", 1),
                )

                # enhance wave contrast
                out = np.clip((out - 0.5) * 1.35 + 0.5, 0.0, 1.0)

                # banding (THIS creates visible wave steps)
                band_levels = int(3 + pattern_strength * 6)
                band_levels = np.clip(band_levels, 3, 9)
                out = soft_quantize(out, band_levels)

                # break band edges (THIS is the dither)
                out = micro_dither(
                    out,
                    strength=0.04 + 0.04 * glitch_strength
                )

            elif glitch_type == "dither_wave_y":
                out = dither_wave_y(
                    gray,
                    pattern_strength=params.get("pattern_strength", 1.0),
                    glitch_strength=glitch_strength,
                    wave_density=params.get("wave_density", 1),
                )

                # enhance wave contrast
                out = np.clip((out - 0.5) * 1.35 + 0.5, 0.0, 1.0)

                # banding (THIS creates visible wave steps)
                band_levels = int(3 + pattern_strength * 6)
                band_levels = np.clip(band_levels, 3, 9)
                out = soft_quantize(out, band_levels)

                # break band edges (THIS is the dither)
                out = micro_dither(
                    out,
                    strength=0.04 + 0.04 * glitch_strength
                )


            elif glitch_type == "dither_wave_alt":
                out = dither_wave_alt(
                    gray,
                    pattern_strength=params.get("pattern_strength", 1.0),
                    glitch_strength=glitch_strength,
                    wave_density=params.get("wave_density", 1),
                )

                # enhance wave contrast
                out = np.clip((out - 0.5) * 1.35 + 0.5, 0.0, 1.0)

                # banding (THIS creates visible wave steps)
                band_levels = int(3 + pattern_strength * 6)
                band_levels = np.clip(band_levels, 3, 9)
                out = soft_quantize(out, band_levels)

                # break band edges (THIS is the dither)
                out = micro_dither(
                    out,
                    strength=0.04 + 0.04 * glitch_strength
                )


            elif glitch_type == "block":
                base_block = cfg.get("block_size", 24)
                base_intensity = cfg.get("intensity", 0.4)
                intensity = np.clip(base_intensity * glitch_power * mask_strength,0.0,1.0,)
                block_size = int(base_block * (1 + abs(glitch_dir)))
                out = glitch_block_shift(gray,block_size,intensity)

            elif glitch_type == "artifact":
                block_size = cfg.get("block_size", 16)
                intensity = np.clip(cfg.get("intensity", 0.6) * glitch_power,0.0,1.0,)
                out = glitch_artifact(gray,block_size,intensity)

            elif glitch_type == "vhs":
                block_size = cfg.get("block_size", 32)
                intensity = np.clip(cfg.get("intensity", 0.5) * glitch_power,0.0,1.0,)
                out = glitch_vhs(gray,block_size,intensity)


            elif glitch_type == "threshold":
                base_strength = cfg.get("strength", 0.15)
                strength = base_strength * glitch_power

                if glitch_type == "glitch":
                    noise = np.random.randn(*gray.shape) * strength
                    mask = (np.random.rand(*gray.shape) < glitch_power * 0.5)
                    out = np.where(mask, gray + noise, gray)
                    out = np.clip(out, 0, 1)
                else:
                    out = func(gray, strength * glitch_mask)



            elif glitch_type == "topography":
                base_levels = cfg.get("levels", 12)

                # More glitch strength → more contour density
                levels = int(base_levels * glitch_power)
                levels = max(2, levels)

                out = func(
                    gray,
                    levels,
                    contrast_amt,
                )
            elif glitch_type == "displace_contour":
                base_strength = cfg.get("strength", 1.0)

                # Scale displacement with glitch strength
                disp = base_strength * glitch_power

                out = func(
                    gray,
                    disp,
                    contrast_amt,
                )


        # ===============================
        # UNKNOWN ENGINE
        # ===============================
        else:
            raise ValueError(f"Unsupported dither engine: {style['engine']}")

        # Convert back to image
        out = denormalize(out)
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        # -------------------------------
        # Pixel size (visibility control)
        # -------------------------------
        pixel_size = int(params.get("pixel_size", 1))

        if pixel_size > 1:
            h, w = out.shape[:2]

            out = cv2.resize(
                out,
                (w // pixel_size, h // pixel_size),
                interpolation=cv2.INTER_NEAREST,
            )
            out = cv2.resize(
                out,
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )

        return out
