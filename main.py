import cv2
from effects.dither import DitherEffect

# ─────────────────────────────
# Load test image
# ─────────────────────────────

img = cv2.imread("assets/test.jpg")
if img is None:
    raise FileNotFoundError("assets/test.jpg not found")

fx = DitherEffect()

# ─────────────────────────────
# Styles to audit
# ─────────────────────────────

audit_styles = [
    "wave_x",
    "wave_y",
]

# ─────────────────────────────
# BASE PARAMETER PRESET
# (this is what you tweak)
# ─────────────────────────────

base_params = {
    "cutoff": 0.0,
    "pixel_size": 1,             
    "contrast":1,
    "midtones": 1,
    "highlights": 1,
    "luminance_threshold": 1,
    "invert": "off",

    # keep these stable for now
    "pattern_strength": 4,
    "glitch_strength": 30,
    "wave_density": 15,
    "depth": 5,
    "luminance": 0.8,


}

# ─────────────────────────────
# Run audit
# ─────────────────────────────

for style in audit_styles:
    # Start from effect defaults
    params = {k: v.value for k, v in fx.params.items()}

    # Apply global overrides
    params.update(base_params)

    # Per-style override (if needed later)
    params["style"] = style

    try:
        out = fx.apply(img, params)
        cv2.imwrite(f"_audit_{style}.png", out)
        print(f"[OK] {style}")
    except Exception as e:
        print(f"[FAIL] {style}: {e}")

print("Dither audit complete")
