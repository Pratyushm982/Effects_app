import numpy as np

from app.effect import Effect
from app.params import NumericParam
from app.math_utils import to_gray


class SolidPaletteMapEffect(Effect):
    name = "Solid Palette Map+"

    params = {
        "palette_size": NumericParam("Palette Size", 2, 1, 5, 1),

        # UI-controlled BGR colors
        "color1": (0, 0, 0),
        "color2": (255, 255, 255),
        "color3": (255, 0, 0),
        "color4": (0, 255, 0),
        "color5": (0, 0, 255),
    }

    def apply(self, image, params):
        # Luminance [0,1]
        gray = to_gray(image).astype(np.float32) / 255.0

        palette_size = int(params["palette_size"])
        palette_size = max(1, min(5, palette_size))

        # Collect palette colors
        colors = []
        for i in range(1, palette_size + 1):
            c = np.array(params[f"color{i}"], dtype=np.float32)
            colors.append(c)

        colors = np.stack(colors, axis=0)  # (N,3)

        # ─────────────────────────────
        # BUILD DISCRETE LUT (256)
        # ─────────────────────────────

        lut = np.zeros((256, 3), dtype=np.float32)

        bins = np.linspace(0, 256, palette_size + 1, dtype=np.int32)

        for i in range(palette_size):
            lut[bins[i] : bins[i + 1]] = colors[i]

        # ─────────────────────────────
        # APPLY LUT (VECTORIZED)
        # ─────────────────────────────

        idx = np.clip((gray * 255).astype(np.int32), 0, 255)
        out = lut[idx]

        return np.clip(out, 0, 255).astype(np.uint8)
