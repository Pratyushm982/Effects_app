import numpy as np

from app.effect import Effect
from app.params import NumericParam
from app.math_utils import to_gray


class GradientMapEffect(Effect):
    name = "Gradient Map+"

    params = {
        "stop_count": NumericParam("Stops", 2, 1, 5, 1),

        "stop1_pos": NumericParam("Stop 1 Pos", 0.0, 0.0, 10.0, 0.1),
        "stop2_pos": NumericParam("Stop 2 Pos", 10.0, 0.0, 10.0, 0.1),
        "stop3_pos": NumericParam("Stop 3 Pos", 5.0, 0.0, 10.0, 0.1),
        "stop4_pos": NumericParam("Stop 4 Pos", 7.5, 0.0, 10.0, 0.1),
        "stop5_pos": NumericParam("Stop 5 Pos", 10.0, 0.0, 10.0, 0.1),

        # UI-controlled BGR colors
        "stop1_color": (0, 0, 0),
        "stop2_color": (255, 255, 255),
        "stop3_color": (255, 0, 0),
        "stop4_color": (0, 255, 0),
        "stop5_color": (0, 0, 255),
    }

    def apply(self, image, params):
        # Luminance [0,1]
        gray = to_gray(image).astype(np.float32) / 255.0

        # Collect stops
        stops = []
        count = int(params["stop_count"])

        for i in range(1, count + 1):
            pos = np.clip(params[f"stop{i}_pos"] / 10.0, 0.0, 1.0)
            color = np.array(params[f"stop{i}_color"], dtype=np.float32)
            stops.append((pos, color))

        # Sort (Photoshop-style)
        stops.sort(key=lambda s: s[0])

        # ─────────────────────────────
        # BUILD GRADIENT LUT (256)
        # ─────────────────────────────

        lut = np.zeros((256, 3), dtype=np.float32)

        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i + 1]

            i0 = int(p0 * 255)
            i1 = int(p1 * 255)

            if i1 <= i0:
                lut[i0] = c0
                continue

            t = np.linspace(0.0, 1.0, i1 - i0 + 1)
            lut[i0 : i1 + 1] = c0 * (1.0 - t[:, None]) + c1 * t[:, None]

        # Clamp ends
        lut[: int(stops[0][0] * 255)] = stops[0][1]
        lut[int(stops[-1][0] * 255) :] = stops[-1][1]

        # ─────────────────────────────
        # APPLY LUT (VECTORIZED)
        # ─────────────────────────────

        idx = np.clip((gray * 255).astype(np.int32), 0, 255)
        out = lut[idx]

        return np.clip(out, 0, 255).astype(np.uint8)
