import cv2
import numpy as np

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize, to_gray


class GlowEffect(Effect):
    name = "Glow+"

    params = {
        "mode": ChoiceParam(
            "Mode",
            "multiscale",
            ["basic", "multiscale", "gamma", "edge_aware"],
        ),

        "threshold": NumericParam("Threshold", 0.5, 0.0, 1.0, 0.01),
        "softness": NumericParam("Softness", 0.3, 0.0, 1.0, 0.01),

        # FIXED RANGE
        "radius": NumericParam("Radius", 20, 1, 100, 1),

        # STRONG BUT CONTROLLED
        "intensity": NumericParam("Intensity", 1.0, 0.0, 5.0, 0.05),

        "glow_gamma": NumericParam("Glow Gamma", 1.0, 0.4, 2.0, 0.01),
        "edge_threshold": NumericParam("Edge Threshold", 0.4, 0.0, 1.0, 0.01),

        "use_color": ChoiceParam("Color Glow", "off", ["off", "on"]),
        "color": NumericParam("Color (Hue)", 0.0, 0.0, 1.0, 0.01),
        "color_amount": NumericParam("Color Amount", 0.0, -1.0, 1.0, 0.01),
    }

    def apply(self, image, params):
        mode = params["mode"]

        threshold = params["threshold"]
        softness = params["softness"]
        radius = int(params["radius"])
        glow_gamma = params["glow_gamma"]
        edge_th = params["edge_threshold"]

        # CONTROLLED POWER
        intensity = params["intensity"] ** 1.6

        use_color = params["use_color"] == "on"
        hue = params["color"]
        color_amt = params["color_amount"]

        if radius % 2 == 0:
            radius += 1

        img = normalize(image)
        gray = normalize(to_gray(image))

        # ─────────────────────────────
        # LIGHT SOURCE
        # ─────────────────────────────

        if mode == "edge_aware":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edges = np.sqrt(gx * gx + gy * gy)
            edges /= edges.max() + 1e-6

            edges = np.clip(
                (edges - edge_th) / max(1e-6, (1.0 - edge_th)),
                0.0,
                1.0,
            )

            light = np.dstack([edges] * 3)

        else:
            # SOFT THRESHOLD (fixed)
            soft = softness * 0.5 + 0.02
            mask = np.clip((gray - threshold) / soft, 0.0, 1.0)
            light = img * np.dstack([mask] * 3)

        # ─────────────────────────────
        # BLOOM
        # ─────────────────────────────

        glow = np.zeros_like(light)

        if mode == "basic":
            scales = [(radius, 1.0)]
        else:
            scales = [
                (radius // 4, 1.4),
                (radius, 1.0),
                (radius * 2, 0.8),
                (radius * 4, 0.4),
            ]

        for r, w in scales:
            r = max(3, int(r) | 1)
            blur = cv2.GaussianBlur(
                light,
                (r, r),
                sigmaX=r * 0.5,
                sigmaY=r * 0.5,
            )
            glow += blur * w

        glow = np.clip(glow, 0.0, 1.0)

        # ─────────────────────────────
        # GAMMA
        # ─────────────────────────────

        if mode == "gamma":
            glow = np.power(glow, glow_gamma)

        # ─────────────────────────────
        # SOFTNESS
        # ─────────────────────────────

        if softness > 0:
            glow = cv2.GaussianBlur(
                glow,
                (0, 0),
                sigmaX=softness * radius,
                sigmaY=softness * radius,
            )

        # ─────────────────────────────
        # COLOR (FIXED, ALL MODES)
        # ─────────────────────────────

        if use_color:
            hsv = np.array(
                [[[np.clip(hue, 0, 1) * 180, 255, 255]]],
                dtype=np.uint8,
            )
            base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0] / 255.0

            if color_amt < 0:
                tint = base * (1 + color_amt)
            else:
                tint = base + (1 - base) * color_amt

            glow *= tint.astype(np.float32)

        out = img + glow * intensity
        out = np.clip(out, 0.0, 1.0)

        return denormalize(out)
