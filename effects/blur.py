import cv2
import numpy as np

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize, to_gray


class BlurEffect(Effect):
    name = "Blur+"

    params = {
        "mode": ChoiceParam(
            "Mode",
            "gaussian",
            [
                "box",
                "gaussian",
                "blur_more",
                "surface",
                "smart",
            ],
        ),
        "radius": NumericParam("Radius", 9, 1, 51, 2),
        "strength": NumericParam("Strength", 1.0, 0.5, 3.0, 0.01),
        "edge_preserve": NumericParam("Edge Preserve", 0.7, 0.0, 1.0, 0.01),
        "threshold": NumericParam("Threshold", 0.15, 0.0, 1.0, 0.01),
    }

    def apply(self, image, params):
        mode = params["mode"]
        radius = int(params["radius"])
        strength = params["strength"]
        edge_amt = params["edge_preserve"]
        thresh = params["threshold"]

        if radius % 2 == 0:
            radius += 1

        img = normalize(image)
        eff_radius = max(3, int(radius * strength) | 1)

        # ── BOX BLUR ───────────────────────
        if mode == "box":
            blurred = cv2.blur(img, (eff_radius, eff_radius))

        # ── GAUSSIAN ───────────────────────
        elif mode == "gaussian":
            blurred = cv2.GaussianBlur(
                img,
                (eff_radius, eff_radius),
                eff_radius * 0.45,
            )

        # ── BLUR MORE ──────────────────────
        elif mode == "blur_more":
            r = eff_radius * 2 | 1
            blurred = cv2.GaussianBlur(
                img,
                (r, r),
                r * 0.5,
            )

        # ── SURFACE BLUR ───────────────────
        elif mode == "surface":
            gray = to_gray(image) / 255.0
            base = cv2.GaussianBlur(img, (eff_radius, eff_radius), eff_radius * 0.45)

            diff = np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0))
            mask = diff < (thresh * edge_amt)

            blurred = img.copy()
            blurred[mask] = base[mask]

        # ── SMART BLUR ─────────────────────
        else:
            gray = to_gray(image) / 255.0
            base = cv2.GaussianBlur(img, (eff_radius, eff_radius), eff_radius * 0.45)

            diff = np.abs(gray - cv2.GaussianBlur(gray, (3, 3), 0))
            mask = diff < thresh

            blurred = img * (~mask[..., None]) + base * mask[..., None]

        blurred = np.clip(blurred, 0.0, 1.0)
        return denormalize(blurred)
