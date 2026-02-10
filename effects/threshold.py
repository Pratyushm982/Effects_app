import numpy as np
import cv2

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize, to_gray


class ThresholdEffect(Effect):
    name = "Threshold+"

    params = {
        "mode": ChoiceParam(
            "Mode",
            "luma",
            ["luma", "rgb"],
        ),
        "threshold": NumericParam("Threshold", 0.5, 0.0, 1.0, 0.01),
        "softness": NumericParam("Softness", 0.0, 0.0, 1.0, 0.01),
        "invert": ChoiceParam(
            "Invert",
            "off",
            ["off", "on"],
        ),
        "preserve_color": ChoiceParam(
            "Preserve Color",
            "off",
            ["off", "on"],
        ),
    }

    def apply(self, image, params):
        mode = params["mode"]
        threshold = params["threshold"]
        softness = params["softness"]
        invert = params["invert"] == "on"
        preserve_color = params["preserve_color"] == "on"

        img = normalize(image)

        # ── Source signal ───────────────────────────
        if mode == "luma":
            src = normalize(to_gray(image))
        else:
            src = img.mean(axis=2)

        # ── Soft threshold ──────────────────────────
        if softness > 0:
            width = softness * 0.5 + 1e-6
            mask = np.clip((src - threshold) / width + 0.5, 0.0, 1.0)
        else:
            mask = (src >= threshold).astype(np.float32)

        if invert:
            mask = 1.0 - mask

        # ── Output ──────────────────────────────────
        if preserve_color:
            out = img * mask[..., None]
        else:
            out = np.dstack([mask, mask, mask])

        out = np.clip(out, 0.0, 1.0)
        return denormalize(out)
