import cv2
import numpy as np
import random

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize


class NoiseEffect(Effect):
    name = "Noise+"

    params = {
        "mode": ChoiceParam(
            "Mode",
            "film",
            ["film", "gaussian", "salt", "pepper", "color"],
        ),
        "pattern": ChoiceParam(
            "Pattern",
            "grain",
            ["grain", "lines"],
        ),
        "strength": NumericParam("Strength", 0.5, 0.0, 2.0, 0.01),
        "size": NumericParam("Size", 1.0, 0.5, 5.0, 0.01),
        "randomness": NumericParam("Randomness", 0.5, 0.0, 1.0, 0.01),
        "density": NumericParam("Density", 0.1, 0.0, 1.0, 0.01),
        "line_length": NumericParam("Line Length", 10, 3, 50, 1),
        "line_angle": NumericParam("Line Angle", 0, 0, 90, 1),
    }

    def apply(self, image, params):
        mode = params["mode"]
        pattern = params["pattern"]
        strength = params["strength"]
        size = int(round(params["size"]))
        randomness = params["randomness"]
        density = params["density"]
        max_len = int(params["line_length"])
        angle_deg = params["line_angle"]

        max_len =int(np.clip(max_len,1,100))
        density = np.clip(density, 0.0, 10.0)
        if pattern == "grain":
            density = min(density, 1.5)

        img = normalize(image)
        h, w, _ = img.shape

        eff = strength ** 1.3

        # ───────────────────────────────
        # GRAIN PATH (NO LINE PARAMS USED)
        # ───────────────────────────────
        if pattern == "grain":

            # Salt & pepper are PURE impulse noise
            if mode in ("salt", "pepper"):
                out = img.copy()
                p = density * 0.15

                mask = np.random.rand(h, w) < p
                if mode == "salt":
                    out[mask] = 1.0
                else:
                    out[mask] = 0.0

                return denormalize(out)

            # Film / Gaussian / Color grain
            if mode == "color":
                noise = np.random.randn(h, w, 3).astype(np.float32)
            else:
                n = np.random.randn(h, w).astype(np.float32)
                noise = np.dstack([n, n, n])

            if size > 1:
                k = int(size * 3) | 1
                noise = cv2.GaussianBlur(noise, (k, k), size)

            if randomness > 0:
                k = int(1 + randomness * 3) | 1
                noise = cv2.GaussianBlur(noise, (k, k), randomness * 0.6)

            if mode == "film":
                scale = 0.10
                noise = np.tanh(noise * 0.8)
            else:  # gaussian / color
                scale = 0.22

            out = img + noise * eff * scale * density
            out = np.clip(out, 0.0, 1.0)
            return denormalize(out)

        # ───────────────────────────────
        # LINE PATH (USES LINE PARAMS)
        # ───────────────────────────────
        out = img.copy()
        canvas = np.zeros((h, w, 3), dtype=np.float32)

        count = int(density * h * w * 0.002)

        theta = np.deg2rad(angle_deg)
        dx = np.cos(theta)
        dy = np.sin(theta)

        for _ in range(count):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)

            length = random.randint(1, max_len)
            x2 = int(np.clip(x1 + dx * length, 0, w - 1))
            y2 = int(np.clip(y1 + dy * length, 0, h - 1))

            val = random.uniform(-1.0, 1.0)

            cv2.line(
                canvas,
                (x1, y1),
                (x2, y2),
                (val, val, val),
                thickness=size,
                lineType=cv2.LINE_AA,
            )

        if randomness > 0:
            k = int(1 + randomness * 3) | 1
            canvas = cv2.GaussianBlur(canvas, (k, k), randomness * 0.5)

        if mode == "color":
            canvas *= np.random.randn(h, w, 3).astype(np.float32)
            scale = 0.25
        elif mode == "film":
            canvas = np.tanh(canvas * 0.7)
            scale = 0.12
        elif mode == "gaussian":
            scale = 0.25
        elif mode == "salt":
            canvas = (canvas > 0).astype(np.float32)
            scale = 1.0
        else:  # pepper
            canvas = -(canvas < 0).astype(np.float32)
            scale = 1.0

        out = out + canvas * eff * scale
        out = np.clip(out, 0.0, 1.0)
        return denormalize(out)
