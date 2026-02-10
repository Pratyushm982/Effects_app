import cv2
import numpy as np

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize, to_gray


class BevelEffect(Effect):
    name = "Bevel+"

    params = {
        "type": ChoiceParam(
            "Type",
            "bevel",
            ["bevel", "emboss"],
        ),
        "mode": ChoiceParam(
            "Mode",
            "simple",
            ["simple", "soft", "hard"],
        ),
        "depth": NumericParam("Depth", 1.0, 0.1, 5.0, 0.01),
        "light_angle": NumericParam("Light Angle", 45, 0, 360, 1),
        "strength": NumericParam("Strength", 1.0, 0.0, 2.0, 0.01),
    }

    def apply(self, image, params):
        mode = params["mode"]
        kind = params["type"]
        depth = params["depth"]
        angle = np.deg2rad(params["light_angle"])
        strength = params["strength"]

        img = normalize(image)
        gray = to_gray(image) / 255.0

        if mode == "soft":
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        nx = -gx * depth
        ny = -gy * depth
        nz = np.ones_like(nx)

        norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx /= norm
        ny /= norm
        nz /= norm

        lx = np.cos(angle)
        ly = np.sin(angle)
        lz = 1.0
        ln = np.sqrt(lx * lx + ly * ly + lz * lz)
        lx /= ln
        ly /= ln
        lz /= ln

        light = nx * lx + ny * ly + nz * lz

        if mode == "hard":
            light = np.sign(light) * (np.abs(light) ** 0.6)

        # Normalize light to [-1, 1]
        light = np.clip(light, -1.0, 1.0)

        if kind == "emboss":
            base = np.full_like(light, 0.5)
            v = base + light * strength * 0.5
            v = np.clip(v, 0.0, 1.0)
            out = np.dstack([v, v, v])

        else:
            ycrcb = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YCrCb)
            Y = ycrcb[..., 0]

            highlight = np.clip(light, 0.0, 1.0)
            shadow = np.clip(-light, 0.0, 1.0)

            Y = Y + highlight * strength * (1.0 - Y)
            Y = Y * (1.0 - shadow * strength)

            ycrcb[..., 0] = np.clip(Y, 0.0, 1.0)
            out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return denormalize(out)
