import cv2
import numpy as np

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize, to_gray


class PixelateEffect(Effect):
    name = "Pixelate+"

    params = {
        "mode": ChoiceParam(
            "Mode",
            "full",
            [
                "basic",
                "perceptual",
                "contrast",
                "soft",
                "luma",
                "poster",
                "grid",
                "full",
            ],
        ),
        "size": NumericParam("Pixel Size", 8, 2, 64, 1),
        "strength": NumericParam("Strength", 1.0, 0.0, 1.0, 0.01),
        "contrast": NumericParam("Contrast Sensitivity", 0.6, 0.0, 1.0, 0.01),
    }

    def apply(self, image, params):
        mode = params["mode"]
        base_size = int(params["size"])
        strength = params["strength"]
        contrast_amt = params["contrast"]

        img = normalize(image)
        h, w, _ = img.shape

        size = base_size
        if mode in ("perceptual", "full"):
            size = int(base_size ** 1.15)
        size = max(2, size)

        gray = to_gray(image) / 255.0

        luma = (
            0.2126 * gray
            + 0.7152 * gray
            + 0.0722 * gray
        )

        contrast_map = None
        if mode in ("contrast", "full", "crystallize"):
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx * gx + gy * gy)
            grad /= grad.max() + 1e-6
            contrast_map = 1.0 - contrast_amt * grad

        pixelated = img.copy()

        y = 0
        while y < h:
            x = 0
            while x < w:
                block_size = size

                if mode == "luma":
                    block_size = int(size * (1.2 - luma[y, x]))
                    block_size = max(2, block_size)

                if contrast_map is not None:
                    block_size = int(block_size * contrast_map[y, x])
                    block_size = max(2, block_size)

                y2 = min(y + block_size, h)
                x2 = min(x + block_size, w)

                block = img[y:y2, x:x2]

                if mode == "poster":
                    color = np.median(block.reshape(-1, 3), axis=0)

                elif mode == "grid":
                    cy = min(y + block_size // 2, h - 1)
                    cx = min(x + block_size // 2, w - 1)
                    color = img[cy, cx]

                else:
                    color = block.mean(axis=(0, 1))

                pixelated[y:y2, x:x2] = color
                x += block_size
            y += block_size

        if mode in ("soft", "full"):
            pixelated = img * (1.0 - strength) + pixelated * strength

        pixelated = np.clip(pixelated, 0.0, 1.0)
        return denormalize(pixelated)
