import numpy as np
import cv2

from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import normalize, denormalize, to_gray


class GammaEffect(Effect):
    name = "Gamma+"

    params = {
        "mode": ChoiceParam(
            "Mode",
            "luma",
            ["luma", "rgb"],
        ),
        "gamma": NumericParam("Gamma", 1.0, 0.2, 3.0, 0.01),
        "preserve_black": ChoiceParam(
            "Preserve Black",
            "on",
            ["off", "on"],
        ),
        "preserve_white": ChoiceParam(
            "Preserve White",
            "on",
            ["off", "on"],
        ),
    }

    def apply(self, image, params):
        mode = params["mode"]
        gamma = params["gamma"]
        keep_black = params["preserve_black"] == "on"
        keep_white = params["preserve_white"] == "on"

        img = normalize(image)

        if mode == "luma":
            gray = normalize(to_gray(image))
            gray_gamma = np.power(gray, gamma)

            if keep_black:
                gray_gamma = np.maximum(gray_gamma, gray.min())
            if keep_white:
                gray_gamma = np.minimum(gray_gamma, gray.max())

            ratio = gray_gamma / (gray + 1e-6)
            out = img * ratio[..., None]

        else:  # rgb
            out = np.power(img, gamma)

            if keep_black:
                out = np.maximum(out, img.min())
            if keep_white:
                out = np.minimum(out, img.max())

        out = np.clip(out, 0.0, 1.0)
        return denormalize(out)
