import numpy as np

from app.effect import Effect
from app.params import NumericParam
from app.math_utils import normalize, denormalize, to_gray


class WhitePointEffect(Effect):
    name = "White Point+"

    params = {
        "white_point": NumericParam("White Point", 1.0, 0.0, 1.0, 0.01),
        "softness": NumericParam("Softness", 0.1, 0.0, 1.0, 0.01),
    }

    def apply(self, image, params):
        wp = params["white_point"]
        softness = params["softness"]

        img = normalize(image)
        gray = normalize(to_gray(image))

        width = softness * 0.5 + 1e-6
        mask = np.clip((wp - gray) / width, 0.0, 1.0)

        out = 1.0 - (1.0 - img) * mask[..., None]
        out = np.clip(out, 0.0, 1.0)

        return denormalize(out)
