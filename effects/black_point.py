import numpy as np

from app.effect import Effect
from app.params import NumericParam
from app.math_utils import normalize, denormalize, to_gray


class BlackPointEffect(Effect):
    name = "Black Point+"

    params = {
        "black_point": NumericParam("Black Point", 0.0, 0.0, 1.0, 0.01),
        "softness": NumericParam("Softness", 0.1, 0.0, 1.0, 0.01),
    }

    def apply(self, image, params):
        bp = params["black_point"]
        softness = params["softness"]

        img = normalize(image)
        gray = normalize(to_gray(image))

        width = softness * 0.5 + 1e-6
        mask = np.clip((gray - bp) / width, 0.0, 1.0)

        out = img * mask[..., None]
        out = np.clip(out, 0.0, 1.0)

        return denormalize(out)
