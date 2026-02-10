import cv2
import numpy as np
from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import to_gray, normalize, soft_threshold

class ASCIIEffect(Effect):
    name = "ASCII+"

    params = {
        "cell_size": NumericParam("Cell Size", 8, 4, 32, 1),

        "glyph_set": ChoiceParam(
            "Glyph Set",
            "standard",
            ["standard", "dense", "light"]
        ),

        "edge_weight": NumericParam(
            "Edge Weight",
            0.0, 0.0, 1.0, 0.05
        ),

        "cutoff": NumericParam(
            "Cutoff",
            0.2, 0.0, 1.0, 0.01
        ),

        "softness": NumericParam(
            "Softness",
            0.0, 0.0, 1.0, 0.01
        ),
    }

    GLYPHS = {
        "dense": "@%#*+=-:. ",
        "standard": "#*+=-:. ",
        "light": "+-:. ",
    }

    def apply(self, image, params):
        cell = int(params["cell_size"])
        glyph_set = params["glyph_set"]
        edge_weight = params["edge_weight"]
        cutoff = params["cutoff"]
        softness = params["softness"]

        gray = to_gray(image)
        gray_n = normalize(gray)

        h, w = gray.shape
        out = np.zeros_like(image)

        chars = self.GLYPHS[glyph_set]
        n_chars = len(chars)

        # Optional edge guidance
        if edge_weight > 0:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            edges = normalize(cv2.magnitude(gx, gy))
        else:
            edges = 0

        for y in range(0, h, cell):
            for x in range(0, w, cell):
                block = gray_n[y:y+cell, x:x+cell]
                if block.size == 0:
                    continue

                avg = np.mean(block)

                if edge_weight > 0:
                    e = np.mean(edges[y:y+cell, x:x+cell])
                    avg = (1 - edge_weight) * avg + edge_weight * e

                density = soft_threshold(avg, cutoff, softness)

                idx = int(density * (n_chars - 1))
                val = int((idx / (n_chars - 1)) * 255)

                out[y:y+cell, x:x+cell] = val

        return out
