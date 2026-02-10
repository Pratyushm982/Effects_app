import cv2
import numpy as np
from app.effect import Effect
from app.params import NumericParam, ChoiceParam
from app.math_utils import to_gray, normalize, denormalize, soft_threshold

class EdgeEffect(Effect):
    name = "Edge+"

    params = {
    "mode": ChoiceParam("Mode", "sobel", ["sobel", "scharr", "sketch"]),
    "sensitivity": NumericParam("Sensitivity", 1.5, 0.5, 5.0, 0.1),
    "cutoff": NumericParam("Cutoff", 0.25, 0.0, 1.0, 0.01),
    "softness": NumericParam("Softness", 0.0, 0.0, 1.0, 0.01),
    "invert": NumericParam("Invert", 0, 0, 1, 1),
    }

    def apply(self, image, params):
        mode = params["mode"]
        sensitivity = params["sensitivity"]
        cutoff = params["cutoff"]
        softness = params["softness"]
        invert = params["invert"]

        gray = to_gray(image)

        # --- gradient computation ---
        if mode == "sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        elif mode == "scharr":
            gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)

        elif mode == "sketch":
            gx = cv2.Laplacian(gray, cv2.CV_32F)
            gy = np.zeros_like(gx)

        mag = cv2.magnitude(gx, gy) * sensitivity
        mag = normalize(mag)

        # --- soft thresholding ---
        edges = soft_threshold(mag, cutoff, softness)

        if invert:
            edges = 1.0 - edges

        # convert to image
        edges = denormalize(edges)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        return edges
