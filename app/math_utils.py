import numpy as np
import cv2

def to_gray(image):
    if len(image.shape) == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

def normalize(img):
    return np.clip(img / 255.0, 0.0, 1.0)

def denormalize(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def soft_threshold(x, cutoff, softness):
    """
    cutoff: 0–1
    softness: 0–1
    """
    if softness <= 0:
        return (x > cutoff).astype(np.float32)

    return np.clip((x - cutoff) / max(softness, 1e-5), 0, 1)
