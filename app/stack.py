import uuid
import hashlib
from copy import deepcopy
import json


# ─────────────────────────────
# Internal helpers
# ─────────────────────────────

def _image_signature(img):
    """
    Deterministic signature for cache invalidation.
    This is intentionally simple and correct.
    """
    h = hashlib.sha1()
    h.update(str(img.shape).encode())
    h.update(str(img.dtype).encode())
    h.update(img.tobytes())
    return h.hexdigest()


# ─────────────────────────────
# EffectNode
# ─────────────────────────────

class EffectNode:
    """
    Represents a single effect instance inside a stack.
    """

    def __init__(self, effect):
        self.id = str(uuid.uuid4())
        self.effect = effect
        self.enabled = True

        # Snapshot parameter values from effect definition
        self.params = {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in effect.params.items()
        }

        # Cache
        self.cached_output = None
        self.last_input_sig = None

    def invalidate(self):
        """
        Clear cached output.
        """
        self.cached_output = None
        self.last_input_sig = None

    def copy(self):
        """
        Deep copy (useful later for undo/redo).
        """
        new = EffectNode(self.effect)
        new.enabled = self.enabled
        new.params = deepcopy(self.params)
        return new


# ─────────────────────────────
# EffectStack
# ─────────────────────────────

class EffectStack:
    """
    Ordered collection of EffectNodes.
    """

    def __init__(self):
        self.nodes = []

    # ─────────────────────────────
    # Stack manipulation (UI-driven)
    # ─────────────────────────────

    def add_effect(self, effect):
        node = EffectNode(effect)
        self.nodes.append(node)
        return node

    def remove_node(self, node_id):
        idx = None
        for i, n in enumerate(self.nodes):
            if n.id == node_id:
                idx = i
                break

        if idx is not None:
            # Invalidate downstream
            for n in self.nodes[idx:]:
                n.invalidate()
            self.nodes.pop(idx)

    def move_node(self, from_index, to_index):
        node = self.nodes.pop(from_index)
        self.nodes.insert(to_index, node)

        # Invalidate from earliest affected index
        start = min(from_index, to_index)
        for n in self.nodes[start:]:
            n.invalidate()

    def toggle_node(self, node_id, enabled):
        for i, n in enumerate(self.nodes):
            if n.id == node_id:
                if n.enabled != enabled:
                    n.enabled = enabled
                    # Invalidate this node and downstream
                    for dn in self.nodes[i:]:
                        dn.invalidate()
                return

    def clear(self):
        self.nodes.clear()

    # ─────────────────────────────
    # Stack execution (cached)
    # ─────────────────────────────

    def apply(self, image):
        out = image
        input_sig = _image_signature(image)

        for node in self.nodes:
            if not node.enabled:
                node.invalidate()
                continue

            if node.cached_output is not None and node.last_input_sig == input_sig:
                out = node.cached_output
            else:
                out = node.effect.apply(out, node.params)
                node.cached_output = out
                node.last_input_sig = input_sig

            input_sig = _image_signature(out)

        return out
    
    def to_dict(self):
        return {
            "nodes": [
                {
                    "effect": node.effect.name,
                    "enabled": node.enabled,
                    "params": node.params,
                }
                for node in self.nodes
            ]
        }


    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


    @classmethod
    def from_dict(cls, data):
        stack = cls()

        for item in data["nodes"]:
            effect_name = item["effect"]
            effect_cls = EFFECT_REGISTRY.get(effect_name)

            if effect_cls is None:
                raise ValueError(f"Unknown effect: {effect_name}")

            node = stack.add_effect(effect_cls())
            node.enabled = item["enabled"]
            node.params.update(item["params"])

        return stack


    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)

        return cls.from_dict(data)

# ─────────────────────────────
# Effect registration
# ─────────────────────────────

EFFECT_REGISTRY = {}

from effects.threshold import ThresholdEffect
from effects.blur import BlurEffect
from effects.glow import GlowEffect
from effects.dither import DitherEffect
from effects.gamma import GammaEffect
from effects.black_point import BlackPointEffect
from effects.white_point import WhitePointEffect
from effects.gradient_map import GradientMapEffect
from effects.solid_palette_map import SolidPaletteMapEffect
from effects.noise import NoiseEffect
from effects.pixelate import PixelateEffect
from effects.bevel import BevelEffect
from effects.edge import EdgeEffect
from effects.ascii import ASCIIEffect

EFFECT_REGISTRY.update({
    ThresholdEffect.name: ThresholdEffect,
    BlurEffect.name: BlurEffect,
    GlowEffect.name: GlowEffect,
    DitherEffect.name: DitherEffect,
    GammaEffect.name: GammaEffect,
    BlackPointEffect.name: BlackPointEffect,
    WhitePointEffect.name: WhitePointEffect,
    GradientMapEffect.name: GradientMapEffect,
    SolidPaletteMapEffect.name: SolidPaletteMapEffect,
    NoiseEffect.name: NoiseEffect,
    PixelateEffect.name: PixelateEffect,
    BevelEffect.name: BevelEffect,
    EdgeEffect.name: EdgeEffect,
    ASCIIEffect.name: ASCIIEffect,
})