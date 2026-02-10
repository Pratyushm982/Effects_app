# effects/dither_registry.py

DITHER_STYLES = {
    # --- Error Diffusion ---
    "floyd": {
        "label": "Floyd–Steinberg",
        "engine": "error_diffusion",
        "config": {"kernel": "floyd"},
    },
    "atkinson": {
        "label": "Atkinson",
        "engine": "error_diffusion",
        "config": {"kernel": "atkinson"},
    },
    "stucki": {
        "label": "Stucki",
        "engine": "error_diffusion",
        "config": {"kernel": "stucki"},
    },
    #"jarvis": {
    #    "label": "Jarvis–Judice–Ninke",
    #    "engine": "error_diffusion",
    #    "config": {"kernel": "jarvis"},
    #},
    #"burkes": {
    #    "label": "Burkes",
    #    "engine": "error_diffusion",
    #    "config": {"kernel": "burkes"},
    #},
    #"sierra_3": {
    #    "label": "Sierra-3",
    #    "engine": "error_diffusion",
    #    "config": {"kernel": "sierra3"},
    #},
    #"sierra_lite": {
    #    "label": "Sierra Lite",
    #    "engine": "error_diffusion",
    #    "config": {"kernel": "sierra_lite"},
    #},
    "two_row_sierra": {
        "label": "Two-Row Sierra",
        "engine": "error_diffusion",
        "config": {"kernel": "two_row_sierra"},
    },
    "stevenson_arce": {
        "label": "Stevenson–Arce",
        "engine": "error_diffusion",
        "config": {"kernel": "stevenson"},
    },
    "ostromoukhov": {
        "label": "Ostromoukhov (Adaptive)",
        "engine": "error_diffusion",
        "config": {"kernel": "ostromoukhov"},
    },
    # --- Ordered ---
    "bayer_2": {
        "label": "Bayer 2×2",
        "engine": "ordered",
        "config": {"matrix": "bayer2"},
    },
    "bayer_4": {
        "label": "Bayer 4×4",
        "engine": "ordered",
        "config": {"matrix": "bayer4"},
    },
    "bayer_8": {
        "label": "Bayer 8×8",
        "engine": "ordered",
        "config": {"matrix": "bayer8"},
    },
    #"bayer_16": {
    #    "label": "Bayer 16×16",
    #    "engine": "ordered",
    #    "config": {"matrix": "bayer16"},
    #},
    "random_ordered": {
        "label": "Random Ordered",
        "engine": "ordered",
        "config": {"matrix": "random"},
    },
    "bayer_void": {
        "label": "Bayer Void",
        "engine": "ordered",
        "config": {"matrix": "bayer_void"},
    },
    "bit_tone": {
        "label": "Bit Tone",
        "engine": "ordered",
        "config": {"matrix": "bitone"},
    },
    #"mosaic": {
    #    "label": "Mosaic",
    #    "engine": "ordered",
    #    "config": {"matrix": "mosaic"},
    #},

    # --- Pattern ---
    "checkers_s": {
        "label": "Checkers – Small",
        "engine": "pattern",
        "config": {"pattern": "checkers", "scale": 4},
    },
    "checkers_m": {
        "label": "Checkers – Medium",
        "engine": "pattern",
        "config": {"pattern": "checkers", "scale": 8},
    },
    "checkers_l": {
        "label": "Checkers – Large",
        "engine": "pattern",
        "config": {"pattern": "checkers", "scale": 16},
    },
    "gridlock": {
        "label": "Gridlock / Traffic",
        "engine": "pattern",
        "config": {"pattern": "grid", "scale": 8},
    },
    "diamonds": {
        "label": "Diamonds",
        "engine": "pattern",
        "config": {"pattern": "diamonds", "scale": 8},
    },
    "stippling": {
        "label": "Stippling",
        "engine": "pattern",
        "config": {"pattern": "stipple", "density": 1.0},
    },
    "crosshatch": {
        "label": "Crosshatch",
        "engine": "pattern",
        "config": {"pattern": "crosshatch", "scale": 8},
    },
    "diagonal": {
        "label": "Diagonal",
        "engine": "pattern",
        "config": {"pattern": "crosshatch", "scale": 12},
    },
    # --- Glitch / Modulation ---
    "waveform": {
        "label": "Waveform",
        "engine": "glitch",
        "config": {"type": "waveform"},
    },

    "waveform_x": {
        "label": "Waveform X",
        "engine": "glitch",
        "config": {"type": "dither_wave_x", "frequency": 1.0},
    },
    "waveform_y": {
        "label": "Waveform Y",
        "engine": "glitch",
        "config": {"type": "dither_wave_y", "frequency": 1.0},
    },
    "glitch_block": {
        "label": "Block Glitch",
        "engine": "glitch",
        "config": {"type": "block", "block_size": 24, "intensity": 0.4},
    },
    "threshold_noise": {
        "label": "Threshold Noise",
        "engine": "glitch",
        "config": {"type": "threshold", "strength": 0.15},
    },
    "artifact_modulation": {
        "label": "Artifact Modulation",
        "engine": "glitch",
        "config": {"type": "artifact", "block_size": 16, "intensity": 0.6},
    },
    "atkinson_vhs": {
        "label": "Atkinson VHS",
        "engine": "glitch",
        "config": {"type": "vhs", "block_size": 32, "intensity": 0.5},
    },
    "glitch": {
        "label": "Glitch",
        "engine": "glitch",
        "config": {"type": "threshold", "strength": 0.25},
    },
    #"uniform_mod_x": {
    #    "label": "Uniform Modulation X",
    #    "engine": "glitch",
    #    "config": {"type": "wave_x", "amplitude": 10, "frequency": 0.03},
    #},
    #"uniform_mod_y": {
    #    "label": "Uniform Modulation Y",
    #    "engine": "glitch",
    #    "config": {"type": "wave_y", "amplitude": 10, "frequency": 0.03},
    #},
    "waveform_alt": {
        "label": "Waveform Alt",
        "engine": "glitch",
        "config": {"type": "dither_wave_alt", "frequency": 1.0},
    },


    # --- Luminance / Directional ---
    "smooth_diffuse": {
        "label": "Smooth Diffuse",
        "engine": "error_diffusion",
        "config": {"kernel": "atkinson"},
    },
    "stucki_lines": {
        "label": "Stucki Diffusion Lines",
        "engine": "error_diffusion",
        "config": {"kernel": "stucki"},
    },
    "contrast_aware_x": {
        "label": "Contrast Aware X",
        "engine": "glitch",
        "config": {"type": "wave_x", "amplitude": 14, "frequency": 0.04},
    },
    "contrast_aware_y": {
        "label": "Contrast Aware Y",
        "engine": "glitch",
        "config": {"type": "wave_y", "amplitude": 14, "frequency": 0.04},
    },


    # --- Special ---
    "radial_burst": {
        "label": "Radial Burst",
        "engine": "glitch",
        "config": {"type": "radial_threshold", "strength": 0.3},
    },
    "noise": {
        "label": "Noise",
        "engine": "glitch",
        "config": {"type": "threshold", "strength": 0.1},
    },
    "threshold": {
        "label": "Threshold",
        "engine": "glitch",
        "config": {"type": "threshold", "strength": 0.5},
    },
    
    "sine_wave_mod": {
        "label": "Sine Wave Modulation",
        "engine": "glitch",
        "config": {"type": "wave_x", "amplitude": 20, "frequency": 0.12},
    },
    "displace_contour": {
        "label": "Displace Contour",
        "engine": "glitch",
        "config": {
            "type": "displace_contour",
            "strength": 1.0,},
    },
    "topography": {
        "label": "Topography",
        "engine": "glitch",
        "config": {
            "type": "topography",
            "levels": 12,},
    },
    "ordered_modulation": {
        "label": "Ordered Modulation",
        "engine": "ordered",
        "config": {"matrix": "bayer8"},
    },
    "atkinson_line_modulation": {
        "label": "Atkinson Line Modulation",
        "engine": "error_diffusion",
        "config": {"kernel": "atkinson"},
    },



}


def get_style_keys():
    return list(DITHER_STYLES.keys())


def get_style_labels():
    return [v["label"] for v in DITHER_STYLES.values()]


def get_style(style_key):
    if style_key not in DITHER_STYLES:
        raise ValueError(f"Unknown dither style: {style_key}")
    return DITHER_STYLES[style_key]
