# taxonomy.py (excerpt)

from typing import Dict, Any, List

taxonomy: Dict[str, Any] = {
    "meta": {
        "sources": {
            "E": {"code": "E", "name": "Environment/Scene"},
            "S": {"code": "S", "name": "Optics/Sensor"},
            "R": {"code": "R", "name": "ISP/Rendering/Codec"},
            "T": {"code": "T", "name": "System/Transfer"},
        },
        "effects": {
            "N":  {"code": "N",  "name": "Noise"},
            "B":  {"code": "B",  "name": "Blur"},
            "WX": {"code": "WX", "name": "Weather/Medium"},
            "CP": {"code": "CP", "name": "Compression/Quantization"},
            "CL": {"code": "CL", "name": "Color/White balance"},
            "IL": {"code": "IL", "name": "Brightness/Exposure"},
            "GD": {"code": "GD", "name": "Geometry/Spatial"},
            "RZ": {"code": "RZ", "name": "Resolution/Sampling"},
            "OC": {"code": "OC", "name": "Occlusion/Obstruction"},
            "TX": {"code": "TX", "name": "Sharpness/Contrast/Texture"},
            "TV": {"code": "TV", "name": "Temporal/Video"},
        },
        "groups": {
            "G1":  {"id": "G1",  "name": "Noise",                        "source": "S",  "source_name": "Optics/Sensor",        "effect": "N",  "effect_name": "Noise"},
            "G2":  {"id": "G2",  "name": "Blur",                         "source": "S",  "source_name": "Optics/Sensor",        "effect": "B",  "effect_name": "Blur"},
            "G3":  {"id": "G3",  "name": "Resolution/Sampling",          "source": "R",  "source_name": "ISP/Rendering/Codec", "effect": "RZ", "effect_name": "Resolution/Sampling"},
            "G4":  {"id": "G4",  "name": "Compression/Quantization",     "source": "R",  "source_name": "ISP/Rendering/Codec", "effect": "CP", "effect_name": "Compression/Quantization"},
            "G5":  {"id": "G5",  "name": "Color/White balance",          "source": "R",  "source_name": "ISP/Rendering/Codec", "effect": "CL", "effect_name": "Color/White balance"},
            "G6":  {"id": "G6",  "name": "Brightness/Exposure",          "source": "R",  "source_name": "ISP/Rendering/Codec", "effect": "IL", "effect_name": "Brightness/Exposure"},
            "G7":  {"id": "G7",  "name": "Geometry/Spatial",             "source": "R",  "source_name": "ISP/Rendering/Codec", "effect": "GD", "effect_name": "Geometry/Spatial"},
            "G8":  {"id": "G8",  "name": "Weather/Medium",               "source": "E",  "source_name": "Environment/Scene",   "effect": "WX", "effect_name": "Weather/Medium"},
            "G9":  {"id": "G9",  "name": "Occlusion/Obstruction",        "source": "E",  "source_name": "Environment/Scene",   "effect": "OC", "effect_name": "Occlusion/Obstruction"},
            "G10": {"id": "G10", "name": "Sharpness/Contrast/Texture",   "source": "R",  "source_name": "ISP/Rendering/Codec", "effect": "TX", "effect_name": "Sharpness/Contrast/Texture"},
            "G11": {"id": "G11", "name": "System/Transfer/Board",        "source": "T",  "source_name": "System/Transfer",     "effect": None, "effect_name": None},
            "G12": {"id": "G12", "name": "Temporal/Video",               "source": None, "source_name": "Variable (E/S/R/T)",  "effect": "TV", "effect_name": "Temporal/Video"},
        },
    },
    # ======================================================================
    # Add temporal/video (G12) subtypes into the taxonomy meta section
    # ======================================================================
    "subtypes": {
                "tv": {
                    "tv_flicker": {
                        "name": "Temporal flicker",
                        "default_source": "E",
                        "effect": "TV",
                        "description": "Illumination flicker, exposure/gamma pumping."
                    },
                    "tv_awb_osc": {
                        "name": "AWB oscillation",
                        "default_source": "R",
                        "effect": "TV",
                        "description": "Auto–white–balance oscillation over time."
                    },
                    "tv_rs_wobble": {
                        "name": "Rolling shutter wobble",
                        "default_source": "S",
                        "effect": "TV",
                        "description": "RS wobble/skew under motion."
                    },
                    "tv_af_hunting": {
                        "name": "Autofocus hunting",
                        "default_source": "S",
                        "effect": "TV",
                        "description": "AF hunting or focus breathing."
                    },
                    "tv_ois_jitter": {
                        "name": "OIS jitter",
                        "default_source": "S",
                        "effect": "TV",
                        "description": "Optical-image-stabilizer jitter/instability."
                    },
                    "tv_ghosting": {
                        "name": "Temporal ghosting",
                        "default_source": "R",
                        "effect": "TV",
                        "description": "Temporal denoise/deinterlace ghosting."
                    },
                    "tv_stab_jitter": {
                        "name": "Stabilization jitter",
                        "default_source": "R",
                        "effect": "TV",
                        "description": "Temporal jitter caused by digital stabilization."
                    },
                    "tv_vfr": {
                        "name": "Variable frame-rate",
                        "default_source": "R",  # or T
                        "effect": "TV",
                        "description": "Variable frame-rate or resampling artefacts."
                    },
                    "tv_drop_repeat": {
                        "name": "Frame drop/repeat",
                        "default_source": "T",
                        "effect": "TV",
                        "description": "Frame missing or repeated due to SE/ME/THE."
                    },
                    "tv_desync": {
                        "name": "Temporal desynchronization",
                        "default_source": "T",
                        "effect": "TV",
                        "description": "AV or PTS desync / timing jitter."
                    },
                    "tv_gop_loss": {
                        "name": "GOP/slice loss",
                        "default_source": "T",  # sometimes codec (R)
                        "effect": "TV",
                        "description": "Loss of GOP/slices from transfer or codec."
                    },
                    "tv_seq_transforms": {
                        "name": "Sequential transforms",
                        "default_source": "R",
                        "effect": "TV",
                        "description": "Small sequential transforms (translate/rotate/scale/tilt), e.g. ImageNet-P."
                    },
                }
            },
    "papers": {

        # ------------- ARNIQA / Agnolucci WACV 2024 -------------------
        "Agnolucci_WACV_2024": {
            "meta": {
                "title": "ARNIQA: Learning Distortion Manifold for Image Quality Assessment",
                "venue": "WACV 2024",
                "url": "https://github.com/miccunifi/ARNIQA",
                "based": ""
            },
            # Group-level mapping (as in the paper)
            "groups": {
                "Brightness change": {
                    "group_id": "G6",
                    "group_name": "Brightness/Exposure",
                    "source": "R",
                    "effect": "IL",
                },
                "Blur": {
                    "group_id": "G2",
                    "group_name": "Blur",
                    "source": "S",
                    "effect": "B",
                },
                "Spatial distortions": {
                    "group_id": "G7",
                    "group_name": "Geometry/Spatial",
                    "source": "R",
                    "effect": "GD",
                },
                "Noise": {
                    "group_id": "G1",
                    "group_name": "Noise",
                    "source": "S",
                    "effect": "N",
                },
                "Color distortions": {
                    "group_id": "G5",
                    "group_name": "Color/White balance",
                    "source": "R",
                    "effect": "CL",
                },
                "Compression": {
                    "group_id": "G4",
                    "group_name": "Compression/Quantization",
                    "source": "R",
                    "effect": "CP",
                },
                "Sharpness & contrast": {
                    "group_id": "G10",
                    "group_name": "Sharpness/Contrast/Texture",
                    "source": "R",
                    "effect": "TX",
                },
            },

            # Type-level mapping: *links backend keys / names to groups*
            "types": {
                # Blur (G2)
                "gaublur":   {"group_id": "G2", "source": "S", "effect": "B"},
                "lensblur":  {"group_id": "G2", "source": "S", "effect": "B"},
                "motionblur":{"group_id": "G2", "source": "S", "effect": "B"},

                # Noise (G1)
                "whitenoise":   {"group_id": "G1", "source": "S", "effect": "N"},
                "whitenoiseCC": {"group_id": "G1", "source": "S", "effect": "N"},
                "impulsenoise": {"group_id": "G1", "source": "S", "effect": "N"},
                "multnoise":    {"group_id": "G1", "source": "S", "effect": "N"},

                # Brightness/Exposure (G6)
                "brighten":  {"group_id": "G6", "source": "R", "effect": "IL"},
                "darken":    {"group_id": "G6", "source": "R", "effect": "IL"},
                "meanshift": {"group_id": "G6", "source": "R", "effect": "IL"},

                # Color (G5)
                "colordiff": {"group_id": "G5", "source": "R", "effect": "CL"},
                "colorshift":{"group_id": "G5", "source": "R", "effect": "CL"},
                "colorsat1": {"group_id": "G5", "source": "R", "effect": "CL"},
                "colorsat2": {"group_id": "G5", "source": "R", "effect": "CL"},

                # Compression (G4)
                "jpeg2000": {"group_id": "G4", "source": "R", "effect": "CP"},
                "jpeg":     {"group_id": "G4", "source": "R", "effect": "CP"},

                # Spatial (G7)
                "jitter":      {"group_id": "G7", "source": "R", "effect": "GD"},
                "noneccpatch": {"group_id": "G7", "source": "R", "effect": "GD"},
                "pixelate":    {"group_id": "G7", "source": "R", "effect": "RZ"},
                "quantization": {"group_id": "G4", "source": "R", "effect": "CP"},
                "colorblock":  {"group_id": "G9", "source": "R", "effect": "OC"},

                # Sharpness & contrast (G10)
                "highsharpen":     {"group_id": "G10", "source": "R", "effect": "TX"},
                "lincontrchange":  {"group_id": "G10", "source": "R", "effect": "TX"},
                "nonlincontrchange":{"group_id": "G10", "source": "R", "effect": "TX"},
            },
        },
        "Hendrycks_ICLR_2019": {
            "meta": {
                "title": "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations",
                "venue": "ICLR 2019",
                "url": "https://github.com/hendrycks/robustness",
            },

            "ImageNet-C": {
                # Noise (G1)
                "gaussian_noise":   {"group_id": "G1", "source": "S", "effect": "N"},
                "shot_noise":       {"group_id": "G1", "source": "S", "effect": "N"},
                "impulse_noise":    {"group_id": "G1", "source": "S", "effect": "N"},
                "speckle_noise":    {"group_id": "G1", "source": "S", "effect": "N"},

                # Blur (G2)
                "defocus_blur":     {"group_id": "G2", "source": "S", "effect": "B"},
                "gaussian_blur":    {"group_id": "G2", "source": "S", "effect": "B"},
                "glass_blur":       {"group_id": "G2", "source": "S", "effect": "B"},
                "motion_blur":      {"group_id": "G2", "source": "S", "effect": "B"},
                "zoom_blur":        {"group_id": "G2", "source": "S", "effect": "B"},

                # Weather / medium (G8)
                "fog":              {"group_id": "G8", "source": "E", "effect": "WX"},
                "frost":            {"group_id": "G8", "source": "E", "effect": "WX"},
                "snow":             {"group_id": "G8", "source": "E", "effect": "WX"},
                "spatter":          {"group_id": "G8", "source": "E", "effect": "WX"},

                # Brightness/Exposure (G6)
                "brightness":       {"group_id": "G6", "source": "R", "effect": "IL"},

                # Sharpness/Contrast/Texture (G10)
                "contrast":         {"group_id": "G10", "source": "R", "effect": "TX"},

                # Color (G5)
                "saturate":         {"group_id": "G5", "source": "R", "effect": "CL"},

                # Compression (G4)
                "jpeg_compression": {"group_id": "G4", "source": "R", "effect": "CP"},

                # Resolution/Sampling (G3)
                "pixelate":         {"group_id": "G3", "source": "R", "effect": "RZ"},#

                # Geometry/Spatial (G7)
                "elastic_transform":{"group_id": "G7", "source": "R", "effect": "GD"},
            },

            # Optional: if you later add ImageNet-P, you can link to TV subtypes:
            # "ImageNet-P": {
            #     "translate": {"group_id": "G12", "source": "R", "effect": "TV", "subtype": "tv_seq_transforms"},
            #     ...
            # },
        },
        "Liu_IJCV_2024": {
            "meta": {
                "title": "Benchmarking Object Detection Robustness against Real-World Corruptions",
                "venue": "IJCV 2024",
                "url": "https://sites.google.com/view/real-worldbenchmark/real-world-benchmarking",
                "based": "",
            },

            # High-level grouping as described in Liu et al.
            "groups": {
                "Camera damage": {
                    "group_id": None,
                    "description": "Physical damage or environmental interference affecting optics/sensor.",
                },
                "ISP damage": {
                    "group_id": None,
                    "description": "Faults in the ISP pipeline (BLC, shading, AWB, gamma, CFA, color space).",
                },
                "Board-level damage": {
                    "group_id": None,
                    "description": "Corruptions due to memory, transfer, or board-level faults.",
                },
            },

            # Type-level mapping: backend keys -> canonical taxonomy
            "types": {

                # ---------------------------------------------------
                # Camera damage
                # ---------------------------------------------------
                "fog": {
                    "group_id": "G8",
                    "source": "E",
                    "effect": "WX",
                    "original_name": "Fog (F)",
                },
                "lens_obstruction": {
                    "group_id": "G9",
                    "source": "S",
                    "effect": "OC",
                    "original_name": "Lens Obstruction (LO)",
                },
                "focus_motor_damage": {
                    "group_id": "G2",
                    "source": "S",
                    "effect": "B",
                    "original_name": "Focus Motor Damage (FM-D)",
                },
                "ccd_sensor_damage": {
                    "group_id": "G1",
                    "source": "S",
                    "effect": "N",
                    "original_name": "CCD Sensor Damage (CCD-D)",
                },
                "cmos_sensor_damage": {
                    "group_id": "G1",
                    "source": "S",
                    "effect": "N",
                    "original_name": "CMOS Sensor Damage (CMOS-D)",
                },

                # ---------------------------------------------------
                # ISP damage
                # ---------------------------------------------------
                "insufficient_blc": {
                    "group_id": "G6",
                    "source": "R",
                    "effect": "IL",
                    "original_name": "Insufficient Black Level Correction (I-BLC)",
                },
                "excessive_blc": {
                    "group_id": "G6",
                    "source": "R",
                    "effect": "IL",
                    "original_name": "Excessive Black Level Correction (E-BLC)",
                },
                "lens_shading_damage": {
                    "group_id": "G6",
                    "source": "R",
                    "effect": "IL",
                    "original_name": "Lens Shading Correction Damage (LSC-D)",
                },
                "awb_damage": {
                    "group_id": "G5",
                    "source": "R",
                    "effect": "CL",
                    "original_name": "Automated White Balance Damage (AWB-D)",
                },
                "bad_pixel_correction": {
                    "group_id": "G1",
                    "source": "R",
                    "effect": "N",
                    "original_name": "Bad Pixel Correction Damage (BPC-D)",
                },
                "cfa_interpolation_damage": {
                    "group_id": "G10",
                    "source": "R",
                    "effect": "TX",
                    "original_name": "CFA Interpolation Damage (CFAI-D)",
                },
                "gamma_damage": {
                    "group_id": "G10",
                    "source": "R",
                    "effect": "TX",
                    "original_name": "Gamma Correction Damage (GC-D)",
                },
                "color_space_conversion_damage": {
                    "group_id": "G5",
                    "source": "R",
                    "effect": "CL",
                    "original_name": "Color Space Conversion Damage (CSC-D)",
                },

                # ---------------------------------------------------
                # Board-level damage
                # (matching your BACKEND_CAPS list)
                # ---------------------------------------------------
                "memory_exceptions": {
                    "group_id": "G11",
                    "source": "T",
                    "effect": "TV",
                    "original_name": "Memory Exceptions (ME)",
                },
                "transfer_harness_exceptions": {
                    "group_id": "G11",
                    "source": "T",
                    "effect": "TV",
                    "original_name": "Transfer Harness Exceptions (THE)",
                },
                "sensor_broken": {
                    "group_id": "G11",
                    "source": "T",
                    "effect": "TV",
                    "original_name": "Sensor Broken",
                },
            },
        },#  "XXXX et al. XXXX": {
    },#  "papers": {
}# taxonomy: Dict[str, Any] = {


# ======================================================================
# Helper functions
# ======================================================================
def get_group_full(group_id: str) -> Dict[str, Any]:
    return taxonomy["meta"]["groups"][group_id]


def map_paper_term(paper: str, term: str) -> Dict[str, Any]:
    paper_node = taxonomy["papers"][paper]
    entry = None
    for section in ("types", "ImageNet-C", "ImageNet-P", "groups"):
        node = paper_node.get(section, {})
        if term in node:
            entry = node[term]
            break
    if entry is None:
        raise KeyError(f"Term {term!r} not found for paper {paper!r}.")

    g = get_group_full(entry["group_id"])
    return {
        **entry,
        "group": g,
    }


def list_paper_types(paper_key: str) -> List[Dict[str, Any]]:
    paper = taxonomy["papers"].get(paper_key)
    if not paper:
        raise KeyError(f"Paper {paper_key!r} not found.")
    out: List[Dict[str, Any]] = []
    #for section in ("types", "ImageNet-C", "ImageNet-P", "groups"):
    for section in ("types", "ImageNet-C", "ImageNet-P"):
        node = paper.get(section, {})

        for term, info in node.items():
            # Some entries (e.g. high-level Liu groups) have no group_id.
            group_id = info.get("group_id")
            if not group_id:
                # Skip non-degradation entries
                continue
            g = taxonomy["meta"]["groups"][group_id]
            out.append({
                "term": term,
                "group_id": group_id,
                "group_name": g["name"],
                "source": info.get("source") or g["source"],
                "effect": info.get("effect") or g["effect"],
            })
    return out

def get_subtype_meta(subtype_code: str) -> Dict[str, Any]:
    """
    Look up metadata for a given subtype code (e.g. 'tv_drop_repeat').

    Searches over taxonomy['meta']['subtypes'][*].
    Raises KeyError if not found.

    usage
    meta = get_subtype_meta("tv_drop_repeat")
    print(meta["name"], meta["default_source"], meta["effect"])
    # -> "Frame drop/repeat", "T", "TV"

    """
    subtypes_root = taxonomy["meta"].get("subtypes", {})
    for family_name, family_dict in subtypes_root.items():
        if subtype_code in family_dict:
            return family_dict[subtype_code]
    raise KeyError(f"Subtype {subtype_code!r} not found in taxonomy['meta']['subtypes'].")


def list_taxonomy_papers() -> List[str]:
    """
    Return all paper keys defined under taxonomy["papers"],
    regardless of whether a backend is implemented.
    """
    return sorted(taxonomy.get("papers", {}).keys())




