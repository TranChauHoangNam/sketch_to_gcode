"""
Sketch to G-Code.

The active app accepts already-sketch/line-art images by upload or clipboard
paste, then builds line art, hatch vectors, optimized strokes, an inline QA
gate, and robot-ready G-code. The UI pipeline does not expose camera capture,
background removal, report sidecars, or legacy mode switching.
"""

import math
import os
import re
import sys
import time
import threading
import tempfile
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageGrab
try:
    from .hatching import generate_hatching
except ImportError:
    from hatching import generate_hatching

# ═══════════════════════════════════════════════════════════════════
#  CAU HINH CHUNG
# ═══════════════════════════════════════════════════════════════════

PAGE_MAX_X = 300.0      # mm
PAGE_MAX_Y = 400.0      # mm
PAGE_SAFE_MARGIN_MM = 2.0  # Le an toan de preview/export khong cham sat mep giay
FEED_RATE  = 1600       # mm/min, kept for compatibility
DRAW_FEED_RATE = 1500   # mm/min while drawing
TRAVEL_FEED_RATE = 4200 # mm/min while moving with pen up
MIN_DRAW_FEED_FACTOR = 0.45
RAMP_SEGMENTS = 4
ACCEL_DISTANCE_MM = 8.0       # mm dung cho trapezoidal velocity ramp moi stroke
MIN_DRAW_FEED_RATE = 650      # feedrate thap nhat khi tang/giam toc dau-cuoi stroke
MAX_DRAW_FEED_RATE = DRAW_FEED_RATE
DRAW_FEED_QUANTUM = 25        # Giam thay doi F lien tuc ma khong doi toc do dang ke
Z_DRAW = 0.0
Z_HOVER = 0.45
Z_TRAVEL = 1.4
HOVER_TRAVEL_MM = 12.0
PEN_Z_MOVE_TIME_S = 0.28
COLLINEAR_ANGLE_DEG = 2.2
COLLINEAR_DIST_MM = 0.035
TWO_OPT_MAX_PATHS = 900
TWO_OPT_MAX_PASSES = 2
OR_OPT_MAX_PATHS = 650        # Or-opt cham hon 2-opt nen gioi han path thap hon
OR_OPT_MAX_PASSES = 2
OR_OPT_MAX_BLOCK = 3          # Di chuyen cum 1-3 stroke lien tiep
ROUTE_IMPROVE_TIME_BUDGET_S = 1.6
RDP_EPS_WARNING_MM = 2.5
RDP_REBALANCE_MAX_PASSES = 4
STROKE_COUNT_SOFT_TARGET_RATIO = 0.24
STROKE_COUNT_SOFT_TARGET_MIN = 90
STROKE_COUNT_SOFT_TARGET_MAX = 620
MICRO_TEXTURE_GRID_MM = 8.0
MICRO_TEXTURE_DENSE_COUNT = 12
MICRO_TEXTURE_KEEP_RATIO = 0.38
CLIP_EPS_MM = 0.02            # Dung sai cat path vao trong vung giay an toan
DEDUP_ENDPOINT_TOL_MM = 0.65  # Nguong coi hai stroke la trung dau/cuoi
DEDUP_LENGTH_TOL_RATIO = 0.08
BUDGET_GRID_MM = 32.0
BUDGET_NEARBY_LIMIT = 180
ADAPTIVE_RDP_CURVE_GAIN = 0.65
SMOOTHING_MAX_POINTS = 16000  # Qua nhieu diem thi bo smoothing de giu UI nhanh
SMOOTHING_WEIGHT = 0.18
SMOOTHING_TURN_KEEP_DEG = 28.0
ISLAND_GRID_MM = 28.0
ISLAND_MAX_PATHS = 5000
STITCH_SCORE_THRESHOLD = 0.62
STITCH_MAX_GAP_MM = 4.0
PEN_LIFT_BRIDGE_MM = 0.85       # Gap rat ngan se ve noi luon de giam nhac but
PEN_LIFT_BRIDGE_MAX_MM = 1.35   # Gap dai hon chi noi neu huong/score du tot
PEN_LIFT_BRIDGE_SCORE = 0.70
CORNER_SLOWDOWN_DEG = 22.0
CORNER_LOOKAHEAD_SEGMENTS = 2
GCODE_MIN_MOVE_MM = 0.015
GCODE_DECIMALS = 3
RESAMPLE_TARGET_MM = 1.15
RESAMPLE_MAX_SEGMENT_MM = 2.6
RESAMPLE_MAX_EXTRA_POINTS = 9000
RESAMPLE_MAX_TOTAL_POINTS = 36000
STROKE_DIRECTION_MAX_PASSES = 2
DRY_RUN_BOUNDS_TOL_MM = 0.05
STROKE_MERGE_ANGLE_TOLERANCE_DEG = 12.0
STROKE_MERGE_ENDPOINT_GAP_MM = 0.60
STROKE_MERGE_RDP_EPSILON_MM = 0.06
STROKE_MERGE_MAX_PASSES = 18
OVERLAP_REDUCTION_DISTANCE_MM = 0.35
OVERLAP_REDUCTION_FRACTION = 0.78
OVERLAP_REDUCTION_ANGLE_DEG = 12.0
OVERLAP_REDUCTION_GRID_MM = 6.0
OVERLAP_REDUCTION_MAX_PATHS = 5200
QA_MIN_ITERATIONS = 1
QA_TIMEOUT_S = 60.0
QA_RENDER_LINE_WIDTH_PX = 1
QA_OVERLAY_REFERENCE_INK_THRESHOLD = 210
QA_OVERLAY_PREVIEW_INK_THRESHOLD = 245
QA_OVERLAY_TOLERANCE_PX = 3
QA_OVERLAY_MIN_MISSING_PIXELS = 16
QA_OVERLAY_MIN_MISSING_FRACTION = 0.006
QA_OVERLAY_PASS_COVERAGE = 1.0 - QA_OVERLAY_MIN_MISSING_FRACTION
QA_OVERLAY_MISSING_COMPONENT_MIN_AREA_PX = 8
QA_OVERLAY_MAX_NEW_CANDIDATES = 120
FACE_FIXED_BUDGET_RATIO = 0.26
FACE_FIXED_BUDGET_MIN = 36
FACE_FIXED_BUDGET_MAX_RATIO = 0.42
FACE_RDP_EPSILON_MM = 0.045
MAIN_OUTLINE_RDP_EPSILON_MM = 0.08
GARMENT_OUTLINE_BUDGET_RATIO = 0.46
MANUAL_BRUSH_MIN_PX = 5
MANUAL_BRUSH_MAX_PX = 150
MANUAL_BRUSH_DEFAULT_PX = 42
MANUAL_ADD_DETAIL_MAX_NEW = 160
MANUAL_ADD_OVERLAP_THRESHOLD = 0.07
MANUAL_REDUCE_OVERLAP_NORMAL = 0.30
MANUAL_REDUCE_OVERLAP_STRONG = 0.20
MANUAL_REDUCE_SCORE_NORMAL = 3.55
MANUAL_REDUCE_SCORE_STRONG = 4.20
MANUAL_REDUCE_FACE_WARN_RATIO = 0.18
MANUAL_QA_QUICK_TIMEOUT_S = 18.0

MIN_STROKE_BUDGET     = 300
MAX_STROKE_BUDGET     = 30000
DEFAULT_STROKE_BUDGET = 1600
AUTO_BUDGET_BASE_SEGMENTS = 520
AUTO_BUDGET_SQRT_SCALE = 12.0
AUTO_BUDGET_MIN_SEGMENTS = 700
AUTO_BUDGET_MAX_SEGMENTS = 2200
LINE_ART_DARK_THRESHOLD = 120
LINE_ART_LIGHT_THRESHOLD = 225
LINE_ART_MAX_MIDTONE_FRACTION = 0.12
LINE_ART_MIN_DARK_FRACTION = 0.0004
LINE_ART_MAX_DARK_FRACTION = 0.34
LINE_ART_TRACE_MIN_KEEP_PX = 14.0
LINE_ART_AUTO_RDP_EPSILON_MM = 0.42
LINE_ART_AUTO_BUDGET_HEADROOM = 1.18
FILLED_REGION_MIN_AREA_PX = 24
FILLED_REGION_MIN_SIZE_PX = 8
FILLED_REGION_FILL_RATIO = 0.56
FILLED_REGION_MAX_ASPECT = 8.0
FILLED_REGION_MIN_DISTANCE_PX = 2.4
FILLED_REGION_HATCH_SPACING_PX = 8
FILLED_REGION_HATCH_ANGLE_DEG = -38.0

VECTOR_MAX_DIM = 850
HATCH_MAX_DIM = 900
DEFAULT_HATCH_CELL_SIZE = 18
DEFAULT_HATCH_ANGLE_DEG = 35.0
DEFAULT_HATCH_MIN_SPACING = 3
DEFAULT_HATCH_MAX_SPACING = 18
DEFAULT_HATCH_DARK_THRESHOLD = 160.0
HATCH_DARK_EDGE_MARGIN = 10.0
HATCH_PRESETS = OrderedDict([
    ("Compact", dict(cell_size=24, angle_deg=35.0, min_spacing=7,
                     max_spacing=28, dark_threshold=145.0)),
    ("Balanced", dict(cell_size=18, angle_deg=35.0, min_spacing=3,
                      max_spacing=18, dark_threshold=160.0)),
    ("Quality", dict(cell_size=14, angle_deg=35.0, min_spacing=3,
                     max_spacing=14, dark_threshold=170.0)),
])
HATCH_CANDIDATE_IMPORTANCE = 0.42  # Hatch di rieng vao budget, uu tien vua phai de khong lan at contour.
HATCH_CANDIDATE_SALIENCY = 0.40    # Saliency mac dinh giup hatch khong bi cat sach o budget thap.
HATCH_CANDIDATE_DETAIL_TIER = 2    # Hatch la fine detail: uu tien net dai, cat truoc contour khi budget thap.
HATCH_VECTOR_MIN_POINT_DIST_PX = 0.55
HATCH_VECTOR_COLLINEAR_DIST_PX = 0.18
HATCH_VECTOR_MIN_KEEP_MM = 0.65
HATCH_CLIP_SAMPLE_STEP_PX = 1.6
DEFAULT_FACE_MASK_FEATHER_PX = 6
FACE_DNN_CONFIDENCE = 0.50
FOREGROUND_MAX_DIM = 900
FOREGROUND_BORDER_FRAC = 0.055
FOREGROUND_MIN_FRACTION = 0.002
FOREGROUND_MAX_FRACTION = 0.94
DETAIL_TIER_BASE_RATIO = 0.55   # Phan budget uu tien giu contour/base stroke
DETAIL_TIER_MED_RATIO = 0.32    # Phan budget tiep theo cho detail trung binh
FINE_DETAIL_TRAVEL_PENALTY = 0.065
BASE_DETAIL_TRAVEL_PENALTY = 0.018

# G-code optimization thresholds
STITCH_THRESHOLD_MM = 3.2    # max gap (mm) to bridge without pen lift
STITCH_ANGLE_COS    = 0.5    # min cosine similarity for stitching (~60 deg)
MICRO_MM            = 0.8    # paths shorter than this (isolated) are removed
SEAMLESS_MM         = 0.35   # within this distance, skip G0 lift entirely
GREEDY_NN_LIMIT     = 4000   # above this, fall back to serpentine ordering


# ═══════════════════════════════════════════════════════════════════
#  DATA MODEL
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CandidatePath:
    points:     np.ndarray  # shape (N,2), coordinates in mm (x, y)
    importance: float       # 0..1
    length_mm:  float
    detail_tier: int = 1     # 0=base contour, 1=medium detail, 2=fine texture
    saliency: float = 0.0    # Diem thi giac dung de cat giam stroke thong minh
    region: str = "detail"   # face, garment_outline, garment_detail, hatch, detail
    source: str = "lineart"  # lineart, hatch, manual_add
    protected: bool = False
    classifier_scores: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AIMaps:
    foreground: np.ndarray
    saliency: np.ndarray
    hatch_weight: np.ndarray
    backend: str = "optional"
    elapsed_s: float = 0.0


@dataclass
class ForegroundMaskInfo:
    source: str = "full-frame"
    has_alpha: bool = False
    estimated: bool = False
    fallback_full: bool = False
    warning: str = ""
    foreground_fraction: float = 1.0
    background_mask: Optional[np.ndarray] = None


@dataclass
class StrokeTier:
    detail_tier: int = 1
    importance_score: float = 0.5
    saliency_score: float = 0.5
    region: str = "detail"
    protected: bool = False
    notes: str = ""


@dataclass
class StrokeClassificationContext:
    canvas_size: tuple = (1, 1)
    reference_img: Optional[Image.Image] = None
    lineart_img: Optional[Image.Image] = None
    ai_maps: Optional[object] = None
    face_mask: Optional[np.ndarray] = None
    extra_masks: dict = field(default_factory=dict)
    cache: dict = field(default_factory=dict)


@dataclass
class QAIteration:
    index: int
    ssim: float
    stroke_count: int
    pen_lifts: int
    draw_distance_mm: float
    estimated_time_s: float
    changes_applied: list = field(default_factory=list)
    model_score: Optional[float] = None
    redundancy_notes: str = ""
    missing_detail_notes: str = ""
    elapsed_s: float = 0.0


@dataclass
class QAReport:
    iterations: list = field(default_factory=list)
    final_stroke_count: int = 0
    final_ssim: float = 0.0
    final_model_score: Optional[float] = None
    passed: bool = False
    timed_out: bool = False
    total_time_s: float = 0.0
    model_status: str = "not_run"

    def to_dict(self):
        return {
            "iterations": [
                {
                    "index": it.index,
                    "ssim": it.ssim,
                    "model_score": it.model_score,
                    "stroke_count": it.stroke_count,
                    "pen_lifts": it.pen_lifts,
                    "draw_distance_mm": it.draw_distance_mm,
                    "estimated_time_s": it.estimated_time_s,
                    "changes_applied": list(it.changes_applied),
                    "redundancy_notes": it.redundancy_notes,
                    "missing_detail_notes": it.missing_detail_notes,
                    "elapsed_s": it.elapsed_s,
                }
                for it in self.iterations
            ],
            "final_stroke_count": self.final_stroke_count,
            "final_ssim": self.final_ssim,
            "final_model_score": self.final_model_score,
            "passed": self.passed,
            "timed_out": self.timed_out,
            "total_time_s": self.total_time_s,
            "model_status": self.model_status,
        }


@dataclass
class GCodePlan:
    gcode_lines:       list
    paths:             list
    target_segments:   int
    actual_segments:   int
    stroke_count:      int
    pen_lifts:         int
    command_count:     int
    raw_segment_count: int
    used_epsilon:      float
    stitched_count:    int   = 0
    travel_distance_mm: float = 0.0
    draw_distance_mm: float = 0.0
    estimated_time_s: float = 0.0
    travel_feed_rate: int = TRAVEL_FEED_RATE
    draw_feed_rate: int = DRAW_FEED_RATE
    command_count_before_compression: int = 0
    compressed_segment_count: int = 0
    compression_removed_count: int = 0
    route_improvement_mm: float = 0.0
    route_improvement_pct: float = 0.0
    clipped_path_count: int = 0
    deduplicated_path_count: int = 0
    island_count: int = 0
    gcode_postprocess_removed_count: int = 0
    resampled_point_count: int = 0
    stroke_direction_flip_count: int = 0
    dry_run_warning_count: int = 0
    selected_detail_tier_counts: dict = field(default_factory=dict)
    pen_lift_bridge_count: int = 0
    planning_time_s: float = 0.0
    gcode_size_bytes: int = 0
    ai_backend: str = "classical"
    ai_elapsed_s: float = 0.0
    hatch_candidate_count: int = 0
    stroke_merge_count: int = 0
    overlap_removed_count: int = 0
    stroke_merge_reduction_pct: float = 0.0
    qa_report: Optional[QAReport] = None
    rdp_eps_warning: bool = False
    validation_errors: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
#  IMAGE PROCESSING — UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _soften_alpha(alpha):
    """Apply a small anti-aliasing blur before alpha compositing/thresholding."""
    return cv2.GaussianBlur(
        np.asarray(alpha, dtype=np.uint8), (3, 3), 0,
        borderType=cv2.BORDER_REPLICATE,
    )


def _mask_fraction(mask):
    arr = np.asarray(mask, dtype=np.uint8)
    return cv2.countNonZero(arr) / max(1, arr.size)


def _has_valid_alpha_mask(alpha):
    arr = np.asarray(alpha, dtype=np.uint8)
    if arr.size == 0:
        return False
    transparent = int(np.count_nonzero(arr <= 245))
    opaque = int(np.count_nonzero(arr > 12))
    return transparent >= max(8, int(arr.size * 0.002)) and opaque > 0


def _border_sample_mask(height, width, border_px=None):
    border = max(2, int(round(min(height, width) * FOREGROUND_BORDER_FRAC))
                 if border_px is None else int(border_px))
    border = min(border, max(2, height // 3), max(2, width // 3))
    mask = np.zeros((height, width), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask


def _connected_border_components(binary_bool):
    src = np.asarray(binary_bool, dtype=np.uint8)
    if src.size == 0 or cv2.countNonZero(src) == 0:
        return np.zeros_like(src, dtype=np.uint8)
    num, labels = cv2.connectedComponents(src, connectivity=8)
    if num <= 1:
        return np.zeros_like(src, dtype=np.uint8)
    h, w = src.shape
    border = _border_sample_mask(h, w, border_px=2)
    touching = set(int(v) for v in np.unique(labels[border]) if int(v) != 0)
    if not touching:
        return np.zeros_like(src, dtype=np.uint8)
    out = np.isin(labels, list(touching)).astype(np.uint8) * 255
    return out.astype(np.uint8)


def _estimate_paper_background_mask(rgb):
    """
    Detect paper/photo background connected to the frame border.

    This is intentionally conservative: only bright-ish, low-chroma, border-
    connected regions are removed, so dark drawing strokes remain available to
    Canny/DoG while paper gradients stop driving hatch coverage.
    """
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    h, w = arr.shape[:2]
    if h < 8 or w < 8:
        return np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1] - 128.0
    b_chan = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a_chan * a_chan + b_chan * b_chan)
    border = _border_sample_mask(h, w)
    border_lab = lab[border]
    border_l = l_chan[border]
    border_chroma = chroma[border]
    if border_lab.size == 0:
        return np.zeros((h, w), dtype=np.uint8)

    median_lab = np.median(border_lab, axis=0)
    diff = lab - median_lab[None, None, :]
    diff[:, :, 0] *= 0.70
    color_dist = np.sqrt(np.sum(diff * diff, axis=2))
    border_dist = color_dist[border]
    dist_thr = max(
        14.0,
        min(64.0, float(np.percentile(border_dist, 92.0)) * 2.35 + 5.0),
    )
    l_floor = max(92.0, float(np.percentile(border_l, 12.0)) - 38.0)
    chroma_limit = max(24.0, min(70.0, float(np.percentile(border_chroma, 90.0)) + 24.0))

    paper_like = (
        ((color_dist <= dist_thr) |
         ((l_chan >= l_floor) & (chroma <= chroma_limit))) &
        (l_chan >= 78.0)
    )
    background = _connected_border_components(paper_like)
    if cv2.countNonZero(background) == 0:
        return background

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    background = cv2.morphologyEx(background, cv2.MORPH_CLOSE, kernel, iterations=2)
    background = cv2.morphologyEx(background, cv2.MORPH_OPEN, kernel, iterations=1)
    return np.where(background > 0, 255, 0).astype(np.uint8)


def _cleanup_foreground_mask(mask, preserve_thin=True):
    src = np.where(np.asarray(mask, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    if src.size == 0 or cv2.countNonZero(src) == 0:
        return src
    h, w = src.shape
    min_area = max(3, int(src.size * (0.000015 if preserve_thin else 0.00012)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    cleaned = np.zeros_like(src)
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == lbl] = 255
    if cv2.countNonZero(cleaned) == 0:
        cleaned = src
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return np.where(cleaned > 0, 255, 0).astype(np.uint8)


def _estimate_foreground_grabcut(rgb, background_seed=None):
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h < 20 or w < 20:
        return None
    try:
        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        if background_seed is not None:
            bg = _resize_binary_mask(background_seed, w, h)
            if bg is not None:
                gc_mask[bg > 0] = cv2.GC_BGD
                gc_mask[bg == 0] = cv2.GC_PR_FGD
        margin_x = max(2, int(round(w * 0.08)))
        margin_y = max(2, int(round(h * 0.08)))
        gc_mask[margin_y:h - margin_y, margin_x:w - margin_x] = np.where(
            gc_mask[margin_y:h - margin_y, margin_x:w - margin_x] == cv2.GC_BGD,
            cv2.GC_BGD,
            cv2.GC_PR_FGD,
        )
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        border = _border_sample_mask(h, w)
        dark_seed = gray < max(35.0, float(np.percentile(gray[border], 8.0)) - 28.0)
        gc_mask[dark_seed] = cv2.GC_FGD
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(arr, gc_mask, None, bgd_model, fgd_model, 3,
                    cv2.GC_INIT_WITH_MASK)
        fg = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
        return _cleanup_foreground_mask(fg, preserve_thin=False)
    except Exception as error:
        print(f"Foreground GrabCut warning: {type(error).__name__}: {error}")
        return None


def _estimate_foreground_mask_non_alpha(rgb):
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8), ForegroundMaskInfo()

    scale = min(1.0, float(FOREGROUND_MAX_DIM) / float(max(h, w)))
    if scale < 1.0:
        work_w = max(1, int(round(w * scale)))
        work_h = max(1, int(round(h * scale)))
        work = cv2.resize(arr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    else:
        work = arr

    paper_bg = _estimate_paper_background_mask(work)
    paper_fg = cv2.bitwise_not(paper_bg)
    paper_fg = _cleanup_foreground_mask(paper_fg, preserve_thin=True)
    if scale < 1.0:
        paper_bg_full = cv2.resize(paper_bg, (w, h), interpolation=cv2.INTER_NEAREST)
        paper_fg_full = cv2.resize(paper_fg, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        paper_bg_full = paper_bg
        paper_fg_full = paper_fg

    frac = _mask_fraction(paper_fg_full)
    if FOREGROUND_MIN_FRACTION <= frac <= FOREGROUND_MAX_FRACTION:
        info = ForegroundMaskInfo(
            source="paper-border",
            has_alpha=False,
            estimated=True,
            fallback_full=False,
            warning=(
                "Input has no alpha; foreground was estimated from paper/background."
            ),
            foreground_fraction=float(frac),
            background_mask=np.where(paper_bg_full > 0, 255, 0).astype(np.uint8),
        )
        print(
            f"Foreground mask: estimated paper-border foreground "
            f"{frac * 100.0:.1f}% of image"
        )
        return np.where(paper_fg_full > 0, 255, 0).astype(np.uint8), info

    grabcut_fg = _estimate_foreground_grabcut(work, paper_bg)
    if grabcut_fg is not None:
        if scale < 1.0:
            grabcut_fg = cv2.resize(grabcut_fg, (w, h), interpolation=cv2.INTER_NEAREST)
        grabcut_fg = _cleanup_foreground_mask(grabcut_fg, preserve_thin=False)
        frac = _mask_fraction(grabcut_fg)
        if FOREGROUND_MIN_FRACTION <= frac <= FOREGROUND_MAX_FRACTION:
            background = cv2.bitwise_not(grabcut_fg)
            info = ForegroundMaskInfo(
                source="grabcut",
                has_alpha=False,
                estimated=True,
                fallback_full=False,
                warning=(
                    "Input has no alpha; foreground was estimated with GrabCut."
                ),
                foreground_fraction=float(frac),
                background_mask=background,
            )
            print(
                f"Foreground mask: GrabCut foreground {frac * 100.0:.1f}% of image"
            )
            return grabcut_fg, info

    full = np.full((h, w), 255, dtype=np.uint8)
    warning = (
        "Input has no alpha and foreground segmentation failed; "
        "using full image, hatch may include paper/background."
    )
    print(f"Foreground mask warning: {warning}")
    return full, ForegroundMaskInfo(
        source="full-frame",
        has_alpha=False,
        estimated=False,
        fallback_full=True,
        warning=warning,
        foreground_fraction=1.0,
        background_mask=np.zeros((h, w), dtype=np.uint8),
    )


def _composite_rgb_and_mask(pil_img, return_info=False):
    info = ForegroundMaskInfo()
    if pil_img.mode == "RGBA":
        rgba    = np.array(pil_img.convert("RGBA"))
        alpha   = _soften_alpha(rgba[:, :, 3])
        rgb     = rgba[:, :, :3].astype(np.float32)
        a       = (alpha.astype(np.float32) / 255.0)[..., None]
        rgb     = rgb * a + 255.0 * (1.0 - a)
        rgb     = np.clip(rgb, 0, 255).astype(np.uint8)
        if _has_valid_alpha_mask(alpha):
            fg_mask = np.where(alpha > 12, 255, 0).astype(np.uint8)
            frac = _mask_fraction(fg_mask)
            info = ForegroundMaskInfo(
                source="alpha",
                has_alpha=True,
                estimated=False,
                fallback_full=False,
                foreground_fraction=float(frac),
                background_mask=np.where(fg_mask > 0, 0, 255).astype(np.uint8),
            )
            print(f"Foreground mask: using alpha foreground {frac * 100.0:.1f}%")
        else:
            fg_mask, info = _estimate_foreground_mask_non_alpha(rgb)
    else:
        rgb     = np.array(pil_img.convert("RGB"))
        fg_mask, info = _estimate_foreground_mask_non_alpha(rgb)
    return (rgb, fg_mask, info) if return_info else (rgb, fg_mask)


def _safe_percentile(values, q, fallback):
    if values.size == 0:
        return fallback
    return float(np.percentile(values, q))


def _normalize_u8(arr):
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    out = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(out, 0, 255).astype(np.uint8)


def _looks_like_clean_line_art_gray(gray):
    """Detect black/white line-art so it is not edge-detected a second time."""
    values = np.asarray(gray, dtype=np.uint8)
    if values.size == 0:
        return False
    flat = values.reshape(-1)
    dark_fraction = float(np.count_nonzero(flat <= LINE_ART_DARK_THRESHOLD)) / flat.size
    light_fraction = float(np.count_nonzero(flat >= LINE_ART_LIGHT_THRESHOLD)) / flat.size
    mid_fraction = 1.0 - dark_fraction - light_fraction
    contrast = float(np.percentile(flat, 96) - np.percentile(flat, 4))
    return (
        contrast >= 95.0 and
        light_fraction >= 0.52 and
        LINE_ART_MIN_DARK_FRACTION <= dark_fraction <= LINE_ART_MAX_DARK_FRACTION and
        mid_fraction <= LINE_ART_MAX_MIDTONE_FRACTION
    )


def _looks_like_clean_line_art(rgb):
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim == 2:
        gray = arr
    else:
        gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    return _looks_like_clean_line_art_gray(gray)


def _clean_line_art_binary_from_rgb(rgb, fg_mask):
    """Extract actual ink from clean line-art with one threshold and light cleanup."""
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    if gray.size == 0:
        return np.zeros_like(gray, dtype=np.uint8)

    otsu_threshold, _ = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    ink_threshold = int(np.clip(max(150.0, otsu_threshold + 28.0), 80.0, 220.0))
    binary = np.where(gray <= ink_threshold, 255, 0).astype(np.uint8)

    fg = _resize_binary_mask(fg_mask, gray.shape[1], gray.shape[0])
    if fg is not None and cv2.countNonZero(fg) > 0:
        binary = cv2.bitwise_and(binary, fg)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    min_area = max(2, int(round(binary.size * 0.00001)))
    return _adaptive_component_cleanup(binary, min_area=min_area)


def _filled_component_score(comp, area, width, height):
    if area < FILLED_REGION_MIN_AREA_PX:
        return False, 0.0, 0.0
    if min(width, height) < FILLED_REGION_MIN_SIZE_PX:
        return False, 0.0, 0.0
    aspect = max(width, height) / float(max(1, min(width, height)))
    if aspect > FILLED_REGION_MAX_ASPECT:
        return False, 0.0, aspect
    fill_ratio = area / float(max(1, width * height))
    if fill_ratio < FILLED_REGION_FILL_RATIO:
        return False, fill_ratio, aspect
    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    max_dist = float(dist.max()) if dist.size else 0.0
    return max_dist >= FILLED_REGION_MIN_DISTANCE_PX, fill_ratio, aspect


def _diagonal_hatch_mask_for_component(comp, spacing_px=FILLED_REGION_HATCH_SPACING_PX,
                                       angle_deg=FILLED_REGION_HATCH_ANGLE_DEG):
    mask = np.where(np.asarray(comp, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    h, w = mask.shape
    if h <= 1 or w <= 1 or cv2.countNonZero(mask) == 0:
        return np.zeros_like(mask)

    inset = 2 if min(w, h) >= 14 else 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    interior = cv2.erode(mask, kernel, iterations=inset)
    if cv2.countNonZero(interior) == 0:
        interior = mask

    spacing = max(4, int(round(float(spacing_px))))
    angle = math.radians(float(angle_deg))
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    center = np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float32)
    span = int(math.ceil(math.hypot(w, h))) + spacing + 2

    lines = np.zeros_like(mask)
    for offset in range(-span, span + 1, spacing):
        p0 = center + normal * float(offset) - direction * float(span)
        p1 = center + normal * float(offset) + direction * float(span)
        cv2.line(
            lines,
            (int(round(p0[0])), int(round(p0[1]))),
            (int(round(p1[0])), int(round(p1[1]))),
            255,
            thickness=1,
            lineType=cv2.LINE_8,
        )
    return cv2.bitwise_and(lines, interior)


def _symbolic_filled_component(comp):
    comp = np.where(np.asarray(comp, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    out = np.zeros_like(comp)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, 255, 1, lineType=cv2.LINE_8)
    hatch = _diagonal_hatch_mask_for_component(comp)
    return cv2.bitwise_or(out, hatch)


def _clean_line_art_draw_mask(binary_255):
    """
    Convert thresholded sketch ink into plotter strokes.

    Thin line strokes are skeletonized to one centerline. Filled black regions
    are rendered as an outline plus sparse diagonal hatching so the robot does
    not waste commands filling a blob.
    """
    src = np.where(np.asarray(binary_255, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    if src.size == 0 or cv2.countNonZero(src) == 0:
        return src, 0

    out = np.zeros_like(src)
    filled_count = 0
    num, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area <= 0 or w <= 0 or h <= 0:
            continue

        comp = np.where(labels[y:y + h, x:x + w] == lbl, 255, 0).astype(np.uint8)
        looks_filled, _fill_ratio, _aspect = _filled_component_score(comp, area, w, h)
        rendered = _symbolic_filled_component(comp) if looks_filled else _thin_binary(comp)
        if looks_filled:
            filled_count += 1
        out[y:y + h, x:x + w] = cv2.bitwise_or(out[y:y + h, x:x + w], rendered)

    return np.where(out > 0, 255, 0).astype(np.uint8), filled_count


def _adaptive_component_cleanup(binary_255, min_area=None):
    """Remove tiny connected components (noise/dust) while preserving real lines."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary_255, connectivity=8)
    total_active = max(1, int(np.count_nonzero(binary_255)))
    if min_area is None:
        min_area = max(4, int(total_active * 0.003))
    out = np.zeros_like(binary_255)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            out[labels == lbl] = 255
    return out


def _force_outline_only(binary_255):
    mask, _filled_count = _clean_line_art_draw_mask(binary_255)
    return mask


def _apply_exclusion_mask(pil_img, exclude_bottom_pct=0.0, exclude_right_pct=0.0):
    """Xoa vung watermark/logo o goc duoi phai truoc khi dua vao pipeline chinh."""
    bottom_pct = float(np.clip(exclude_bottom_pct, 0.0, 45.0))
    right_pct = float(np.clip(exclude_right_pct, 0.0, 45.0))
    if bottom_pct <= 0.0 or right_pct <= 0.0:
        return pil_img

    img = pil_img.convert("RGBA")
    w, h = img.size
    x0 = int(round(w * (1.0 - right_pct / 100.0)))
    y0 = int(round(h * (1.0 - bottom_pct / 100.0)))

    # To trang va alpha=0 trong vung loai tru de line-art khong nhin thay logo.
    arr = np.array(img)
    arr[y0:h, x0:w, :3] = 255
    arr[y0:h, x0:w, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


_mediapipe_face_warning_printed = False
_opencv_face_dnn_warning_printed = False

_MEDIAPIPE_FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]


def _feather_binary_mask(mask, feather_px=DEFAULT_FACE_MASK_FEATHER_PX):
    """Blur and re-threshold a 0/255 mask so hatch clipping does not look hard."""
    src = np.where(np.asarray(mask, dtype=np.uint8) > 0, 255, 0).astype(np.uint8)
    feather = max(0, int(round(float(feather_px))))
    if src.size == 0 or feather <= 0 or cv2.countNonZero(src) == 0:
        return src
    sigma = max(0.1, feather / 3.0)
    blurred = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma,
                               borderType=cv2.BORDER_REPLICATE)
    _, smoothed = cv2.threshold(blurred, 64, 255, cv2.THRESH_BINARY)
    kernel_size = max(1, min(9, feather | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel, iterations=1)


def _resize_binary_mask(mask, width, height):
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0 or arr.ndim != 2:
        return None
    if arr.shape != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height),
                         interpolation=cv2.INTER_NEAREST)
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _opencv_face_dnn_paths():
    proto_env = os.environ.get("OPENCV_FACE_PROTO")
    model_env = os.environ.get("OPENCV_FACE_MODEL")
    if proto_env and model_env and os.path.isfile(proto_env) and os.path.isfile(model_env):
        return proto_env, model_env

    cwd = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    proto_names = ["deploy.prototxt", "deploy.prototxt.txt"]
    model_names = [
        "res10_300x300_ssd_iter_140000.caffemodel",
        "opencv_face_detector.caffemodel",
    ]
    folders = []
    for base in (cwd, module_dir):
        for folder in (base, os.path.join(base, "models"), os.path.join(base, "data")):
            if folder not in folders:
                folders.append(folder)
    for folder in folders:
        for proto_name in proto_names:
            proto = os.path.join(folder, proto_name)
            if not os.path.isfile(proto):
                continue
            for model_name in model_names:
                model = os.path.join(folder, model_name)
                if os.path.isfile(model):
                    return proto, model
    return None, None


def _detect_face_region_opencv_dnn(rgb, feather_px):
    global _opencv_face_dnn_warning_printed

    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    proto, model = _opencv_face_dnn_paths()
    if not proto or not model:
        if not _opencv_face_dnn_warning_printed:
            print(
                "Face mask warning: OpenCV DNN face model not found "
                "(set OPENCV_FACE_PROTO and OPENCV_FACE_MODEL to enable fallback)."
            )
            _opencv_face_dnn_warning_printed = True
        return mask, 0

    detections_count = 0
    try:
        net = cv2.dnn.readNetFromCaffe(proto, model)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(
            cv2.resize(bgr, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        net.setInput(blob)
        detections = net.forward()
        for idx in range(detections.shape[2]):
            confidence = float(detections[0, 0, idx, 2])
            if confidence < FACE_DNN_CONFIDENCE:
                continue
            x0, y0, x1, y1 = detections[0, 0, idx, 3:7] * np.array(
                [width, height, width, height], dtype=np.float32)
            x0 = int(np.clip(round(x0), 0, width - 1))
            y0 = int(np.clip(round(y0), 0, height - 1))
            x1 = int(np.clip(round(x1), 0, width - 1))
            y1 = int(np.clip(round(y1), 0, height - 1))
            if x1 <= x0 or y1 <= y0:
                continue

            box_w = float(x1 - x0)
            box_h = float(y1 - y0)
            center = (int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.54)))
            axes = (
                max(1, int(round(box_w * 0.56))),
                max(1, int(round(box_h * 0.68))),
            )
            cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 255,
                        thickness=-1, lineType=cv2.LINE_AA)
            detections_count += 1
    except Exception as error:
        if not _opencv_face_dnn_warning_printed:
            print(f"Face mask warning: OpenCV DNN fallback failed: {error}")
            _opencv_face_dnn_warning_printed = True
        return np.zeros((height, width), dtype=np.uint8), 0

    if detections_count:
        mask = _feather_binary_mask(mask, feather_px)
    return mask, detections_count


def detect_face_region(pil_img, feather_px=DEFAULT_FACE_MASK_FEATHER_PX):
    """
    Return a 0/255 face-oval mask. Multiple faces are unioned.

    MediaPipe FaceMesh gives the precise oval. If it is unavailable, an OpenCV
    DNN detector can supply an ellipse fallback when its Caffe files are present.
    Missing optional dependencies never crash the app; the safe result is no
    hatch exclusion.
    """
    global _mediapipe_face_warning_printed

    if not isinstance(pil_img, Image.Image):
        return np.zeros((1, 1), dtype=np.uint8)
    width, height = pil_img.size
    mask = np.zeros((height, width), dtype=np.uint8)
    if width <= 0 or height <= 0:
        return mask

    rgb = np.asarray(pil_img.convert("RGB"))
    try:
        import mediapipe as mp

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=8,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        try:
            result = face_mesh.process(rgb)
        finally:
            face_mesh.close()

        face_count = 0
        for face_landmarks in (result.multi_face_landmarks or []):
            landmarks = face_landmarks.landmark
            points = []
            for index in _MEDIAPIPE_FACE_OVAL_LANDMARKS:
                if index >= len(landmarks):
                    continue
                lm = landmarks[index]
                x = int(np.clip(round(lm.x * (width - 1)), 0, width - 1))
                y = int(np.clip(round(lm.y * (height - 1)), 0, height - 1))
                points.append((x, y))
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255,
                             lineType=cv2.LINE_AA)
                face_count += 1

        if face_count:
            mask = _feather_binary_mask(mask, feather_px)
            coverage = 100.0 * cv2.countNonZero(mask) / max(1, width * height)
            print(
                f"Face mask: MediaPipe detected {face_count} face(s), "
                f"coverage={coverage:.1f}%, feather={int(round(float(feather_px)))}px"
            )
            return mask

        print("Face mask: MediaPipe detected 0 faces; hatch exclusion disabled")
        return mask

    except Exception as error:
        if not _mediapipe_face_warning_printed:
            print(
                "Face mask warning: MediaPipe FaceMesh unavailable; "
                f"trying OpenCV DNN fallback ({type(error).__name__}: {error})."
            )
            _mediapipe_face_warning_printed = True

    dnn_mask, dnn_count = _detect_face_region_opencv_dnn(rgb, feather_px)
    if dnn_count:
        coverage = 100.0 * cv2.countNonZero(dnn_mask) / max(1, width * height)
        print(
            f"Face mask: OpenCV DNN detected {dnn_count} face(s), "
            f"coverage={coverage:.1f}%, feather={int(round(float(feather_px)))}px"
        )
    else:
        print("Face mask: no face detected; hatch exclusion disabled")
    return dnn_mask


# ═══════════════════════════════════════════════════════════════════
#  AUTO LINE-ART PIPELINE  (sketch/line-art input)
# ═══════════════════════════════════════════════════════════════════

def _mask_value_at(mask, x, y):
    xi = int(round(float(x)))
    yi = int(round(float(y)))
    h, w = mask.shape
    return 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0


def _append_if_distinct(points, point, min_dist_px=0.25):
    pt = (float(point[0]), float(point[1]))
    if not points or float(np.hypot(pt[0] - points[-1][0], pt[1] - points[-1][1])) >= min_dist_px:
        points.append(pt)


def _resize_ai_map(ai_map, width, height, binary=False):
    if ai_map is None:
        return None
    arr = np.asarray(ai_map)
    if arr.size == 0 or arr.ndim != 2 or not np.all(np.isfinite(arr)):
        return None
    interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    if arr.shape != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height), interpolation=interpolation)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _clip_hatch_vectors_to_mask(hatch_lines, fg_mask):
    """
    Cat hatch vector theo fg_mask/exclude mask truoc khi dua vao G-code.
    Truoc day hatch raster bi AND voi mask; khi tach vector rieng, ta sample segment
    va tach polyline tai cac doan roi khoi foreground de khong ve xuyen nen/logo.
    """
    clipped = []
    for polyline in hatch_lines:
        if len(polyline) < 2:
            continue
        current = []
        for p0, p1 in zip(polyline[:-1], polyline[1:]):
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            dist = float(np.hypot(x1 - x0, y1 - y0))
            if dist <= 1e-6:
                continue

            mid = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
            if (_mask_value_at(fg_mask, x0, y0) and
                    _mask_value_at(fg_mask, mid[0], mid[1]) and
                    _mask_value_at(fg_mask, x1, y1)):
                if not current:
                    _append_if_distinct(current, (x0, y0))
                _append_if_distinct(current, (x1, y1))
                continue

            sample_count = max(2, int(math.ceil(dist / HATCH_CLIP_SAMPLE_STEP_PX)) + 1)
            last_inside = False
            for idx in range(sample_count):
                t = idx / float(sample_count - 1)
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                inside = _mask_value_at(fg_mask, x, y)
                if inside:
                    if not last_inside and not current:
                        _append_if_distinct(current, (x, y))
                    _append_if_distinct(current, (x, y))
                elif last_inside:
                    if len(current) >= 2:
                        clipped.append(current)
                    current = []
                last_inside = inside

        if len(current) >= 2:
            clipped.append(current)
    return clipped


def _local_hatch_orientation_map(gray_work, fallback_angle_deg, cell_size):
    """Estimate a smooth contour-tangent direction map on the resized hatch image."""
    gray = np.asarray(gray_work, dtype=np.float32)
    if gray.size == 0 or gray.ndim != 2:
        return None

    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0, sigmaY=1.0)
    gx = cv2.Scharr(smoothed, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(smoothed, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    magnitude_scale = max(1e-6, float(np.percentile(magnitude, 92)))
    weight = np.clip(magnitude / magnitude_scale, 0.0, 1.0)

    # Gradient points across an edge; adding 90 degrees produces the local
    # contour tangent. Double-angle vectors make 0 and 180 degrees equivalent.
    tangent = np.arctan2(gy, gx) + np.pi * 0.5
    cos2 = np.cos(2.0 * tangent).astype(np.float32) * weight
    sin2 = np.sin(2.0 * tangent).astype(np.float32) * weight
    sigma = max(1.5, float(cell_size) * 0.55)
    cos2 = cv2.GaussianBlur(cos2, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sin2 = cv2.GaussianBlur(sin2, (0, 0), sigmaX=sigma, sigmaY=sigma)

    fallback = np.deg2rad(float(fallback_angle_deg) * 2.0)
    fallback_weight = 0.08
    cos2 += fallback_weight * np.cos(fallback)
    sin2 += fallback_weight * np.sin(fallback)
    orientation = np.mod(
        np.rad2deg(0.5 * np.arctan2(sin2, cos2)), 180.0)
    return orientation.astype(np.float32)


def _adaptive_hatch_threshold(tone_work, foreground_work, requested_threshold,
                              cell_size, allowed_work=None,
                              background_work=None):
    """Choose a shadow threshold from foreground luminance and estimate coverage."""
    tone = np.asarray(tone_work, dtype=np.uint8)
    foreground = np.asarray(foreground_work) > 0
    if allowed_work is not None:
        foreground &= np.asarray(allowed_work) > 0
    if background_work is not None:
        bg = np.asarray(background_work) > 0
        if bg.shape == foreground.shape:
            foreground &= ~bg
    foreground_count = int(np.count_nonzero(foreground))
    requested = float(np.clip(requested_threshold, 0.0, 255.0))
    if foreground_count == 0:
        return requested, np.zeros(tone.shape, dtype=bool), 0

    kernel_size = max(1, int(cell_size))
    local_mean = (cv2.boxFilter(
        tone, ddepth=-1, ksize=(kernel_size, kernel_size), normalize=True)
        if kernel_size > 1 else tone.astype(np.float32))
    local_foreground_values = local_mean[foreground]

    # The 28th percentile targets the middle of the desired 15-35% shadow
    # coverage range. The UI value remains a hard ceiling so bright skin or
    # clothing is not hatched merely because it is the darkest part of an image.
    percentile_threshold = float(np.percentile(local_foreground_values, 28.0))
    effective = min(requested, percentile_threshold)
    eligible = (foreground & (local_mean < effective) &
                (tone < effective + HATCH_DARK_EDGE_MARGIN))
    return effective, eligible, foreground_count


def _create_hatching_vectors(l_enh, fg_mask, cell_size=DEFAULT_HATCH_CELL_SIZE,
                             angle_deg=DEFAULT_HATCH_ANGLE_DEG,
                             min_spacing=DEFAULT_HATCH_MIN_SPACING,
                             max_spacing=DEFAULT_HATCH_MAX_SPACING,
                             dark_threshold=DEFAULT_HATCH_DARK_THRESHOLD,
                             max_dim=HATCH_MAX_DIM,
                             importance_map=None,
                             allowed_mask=None,
                             exclude_mask=None,
                             contour_following=True):
    h, w = l_enh.shape
    if h == 0 or w == 0:
        return []

    cell_size = max(1, int(round(cell_size)))
    min_spacing = max(1, int(round(min_spacing)))
    max_spacing = max(min_spacing, int(round(max_spacing)))
    dark_threshold = float(np.clip(dark_threshold, 0.0, 255.0))

    fg_mask_for_hatch = np.where(fg_mask > 0, 255, 0).astype(np.uint8)
    exclude = _resize_binary_mask(exclude_mask, w, h)
    if exclude is not None and cv2.countNonZero(exclude) > 0:
        before_px = int(cv2.countNonZero(fg_mask_for_hatch))
        fg_mask_for_hatch = cv2.bitwise_and(
            fg_mask_for_hatch, cv2.bitwise_not(exclude))
        after_px = int(cv2.countNonZero(fg_mask_for_hatch))
        print(
            f"  Face hatch exclusion: foreground {before_px} -> {after_px} px "
            f"(removed {before_px - after_px})"
        )

    tone = np.where(fg_mask_for_hatch > 0, l_enh, 255).astype(np.uint8)
    scale = min(1.0, float(max_dim) / float(max(h, w)))

    foreground = np.where(fg_mask_for_hatch > 0, 255, 0).astype(np.uint8)
    if scale < 1.0:
        small_w = max(1, int(round(w * scale)))
        small_h = max(1, int(round(h * scale)))
        tone_work = cv2.resize(tone, (small_w, small_h), interpolation=cv2.INTER_AREA)
        foreground_work = cv2.resize(
            foreground, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        background_work = cv2.resize(
            cv2.bitwise_not(foreground), (small_w, small_h),
            interpolation=cv2.INTER_NEAREST)
        spacing_scale = scale
    else:
        tone_work = tone
        foreground_work = foreground
        background_work = cv2.bitwise_not(foreground)
        spacing_scale = 1.0

    work_h, work_w = tone_work.shape
    scaled_cell_size = max(1, int(round(cell_size * spacing_scale)))
    importance_work = _resize_ai_map(importance_map, work_w, work_h)
    allowed_work = _resize_ai_map(allowed_mask, work_w, work_h, binary=True)
    effective_threshold, hatch_region, foreground_count = _adaptive_hatch_threshold(
        tone_work, foreground_work, dark_threshold, scaled_cell_size,
        allowed_work, background_work)
    hatch_pixel_count = int(np.count_nonzero(hatch_region))
    hatch_coverage_pct = (
        100.0 * hatch_pixel_count / foreground_count if foreground_count else 0.0)
    coverage_state = (
        "target" if 15.0 <= hatch_coverage_pct <= 35.0
        else "low" if hatch_coverage_pct < 15.0 else "high")
    print(
        f"  Hatch coverage: {hatch_coverage_pct:.1f}% foreground "
        f"({hatch_pixel_count}/{foreground_count} px, {coverage_state}; "
        f"threshold {dark_threshold:.0f} -> {effective_threshold:.1f}, target 15-35%)"
    )
    orientation_work = (
        _local_hatch_orientation_map(
            tone_work, angle_deg, scaled_cell_size)
        if contour_following else None
    )

    hatch_lines = generate_hatching(
        tone_work,
        cell_size=scaled_cell_size,
        angle_deg=float(angle_deg),
        min_spacing=max(1, int(round(min_spacing * spacing_scale))),
        max_spacing=max(1, int(round(max_spacing * spacing_scale))),
        invert=False,
        dark_threshold=effective_threshold,
        importance_map=importance_work,
        allowed_mask=allowed_work,
        orientation_map=orientation_work,
    )

    hatch_lines_original = []
    inv_scale = 1.0 / max(1e-6, float(scale))
    for polyline in hatch_lines:
        if len(polyline) < 2:
            continue
        restored = []
        for x, y in polyline:
            # generate_hatching() chay tren anh resize nho, nen scale nguoc ve toa do pixel goc.
            restored.append((
                float(np.clip(float(x) * inv_scale, 0.0, max(0.0, w - 1.0))),
                float(np.clip(float(y) * inv_scale, 0.0, max(0.0, h - 1.0))),
            ))
        hatch_lines_original.append(restored)

    return _clip_hatch_vectors_to_mask(hatch_lines_original, fg_mask_for_hatch)


def _simplify_hatch_polyline_px(polyline):
    """
    Nen hatch o toa do pixel truoc khi doi sang mm.
    Hatch chu yeu la cac doan thang/zigzag, nen bo diem gan trung va diem gan thang
    hang som giup giam segment ma khong lam mat tone chinh.
    """
    if len(polyline) <= 2:
        return list(polyline)

    pts = []
    for point in polyline:
        _append_if_distinct(pts, point, HATCH_VECTOR_MIN_POINT_DIST_PX)
    if len(pts) <= 2:
        return pts

    kept = [pts[0]]
    removed = 0
    for i in range(1, len(pts) - 1):
        a = np.asarray(kept[-1], dtype=np.float32)
        b = np.asarray(pts[i], dtype=np.float32)
        c = np.asarray(pts[i + 1], dtype=np.float32)
        ab = b - a
        bc = c - b
        ac = c - a
        n_ab = float(np.linalg.norm(ab))
        n_bc = float(np.linalg.norm(bc))
        n_ac = float(np.linalg.norm(ac))
        if n_ab <= 1e-6 or n_bc <= 1e-6:
            removed += 1
            continue
        cross = float(ac[0] * (b[1] - a[1]) - ac[1] * (b[0] - a[0]))
        perp = 0.0 if n_ac <= 1e-6 else abs(cross / n_ac)
        cos_a = float(np.dot(ab / n_ab, bc / n_bc))
        if perp <= HATCH_VECTOR_COLLINEAR_DIST_PX and cos_a > 0.996:
            removed += 1
            continue
        kept.append(tuple(pts[i]))

    kept.append(tuple(pts[-1]))
    return kept


def transform_to_lineart(
    pil_img, hatch_settings=None, return_hatch_vectors=False, ai_maps=None,
    exclude_mask=None, return_mask_info=False
):
    """
    Convert a black/white sketch into clean robot ink.

    The robot should draw centerlines, not re-fill thick pixels. Filled black
    regions are converted to outline plus sparse diagonal hatching.
    """
    rgb, fg_mask, mask_info = _composite_rgb_and_mask(
        pil_img, return_info=True)
    transform_to_lineart.last_foreground_mask_info = mask_info

    ink = _clean_line_art_binary_from_rgb(rgb, fg_mask)
    draw_mask, filled_count = _clean_line_art_draw_mask(ink)
    result = np.full_like(draw_mask, 255)
    result[draw_mask > 0] = 0
    pil_lineart = Image.fromarray(result)
    pil_lineart.info["vector_mode"] = "clean_lineart"
    pil_lineart.info["contains_symbolic_fills"] = bool(filled_count)
    pil_lineart.info["filled_region_count"] = int(filled_count)
    print(
        "Line-art pipeline: threshold -> centerline, "
        f"ink={int(cv2.countNonZero(ink))}, "
        f"draw={int(cv2.countNonZero(draw_mask))}, "
        f"filled_regions={filled_count}"
    )
    if return_hatch_vectors and return_mask_info:
        return pil_lineart, [], mask_info
    if return_hatch_vectors:
        return pil_lineart, []
    if return_mask_info:
        return pil_lineart, mask_info
    return pil_lineart


transform_to_lineart.last_foreground_mask_info = ForegroundMaskInfo()


# ═══════════════════════════════════════════════════════════════════
#  LINE ART -> VECTOR PATHS
# ═══════════════════════════════════════════════════════════════════

def _thin_binary(binary_255):
    """Skeletonize. Uses ximgproc.thinning if available, else morphology fallback."""
    src = np.where(binary_255 > 0, 255, 0).astype(np.uint8)
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(src)
    except Exception:
        pass

    img     = src.copy()
    skel    = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iter = max(img.shape) + 8
    for _ in range(max_iter):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp   = cv2.subtract(img, opened)
        skel   = cv2.bitwise_or(skel, temp)
        img    = cv2.erode(img, element)
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _reference_gradient_map(reference_img, width, height):
    if reference_img is None:
        return np.zeros((height, width), dtype=np.uint8)
    if reference_img.mode == "RGBA":
        rgb, _ = _composite_rgb_and_mask(reference_img)
    else:
        rgb = np.asarray(reference_img.convert("RGB"))
    gray   = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray   = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
    gray   = cv2.GaussianBlur(gray, (3, 3), 0)
    gx     = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy     = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    return _normalize_u8(cv2.magnitude(gx, gy))


def _fit_pixel_to_page(col, row, width, height):
    """Map pixel coordinate to mm on the 300x400 mm page, preserving aspect ratio."""
    span_w   = max(1, width  - 1)
    span_h   = max(1, height - 1)
    # Chua mot le nho ben trong kho giay de preview khong cat vien va may khong cham bien.
    usable_w = max(1.0, PAGE_MAX_X - 2.0 * PAGE_SAFE_MARGIN_MM)
    usable_h = max(1.0, PAGE_MAX_Y - 2.0 * PAGE_SAFE_MARGIN_MM)
    scale    = min(usable_w / span_w, usable_h / span_h)
    draw_w   = span_w * scale
    draw_h   = span_h * scale
    offset_x = (PAGE_MAX_X - draw_w) / 2.0
    offset_y = (PAGE_MAX_Y - draw_h) / 2.0
    x = offset_x + col * scale
    y = offset_y + (span_h - row) * scale
    return x, y


def _sample_polyline_ai_weight(polyline, ai_map, width, height):
    if ai_map is None or not polyline:
        return None
    arr = np.asarray(ai_map)
    if arr.size == 0 or arr.ndim != 2:
        return None
    map_h, map_w = arr.shape
    values = []
    for x, y in polyline:
        mx = int(round(float(x) * max(0, map_w - 1) / max(1, width - 1)))
        my = int(round(float(y) * max(0, map_h - 1) / max(1, height - 1)))
        values.append(float(arr[np.clip(my, 0, map_h - 1),
                                np.clip(mx, 0, map_w - 1)]))
    if not values:
        return None
    return float(np.clip(0.65 * np.mean(values) + 0.35 * np.percentile(values, 90),
                         0.0, 255.0)) / 255.0


def _hatch_vectors_to_candidates(hatch_lines, width, height, ai_hatch_map=None):
    """
    Doi hatch polyline pixel thanh CandidatePath mm de vao chung pipeline G-code.
    Hatch duoc gan tier medium de contour/base van duoc budget uu tien truoc.
    """
    candidates = []
    for polyline in hatch_lines:
        polyline = _simplify_hatch_polyline_px(polyline)
        if len(polyline) < 2:
            continue
        mm_pts = np.array(
            [_fit_pixel_to_page(float(x), float(y), width, height) for x, y in polyline],
            dtype=np.float32,
        )
        if len(mm_pts) < 2:
            continue
        diffs = np.diff(mm_pts, axis=0)
        length_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))
        if length_mm < HATCH_VECTOR_MIN_KEEP_MM:
            continue
        length_score = min(1.0, math.log1p(length_mm) / math.log1p(180.0))
        ai_weight = _sample_polyline_ai_weight(
            polyline, ai_hatch_map, width, height)
        ai_factor = 1.0 if ai_weight is None else 0.72 + 0.28 * ai_weight
        saliency = (0.24 + 0.34 * length_score) * ai_factor
        importance = (0.32 + 0.22 * length_score) * ai_factor
        detail_tier = HATCH_CANDIDATE_DETAIL_TIER if length_mm >= 7.0 else 2
        candidates.append(CandidatePath(
            points=mm_pts,
            importance=importance,
            length_mm=length_mm,
            detail_tier=detail_tier,
            saliency=saliency,
            region="hatch",
            source="hatch",
            protected=False,
            classifier_scores={
                "length_score": length_score,
                "ai_hatch_weight": ai_weight if ai_weight is not None else 0.0,
            },
        ))
    return candidates


def _trace_skeleton_paths(skeleton):
    """
    Convert skeleton to continuous trails on an 8-neighbor graph.

    Unlike naive "stop at every branch point" approaches, this function
    threads through junctions by choosing the most collinear neighbor,
    resulting in fewer, longer strokes and fewer pen lifts.
    """
    h, w       = skeleton.shape
    rows, cols = np.where(skeleton > 0)
    if rows.size == 0:
        return []

    line_ids = {int(r) * w + int(c) for r, c in zip(rows, cols)}
    offsets  = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    neighbor_cache = {}

    def neighbors(pid):
        cached = neighbor_cache.get(pid)
        if cached is not None:
            return cached
        r, c = divmod(pid, w)
        nbs  = []
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                nid = rr * w + cc
                if nid in line_ids:
                    nbs.append(nid)
        result = tuple(nbs)
        neighbor_cache[pid] = result
        return result

    degree     = {pid: len(neighbors(pid)) for pid in line_ids}
    used_edges = set()

    def edge_key(a, b):
        return (a, b) if a < b else (b, a)

    def has_unused(pid):
        return any(edge_key(pid, nb) not in used_edges for nb in neighbors(pid))

    def choose_next(prev, cur, candidates):
        if prev is None or len(candidates) == 1:
            return min(candidates, key=lambda n: degree.get(n, 0))
        pr, pc = divmod(prev, w)
        cr, cc = divmod(cur,  w)
        v1     = np.array([cr - pr, cc - pc], dtype=np.float32)
        norm1  = float(np.linalg.norm(v1)) or 1.0
        best       = candidates[0]
        best_score = -10.0
        for cand in candidates:
            nr, nc = divmod(cand, w)
            v2     = np.array([nr - cr, nc - cc], dtype=np.float32)
            norm2  = float(np.linalg.norm(v2)) or 1.0
            cos_a  = float(np.dot(v1, v2)) / (norm1 * norm2)
            score  = cos_a - 0.08 * degree.get(cand, 1)
            if score > best_score:
                best_score = score
                best = cand
        return best

    trails = []
    # Prioritize endpoints (deg=1), then junctions (deg>=3), then interior (deg=2)
    endpoints   = [p for p in line_ids if degree.get(p, 0) == 1]
    junctions   = [p for p in line_ids if degree.get(p, 0) >= 3]
    others      = [p for p in line_ids if degree.get(p, 0) == 2]
    start_order = endpoints + junctions + others

    for start in start_order:
        if not has_unused(start):
            continue
        trail    = [divmod(start, w)]
        prev_pid = None
        cur_pid  = start
        while True:
            nbs        = neighbors(cur_pid)
            candidates = [nb for nb in nbs if edge_key(cur_pid, nb) not in used_edges]
            if not candidates:
                break
            nxt_pid = choose_next(prev_pid, cur_pid, candidates)
            used_edges.add(edge_key(cur_pid, nxt_pid))
            prev_pid = cur_pid
            cur_pid  = nxt_pid
            trail.append(divmod(cur_pid, w))
        if len(trail) >= 2:
            trails.append(trail)

    return trails


def _pixel_trail_length_px(trail):
    if len(trail) < 2:
        return 0.0
    pts = np.asarray([(float(c), float(r)) for r, c in trail], dtype=np.float32)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _filter_short_trace_artifacts(pixel_trails, min_length_px=LINE_ART_TRACE_MIN_KEEP_PX):
    """
    Drop tiny leftover graph trails from thinned clean line-art.

    Thinning curved, anti-aliased black strokes often creates local 2x2 stair
    steps. In an 8-neighbor graph those become fake junctions, so edge traversal
    leaves many 2-12 px fragments after the real long stroke has already been
    traced. These are not separate plotter strokes.
    """
    trails = [trail for trail in pixel_trails if len(trail) >= 2]
    if len(trails) < 24:
        return trails

    lengths = [_pixel_trail_length_px(trail) for trail in trails]
    long_exists = any(length > float(min_length_px) * 2.0 for length in lengths)
    if not long_exists:
        return trails

    kept = [
        trail for trail, length in zip(trails, lengths)
        if length >= float(min_length_px)
    ]
    if not kept:
        return trails

    removed = len(trails) - len(kept)
    if removed:
        print(
            f"  Trace artifact filter: {len(trails)} -> {len(kept)} trails "
            f"(removed {removed} fragments < {float(min_length_px):.1f}px)"
        )
    return kept


def _polyline_turn_saliency(points):
    """Tinh do dac trung cua stroke dua tren goc/curvature de giu lai net co hinh dang ro."""
    if len(points) < 3:
        return 0.0
    total_turn = 0.0
    sharp_count = 0
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
        turn = math.degrees(math.acos(cos_a))
        total_turn += min(90.0, turn) / 90.0
        if turn >= 28.0:
            sharp_count += 1
    density = total_turn / max(1, len(points) - 2)
    return float(np.clip(density * 0.75 + min(1.0, sharp_count / 5.0) * 0.25, 0.0, 1.0))


def _classify_detail_tier(length_mm, grad_v, turn_v, importance):
    """
    Chia stroke thanh 3 lop chi tiet.
    Tier 0 luon uu tien giu, tier 2 la texture/fine detail se bi cat truoc khi budget thap.
    """
    length_score = min(1.0, math.log1p(length_mm) / math.log1p(240.0))
    saliency = float(np.clip(
        importance * 0.38 + grad_v * 0.26 + turn_v * 0.22 + length_score * 0.14,
        0.0, 1.0,
    ))
    if saliency >= 0.62 or length_mm >= 45.0 or grad_v >= 0.72:
        tier = 0
    elif saliency >= 0.34 or length_mm >= 10.0:
        tier = 1
    else:
        tier = 2
    return tier, saliency


def _candidate_mask_overlap(candidate, mask, canvas_size=None, samples_per_segment=3):
    if mask is None:
        return 0.0
    arr = np.asarray(mask)
    if arr.size == 0 or arr.ndim != 2 or len(candidate.points) < 2:
        return 0.0
    height, width = arr.shape
    if canvas_size is not None:
        width, height = int(canvas_size[0]), int(canvas_size[1])
        arr = _resize_binary_mask(arr, width, height)
        if arr is None:
            return 0.0

    hits = 0
    total = 0
    for p0, p1 in zip(candidate.points[:-1], candidate.points[1:]):
        steps = max(1, int(samples_per_segment))
        for step in range(steps + 1):
            t = step / float(steps)
            p = p0 * (1.0 - t) + p1 * t
            x, y = _fit_page_to_pixel(p[0], p[1], width, height)
            total += 1
            if arr[y, x] > 0:
                hits += 1
    return hits / max(1, total)


def _candidate_bbox_mask_overlap(candidate, mask, canvas_size=None, pad_px=2):
    if mask is None or len(candidate.points) < 2:
        return 0.0
    arr = np.asarray(mask)
    if arr.size == 0 or arr.ndim != 2:
        return 0.0
    height, width = arr.shape
    if canvas_size is not None:
        width, height = int(canvas_size[0]), int(canvas_size[1])
        arr = _resize_binary_mask(arr, width, height)
        if arr is None:
            return 0.0
    cols = []
    rows = []
    for point in candidate.points:
        x, y = _fit_page_to_pixel(point[0], point[1], width, height)
        cols.append(x)
        rows.append(y)
    if not cols or not rows:
        return 0.0
    x0 = max(0, min(cols) - int(pad_px))
    x1 = min(width, max(cols) + int(pad_px) + 1)
    y0 = max(0, min(rows) - int(pad_px))
    y1 = min(height, max(rows) + int(pad_px) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    crop = arr[y0:y1, x0:x1]
    return cv2.countNonZero(crop) / max(1, crop.size)


def _sample_candidate_map(candidate, map_img, canvas_size=None, default=0.0):
    if map_img is None:
        return float(default)
    arr = np.asarray(map_img)
    if arr.size == 0 or arr.ndim != 2:
        return float(default)
    height, width = arr.shape
    if canvas_size is not None:
        width, height = int(canvas_size[0]), int(canvas_size[1])
        if arr.shape != (height, width):
            arr = cv2.resize(arr.astype(np.float32), (width, height),
                             interpolation=cv2.INTER_NEAREST)
    values = []
    for point in candidate.points:
        x, y = _fit_page_to_pixel(point[0], point[1], width, height)
        values.append(float(arr[y, x]))
    if not values:
        return float(default)
    return float(np.mean(values))


def _build_contour_hierarchy_maps(lineart_img):
    if lineart_img is None:
        return None
    gray = np.asarray(lineart_img.convert("L"))
    height, width = gray.shape
    binary = np.where(gray < 210, 255, 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return {
            "depth": np.full((height, width), 3, dtype=np.uint8),
            "outline": np.zeros((height, width), dtype=np.uint8),
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grouped = cv2.dilate(binary, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(
        grouped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    depth_map = np.full((height, width), 3, dtype=np.uint8)
    outline_map = np.zeros((height, width), dtype=np.uint8)
    if hierarchy is None or not contours:
        return {"depth": depth_map, "outline": outline_map}

    hierarchy = hierarchy[0]
    depths = []
    for idx in range(len(contours)):
        depth = 0
        parent = int(hierarchy[idx][3])
        guard = 0
        while parent >= 0 and guard < len(contours):
            depth += 1
            parent = int(hierarchy[parent][3])
            guard += 1
        depths.append(depth)

    canvas_area = float(width * height)
    order = sorted(range(len(contours)),
                   key=lambda i: cv2.contourArea(contours[i]), reverse=True)
    for idx in order:
        contour = contours[idx]
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter < 4.0:
            continue
        depth = min(3, int(depths[idx]))
        area_score = min(1.0, math.sqrt(max(0.0, area) / max(1.0, canvas_area)) * 7.0)
        length_score = min(1.0, perimeter / max(1.0, 0.45 * (width + height)))
        outline_score = int(round(255.0 * np.clip(
            0.56 * (1.0 - min(depth, 3) / 3.0) +
            0.27 * area_score +
            0.17 * length_score,
            0.0, 1.0)))
        cv2.drawContours(depth_map, contours, idx, int(depth),
                         thickness=cv2.FILLED)
        cv2.drawContours(outline_map, contours, idx, outline_score,
                         thickness=3, lineType=cv2.LINE_AA)
    return {"depth": depth_map, "outline": outline_map}


class StrokeClassifier:
    name = "base"
    weight = 1.0
    enabled = True

    def classify(self, candidates, context):
        raise NotImplementedError


class GeometricStrokeClassifier(StrokeClassifier):
    name = "geometry"
    weight = 1.0

    def classify(self, candidates, context):
        tiers = []
        for candidate in candidates:
            metrics = getattr(candidate, "classifier_scores", {}) or {}
            grad_v = float(metrics.get("grad", 0.0))
            turn_v = float(metrics.get(
                "turn", _polyline_turn_saliency(candidate.points)))
            base_importance = float(metrics.get(
                "base_importance", candidate.importance))
            tier, saliency = _classify_detail_tier(
                candidate.length_mm, grad_v, turn_v, base_importance)
            tiers.append(StrokeTier(
                detail_tier=tier,
                importance_score=base_importance,
                saliency_score=saliency,
                region=getattr(candidate, "region", "detail"),
                protected=False,
                notes=f"grad={grad_v:.2f},turn={turn_v:.2f}",
            ))
        return tiers


class ContourHierarchyStrokeClassifier(StrokeClassifier):
    name = "contour_hierarchy"
    weight = 0.88

    def classify(self, candidates, context):
        cache = context.cache if context is not None else {}
        maps = cache.get("contour_hierarchy")
        if maps is None:
            maps = _build_contour_hierarchy_maps(
                context.lineart_img if context is not None else None)
            cache["contour_hierarchy"] = maps
        tiers = []
        canvas_size = context.canvas_size if context is not None else None
        for candidate in candidates:
            if getattr(candidate, "source", "") == "hatch":
                tiers.append(StrokeTier(
                    detail_tier=2,
                    importance_score=min(0.62, candidate.importance),
                    saliency_score=min(0.58, candidate.saliency or candidate.importance),
                    region="hatch",
                    protected=False,
                    notes="hatch",
                ))
                continue
            if maps is None:
                tiers.append(StrokeTier(
                    detail_tier=candidate.detail_tier,
                    importance_score=candidate.importance,
                    saliency_score=candidate.saliency,
                    region=getattr(candidate, "region", "detail"),
                ))
                continue
            outline = _sample_candidate_map(
                candidate, maps["outline"], canvas_size=canvas_size) / 255.0
            depth = _sample_candidate_map(
                candidate, maps["depth"], canvas_size=canvas_size, default=2.0)
            length_score = min(1.0, math.log1p(candidate.length_mm) / math.log1p(280.0))
            turn_score = _polyline_turn_saliency(candidate.points)
            score = float(np.clip(
                outline * 0.54 + length_score * 0.30 + turn_score * 0.16,
                0.0, 1.0))
            if depth <= 0.55 and (candidate.length_mm >= 7.0 or outline >= 0.40):
                tier = 0
                region = "garment_outline"
            elif depth <= 1.60 or candidate.length_mm >= 18.0 or score >= 0.52:
                tier = 1
                region = "garment_detail"
            else:
                tier = 2
                region = "fine_detail"
            tiers.append(StrokeTier(
                detail_tier=tier,
                importance_score=score,
                saliency_score=max(score, candidate.saliency),
                region=region,
                protected=False,
                notes=f"depth={depth:.1f},outline={outline:.2f}",
            ))
        return tiers


class FaceRegionStrokeClassifier(StrokeClassifier):
    name = "face_region"
    weight = 1.15

    def classify(self, candidates, context):
        face_mask = context.face_mask if context is not None else None
        canvas_size = context.canvas_size if context is not None else None
        has_face = (
            face_mask is not None and np.asarray(face_mask).ndim == 2 and
            cv2.countNonZero(np.asarray(face_mask, dtype=np.uint8)) > 0
        )
        tiers = []
        for candidate in candidates:
            if not has_face or getattr(candidate, "source", "") == "hatch":
                tiers.append(StrokeTier(
                    detail_tier=candidate.detail_tier,
                    importance_score=candidate.importance,
                    saliency_score=candidate.saliency,
                    region=getattr(candidate, "region", "detail"),
                    protected=False,
                    notes="no_face" if not has_face else "hatch",
                ))
                continue
            overlap = _candidate_mask_overlap(
                candidate, face_mask, canvas_size=canvas_size)
            if overlap >= 0.18:
                length_score = min(1.0, math.log1p(candidate.length_mm) / math.log1p(180.0))
                turn_score = _polyline_turn_saliency(candidate.points)
                saliency = float(np.clip(
                    0.62 + 0.20 * length_score + 0.18 * turn_score,
                    0.0, 0.96))
                tiers.append(StrokeTier(
                    detail_tier=0,
                    importance_score=max(0.86, saliency),
                    saliency_score=saliency,
                    region="face",
                    protected=True,
                    notes=f"face_overlap={overlap:.2f}",
                ))
            else:
                tiers.append(StrokeTier(
                    detail_tier=candidate.detail_tier,
                    importance_score=candidate.importance,
                    saliency_score=candidate.saliency,
                    region=getattr(candidate, "region", "detail"),
                    protected=False,
                    notes=f"face_overlap={overlap:.2f}",
                ))
        return tiers


STROKE_CLASSIFIER_REGISTRY = []


def register_stroke_classifier(classifier):
    STROKE_CLASSIFIER_REGISTRY.append(classifier)
    return classifier


def _stroke_region_rank(region):
    ranks = {
        "face": 5,
        "garment_outline": 4,
        "garment_detail": 3,
        "detail": 2,
        "fine_detail": 1,
        "hatch": 0,
    }
    return ranks.get(region, 2)


def classify_strokes(candidates, context=None, classifiers=None):
    pool = list(candidates or [])
    if not pool:
        return []
    context = context or StrokeClassificationContext()
    classifiers = classifiers or STROKE_CLASSIFIER_REGISTRY
    active = [classifier for classifier in classifiers
              if getattr(classifier, "enabled", True)]
    if not active:
        return pool

    classified_sets = []
    for classifier in active:
        try:
            result = classifier.classify(pool, context)
            if result is not None and len(result) == len(pool):
                classified_sets.append((classifier, result))
        except Exception as error:
            print(
                f"Stroke classifier warning: {classifier.name} failed: "
                f"{type(error).__name__}: {error}"
            )

    output = []
    for index, candidate in enumerate(pool):
        total_weight = 0.35
        importance_sum = float(candidate.importance) * 0.35
        saliency_sum = float(candidate.saliency or candidate.importance) * 0.35
        best_tier = int(candidate.detail_tier)
        best_region = getattr(candidate, "region", "detail")
        protected = bool(getattr(candidate, "protected", False))
        score_notes = dict(getattr(candidate, "classifier_scores", {}) or {})

        for classifier, tiers in classified_sets:
            tier = tiers[index]
            weight = float(getattr(classifier, "weight", 1.0))
            total_weight += weight
            importance_sum += float(tier.importance_score) * weight
            saliency_sum += float(tier.saliency_score) * weight
            if tier.detail_tier < best_tier:
                best_tier = int(tier.detail_tier)
            if _stroke_region_rank(tier.region) > _stroke_region_rank(best_region):
                best_region = tier.region
            protected = protected or bool(tier.protected)
            score_notes[classifier.name] = {
                "tier": int(tier.detail_tier),
                "importance": float(tier.importance_score),
                "saliency": float(tier.saliency_score),
                "region": tier.region,
                "protected": bool(tier.protected),
                "notes": tier.notes,
            }

        importance = float(np.clip(importance_sum / max(1e-6, total_weight), 0.0, 1.0))
        saliency = float(np.clip(saliency_sum / max(1e-6, total_weight), 0.0, 1.0))
        if protected:
            best_tier = min(best_tier, 0)
            importance = max(importance, 0.88)
            saliency = max(saliency, 0.82)
        output.append(CandidatePath(
            points=np.asarray(candidate.points, dtype=np.float32).copy(),
            importance=importance,
            length_mm=float(candidate.length_mm),
            detail_tier=int(best_tier),
            saliency=saliency,
            region=best_region,
            source=getattr(candidate, "source", "lineart"),
            protected=protected,
            classifier_scores=score_notes,
        ))

    counts = {}
    for candidate in output:
        counts[candidate.region] = counts.get(candidate.region, 0) + 1
    print(
        "Stroke classification: " +
        ", ".join(f"{key}={counts[key]}" for key in sorted(counts))
    )
    return output


register_stroke_classifier(GeometricStrokeClassifier())
register_stroke_classifier(ContourHierarchyStrokeClassifier())
register_stroke_classifier(FaceRegionStrokeClassifier())


def extract_candidate_paths(pil_lineart, reference_img=None, ai_maps=None,
                            face_mask=None):
    """
    Vectorize line-art image into a list of CandidatePath (mm coordinates).
    """
    gray  = np.array(pil_lineart.convert("L"))
    h0, w0 = gray.shape
    explicit_clean_lineart = (
        getattr(pil_lineart, "info", {}).get("vector_mode") == "clean_lineart"
    )
    has_symbolic_fills = bool(
        getattr(pil_lineart, "info", {}).get("contains_symbolic_fills")
    )

    scale_factor = min(1.0, VECTOR_MAX_DIM / max(h0, w0))
    if scale_factor < 1.0:
        new_w = max(1, int(w0 * scale_factor))
        new_h = max(1, int(h0 * scale_factor))
        gray  = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h, w = gray.shape
    clean_lineart = explicit_clean_lineart or _looks_like_clean_line_art_gray(gray)

    binary  = np.where(gray < 180, 255, 0).astype(np.uint8)
    if not explicit_clean_lineart:
        close_k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k, iterations=1)

    skeleton      = _thin_binary(binary)
    ref_grad      = _reference_gradient_map(reference_img, w, h)
    ai_saliency = _resize_ai_map(
        ai_maps.saliency if ai_maps is not None else None, w, h)
    ai_subject = _resize_ai_map(
        ai_maps.foreground if ai_maps is not None else None, w, h, binary=True)
    pixel_trails  = _trace_skeleton_paths(skeleton)
    if clean_lineart:
        min_keep_px = 5.0 if has_symbolic_fills else LINE_ART_TRACE_MIN_KEEP_PX
        pixel_trails = _filter_short_trace_artifacts(
            pixel_trails, min_length_px=min_keep_px)

    candidates = []
    for trail in pixel_trails:
        mm_pts = np.array(
            [_fit_pixel_to_page(c, r, w, h) for r, c in trail],
            dtype=np.float32,
        )
        if len(mm_pts) < 2:
            continue
        diffs     = np.diff(mm_pts, axis=0)
        length_mm = float(np.sum(np.linalg.norm(diffs, axis=1)))
        r_arr     = np.array([int(r) for r, c in trail], dtype=np.int32).clip(0, h - 1)
        c_arr     = np.array([int(c) for r, c in trail], dtype=np.int32).clip(0, w - 1)
        grad_v    = float(np.mean(ref_grad[r_arr, c_arr])) / 255.0
        importance = min(
            1.0,
            math.log1p(length_mm) / math.log1p(300.0) * 0.75 + grad_v * 0.25
        )
        ai_score = None
        if ai_saliency is not None and ai_subject is not None:
            saliency_samples = ai_saliency[r_arr, c_arr].astype(np.float32) / 255.0
            subject_samples = ai_subject[r_arr, c_arr].astype(np.float32) / 255.0
            ai_score = float(np.clip(
                0.55 * np.percentile(saliency_samples, 90) +
                0.25 * np.mean(saliency_samples) +
                0.20 * np.mean(subject_samples),
                0.0, 1.0,
            ))
            ai_factor = 0.82 + 0.28 * ai_score
            if length_mm >= 25.0 or grad_v >= 0.55:
                ai_factor = max(0.98, ai_factor)
            importance = float(np.clip(importance * ai_factor, 0.0, 1.0))
        turn_v = _polyline_turn_saliency(mm_pts)
        detail_tier, saliency = _classify_detail_tier(
            length_mm, grad_v, turn_v, importance)
        if ai_score is not None:
            ai_factor = 0.82 + 0.28 * ai_score
            if detail_tier == 0:
                ai_factor = max(0.98, ai_factor)
            saliency = float(np.clip(saliency * ai_factor, 0.0, 1.0))
        candidates.append(CandidatePath(
            points=mm_pts,
            importance=importance,
            length_mm=length_mm,
            detail_tier=detail_tier,
            saliency=saliency,
            region="detail",
            source="lineart",
            protected=False,
            classifier_scores={
                "grad": grad_v,
                "turn": turn_v,
                "base_importance": importance,
                "initial_tier": detail_tier,
                "initial_saliency": saliency,
            },
        ))

    context = StrokeClassificationContext(
        canvas_size=pil_lineart.size,
        reference_img=reference_img,
        lineart_img=pil_lineart,
        ai_maps=ai_maps,
        face_mask=face_mask,
    )
    candidates = classify_strokes(candidates, context)
    print(f"extract_candidate_paths: {len(candidates)} paths from {len(pixel_trails)} trails")
    return candidates


def _consolidate_dense_micro_candidates(
        candidates, grid_mm=MICRO_TEXTURE_GRID_MM,
        dense_count=MICRO_TEXTURE_DENSE_COUNT,
        keep_ratio=MICRO_TEXTURE_KEEP_RATIO):
    """Downsample dense short texture while preserving strong strokes and directions."""
    pool = list(candidates)
    if len(pool) < dense_count:
        return pool

    groups = {}
    for index, candidate in enumerate(pool):
        is_micro_texture = candidate.length_mm <= 6.0 and (
            candidate.detail_tier == 2 or candidate.length_mm <= 4.0)
        if not is_micro_texture or len(candidate.points) < 2:
            continue
        key = _grid_key(_candidate_center(candidate), grid_mm)
        groups.setdefault(key, []).append(index)

    keep = np.ones(len(pool), dtype=bool)
    dense_cells = 0
    for indexes in groups.values():
        if len(indexes) < dense_count:
            continue
        dense_cells += 1
        keep_count = max(8, int(math.ceil(len(indexes) * keep_ratio)))

        def candidate_score(index):
            candidate = pool[index]
            visual = candidate.saliency if candidate.saliency > 0 else candidate.importance
            return visual * 3.0 + math.log1p(candidate.length_mm)

        ranked = sorted(indexes, key=candidate_score, reverse=True)
        direction_bins = {}
        for index in ranked:
            delta = pool[index].points[-1] - pool[index].points[0]
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0
            direction_bins.setdefault(int(angle // 30.0), []).append(index)

        retained = []
        # Preserve at least one representative of every local stroke direction.
        for bin_indexes in direction_bins.values():
            if len(retained) < keep_count:
                retained.append(bin_indexes[0])
        retained_set = set(retained)
        for index in ranked:
            if len(retained) >= keep_count:
                break
            if index not in retained_set:
                retained.append(index)
                retained_set.add(index)

        for index in indexes:
            if index not in retained_set:
                keep[index] = False

    consolidated = [candidate for index, candidate in enumerate(pool) if keep[index]]
    removed = len(pool) - len(consolidated)
    if removed:
        print(
            f"  Dense micro-texture: {len(pool)} -> {len(consolidated)} candidates "
            f"(removed {removed} from {dense_cells} dense cells)"
        )
    return consolidated


# ═══════════════════════════════════════════════════════════════════
#  G-CODE PLANNING UTILITIES
# ═══════════════════════════════════════════════════════════════════

def _clip_segment_to_rect(p0, p1, xmin, ymin, xmax, ymax):
    """Cat mot segment vao khung giay bang thuat toan Liang-Barsky."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    u1 = 0.0
    u2 = 1.0
    for p, q in [(-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > u2:
                return None
            u1 = max(u1, r)
        else:
            if r < u1:
                return None
            u2 = min(u2, r)
    a = np.array([x0 + u1 * dx, y0 + u1 * dy], dtype=np.float32)
    b = np.array([x0 + u2 * dx, y0 + u2 * dy], dtype=np.float32)
    return a, b


def _clip_paths_to_page(paths, margin_mm=PAGE_SAFE_MARGIN_MM):
    """
    Cat path vao vung giay an toan thay vi chi bao loi khi vuot bien.
    Neu mot polyline cat qua bien, ham se tach thanh cac stroke con nam trong khung.
    """
    xmin = float(margin_mm)
    ymin = float(margin_mm)
    xmax = float(PAGE_MAX_X - margin_mm)
    ymax = float(PAGE_MAX_Y - margin_mm)
    clipped = []
    split_count = 0
    for path in paths:
        if len(path) < 2:
            continue
        current = []
        for p0, p1 in zip(path[:-1], path[1:]):
            seg = _clip_segment_to_rect(p0, p1, xmin, ymin, xmax, ymax)
            if seg is None:
                if len(current) >= 2:
                    clipped.append(np.asarray(current, dtype=np.float32))
                    split_count += 1
                current = []
                continue
            a, b = seg
            if not current:
                current = [a, b]
            elif float(np.linalg.norm(current[-1] - a)) > CLIP_EPS_MM:
                if len(current) >= 2:
                    clipped.append(np.asarray(current, dtype=np.float32))
                    split_count += 1
                current = [a, b]
            else:
                current.append(b)
        if len(current) >= 2:
            clipped.append(np.asarray(current, dtype=np.float32))
    if split_count:
        print(f"  Page clipping: split {split_count} clipped strokes")
    return clipped, split_count


def _candidate_center(candidate):
    return candidate.points.mean(axis=0) if len(candidate.points) else np.zeros(2, dtype=np.float32)


def _candidate_with_points(candidate, points, source=None):
    pts = np.asarray(points, dtype=np.float32)
    length_mm = _path_length_mm(pts) if len(pts) >= 2 else 0.0
    return CandidatePath(
        points=pts,
        importance=float(candidate.importance),
        length_mm=length_mm,
        detail_tier=int(candidate.detail_tier),
        saliency=float(candidate.saliency),
        region=getattr(candidate, "region", "detail"),
        source=source or getattr(candidate, "source", "lineart"),
        protected=bool(getattr(candidate, "protected", False)),
        classifier_scores=dict(getattr(candidate, "classifier_scores", {}) or {}),
    )


def _grid_key(point, cell_mm):
    """Dua toa do mm vao o luoi de truy van lan can nhanh."""
    return (int(float(point[0]) // cell_mm), int(float(point[1]) // cell_mm))


def _neighbor_grid_keys(key, radius=1):
    kx, ky = key
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            yield (kx + dx, ky + dy)


def _deduplicate_candidates(candidates, endpoint_tol=DEDUP_ENDPOINT_TOL_MM,
                            length_tol_ratio=DEDUP_LENGTH_TOL_RATIO):
    """
    Loai bo stroke gan nhu trung nhau do threshold/contour tao net doi.
    So sanh dau-cuoi theo ca hai huong va chenh lech do dai de giu lai net quan trong hon.
    """
    kept = []
    removed = 0
    tol = float(endpoint_tol)
    cell_mm = max(tol * 2.0, 1e-6)
    grid = {}
    for cand in sorted(candidates, key=lambda c: (c.importance, c.length_mm), reverse=True):
        duplicate = False
        center = _candidate_center(cand)
        nearby_indexes = set()
        for key in _neighbor_grid_keys(_grid_key(center, cell_mm), radius=1):
            nearby_indexes.update(grid.get(key, ()))
        for other_idx in nearby_indexes:
            other = kept[other_idx]
            len_ref = max(cand.length_mm, other.length_mm, 1e-6)
            if abs(cand.length_mm - other.length_mm) / len_ref > length_tol_ratio:
                continue
            same = (float(np.linalg.norm(cand.points[0] - other.points[0])) <= tol and
                    float(np.linalg.norm(cand.points[-1] - other.points[-1])) <= tol)
            flipped = (float(np.linalg.norm(cand.points[0] - other.points[-1])) <= tol and
                       float(np.linalg.norm(cand.points[-1] - other.points[0])) <= tol)
            center_close = float(np.linalg.norm(_candidate_center(cand) - _candidate_center(other))) <= tol
            if center_close and (same or flipped):
                duplicate = True
                break
        if duplicate:
            removed += 1
        else:
            kept_idx = len(kept)
            kept.append(cand)
            grid.setdefault(_grid_key(center, cell_mm), []).append(kept_idx)
    if removed:
        print(f"  Path deduplication: removed {removed} duplicated strokes")
    return kept, removed


def _path_curvature_scores(points):
    """Tinh diem cong tai tung diem; goc cang gap thi diem cang cao."""
    scores = np.zeros(len(points), dtype=np.float32)
    if len(points) < 3:
        return scores
    for i in range(1, len(points) - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
        turn = math.degrees(math.acos(cos_a))
        scores[i] = min(1.0, turn / 120.0)
    return scores


def _rdp_simplify(points, epsilon):
    """Iterative Ramer-Douglas-Peucker simplification."""
    if len(points) <= 2:
        return points
    stack  = [(0, len(points) - 1)]
    keep   = set([0, len(points) - 1])
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue
        seg     = points[end] - points[start]
        seg_len = float(np.linalg.norm(seg)) or 1e-9
        seg_u   = seg / seg_len
        vecs    = points[start + 1:end] - points[start]
        dists   = np.abs(vecs[:, 0] * seg_u[1] - vecs[:, 1] * seg_u[0])
        max_i   = int(np.argmax(dists)) + start + 1
        if dists[max_i - start - 1] > epsilon:
            keep.add(max_i)
            stack.append((start, max_i))
            stack.append((max_i, end))
    return points[sorted(keep)]


def _adaptive_rdp_simplify(points, epsilon, curvature=None):
    """
    Adaptive RDP: doan thang dung epsilon lon, vung cong/goc dung epsilon nho hon.
    Cach nay giu lai chi tiet quan trong ma van giam manh diem tren doan it cong.
    """
    if len(points) <= 2:
        return points
    # Standard RDP collapses a closed contour because both endpoints coincide.
    # Split the loop into two open arcs, simplify them, then close it again.
    if (len(points) >= 4 and
            float(np.linalg.norm(points[0] - points[-1])) <= 1e-5):
        loop = points[:-1]
        unique = np.unique(np.round(loop, decimals=6), axis=0)
        if len(unique) < 3:
            return points[[0, -1]]
        split = int(np.argmax(np.linalg.norm(loop - loop[0], axis=1)))
        if 0 < split < len(loop) - 1:
            first_arc = _adaptive_rdp_simplify(
                loop[:split + 1], epsilon, curvature=None)
            second_arc = _adaptive_rdp_simplify(
                np.vstack([loop[split:], loop[0]]), epsilon, curvature=None)
            return np.vstack([first_arc, second_arc[1:]]).astype(np.float32)
    if curvature is None:
        curvature = _path_curvature_scores(points)
    stack = [(0, len(points) - 1)]
    keep = {0, len(points) - 1}
    while stack:
        start, end = stack.pop()
        if end - start <= 1:
            continue
        seg = points[end] - points[start]
        seg_len = float(np.linalg.norm(seg)) or 1e-9
        seg_u = seg / seg_len
        vecs = points[start + 1:end] - points[start]
        dists = np.abs(vecs[:, 0] * seg_u[1] - vecs[:, 1] * seg_u[0])
        local_curve = curvature[start + 1:end]
        effective_eps = epsilon * (1.0 - ADAPTIVE_RDP_CURVE_GAIN * local_curve)
        effective_eps = np.maximum(effective_eps, epsilon * 0.22)
        ratios = dists / effective_eps
        max_rel = int(np.argmax(ratios))
        max_i = max_rel + start + 1
        if ratios[max_rel] > 1.0:
            keep.add(max_i)
            stack.append((start, max_i))
            stack.append((max_i, end))
    return points[sorted(keep)]


def _candidate_rdp_epsilon(candidate, epsilon):
    region = getattr(candidate, "region", "detail")
    if region == "face" or getattr(candidate, "protected", False):
        return min(float(epsilon), FACE_RDP_EPSILON_MM)
    return float(epsilon)


def _simplify_all(candidates, epsilon, curvatures=None):
    if curvatures is None:
        curvatures = [None] * len(candidates)
    return [
        _adaptive_rdp_simplify(c.points, _candidate_rdp_epsilon(c, epsilon), curvature)
        for c, curvature in zip(candidates, curvatures)
        if len(c.points) >= 2
    ]


def _count_segments(paths):
    return sum(max(0, len(p) - 1) for p in paths)


def _candidate_min_segment_cost(candidate):
    if len(candidate.points) < 2:
        return 0
    closed = float(np.linalg.norm(candidate.points[0] - candidate.points[-1])) < 0.35
    return 2 if closed else 1


def _is_auto_prune_protected(candidate):
    region = getattr(candidate, "region", "")
    return bool(getattr(candidate, "protected", False)) or region == "face" or (
        region == "garment_outline" and int(getattr(candidate, "detail_tier", 1)) == 0
    )


def _select_candidates_for_budget_legacy(candidates, target):
    """
    Travel-aware detail budgeting: chon stroke theo ty le gia tri/chi phi di chuyen.
    Nhờ vậy budget thấp vẫn ưu tiên nét quan trọng gần nhau, giảm G0 travel vô ích.
    """
    pool = list(candidates)
    selected = []
    cost = 0
    cur = np.array([0.0, 0.0], dtype=np.float32)
    while pool and cost < target:
        best_i = None
        best_score = -1e18
        for i, c in enumerate(pool):
            min_cost = _candidate_min_segment_cost(c)
            if min_cost <= 0 or cost + min_cost > target:
                continue
            d_start = float(np.linalg.norm(c.points[0] - cur))
            d_end = float(np.linalg.norm(c.points[-1] - cur))
            travel_cost = min(d_start, d_end)
            value = c.importance * 3.0 + math.log1p(c.length_mm) / math.log1p(300.0)
            score = value / (1.0 + travel_cost * 0.035 + min_cost * 0.002)
            if score > best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        chosen = pool.pop(best_i)
        if float(np.linalg.norm(chosen.points[-1] - cur)) < float(np.linalg.norm(chosen.points[0] - cur)):
            chosen = _candidate_with_points(chosen, chosen.points[::-1].copy())
        selected.append(chosen)
        cost += _candidate_min_segment_cost(chosen)
        cur = chosen.points[-1]
    return selected


def _select_candidates_for_budget(candidates, target):
    """
    Travel-aware detail budgeting bang spatial grid.
    Multi-resolution budget: giu base contour truoc, roi medium, fine texture sau.
    Khi budget thap, texture phu bi cat truoc nen hinh tong the van giu chi tiet chinh.
    """
    pool = list(candidates)
    if not pool:
        return []

    values = []
    for c in pool:
        length_value = math.log1p(c.length_mm) / math.log1p(300.0)
        # Within fine-detail/hatch budget, favor one continuous long stroke over
        # many short fragments with similar saliency and aggregate length.
        continuity_bonus = 0.55 * length_value if c.detail_tier == 2 else 0.0
        region = getattr(c, "region", "detail")
        source = getattr(c, "source", "")
        region_bonus = (
            0.72 if region == "face"
            else 0.46 if region == "garment_outline"
            else 0.18 if region == "garment_detail"
            else -0.10 if region == "hatch"
            else 0.0
        )
        if source == "qa_overlay_add":
            region_bonus += 0.90
        protected_bonus = 0.40 if getattr(c, "protected", False) else 0.0
        values.append(
            (c.saliency if c.saliency > 0 else c.importance) * 3.2 +
            length_value + continuity_bonus +
            (0.45 if c.detail_tier == 0 else 0.15 if c.detail_tier == 1 else -0.18) +
            region_bonus + protected_bonus
        )
    active = [True] * len(pool)
    grid = {}
    for i, c in enumerate(pool):
        grid.setdefault(_grid_key(_candidate_center(c), BUDGET_GRID_MM), []).append(i)
    top_indexes = sorted(range(len(pool)), key=lambda i: values[i], reverse=True)

    selected = []
    cost = 0
    active_count = len(pool)
    cur = np.array([0.0, 0.0], dtype=np.float32)

    def select_index(i):
        nonlocal cost, cur, active_count
        active[i] = False
        active_count -= 1
        chosen = pool[i]
        if float(np.linalg.norm(chosen.points[-1] - cur)) < float(np.linalg.norm(chosen.points[0] - cur)):
            chosen = _candidate_with_points(chosen, chosen.points[::-1].copy())
        selected.append(chosen)
        cost += _candidate_min_segment_cost(chosen)
        cur = chosen.points[-1]

    qa_repair_indexes = sorted(
        [
            i for i, c in enumerate(pool)
            if active[i] and getattr(c, "source", "") == "qa_overlay_add"
        ],
        key=lambda i: values[i],
        reverse=True,
    )
    qa_repair_cost = 0
    qa_repair_cap = min(target, max(60, int(target * 0.40)))
    for i in qa_repair_indexes:
        min_cost = _candidate_min_segment_cost(pool[i])
        if min_cost <= 0 or cost + min_cost > target:
            continue
        if qa_repair_cost + min_cost > qa_repair_cap and qa_repair_cost > 0:
            continue
        select_index(i)
        qa_repair_cost += min_cost
    if qa_repair_indexes:
        print(
            f"  Overlay repair budget: selected "
            f"{sum(1 for c in selected if getattr(c, 'source', '') == 'qa_overlay_add')} "
            f"(cost {qa_repair_cost}/{qa_repair_cap})"
        )

    face_indexes = sorted(
        [
            i for i, c in enumerate(pool)
            if active[i] and (getattr(c, "region", "") == "face" or
                              getattr(c, "protected", False))
        ],
        key=lambda i: values[i],
        reverse=True,
    )
    face_cost = 0
    face_cap = min(
        target,
        max(FACE_FIXED_BUDGET_MIN, int(target * FACE_FIXED_BUDGET_RATIO)),
        int(max(1, target * FACE_FIXED_BUDGET_MAX_RATIO)),
    )
    for i in face_indexes:
        min_cost = _candidate_min_segment_cost(pool[i])
        if min_cost <= 0 or cost + min_cost > target:
            continue
        if face_cost + min_cost > face_cap and face_cost > 0:
            continue
        select_index(i)
        face_cost += min_cost
    if face_indexes:
        print(
            f"  Semantic budget: reserved face strokes "
            f"{sum(1 for c in selected if getattr(c, 'region', '') == 'face')} "
            f"(cost {face_cost}/{face_cap})"
        )

    # Coverage pass: reserve part of the budget for at most one strong base
    # contour in each occupied cell before global value ranking can concentrate
    # too many strokes in a small, noisy region.
    coverage_cap = max(0, min(target, int(target * 0.30)))
    tier0_cell_winners = []
    tier0_cell_count = 0
    for key, indexes in grid.items():
        eligible = [
            i for i in indexes
            if (active[i] and pool[i].detail_tier == 0 and
                getattr(pool[i], "region", "") != "face" and
                _candidate_min_segment_cost(pool[i]) > 0)
        ]
        if not eligible:
            continue
        tier0_cell_count += 1
        tier0_cell_winners.append((key, max(eligible, key=lambda i: values[i])))

    # When not every cell fits in the reserved share, retain the strongest
    # representative cells first. Each cell can still contribute only once.
    tier0_cell_winners.sort(key=lambda item: values[item[1]], reverse=True)
    covered_cells = set()
    coverage_cost = 0
    for key, i in tier0_cell_winners:
        min_cost = _candidate_min_segment_cost(pool[i])
        if coverage_cost + min_cost > coverage_cap or cost + min_cost > target:
            continue
        select_index(i)
        coverage_cost += min_cost
        covered_cells.add(key)

    print(
        f"  Budget coverage: {len(covered_cells)}/{len(grid)} grid cells covered "
        f"({tier0_cell_count} tier-0 eligible, cost {coverage_cost}/{coverage_cap})"
    )

    # Giu truoc mot phan contour/base va medium quan trong de khong mat form khi budget thap.
    garment_outline_cap = min(
        target,
        cost + max(1, int(target * GARMENT_OUTLINE_BUDGET_RATIO)),
    )
    garment_outline_cost = 0
    ranked_outlines = sorted(
        [
            i for i, c in enumerate(pool)
            if (active[i] and c.detail_tier == 0 and
                getattr(c, "region", "") in {"garment_outline", "detail", "garment_detail"})
        ],
        key=lambda i: values[i],
        reverse=True,
    )
    for i in ranked_outlines:
        min_cost = _candidate_min_segment_cost(pool[i])
        if min_cost <= 0 or cost + min_cost > target:
            continue
        if garment_outline_cost + min_cost > garment_outline_cap - face_cost:
            continue
        select_index(i)
        garment_outline_cost += min_cost

    tier_caps = {
        0: int(target * DETAIL_TIER_BASE_RATIO),
        1: int(target * (DETAIL_TIER_BASE_RATIO + DETAIL_TIER_MED_RATIO)),
    }
    for tier in (0, 1):
        ranked = sorted(
            [i for i, c in enumerate(pool) if active[i] and c.detail_tier == tier],
            key=lambda i: values[i],
            reverse=True,
        )
        for i in ranked:
            min_cost = _candidate_min_segment_cost(pool[i])
            if min_cost <= 0 or cost + min_cost > target or cost + min_cost > tier_caps[tier]:
                continue
            select_index(i)

    while active_count > 0 and cost < target:
        best_i = None
        best_score = -1e18
        candidate_indexes = set()
        cur_key = _grid_key(cur, BUDGET_GRID_MM)

        for radius in (1, 2):
            for key in _neighbor_grid_keys(cur_key, radius=radius):
                candidate_indexes.update(i for i in grid.get(key, ()) if active[i])
            if len(candidate_indexes) >= 24:
                break

        for i in top_indexes:
            if active[i]:
                candidate_indexes.add(i)
                if len(candidate_indexes) >= BUDGET_NEARBY_LIMIT:
                    break

        for i in candidate_indexes:
            if not active[i]:
                continue
            c = pool[i]
            min_cost = _candidate_min_segment_cost(c)
            if min_cost <= 0 or cost + min_cost > target:
                continue
            d_start = float(np.linalg.norm(c.points[0] - cur))
            d_end = float(np.linalg.norm(c.points[-1] - cur))
            travel_cost = min(d_start, d_end)
            travel_penalty = (
                BASE_DETAIL_TRAVEL_PENALTY if c.detail_tier == 0
                else 0.035 if c.detail_tier == 1
                else FINE_DETAIL_TRAVEL_PENALTY
            )
            score = values[i] / (1.0 + travel_cost * travel_penalty + min_cost * 0.002)
            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break

        select_index(best_i)

    semantic_counts = {
        "face": sum(1 for c in selected if getattr(c, "region", "") == "face"),
        "outline": sum(1 for c in selected if getattr(c, "region", "") == "garment_outline"),
        "small_detail": sum(
            1 for c in selected
            if getattr(c, "region", "") in {"garment_detail", "fine_detail", "detail", "hatch"} and
            getattr(c, "detail_tier", 1) >= 1
        ),
    }
    print(
        "  Semantic budget split: "
        f"face={semantic_counts['face']}, "
        f"garment_outline={semantic_counts['outline']}, "
        f"small_detail={semantic_counts['small_detail']}, "
        f"cost={cost}/{target}"
    )
    return selected


def _candidate_stroke_count_score(candidate):
    length_score = math.log1p(candidate.length_mm) / math.log1p(320.0)
    visual = candidate.saliency if candidate.saliency > 0 else candidate.importance
    tier_bonus = 0.48 if candidate.detail_tier == 0 else 0.18 if candidate.detail_tier == 1 else -0.12
    region = getattr(candidate, "region", "detail")
    region_bonus = (
        0.85 if region == "face"
        else 0.55 if region == "garment_outline"
        else 0.12 if region == "garment_detail"
        else -0.18 if region == "hatch"
        else 0.0
    )
    if getattr(candidate, "source", "") == "qa_overlay_add":
        region_bonus += 0.90
    protected_bonus = 0.7 if getattr(candidate, "protected", False) else 0.0
    return float(visual) * 3.0 + length_score * 2.2 + tier_bonus + region_bonus + protected_bonus


def _apply_stroke_count_soft_cap(selected, target_segments):
    if not selected:
        return selected
    target = int(max(MIN_STROKE_BUDGET, min(MAX_STROKE_BUDGET, target_segments)))
    cap = max(STROKE_COUNT_SOFT_TARGET_MIN,
              int(round(target * STROKE_COUNT_SOFT_TARGET_RATIO)))
    if len(selected) <= cap:
        return selected
    protected = [
        c for c in selected
        if getattr(c, "protected", False) or getattr(c, "region", "") == "face"
    ]
    protected_ids = {id(c) for c in protected}
    removable = [c for c in selected if id(c) not in protected_ids]
    room = max(0, cap - len(protected))
    ranked = sorted(removable, key=_candidate_stroke_count_score, reverse=True)
    retained = protected + ranked[:room]
    print(
        f"  Stroke-count soft cap: {len(selected)} -> {len(retained)} "
        f"selected candidates (cap {cap}, protected {len(protected)}, target {target} segments)"
    )
    return retained


def suggest_auto_detail_budget(candidates):
    """
    Estimate a sane detail budget from simplified geometry, not raw pixels.

    Raw skeleton points can be huge for anti-aliased line art. Auto mode should
    describe the number of useful plotter segments after curve simplification.
    """
    pool = [c for c in candidates or [] if len(getattr(c, "points", ())) >= 2]
    if not pool:
        return DEFAULT_STROKE_BUDGET

    simplified_segments = 0
    for candidate in pool:
        region = getattr(candidate, "region", "detail")
        if region == "face" or getattr(candidate, "protected", False):
            epsilon = FACE_RDP_EPSILON_MM
        elif getattr(candidate, "source", "") == "hatch":
            epsilon = max(LINE_ART_AUTO_RDP_EPSILON_MM, 0.70)
        elif int(getattr(candidate, "detail_tier", 1)) == 0:
            epsilon = max(LINE_ART_AUTO_RDP_EPSILON_MM * 0.72, 0.28)
        else:
            epsilon = LINE_ART_AUTO_RDP_EPSILON_MM
        simplified = _adaptive_rdp_simplify(candidate.points, epsilon)
        simplified_segments += max(
            _candidate_min_segment_cost(candidate),
            max(0, len(simplified) - 1),
        )

    stroke_headroom = min(260.0, len(pool) * 0.75)
    suggested = int(round(
        (simplified_segments * LINE_ART_AUTO_BUDGET_HEADROOM + stroke_headroom) / 100.0
    ) * 100)
    return max(
        MIN_STROKE_BUDGET,
        min(AUTO_BUDGET_MAX_SEGMENTS, suggested),
    )


def _fit_paths_to_budget(candidates, target_segments):
    target   = int(max(MIN_STROKE_BUDGET, min(MAX_STROKE_BUDGET, target_segments)))
    selected = _select_candidates_for_budget(candidates, target)
    if not selected:
        return [], 0.0, 0, {"base": 0, "medium": 0, "fine": 0}, False

    rdp_eps_warning = False
    initial_selected_count = len(selected)

    # Before allowing RDP to exceed the visual-fidelity ceiling, reduce the
    # number of selected candidates proportionally. This preserves curvature
    # on retained hair/face strokes instead of turning every stroke angular.
    for rebalance_pass in range(RDP_REBALANCE_MAX_PASSES):
        curvatures = [_path_curvature_scores(c.points) for c in selected]
        warning_paths = _simplify_all(
            selected, RDP_EPS_WARNING_MM, curvatures)
        warning_count = _count_segments(warning_paths)
        if warning_count <= target:
            break

        rdp_eps_warning = True
        selection_cost = sum(
            _candidate_min_segment_cost(candidate) for candidate in selected)
        next_selection_budget = max(
            1,
            int(selection_cost * target / max(1, warning_count) * 0.92),
        )
        if next_selection_budget >= selection_cost:
            next_selection_budget = max(1, selection_cost - 1)
        reselected = _select_candidates_for_budget(
            candidates, next_selection_budget)
        if not reselected or len(reselected) >= len(selected):
            break
        print(
            f"  RDP rebalance pass {rebalance_pass + 1}: "
            f"candidates {len(selected)} -> {len(reselected)}, "
            f"segments@{RDP_EPS_WARNING_MM:.1f}mm={warning_count} > {target}"
        )
        selected = reselected

    selected = _apply_stroke_count_soft_cap(selected, target)

    raw_segments = sum(max(1, len(c.points) - 1) for c in selected)
    tier_counts = {
        "base": sum(1 for c in selected if c.detail_tier == 0),
        "medium": sum(1 for c in selected if c.detail_tier == 1),
        "fine": sum(1 for c in selected if c.detail_tier == 2),
        "face": sum(1 for c in selected if getattr(c, "region", "") == "face"),
        "garment_outline": sum(
            1 for c in selected if getattr(c, "region", "") == "garment_outline"),
        "small_detail": sum(
            1 for c in selected
            if getattr(c, "region", "") in {"garment_detail", "fine_detail", "detail", "hatch"} and
            getattr(c, "detail_tier", 1) >= 1),
    }
    base_epsilon  = 0.12
    curvatures = [_path_curvature_scores(c.points) for c in selected]
    base_paths    = _simplify_all(selected, base_epsilon, curvatures)
    base_count    = _count_segments(base_paths)

    if base_count <= target:
        if rdp_eps_warning:
            print(
                f"  RDP warning: candidate pressure rebalanced "
                f"{initial_selected_count} -> {len(selected)} paths"
            )
        return base_paths, base_epsilon, raw_segments, tier_counts, rdp_eps_warning

    lo = base_epsilon; hi = 1.0
    hi_paths = _simplify_all(selected, hi, curvatures)
    hi_count = _count_segments(hi_paths)

    while hi_count > target and hi < 40.0:
        hi = (min(RDP_EPS_WARNING_MM, hi * 1.8)
              if hi < RDP_EPS_WARNING_MM else hi * 1.8)
        hi_paths = _simplify_all(selected, hi, curvatures)
        hi_count = _count_segments(hi_paths)

    best_paths = hi_paths; best_epsilon = hi; best_count = hi_count

    for _ in range(11):
        mid       = (lo + hi) / 2.0
        mid_paths = _simplify_all(selected, mid, curvatures)
        mid_count = _count_segments(mid_paths)
        if mid_count > target:
            lo = mid
        else:
            hi = mid; best_paths = mid_paths
            best_epsilon = mid;  best_count = mid_count
        if abs(best_count - target) <= max(20, int(target * 0.01)):
            break

    if best_epsilon > RDP_EPS_WARNING_MM:
        rdp_eps_warning = True
    if rdp_eps_warning:
        print(
            f"  RDP warning: final epsilon={best_epsilon:.3f}mm; "
            f"candidates {initial_selected_count} -> {len(selected)}"
        )
    return best_paths, best_epsilon, raw_segments, tier_counts, rdp_eps_warning


# ═══════════════════════════════════════════════════════════════════
#  MICRO-STROKE FILTERING  [NEW]
# ═══════════════════════════════════════════════════════════════════

def _filter_micro_strokes(candidates, micro_mm=MICRO_MM,
                          isolation_mm=STITCH_THRESHOLD_MM):
    """
    Remove isolated CandidatePaths shorter than micro_mm.
    Short paths near a long path are kept (they are real detail).
    Short isolated paths are discarded (they are dust/noise dots).
    """
    if not candidates:
        return candidates

    protected_paths = [
        c for c in candidates
        if getattr(c, "protected", False) or getattr(c, "region", "") in {
            "face", "garment_outline"
        }
    ]
    protected_ids = {id(c) for c in protected_paths}
    review_paths = [c for c in candidates if id(c) not in protected_ids]
    long_paths  = [c for c in review_paths if c.length_mm >= micro_mm]
    short_paths = [c for c in review_paths if c.length_mm <  micro_mm]
    if not short_paths:
        return protected_paths + long_paths

    long_centers = (np.array([c.points.mean(axis=0) for c in long_paths],
                              dtype=np.float32)
                    if long_paths else np.empty((0, 2), dtype=np.float32))

    kept = protected_paths + list(long_paths)
    for s in short_paths:
        center = s.points.mean(axis=0)
        if len(long_centers) > 0:
            dists = np.linalg.norm(long_centers - center, axis=1)
            if float(dists.min()) <= isolation_mm:
                kept.append(s)

    removed = len(candidates) - len(kept)
    print(f"  Micro-stroke filter: {len(candidates)} -> {len(kept)} "
          f"(removed {removed} isolated strokes < {micro_mm}mm, "
          f"protected {len(protected_paths)})")
    return kept


# ═══════════════════════════════════════════════════════════════════
#  GREEDY NEAREST-NEIGHBOR PATH ORDERING  [UPGRADED]
# ═══════════════════════════════════════════════════════════════════

def optimize_path_order(paths):
    """
    Sort paths to minimize total G0 rapid travel distance.

    Strategy:
      n <= GREEDY_NN_LIMIT: Greedy nearest-neighbor with angle-priority bonus.
        At each step, picks the unvisited path whose nearest endpoint is closest
        to the current pen position. An angle bonus reduces effective distance
        when the next path continues in the same direction (reduces sharp turns).
      n > GREEDY_NN_LIMIT: Falls back to fast serpentine row-sort (O(n log n)).

    Each path is also flipped if its endpoint is closer than its start.
    """
    if len(paths) <= 1:
        return paths

    n = len(paths)
    if n > GREEDY_NN_LIMIT:
        return _serpentine_order(paths)

    starts = np.array([p[0]  for p in paths], dtype=np.float32)
    ends   = np.array([p[-1] for p in paths], dtype=np.float32)
    used   = np.zeros(n, dtype=bool)
    ordered = []

    cur_pos     = np.array([0.0, 0.0], dtype=np.float32)
    cur_dir     = np.array([1.0, 0.0], dtype=np.float32)
    pen_is_down = False

    for _ in range(n):
        ds = np.linalg.norm(starts - cur_pos, axis=1)
        de = np.linalg.norm(ends   - cur_pos, axis=1)
        best_dist = np.minimum(ds, de)
        best_dist[used] = 1e18

        effective_start = ds.copy()
        effective_end = de.copy()
        if pen_is_down:
            mask = ~used
            for points, distances, effective_endpoint in (
                    (starts, ds, effective_start), (ends, de, effective_end)):
                vectors = points - cur_pos
                safe_distances = np.maximum(distances, 1e-6)
                cosines = np.sum(vectors * cur_dir, axis=1) / safe_distances
                bonuses = cosines * 0.3 * distances
                bonuses[distances < 1e-6] = 2.0
                effective_endpoint[mask] -= bonuses[mask]

        effective = np.minimum(effective_start, effective_end)
        effective[used] = 1e18

        idx       = int(np.argmin(effective))
        used[idx] = True
        p = paths[idx]
        q = p if effective_start[idx] <= effective_end[idx] else p[::-1].copy()
        ordered.append(q)

        if len(q) >= 2:
            tail    = q[-1] - q[-2]
            tl      = float(np.linalg.norm(tail))
            cur_dir = tail / tl if tl > 1e-6 else cur_dir
        cur_pos     = q[-1]
        pen_is_down = True

    return ordered


def _serpentine_order(paths):
    """Fast O(n log n) serpentine row-sort fallback for large path counts."""
    row_height = 24.0
    groups     = {}
    for p in paths:
        center = np.mean(p, axis=0)
        row    = int(center[1] // row_height)
        groups.setdefault(row, []).append(p)

    ordered    = []
    last_point = None
    for idx, row in enumerate(sorted(groups.keys())):
        group = groups[row]
        group.sort(key=lambda p: float(np.mean(p[:, 0])), reverse=(idx % 2 == 1))
        for p in group:
            q = p
            if last_point is not None:
                if float(np.linalg.norm(p[-1] - last_point)) < \
                   float(np.linalg.norm(p[0]  - last_point)):
                    q = p[::-1].copy()
            ordered.append(q)
            last_point = q[-1]
    return ordered


def _group_paths_by_islands(paths, grid_mm=ISLAND_GRID_MM):
    """
    Island grouping: gom stroke theo o luoi gan nhau thanh cum cuc bo.
    Route se toi uu trong tung cum truoc, roi moi sap thu tu cac cum de giam G0 xa.
    """
    if len(paths) <= 1:
        return [paths]
    groups = {}
    for p in paths:
        center = np.mean(p, axis=0)
        key = (int(center[0] // grid_mm), int(center[1] // grid_mm))
        groups.setdefault(key, []).append(p)
    islands = list(groups.values())
    islands.sort(key=lambda group: (-len(group), float(np.mean([np.mean(p[:, 1]) for p in group]))))
    return islands


def _order_route_groups_nearest(route_groups):
    """Join already-optimized local route groups by nearest available endpoint."""
    pending = [list(group) for group in route_groups if group]
    ordered = []
    cur = np.array([0.0, 0.0], dtype=np.float32)
    while pending:
        best_i = 0
        best_flip = False
        best_dist = float("inf")
        for i, route in enumerate(pending):
            start_dist = float(np.linalg.norm(route[0][0] - cur))
            end_dist = float(np.linalg.norm(route[-1][-1] - cur))
            if start_dist < best_dist:
                best_i, best_flip, best_dist = i, False, start_dist
            if end_dist < best_dist:
                best_i, best_flip, best_dist = i, True, end_dist
        route = pending.pop(best_i)
        if best_flip:
            route = [path[::-1].copy() for path in reversed(route)]
        ordered.extend(route)
        cur = ordered[-1][-1]
    return ordered


def _optimize_paths_with_islands(paths):
    """
    Toi uu route theo cum: Greedy NN trong tung island, sau do sap island gan nhat.
    Giu interface optimize_path_order cu, chi them lop route-aware truoc 2-opt/Or-opt.
    """
    islands = _group_paths_by_islands(paths)
    if len(islands) <= 1:
        return optimize_path_order(paths), len(islands)

    island_routes = [optimize_path_order(group) for group in islands if group]
    ordered = _order_route_groups_nearest(island_routes)
    print(f"  Island grouping: {len(paths)} paths in {len(islands)} islands")
    if len(paths) <= min(1200, GREEDY_NN_LIMIT):
        global_route = optimize_path_order(paths)
        if _route_travel_distance(global_route) + 1e-6 < _route_travel_distance(ordered):
            print("  Route selection: global NN beat island route")
            return global_route, 1
    return ordered, len(islands)


def _route_travel_distance(paths):
    """Tinh tong quang duong G0 giua cac stroke theo thu tu hien tai."""
    if not paths:
        return 0.0
    total = 0.0
    cur = np.array([0.0, 0.0], dtype=np.float32)
    for p in paths:
        if len(p) < 2:
            continue
        total += float(np.linalg.norm(p[0] - cur))
        cur = p[-1]
    return total


def _route_edge_cost(prev_path, next_path):
    """Tinh chi phi travel tu stroke truoc sang stroke sau, hoac tu home neu prev=None."""
    if next_path is None:
        return 0.0
    prev_end = np.array([0.0, 0.0], dtype=np.float32) if prev_path is None else prev_path[-1]
    return float(np.linalg.norm(next_path[0] - prev_end))


def _optimize_stroke_directions(paths, max_passes=STROKE_DIRECTION_MAX_PASSES):
    """
    Toi uu huong ve tung stroke ma khong doi thu tu route.
    Moi stroke duoc thu dao chieu neu tong travel tu stroke truoc -> stroke nay -> stroke sau giam.
    """
    if len(paths) <= 1:
        return paths, 0
    route = [p.copy() for p in paths]
    flip_count = 0
    for _ in range(max_passes):
        changed = False
        for i, path in enumerate(route):
            if len(path) < 2:
                continue
            prev_path = None if i == 0 else route[i - 1]
            next_path = None if i + 1 >= len(route) else route[i + 1]
            old_cost = _route_edge_cost(prev_path, path) + _route_edge_cost(path, next_path)
            rev = path[::-1].copy()
            new_cost = _route_edge_cost(prev_path, rev) + _route_edge_cost(rev, next_path)
            if new_cost + 1e-6 < old_cost:
                route[i] = rev
                flip_count += 1
                changed = True
        if not changed:
            break
    if flip_count:
        print(f"  Stroke direction: flipped {flip_count} strokes")
    return route, flip_count


def _two_opt_route_improve(paths, max_paths=TWO_OPT_MAX_PATHS,
                           max_passes=TWO_OPT_MAX_PASSES,
                           time_budget_s=ROUTE_IMPROVE_TIME_BUDGET_S,
                           log_improvement=True):
    """
    Cai thien route sau greedy bang 2-opt cho open path.
    Khi dao mot doan route, ta dao ca thu tu stroke va huong tung stroke de
    dau/cuoi moi khop voi duong di moi. Gioi han so path de tranh treo UI.
    """
    if len(paths) < 4 or len(paths) > max_paths:
        return paths, 0.0

    route = [p.copy() for p in paths]
    before = _route_travel_distance(route)

    n = len(route)
    t0 = time.perf_counter()
    for _pass in range(max_passes):
        improved = False
        for i in range(n - 2):
            if time.perf_counter() - t0 > time_budget_s:
                break
            prev_path = None if i == 0 else route[i - 1]
            old_a = _route_edge_cost(prev_path, route[i])
            for k in range(i + 1, n):
                if time.perf_counter() - t0 > time_budget_s:
                    break
                next_path = None if k + 1 >= n else route[k + 1]
                old_b = _route_edge_cost(route[k], next_path)

                new_first = route[k][::-1].copy()
                new_last = route[i][::-1].copy()
                new_a = _route_edge_cost(prev_path, new_first)
                new_b = _route_edge_cost(new_last, next_path)

                if new_a + new_b + 1e-6 < old_a + old_b:
                    route[i:k + 1] = [p[::-1].copy() for p in reversed(route[i:k + 1])]
                    improved = True
                    break
            if improved:
                break
        if time.perf_counter() - t0 > time_budget_s:
            break
        if not improved:
            break

    after = _route_travel_distance(route)
    improvement = max(0.0, before - after)
    if log_improvement and improvement > 0.01:
        print(f"  2-opt route: travel {before:.1f}mm -> {after:.1f}mm "
              f"(saved {improvement:.1f}mm)")
    return route, improvement


def _or_opt_route_improve(paths, max_paths=OR_OPT_MAX_PATHS,
                          max_passes=OR_OPT_MAX_PASSES,
                          max_block=OR_OPT_MAX_BLOCK,
                          time_budget_s=ROUTE_IMPROVE_TIME_BUDGET_S,
                          log_improvement=True):
    """
    Or-opt: di chuyen 1 stroke hoac cum 1-3 stroke sang vi tri khac trong route.
    Moi lan thu chi tinh delta travel o cac canh bien, nen nhanh hon tinh lai ca route.
    Co time budget de UI khong bi treo khi anh co qua nhieu stroke.
    """
    if len(paths) < 4 or len(paths) > max_paths:
        return paths, 0.0

    route = [p.copy() for p in paths]
    before = _route_travel_distance(route)
    n = len(route)
    t0 = time.perf_counter()

    for _pass in range(max_passes):
        improved = False
        for block_len in range(1, max(1, max_block) + 1):
            if block_len >= n:
                continue
            for i in range(0, n - block_len + 1):
                if time.perf_counter() - t0 > time_budget_s:
                    break

                block = route[i:i + block_len]
                prev_path = None if i == 0 else route[i - 1]
                next_path = None if i + block_len >= n else route[i + block_len]
                old_remove = (_route_edge_cost(prev_path, block[0]) +
                              _route_edge_cost(block[-1], next_path))
                new_bridge = _route_edge_cost(prev_path, next_path)

                remaining = route[:i] + route[i + block_len:]
                for insert_at in range(0, len(remaining) + 1):
                    if insert_at == i:
                        continue
                    insert_prev = None if insert_at == 0 else remaining[insert_at - 1]
                    insert_next = None if insert_at >= len(remaining) else remaining[insert_at]
                    old_insert = _route_edge_cost(insert_prev, insert_next)

                    # Thu giu nguyen huong cum stroke.
                    new_insert = (_route_edge_cost(insert_prev, block[0]) +
                                  _route_edge_cost(block[-1], insert_next))
                    delta = (new_bridge + new_insert) - (old_remove + old_insert)
                    best_block = block
                    best_delta = delta

                    # Thu dao ca cum va dao huong tung stroke trong cum.
                    reversed_block = [p[::-1].copy() for p in reversed(block)]
                    rev_insert = (_route_edge_cost(insert_prev, reversed_block[0]) +
                                  _route_edge_cost(reversed_block[-1], insert_next))
                    rev_delta = (new_bridge + rev_insert) - (old_remove + old_insert)
                    if rev_delta < best_delta:
                        best_delta = rev_delta
                        best_block = reversed_block

                    if best_delta < -1e-6:
                        route = remaining[:insert_at] + [p.copy() for p in best_block] + remaining[insert_at:]
                        n = len(route)
                        improved = True
                        break
                if improved:
                    break
            if improved or time.perf_counter() - t0 > time_budget_s:
                break
        if not improved or time.perf_counter() - t0 > time_budget_s:
            break

    after = _route_travel_distance(route)
    improvement = max(0.0, before - after)
    if log_improvement and improvement > 0.01:
        print(f"  Or-opt route: travel {before:.1f}mm -> {after:.1f}mm "
              f"(saved {improvement:.1f}mm)")
    return route, improvement


def _postprocess_route_improve(paths):
    """
    Hau xu ly route sau Greedy NN: chay 2-opt roi Or-opt trong time budget tong.
    Tra ve tong mm va % travel duoc cai thien so voi route greedy ban dau.
    """
    if len(paths) <= 1:
        return paths, 0.0, 0.0

    before = _route_travel_distance(paths)
    if before <= 1e-6:
        return paths, 0.0, 0.0

    total_paths = len(paths)
    t0 = time.perf_counter()
    if total_paths <= min(TWO_OPT_MAX_PATHS, OR_OPT_MAX_PATHS):
        route, _ = _two_opt_route_improve(
            paths, time_budget_s=ROUTE_IMPROVE_TIME_BUDGET_S * 0.55)
        elapsed = time.perf_counter() - t0
        remaining_budget = max(0.0, ROUTE_IMPROVE_TIME_BUDGET_S - elapsed)
        route, _ = _or_opt_route_improve(route, time_budget_s=remaining_budget)
        optimized_paths = total_paths
        optimized_chunks = 1
        total_chunks = 1
    else:
        # Large routes are optimized locally instead of being rejected by the
        # hard MAX_PATHS guards. Spatial islands are split again when a dense
        # region itself exceeds the smaller Or-opt limit.
        chunk_limit = min(TWO_OPT_MAX_PATHS, OR_OPT_MAX_PATHS)
        route_chunks = []
        for island in _group_paths_by_islands(paths):
            for start in range(0, len(island), chunk_limit):
                route_chunks.append(list(island[start:start + chunk_limit]))

        total_chunks = len(route_chunks)
        optimized_paths = 0
        optimized_chunks = 0
        for chunk_index, chunk in enumerate(route_chunks):
            remaining = ROUTE_IMPROVE_TIME_BUDGET_S - (time.perf_counter() - t0)
            if remaining <= 0.0:
                break
            chunks_left = max(1, total_chunks - chunk_index)
            chunk_budget = remaining / chunks_left
            if len(chunk) >= 4:
                improved_chunk, _ = _two_opt_route_improve(
                    chunk, time_budget_s=chunk_budget * 0.55,
                    log_improvement=False)
                elapsed_chunk = time.perf_counter() - t0
                remaining_chunk = max(
                    0.0,
                    min(chunk_budget * 0.45,
                        ROUTE_IMPROVE_TIME_BUDGET_S - elapsed_chunk),
                )
                improved_chunk, _ = _or_opt_route_improve(
                    improved_chunk, time_budget_s=remaining_chunk,
                    log_improvement=False)
                route_chunks[chunk_index] = improved_chunk
            optimized_paths += len(chunk)
            optimized_chunks += 1

        route = _order_route_groups_nearest(route_chunks)

    coverage_pct = 100.0 * optimized_paths / max(1, total_paths)
    limited_text = "time-limited" if optimized_paths < total_paths else "full"
    print(
        f"  Route optimization coverage: {optimized_paths}/{total_paths} paths "
        f"({coverage_pct:.1f}%), chunks {optimized_chunks}/{total_chunks} "
        f"({limited_text})"
    )

    after = _route_travel_distance(route)
    if after > before + 1e-6:
        route = [path.copy() for path in paths]
        after = before
    saved_mm = max(0.0, before - after)
    saved_pct = saved_mm / before * 100.0
    if saved_mm > 0.01:
        print(f"  Route postprocess: saved {saved_mm:.1f}mm ({saved_pct:.1f}%)")
    return route, saved_mm, saved_pct


def _compress_polyline_collinear(points, angle_deg=COLLINEAR_ANGLE_DEG,
                                 dist_tol=COLLINEAR_DIST_MM):
    """
    Nen polyline sau RDP: bo diem giua neu ba diem gan thang hang.
    Dung dung sai goc nho va sai lech vuong goc tinh bang mm de khong lam meo net.
    """
    if len(points) <= 2:
        return points

    cos_limit = math.cos(math.radians(angle_deg))
    kept = [points[0]]
    removed = 0

    for i in range(1, len(points) - 1):
        a = kept[-1]
        b = points[i]
        c = points[i + 1]
        v1 = b - a
        v2 = c - b
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            removed += 1
            continue
        cos_a = float(np.dot(v1 / n1, v2 / n2))
        line = c - a
        line_len = float(np.linalg.norm(line))
        cross_2d = float(line[0] * (b - a)[1] - line[1] * (b - a)[0])
        perp = 0.0 if line_len < 1e-6 else abs(cross_2d / line_len)
        if cos_a >= cos_limit and perp <= dist_tol:
            removed += 1
            continue
        kept.append(b)

    kept.append(points[-1])
    return np.asarray(kept, dtype=np.float32)


def _compress_paths_collinear(paths):
    compressed = []
    before = _count_segments(paths)
    for p in paths:
        q = _compress_polyline_collinear(p)
        if len(q) >= 2:
            compressed.append(q)
    after = _count_segments(compressed)
    removed = max(0, before - after)
    if removed:
        print(f"  Polyline compression: {before} -> {after} segments "
              f"(removed {removed})")
    return compressed, before, after, removed


# ═══════════════════════════════════════════════════════════════════
#  SMART GAP-BRIDGING / PATH STITCHING  [NEW]
# ═══════════════════════════════════════════════════════════════════

def _stitch_paths(paths, threshold_mm=STITCH_THRESHOLD_MM,
                  angle_cos_min=STITCH_ANGLE_COS, segment_cap=None):
    """
    Merge consecutive paths where end(A) -> start(B) gap is small AND direction
    is sufficiently collinear (cosine >= angle_cos_min).

    This eliminates pen-lift -> rapid -> pen-down sequences for near-continuous
    strokes, producing smoother G-code with fewer Z-moves.

    Returns: (merged_paths_list, stitch_count)
    """
    if len(paths) <= 1:
        return paths, 0

    stitched_count = 0
    result = [paths[0].copy()]
    segment_count = _count_segments(paths)

    for i in range(1, len(paths)):
        prev  = result[-1]
        curr  = paths[i]
        gap   = float(np.linalg.norm(curr[0] - prev[-1]))

        if gap > STITCH_MAX_GAP_MM:
            result.append(curr.copy())
            continue

        # Scoring thong minh: gan nhau + cung huong + stroke khong qua lech nhau thi noi.
        direction_score = 1.0
        if len(prev) >= 2 and len(curr) >= 2:
            d1 = prev[-1] - prev[-2]
            d2 = curr[1]  - curr[0]
            n1 = float(np.linalg.norm(d1))
            n2 = float(np.linalg.norm(d2))
            if n1 > 1e-6 and n2 > 1e-6:
                direction_score = max(0.0, (float(np.dot(d1 / n1, d2 / n2)) + 1.0) * 0.5)

        gap_score = max(0.0, 1.0 - gap / max(1e-6, STITCH_MAX_GAP_MM))
        len_prev = _path_length_mm(prev)
        len_curr = _path_length_mm(curr)
        length_balance = min(len_prev, len_curr) / max(len_prev, len_curr, 1e-6)
        stitch_score = gap_score * 0.55 + direction_score * 0.35 + length_balance * 0.10
        can_stitch = (gap <= threshold_mm and direction_score >= (angle_cos_min + 1.0) * 0.5) or \
                     stitch_score >= STITCH_SCORE_THRESHOLD

        adds_connector = gap >= 1e-4
        if can_stitch and (not adds_connector or segment_cap is None or
                           segment_count < segment_cap):
            # Merge: append current path (skip its first point to avoid duplicate).
            merged     = np.vstack([prev, curr[1:]]) if gap < 1e-4 \
                         else np.vstack([prev, curr[0:1], curr[1:]])
            result[-1] = merged
            stitched_count += 1
            segment_count += int(adds_connector)
        else:
            result.append(curr.copy())

    print(f"  Path stitching: {len(paths)} -> {len(result)} "
          f"(merged {stitched_count} gaps < {threshold_mm}mm)")
    return result, stitched_count


def _coerce_stroke_paths(candidates):
    if isinstance(candidates, np.ndarray):
        arr = np.asarray(candidates, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 2:
            candidates = [arr]
    paths = []
    priorities = []
    for item in candidates:
        if isinstance(item, CandidatePath):
            raw_points = item.points
            visual = item.saliency if item.saliency > 0 else item.importance
            priority = float(visual) * 150.0 + float(item.length_mm)
        else:
            raw_points = item
            priority = None
        pts = np.asarray(raw_points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            continue
        if not np.all(np.isfinite(pts)):
            continue
        paths.append(pts.copy())
        priorities.append(priority)
    return paths, priorities


def _path_is_closed(path, tol_mm=0.35):
    return len(path) >= 4 and float(np.linalg.norm(path[0] - path[-1])) <= tol_mm


def _path_visual_priority(path, explicit_priority=None):
    if explicit_priority is not None:
        return float(explicit_priority)
    length = _path_length_mm(path)
    turn = _polyline_turn_saliency(path)
    return length * (1.0 + 0.25 * turn) + math.log1p(max(0, len(path) - 1))


def _try_merge_oriented_paths(path_a, path_b, angle_tolerance_deg, endpoint_gap_mm):
    if len(path_a) < 2 or len(path_b) < 2:
        return None
    if _path_is_closed(path_a) or _path_is_closed(path_b):
        return None

    cos_limit = math.cos(math.radians(float(angle_tolerance_deg)))
    best = None
    best_score = float("inf")

    for reverse_a in (False, True):
        first = path_a[::-1].copy() if reverse_a else path_a
        v1 = first[-1] - first[-2]
        n1 = float(np.linalg.norm(v1))
        if n1 <= 1e-6:
            continue
        d1 = v1 / n1

        for reverse_b in (False, True):
            second = path_b[::-1].copy() if reverse_b else path_b
            gap = float(np.linalg.norm(second[0] - first[-1]))
            if gap > endpoint_gap_mm:
                continue
            v2 = second[1] - second[0]
            n2 = float(np.linalg.norm(v2))
            if n2 <= 1e-6:
                continue
            cos_a = float(np.dot(d1, v2 / n2))
            if cos_a < cos_limit:
                continue

            score = gap + (1.0 - cos_a) * max(0.25, endpoint_gap_mm)
            if score < best_score:
                if gap <= 1e-4:
                    merged = np.vstack([first, second[1:]])
                else:
                    merged = np.vstack([first, second])
                best = merged.astype(np.float32)
                best_score = score

    return best


def _path_bbox(path, pad=0.0):
    mins = np.min(path, axis=0)
    maxs = np.max(path, axis=0)
    return (
        float(mins[0] - pad),
        float(mins[1] - pad),
        float(maxs[0] + pad),
        float(maxs[1] + pad),
    )


def _bboxes_touch(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bbox_grid_keys(bbox, cell_mm):
    x0, y0, x1, y1 = bbox
    gx0 = int(math.floor(x0 / cell_mm))
    gy0 = int(math.floor(y0 / cell_mm))
    gx1 = int(math.floor(x1 / cell_mm))
    gy1 = int(math.floor(y1 / cell_mm))
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            yield (gx, gy)


def _segment_overlap_fraction(path_a, path_b,
                              distance_mm=OVERLAP_REDUCTION_DISTANCE_MM,
                              angle_deg=OVERLAP_REDUCTION_ANGLE_DEG):
    if len(path_a) < 2 or len(path_b) < 2:
        return 0.0

    b0 = path_b[:-1].astype(np.float32)
    b1 = path_b[1:].astype(np.float32)
    bv = b1 - b0
    b_len = np.linalg.norm(bv, axis=1)
    valid_b = b_len > 1e-6
    if not bool(np.any(valid_b)):
        return 0.0
    b0 = b0[valid_b]
    bv = bv[valid_b]
    b_len = b_len[valid_b]
    b_dir = bv / b_len[:, None]

    cos_limit = math.cos(math.radians(float(angle_deg)))
    total = 0.0
    overlapped = 0.0
    for a0, a1 in zip(path_a[:-1], path_a[1:]):
        av = a1 - a0
        a_len = float(np.linalg.norm(av))
        if a_len <= 1e-6:
            continue
        total += a_len
        a_dir = av / a_len
        parallel = np.abs(b_dir @ a_dir) >= cos_limit
        if not bool(np.any(parallel)):
            continue

        mid = (a0 + a1) * 0.5
        seg0 = b0[parallel]
        segv = bv[parallel]
        denom = np.sum(segv * segv, axis=1)
        denom = np.where(denom <= 1e-9, 1.0, denom)
        t = np.sum((mid - seg0) * segv, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        closest = seg0 + segv * t[:, None]
        if float(np.min(np.linalg.norm(closest - mid, axis=1))) <= distance_mm:
            overlapped += a_len

    return overlapped / max(1e-6, total)


def _remove_overlapping_strokes(paths, priorities=None,
                                distance_mm=OVERLAP_REDUCTION_DISTANCE_MM,
                                overlap_fraction=OVERLAP_REDUCTION_FRACTION,
                                grid_mm=OVERLAP_REDUCTION_GRID_MM):
    if len(paths) <= 1:
        return paths, 0
    if len(paths) > OVERLAP_REDUCTION_MAX_PATHS:
        print(
            f"  Overlap reduction: skipped {len(paths)} paths "
            f"(limit {OVERLAP_REDUCTION_MAX_PATHS})"
        )
        return paths, 0

    if priorities is None:
        priorities = [None] * len(paths)
    ranked = sorted(
        range(len(paths)),
        key=lambda idx: _path_visual_priority(paths[idx], priorities[idx]),
        reverse=True,
    )
    kept = []
    kept_bboxes = []
    grid = {}
    removed = 0
    expanded = float(distance_mm) * 2.5
    cell = max(float(grid_mm), expanded * 2.0, 1e-6)

    for idx in ranked:
        path = paths[idx]
        bbox = _path_bbox(path, pad=expanded)
        nearby = set()
        for key in _bbox_grid_keys(bbox, cell):
            nearby.update(grid.get(key, ()))

        duplicate = False
        for kept_idx in nearby:
            if not _bboxes_touch(bbox, kept_bboxes[kept_idx]):
                continue
            fraction = _segment_overlap_fraction(path, kept[kept_idx],
                                                 distance_mm=distance_mm)
            if fraction >= overlap_fraction:
                duplicate = True
                break

        if duplicate:
            removed += 1
            continue

        kept_idx = len(kept)
        kept.append(path.copy())
        kept_bboxes.append(bbox)
        for key in _bbox_grid_keys(bbox, cell):
            grid.setdefault(key, []).append(kept_idx)

    if removed:
        print(
            f"  Overlap reduction: {len(paths)} -> {len(kept)} strokes "
            f"(removed {removed} overlapped strokes)"
        )
    return kept, removed


def merge_collinear_and_touching_strokes(
        candidates,
        angle_tolerance_deg=STROKE_MERGE_ANGLE_TOLERANCE_DEG,
        endpoint_gap_mm=STROKE_MERGE_ENDPOINT_GAP_MM,
        return_stats=False):
    """
    Merge strokes whose endpoints touch and continue in the same direction.

    The function accepts either CandidatePath objects or raw mm polylines and
    repeats merge passes until no eligible endpoint pair remains. It then removes
    shorter/lower-priority strokes that substantially overlap a retained stroke.
    """
    paths, priorities = _coerce_stroke_paths(candidates)
    before = len(paths)
    if before <= 1:
        stats = {
            "before": before,
            "after": before,
            "merged_count": 0,
            "overlap_removed_count": 0,
            "reduction_pct": 0.0,
        }
        return (paths, stats) if return_stats else paths

    total_merged = 0
    endpoint_gap = float(max(0.0, endpoint_gap_mm))
    cell_mm = max(endpoint_gap, 1e-6)
    max_passes = max(1, int(STROKE_MERGE_MAX_PASSES))

    for pass_index in range(max_passes):
        endpoint_grid = {}
        for index, path in enumerate(paths):
            if len(path) < 2 or _path_is_closed(path):
                continue
            endpoint_grid.setdefault(_grid_key(path[0], cell_mm), []).append(index)
            endpoint_grid.setdefault(_grid_key(path[-1], cell_mm), []).append(index)

        used = np.zeros(len(paths), dtype=bool)
        next_paths = []
        next_priorities = []
        merged_this_pass = 0

        for i, path in enumerate(paths):
            if used[i]:
                continue

            candidate_indexes = set()
            for endpoint in (path[0], path[-1]):
                for key in _neighbor_grid_keys(_grid_key(endpoint, cell_mm), radius=1):
                    candidate_indexes.update(endpoint_grid.get(key, ()))
            candidate_indexes.discard(i)

            best = None
            best_j = None
            best_gap = float("inf")
            for j in candidate_indexes:
                if used[j]:
                    continue
                merged = _try_merge_oriented_paths(
                    path, paths[j], angle_tolerance_deg, endpoint_gap)
                if merged is None:
                    continue
                endpoint_dist = min(
                    float(np.linalg.norm(path[0] - paths[j][0])),
                    float(np.linalg.norm(path[0] - paths[j][-1])),
                    float(np.linalg.norm(path[-1] - paths[j][0])),
                    float(np.linalg.norm(path[-1] - paths[j][-1])),
                )
                if endpoint_dist < best_gap:
                    best = merged
                    best_j = j
                    best_gap = endpoint_dist

            if best is not None and best_j is not None:
                simplified = _adaptive_rdp_simplify(
                    best, STROKE_MERGE_RDP_EPSILON_MM)
                if len(simplified) >= 2:
                    next_paths.append(simplified.astype(np.float32))
                    priority = max(
                        _path_visual_priority(path, priorities[i]),
                        _path_visual_priority(paths[best_j], priorities[best_j]),
                    )
                    next_priorities.append(priority)
                    used[i] = True
                    used[best_j] = True
                    merged_this_pass += 1
                    continue

            next_paths.append(path.copy())
            next_priorities.append(priorities[i])
            used[i] = True

        paths = next_paths
        priorities = next_priorities
        total_merged += merged_this_pass
        if merged_this_pass:
            print(
                f"  Stroke merge pass {pass_index + 1}: "
                f"merged {merged_this_pass}, remaining {len(paths)}"
            )
        else:
            break

    paths, overlap_removed = _remove_overlapping_strokes(paths, priorities)
    after = len(paths)
    reduction_pct = 100.0 * (before - after) / max(1, before)
    print(
        f"  Stroke merge/overlap: {before} -> {after} strokes "
        f"(-{reduction_pct:.1f}%; endpoint merges={total_merged}, "
        f"overlap removed={overlap_removed})"
    )
    stats = {
        "before": before,
        "after": after,
        "merged_count": total_merged,
        "overlap_removed_count": overlap_removed,
        "reduction_pct": reduction_pct,
    }
    return (paths, stats) if return_stats else paths


# ═══════════════════════════════════════════════════════════════════
#  PATHS -> G-CODE  (SEAMLESS G1 CONTINUATION)  [UPGRADED]
# ═══════════════════════════════════════════════════════════════════

def _bridge_short_pen_lift_gaps(paths, segment_cap=None):
    """
    Giam nhac but bang cach ve noi cac gap rat ngan giua hai stroke lien tiep.
    Khac stitching theo contour, bridge nay chap nhan goc lech neu gap du nho;
    duong noi ngan thuong it thay hon thao tac nhac/ha but tren plotter.
    """
    if len(paths) <= 1:
        return paths, 0

    bridged = [paths[0].copy()]
    bridge_count = 0
    bridge_mm = 0.0
    segment_count = _count_segments(paths)

    for curr in paths[1:]:
        prev = bridged[-1]
        if len(prev) < 2 or len(curr) < 2:
            bridged.append(curr.copy())
            continue

        gap = float(np.linalg.norm(curr[0] - prev[-1]))
        d1 = prev[-1] - prev[-2]
        d2 = curr[1] - curr[0]
        n1 = float(np.linalg.norm(d1))
        n2 = float(np.linalg.norm(d2))
        direction_score = 0.5
        connector_score = 0.0
        if n1 > 1e-6 and n2 > 1e-6:
            direction_score = (float(np.dot(d1 / n1, d2 / n2)) + 1.0) * 0.5
        if gap > 1e-6 and n1 > 1e-6 and n2 > 1e-6:
            connector = (curr[0] - prev[-1]) / gap
            connector_score = min(
                float(np.dot(d1 / n1, connector)),
                float(np.dot(connector, d2 / n2)),
            )

        gap_score = max(0.0, 1.0 - gap / max(1e-6, PEN_LIFT_BRIDGE_MAX_MM))
        bridge_score = gap_score * 0.72 + direction_score * 0.28
        can_bridge = (
            gap <= 0.20 or
            (gap <= 0.55 and direction_score >= 0.55 and connector_score >= 0.35) or
            (gap <= PEN_LIFT_BRIDGE_MAX_MM and direction_score >= 0.75 and
             connector_score >= 0.60 and bridge_score >= PEN_LIFT_BRIDGE_SCORE)
        )
        if segment_cap is not None and segment_count >= segment_cap and gap >= 1e-4:
            can_bridge = False

        if can_bridge:
            # Giu curr[0] de G1 ve duong noi ngan tu prev[-1] sang dau stroke tiep theo.
            bridged[-1] = np.vstack([prev, curr[0:1], curr[1:]]).astype(np.float32)
            bridge_count += 1
            bridge_mm += gap
            segment_count += int(gap >= 1e-4)
        else:
            bridged.append(curr.copy())

    if bridge_count:
        print(f"  Pen-lift bridge: merged {bridge_count} tiny gaps "
              f"({bridge_mm:.1f}mm drawn connectors)")
    return bridged, bridge_count


def _legacy_paths_to_gcode_removed(paths, feed_rate=FEED_RATE, seamless_mm=SEAMLESS_MM):
    """
    Convert ordered mm-paths to G-code lines.

    Enhancement: if the pen is already within seamless_mm of the next path's
    start point (pen already down), emit only G1 continuation commands —
    skipping the G0 Z1 -> G0 XY -> G1 Z0 sequence entirely.

    Returns: (gcode_lines, pen_lift_count, total_g0_travel_mm)
    """
    gcode = [
        "G21 (programming in millimeters, mm)",
        "G90 (programming in absolute positioning)",
        "G28 (auto homing)",
        f"G1 F{feed_rate}",
        "G0 Z1 ; Initial pen raise",
    ]

    pen_lifts = 0
    travel_mm = 0.0
    cur_x     = 0.0
    cur_y     = 0.0
    pen_down  = False

    for path in paths:
        if len(path) < 2:
            continue

        start          = path[0]
        dist_to_start  = math.hypot(float(start[0]) - cur_x,
                                    float(start[1]) - cur_y)

        # Seamless: pen already down and close enough — no lift needed.
        if pen_down and dist_to_start <= seamless_mm:
            for pt in path:
                gcode.append(f"G1 X{float(pt[0]):.3f} Y{float(pt[1]):.3f}")
            cur_x, cur_y = float(path[-1][0]), float(path[-1][1])
            continue

        # Normal sequence: lift (if down) -> rapid -> lower -> draw.
        if pen_down:
            gcode.append("G0 Z1 ; Pen raise")
            pen_lifts += 1
            pen_down   = False

        travel_mm += dist_to_start
        gcode.append(f"G0 X{float(start[0]):.3f} Y{float(start[1]):.3f}")
        gcode.append("G1 Z0 ; Pen lower")
        pen_down = True
        cur_x, cur_y = float(start[0]), float(start[1])

        for pt in path[1:]:
            gcode.append(f"G1 X{float(pt[0]):.3f} Y{float(pt[1]):.3f}")
        cur_x, cur_y = float(path[-1][0]), float(path[-1][1])

    if pen_down:
        gcode.append("G0 Z1 ; Final pen raise")
        pen_lifts += 1

    return gcode, pen_lifts, travel_mm


def _path_length_mm(path):
    if len(path) < 2:
        return 0.0
    diffs = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _clone_candidate_list(candidates):
    return [_candidate_with_points(candidate, candidate.points.copy())
            for candidate in (candidates or [])]


def _page_mm_per_pixel(canvas_size):
    width, height = _normalize_canvas_size(canvas_size)
    span_w = max(1, width - 1)
    span_h = max(1, height - 1)
    usable_w = max(1.0, PAGE_MAX_X - 2.0 * PAGE_SAFE_MARGIN_MM)
    usable_h = max(1.0, PAGE_MAX_Y - 2.0 * PAGE_SAFE_MARGIN_MM)
    return min(usable_w / span_w, usable_h / span_h)


def make_brush_mask(points_mm, radius_mm, canvas_size):
    width, height = _normalize_canvas_size(canvas_size)
    mask = np.zeros((height, width), dtype=np.uint8)
    if not points_mm:
        return mask
    mm_per_px = max(1e-6, _page_mm_per_pixel((width, height)))
    radius_px = max(1, int(round(float(radius_mm) / mm_per_px)))
    points_px = [
        _fit_page_to_pixel(float(point[0]), float(point[1]), width, height)
        for point in points_mm
    ]
    if len(points_px) == 1:
        cv2.circle(mask, points_px[0], radius_px, 255,
                   thickness=-1, lineType=cv2.LINE_AA)
    else:
        for p0, p1 in zip(points_px[:-1], points_px[1:]):
            cv2.line(mask, p0, p1, 255, thickness=radius_px * 2,
                     lineType=cv2.LINE_AA)
        for point in (points_px[0], points_px[-1]):
            cv2.circle(mask, point, radius_px, 255,
                       thickness=-1, lineType=cv2.LINE_AA)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _mask_bbox(mask, pad_px=2):
    arr = np.asarray(mask)
    rows, cols = np.where(arr > 0)
    if rows.size == 0:
        return None
    height, width = arr.shape
    x0 = max(0, int(cols.min()) - int(pad_px))
    x1 = min(width, int(cols.max()) + int(pad_px) + 1)
    y0 = max(0, int(rows.min()) - int(pad_px))
    y1 = min(height, int(rows.max()) + int(pad_px) + 1)
    return x0, y0, x1, y1


def _manual_sensitive_lineart(pil_img, brush_mask):
    gray = np.asarray(pil_img.convert("L"))
    height, width = gray.shape
    brush = _resize_binary_mask(brush_mask, width, height)
    if brush is None or cv2.countNonZero(brush) == 0:
        return Image.fromarray(np.full((height, width), 255, dtype=np.uint8))

    bbox = _mask_bbox(brush, pad_px=max(4, int(min(width, height) * 0.015)))
    if bbox is None:
        return Image.fromarray(np.full((height, width), 255, dtype=np.uint8))
    x0, y0, x1, y1 = bbox
    crop = gray[y0:y1, x0:x1]
    crop_mask = brush[y0:y1, x0:x1]
    if crop.size == 0:
        return Image.fromarray(np.full((height, width), 255, dtype=np.uint8))

    kernel_size = max(21, int(min(crop.shape) * 0.18) | 1)
    background = cv2.morphologyEx(
        crop,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
    )
    normalized = np.clip(
        crop.astype(np.float32) / (background.astype(np.float32) + 1e-3) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    smooth = cv2.bilateralFilter(normalized, d=5, sigmaColor=38, sigmaSpace=38)
    block = max(11, int(min(crop.shape) * 0.12) | 1)
    adaptive = cv2.adaptiveThreshold(
        smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, C=2)
    edge = cv2.Canny(smooth, 18, 70, L2gradient=True)
    ink_crop = cv2.bitwise_or(adaptive, edge)
    ink_crop = cv2.bitwise_and(ink_crop, crop_mask)
    ink_crop = cv2.morphologyEx(
        ink_crop,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
        iterations=1,
    )
    ink_crop = _adaptive_component_cleanup(ink_crop, min_area=2)
    ink_crop = _thin_binary(ink_crop)

    ink = np.zeros((height, width), dtype=np.uint8)
    ink[y0:y1, x0:x1] = ink_crop
    result = np.full((height, width), 255, dtype=np.uint8)
    result[ink > 0] = 0
    return Image.fromarray(result)


def extract_manual_detail_candidates(pil_img, brush_mask, face_mask=None,
                                     max_new=MANUAL_ADD_DETAIL_MAX_NEW,
                                     overlap_threshold=MANUAL_ADD_OVERLAP_THRESHOLD,
                                     return_stats=False):
    if pil_img is None:
        stats = {"seen": 0, "kept": 0, "low_overlap": 0, "threshold": overlap_threshold}
        return ([], stats) if return_stats else []
    lineart = _manual_sensitive_lineart(pil_img, brush_mask)
    candidates = extract_candidate_paths(
        lineart, reference_img=pil_img, ai_maps=None, face_mask=face_mask)
    canvas_size = lineart.size
    kept = []
    low_overlap = 0
    for candidate in candidates:
        sample_overlap = _candidate_mask_overlap(
            candidate, brush_mask, canvas_size, samples_per_segment=6)
        bbox_overlap = _candidate_bbox_mask_overlap(
            candidate, brush_mask, canvas_size, pad_px=3)
        overlap = max(sample_overlap, bbox_overlap)
        if overlap < float(overlap_threshold):
            if overlap > 0.0:
                low_overlap += 1
            continue
        adjusted = _candidate_with_points(candidate, candidate.points.copy(),
                                          source="manual_add")
        adjusted.importance = max(adjusted.importance, 0.74)
        adjusted.saliency = max(adjusted.saliency, 0.70)
        if adjusted.region not in {"face", "garment_outline"}:
            adjusted.region = "garment_detail"
            adjusted.detail_tier = min(adjusted.detail_tier, 1)
        adjusted.classifier_scores["manual_add_overlap"] = overlap
        kept.append(adjusted)
    kept.sort(key=_candidate_stroke_count_score, reverse=True)
    if len(kept) > max_new:
        kept = kept[:max_new]
    stats = {
        "seen": len(candidates),
        "kept": len(kept),
        "low_overlap": int(low_overlap),
        "threshold": float(overlap_threshold),
    }
    print(
        f"Manual add: extracted {len(kept)} candidate strokes "
        f"(seen={len(candidates)}, low_overlap={low_overlap}, "
        f"threshold={float(overlap_threshold):.2f})"
    )
    return (kept, stats) if return_stats else kept


def brush_overlaps_face(brush_mask, face_mask,
                        threshold=MANUAL_REDUCE_FACE_WARN_RATIO):
    if brush_mask is None or face_mask is None:
        return False
    brush = np.asarray(brush_mask, dtype=np.uint8)
    if brush.ndim != 2 or cv2.countNonZero(brush) == 0:
        return False
    face = _resize_binary_mask(face_mask, brush.shape[1], brush.shape[0])
    if face is None or cv2.countNonZero(face) == 0:
        return False
    overlap = cv2.bitwise_and(brush, face)
    ratio = cv2.countNonZero(overlap) / max(1, cv2.countNonZero(brush))
    return ratio >= float(threshold)


def apply_manual_brush_adjustment(candidates, pil_img, brush_mask, mode,
                                  face_mask=None, allow_face_reduce=False,
                                  aggressiveness="normal"):
    pool = _clone_candidate_list(candidates)
    brush_arr = np.asarray(brush_mask, dtype=np.uint8)
    if brush_arr.ndim != 2 or cv2.countNonZero(brush_arr) == 0:
        return pool, {
            "mode": mode, "added": 0, "removed": 0, "merged": 0,
            "diagnostics": {"overlapped": 0},
        }

    mode = "reduce" if str(mode).lower().startswith("reduce") else "add"
    canvas_size = (brush_arr.shape[1], brush_arr.shape[0])
    if mode == "add":
        added, add_stats = extract_manual_detail_candidates(
            pil_img, brush_arr, face_mask=face_mask, return_stats=True)
        combined = pool + added
        combined, dedup_removed = _deduplicate_candidates(combined)
        context = StrokeClassificationContext(
            canvas_size=canvas_size,
            reference_img=pil_img,
            lineart_img=None,
            face_mask=face_mask,
        )
        combined = classify_strokes(combined, context)
        return combined, {
            "mode": mode,
            "added": len(added),
            "removed": int(dedup_removed),
            "merged": 0,
            "diagnostics": add_stats,
        }

    strong = str(aggressiveness).lower() in {"strong", "high", "aggressive"}
    overlap_threshold = (
        MANUAL_REDUCE_OVERLAP_STRONG if strong
        else MANUAL_REDUCE_OVERLAP_NORMAL
    )
    soft_overlap_threshold = max(0.10, overlap_threshold * 0.58)
    score_threshold = (
        MANUAL_REDUCE_SCORE_STRONG if strong
        else MANUAL_REDUCE_SCORE_NORMAL
    )
    kept = []
    affected = []
    removed = 0
    diagnostics = {
        "overlapped": 0,
        "protected_face": 0,
        "protected_outline": 0,
        "low_overlap": 0,
        "score_high": 0,
        "removed_high_overlap": 0,
        "removed_soft": 0,
        "threshold": float(overlap_threshold),
        "score_threshold": float(score_threshold),
        "aggressiveness": "strong" if strong else "normal",
    }
    for candidate in pool:
        overlap = _candidate_mask_overlap(
            candidate, brush_arr, canvas_size, samples_per_segment=6)
        if overlap <= 0.0:
            kept.append(candidate)
            continue
        diagnostics["overlapped"] += 1
        face_overlap = (
            _candidate_mask_overlap(candidate, face_mask, canvas_size)
            if face_mask is not None else 0.0
        )
        is_face = getattr(candidate, "region", "") == "face" or face_overlap >= 0.18
        if is_face and not allow_face_reduce:
            diagnostics["protected_face"] += 1
            kept.append(candidate)
            continue
        is_primary_outline = (
            getattr(candidate, "region", "") == "garment_outline" and
            int(getattr(candidate, "detail_tier", 1)) == 0
        )
        if is_primary_outline:
            diagnostics["protected_outline"] += 1
            kept.append(candidate)
            continue
        if overlap < soft_overlap_threshold:
            diagnostics["low_overlap"] += 1
            kept.append(candidate)
            continue

        score = _candidate_stroke_count_score(candidate)
        high_overlap_remove = overlap >= overlap_threshold
        soft_remove = (
            overlap >= soft_overlap_threshold and
            score <= score_threshold and
            (
                getattr(candidate, "source", "") == "hatch" or
                getattr(candidate, "region", "") in {
                    "hatch", "fine_detail", "garment_detail", "detail"
                } or
                int(getattr(candidate, "detail_tier", 1)) >= 1 or
                (allow_face_reduce and is_face)
            )
        )
        if high_overlap_remove or soft_remove:
            removed += 1
            if high_overlap_remove:
                diagnostics["removed_high_overlap"] += 1
            else:
                diagnostics["removed_soft"] += 1
            continue
        diagnostics["score_high"] += 1
        affected.append(candidate)

    merged_count = 0
    if len(affected) >= 2:
        merged_paths, stats = merge_collinear_and_touching_strokes(
            [candidate.points for candidate in affected],
            angle_tolerance_deg=18.0,
            endpoint_gap_mm=1.05,
            return_stats=True,
        )
        merged_count = int(stats.get("merged_count", 0))
        template = max(affected, key=_candidate_stroke_count_score)
        for path in merged_paths:
            merged = _candidate_with_points(template, path, source="manual_reduce")
            merged.importance = max(0.45, min(template.importance, 0.88))
            merged.saliency = max(0.40, min(template.saliency, 0.82))
            if merged.region == "face" and not allow_face_reduce:
                merged.protected = True
            kept.append(merged)
    else:
        kept.extend(affected)

    print(
        f"Manual reduce: removed {removed}, locally merged {merged_count}, "
        f"remaining {len(kept)}, diagnostics={diagnostics}"
    )
    return kept, {
        "mode": mode,
        "added": 0,
        "removed": removed,
        "merged": merged_count,
        "diagnostics": diagnostics,
    }


def _resample_polyline_limited(path, target_mm=RESAMPLE_TARGET_MM,
                               max_segment_mm=RESAMPLE_MAX_SEGMENT_MM):
    """
    Chi chen diem tai cac moc tang/giam toc. Dinh hinh hoc ban dau duoc giu nguyen,
    nen doan thang dai khong bi chia deu thanh hang tram lenh G-code.
    """
    if len(path) < 2:
        return path, 0
    del target_mm
    max_segment_mm = max(0.2, float(max_segment_mm))
    cumulative = _stroke_cumulative_lengths(path)
    total_len = float(cumulative[-1])
    if total_len <= 2.0 * ACCEL_DISTANCE_MM or RAMP_SEGMENTS <= 0:
        return path, 0

    ramp_step = ACCEL_DISTANCE_MM / float(RAMP_SEGMENTS)
    events = sorted({
        distance
        for step in range(1, RAMP_SEGMENTS + 1)
        for distance in (step * ramp_step, total_len - step * ramp_step)
        if 1e-6 < distance < total_len - 1e-6
    })
    out = [path[0]]
    added = 0
    event_index = 0
    for seg_index, (p0, p1) in enumerate(zip(path[:-1], path[1:])):
        seg_len = float(np.linalg.norm(p1 - p0))
        seg_start = float(cumulative[seg_index])
        seg_end = float(cumulative[seg_index + 1])
        while event_index < len(events) and events[event_index] <= seg_start + 1e-6:
            event_index += 1
        while event_index < len(events) and events[event_index] < seg_end - 1e-6:
            event_distance = events[event_index]
            if seg_len > max_segment_mm:
                t = (event_distance - seg_start) / max(seg_len, 1e-9)
                out.append((p0 * (1.0 - t) + p1 * t).astype(np.float32))
                added += 1
            event_index += 1
        out.append(p1)
    return np.asarray(out, dtype=np.float32), added


def _resample_paths_limited(paths, max_segments=None):
    """
    Resample stroke sau compression, truoc khi sinh G-code.
    Gioi han extra/total points giup UI khong cham voi anh qua phuc tap.
    """
    total_points = sum(len(p) for p in paths)
    if total_points >= RESAMPLE_MAX_TOTAL_POINTS:
        return paths, 0
    current_segments = _count_segments(paths)
    if max_segments is not None and current_segments >= max_segments:
        return paths, 0
    out = []
    total_added = 0
    for path in paths:
        if total_added >= RESAMPLE_MAX_EXTRA_POINTS:
            out.append(path)
            continue
        q, added = _resample_polyline_limited(path)
        if max_segments is not None and current_segments + total_added + added > max_segments:
            out.append(path)
            continue
        if total_points + total_added + added > RESAMPLE_MAX_TOTAL_POINTS:
            out.append(path)
            continue
        if total_added + added > RESAMPLE_MAX_EXTRA_POINTS:
            out.append(path)
            continue
        out.append(q if len(q) >= 2 else path)
        total_added += added
    if total_added:
        print(f"  Segment resampling: added {total_added} points")
    return out, total_added


def _smooth_polyline_preserve_corners(path, weight=SMOOTHING_WEIGHT,
                                      keep_turn_deg=SMOOTHING_TURN_KEEP_DEG):
    """
    B-spline smoothing nhe cho polyline: lam mem diem noi bo de bot rung/gay khuc.
    Neu diem la goc nhon thi giu nguyen de khong lam mat form cua line-art.
    """
    if len(path) < 5:
        return path
    out = path.copy()
    for i in range(1, len(path) - 1):
        v1 = path[i] - path[i - 1]
        v2 = path[i + 1] - path[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 1e-6 or n2 <= 1e-6:
            continue
        cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
        turn_deg = math.degrees(math.acos(cos_a))
        if turn_deg >= keep_turn_deg:
            continue
        spline_point = (path[i - 1] + 4.0 * path[i] + path[i + 1]) / 6.0
        out[i] = path[i] * (1.0 - weight) + spline_point * weight
    return out.astype(np.float32)


def _smooth_paths(paths):
    """Ap dung smoothing co gioi han de UI khong cham voi anh qua phuc tap."""
    total_points = sum(len(p) for p in paths)
    if total_points > SMOOTHING_MAX_POINTS:
        return paths
    smoothed = []
    for p in paths:
        q = _smooth_polyline_preserve_corners(p)
        if len(q) >= 2:
            smoothed.append(q)
    return smoothed


def _stroke_cumulative_lengths(path):
    """Tinh quang duong tich luy tren stroke de ramp feedrate theo mm that."""
    if len(path) < 2:
        return np.array([0.0], dtype=np.float32)
    seg_lens = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lens))).astype(np.float32)


def _segment_draw_feed(path, seg_index, base_feed=DRAW_FEED_RATE,
                       accel_distance_mm=ACCEL_DISTANCE_MM,
                       min_feed_rate=MIN_DRAW_FEED_RATE,
                       max_feed_rate=None, cumulative_lengths=None):
    """
    Trapezoidal velocity ramp cho tung doan ve.
    Stroke du dai se co 3 pha: tang toc, cruise, giam toc. Stroke ngan hon
    2*accel_distance_mm giu feedrate co dinh de G-code khong bi roi/qua day lenh.
    """
    if max_feed_rate is None:
        max_feed_rate = base_feed
    max_feed_rate = int(max(1, min(int(max_feed_rate), int(base_feed))))
    min_feed_rate = int(max(1, min(int(min_feed_rate), max_feed_rate)))
    accel_distance_mm = max(0.0, float(accel_distance_mm))

    cum = (cumulative_lengths if cumulative_lengths is not None
           else _stroke_cumulative_lengths(path))
    total_len = float(cum[-1]) if len(cum) else 0.0
    if total_len <= 1e-6 or accel_distance_mm <= 1e-6:
        return max_feed_rate
    if total_len <= 2.0 * accel_distance_mm:
        return max_feed_rate

    seg_index = int(max(0, min(len(path) - 2, seg_index)))
    seg_mid = float((cum[seg_index] + cum[seg_index + 1]) * 0.5)

    # He so ramp = 0 o dau/cuoi stroke, =1 o phan cruise giua stroke.
    ramp_t = min(1.0, seg_mid / accel_distance_mm,
                 (total_len - seg_mid) / accel_distance_mm)
    ramp_t = max(0.0, ramp_t)
    feed = min_feed_rate + (max_feed_rate - min_feed_rate) * ramp_t

    curve_factor = 1.0
    # Corner look-ahead: giam toc truoc/sau goc nhon trong vai segment gan ke.
    for j in range(max(1, seg_index - CORNER_LOOKAHEAD_SEGMENTS),
                   min(len(path) - 1, seg_index + CORNER_LOOKAHEAD_SEGMENTS + 1)):
        v1 = path[j] - path[j - 1]
        v2 = path[j + 1] - path[j]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 > 1e-6 and n2 > 1e-6:
            cos_a = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
            turn_deg = math.degrees(math.acos(cos_a))
            if turn_deg > CORNER_SLOWDOWN_DEG:
                distance_from_corner = abs(j - seg_index)
                influence = max(0.0, 1.0 - distance_from_corner / (CORNER_LOOKAHEAD_SEGMENTS + 1.0))
                local_factor = 1.0 - (turn_deg / 180.0) * influence
                curve_factor = min(curve_factor, max(MIN_DRAW_FEED_FACTOR, local_factor))

    # Van giam toc tai goc gap de han che rung, nhung khong thap hon min_feed_rate.
    quantized = round((feed * curve_factor) / DRAW_FEED_QUANTUM) * DRAW_FEED_QUANTUM
    return int(np.clip(quantized, min_feed_rate, max_feed_rate))


def _validate_paths_bounds(paths):
    errors = []
    for pi, path in enumerate(paths):
        if len(path) < 2:
            continue
        if not np.all(np.isfinite(path)):
            errors.append(f"Path {pi} contains NaN/Infinity")
            continue
        xs = path[:, 0]
        ys = path[:, 1]
        if np.any(xs < -1e-4) or np.any(xs > PAGE_MAX_X + 1e-4) or \
           np.any(ys < -1e-4) or np.any(ys > PAGE_MAX_Y + 1e-4):
            errors.append(
                f"Path {pi} vuot bien trang: "
                f"x=[{float(xs.min()):.2f},{float(xs.max()):.2f}], "
                f"y=[{float(ys.min()):.2f},{float(ys.max()):.2f}]")
    return errors


def _postprocess_gcode_lines(gcode_lines, min_move_mm=GCODE_MIN_MOVE_MM,
                             decimals=GCODE_DECIMALS):
    """
    G-code post-processor: bo XY move qua ngan va loai F lap lai.
    Muc tieu la file gon hon, may chay muot hon, nhung khong doi thu tu logic Z.
    """
    optimized = []
    removed = 0
    last_feed = None
    x = 0.0
    y = 0.0
    pen_down = False
    for raw in gcode_lines:
        line = raw.strip()
        clean = line.split(";", 1)[0].split("(", 1)[0].strip().upper()
        if not clean:
            optimized.append(raw)
            continue
        values = {m.group(1).upper(): float(m.group(2)) for m in _AXIS_RE.finditer(clean)}
        has_xy = "X" in values or "Y" in values
        has_z = "Z" in values
        is_motion = clean.startswith("G0") or clean.startswith("G1")
        if has_z:
            pen_down = values["Z"] <= Z_DRAW + 0.05

        if is_motion and has_xy and not has_z:
            nx = values.get("X", x)
            ny = values.get("Y", y)
            is_draw_move = clean.startswith("G1") and pen_down
            if not is_draw_move and math.hypot(nx - x, ny - y) < min_move_mm:
                removed += 1
                x, y = nx, ny
                continue
            x, y = nx, ny

        if is_motion and "F" in clean:
            feed_match = re.search(r"\sF(\d+(?:\.\d+)?)", line, re.IGNORECASE)
            if feed_match:
                feed_value = float(feed_match.group(1))
                if last_feed is not None and abs(feed_value - last_feed) <= 1e-9:
                    line = re.sub(r"\sF\d+(?:\.\d+)?", "", line, count=1, flags=re.IGNORECASE)
                    removed += 1
                else:
                    last_feed = feed_value

        if is_motion and has_xy:
            for axis in ("X", "Y"):
                if axis in values:
                    line = re.sub(
                        rf"{axis}-?\d+(?:\.\d+)?",
                        f"{axis}{values[axis]:.{decimals}f}",
                        line,
                        count=1,
                        flags=re.IGNORECASE,
                    )
        optimized.append(line)

    if removed:
        print(f"  G-code postprocess: removed/compacted {removed} commands/fields")
    return optimized, removed


def _dry_run_validate_gcode(gcode_lines):
    """
    Dry-run simulator nhe: doc lai G-code de kiem tra bien giay, trang thai but,
    feedrate min/max va bounding box truoc khi export.
    """
    errors = []
    x = 0.0
    y = 0.0
    z = Z_TRAVEL
    pen_down = False
    feed = None
    draw_mm = 0.0
    travel_mm = 0.0
    motion_time_s = 0.0
    z_move_count = 0
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for ln, raw in enumerate(gcode_lines, 1):
        clean = raw.split(";", 1)[0].split("(", 1)[0].strip().upper()
        if not clean:
            continue
        values = {m.group(1).upper(): float(m.group(2)) for m in _AXIS_RE.finditer(clean)}
        feed_match = re.search(r"\bF\s*(-?\d+(?:\.\d+)?)", clean, re.IGNORECASE)
        if feed_match:
            feed = float(feed_match.group(1))
            if feed <= 0:
                errors.append(f"Line {ln}: feedrate khong hop le F{feed:g}")

        if "Z" in values:
            next_z = values["Z"]
            if abs(next_z - z) > 1e-6:
                z_move_count += 1
            z = next_z
            pen_down = z <= Z_DRAW + 0.05

        has_xy = "X" in values or "Y" in values
        if not has_xy:
            continue

        nx = values.get("X", x)
        ny = values.get("Y", y)
        if nx < -DRY_RUN_BOUNDS_TOL_MM or nx > PAGE_MAX_X + DRY_RUN_BOUNDS_TOL_MM or \
           ny < -DRY_RUN_BOUNDS_TOL_MM or ny > PAGE_MAX_Y + DRY_RUN_BOUNDS_TOL_MM:
            errors.append(f"Line {ln}: XY vuot bien ({nx:.3f}, {ny:.3f})")

        dist = math.hypot(nx - x, ny - y)
        if dist > 1e-9:
            if feed is None or feed <= 0:
                errors.append(f"Line {ln}: XY move without valid modal feed")
            else:
                motion_time_s += dist / feed * 60.0
        if clean.startswith("G0") and pen_down and dist > 1e-6:
            errors.append(f"Line {ln}: G0 XY khi but dang ha")
        if clean.startswith("G1") and not pen_down and dist > 1e-6:
            errors.append(f"Line {ln}: G1 XY khi but chua ha")
        if clean.startswith("G1") and pen_down:
            draw_mm += dist
        else:
            travel_mm += dist

        min_x = min(min_x, nx)
        min_y = min(min_y, ny)
        max_x = max(max_x, nx)
        max_y = max(max_y, ny)
        x, y = nx, ny

    bbox = None if min_x == float("inf") else (min_x, min_y, max_x, max_y)
    stats = {
        "draw_mm": draw_mm,
        "travel_mm": travel_mm,
        "bbox": bbox,
        "last_feed": feed,
        "motion_time_s": motion_time_s,
        "z_move_count": z_move_count,
    }
    if errors:
        print(f"  Dry-run validator: {len(errors)} warnings/errors")
    return errors, stats


def _validate_gcode_motion(gcode_lines):
    """Kiem tra khong ve khi but chua ha, va khong G0 XY khi but dang ha."""
    errors = []
    pen_down = False
    for ln, raw in enumerate(gcode_lines, 1):
        line = raw.split(";", 1)[0].split("(", 1)[0].strip().upper()
        if not line:
            continue
        values = {m.group(1).upper(): float(m.group(2))
                  for m in _AXIS_RE.finditer(line)}
        if "Z" in values:
            pen_down = values["Z"] <= Z_DRAW + 0.05
        has_xy = "X" in values or "Y" in values
        if line.startswith("G1") and has_xy and not pen_down:
            errors.append(f"Line {ln}: G1 XY khi but chua ha")
        if line.startswith("G0") and has_xy and pen_down:
            errors.append(f"Line {ln}: G0 XY khi but dang ha")
    return errors


def _format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _estimate_time_s(draw_mm, travel_mm, pen_lifts,
                     draw_feed=DRAW_FEED_RATE,
                     travel_feed=TRAVEL_FEED_RATE):
    draw_s = draw_mm / max(1.0, draw_feed) * 60.0
    travel_s = travel_mm / max(1.0, travel_feed) * 60.0
    z_s = pen_lifts * PEN_Z_MOVE_TIME_S * 2.0
    return draw_s + travel_s + z_s


def _estimate_path_pen_lifts(paths, seamless_mm=SEAMLESS_MM):
    """Count Z-lift cycles the current G-code writer would need for path gaps."""
    valid_paths = [path for path in paths if len(path) >= 2]
    if not valid_paths:
        return 0
    lifts = 1  # Final raise after the last continuous stroke.
    for previous, current in zip(valid_paths[:-1], valid_paths[1:]):
        gap = float(np.linalg.norm(current[0] - previous[-1]))
        if gap > seamless_mm:
            lifts += 1
    return lifts


def _add_gcode_header(gcode, stats):
    """Chen metadata vao dau file G-code de de debug va log lich su ve."""
    header = [
        "; Sketch-to-GCode professional output",
        f"; Created: {datetime.now().isoformat(timespec='seconds')}",
        f"; Page: {PAGE_MAX_X:.1f} x {PAGE_MAX_Y:.1f} mm",
        f"; Draw feed: F{stats['draw_feed']} mm/min",
        f"; Travel feed: F{stats['travel_feed']} mm/min",
        f"; Ramp accel distance: {stats['accel_distance_mm']:.1f} mm",
        f"; Ramp min/max feed: F{stats['min_feed_rate']} / F{stats['max_feed_rate']}",
        f"; Z draw/hover/travel: {Z_DRAW:.2f}/{Z_HOVER:.2f}/{Z_TRAVEL:.2f}",
        f"; Strokes: {stats['stroke_count']}",
        f"; Pen lifts: {stats['pen_lifts']}",
        f"; Draw distance: {stats['draw_mm']:.1f} mm",
        f"; Travel distance: {stats['travel_mm']:.1f} mm",
        f"; Estimated time: {_format_duration(stats['estimated_time_s'])}",
        f"; G-code postprocess removed/compacted: {stats['postprocess_removed']}",
        f"; Dry-run warnings: {stats['dry_run_warning_count']}",
        f"; Commands: {len(gcode)}",
        ";",
    ]
    return header + gcode


def paths_to_gcode(paths, draw_feed=DRAW_FEED_RATE, travel_feed=TRAVEL_FEED_RATE,
                   seamless_mm=SEAMLESS_MM,
                   accel_distance_mm=ACCEL_DISTANCE_MM,
                   min_feed_rate=MIN_DRAW_FEED_RATE,
                   max_feed_rate=MAX_DRAW_FEED_RATE):
    """
    Chuyen path da sap thu tu sang G-code chuyen nghiep.
    Co feedrate rieng, ramp toc do, Z hover/travel, validate va metadata.
    """
    validation_errors = _validate_paths_bounds(paths)
    gcode = [
        "G21 ; mm",
        "G90 ; absolute positioning",
        "G28 ; auto homing",
        f"G0 F{int(travel_feed)}",
        f"G0 Z{Z_TRAVEL:.3f} ; initial full pen raise",
    ]

    pen_lifts = 0
    travel_mm = 0.0
    draw_mm = 0.0
    cur_x = 0.0
    cur_y = 0.0
    cur_z = Z_TRAVEL
    pen_down = False

    for path in paths:
        if len(path) < 2:
            continue

        cumulative_lengths = _stroke_cumulative_lengths(path)

        start = path[0]
        dist_to_start = math.hypot(float(start[0]) - cur_x,
                                   float(start[1]) - cur_y)

        if pen_down and dist_to_start <= seamless_mm:
            for i, pt in enumerate(path):
                feed = _segment_draw_feed(
                    path, max(0, i - 1), draw_feed,
                    accel_distance_mm, min_feed_rate, max_feed_rate,
                    cumulative_lengths)
                gcode.append(
                    f"G1 X{float(pt[0]):.3f} Y{float(pt[1]):.3f} F{feed}")
            draw_mm += _path_length_mm(path)
            cur_x, cur_y = float(path[-1][0]), float(path[-1][1])
            continue

        if pen_down:
            lift_z = Z_HOVER if dist_to_start <= HOVER_TRAVEL_MM else Z_TRAVEL
            gcode.append(f"G0 Z{lift_z:.3f} F{int(travel_feed)} ; pen lift")
            pen_lifts += 1
            cur_z = lift_z
            pen_down = False

        if cur_z < Z_HOVER - 1e-6:
            gcode.append(f"G0 Z{Z_HOVER:.3f} F{int(travel_feed)} ; hover")
            cur_z = Z_HOVER

        travel_mm += dist_to_start
        gcode.append(
            f"G0 X{float(start[0]):.3f} Y{float(start[1]):.3f} F{int(travel_feed)}")
        # Keep feed-only and Z commands separate for gcode_to_robot compatibility.
        gcode.append(f"G1 F{int(draw_feed)} ; drawing feed")
        gcode.append(f"G1 Z{Z_DRAW:.3f} ; pen lower")
        pen_down = True
        cur_z = Z_DRAW
        cur_x, cur_y = float(start[0]), float(start[1])

        for i, pt in enumerate(path[1:], start=0):
            feed = _segment_draw_feed(
                path, i, draw_feed, accel_distance_mm,
                min_feed_rate, max_feed_rate, cumulative_lengths)
            gcode.append(
                f"G1 X{float(pt[0]):.3f} Y{float(pt[1]):.3f} F{feed}")
        draw_mm += _path_length_mm(path)
        cur_x, cur_y = float(path[-1][0]), float(path[-1][1])

    if pen_down:
        gcode.append(f"G0 Z{Z_TRAVEL:.3f} F{int(travel_feed)} ; final pen raise")
        pen_lifts += 1

    gcode, post_removed = _postprocess_gcode_lines(gcode)
    dry_errors, dry_stats = _dry_run_validate_gcode(gcode)
    validation_errors.extend(_validate_gcode_motion(gcode))
    validation_errors.extend(dry_errors)
    estimated_time_s = (
        float(dry_stats.get("motion_time_s", 0.0)) +
        int(dry_stats.get("z_move_count", 0)) * PEN_Z_MOVE_TIME_S
    )
    stats = {
        "draw_feed": int(draw_feed),
        "travel_feed": int(travel_feed),
        "stroke_count": len(paths),
        "pen_lifts": pen_lifts,
        "draw_mm": draw_mm,
        "travel_mm": travel_mm,
        "estimated_time_s": estimated_time_s,
        "accel_distance_mm": float(accel_distance_mm),
        "min_feed_rate": int(min_feed_rate),
        "max_feed_rate": int(max_feed_rate),
        "postprocess_removed": post_removed,
        "dry_run_warning_count": len(dry_errors),
        "dry_run_bbox": dry_stats.get("bbox"),
    }
    gcode = _add_gcode_header(gcode, stats)
    return gcode, pen_lifts, travel_mm, draw_mm, estimated_time_s, validation_errors


# ═══════════════════════════════════════════════════════════════════
#  BUILD G-CODE PLAN  (ORCHESTRATOR)
# ═══════════════════════════════════════════════════════════════════

def build_gcode_plan(
        candidates,
        target_segments,
        merge_angle_tolerance_deg=STROKE_MERGE_ANGLE_TOLERANCE_DEG,
        merge_endpoint_gap_mm=STROKE_MERGE_ENDPOINT_GAP_MM):
    """
    Full G-code pipeline:
      1. Micro-stroke filtering
      2. RDP simplification / budget fitting
      3. Greedy NN path ordering (angle-priority)
      4. Smart gap-bridging (path stitching)
      5. G-code generation with seamless G1 continuation
    """
    planning_started = time.perf_counter()
    target = int(max(MIN_STROKE_BUDGET, min(MAX_STROKE_BUDGET, target_segments)))

    filtered              = _filter_micro_strokes(candidates)
    filtered, dedup_removed = _deduplicate_candidates(filtered)
    simplified, epsilon, raw_segments, tier_counts, rdp_eps_warning = \
        _fit_paths_to_budget(filtered, target)
    simplified            = [p for p in simplified if len(p) >= 2]
    clipped, clipped_count = _clip_paths_to_page(simplified)
    smoothed              = _smooth_paths(clipped)
    preordered, _ = _optimize_paths_with_islands(smoothed)
    preordered, pre_direction_flips = _optimize_stroke_directions(preordered)
    strokes_before_stitch = len(preordered)
    lifts_before_stitch = _estimate_path_pen_lifts(preordered)
    draw_before_stitch = sum(_path_length_mm(path) for path in preordered)
    avg_draw_before = draw_before_stitch / max(1, strokes_before_stitch)
    stitched, stitch_count = _stitch_paths(preordered, segment_cap=target)
    stitched, pen_lift_bridge_count = _bridge_short_pen_lift_gaps(
        stitched, segment_cap=target)
    merged, merge_stats = merge_collinear_and_touching_strokes(
        stitched,
        angle_tolerance_deg=merge_angle_tolerance_deg,
        endpoint_gap_mm=merge_endpoint_gap_mm,
        return_stats=True)
    ordered, island_count = _optimize_paths_with_islands(merged)
    ordered, route_saved_mm, route_saved_pct = _postprocess_route_improve(ordered)
    ordered, final_direction_flips = _optimize_stroke_directions(ordered)
    direction_flips = pre_direction_flips + final_direction_flips
    compressed, before_compress, actual_segments, removed_segments = \
        _compress_paths_collinear(ordered)
    resample_segment_cap = min(MAX_STROKE_BUDGET, max(target, actual_segments))
    compressed, resampled_added = _resample_paths_limited(compressed, resample_segment_cap)
    actual_segments = _count_segments(compressed)

    stroke_count    = len(compressed)
    gcode_lines, pen_lifts, travel_mm, draw_mm, estimated_time_s, validation_errors = \
        paths_to_gcode(compressed)
    avg_draw_after = draw_mm / max(1, stroke_count)
    stroke_reduction_pct = 100.0 * (
        strokes_before_stitch - stroke_count) / max(1, strokes_before_stitch)
    lift_reduction_pct = 100.0 * (
        lifts_before_stitch - pen_lifts) / max(1, lifts_before_stitch)
    print(
        f"  Pen-lift comparison: strokes {strokes_before_stitch} -> {stroke_count} "
        f"(-{stroke_reduction_pct:.1f}%), lifts {lifts_before_stitch} -> {pen_lifts} "
        f"(-{lift_reduction_pct:.1f}%), avg draw/stroke "
        f"{avg_draw_before:.1f} -> {avg_draw_after:.1f}mm"
    )
    post_removed = 0
    dry_run_warnings = 0
    for ln in gcode_lines:
        if ln.startswith("; G-code postprocess"):
            m = re.search(r"(\d+)", ln)
            post_removed = int(m.group(1)) if m else 0
        if ln.startswith("; Dry-run warnings"):
            m = re.search(r"(\d+)", ln)
            dry_run_warnings = int(m.group(1)) if m else 0
            break
    # Khong raise tai day de preview van hien thi; export se chan bang popup ro rang.

    print(
        f"G-code plan: target<={target}, actual={actual_segments}, "
        f"strokes={stroke_count}, lifts={pen_lifts}, "
        f"stitched={stitch_count}, travel={travel_mm:.1f}mm, "
        f"route_saved={route_saved_mm:.1f}mm ({route_saved_pct:.1f}%), "
        f"dir_flip={direction_flips}, bridge={pen_lift_bridge_count}, "
        f"merge={merge_stats['merged_count']}, overlap={merge_stats['overlap_removed_count']}, "
        f"resample=+{resampled_added}, "
        f"islands={island_count}, dedup={dedup_removed}, clipped={clipped_count}, "
        f"draw={draw_mm:.1f}mm, est={_format_duration(estimated_time_s)}, "
        f"eps={epsilon:.3f}mm, cmds={len(gcode_lines)}"
    )
    travel_draw_ratio = travel_mm / max(1e-6, draw_mm)
    print(
        f"  Plan diagnostics: RDP epsilon={epsilon:.3f}mm "
        f"(warning={'yes' if rdp_eps_warning else 'no'}), "
        f"travel/draw={travel_draw_ratio:.2f}x"
    )

    return GCodePlan(
        gcode_lines=gcode_lines,
        paths=compressed,
        target_segments=target,
        actual_segments=actual_segments,
        stroke_count=stroke_count,
        pen_lifts=pen_lifts,
        command_count=len(gcode_lines),
        raw_segment_count=raw_segments,
        used_epsilon=epsilon,
        stitched_count=stitch_count,
        travel_distance_mm=travel_mm,
        draw_distance_mm=draw_mm,
        estimated_time_s=estimated_time_s,
        travel_feed_rate=TRAVEL_FEED_RATE,
        draw_feed_rate=DRAW_FEED_RATE,
        command_count_before_compression=before_compress,
        compressed_segment_count=actual_segments,
        compression_removed_count=removed_segments,
        route_improvement_mm=route_saved_mm,
        route_improvement_pct=route_saved_pct,
        clipped_path_count=clipped_count,
        deduplicated_path_count=dedup_removed,
        island_count=island_count,
        gcode_postprocess_removed_count=post_removed,
        resampled_point_count=resampled_added,
        stroke_direction_flip_count=direction_flips,
        dry_run_warning_count=dry_run_warnings,
        selected_detail_tier_counts=tier_counts,
        pen_lift_bridge_count=pen_lift_bridge_count,
        planning_time_s=time.perf_counter() - planning_started,
        gcode_size_bytes=len("\n".join(gcode_lines).encode("utf-8")),
        stroke_merge_count=merge_stats["merged_count"],
        overlap_removed_count=merge_stats["overlap_removed_count"],
        stroke_merge_reduction_pct=merge_stats["reduction_pct"],
        rdp_eps_warning=rdp_eps_warning,
        validation_errors=validation_errors,
    )


def generate_gcode(pil_image, target_segments=DEFAULT_STROKE_BUDGET):
    """Compatibility wrapper using the current line-art vector pipeline."""
    candidates = _consolidate_dense_micro_candidates(
        extract_candidate_paths(pil_image))
    return build_gcode_plan(candidates, target_segments).gcode_lines


# ═══════════════════════════════════════════════════════════════════
#  G-CODE PARSER FOR PREVIEW
# ═══════════════════════════════════════════════════════════════════

_AXIS_RE = re.compile(r"([XYZ])\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_gcode_for_preview(gcode_lines):
    drawing_paths   = []
    travel_segments = []
    x = 0.0; y = 0.0; z = 1.0
    pen_down    = False
    active_path = None

    for raw in gcode_lines:
        line = raw.split(";", 1)[0].split("(", 1)[0].strip().upper()
        if not line:
            continue
        values = {m.group(1).upper(): float(m.group(2)) for m in _AXIS_RE.finditer(line)}

        if "Z" in values:
            z            = values["Z"]
            new_pen_down = z <= Z_DRAW + 0.05
            if pen_down and not new_pen_down and active_path and len(active_path) >= 2:
                drawing_paths.append(active_path)
                active_path = None
            pen_down = new_pen_down

        has_xy = ("X" in values) or ("Y" in values)
        if not has_xy:
            continue

        nx  = values.get("X", x)
        ny  = values.get("Y", y)
        old = (x, y)
        new_pos = (nx, ny)

        if old != new_pos:
            if pen_down:
                if active_path is None:
                    active_path = [old, new_pos]
                else:
                    if active_path[-1] != old:
                        active_path.append(old)
                    active_path.append(new_pos)
            else:
                travel_segments.append((old, new_pos))
        x, y = nx, ny

    if active_path and len(active_path) >= 2:
        drawing_paths.append(active_path)

    return drawing_paths, travel_segments


def _fit_page_to_pixel(x, y, width, height):
    span_w = max(1, width - 1)
    span_h = max(1, height - 1)
    usable_w = max(1.0, PAGE_MAX_X - 2.0 * PAGE_SAFE_MARGIN_MM)
    usable_h = max(1.0, PAGE_MAX_Y - 2.0 * PAGE_SAFE_MARGIN_MM)
    scale = min(usable_w / span_w, usable_h / span_h)
    draw_w = span_w * scale
    draw_h = span_h * scale
    offset_x = (PAGE_MAX_X - draw_w) / 2.0
    offset_y = (PAGE_MAX_Y - draw_h) / 2.0
    col = (float(x) - offset_x) / max(1e-6, scale)
    row = span_h - (float(y) - offset_y) / max(1e-6, scale)
    return (
        int(np.clip(round(col), 0, width - 1)),
        int(np.clip(round(row), 0, height - 1)),
    )


def _normalize_canvas_size(canvas_size):
    if isinstance(canvas_size, Image.Image):
        return canvas_size.size
    width, height = canvas_size
    return max(1, int(width)), max(1, int(height))


def _stroke_paths_from_gcode_or_paths(gcode_or_strokes):
    if isinstance(gcode_or_strokes, str):
        lines = gcode_or_strokes.splitlines()
        drawing_paths, _ = parse_gcode_for_preview(lines)
        return [np.asarray(path, dtype=np.float32) for path in drawing_paths]
    if (isinstance(gcode_or_strokes, (list, tuple)) and gcode_or_strokes and
            isinstance(gcode_or_strokes[0], str)):
        drawing_paths, _ = parse_gcode_for_preview(gcode_or_strokes)
        return [np.asarray(path, dtype=np.float32) for path in drawing_paths]

    paths, _ = _coerce_stroke_paths([] if gcode_or_strokes is None else gcode_or_strokes)
    return paths


def render_gcode_preview_to_image(gcode_or_strokes, canvas_size):
    """
    Render final draw strokes into a white bitmap using the same mm-to-page fit
    as vectorization. This is the QA image of what the robot will draw.
    """
    width, height = _normalize_canvas_size(canvas_size)
    canvas = np.full((height, width), 255, dtype=np.uint8)
    for path in _stroke_paths_from_gcode_or_paths(gcode_or_strokes):
        if len(path) < 2:
            continue
        for p0, p1 in zip(path[:-1], path[1:]):
            x0, y0 = _fit_page_to_pixel(p0[0], p0[1], width, height)
            x1, y1 = _fit_page_to_pixel(p1[0], p1[1], width, height)
            cv2.line(canvas, (x0, y0), (x1, y1), 0,
                     thickness=QA_RENDER_LINE_WIDTH_PX, lineType=cv2.LINE_AA)
    return canvas


def _as_gray_array(image, size=None):
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("L"))
    else:
        arr = np.asarray(image)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        arr = arr.astype(np.uint8)
    if size is not None:
        width, height = _normalize_canvas_size(size)
        if arr.shape != (height, width):
            arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_AREA)
    return arr.astype(np.uint8)


def _path_hash(paths):
    digest = hashlib.sha256()
    for path in paths:
        arr = np.round(np.asarray(path, dtype=np.float32), 3)
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _render_plan_preview_cached(plan, canvas_size, cache):
    key = (_path_hash(plan.paths), tuple(_normalize_canvas_size(canvas_size)))
    cached = cache.get(key)
    if cached is not None:
        return cached
    preview = render_gcode_preview_to_image(plan.paths, canvas_size)
    cache[key] = preview
    return preview



def _overlay_missing_mask(plan, reference_lineart_img, render_cache=None):
    canvas_size = reference_lineart_img.size
    preview = (
        _render_plan_preview_cached(plan, canvas_size, render_cache)
        if render_cache is not None else
        render_gcode_preview_to_image(plan.paths, canvas_size)
    )
    reference = _as_gray_array(reference_lineart_img, size=canvas_size)
    ref_ink = reference < QA_OVERLAY_REFERENCE_INK_THRESHOLD
    ref_pixels = int(np.count_nonzero(ref_ink))
    if ref_pixels == 0:
        return {
            "preview": preview,
            "missing_mask": np.zeros_like(reference, dtype=np.uint8),
            "coverage": 1.0,
            "missing_pixels": 0,
            "missing_fraction": 0.0,
            "reference_pixels": 0,
        }

    prev_ink = np.where(
        preview < QA_OVERLAY_PREVIEW_INK_THRESHOLD, 255, 0).astype(np.uint8)
    tolerance = max(0, int(QA_OVERLAY_TOLERANCE_PX))
    if tolerance:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))
        covered = cv2.dilate(prev_ink, kernel, iterations=1) > 0
    else:
        covered = prev_ink > 0

    missing = np.where(ref_ink & ~covered, 255, 0).astype(np.uint8)
    missing = _adaptive_component_cleanup(
        missing, min_area=QA_OVERLAY_MISSING_COMPONENT_MIN_AREA_PX)
    missing_pixels = int(cv2.countNonZero(missing))
    missing_fraction = missing_pixels / float(max(1, ref_pixels))
    return {
        "preview": preview,
        "missing_mask": missing,
        "coverage": float(np.clip(1.0 - missing_fraction, 0.0, 1.0)),
        "missing_pixels": missing_pixels,
        "missing_fraction": float(missing_fraction),
        "reference_pixels": ref_pixels,
    }


def _missing_mask_to_candidates(missing_mask, reference_lineart_img,
                                face_mask=None,
                                max_new=QA_OVERLAY_MAX_NEW_CANDIDATES):
    missing = np.where(np.asarray(missing_mask, dtype=np.uint8) > 0, 255, 0)
    if missing.size == 0 or cv2.countNonZero(missing) == 0:
        return []

    repair_img = np.full_like(missing, 255, dtype=np.uint8)
    repair_img[missing > 0] = 0
    pil_missing = Image.fromarray(repair_img)
    pil_missing.info["vector_mode"] = "clean_lineart"
    pil_missing.info["contains_symbolic_fills"] = True
    raw = extract_candidate_paths(
        pil_missing,
        reference_img=reference_lineart_img,
        face_mask=face_mask,
    )
    ranked = sorted(
        raw,
        key=lambda c: (
            _candidate_mask_overlap(c, face_mask, canvas_size=reference_lineart_img.size)
            if face_mask is not None else 0.0,
            c.length_mm,
            c.saliency,
        ),
        reverse=True,
    )

    promoted = []
    for candidate in ranked[:max(1, int(max_new))]:
        repaired = _candidate_with_points(
            candidate, candidate.points.copy(), source="qa_overlay_add")
        face_overlap = _candidate_mask_overlap(
            repaired, face_mask, canvas_size=reference_lineart_img.size)
        repaired.importance = 1.0
        repaired.saliency = 1.0
        repaired.detail_tier = 0
        repaired.region = "face" if face_overlap >= 0.18 else "garment_detail"
        repaired.protected = True
        repaired.classifier_scores["qa_overlay"] = {
            "missing_repair": True,
            "face_overlap": float(face_overlap),
        }
        promoted.append(repaired)
    return promoted


def _overlay_repair_budget(plan, budget, additions):
    if not additions:
        return int(budget)
    repair_paths = [
        _adaptive_rdp_simplify(
            candidate.points,
            _candidate_rdp_epsilon(candidate, FACE_RDP_EPSILON_MM),
        )
        for candidate in additions
        if len(candidate.points) >= 2
    ]
    repair_segments = _count_segments(repair_paths)
    repair_cost = sum(_candidate_min_segment_cost(candidate) for candidate in additions)
    return int(min(
        MAX_STROKE_BUDGET,
        max(
            MIN_STROKE_BUDGET,
            int(budget),
            int(plan.actual_segments) + max(24, repair_segments + repair_cost * 2),
        ),
    ))


def repair_missing_details_with_overlay(candidates, plan, budget,
                                        reference_lineart_img, face_mask=None,
                                        render_cache=None):
    before = _overlay_missing_mask(plan, reference_lineart_img, render_cache)
    stats = {
        "coverage_before": before["coverage"],
        "coverage_after": before["coverage"],
        "missing_pixels_before": before["missing_pixels"],
        "missing_pixels_after": before["missing_pixels"],
        "missing_fraction_before": before["missing_fraction"],
        "missing_fraction_after": before["missing_fraction"],
        "reference_pixels": before["reference_pixels"],
        "added_candidates": 0,
        "accepted": False,
    }
    needs_repair = (
        before["missing_pixels"] >= QA_OVERLAY_MIN_MISSING_PIXELS and
        before["missing_fraction"] >= QA_OVERLAY_MIN_MISSING_FRACTION
    )
    if not needs_repair:
        return list(candidates), plan, stats

    additions = _missing_mask_to_candidates(
        before["missing_mask"],
        reference_lineart_img,
        face_mask=face_mask,
    )
    stats["added_candidates"] = len(additions)
    if not additions:
        return list(candidates), plan, stats

    repaired_candidates, _ = _deduplicate_candidates(list(candidates) + additions)
    repaired_budget = _overlay_repair_budget(plan, budget, additions)
    repaired_plan = build_gcode_plan(repaired_candidates, repaired_budget)
    after = _overlay_missing_mask(repaired_plan, reference_lineart_img, render_cache)
    stats.update({
        "coverage_after": after["coverage"],
        "missing_pixels_after": after["missing_pixels"],
        "missing_fraction_after": after["missing_fraction"],
        "accepted": after["missing_pixels"] < before["missing_pixels"],
    })
    if not stats["accepted"]:
        return list(candidates), plan, stats

    print(
        "QA overlay repair: "
        f"missing {before['missing_pixels']} -> {after['missing_pixels']} px, "
        f"coverage {before['coverage']:.3f} -> {after['coverage']:.3f}, "
        f"added={len(additions)}, budget={repaired_budget}"
    )
    return repaired_candidates, repaired_plan, stats


def _qa_iteration_from_plan(index, plan, ssim_score, changes, elapsed_s,
                            model_result=None):
    model_result = model_result or {}
    model_score = model_result.get("fidelity_score")
    try:
        model_score = None if model_score is None else float(model_score)
    except Exception:
        model_score = None
    return QAIteration(
        index=index,
        ssim=float(ssim_score),
        model_score=model_score,
        stroke_count=int(plan.stroke_count),
        pen_lifts=int(plan.pen_lifts),
        draw_distance_mm=float(plan.draw_distance_mm),
        estimated_time_s=float(plan.estimated_time_s),
        changes_applied=list(changes),
        redundancy_notes=str(model_result.get("redundancy_notes", "") or ""),
        missing_detail_notes=str(model_result.get("missing_detail_notes", "") or ""),
        elapsed_s=float(elapsed_s),
    )


def run_quality_gate(candidates, budget, reference_lineart_img,
                     original_img=None, face_mask=None, initial_plan=None,
                     timeout_s=QA_TIMEOUT_S):
    started = time.perf_counter()
    render_cache = {}
    iterations = []
    canvas_size = reference_lineart_img.size
    candidate_source = list(candidates)
    best_plan = initial_plan if initial_plan is not None else build_gcode_plan(
        candidate_source, budget)

    def elapsed():
        return time.perf_counter() - started

    def assess(index, plan, changes):
        t0 = time.perf_counter()
        overlay = _overlay_missing_mask(plan, reference_lineart_img, render_cache)
        iter_elapsed = time.perf_counter() - t0
        iteration = _qa_iteration_from_plan(
            index, plan, overlay["coverage"], changes, iter_elapsed)
        iteration.missing_detail_notes = (
            f"missing={overlay['missing_pixels']}px "
            f"({overlay['missing_fraction'] * 100.0:.2f}%)"
        )
        print(
            f"QA overlay {index}: coverage={overlay['coverage']:.3f}, "
            f"missing={overlay['missing_pixels']}px, "
            f"strokes={plan.stroke_count}, lifts={plan.pen_lifts}, "
            f"elapsed={iter_elapsed:.2f}s"
        )
        iterations.append(iteration)
        return iteration, overlay

    iter1, overlay1 = assess(1, best_plan, ["Overlay check against target ink"])
    best_coverage = iter1.ssim
    final_overlay = overlay1

    if elapsed() < float(timeout_s):
        candidate_source, repaired_plan, overlay_stats = \
            repair_missing_details_with_overlay(
                candidate_source,
                best_plan,
                budget,
                reference_lineart_img,
                face_mask=face_mask,
                render_cache=render_cache,
            )
        if overlay_stats.get("accepted"):
            best_plan = repaired_plan
            changes = [
                "Overlay repair added "
                f"{overlay_stats['added_candidates']} missing strokes",
                "Coverage "
                f"{overlay_stats['coverage_before']:.3f} -> "
                f"{overlay_stats['coverage_after']:.3f}",
            ]
            iter2, final_overlay = assess(2, best_plan, changes)
            best_coverage = max(best_coverage, iter2.ssim)
        else:
            best_coverage = max(best_coverage, overlay_stats["coverage_after"])
            final_overlay = {
                **final_overlay,
                "coverage": overlay_stats["coverage_after"],
                "missing_pixels": overlay_stats["missing_pixels_after"],
                "missing_fraction": overlay_stats["missing_fraction_after"],
            }

    total_time = elapsed()
    timed_out_flag = total_time >= float(timeout_s)
    final_missing_fraction = float(final_overlay.get("missing_fraction", 0.0))
    final_missing_pixels = int(final_overlay.get("missing_pixels", 0))
    passed = (
        len(iterations) >= QA_MIN_ITERATIONS and
        (best_coverage >= QA_OVERLAY_PASS_COVERAGE or
         final_missing_pixels < QA_OVERLAY_MIN_MISSING_PIXELS) and
        not timed_out_flag
    )
    report = QAReport(
        iterations=iterations,
        final_stroke_count=int(best_plan.stroke_count),
        final_ssim=float(best_coverage),
        final_model_score=None,
        passed=bool(passed),
        timed_out=bool(timed_out_flag),
        total_time_s=float(total_time),
        model_status=(
            "overlay_only"
            f": missing {final_missing_pixels}px "
            f"({final_missing_fraction * 100.0:.2f}%)"
        ),
    )
    best_plan.qa_report = report
    print(
        f"QA overlay report: passed={report.passed}, "
        f"coverage={report.final_ssim:.3f}, "
        f"missing={final_missing_pixels}px, "
        f"strokes={report.final_stroke_count}, time={report.total_time_s:.2f}s"
    )
    return best_plan, report


# ═══════════════════════════════════════════════════════════════════
#  THEME / DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════

class Theme:
    BG_APP       = "#11111b"
    BG_SIDEBAR   = "#181825"
    BG_CARD      = "#1e1e2e"
    BG_PREVIEW   = "#181825"
    BG_SHADOW    = "#0a0a11"
    BORDER       = "#313244"
    BORDER_SOFT  = "#26263a"

    TEXT_PRIMARY   = "#cdd6f4"
    TEXT_SECOND    = "#a6adc8"
    TEXT_MUTED     = "#6c7086"
    TEXT_ON_ACCENT = "#11111b"

    ACCENT_PURPLE = "#cba6f7"
    ACCENT_BLUE   = "#89b4fa"
    ACCENT_PINK   = "#f5c2e7"
    ACCENT_GREEN  = "#a6e3a1"
    ACCENT_RED    = "#f38ba8"
    ACCENT_YELLOW = "#f9e2af"
    ACCENT_TEAL   = "#94e2d5"

    FONT_FAMILY = "Segoe UI"


def _lighten(hex_color, factor=0.16):
    hex_color = hex_color.lstrip("#")
    r, g, b   = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return f"#{r:02x}{g:02x}{b:02x}"


def _darken(hex_color, factor=0.12):
    hex_color = hex_color.lstrip("#")
    r, g, b   = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r = max(0, int(r * (1 - factor)))
    g = max(0, int(g * (1 - factor)))
    b = max(0, int(b * (1 - factor)))
    return f"#{r:02x}{g:02x}{b:02x}"


# ═══════════════════════════════════════════════════════════════════
#  WIDGET: ROUNDED BUTTON
# ═══════════════════════════════════════════════════════════════════

class RoundedButton(tk.Canvas):
    """Canvas-based button with rounded corners and hover/press animation."""

    def __init__(self, parent, text, icon="",
                 bg=Theme.ACCENT_PURPLE, width=200, height=44,
                 radius=12, parent_bg=Theme.BG_APP, command=None, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=parent_bg, highlightthickness=0, bd=0, **kw)
        self._text      = text
        self._icon      = icon
        self._bg_normal = bg
        self._bg_hover  = _lighten(bg, 0.18)
        self._bg_press  = _darken(bg, 0.15)
        self._bg_cur    = bg
        self._radius    = radius
        self._width     = width
        self._height    = height
        self._command   = command
        self._disabled  = False
        self._draw()
        self.bind("<Enter>",           self._on_enter)
        self.bind("<Leave>",           self._on_leave)
        self.bind("<ButtonPress-1>",   self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _draw(self):
        self.delete("all")
        w, h, r = self._width, self._height, self._radius
        pts = [r,0, w-r,0, w,0, w,r, w,h-r, w,h, w-r,h, r,h, 0,h, 0,h-r, 0,r, 0,0, r,0]
        self.create_polygon(pts, smooth=True, fill=self._bg_cur, outline="")
        label = f"{self._icon}  {self._text}" if self._icon else self._text
        fg    = Theme.TEXT_ON_ACCENT if not self._disabled else Theme.TEXT_MUTED
        self.create_text(w / 2, h / 2, text=label, fill=fg,
                         font=(Theme.FONT_FAMILY, 10, "bold"), anchor="center")

    def _on_enter(self, _e):
        if self._disabled: return
        self._bg_cur = self._bg_hover; self._draw()

    def _on_leave(self, _e):
        if self._disabled: return
        self._bg_cur = self._bg_normal; self._draw()

    def _on_press(self, _e):
        if self._disabled: return
        self._bg_cur = self._bg_press; self._draw()

    def _on_release(self, _e):
        if self._disabled: return
        self._bg_cur = self._bg_hover; self._draw()
        if self._command:
            self._command()

    def config(self, **kw):
        if "state" in kw:
            self._disabled = (kw["state"] == tk.DISABLED)
            self._bg_cur   = _darken(self._bg_normal, 0.35) if self._disabled \
                             else self._bg_normal
            self._draw()


# ═══════════════════════════════════════════════════════════════════
#  WIDGET: CARD PANEL
# ═══════════════════════════════════════════════════════════════════

class CardPanel(tk.Frame):
    """Card container with title, status dot and body frame."""

    def __init__(self, parent, title, icon="", **kw):
        super().__init__(parent, bg=Theme.BG_CARD,
                         highlightthickness=1,
                         highlightbackground=Theme.BORDER, **kw)
        header = tk.Frame(self, bg=Theme.BG_CARD)
        header.pack(fill="x", padx=12, pady=(10, 6))

        self._dot    = tk.Canvas(header, width=8, height=8,
                                 bg=Theme.BG_CARD, highlightthickness=0)
        self._dot_id = self._dot.create_oval(0, 0, 8, 8,
                                             fill=Theme.TEXT_MUTED, outline="")
        self._dot.pack(side="left", padx=(0, 6))

        lbl_text = f"{icon}  {title}" if icon else title
        tk.Label(header, text=lbl_text,
                 font=(Theme.FONT_FAMILY, 9, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(side="left")

        self._status_lbl = tk.Label(header, text="",
                                    font=(Theme.FONT_FAMILY, 8),
                                    bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED)
        self._status_lbl.pack(side="right")

        tk.Frame(self, bg=Theme.BORDER, height=1).pack(fill="x")

        self.body = tk.Frame(self, bg=Theme.BG_CARD)
        self.body.pack(fill="both", expand=True, padx=8, pady=8)

    def set_status(self, text, color=Theme.TEXT_MUTED):
        self._status_lbl.config(text=text, fg=color)
        self._dot.itemconfig(self._dot_id, fill=color)


# ═══════════════════════════════════════════════════════════════════
#  WIDGET: G-CODE PREVIEW CANVAS
# ═══════════════════════════════════════════════════════════════════

class GCodePreviewCanvas(tk.Canvas):
    """Canvas that renders G-code paths as a visual drawing preview."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=Theme.BG_PREVIEW,
                         highlightthickness=0, **kw)
        self.drawing_paths   = []
        self.travel_segments = []
        self.show_travel     = False
        self.placeholder     = "No G-code yet"
        self._zoom           = 1.0
        self._pan_x          = 0.0
        self._pan_y          = 0.0
        self._drag_start     = None
        self.brush_enabled   = False
        self.brush_size_px   = MANUAL_BRUSH_DEFAULT_PX
        self.brush_mode      = "add"
        self.brush_callback  = None
        self._brush_active   = False
        self._brush_points_mm = []
        self._brush_live_mode = "add"
        self._last_mouse_xy  = None
        self.pending_brush_masks = {"add": None, "reduce": None}
        self.pending_brush_canvas_size = None
        self._brush_overlay_tk = None
        self.bind("<Configure>", lambda _e: self.after_idle(self._redraw))
        self.bind("<MouseWheel>", self._on_mousewheel)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave_canvas)
        self.bind("<ButtonPress-1>", self._on_pan_start)
        self.bind("<B1-Motion>", self._on_pan_move)
        self.bind("<ButtonRelease-1>", self._on_pan_end)
        self.bind("<ButtonPress-2>", self._on_pan_start)
        self.bind("<B2-Motion>", self._on_pan_move)
        self.bind("<ButtonRelease-2>", self._on_pan_end)
        self.bind("<Double-Button-1>", self._reset_zoom)

    def clear(self, placeholder="No G-code yet"):
        self.drawing_paths   = []
        self.travel_segments = []
        self.placeholder     = placeholder
        self._redraw()

    def set_gcode(self, gcode_lines):
        self.drawing_paths, self.travel_segments = parse_gcode_for_preview(gcode_lines)
        self.placeholder = ""
        self._redraw()

    def set_show_travel(self, show):
        self.show_travel = bool(show); self._redraw()

    def set_pending_brush_overlay(self, add_mask=None, reduce_mask=None,
                                  canvas_size=None):
        self.pending_brush_masks = {
            "add": None if add_mask is None else np.asarray(add_mask, dtype=np.uint8).copy(),
            "reduce": None if reduce_mask is None else np.asarray(reduce_mask, dtype=np.uint8).copy(),
        }
        if canvas_size is None:
            for mask in self.pending_brush_masks.values():
                if mask is not None and mask.ndim == 2:
                    canvas_size = (mask.shape[1], mask.shape[0])
                    break
        self.pending_brush_canvas_size = canvas_size
        self._redraw()

    def set_manual_brush(self, enabled, size_px=None, mode=None, callback=None):
        self.brush_enabled = bool(enabled)
        if size_px is not None:
            self.brush_size_px = int(max(MANUAL_BRUSH_MIN_PX,
                                         min(MANUAL_BRUSH_MAX_PX, round(float(size_px)))))
        if mode is not None:
            self.brush_mode = "reduce" if str(mode).lower().startswith("reduce") else "add"
        if callback is not None:
            self.brush_callback = callback
        if not self.brush_enabled:
            self._brush_active = False
            self._brush_points_mm = []
            self._last_mouse_xy = None
            self.delete("brush_cursor")
        self._update_pan_cursor()
        self._redraw()

    def _transform(self, x, y, ox, oy, pw, ph):
        cx = ox + (x / PAGE_MAX_X) * pw
        cy = oy + (1.0 - y / PAGE_MAX_Y) * ph
        return cx, cy

    def _page_geometry(self):
        w = max(10, self.winfo_width())
        h = max(10, self.winfo_height())
        pw, ph = self._fit_page_size(w, h, self._zoom)
        self._clamp_pan(w, h, pw, ph)
        ox = (w - pw) / 2.0 + self._pan_x
        oy = (h - ph) / 2.0 + self._pan_y
        return w, h, ox, oy, pw, ph

    def canvas_to_page_mm(self, x, y, clamp=False):
        _w, _h, ox, oy, pw, ph = self._page_geometry()
        if pw <= 1e-6 or ph <= 1e-6:
            return None
        rel_x = (float(x) - ox) / pw
        rel_y = (float(y) - oy) / ph
        if not clamp and not (0.0 <= rel_x <= 1.0 and 0.0 <= rel_y <= 1.0):
            return None
        rel_x = float(np.clip(rel_x, 0.0, 1.0))
        rel_y = float(np.clip(rel_y, 0.0, 1.0))
        return np.asarray([rel_x * PAGE_MAX_X, (1.0 - rel_y) * PAGE_MAX_Y],
                          dtype=np.float32)

    def brush_radius_mm(self):
        _w, _h, _ox, _oy, pw, _ph = self._page_geometry()
        return max(0.05, (float(self.brush_size_px) * 0.5) * PAGE_MAX_X / max(1e-6, pw))

    @staticmethod
    def _fit_page_size(w, h, zoom):
        pad = 24
        avail_w = max(10, w - 2 * pad)
        avail_h = max(10, h - 2 * pad)
        ratio = PAGE_MAX_X / PAGE_MAX_Y
        if avail_w / avail_h > ratio:
            ph = avail_h
            pw = ph * ratio
        else:
            pw = avail_w
            ph = pw / ratio
        return pw * zoom, ph * zoom

    def _clamp_pan(self, w=None, h=None, pw=None, ph=None):
        w = max(10, self.winfo_width()) if w is None else w
        h = max(10, self.winfo_height()) if h is None else h
        if pw is None or ph is None:
            pw, ph = self._fit_page_size(w, h, self._zoom)

        if self._zoom <= 1.0:
            self._pan_x = 0.0
            self._pan_y = 0.0
            return

        # A small overscroll keeps the edge easy to inspect without allowing
        # the page to be dragged completely away from the viewport.
        max_x = ((pw - w) / 2.0 + 24) if pw > w else (w - pw) / 2.0
        max_y = ((ph - h) / 2.0 + 24) if ph > h else (h - ph) / 2.0
        self._pan_x = max(-max_x, min(max_x, self._pan_x))
        self._pan_y = max(-max_y, min(max_y, self._pan_y))

    def _can_pan(self):
        w = max(10, self.winfo_width())
        h = max(10, self.winfo_height())
        return self._zoom > 1.0

    def _update_pan_cursor(self, dragging=False):
        if self.brush_enabled:
            self.configure(cursor="none")
            return
        can_pan = self._can_pan()
        self.configure(cursor="fleur" if dragging and can_pan
                       else "hand2" if can_pan else "")

    def _draw_brush_cursor(self, x=None, y=None, mode=None):
        self.delete("brush_cursor")
        if not self.brush_enabled:
            return
        if x is None or y is None:
            if self._last_mouse_xy is None:
                return
            x, y = self._last_mouse_xy
        radius = max(2.0, float(self.brush_size_px) * 0.5)
        mode = mode or self._event_brush_mode(None)
        color = Theme.ACCENT_GREEN if mode == "add" else Theme.ACCENT_RED
        self.create_oval(x - radius, y - radius, x + radius, y + radius,
                         outline=color, width=2, tags=("brush_cursor",))

    def _draw_pending_brush_overlay(self, ox, oy, pw, ph):
        canvas_size = self.pending_brush_canvas_size
        if canvas_size is None:
            return
        width, height = _normalize_canvas_size(canvas_size)
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        colors = {
            "add": (42, 175, 120, 112),
            "reduce": (239, 89, 120, 120),
        }

        for mode in ("add", "reduce"):
            mask = self.pending_brush_masks.get(mode)
            resized = _resize_binary_mask(mask, width, height)
            if resized is None or cv2.countNonZero(resized) == 0:
                continue
            overlay[resized > 0] = colors[mode]

        if self.brush_enabled and self._brush_active and self._brush_points_mm:
            live_mask = make_brush_mask(
                self._brush_points_mm,
                self.brush_radius_mm(),
                (width, height),
            )
            if cv2.countNonZero(live_mask) > 0:
                overlay[live_mask > 0] = colors.get(
                    self._brush_live_mode, colors["add"])

        if not np.any(overlay[:, :, 3] > 0):
            self._brush_overlay_tk = None
            return

        resampling = getattr(Image, "Resampling", None)
        resample = getattr(
            resampling, "NEAREST",
            getattr(Image, "NEAREST", 0),
        )
        img = Image.fromarray(overlay, "RGBA").resize(
            (max(1, int(round(pw))), max(1, int(round(ph)))),
            resample,
        )
        self._brush_overlay_tk = ImageTk.PhotoImage(img)
        self.create_image(ox, oy, anchor="nw", image=self._brush_overlay_tk,
                          tags=("brush_overlay",))

    def _event_brush_mode(self, event):
        mode = self.brush_mode
        if event is not None and (int(getattr(event, "state", 0)) & 0x0001):
            mode = "reduce" if mode == "add" else "add"
        return mode

    def _append_brush_point(self, event):
        point = self.canvas_to_page_mm(event.x, event.y, clamp=False)
        if point is None:
            return
        if self._brush_points_mm:
            if float(np.linalg.norm(point - self._brush_points_mm[-1])) < 0.25:
                return
        self._brush_points_mm.append(point)

    def _on_motion(self, event):
        self._last_mouse_xy = (event.x, event.y)
        if self.brush_enabled:
            self._draw_brush_cursor(event.x, event.y,
                                    mode=self._event_brush_mode(event))

    def _on_leave_canvas(self, _event=None):
        if self.brush_enabled and not self._brush_active:
            self._last_mouse_xy = None
            self.delete("brush_cursor")

    def _on_mousewheel(self, event):
        factor = 1.12 if event.delta > 0 else 1.0 / 1.12
        old_zoom = self._zoom
        new_zoom = max(0.5, min(5.0, old_zoom * factor))
        actual_factor = new_zoom / old_zoom

        # Keep the drawing point below the mouse at the same screen position.
        w = max(10, self.winfo_width())
        h = max(10, self.winfo_height())
        self._pan_x += (event.x - w / 2.0 - self._pan_x) * (1.0 - actual_factor)
        self._pan_y += (event.y - h / 2.0 - self._pan_y) * (1.0 - actual_factor)
        self._zoom = new_zoom
        pw, ph = self._fit_page_size(w, h, self._zoom)
        self._clamp_pan(w, h, pw, ph)
        self._update_pan_cursor()
        self._redraw()
        return "break"

    def _on_pan_start(self, event):
        if self.brush_enabled:
            self._brush_active = True
            self._brush_points_mm = []
            self._last_mouse_xy = (event.x, event.y)
            self._brush_live_mode = self._event_brush_mode(event)
            self._append_brush_point(event)
            self._draw_brush_cursor(event.x, event.y,
                                    mode=self._brush_live_mode)
            self._redraw()
            return "break"
        if not self._can_pan():
            return "break"
        self._drag_start = (event.x, event.y, self._pan_x, self._pan_y)
        self._update_pan_cursor(dragging=True)
        return "break"

    def _on_pan_move(self, event):
        if self.brush_enabled:
            if self._brush_active:
                self._brush_live_mode = self._event_brush_mode(event)
                self._append_brush_point(event)
            self._last_mouse_xy = (event.x, event.y)
            self._draw_brush_cursor(event.x, event.y,
                                    mode=self._brush_live_mode)
            if self._brush_active:
                self._redraw()
            return "break"
        if self._drag_start is None:
            return "break"
        sx, sy, px, py = self._drag_start
        self._pan_x = px + event.x - sx
        self._pan_y = py + event.y - sy
        self._clamp_pan()
        self._redraw()
        return "break"

    def _on_pan_end(self, _event=None):
        if self.brush_enabled:
            event = _event
            if event is not None and self._brush_active:
                self._append_brush_point(event)
            points = [p.copy() for p in self._brush_points_mm]
            radius_mm = self.brush_radius_mm()
            mode = self._event_brush_mode(event)
            self._brush_active = False
            self._brush_points_mm = []
            if len(points) >= 1 and self.brush_callback is not None:
                self.brush_callback(points, radius_mm, mode)
            self._redraw()
            return "break"
        self._drag_start = None
        self._update_pan_cursor()
        return "break"

    def _reset_zoom(self, _event=None):
        if self.brush_enabled:
            return "break"
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start = None
        self._update_pan_cursor()
        self._redraw()
        return "break"

    def _draw_zoom_badge(self, w):
        self.create_rectangle(w - 70, 8, w - 10, 28,
                              fill=Theme.BG_CARD, outline=Theme.BORDER)
        self.create_text(w - 40, 18, text=f"{self._zoom * 100:.0f}%",
                         fill=Theme.ACCENT_TEAL,
                         font=(Theme.FONT_FAMILY, 8, "bold"))

    def _redraw(self):
        if not self.winfo_exists():
            return
        self.delete("all")
        w = max(10, self.winfo_width())
        h = max(10, self.winfo_height())
        pw, ph = self._fit_page_size(w, h, self._zoom)
        self._clamp_pan(w, h, pw, ph)
        self._update_pan_cursor(dragging=self._drag_start is not None)
        ox = (w - pw) / 2.0 + self._pan_x
        oy = (h - ph) / 2.0 + self._pan_y

        self.create_rectangle(ox, oy, ox + pw, oy + ph,
                              fill="#f7f7f7", outline=Theme.BORDER, width=1)
        for gx in range(50, int(PAGE_MAX_X), 50):
            x1, _ = self._transform(gx, 0, ox, oy, pw, ph)
            self.create_line(x1, oy, x1, oy + ph, fill="#dddddd", width=1)
        for gy in range(50, int(PAGE_MAX_Y), 50):
            _, y1 = self._transform(0, gy, ox, oy, pw, ph)
            self.create_line(ox, y1, ox + pw, y1, fill="#dddddd", width=1)

        if not self.drawing_paths:
            self.create_text(w / 2, h / 2, text=self.placeholder or "No paths",
                             fill=Theme.TEXT_MUTED,
                             font=(Theme.FONT_FAMILY, 10))
            self._draw_pending_brush_overlay(ox, oy, pw, ph)
            self._draw_zoom_badge(w)
            self._draw_brush_cursor()
            return

        if self.show_travel:
            for a, b in self.travel_segments:
                x1, y1 = self._transform(a[0], a[1], ox, oy, pw, ph)
                x2, y2 = self._transform(b[0], b[1], ox, oy, pw, ph)
                self.create_line(x1, y1, x2, y2, fill="#a0a0a0", width=1, dash=(3,3))

        for path in self.drawing_paths:
            if len(path) < 2:
                continue
            coords = []
            for x, y in path:
                cx, cy = self._transform(x, y, ox, oy, pw, ph)
                coords.extend([cx, cy])
            self.create_line(*coords, fill="#111111", width=1.35,
                             capstyle=tk.ROUND, joinstyle=tk.ROUND)

        self._draw_pending_brush_overlay(ox, oy, pw, ph)
        self._draw_zoom_badge(w)
        self._draw_brush_cursor()


class ImageZoomWindow:
    """Simple popup image viewer with mousewheel zoom and drag pan."""

    def __init__(self, parent, pil_img, title="Image Preview"):
        self.parent = parent
        self.original = pil_img.copy()
        if self.original.mode == "RGBA":
            bg = Image.new("RGBA", self.original.size, (24, 24, 37, 255))
            bg.alpha_composite(self.original)
            self.original = bg.convert("RGB")
        else:
            self.original = self.original.convert("RGB")

        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_start = None
        self._tk_img = None
        self._tk_img_size = None

        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.configure(bg=Theme.BG_APP)
        self.window.geometry("980x720")

        toolbar = tk.Frame(self.window, bg=Theme.BG_APP)
        toolbar.pack(fill="x", padx=10, pady=(10, 6))
        self.zoom_var = tk.StringVar(value="100%")
        tk.Button(toolbar, text="-", width=3, command=lambda: self._zoom_by(1 / 1.2),
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                  activebackground=Theme.BORDER, activeforeground=Theme.TEXT_PRIMARY,
                  bd=0).pack(side="left")
        tk.Button(toolbar, text="+", width=3, command=lambda: self._zoom_by(1.2),
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                  activebackground=Theme.BORDER, activeforeground=Theme.TEXT_PRIMARY,
                  bd=0).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Reset", command=self._reset,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                  activebackground=Theme.BORDER, activeforeground=Theme.TEXT_PRIMARY,
                  bd=0).pack(side="left", padx=(6, 0))
        tk.Label(toolbar, textvariable=self.zoom_var,
                 font=(Theme.FONT_FAMILY, 9, "bold"),
                 bg=Theme.BG_APP, fg=Theme.ACCENT_TEAL).pack(side="left", padx=(10, 0))

        self.canvas = tk.Canvas(self.window, bg=Theme.BG_PREVIEW,
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _e: self._redraw())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<ButtonPress-2>", self._on_drag_start)
        self.canvas.bind("<B2-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_drag_end)
        self.canvas.bind("<Double-Button-1>", lambda _e: self._reset())
        self.window.bind("<Escape>", lambda _e: self.window.destroy())

        self._redraw()

    def _zoom_by(self, factor, anchor=None):
        old_zoom = self.zoom
        new_zoom = max(0.1, min(8.0, old_zoom * factor))
        actual_factor = new_zoom / old_zoom
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())
        ax, ay = anchor if anchor is not None else (cw / 2.0, ch / 2.0)

        self.pan_x += (ax - cw / 2.0 - self.pan_x) * (1.0 - actual_factor)
        self.pan_y += (ay - ch / 2.0 - self.pan_y) * (1.0 - actual_factor)
        self.zoom = new_zoom
        self._clamp_pan(cw, ch)
        self._redraw()

    def _reset(self):
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag_start = None
        self._redraw()

    def _on_mousewheel(self, event):
        factor = 1.12 if event.delta > 0 else 1.0 / 1.12
        self._zoom_by(factor, anchor=(event.x, event.y))
        return "break"

    def _on_drag_start(self, event):
        if not self._can_pan():
            return "break"
        self._drag_start = (event.x, event.y, self.pan_x, self.pan_y)
        self.canvas.configure(cursor="fleur")
        return "break"

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return "break"
        sx, sy, px, py = self._drag_start
        self.pan_x = px + event.x - sx
        self.pan_y = py + event.y - sy
        self._clamp_pan()
        self._redraw()
        return "break"

    def _on_drag_end(self, _event=None):
        self._drag_start = None
        self._update_pan_cursor()
        return "break"

    def _rendered_size(self, cw=None, ch=None):
        cw = max(10, self.canvas.winfo_width()) if cw is None else cw
        ch = max(10, self.canvas.winfo_height()) if ch is None else ch
        iw, ih = self.original.size
        fit = min(cw / max(1, iw), ch / max(1, ih), 1.0)
        return max(1, int(round(iw * fit * self.zoom))), \
               max(1, int(round(ih * fit * self.zoom)))

    def _can_pan(self, cw=None, ch=None, rw=None, rh=None):
        return self.zoom > 1.0

    def _clamp_pan(self, cw=None, ch=None, rw=None, rh=None):
        cw = max(10, self.canvas.winfo_width()) if cw is None else cw
        ch = max(10, self.canvas.winfo_height()) if ch is None else ch
        if rw is None or rh is None:
            rw, rh = self._rendered_size(cw, ch)
        if self.zoom <= 1.0:
            self.pan_x = 0.0
            self.pan_y = 0.0
            return
        max_x = ((rw - cw) / 2.0 + 24) if rw > cw else (cw - rw) / 2.0
        max_y = ((rh - ch) / 2.0 + 24) if rh > ch else (ch - rh) / 2.0
        self.pan_x = max(-max_x, min(max_x, self.pan_x))
        self.pan_y = max(-max_y, min(max_y, self.pan_y))

    def _update_pan_cursor(self, cw=None, ch=None, rw=None, rh=None):
        can_pan = self._can_pan(cw, ch, rw, rh)
        self.canvas.configure(cursor="fleur" if self._drag_start is not None and can_pan
                              else "hand2" if can_pan else "")

    def _redraw(self):
        if not self.canvas.winfo_exists():
            return
        self.canvas.delete("all")
        cw = max(10, self.canvas.winfo_width())
        ch = max(10, self.canvas.winfo_height())
        iw, ih = self.original.size
        fit = min(cw / max(1, iw), ch / max(1, ih), 1.0)
        scale = fit * self.zoom
        rw = max(1, int(round(iw * scale)))
        rh = max(1, int(round(ih * scale)))
        self._clamp_pan(cw, ch, rw, rh)
        if self._tk_img is None or self._tk_img_size != (rw, rh):
            resized = self.original.resize((rw, rh), Image.Resampling.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(resized)
            self._tk_img_size = (rw, rh)
        self.canvas.create_image(cw / 2 + self.pan_x, ch / 2 + self.pan_y,
                                 image=self._tk_img, anchor="center")
        self.zoom_var.set(f"{self.zoom * 100:.0f}%")
        self._update_pan_cursor(cw, ch, rw, rh)


# ═══════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Sketch -> G-Code  |  Drawing Robot")
        self.root.configure(bg=Theme.BG_APP)
        self.root.minsize(1120, 680)

        self.input_path          = None
        self.pil_original        = None
        self.pil_work            = None
        self.pil_lineart         = None
        self.candidate_paths     = None
        self.auto_candidate_paths = None
        self.current_plan        = None
        self.current_plan_budget = None
        self.auto_budget         = DEFAULT_STROKE_BUDGET
        self.last_raw_segments   = 0
        self.current_ai_backend  = "classical"
        self.current_ai_elapsed_s = 0.0
        self.current_hatch_candidate_count = 0
        self.current_face_mask = None
        self.current_foreground_mask_info = ForegroundMaskInfo()
        self.current_qa_report = None
        self.manual_undo_stack = []
        self.manual_adjust_active = False
        self.manual_adjust_has_changes = False
        self._pending_brush_masks = {"add": None, "reduce": None}
        self._pending_brush_mode = "add"
        self._pending_brush_max_size_px = 0

        self._image_job_id     = 0
        self._preview_job_id   = 0
        self._qa_job_id        = 0
        self._plan_cache       = {}
        self._qa_cache         = {}
        self._preview_after_id = None
        self._hatch_after_id   = None
        self._manual_detail_dirty = False
        self._manual_hatch_dirty  = False

        self._setup_ttk_styles()
        self._build_ui()
        self.root.bind_all("<Control-v>", self._on_paste_image, add="+")
        self.root.bind_all("<Control-V>", self._on_paste_image, add="+")
        self.root.bind_all("<Escape>", self._on_escape_key, add="+")

    # ─────────────────────────────────────────────────────────────
    def _setup_ttk_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Detail.Horizontal.TScale",
                        background=Theme.BG_APP, troughcolor=Theme.BG_CARD,
                        slidercolor=Theme.ACCENT_PURPLE, bordercolor=Theme.BORDER,
                        lightcolor=Theme.ACCENT_PURPLE, darkcolor=Theme.ACCENT_PURPLE)
        style.configure("TProgressbar", troughcolor=Theme.BG_CARD,
                        background=Theme.ACCENT_BLUE, bordercolor=Theme.BORDER)
        # Copy layout trước, nếu không sẽ báo lỗi "Layout ... not found"
        style.layout("Slim.TProgressbar",
                     style.layout("Horizontal.TProgressbar"))
        style.configure("Slim.TProgressbar", troughcolor=Theme.BG_CARD,
                        background=Theme.ACCENT_BLUE, bordercolor=Theme.BORDER,
                        thickness=4)
    # ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()

    def _action_buttons(self):
        names = (
            "btn_upload", "btn_export", "btn_force_export",
            "btn_apply_changes", "btn_manual_undo", "btn_manual_reset",
            "btn_clear_brush",
        )
        return [
            getattr(self, name)
            for name in names
            if hasattr(self, name) and getattr(self, name) is not None
        ]

    # ───────────────── SIDEBAR ───────────────────────────────────
    def _build_sidebar(self):
        sidebar = tk.Frame(self.root, bg=Theme.BG_SIDEBAR, width=264)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        # Brand
        brand = tk.Frame(sidebar, bg=Theme.BG_SIDEBAR)
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 6))
        tk.Label(brand, text="Sketch -> G-Code",
                 font=(Theme.FONT_FAMILY, 13, "bold"),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_PRIMARY).pack(anchor="w")
        tk.Label(brand, text="Drawing Robot Controller",
                 font=(Theme.FONT_FAMILY, 8),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED).pack(anchor="w")

        tk.Frame(sidebar, bg=Theme.BORDER, height=1).grid(
            row=1, column=0, sticky="ew", padx=12, pady=(4, 10))

        # ── MODE SELECTOR [NEW] ───────────────────────────────
        mode_frame = tk.Frame(sidebar, bg=Theme.BG_SIDEBAR)
        mode_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 8))

        tk.Label(mode_frame, text="Auto Pipeline",
                 font=(Theme.FONT_FAMILY, 9, "bold"),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_SECOND).pack(anchor="w")

        self.ai_status_var = tk.StringVar(value="Input: sketch/line-art only")
        tk.Label(
            mode_frame, textvariable=self.ai_status_var,
            bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED,
            font=(Theme.FONT_FAMILY, 7)).pack(
                anchor="w", pady=(3, 0))

        self.face_detect_var = tk.BooleanVar(value=True)
        self.chk_face_detect = tk.Checkbutton(
            mode_frame, text="  Auto-detect face (skip hatch on face)",
            variable=self.face_detect_var, command=self._on_face_mask_change,
            bg=Theme.BG_SIDEBAR, fg=Theme.ACCENT_TEAL,
            activebackground=Theme.BG_SIDEBAR, activeforeground=Theme.ACCENT_TEAL,
            selectcolor=Theme.BG_CARD, font=(Theme.FONT_FAMILY, 8),
            bd=0, highlightthickness=0)
        self.chk_face_detect.pack(anchor="w", padx=(14, 0), pady=(6, 0))

        self.face_feather_var = tk.DoubleVar(value=DEFAULT_FACE_MASK_FEATHER_PX)
        self.face_feather_value_var = tk.StringVar(
            value=f"{DEFAULT_FACE_MASK_FEATHER_PX}px")
        self.face_feather_frame = tk.Frame(mode_frame, bg=Theme.BG_SIDEBAR)
        self.face_feather_frame.pack(fill="x", padx=(28, 0), pady=(2, 0))
        tk.Label(self.face_feather_frame, text="Face mask feather",
                 font=(Theme.FONT_FAMILY, 7),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED).grid(
                     row=0, column=0, sticky="w")
        tk.Label(self.face_feather_frame, textvariable=self.face_feather_value_var,
                 font=(Theme.FONT_FAMILY, 7, "bold"),
                 bg=Theme.BG_SIDEBAR, fg=Theme.ACCENT_TEAL).grid(
                     row=0, column=1, sticky="e", padx=(6, 0))
        self.face_feather_scale = ttk.Scale(
            self.face_feather_frame, from_=0, to=24,
            variable=self.face_feather_var, command=self._on_face_feather_slider,
            orient="horizontal", style="Detail.Horizontal.TScale")
        self.face_feather_scale.grid(row=1, column=0, columnspan=2,
                                     sticky="ew", pady=(1, 0))
        self.face_feather_frame.columnconfigure(0, weight=1)

        # Mask watermark/logo o goc duoi phai truoc khi xu ly anh.
        self.exclude_mark_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            mode_frame, text="  Remove bottom-right mark",
            variable=self.exclude_mark_var, command=self._on_mask_slider,
            bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED,
            activebackground=Theme.BG_SIDEBAR, activeforeground=Theme.TEXT_SECOND,
            selectcolor=Theme.BG_CARD, font=(Theme.FONT_FAMILY, 8),
            bd=0, highlightthickness=0).pack(anchor="w", padx=(14, 0), pady=(6, 0))

        mask_grid = tk.Frame(mode_frame, bg=Theme.BG_SIDEBAR)
        mask_grid.pack(fill="x", padx=(28, 0), pady=(2, 0))
        self.exclude_bottom_var = tk.DoubleVar(value=14.0)
        self.exclude_right_var = tk.DoubleVar(value=28.0)
        tk.Label(mask_grid, text="Bottom %", font=(Theme.FONT_FAMILY, 7),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED).grid(row=0, column=0, sticky="w")
        ttk.Scale(mask_grid, from_=4, to=35, variable=self.exclude_bottom_var,
                  command=self._on_mask_slider,
                  orient="horizontal", style="Detail.Horizontal.TScale").grid(
                      row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(mask_grid, text="Right %", font=(Theme.FONT_FAMILY, 7),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED).grid(row=1, column=0, sticky="w")
        ttk.Scale(mask_grid, from_=6, to=45, variable=self.exclude_right_var,
                  command=self._on_mask_slider,
                  orient="horizontal", style="Detail.Horizontal.TScale").grid(
                      row=1, column=1, sticky="ew", padx=(6, 0))
        mask_grid.columnconfigure(1, weight=1)

        tk.Frame(sidebar, bg=Theme.BORDER, height=1).grid(
            row=3, column=0, sticky="ew", padx=12, pady=(0, 10))

        # Action buttons
        btn_area = tk.Frame(sidebar, bg=Theme.BG_SIDEBAR)
        btn_area.grid(row=4, column=0, sticky="ew", padx=18)

        self.btn_upload = RoundedButton(
            btn_area, "Upload Image",
            bg=Theme.ACCENT_PURPLE, width=228, height=46,
            parent_bg=Theme.BG_SIDEBAR, command=self._on_upload)
        self.btn_upload.pack(pady=(0, 10))
        self.input_thumb_label = tk.Label(
            btn_area, bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            text="No image", font=(Theme.FONT_FAMILY, 7),
            width=28, height=5, compound="left",
            padx=8, pady=6, anchor="w", justify="left",
            highlightthickness=1, highlightbackground=Theme.BORDER)
        self.input_thumb_label.pack(fill="x", pady=(0, 10))
        self._input_thumb_tk = None

        self.qa_status_var = tk.StringVar(value="QA: waiting for preview")
        self.qa_panel = tk.Label(
            btn_area, textvariable=self.qa_status_var,
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
            font=(Theme.FONT_FAMILY, 7), justify="left", anchor="w",
            wraplength=210, padx=10, pady=8,
            highlightthickness=1, highlightbackground=Theme.BORDER)
        self.qa_panel.pack(fill="x", pady=(0, 10))

        self.btn_export = RoundedButton(
            btn_area, "Export G-Code",
            bg=Theme.ACCENT_GREEN, width=228, height=46,
            parent_bg=Theme.BG_SIDEBAR, command=self._on_export_gcode)
        self.btn_export.pack()

        self.btn_force_export = RoundedButton(
            btn_area, "Force export",
            bg=Theme.ACCENT_RED, width=228, height=36,
            parent_bg=Theme.BG_SIDEBAR,
            command=lambda: self._on_export_gcode(force=True))
        self.btn_force_export.pack(pady=(8, 0))
        self.btn_export.config(state=tk.DISABLED)
        self.btn_force_export.config(state=tk.DISABLED)

        manual_buttons = tk.Frame(btn_area, bg=Theme.BG_SIDEBAR)
        manual_buttons.pack(fill="x", pady=(8, 0))
        self.btn_manual_undo = RoundedButton(
            manual_buttons, "Undo brush",
            bg=Theme.ACCENT_BLUE, width=108, height=34,
            parent_bg=Theme.BG_SIDEBAR, command=self._on_manual_undo)
        self.btn_manual_reset = RoundedButton(
            manual_buttons, "Reset Auto",
            bg=Theme.ACCENT_YELLOW, width=108, height=34,
            parent_bg=Theme.BG_SIDEBAR, command=self._on_manual_reset)
        self.btn_manual_undo.pack(side="left")
        self.btn_manual_reset.pack(side="right")
        self.btn_manual_undo.config(state=tk.DISABLED)
        self.btn_manual_reset.config(state=tk.DISABLED)

        self.btn_clear_brush = RoundedButton(
            btn_area, "Clear brush",
            bg=Theme.BG_CARD, width=228, height=32,
            parent_bg=Theme.BG_SIDEBAR, command=self._on_clear_brush)
        self.btn_clear_brush.pack(pady=(8, 0))
        self.btn_clear_brush.config(state=tk.DISABLED)

        self.btn_apply_changes = RoundedButton(
            btn_area, "Apply changes",
            bg=Theme.ACCENT_YELLOW, width=228, height=40,
            parent_bg=Theme.BG_SIDEBAR, command=self._on_apply_changes)
        self.btn_apply_changes.pack(pady=(10, 0))
        self.btn_apply_changes.pack_forget()

        self.progress = ttk.Progressbar(sidebar, mode="indeterminate",
                                        style="Slim.TProgressbar")
        self.progress.grid(row=5, column=0, sticky="ew", padx=18, pady=(12, 0))
        self.progress.grid_remove()

        self.file_info_var = tk.StringVar(value="No file loaded")
        tk.Label(sidebar, textvariable=self.file_info_var,
                 font=(Theme.FONT_FAMILY, 7), bg=Theme.BG_SIDEBAR,
                 fg=Theme.TEXT_MUTED, wraplength=228,
                 justify="left", anchor="w").grid(
                     row=6, column=0, sticky="ew", padx=18, pady=(8, 0))

        sidebar.rowconfigure(7, weight=1)

        tk.Label(sidebar, text="Drawing Robot  v2.1",
                 font=(Theme.FONT_FAMILY, 7),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_MUTED).grid(
                     row=8, column=0, pady=(0, 12))

    # ───────────────── MAIN AREA ─────────────────────────────────
    def _build_main(self):
        main_shell = tk.Frame(self.root, bg=Theme.BG_APP)
        main_shell.grid(row=0, column=1, sticky="nsew")
        main_shell.columnconfigure(0, weight=1)
        main_shell.rowconfigure(0, weight=1)

        self.main_canvas = tk.Canvas(
            main_shell, bg=Theme.BG_APP, highlightthickness=0,
            bd=0, yscrollincrement=24)
        main_scroll = ttk.Scrollbar(
            main_shell, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=main_scroll.set)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        main_scroll.grid(row=0, column=1, sticky="ns")

        main = tk.Frame(self.main_canvas, bg=Theme.BG_APP)
        main_window = self.main_canvas.create_window(
            (18, 18), window=main, anchor="nw")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1, minsize=560)
        main.rowconfigure(1, weight=0)

        def update_main_scrollregion(_event=None):
            self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

        def fit_main_width(event):
            width = max(600, event.width - 36)
            self.main_canvas.itemconfigure(main_window, width=width)
            update_main_scrollregion()

        def on_main_wheel(event):
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def bind_main_wheel(_event=None):
            self.main_canvas.bind_all("<MouseWheel>", on_main_wheel)

        def unbind_main_wheel(_event=None):
            self.main_canvas.unbind_all("<MouseWheel>")

        main.bind("<Configure>", update_main_scrollregion)
        self.main_canvas.bind("<Configure>", fit_main_width)
        self.main_canvas.bind("<Enter>", bind_main_wheel)
        self.main_canvas.bind("<Leave>", unbind_main_wheel)
        main.bind("<Enter>", bind_main_wheel, add="+")
        main.bind("<Leave>", unbind_main_wheel, add="+")

        # Single preview surface.
        self.card_gcode = CardPanel(main, "G-Code Preview", icon="[GC]")
        self.card_gcode.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        preview_toolbar = tk.Frame(self.card_gcode.body, bg=Theme.BG_CARD)
        preview_toolbar.pack(fill="x", pady=(0, 8))
        tk.Label(preview_toolbar, text="Mode",
                 font=(Theme.FONT_FAMILY, 8, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_SECOND).pack(side="left")
        self.preview_mode_var = tk.StringVar(value="Auto")
        self.preview_mode_combo = ttk.Combobox(
            preview_toolbar, textvariable=self.preview_mode_var,
            values=["Auto", "Manual Adjust"], state="readonly",
            width=15, font=(Theme.FONT_FAMILY, 8))
        self.preview_mode_combo.pack(side="left", padx=(8, 12))
        self.preview_mode_combo.bind("<<ComboboxSelected>>",
                                     self._on_preview_mode_change)

        self.manual_toolbar = tk.Frame(preview_toolbar, bg=Theme.BG_CARD)
        self.manual_mode_var = tk.StringVar(value="add")
        tk.Radiobutton(
            self.manual_toolbar, text="+", value="add",
            variable=self.manual_mode_var, command=self._refresh_manual_brush,
            bg=Theme.BG_CARD, fg=Theme.ACCENT_GREEN,
            activebackground=Theme.BG_CARD, activeforeground=Theme.ACCENT_GREEN,
            selectcolor=Theme.BG_PREVIEW, bd=0, highlightthickness=0,
            font=(Theme.FONT_FAMILY, 10, "bold")).pack(side="left")
        tk.Radiobutton(
            self.manual_toolbar, text="-", value="reduce",
            variable=self.manual_mode_var, command=self._refresh_manual_brush,
            bg=Theme.BG_CARD, fg=Theme.ACCENT_RED,
            activebackground=Theme.BG_CARD, activeforeground=Theme.ACCENT_RED,
            selectcolor=Theme.BG_PREVIEW, bd=0, highlightthickness=0,
            font=(Theme.FONT_FAMILY, 10, "bold")).pack(side="left", padx=(2, 8))
        tk.Label(self.manual_toolbar, text="Brush",
                 font=(Theme.FONT_FAMILY, 8),
                 bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(side="left")
        self.brush_size_var = tk.DoubleVar(value=MANUAL_BRUSH_DEFAULT_PX)
        self.brush_size_value_var = tk.StringVar(
            value=f"{MANUAL_BRUSH_DEFAULT_PX}px")
        self.brush_size_scale = ttk.Scale(
            self.manual_toolbar, from_=MANUAL_BRUSH_MIN_PX,
            to=MANUAL_BRUSH_MAX_PX, variable=self.brush_size_var,
            command=self._on_brush_size_slider,
            orient="horizontal", style="Detail.Horizontal.TScale")
        self.brush_size_scale.pack(side="left", fill="x", expand=True,
                                   padx=(6, 6))
        tk.Label(self.manual_toolbar, textvariable=self.brush_size_value_var,
                 font=(Theme.FONT_FAMILY, 8, "bold"),
                 bg=Theme.BG_CARD, fg=Theme.ACCENT_TEAL).pack(side="left")
        self.manual_toolbar.pack_forget()

        self.gcode_preview = GCodePreviewCanvas(self.card_gcode.body)
        self.gcode_preview.pack(fill="both", expand=True)
        self.gcode_preview.set_manual_brush(
            False, callback=self._on_manual_brush_commit)

        # Controls row
        controls_frame = tk.Frame(main, bg=Theme.BG_APP)
        controls_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        controls_frame.columnconfigure(0, weight=1)
        self._build_controls(controls_frame)

        def bind_main_wheel_tree(widget):
            widget.bind("<Enter>", bind_main_wheel, add="+")
            widget.bind("<Leave>", unbind_main_wheel, add="+")
            for child in widget.winfo_children():
                bind_main_wheel_tree(child)

        bind_main_wheel_tree(main)

    def _build_controls(self, parent):
        controls = tk.Frame(parent, bg=Theme.BG_APP)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)

        top_controls = tk.Frame(controls, bg=Theme.BG_APP)
        top_controls.grid(row=0, column=0, sticky="ew")
        top_controls.columnconfigure(1, weight=1)

        self.auto_detail_var = tk.BooleanVar(value=True)
        self.chk_auto = tk.Checkbutton(
            top_controls, text="Auto",
            variable=self.auto_detail_var, command=self._on_auto_toggle,
            bg=Theme.BG_APP, fg=Theme.TEXT_SECOND,
            activebackground=Theme.BG_APP, activeforeground=Theme.TEXT_PRIMARY,
            selectcolor=Theme.BG_CARD, font=(Theme.FONT_FAMILY, 8),
            bd=0, highlightthickness=0)
        self.chk_auto.grid(row=0, column=0, sticky="w")

        self.detail_value_var = tk.StringVar(
            value=f"{DEFAULT_STROKE_BUDGET:,}".replace(",", "."))
        tk.Label(top_controls, textvariable=self.detail_value_var,
                 font=(Theme.FONT_FAMILY, 9, "bold"),
                 bg=Theme.BG_APP, fg=Theme.ACCENT_YELLOW).grid(
                     row=0, column=2, sticky="e")

        self.live_preview_var = tk.BooleanVar(value=False)
        self.chk_live_preview = tk.Checkbutton(
            top_controls, text="Live preview",
            variable=self.live_preview_var, command=self._on_live_preview_toggle,
            bg=Theme.BG_APP, fg=Theme.TEXT_SECOND,
            activebackground=Theme.BG_APP, activeforeground=Theme.TEXT_PRIMARY,
            selectcolor=Theme.BG_CARD, font=(Theme.FONT_FAMILY, 8),
            bd=0, highlightthickness=0)
        self.chk_live_preview.grid(row=0, column=3, sticky="e", padx=(10, 0))

        tk.Label(controls, text="Max detail budget (segments)",
                 font=(Theme.FONT_FAMILY, 8),
                 bg=Theme.BG_APP, fg=Theme.TEXT_MUTED).grid(
                     row=1, column=0, sticky="w", pady=(4, 2))

        self.detail_slider_var = tk.DoubleVar(value=DEFAULT_STROKE_BUDGET)
        self.detail_scale = ttk.Scale(
            controls, from_=MIN_STROKE_BUDGET, to=MAX_STROKE_BUDGET,
            variable=self.detail_slider_var, command=self._on_detail_slider,
            orient="horizontal", style="Detail.Horizontal.TScale")
        self.detail_scale.grid(row=2, column=0, sticky="ew")

        ends = tk.Frame(controls, bg=Theme.BG_APP)
        ends.grid(row=3, column=0, sticky="ew")
        tk.Label(ends, text="300", font=(Theme.FONT_FAMILY, 7),
                 bg=Theme.BG_APP, fg=Theme.TEXT_MUTED).pack(side="left")
        tk.Label(ends, text="30.000", font=(Theme.FONT_FAMILY, 7),
                 bg=Theme.BG_APP, fg=Theme.TEXT_MUTED).pack(side="right")

        hatch_quick = tk.Frame(controls, bg=Theme.BG_APP)
        hatch_quick.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        hatch_quick.columnconfigure(1, weight=1)
        tk.Label(hatch_quick, text="Hatching", font=(Theme.FONT_FAMILY, 8, "bold"),
                 bg=Theme.BG_APP, fg=Theme.TEXT_SECOND).grid(row=0, column=0, sticky="w")
        self.hatch_preset_var = tk.StringVar(value="Balanced")
        self.hatch_preset_combo = ttk.Combobox(
            hatch_quick, textvariable=self.hatch_preset_var,
            values=list(HATCH_PRESETS) + ["Custom"], state="readonly", width=12,
            font=(Theme.FONT_FAMILY, 8))
        self.hatch_preset_combo.grid(row=0, column=1, sticky="e", padx=(8, 6))
        self.hatch_preset_combo.bind("<<ComboboxSelected>>", self._on_hatch_preset)
        self.btn_hatch_advanced = tk.Button(
            hatch_quick, text="Advanced", command=self._toggle_hatch_advanced,
            bg=Theme.BG_CARD, fg=Theme.TEXT_SECOND,
            activebackground=Theme.BORDER, activeforeground=Theme.TEXT_PRIMARY,
            bd=0, padx=8, pady=3, font=(Theme.FONT_FAMILY, 8))
        self.btn_hatch_advanced.grid(row=0, column=2, sticky="e")

        self.contour_hatch_var = tk.BooleanVar(value=True)
        self.chk_contour_hatch = tk.Checkbutton(
            hatch_quick, text="Contour-following hatching",
            variable=self.contour_hatch_var, command=self._on_hatch_slider,
            bg=Theme.BG_APP, fg=Theme.TEXT_SECOND,
            activebackground=Theme.BG_APP, activeforeground=Theme.TEXT_PRIMARY,
            selectcolor=Theme.BG_CARD, font=(Theme.FONT_FAMILY, 8),
            bd=0, highlightthickness=0)
        self.chk_contour_hatch.grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        hatch_grid = tk.Frame(controls, bg=Theme.BG_APP)
        hatch_grid.grid(row=5, column=0, sticky="ew", pady=(4, 0))
        self.hatch_advanced_frame = hatch_grid
        self.hatch_advanced_visible = False
        hatch_grid.columnconfigure(0, weight=1)
        hatch_grid.columnconfigure(1, weight=0)

        self.hatch_value_var = tk.StringVar(value="")
        self.hatch_slider_value_vars = {}

        def add_hatch_slider(row, label, var_name, default, from_, to_, fmt):
            value_var = tk.DoubleVar(value=default)
            label_var = tk.StringVar(value=fmt(default))
            setattr(self, var_name, value_var)
            self.hatch_slider_value_vars[var_name] = (label_var, fmt)
            tk.Label(hatch_grid, text=label, font=(Theme.FONT_FAMILY, 8),
                     bg=Theme.BG_APP, fg=Theme.TEXT_MUTED).grid(
                         row=row * 2, column=0, sticky="w", pady=(4, 0))
            tk.Label(hatch_grid, textvariable=label_var,
                     font=(Theme.FONT_FAMILY, 8, "bold"),
                     bg=Theme.BG_APP, fg=Theme.ACCENT_TEAL).grid(
                         row=row * 2, column=1, sticky="e", padx=(8, 0))
            scale = ttk.Scale(
                hatch_grid, from_=from_, to=to_, variable=value_var,
                command=self._on_hatch_slider,
                orient="horizontal", style="Detail.Horizontal.TScale")
            scale.grid(row=row * 2 + 1, column=0, columnspan=2, sticky="ew")

        add_hatch_slider(0, "Hatch angle", "hatch_angle_var",
                         DEFAULT_HATCH_ANGLE_DEG, 0, 180,
                         lambda v: f"{float(v):.0f}deg")
        add_hatch_slider(1, "Hatch cell size", "hatch_cell_var",
                         DEFAULT_HATCH_CELL_SIZE, 6, 40,
                         lambda v: f"{int(round(float(v)))}px")
        add_hatch_slider(2, "Hatch min spacing", "hatch_min_spacing_var",
                         DEFAULT_HATCH_MIN_SPACING, 2, 12,
                         lambda v: f"{int(round(float(v)))}px")
        add_hatch_slider(3, "Hatch max spacing", "hatch_max_spacing_var",
                         DEFAULT_HATCH_MAX_SPACING, 8, 36,
                         lambda v: f"{int(round(float(v)))}px")
        add_hatch_slider(4, "Hatch dark threshold", "hatch_dark_threshold_var",
                         DEFAULT_HATCH_DARK_THRESHOLD, 100, 250,
                         lambda v: f"{float(v):.0f}")
        self._refresh_hatch_value_label()
        self.hatch_advanced_frame.grid_remove()

        self.show_travel_var = tk.BooleanVar(value=False)
        self.chk_travel = tk.Checkbutton(
            controls, text="Show G0 travel paths",
            variable=self.show_travel_var,
            command=lambda: self.gcode_preview.set_show_travel(
                self.show_travel_var.get()),
            bg=Theme.BG_APP, fg=Theme.TEXT_SECOND,
            activebackground=Theme.BG_APP, activeforeground=Theme.TEXT_PRIMARY,
            selectcolor=Theme.BG_CARD, font=(Theme.FONT_FAMILY, 8),
            bd=0, highlightthickness=0)
        self.chk_travel.grid(row=6, column=0, sticky="w", pady=(6, 0))

        # Stats label — shows optimization metrics
        self.gcode_stats_var = tk.StringVar(
            value="Actual: --  |  Stroke: --  |  Pen lift: --\n"
                  "Travel: -- mm  |  Draw: -- mm  |  Est: --")
        tk.Label(controls, textvariable=self.gcode_stats_var,
                 font=(Theme.FONT_FAMILY, 8), bg=Theme.BG_APP,
                 fg=Theme.TEXT_MUTED, justify="left", anchor="w",
                 wraplength=500).grid(row=7, column=0, sticky="ew", pady=(4, 0))

        self.detail_scale.state(["disabled"])
        self._refresh_apply_button()

        # Status bar
        status_bar = tk.Frame(parent, bg=Theme.BG_SIDEBAR, height=42)
        status_bar.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        status_bar.grid_propagate(False)

        self.status_dot = tk.Canvas(status_bar, width=10, height=10,
                                    bg=Theme.BG_SIDEBAR, highlightthickness=0)
        self.status_dot_id = self.status_dot.create_oval(
            0, 0, 10, 10, fill=Theme.TEXT_MUTED, outline="")
        self.status_dot.pack(side="left", padx=(20, 8))

        self.status_var = tk.StringVar(
            value="Ready. Paste or Upload Image to begin.")
        self.status_var.trace_add("write", lambda *_: self._refresh_status_dot())

        tk.Label(status_bar, textvariable=self.status_var,
                 font=(Theme.FONT_FAMILY, 9),
                 bg=Theme.BG_SIDEBAR, fg=Theme.TEXT_SECOND,
                 anchor="w").pack(side="left", fill="x", expand=True)

    # ─────────────────────────────────────────────────────────────
    def _refresh_status_dot(self):
        text = self.status_var.get()
        if text.startswith("Done") or text.startswith("Ready:"):
            color = Theme.ACCENT_GREEN
        elif text.startswith("Error") or "error" in text.lower():
            color = Theme.ACCENT_RED
        elif text.startswith("Processing") or text.startswith("Building"):
            color = Theme.ACCENT_YELLOW
        elif "Loading" in text or "Step" in text or "Analyzing" in text or "QA" in text:
            color = Theme.ACCENT_BLUE
        else:
            color = Theme.TEXT_MUTED
        self.status_dot.itemconfig(self.status_dot_id, fill=color)

    # ─────────────────────────────────────────────────────────────
    #  Mode selector callback
    # ─────────────────────────────────────────────────────────────
    def _refresh_ai_status(self):
        if hasattr(self, "ai_status_var"):
            self.ai_status_var.set("Input: sketch/line-art only")

    def _refresh_face_controls(self):
        state = tk.NORMAL
        if hasattr(self, "chk_face_detect"):
            self.chk_face_detect.config(state=state)
        if hasattr(self, "face_feather_scale"):
            self.face_feather_scale.state(["!disabled"] if state == tk.NORMAL else ["disabled"])
        if hasattr(self, "face_feather_value_var"):
            self.face_feather_value_var.set(
                f"{int(round(float(self.face_feather_var.get())))}px")

    def _current_face_settings(self):
        enabled = bool(self.face_detect_var.get())
        feather = int(round(float(self.face_feather_var.get())))
        return enabled, max(0, min(24, feather))

    def _on_face_mask_change(self, _event=None):
        self._refresh_face_controls()
        if not self._live_preview_enabled():
            self._mark_manual_dirty(hatch=True)
            return
        if self.input_path:
            self._on_mask_slider()

    def _on_face_feather_slider(self, _value=None):
        self._refresh_face_controls()
        self._on_face_mask_change()

    # ─────────────────────────────────────────────────────────────
    #  Budget controls
    # ─────────────────────────────────────────────────────────────
    def _format_budget(self, value):
        return f"{int(value):,}".replace(",", ".")

    def _current_budget(self):
        value = int(round(self.detail_slider_var.get() / 100.0) * 100)
        return max(MIN_STROKE_BUDGET, min(MAX_STROKE_BUDGET, value))

    def _budget_warning_text(self, budget):
        """Tao goi y/canh bao Max detail budget dua tren do phuc tap anh."""
        if not self.candidate_paths:
            return "Suggested: --"
        suggested = int(max(MIN_STROKE_BUDGET,
                            min(MAX_STROKE_BUDGET, self.auto_budget)))
        raw_segments = max(1, self.last_raw_segments)
        path_count = len(self.candidate_paths)
        if budget < suggested * 0.55:
            level = "LOW: co the mat net nho"
        elif budget > min(MAX_STROKE_BUDGET, suggested * 1.85) and raw_segments > suggested:
            level = "HIGH: nhieu stroke, thoi gian ve lau"
        else:
            level = "OK"
        return (f"Suggested: {self._format_budget(suggested)}"
                f"  |  {level}"
                f"  |  Source paths: {path_count}")

    def _current_hatch_settings(self):
        min_spacing = int(round(self.hatch_min_spacing_var.get()))
        max_spacing = int(round(self.hatch_max_spacing_var.get()))
        if max_spacing < min_spacing:
            max_spacing = min_spacing
            self.hatch_max_spacing_var.set(max_spacing)
        return {
            "cell_size": int(round(self.hatch_cell_var.get())),
            "angle_deg": float(self.hatch_angle_var.get()),
            "min_spacing": min_spacing,
            "max_spacing": max_spacing,
            "dark_threshold": float(self.hatch_dark_threshold_var.get()),
            "contour_following": bool(self.contour_hatch_var.get()),
        }

    def _refresh_hatch_value_label(self):
        settings = self._current_hatch_settings()
        for var_name, (label_var, fmt) in self.hatch_slider_value_vars.items():
            value_var = getattr(self, var_name)
            label_var.set(fmt(value_var.get()))
        self.hatch_value_var.set(
            f"{'Contour' if settings['contour_following'] else 'Fixed'} / "
            f"{settings['angle_deg']:.0f}deg / "
            f"{settings['cell_size']}px / "
            f"{settings['min_spacing']}-{settings['max_spacing']}px / "
            f"T{settings['dark_threshold']:.0f}"
        )

    def _live_preview_enabled(self):
        return bool(self.live_preview_var.get())

    def _pending_brush_canvas_size(self):
        if self.pil_lineart is not None:
            return self.pil_lineart.size
        if self.pil_original is not None:
            return self.pil_original.size
        return None

    def _has_pending_brush(self):
        for mask in getattr(self, "_pending_brush_masks", {}).values():
            if mask is not None and cv2.countNonZero(np.asarray(mask, dtype=np.uint8)) > 0:
                return True
        return False

    def _pending_brush_count(self):
        return sum(
            cv2.countNonZero(np.asarray(mask, dtype=np.uint8))
            for mask in getattr(self, "_pending_brush_masks", {}).values()
            if mask is not None
        )

    def _merge_pending_brush_mask(self, mode, brush_mask):
        mode = "reduce" if str(mode).lower().startswith("reduce") else "add"
        self._pending_brush_mode = mode
        canvas_size = self._pending_brush_canvas_size()
        if canvas_size is None:
            return
        width, height = _normalize_canvas_size(canvas_size)
        brush = _resize_binary_mask(brush_mask, width, height)
        if brush is None or cv2.countNonZero(brush) == 0:
            return
        current = self._pending_brush_masks.get(mode)
        current = _resize_binary_mask(current, width, height)
        if current is None:
            current = np.zeros((height, width), dtype=np.uint8)
        self._pending_brush_masks[mode] = cv2.bitwise_or(current, brush)

    def _refresh_pending_brush_overlay(self):
        if not hasattr(self, "gcode_preview"):
            return
        canvas_size = self._pending_brush_canvas_size()
        if canvas_size is None:
            self.gcode_preview.set_pending_brush_overlay(None, None, None)
            return
        add_mask = self._pending_brush_masks.get("add")
        reduce_mask = self._pending_brush_masks.get("reduce")
        self.gcode_preview.set_pending_brush_overlay(
            add_mask=add_mask,
            reduce_mask=reduce_mask,
            canvas_size=canvas_size,
        )

    def _clear_pending_brush(self, silent=False):
        self._pending_brush_masks = {"add": None, "reduce": None}
        self._pending_brush_mode = (
            self.manual_mode_var.get() if hasattr(self, "manual_mode_var") else "add"
        )
        self._pending_brush_max_size_px = 0
        self._refresh_pending_brush_overlay()
        self._refresh_apply_button()
        self._refresh_export_buttons()
        if not silent:
            self._show_toast("Pending brush cleared")

    def _on_clear_brush(self):
        if self._has_pending_brush():
            self._clear_pending_brush()

    def _on_escape_key(self, _event=None):
        if self._has_pending_brush():
            self._clear_pending_brush()
            return "break"
        return None

    def _refresh_apply_button(self):
        if not hasattr(self, "btn_apply_changes"):
            return
        has_pending_brush = self._has_pending_brush()
        if self._live_preview_enabled() and not has_pending_brush:
            self.btn_apply_changes.pack_forget()
            return
        if not self.btn_apply_changes.winfo_ismapped():
            self.btn_apply_changes.pack(pady=(10, 0))
        dirty = self._manual_detail_dirty or self._manual_hatch_dirty or has_pending_brush
        self.btn_apply_changes.config(state=tk.NORMAL if dirty else tk.DISABLED)

    def _qa_allows_export(self):
        report = self.current_qa_report
        return bool(
            self.current_plan is not None and
            report is not None and
            len(report.iterations) >= QA_MIN_ITERATIONS and
            report.passed
        )

    def _format_qa_report(self, report=None):
        report = self.current_qa_report if report is None else report
        if report is None:
            return "QA: waiting for preview"
        state = "PASS" if report.passed else "REVIEW"
        lines = [f"QA: {state} overlay ({report.total_time_s:.1f}s)"]
        for iteration in report.iterations:
            lines.append(
                f"{iteration.index}. C{iteration.ssim:.2f} | "
                f"{iteration.stroke_count} strokes | {iteration.pen_lifts} lifts"
            )
        if report.model_status:
            lines.append(report.model_status[:64])
        return "\n".join(lines)

    def _refresh_export_buttons(self):
        if not hasattr(self, "btn_export") or not hasattr(self, "btn_force_export"):
            return
        has_plan = self.current_plan is not None
        can_export = self._qa_allows_export()
        self.btn_export.config(state=tk.NORMAL if can_export else tk.DISABLED)
        self.btn_force_export.config(
            state=tk.NORMAL if has_plan and not can_export else tk.DISABLED)
        if hasattr(self, "btn_manual_undo"):
            self.btn_manual_undo.config(
                state=tk.NORMAL if self.manual_undo_stack else tk.DISABLED)
        if hasattr(self, "btn_manual_reset"):
            self.btn_manual_reset.config(
                state=tk.NORMAL if self.manual_adjust_has_changes else tk.DISABLED)
        if hasattr(self, "btn_clear_brush"):
            self.btn_clear_brush.config(
                state=tk.NORMAL if self._has_pending_brush() else tk.DISABLED)
        if hasattr(self, "qa_status_var") and self.current_qa_report is not None:
            self.qa_status_var.set(self._format_qa_report(self.current_qa_report))

    def _refresh_manual_brush(self):
        if not hasattr(self, "gcode_preview"):
            return
        enabled = (
            hasattr(self, "preview_mode_var") and
            self.preview_mode_var.get() == "Manual Adjust"
        )
        self.manual_adjust_active = bool(enabled)
        size_px = int(round(float(self.brush_size_var.get()))) if hasattr(self, "brush_size_var") else MANUAL_BRUSH_DEFAULT_PX
        if hasattr(self, "brush_size_value_var"):
            mm_text = ""
            try:
                radius_mm = self.gcode_preview.brush_radius_mm()
                mm_text = f" / {radius_mm * 2.0:.1f}mm"
            except Exception:
                pass
            self.brush_size_value_var.set(f"{size_px}px{mm_text}")
        mode = self.manual_mode_var.get() if hasattr(self, "manual_mode_var") else "add"
        self.gcode_preview.set_manual_brush(
            enabled and self.current_plan is not None,
            size_px=size_px,
            mode=mode,
            callback=self._on_manual_brush_commit,
        )
        if hasattr(self, "manual_toolbar"):
            if enabled:
                self.manual_toolbar.pack(side="left", fill="x", expand=True)
            else:
                self.manual_toolbar.pack_forget()
        self._refresh_pending_brush_overlay()

    def _on_preview_mode_change(self, _event=None):
        if self.preview_mode_var.get() == "Manual Adjust" and self.current_plan is None:
            self.preview_mode_var.set("Auto")
            self._show_toast("Load an image and wait for preview first")
        self._refresh_manual_brush()

    def _on_brush_size_slider(self, _value=None):
        value = int(round(float(self.brush_size_var.get())))
        value = max(MANUAL_BRUSH_MIN_PX, min(MANUAL_BRUSH_MAX_PX, value))
        self.brush_size_var.set(value)
        self._refresh_manual_brush()

    def _on_manual_brush_commit(self, points_mm, radius_mm, mode):
        if self.pil_original is None or self.candidate_paths is None:
            return
        canvas_size = self.pil_lineart.size if self.pil_lineart is not None else self.pil_original.size
        brush_mask = make_brush_mask(points_mm, radius_mm, canvas_size)
        if cv2.countNonZero(brush_mask) == 0:
            return
        self._merge_pending_brush_mask(mode, brush_mask)
        if hasattr(self, "brush_size_var"):
            self._pending_brush_max_size_px = max(
                self._pending_brush_max_size_px,
                int(round(float(self.brush_size_var.get()))),
            )
        self.current_qa_report = None
        self._qa_job_id += 1
        if self.current_plan is not None:
            self.current_plan.qa_report = None
        if hasattr(self, "qa_status_var"):
            self.qa_status_var.set("QA: brush pending, apply to re-check")
        self._refresh_pending_brush_overlay()
        self._refresh_apply_button()
        self._refresh_export_buttons()
        self._show_toast(
            f"Brush region pending ({'add' if mode == 'add' else 'reduce'})")
        self.status_var.set("Manual Adjust: brush region pending; click Apply changes")

    def _manual_brush_aggressiveness(self):
        size_px = self._pending_brush_max_size_px
        if size_px <= 0 and hasattr(self, "brush_size_var"):
            size_px = int(round(float(self.brush_size_var.get())))
        return "strong" if size_px >= max(78, int(MANUAL_BRUSH_MAX_PX * 0.52)) else "normal"

    def _apply_pending_manual_brush(self):
        if self.pil_original is None or self.candidate_paths is None:
            return
        pending = {}
        for mode, mask in self._pending_brush_masks.items():
            if mask is not None and cv2.countNonZero(np.asarray(mask, dtype=np.uint8)) > 0:
                pending[mode] = np.asarray(mask, dtype=np.uint8).copy()
        if not pending:
            return

        allow_face = False
        reduce_mask = pending.get("reduce")
        if reduce_mask is not None and brush_overlaps_face(reduce_mask, self.current_face_mask):
            self._show_toast(
                "Protected face region: confirm on Apply to reduce it",
                duration_ms=2600,
            )
            allow_face = messagebox.askyesno(
                "Reduce Face Detail?",
                "This pending reduce brush overlaps the protected face region.\n\n"
                "Reducing strokes here may remove eyes, nose, mouth, or face contour detail.\n"
                "Apply the reduction to face strokes anyway?")

        self.manual_undo_stack.append(_clone_candidate_list(self.candidate_paths))
        if len(self.manual_undo_stack) > 20:
            self.manual_undo_stack = self.manual_undo_stack[-20:]
        self.status_var.set("Manual Adjust: applying pending brush changes...")
        self.card_gcode.set_status("Manual...", Theme.ACCENT_BLUE)
        for btn in self._action_buttons():
            btn.config(state=tk.DISABLED)
        candidates = _clone_candidate_list(self.candidate_paths)
        pil_source = self.pil_original.copy()
        face_mask = None if self.current_face_mask is None else self.current_face_mask.copy()
        aggressiveness = self._manual_brush_aggressiveness()
        threading.Thread(
            target=self._run_manual_adjustment,
            args=(candidates, pil_source, pending, face_mask, allow_face, aggressiveness),
            daemon=True,
        ).start()

    def _run_manual_adjustment(self, candidates, pil_source, pending_masks,
                               face_mask, allow_face, aggressiveness):
        try:
            updated = candidates
            combined = {
                "mode": "brush",
                "added": 0,
                "removed": 0,
                "merged": 0,
                "passes": [],
                "aggressiveness": aggressiveness,
            }
            for mode in ("reduce", "add"):
                brush_mask = pending_masks.get(mode)
                if brush_mask is None or cv2.countNonZero(brush_mask) == 0:
                    continue
                updated, stats = apply_manual_brush_adjustment(
                    updated, pil_source, brush_mask, mode,
                    face_mask=face_mask,
                    allow_face_reduce=(allow_face if mode == "reduce" else False),
                    aggressiveness=aggressiveness,
                )
                combined["passes"].append(stats)
                combined["added"] += int(stats.get("added", 0))
                combined["removed"] += int(stats.get("removed", 0))
                combined["merged"] += int(stats.get("merged", 0))
            self.root.after(
                0, lambda: self._on_manual_adjustment_done(updated, combined))
        except Exception as error:
            self.root.after(
                0, lambda: self._on_manual_adjustment_error(str(error)))

    def _format_manual_stats_toast(self, stats):
        added = int(stats.get("added", 0))
        removed = int(stats.get("removed", 0))
        if added or removed:
            return f"Manual brush: +{added} / -{removed}"
        passes = stats.get("passes") or [stats]
        reasons = {
            "protected_face": 0,
            "protected_outline": 0,
            "low_overlap": 0,
            "score_high": 0,
            "overlapped": 0,
        }
        for item in passes:
            diagnostics = item.get("diagnostics", {}) or {}
            for key in reasons:
                reasons[key] += int(diagnostics.get(key, 0))
        if reasons["protected_face"] > 0:
            return (
                f"0 removed - {reasons['protected_face']} strokes overlap protected face; "
                "confirm face reduce to delete"
            )
        if reasons["protected_outline"] > 0:
            return (
                f"0 removed - {reasons['protected_outline']} main outline strokes protected"
            )
        if reasons["overlapped"] > 0:
            return (
                f"0 changed - {reasons['overlapped']} strokes touched brush "
                f"but overlap/score was below remove threshold"
            )
        return "Manual brush: no matching strokes found"

    def _on_manual_adjustment_done(self, candidates, stats):
        self._clear_pending_brush(silent=True)
        self.candidate_paths = _consolidate_dense_micro_candidates(candidates)
        self.manual_adjust_has_changes = True
        self.current_plan = None
        self.current_plan_budget = None
        self._plan_cache.clear()
        self._qa_cache.clear()
        self._refresh_export_buttons()
        self._refresh_manual_brush()
        self._show_toast(self._format_manual_stats_toast(stats), duration_ms=3200)
        self.status_var.set("Manual Adjust: preview updating...")
        self._schedule_preview_generation(0)

    def _on_manual_adjustment_error(self, msg):
        self.status_var.set(f"Manual Adjust error: {msg}")
        self.card_gcode.set_status("Manual error", Theme.ACCENT_RED)
        for btn in self._action_buttons():
            btn.config(state=tk.NORMAL)
        self._refresh_export_buttons()

    def _on_manual_undo(self):
        if not self.manual_undo_stack:
            return
        self._clear_pending_brush(silent=True)
        self.candidate_paths = self.manual_undo_stack.pop()
        self.manual_adjust_has_changes = bool(self.manual_undo_stack)
        self.current_plan = None
        self.current_plan_budget = None
        self._plan_cache.clear()
        self._qa_cache.clear()
        self._show_toast("Undo brush")
        self._refresh_export_buttons()
        self._refresh_manual_brush()
        self._schedule_preview_generation(0)

    def _on_manual_reset(self):
        if not self.auto_candidate_paths:
            return
        self._clear_pending_brush(silent=True)
        self.candidate_paths = _clone_candidate_list(self.auto_candidate_paths)
        self.manual_undo_stack.clear()
        self.manual_adjust_has_changes = False
        self.current_plan = None
        self.current_plan_budget = None
        self._plan_cache.clear()
        self._qa_cache.clear()
        self._show_toast("Reset to Auto")
        self._refresh_export_buttons()
        self._refresh_manual_brush()
        self._schedule_preview_generation(0)

    def _mark_manual_dirty(self, hatch=False, detail=False):
        self._manual_hatch_dirty = self._manual_hatch_dirty or hatch
        self._manual_detail_dirty = self._manual_detail_dirty or detail
        if hatch or detail:
            self.current_qa_report = None
            self._qa_job_id += 1
            if hasattr(self, "qa_status_var"):
                self.qa_status_var.set("QA: settings changed, apply to re-check")
            if self.current_plan is not None:
                self.current_plan.qa_report = None
            self._refresh_export_buttons()
        self._refresh_apply_button()

    def _on_live_preview_toggle(self):
        if self._live_preview_enabled():
            self._manual_detail_dirty = False
            self._manual_hatch_dirty = False
            self._refresh_apply_button()
            if self.input_path:
                self._process_image(self.input_path)
            elif self.candidate_paths is not None:
                self._schedule_preview_generation(120)
        else:
            if self._hatch_after_id is not None:
                try:
                    self.root.after_cancel(self._hatch_after_id)
                except Exception:
                    pass
                self._hatch_after_id = None
            if self._preview_after_id is not None:
                try:
                    self.root.after_cancel(self._preview_after_id)
                except Exception:
                    pass
                self._preview_after_id = None
            self._refresh_apply_button()

    def _on_apply_changes(self):
        if self._has_pending_brush():
            self._manual_detail_dirty = False
            self._apply_pending_manual_brush()
            return
        if self._live_preview_enabled():
            return
        hatch_dirty = self._manual_hatch_dirty
        detail_dirty = self._manual_detail_dirty
        self._manual_detail_dirty = False
        self._manual_hatch_dirty = False
        self._refresh_apply_button()

        if hatch_dirty and self.input_path:
            self._process_image(self.input_path)
        elif detail_dirty and self.candidate_paths is not None:
            self._schedule_preview_generation(0)

    def _on_hatch_slider(self, _value=None):
        if not getattr(self, "_applying_hatch_preset", False):
            self.hatch_preset_var.set("Custom")
        self._refresh_hatch_value_label()
        if not self._live_preview_enabled():
            self._mark_manual_dirty(hatch=True)
            return
        if self.input_path:
            if self._hatch_after_id is not None:
                try:
                    self.root.after_cancel(self._hatch_after_id)
                except Exception:
                    pass
            self._hatch_after_id = self.root.after(
                700, lambda: self._process_image(self.input_path))

    def _on_mask_slider(self, _value=None):
        if not self._live_preview_enabled():
            self._mark_manual_dirty(hatch=True)
            return
        if self.input_path:
            if self._hatch_after_id is not None:
                try:
                    self.root.after_cancel(self._hatch_after_id)
                except Exception:
                    pass
            self._hatch_after_id = self.root.after(
                700, lambda: self._process_image(self.input_path))

    def _on_hatch_preset(self, _event=None):
        settings = HATCH_PRESETS.get(self.hatch_preset_var.get())
        if settings is None:
            return
        self._applying_hatch_preset = True
        try:
            self.hatch_cell_var.set(settings["cell_size"])
            self.hatch_angle_var.set(settings["angle_deg"])
            self.hatch_min_spacing_var.set(settings["min_spacing"])
            self.hatch_max_spacing_var.set(settings["max_spacing"])
            self.hatch_dark_threshold_var.set(settings["dark_threshold"])
            self._refresh_hatch_value_label()
        finally:
            self._applying_hatch_preset = False
        if not self._live_preview_enabled():
            self._mark_manual_dirty(hatch=True)
        elif self.input_path:
            self._on_mask_slider()

    def _toggle_hatch_advanced(self):
        self.hatch_advanced_visible = not self.hatch_advanced_visible
        if self.hatch_advanced_visible:
            self.hatch_advanced_frame.grid()
            self.btn_hatch_advanced.config(text="Hide")
        else:
            self.hatch_advanced_frame.grid_remove()
            self.btn_hatch_advanced.config(text="Advanced")

    def _on_detail_slider(self, value):
        rounded = int(round(float(value) / 100.0) * 100)
        rounded = max(MIN_STROKE_BUDGET, min(MAX_STROKE_BUDGET, rounded))
        self.detail_value_var.set(self._format_budget(rounded))
        if self.candidate_paths is not None and self.current_plan is None:
            self.gcode_stats_var.set(self._budget_warning_text(rounded))
        if not self._live_preview_enabled():
            if not self.auto_detail_var.get():
                self._mark_manual_dirty(detail=True)
            return
        if not self.auto_detail_var.get() and self.candidate_paths is not None:
            self._schedule_preview_generation(350)

    def _on_auto_toggle(self):
        if self.auto_detail_var.get():
            self.detail_scale.state(["disabled"])
            self.detail_slider_var.set(self.auto_budget)
            self.detail_value_var.set(self._format_budget(self.auto_budget))
        else:
            self.detail_scale.state(["!disabled"])
        if not self._live_preview_enabled():
            self._mark_manual_dirty(detail=True)
            return
        if self.candidate_paths is not None:
            self._schedule_preview_generation(120)

    # ─────────────────────────────────────────────────────────────
    def _focus_wants_text_paste(self, widget):
        try:
            klass = widget.winfo_class()
        except Exception:
            return False
        return klass in {"Entry", "Text", "TEntry", "Spinbox", "TSpinbox"}

    def _show_toast(self, text, duration_ms=1800):
        try:
            toast = tk.Toplevel(self.root)
            toast.overrideredirect(True)
            toast.configure(bg=Theme.BG_CARD)
            tk.Label(
                toast, text=text,
                bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY,
                font=(Theme.FONT_FAMILY, 9), padx=14, pady=8,
                highlightthickness=1, highlightbackground=Theme.BORDER,
            ).pack()
            self.root.update_idletasks()
            x = self.root.winfo_rootx() + self.root.winfo_width() - 300
            y = self.root.winfo_rooty() + 70
            toast.geometry(f"+{max(0, x)}+{max(0, y)}")
            toast.after(duration_ms, toast.destroy)
        except Exception:
            self.status_var.set(text)

    def _set_input_thumbnail(self, pil_img, filename=None):
        if not hasattr(self, "input_thumb_label"):
            return
        if pil_img is None:
            self._input_thumb_tk = None
            self.input_thumb_label.config(image="", text="No image")
            return
        thumb = pil_img.convert("RGB").copy()
        resampling = getattr(Image, "Resampling", None)
        resample = getattr(
            resampling, "LANCZOS",
            getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3)),
        )
        thumb.thumbnail((54, 54), resample)
        tk_img = ImageTk.PhotoImage(thumb)
        self._input_thumb_tk = tk_img
        label = filename or "Clipboard image"
        self.input_thumb_label.config(
            image=tk_img,
            text=f"  {label}",
            compound="left",
            fg=Theme.TEXT_SECOND,
        )

    def _process_clipboard_image(self, pil_img):
        tmp = tempfile.NamedTemporaryFile(
            prefix="clipboard_", suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        pil_img.convert("RGBA").save(tmp_path)
        self._show_toast("Đã dán ảnh từ clipboard")
        print(f"Clipboard paste: image saved to {tmp_path}")
        self._process_image(tmp_path)

    def _on_paste_image(self, event=None):
        if event is not None and self._focus_wants_text_paste(event.widget):
            return None
        try:
            clipboard = ImageGrab.grabclipboard()
        except Exception as error:
            print(f"Clipboard paste unavailable: {error}")
            return None

        if isinstance(clipboard, Image.Image):
            self._process_clipboard_image(clipboard)
            return "break"
        if isinstance(clipboard, list):
            for item in clipboard:
                path = str(item)
                if os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                    self._show_toast("Đã dán ảnh từ clipboard")
                    self._process_image(path)
                    return "break"
        return None

    def _on_upload(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        self._process_image(path)

    # ─────────────────────────────────────────────────────────────
    #  Shared pipeline entry
    # ─────────────────────────────────────────────────────────────
    def _reset_gcode_state(self):
        self.candidate_paths     = None
        self.auto_candidate_paths = None
        self.current_plan        = None
        self.current_plan_budget = None
        self.current_qa_report   = None
        self.current_face_mask   = None
        self.current_foreground_mask_info = ForegroundMaskInfo()
        self.manual_undo_stack.clear()
        self.manual_adjust_has_changes = False
        self._clear_pending_brush(silent=True)
        if hasattr(self, "preview_mode_var"):
            self.preview_mode_var.set("Auto")
        if hasattr(self, "gcode_preview"):
            self.gcode_preview.set_manual_brush(False)
        self._plan_cache.clear()
        self._qa_cache.clear()
        self._preview_job_id    += 1
        self._qa_job_id         += 1
        if hasattr(self, "qa_status_var"):
            self.qa_status_var.set("QA: waiting for preview")
        self._refresh_export_buttons()
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
            self._preview_after_id = None

    def _process_image(self, path):
        self.input_path    = path
        self._image_job_id += 1
        job_id             = self._image_job_id
        hatch_settings     = self._current_hatch_settings()
        face_settings      = self._current_face_settings()
        exclude_settings   = (
            self.exclude_mark_var.get(),
            float(self.exclude_bottom_var.get()),
            float(self.exclude_right_var.get()),
        )

        self._reset_gcode_state()

        self.pil_original = None
        self.pil_work     = None
        self.pil_lineart  = None

        self.gcode_preview.clear("Generating G-code...")
        self.card_gcode.set_status("Waiting", Theme.TEXT_MUTED)

        for btn in self._action_buttons():
            btn.config(state=tk.DISABLED)
        self.progress.grid()
        self.progress.start(10)

        fname = os.path.basename(path)
        self.file_info_var.set(f"File: {fname}")
        self.status_var.set(f"Loading image: {fname}")

        threading.Thread(
            target=self._run_pipeline,
            args=(job_id, path, hatch_settings, exclude_settings, face_settings),
            daemon=True).start()

    # ─────────────────────────────────────────────────────────────
    #  Background pipeline thread
    # ─────────────────────────────────────────────────────────────
    def _run_pipeline(self, job_id, path, hatch_settings, exclude_settings,
                      face_settings):
        try:
            # Step 1: load original
            with Image.open(path) as opened:
                pil_orig = ImageOps.exif_transpose(opened).copy()
            self.root.after(0, lambda: self._on_original_loaded(job_id, pil_orig))
            if job_id != self._image_job_id:
                return

            # Xoa truoc vung watermark/logo o goc duoi phai de pipeline khong vector hoa no.
            use_mask, bottom_pct, right_pct = exclude_settings
            pil_input = (_apply_exclusion_mask(pil_orig, bottom_pct, right_pct)
                         if use_mask else pil_orig)

            face_mask = None
            face_enabled, face_feather_px = face_settings
            if face_enabled:
                self.root.after(0, lambda: self.status_var.set(
                    "Step 1/3: Detecting face region..."))
                face_mask = detect_face_region(
                    pil_input, feather_px=face_feather_px)

            ai_maps = None
            pil_work = pil_input.convert("RGB")
            self.root.after(0, lambda: self._on_work_image_done(job_id, pil_work))
            if job_id != self._image_job_id:
                return

            self.root.after(0, lambda: self.status_var.set(
                "Step 2/3: Generating line art and hatch vectors..."))
            pil_lineart, hatch_lines_px, fg_info = transform_to_lineart(
                pil_input,
                hatch_settings,
                return_hatch_vectors=True,
                ai_maps=None,
                exclude_mask=face_mask,
                return_mask_info=True,
            )

            self.root.after(0, lambda info=fg_info: self._on_foreground_mask_info(
                job_id, info))
            self.root.after(0, lambda: self._on_lineart_done(job_id, pil_lineart))
            if job_id != self._image_job_id:
                return

            self.root.after(0, lambda: self.status_var.set(
                "Step 3/3: Analyzing vector paths..."))
            candidates = extract_candidate_paths(
                pil_lineart,
                reference_img=pil_work,
                ai_maps=ai_maps,
                face_mask=face_mask,
            )
            candidates = _consolidate_dense_micro_candidates(candidates)
            hatch_candidate_count = 0
            if hatch_lines_px:
                # Dua hatch vector vao sau extractor de khong bi raster/skeleton tach vo thanh nhieu stroke.
                hatch_candidates = _hatch_vectors_to_candidates(
                    hatch_lines_px,
                    pil_lineart.size[0],
                    pil_lineart.size[1],
                    ai_hatch_map=None,
                )
                candidates.extend(hatch_candidates)
                hatch_candidate_count = len(hatch_candidates)
                print(
                    f"append hatch vectors: {len(hatch_lines_px)} polylines -> "
                    f"{len(hatch_candidates)} candidates")
            context = StrokeClassificationContext(
                canvas_size=pil_lineart.size,
                reference_img=pil_work,
                lineart_img=pil_lineart,
                ai_maps=None,
                face_mask=face_mask,
            )
            candidates = classify_strokes(candidates, context)
            ai_backend = "classical/sketch"
            ai_elapsed_s = 0.0
            self.root.after(0, lambda: self._on_paths_done(
                job_id, candidates, ai_backend, ai_elapsed_s,
                hatch_candidate_count, face_mask, fg_info))

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            self.root.after(0, lambda: self._on_pipeline_error(job_id, str(e)))

    # ─────────────────────────────────────────────────────────────
    #  Pipeline callbacks (main thread)
    # ─────────────────────────────────────────────────────────────
    def _on_original_loaded(self, job_id, pil_orig):
        if job_id != self._image_job_id:
            return
        self.pil_original = pil_orig
        self._set_input_thumbnail(pil_orig, os.path.basename(self.input_path or "image"))

    def _on_work_image_done(self, job_id, pil_work):
        if job_id != self._image_job_id:
            return
        self.pil_work = pil_work

    def _on_foreground_mask_info(self, job_id, mask_info):
        if job_id != self._image_job_id:
            return
        self.current_foreground_mask_info = mask_info or ForegroundMaskInfo()
        info = self.current_foreground_mask_info
        if info.fallback_full:
            text = "Warning: no alpha/foreground mask; background may hatch"
            if hasattr(self, "ai_status_var"):
                self.ai_status_var.set(text)
            self._show_toast(text, duration_ms=3200)
        elif info.estimated and hasattr(self, "ai_status_var"):
            self.ai_status_var.set(
                f"Foreground: estimated {info.foreground_fraction * 100.0:.0f}% "
                f"({info.source})")

    def _on_lineart_done(self, job_id, pil_lineart):
        if job_id != self._image_job_id:
            return
        self.pil_lineart = pil_lineart

    def _on_paths_done(self, job_id, candidates, ai_backend="classical",
                       ai_elapsed_s=0.0, hatch_candidate_count=0,
                       face_mask=None, foreground_mask_info=None):
        if job_id != self._image_job_id:
            return
        self.candidate_paths = candidates
        self.auto_candidate_paths = _clone_candidate_list(candidates)
        self.manual_undo_stack.clear()
        self.manual_adjust_has_changes = False
        self.current_ai_backend = ai_backend
        self.current_ai_elapsed_s = float(ai_elapsed_s)
        self.current_hatch_candidate_count = int(hatch_candidate_count)
        self.current_face_mask = face_mask
        if foreground_mask_info is not None:
            self.current_foreground_mask_info = foreground_mask_info

        raw_total = sum(max(1, len(c.points) - 1) for c in candidates)
        self.last_raw_segments = raw_total
        self.auto_budget = suggest_auto_detail_budget(candidates)
        if self.auto_detail_var.get():
            self.detail_slider_var.set(self.auto_budget)
            self.detail_value_var.set(self._format_budget(self.auto_budget))

        self.progress.stop()
        self.progress.grid_remove()
        for btn in self._action_buttons():
            btn.config(state=tk.NORMAL)
        self._refresh_apply_button()
        self._refresh_export_buttons()
        self._refresh_manual_brush()

        self.status_var.set(
            f"Ready: {len(candidates)} paths — building G-code preview...")
        self.gcode_stats_var.set(self._budget_warning_text(self._current_budget()))
        self._schedule_preview_generation(0)

    def _on_pipeline_error(self, job_id, msg):
        if job_id != self._image_job_id:
            return
        self.progress.stop()
        self.progress.grid_remove()
        for btn in self._action_buttons():
            btn.config(state=tk.NORMAL)
        self._refresh_apply_button()
        self._refresh_export_buttons()
        self.status_var.set(f"Error: {msg}")
        messagebox.showerror("Processing Error", f"An error occurred:\n{msg}")

    # ─────────────────────────────────────────────────────────────
    #  G-Code preview (debounced)
    # ─────────────────────────────────────────────────────────────
    def _schedule_preview_generation(self, delay_ms=300):
        self.current_qa_report = None
        self._qa_job_id += 1
        if self.current_plan is not None:
            self.current_plan.qa_report = None
        if hasattr(self, "qa_status_var"):
            self.qa_status_var.set("QA: waiting for preview")
        self._refresh_export_buttons()
        if self._preview_after_id is not None:
            try:
                self.root.after_cancel(self._preview_after_id)
            except Exception:
                pass
        self._preview_after_id = self.root.after(
            delay_ms, self._start_preview_generation)

    def _start_preview_generation(self):
        self._preview_after_id = None
        if not self.candidate_paths:
            return

        budget = self._current_budget()
        self.detail_slider_var.set(budget)
        self.detail_value_var.set(self._format_budget(budget))

        cached = self._plan_cache.get(budget)
        if cached is not None:
            self._apply_plan_to_preview(cached, budget)
            return

        self._preview_job_id += 1
        preview_id   = self._preview_job_id
        image_job_id = self._image_job_id

        self.card_gcode.set_status("Building...", Theme.ACCENT_BLUE)
        self.gcode_preview.clear("Building G-code preview...")
        self.status_var.set(
            f"Building G-code preview with budget "
            f"{self._format_budget(budget)} segments...")

        candidates = list(self.candidate_paths)
        threading.Thread(target=self._run_preview_plan,
                         args=(image_job_id, preview_id, budget, candidates),
                         daemon=True).start()

    def _run_preview_plan(self, image_job_id, preview_id, budget, candidates):
        try:
            plan = build_gcode_plan(candidates, budget)
            self.root.after(0, lambda: self._on_preview_plan_ready(
                image_job_id, preview_id, budget, plan))
        except Exception as e:
            self.root.after(0, lambda: self._on_preview_plan_error(
                image_job_id, preview_id, str(e)))

    def _on_preview_plan_ready(self, image_job_id, preview_id, budget, plan):
        if image_job_id != self._image_job_id or preview_id != self._preview_job_id:
            return
        self._plan_cache[budget] = plan
        self._apply_plan_to_preview(plan, budget)

    def _apply_plan_to_preview(self, plan, budget, start_qa=True):
        plan.ai_backend = self.current_ai_backend
        plan.ai_elapsed_s = self.current_ai_elapsed_s
        plan.hatch_candidate_count = self.current_hatch_candidate_count
        self.current_plan        = plan
        self.current_plan_budget = budget
        self.current_qa_report   = plan.qa_report

        self.gcode_preview.set_gcode(plan.gcode_lines)
        self.gcode_preview.set_show_travel(self.show_travel_var.get())
        self.card_gcode.set_status("Done", Theme.ACCENT_GREEN)

        rdp_warning_text = (
            "\nDetail warning: too much hair/texture for this budget; "
            "increase the segment budget or simplify fine texture in the source."
            if plan.rdp_eps_warning else ""
        )

        # ── Extended stats display [NEW] ──────────────────────
        self.gcode_stats_var.set(
            f"Actual: {self._format_budget(plan.actual_segments)}"
            f"  |  Stroke: {plan.stroke_count}"
            f"  |  Pen lift: {plan.pen_lifts}\n"
            f"Stitched: {plan.stitched_count}"
            f"  |  Travel: {plan.travel_distance_mm:.1f} mm"
            f"  |  Draw: {plan.draw_distance_mm:.1f} mm"
            f"  |  Est: {_format_duration(plan.estimated_time_s)}\n"
            f"Compressed: -{plan.compression_removed_count}"
            f"  |  Merged: {plan.stroke_merge_count}"
            f"  |  Overlap: -{plan.overlap_removed_count}\n"
            f"  |  Route saved: {plan.route_improvement_mm:.1f} mm"
            f"  |  RDP eps: {plan.used_epsilon:.2f} mm\n"
            f"Face: {plan.selected_detail_tier_counts.get('face', 0)}"
            f"  |  Garment outline: {plan.selected_detail_tier_counts.get('garment_outline', 0)}"
            f"  |  Small detail: {plan.selected_detail_tier_counts.get('small_detail', 0)}\n"
            f"AI: {plan.ai_backend}"
            f"  |  File: {plan.gcode_size_bytes / 1024.0:.1f} KB"
            f"  |  Plan: {plan.planning_time_s:.2f}s\n"
            f"{self._budget_warning_text(budget)}"
            f"{rdp_warning_text}"
        )
        self.status_var.set(
            f"Ready: {self._format_budget(plan.actual_segments)} segments "
            f"(est. {_format_duration(plan.estimated_time_s)})."
        )
        self._refresh_export_buttons()
        self._refresh_manual_brush()
        if start_qa:
            self._start_quality_gate(plan, budget)

    def _start_quality_gate(self, plan, budget):
        if self.pil_lineart is None or not self.candidate_paths:
            return
        if plan.qa_report is not None:
            self.current_qa_report = plan.qa_report
            if hasattr(self, "qa_status_var"):
                self.qa_status_var.set(self._format_qa_report(plan.qa_report))
            self._refresh_export_buttons()
            return

        self._qa_job_id += 1
        qa_id = self._qa_job_id
        image_job_id = self._image_job_id
        self.current_qa_report = None
        self.qa_status_var.set("QA: running overlay check...")
        self.card_gcode.set_status("QA...", Theme.ACCENT_BLUE)
        self._refresh_export_buttons()

        candidates = list(self.candidate_paths)
        reference_lineart = self.pil_lineart.copy()
        original = self.pil_original.copy() if self.pil_original is not None else reference_lineart.copy()
        face_mask = None if self.current_face_mask is None else self.current_face_mask.copy()
        threading.Thread(
            target=self._run_quality_gate,
            args=(image_job_id, qa_id, budget, candidates, reference_lineart,
                  original, face_mask, plan),
            daemon=True).start()

    def _run_quality_gate(self, image_job_id, qa_id, budget, candidates,
                          reference_lineart, original, face_mask, initial_plan):
        try:
            plan, report = run_quality_gate(
                candidates,
                budget,
                reference_lineart,
                original_img=original,
                face_mask=face_mask,
                initial_plan=initial_plan,
                timeout_s=QA_TIMEOUT_S,
            )
            self.root.after(0, lambda: self._on_quality_gate_done(
                image_job_id, qa_id, budget, plan, report))
        except Exception as error:
            self.root.after(0, lambda: self._on_quality_gate_error(
                image_job_id, qa_id, str(error)))

    def _on_quality_gate_done(self, image_job_id, qa_id, budget, plan, report):
        if image_job_id != self._image_job_id or qa_id != self._qa_job_id:
            return
        plan.ai_backend = self.current_ai_backend
        plan.ai_elapsed_s = self.current_ai_elapsed_s
        plan.hatch_candidate_count = self.current_hatch_candidate_count
        plan.qa_report = report
        self._plan_cache[budget] = plan
        self._apply_plan_to_preview(plan, budget, start_qa=False)
        self.current_qa_report = report
        self.qa_status_var.set(self._format_qa_report(report))
        self.card_gcode.set_status(
            "QA passed" if report.passed else "QA review",
            Theme.ACCENT_GREEN if report.passed else Theme.ACCENT_YELLOW)
        self._refresh_export_buttons()

    def _on_quality_gate_error(self, image_job_id, qa_id, msg):
        if image_job_id != self._image_job_id or qa_id != self._qa_job_id:
            return
        report = QAReport(
            iterations=[],
            passed=False,
            timed_out=False,
            model_status=f"error: {msg}",
        )
        self.current_qa_report = report
        if self.current_plan is not None:
            self.current_plan.qa_report = report
        self.qa_status_var.set(self._format_qa_report(report))
        self.card_gcode.set_status("QA error", Theme.ACCENT_RED)
        self.status_var.set(f"QA error: {msg}")
        self._refresh_export_buttons()

    def _on_preview_plan_error(self, image_job_id, preview_id, msg):
        if image_job_id != self._image_job_id or preview_id != self._preview_job_id:
            return
        self.card_gcode.set_status("Error", Theme.ACCENT_RED)
        self.gcode_preview.clear("Failed to build preview")
        self.status_var.set(f"Error in G-code preview: {msg}")

    # ─────────────────────────────────────────────────────────────
    #  Export G-Code
    # ─────────────────────────────────────────────────────────────
    def _on_export_gcode(self, force=False):
        if self.pil_lineart is None or not self.candidate_paths:
            messagebox.showwarning("No image", "Please load and process an image first!")
            return
        if not force and not self._qa_allows_export():
            messagebox.showwarning(
                "QA Gate Required",
                "Export is enabled only after the overlay QA gate passes.\n"
                "Wait for QA to finish, adjust settings, or use Force export.")
            return
        if force:
            report_text = self._format_qa_report(self.current_qa_report)
            if not messagebox.askyesno(
                    "Force Export",
                    "QA has not passed or is not complete.\n"
                    "The G-code preview may miss details or contain redundant strokes.\n\n"
                    f"{report_text}\n\n"
                    "Continue exporting anyway?"):
                return

        default_name = (
            os.path.splitext(os.path.basename(self.input_path))[0]
            + "_drawing_robot.gcode")
        save_path = filedialog.asksaveasfilename(
            title="Save G-Code File",
            initialfile=default_name,
            defaultextension=".gcode",
            filetypes=[("G-Code Files", "*.gcode"), ("Text Files", "*.txt")])
        if not save_path:
            return

        budget = self._current_budget()
        for btn in self._action_buttons():
            btn.config(state=tk.DISABLED)
        self.status_var.set(
            f"Exporting G-Code (budget {self._format_budget(budget)} segments)...")
        self.progress.grid()
        self.progress.start(10)

        candidates = list(self.candidate_paths)
        cached     = self._plan_cache.get(budget)
        threading.Thread(target=self._run_export,
                         args=(save_path, budget, candidates, cached),
                         daemon=True).start()

    def _run_export(self, save_path, budget, candidates, cached):
        t0 = time.time()
        try:
            plan = cached if cached is not None else build_gcode_plan(candidates, budget)
            plan.ai_backend = self.current_ai_backend
            plan.ai_elapsed_s = self.current_ai_elapsed_s
            plan.hatch_candidate_count = self.current_hatch_candidate_count
            # Validate lan cuoi truoc khi ghi file: neu co diem vuot bien hoac logic pen sai thi chan export.
            if plan.validation_errors:
                self.root.after(0, lambda: self._on_export_validation_error(
                    plan.validation_errors))
                return
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(plan.gcode_lines))
            elapsed = time.time() - t0
            self.root.after(0, lambda: self._on_export_done(save_path, plan, elapsed))
        except Exception as e:
            self.root.after(0, lambda: self._on_export_error(str(e)))

    def _on_export_validation_error(self, errors):
        self.progress.stop()
        self.progress.grid_remove()
        for btn in self._action_buttons():
            btn.config(state=tk.NORMAL)
        self._refresh_apply_button()
        self._refresh_export_buttons()
        preview = "\n".join(f"- {err}" for err in errors[:8])
        if len(errors) > 8:
            preview += f"\n- ... and {len(errors) - 8} more"
        self.status_var.set("Export blocked: G-code validation failed")
        messagebox.showwarning(
            "Cannot Export G-Code",
            "Validation failed. Please reduce detail, adjust mask/crop, or reload image.\n\n"
            f"{preview}")

    def _on_export_done(self, save_path, plan, elapsed):
        self.progress.stop()
        self.progress.grid_remove()
        for btn in self._action_buttons():
            btn.config(state=tk.NORMAL)
        self._refresh_apply_button()
        self._refresh_export_buttons()
        self.status_var.set(
            f"Done: {os.path.basename(save_path)}  "
            f"({self._format_budget(plan.actual_segments)} segments, "
            f"{plan.command_count} commands, {elapsed:.1f}s)")
        messagebox.showinfo(
            "G-Code Export Successful!",
            f"G-Code saved to:\n{save_path}\n\n"
            f"Segment budget:       {self._format_budget(plan.target_segments)}\n"
            f"Actual segments:      {self._format_budget(plan.actual_segments)}\n"
            f"Continuous strokes:   {plan.stroke_count}\n"
            f"Pen lifts:            {plan.pen_lifts}\n"
            f"Stitched (no-lift):   {plan.stitched_count}\n"
            f"G0 travel distance:   {plan.travel_distance_mm:.1f} mm\n"
            f"Draw distance:        {plan.draw_distance_mm:.1f} mm\n"
            f"Estimated machine time:{_format_duration(plan.estimated_time_s)}\n"
            f"Draw / travel feed:   F{plan.draw_feed_rate} / F{plan.travel_feed_rate}\n"
            f"Compression removed:  {plan.compression_removed_count} segments\n"
            f"Route 2-opt saved:    {plan.route_improvement_mm:.1f} mm\n"
            f"Total G-code commands:{plan.command_count}\n"
            f"G-code size:          {plan.gcode_size_bytes / 1024.0:.1f} KB\n"
            f"AI backend / time:    {plan.ai_backend} / {plan.ai_elapsed_s:.1f}s\n"
            f"Processing time:      {elapsed:.1f}s"
        )

    def _on_export_error(self, msg):
        self.progress.stop()
        self.progress.grid_remove()
        for btn in self._action_buttons():
            btn.config(state=tk.NORMAL)
        self._refresh_apply_button()
        self._refresh_export_buttons()
        self.status_var.set(f"Error exporting G-Code: {msg}")
        messagebox.showerror("Export Error", f"An error occurred:\n{msg}")

    # ─────────────────────────────────────────────────────────────
    def _show_pil_in_label(self, pil_img, label):
        if pil_img is None:
            return
        label._full_pil_image = pil_img.copy()
        label.bind("<Double-Button-1>",
                   lambda _e, target=label: self._open_image_zoom(target))
        label.config(cursor="hand2")
        label.update_idletasks()
        max_w = max(label.winfo_width()  - 8, 200)
        max_h = max(label.winfo_height() - 8, 200)

        img_copy = pil_img.copy()
        if img_copy.mode == "RGBA":
            bg = Image.new("RGBA", img_copy.size, (24, 24, 37, 255))
            bg.alpha_composite(img_copy)
            img_copy = bg.convert("RGB")
        else:
            img_copy = img_copy.convert("RGB")

        img_copy.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(img_copy)
        label.config(image=tk_img, text="", bg=Theme.BG_PREVIEW)
        label.image = tk_img

    def _open_image_zoom(self, label):
        pil_img = getattr(label, "_full_pil_image", None)
        if pil_img is None:
            return
        title = getattr(label, "_preview_title", "Image Preview")
        ImageZoomWindow(self.root, pil_img, title)


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main():
    try:
        root = tk.Tk()
        app  = App(root)
        root.mainloop()
    except tk.TclError as exc:
        print("Cannot start Tkinter UI. Your Python Tcl/Tk runtime is not usable.")
        print("Try repairing/reinstalling Python with Tcl/Tk, or select another IDE interpreter.")
        print(f"Details: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
