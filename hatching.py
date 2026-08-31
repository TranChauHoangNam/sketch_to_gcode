"""Utility functions for generating hatching patterns from grayscale images.

The goal of this module is to convert grayscale shading into a set of hatch lines
that can be fed into a pen-plotter or CNC drawing pipeline. The generated output is
kept in a simple polyline representation so it can be merged with existing contour
(edge/lineart) data and later converted to G-code without depending on a specific
Tkinter UI layer.

The core idea is:
- rotate the image so the hatch direction becomes horizontal,
- compute darkness statistics in local windows,
- use a spacing that shrinks as the patch gets darker,
- only draw hatch segments where the local intensity is sufficiently dark.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]
Polyline = List[Point]
HATCH_CONTOUR_OVERLAP_THRESHOLD_MM = 0.5
HATCH_MASK_EDGE_MARGIN = 10.0
HATCH_MIN_SEGMENT_PX = 2.0
HATCH_THINNING_STRENGTH = 0.34
HATCH_MIN_KEEP_PROB = 0.42
HATCH_JOIN_GAP_PX = 3.0
HATCH_TILE_SIZE_PX = 48
HATCH_TILE_DENSITY_SOFT_CAP = 6
HATCH_TILE_DENSITY_HARD_CAP = 10
HATCH_ZIGZAG_ROW_GAP_FACTOR = 1.4
HATCH_ZIGZAG_CONNECT_GAP_FACTOR = 1.25


def _as_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input image to a single-channel grayscale float32 array in [0, 255]."""
    arr = np.asarray(image)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    elif arr.ndim != 2:
        raise ValueError(f"Expected grayscale or BGR image, got shape {arr.shape!r}.")

    arr = arr.astype(np.float32)
    arr = np.clip(arr, 0.0, 255.0)
    return arr


def _prepare_guidance_map(
    guidance: np.ndarray | None,
    shape: tuple[int, int],
    binary: bool = False,
) -> np.ndarray | None:
    """Normalize an optional AI map to the working image without changing baseline output."""
    if guidance is None:
        return None
    arr = np.asarray(guidance)
    if arr.size == 0 or arr.ndim != 2 or not np.all(np.isfinite(arr)):
        return None
    arr = arr.astype(np.float32)
    if float(arr.max()) > 1.0:
        arr *= 1.0 / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    height, width = shape
    if arr.shape != shape:
        interpolation = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
        arr = cv2.resize(arr, (width, height), interpolation=interpolation)
    if not binary:
        return arr

    support = np.where(arr >= 0.08, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel, iterations=2)
    support = cv2.dilate(support, kernel, iterations=1)
    return support


def _darkness_map(image: np.ndarray, invert: bool = False) -> np.ndarray:
    """Return a 'darkness' map where larger values mean darker regions.

    For a standard 0-255 grayscale image, black is 0 and white is 255. We convert it
    to a darkness metric so that dark regions have high values and white regions low
    values. If invert=True, the logic is reversed for special use cases.
    """
    gray = _as_grayscale(image)
    if invert:
        return gray
    return 255.0 - gray


def _spacing_from_darkness(darkness_value: float, min_spacing: int, max_spacing: int) -> float:
    """Map a local darkness score to hatch spacing.

    The rule is linear interpolation in the opposite direction of brightness:
    - darker regions -> smaller spacing (more dense hatching)
    - lighter regions -> larger spacing (less dense hatching)

    Using 255 as maximum darkness ensures that nearly-black pixels become spacing
    close to min_spacing while white pixels become spacing close to max_spacing.
    """
    if max_spacing < min_spacing:
        min_spacing, max_spacing = max_spacing, min_spacing

    darkness_value = float(np.clip(darkness_value, 0.0, 255.0))
    if max_spacing == min_spacing:
        return float(max_spacing)

    t = darkness_value / 255.0
    spacing = (1.0 - t) * max_spacing + t * min_spacing
    return float(np.clip(spacing, min_spacing, max_spacing))


def _stable_keep_value(*values: int) -> float:
    """Hash nhe de thinning hatch quyet dinh on dinh, khong dung random."""
    h = 2166136261
    for value in values:
        h ^= int(value) & 0xFFFFFFFF
        h = (h * 16777619) & 0xFFFFFFFF
    return (h % 10000) / 10000.0


def _merge_nearby_runs(
    starts: np.ndarray,
    ends: np.ndarray,
    max_gap_px: float,
    row_tone: np.ndarray | None = None,
    gap_brightness_limit: float = 255.0,
) -> list[tuple[int, int]]:
    """Noi cac hatch run gan nhau tren cung mot hang de giam pen-lift nho le."""
    if len(starts) == 0:
        return []
    merged: list[tuple[int, int]] = []
    cur_start = int(starts[0])
    cur_end = int(ends[0])
    for start, end in zip(starts[1:], ends[1:]):
        start = int(start)
        end = int(end)
        gap_is_dark = True
        if row_tone is not None and start > cur_end:
            gap = row_tone[cur_end:start]
            gap_is_dark = bool(gap.size and np.all(gap < gap_brightness_limit))
        if start - cur_end <= max_gap_px and gap_is_dark:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _tile_density_keep(tile_counts: dict[tuple[int, int], int],
                       tile_key: tuple[int, int],
                       darkness_strength: float,
                       y_index: int,
                       start_x: int,
                       end_x: int) -> bool:
    """
    Cap mat do hatch theo tile: shadow dam duoc vuot soft cap, nhung khong vuot hard cap.
    Vung midtone bi lam thua bot de giam stroke ma van giu cam giac tone.
    """
    count = tile_counts.get(tile_key, 0)
    if count < HATCH_TILE_DENSITY_SOFT_CAP:
        tile_counts[tile_key] = count + 1
        return True
    if count >= HATCH_TILE_DENSITY_HARD_CAP:
        return False
    extra_keep = 0.22 + 0.62 * float(np.clip(darkness_strength, 0.0, 1.0))
    if _stable_keep_value(tile_key[0], tile_key[1], y_index, start_x, end_x) <= extra_keep:
        tile_counts[tile_key] = count + 1
        return True
    return False


def _serpentine_hatch_order(rows: list[tuple[int, list[tuple[int, int]]]]) -> list[tuple[int, int, int]]:
    """
    Sap hatch theo serpentine trong rotated space: hang chan trai->phai, hang le phai->trai.
    Khi downstream giu thu tu nay, travel giua cac hatch line giam hon so voi sap mot chieu.
    """
    ordered: list[tuple[int, int, int]] = []
    for row_idx, (y_index, runs) in enumerate(rows):
        row_runs = sorted(runs, key=lambda item: item[0], reverse=bool(row_idx % 2))
        for start_x, end_x in row_runs:
            last_x = max(start_x, end_x - 1)
            if row_idx % 2:
                ordered.append((y_index, last_x, start_x))
            else:
                ordered.append((y_index, start_x, last_x))
    return ordered


def _zigzag_stitch_runs(
    ordered_runs: Sequence[tuple[int, int, int]],
    max_row_gap_px: float,
    max_connect_gap_px: float,
    hatch_mask: np.ndarray | None = None,
) -> list[list[tuple[float, float]]]:
    """
    Gop cac hatch run da sap serpentine thanh polyline zigzag dai hon trong rotated space.

    Neu hai run lien tiep nam gan nhau theo ca chieu hang va khoang cach dau-noi-cuoi,
    ta noi truc tiep bang net ve thay vi nhac but. Nguong duoc truyen tu generate_hatching()
    theo max_spacing de van noi duoc khi hatch spacing thuc te lon.
    """
    stitched: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    previous_end: tuple[float, float] | None = None
    previous_y: float | None = None

    def connector_is_safe(a: tuple[float, float], b: tuple[float, float]) -> bool:
        if hatch_mask is None:
            return True
        h, w = hatch_mask.shape
        x0, y0 = int(round(a[0])), int(round(a[1]))
        x1, y1 = int(round(b[0])), int(round(b[1]))
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if x < 0 or x >= w or y < 0 or y >= h or not bool(hatch_mask[y, x]):
                return False
            if x == x1 and y == y1:
                break
            err2 = 2 * err
            if err2 > -dy:
                err -= dy
                x += sx
            if err2 < dx:
                err += dx
                y += sy
        return True

    for y_index, start_x, end_x in ordered_runs:
        start = (float(start_x), float(y_index))
        end = (float(end_x), float(y_index))

        if not current or previous_end is None or previous_y is None:
            current = [start, end]
            previous_end = end
            previous_y = float(y_index)
            continue

        row_gap = abs(float(y_index) - previous_y)
        connect_gap = float(np.hypot(start[0] - previous_end[0], start[1] - previous_end[1]))

        # Chi stitch khi duong noi ngan va hang ke nhau du gan; neu xa qua thi tach polyline.
        if (row_gap <= max_row_gap_px and
                connect_gap <= max_connect_gap_px and
                connector_is_safe(previous_end, start)):
            if connect_gap > 1e-6:
                current.append(start)
            current.append(end)
        else:
            stitched.append(current)
            current = [start, end]

        previous_end = end
        previous_y = float(y_index)

    if current:
        stitched.append(current)
    return stitched


def _connector_inside_mask(
    start: tuple[float, float],
    end: tuple[float, float],
    mask: np.ndarray | None,
) -> bool:
    if mask is None:
        return True
    distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    sample_count = max(2, int(np.ceil(distance * 1.5)) + 1)
    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
    xs = np.rint(start[0] + (end[0] - start[0]) * t).astype(np.int32)
    ys = np.rint(start[1] + (end[1] - start[1]) * t).astype(np.int32)
    h, w = mask.shape
    if np.any(xs < 0) or np.any(xs >= w) or np.any(ys < 0) or np.any(ys >= h):
        return False
    return bool(np.all(mask[ys, xs]))


def _stitch_oriented_hatch_segments(
    polylines: Sequence[Sequence[tuple[float, float]]],
    max_gap_px: float,
    hatch_mask: np.ndarray | None,
    max_angle_delta_deg: float = 45.0,
) -> list[list[tuple[float, float]]]:
    """Greedily serpentine-stitch nearby parallel local hatch segments."""
    paths = [
        [(float(x), float(y)) for x, y in polyline]
        for polyline in polylines if len(polyline) >= 2
    ]
    if len(paths) <= 1 or max_gap_px <= 0.0:
        return paths

    cell = max(1.0, float(max_gap_px))
    endpoint_grid: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def grid_key(point: tuple[float, float]) -> tuple[int, int]:
        return int(np.floor(point[0] / cell)), int(np.floor(point[1] / cell))

    for index, path in enumerate(paths):
        endpoint_grid.setdefault(grid_key(path[0]), []).append((index, 0))
        endpoint_grid.setdefault(grid_key(path[-1]), []).append((index, -1))

    used = np.zeros(len(paths), dtype=bool)
    stitched: list[list[tuple[float, float]]] = []
    min_parallel_cos = float(np.cos(np.deg2rad(max_angle_delta_deg)))
    connector_count = 0

    for seed_index in range(len(paths)):
        if used[seed_index]:
            continue
        used[seed_index] = True
        chain = list(paths[seed_index])

        while True:
            end = chain[-1]
            key_x, key_y = grid_key(end)
            current_vec = np.asarray(chain[-1], dtype=np.float32) - \
                np.asarray(chain[-2], dtype=np.float32)
            current_norm = float(np.linalg.norm(current_vec))
            if current_norm <= 1e-6:
                break
            current_dir = current_vec / current_norm

            best = None
            best_score = float("inf")
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    for candidate_index, endpoint_index in endpoint_grid.get(
                            (key_x + dx, key_y + dy), ()):
                        if used[candidate_index]:
                            continue
                        candidate = (paths[candidate_index] if endpoint_index == 0
                                     else list(reversed(paths[candidate_index])))
                        gap = float(np.hypot(
                            candidate[0][0] - end[0], candidate[0][1] - end[1]))
                        if gap > max_gap_px:
                            continue
                        candidate_vec = np.asarray(candidate[1], dtype=np.float32) - \
                            np.asarray(candidate[0], dtype=np.float32)
                        candidate_norm = float(np.linalg.norm(candidate_vec))
                        if candidate_norm <= 1e-6:
                            continue
                        parallel = abs(float(np.dot(
                            current_dir, candidate_vec / candidate_norm)))
                        if parallel < min_parallel_cos:
                            continue
                        if not _connector_inside_mask(end, candidate[0], hatch_mask):
                            continue
                        score = gap * (1.0 + 0.30 * (1.0 - parallel))
                        if score < best_score:
                            best_score = score
                            best = candidate_index, candidate, gap

            if best is None:
                break
            candidate_index, candidate, gap = best
            if gap > 1e-4:
                chain.append(candidate[0])
            chain.extend(candidate[1:])
            used[candidate_index] = True
            connector_count += 1

        stitched.append(chain)

    if connector_count:
        print(
            f"  Oriented hatch stitching: {len(paths)} -> {len(stitched)} polylines "
            f"({connector_count} draw-through connectors <= {max_gap_px:.1f}px)"
        )
    return stitched


def _inverse_rotate_points(points: Sequence[Tuple[float, float]], angle_deg: float, center: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Rotate a list of points back to the original image coordinate system."""
    if not points:
        return []

    arr = np.asarray(points, dtype=np.float32)
    cx, cy = center
    # cv2.getRotationMatrix2D uses image coordinates (positive Y points down),
    # so its inverse corresponds to the standard +angle rotation below.
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    centered = arr - np.array([cx, cy], dtype=np.float32)
    rotated = np.empty_like(centered)
    rotated[:, 0] = centered[:, 0] * cos_t - centered[:, 1] * sin_t
    rotated[:, 1] = centered[:, 0] * sin_t + centered[:, 1] * cos_t
    rotated = rotated + np.array([cx, cy], dtype=np.float32)
    return [(float(x), float(y)) for x, y in rotated]


def _clip_polyline_to_image(
    points: Sequence[Tuple[float, float]], width: int, height: int
) -> list[list[tuple[float, float]]]:
    """Clip a polyline to image bounds without creating artificial border strokes."""
    if len(points) < 2 or width <= 0 or height <= 0:
        return []
    max_x = float(width - 1)
    max_y = float(height - 1)
    result: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    def clip_segment(p0: Point, p1: Point) -> tuple[Point, Point] | None:
        x0, y0 = map(float, p0)
        x1, y1 = map(float, p1)
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0), (dx, max_x - x0),
                     (-dy, y0), (dy, max_y - y0)):
            if abs(p) <= 1e-12:
                if q < 0.0:
                    return None
                continue
            ratio = q / p
            if p < 0.0:
                t0 = max(t0, ratio)
            else:
                t1 = min(t1, ratio)
            if t0 > t1:
                return None
        return ((x0 + t0 * dx, y0 + t0 * dy),
                (x0 + t1 * dx, y0 + t1 * dy))

    for p0, p1 in zip(points[:-1], points[1:]):
        clipped = clip_segment(p0, p1)
        if clipped is None:
            if len(current) >= 2:
                result.append(current)
            current = []
            continue
        q0, q1 = clipped
        if current and np.hypot(current[-1][0] - q0[0], current[-1][1] - q0[1]) <= 1e-4:
            if np.hypot(current[-1][0] - q1[0], current[-1][1] - q1[1]) > 1e-6:
                current.append(q1)
        else:
            if len(current) >= 2:
                result.append(current)
            current = [q0, q1]
    if len(current) >= 2:
        result.append(current)
    return result


def _prepare_orientation_map(
    orientation_map: np.ndarray | None,
    shape: tuple[int, int],
    cell_size: int,
) -> np.ndarray | None:
    """Resize and smooth axial angles in degrees using their doubled vectors."""
    if orientation_map is None:
        return None
    angles = np.asarray(orientation_map, dtype=np.float32)
    if angles.size == 0 or angles.ndim != 2 or not np.all(np.isfinite(angles)):
        return None

    theta2 = np.deg2rad(angles * 2.0)
    cos2 = np.cos(theta2).astype(np.float32)
    sin2 = np.sin(theta2).astype(np.float32)
    height, width = shape
    if angles.shape != shape:
        cos2 = cv2.resize(cos2, (width, height), interpolation=cv2.INTER_LINEAR)
        sin2 = cv2.resize(sin2, (width, height), interpolation=cv2.INTER_LINEAR)

    sigma = max(1.0, float(cell_size) * 0.45)
    cos2 = cv2.GaussianBlur(cos2, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sin2 = cv2.GaussianBlur(sin2, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.mod(np.rad2deg(0.5 * np.arctan2(sin2, cos2)), 180.0).astype(np.float32)


def _clip_infinite_line_to_tile(
    point: np.ndarray,
    direction: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a long parametric line to an inclusive image tile rectangle."""
    extent = float(np.hypot(x1 - x0 + 1, y1 - y0 + 1) * 2.0 + 2.0)
    p0 = point - direction * extent
    p1 = point + direction * extent
    dx, dy = p1 - p0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]),
                 (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if abs(float(p)) <= 1e-12:
            if q < 0.0:
                return None
            continue
        ratio = float(q / p)
        if p < 0.0:
            t0 = max(t0, ratio)
        else:
            t1 = min(t1, ratio)
        if t0 > t1:
            return None
    return p0 + (p1 - p0) * t0, p0 + (p1 - p0) * t1


def _generate_oriented_hatching(
    gray: np.ndarray,
    orientation: np.ndarray,
    cell_size: int,
    min_spacing: int,
    max_spacing: int,
    dark_threshold: float,
    importance: np.ndarray | None,
    allowed: np.ndarray | None,
    fallback_angle_deg: float,
) -> list[list[tuple[float, float]]]:
    """Generate locally oriented hatch segments on a smooth tile direction field."""
    height, width = gray.shape
    kernel_size = max(1, int(cell_size))
    local_mean = (cv2.boxFilter(
        gray, ddepth=-1, ksize=(kernel_size, kernel_size), normalize=True)
        if kernel_size > 1 else gray.astype(np.float32))
    drawable = ((local_mean < dark_threshold) &
                (gray < dark_threshold + HATCH_MASK_EDGE_MARGIN))
    if allowed is not None:
        drawable &= allowed > 0

    # Average tile directions through doubled-angle vectors. Sampling the
    # box-filtered field at tile centers gives bilinear-like transitions between
    # neighboring tiles without the 0/180-degree discontinuity.
    theta2 = np.deg2rad(orientation * 2.0)
    tile_size = max(20, int(cell_size))
    mean_cos2 = cv2.boxFilter(
        np.cos(theta2).astype(np.float32), -1, (tile_size, tile_size), normalize=True)
    mean_sin2 = cv2.boxFilter(
        np.sin(theta2).astype(np.float32), -1, (tile_size, tile_size), normalize=True)

    polylines: list[list[tuple[float, float]]] = []
    min_segment_px = max(
        HATCH_MIN_SEGMENT_PX, min(8.0, min(height, width) * 0.0035))

    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size) - 1
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size) - 1
            tile_drawable = drawable[y0:y1 + 1, x0:x1 + 1]
            if np.count_nonzero(tile_drawable) < 3:
                continue

            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            vector_strength = float(np.hypot(mean_cos2[cy, cx], mean_sin2[cy, cx]))
            angle = (float(np.rad2deg(0.5 * np.arctan2(
                mean_sin2[cy, cx], mean_cos2[cy, cx]))) % 180.0
                     if vector_strength > 0.04 else float(fallback_angle_deg) % 180.0)
            theta = np.deg2rad(angle)
            direction = np.asarray([np.cos(theta), np.sin(theta)], dtype=np.float32)
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)

            supported_tone = local_mean[y0:y1 + 1, x0:x1 + 1][tile_drawable]
            darkness = 255.0 - float(supported_tone.mean())
            spacing = _spacing_from_darkness(darkness, min_spacing, max_spacing)

            corners = np.asarray(
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            projections = corners @ normal
            first_offset = np.ceil(float(projections.min()) / spacing) * spacing
            offsets = np.arange(
                first_offset, float(projections.max()) + spacing * 0.25,
                spacing, dtype=np.float32)
            kept_in_tile = 0
            tile_key = (x0 // tile_size, y0 // tile_size)
            tile_center = np.asarray(
                [(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)

            for line_index, offset in enumerate(offsets):
                point = tile_center + normal * (
                    float(offset) - float(np.dot(tile_center, normal)))
                clipped = _clip_infinite_line_to_tile(
                    point, direction, x0, y0, x1, y1)
                if clipped is None:
                    continue
                start, end = clipped
                length = float(np.linalg.norm(end - start))
                if length < min_segment_px:
                    continue

                sample_count = max(2, int(np.ceil(length)) + 1)
                t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
                samples = start[None, :] + (end - start)[None, :] * t[:, None]
                xs = np.clip(np.rint(samples[:, 0]).astype(np.int32), x0, x1)
                ys = np.clip(np.rint(samples[:, 1]).astype(np.int32), y0, y1)
                inside = drawable[ys, xs]
                changes = np.diff(inside.astype(np.int8))
                run_starts = (np.nonzero(changes == 1)[0] + 1).tolist()
                run_ends = (np.nonzero(changes == -1)[0] + 1).tolist()
                if inside[0]:
                    run_starts.insert(0, 0)
                if inside[-1]:
                    run_ends.append(sample_count)

                for run_index, (run_start, run_end) in enumerate(zip(run_starts, run_ends)):
                    if run_end - run_start < 2:
                        continue
                    p0 = samples[run_start]
                    p1 = samples[run_end - 1]
                    run_length = float(np.linalg.norm(p1 - p0))
                    if run_length < min_segment_px:
                        continue

                    run_tone = float(local_mean[ys[run_start:run_end], xs[run_start:run_end]].mean())
                    darkness_strength = float(np.clip(
                        (dark_threshold - run_tone) / max(1.0, dark_threshold), 0.0, 1.0))
                    keep_prob = HATCH_MIN_KEEP_PROB + \
                        (1.0 - HATCH_MIN_KEEP_PROB) * darkness_strength
                    keep_prob = 1.0 - HATCH_THINNING_STRENGTH * (1.0 - keep_prob)
                    if importance is not None:
                        segment_importance = float(
                            importance[ys[run_start:run_end], xs[run_start:run_end]].mean())
                        keep_prob *= 0.70 + 0.30 * segment_importance
                    if _stable_keep_value(
                            tile_key[0], tile_key[1], line_index, run_index) > keep_prob:
                        continue
                    if kept_in_tile >= HATCH_TILE_DENSITY_HARD_CAP:
                        break

                    polylines.append([
                        (float(p0[0]), float(p0[1])),
                        (float(p1[0]), float(p1[1])),
                    ])
                    kept_in_tile += 1
                if kept_in_tile >= HATCH_TILE_DENSITY_HARD_CAP:
                    break

    connector_gap_px = min(9.0, max(4.0, float(max_spacing) * 0.55))
    return _stitch_oriented_hatch_segments(
        polylines, connector_gap_px, drawable,
        max_angle_delta_deg=45.0)


def generate_hatching(
    image: np.ndarray,
    cell_size: int,
    angle_deg: float,
    min_spacing: int,
    max_spacing: int,
    invert: bool = False,
    dark_threshold: float = 235.0,
    importance_map: np.ndarray | None = None,
    allowed_mask: np.ndarray | None = None,
    orientation_map: np.ndarray | None = None,
) -> list[list[tuple[float, float]]]:
    """Generate grayscale hatching strokes as a list of polylines.

    The method intentionally works in a rotated coordinate system so that the hatch
    lines are effectively horizontal. For each row in that rotated view, the algorithm
    computes a local darkness score and converts it to a spacing value via linear
    interpolation between `min_spacing` and `max_spacing`.

    The logic is:
    - darker pixels => larger local darkness value => smaller spacing => denser hatch lines
    - lighter pixels => smaller darkness value => larger spacing => sparser hatch lines
    - only draw segments where the local pixel intensity is below `dark_threshold` so the pattern
      stays out of white/background regions.

    Parameters
    ----------
    image:
        Input grayscale image in the range [0, 255]. A color image is accepted and
        converted automatically to grayscale.
    cell_size:
        Size of the local sampling window for computing the average darkness used to set
        spacing. Larger values produce smoother density transitions.
    angle_deg:
        Angle in degrees of the hatch direction.
    min_spacing:
        Minimum spacing between neighboring hatch lines in pixels for the darkest regions.
    max_spacing:
        Maximum spacing between neighboring hatch lines in pixels for the lightest regions.
    invert:
        If True, invert the grayscale before computing darkness. This is useful when the
        source data is a light-on-dark sketch or when a reversed tone map is desired.
    dark_threshold:
        Local mean brightness cutoff in the range [0, 255]. Pixels below this value are
        considered dark enough to receive hatching. Lower values restrict hatching to
        deeper shadows; higher values include more midtones and near-white areas.
    orientation_map:
        Optional per-pixel hatch direction in degrees. When supplied, hatching
        follows a smoothed local tile orientation; when omitted, the legacy
        single-angle rotation path is used unchanged.

    Returns
    -------
    list[list[tuple[float, float]]]
        A list of polylines. Each polyline is represented as a list of (x, y) points in
        the original image coordinate space.
    """
    gray = _as_grayscale(image)
    if gray.size == 0:
        return []
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0.")
    if min_spacing <= 0 or max_spacing <= 0:
        raise ValueError("min_spacing and max_spacing must be > 0.")
    dark_threshold = float(np.clip(dark_threshold, 0.0, 255.0))

    # Inverse the grayscale if requested. For a normal pen-and-paper sketch, darker pixels
    # should receive denser hatching, so we treat lower luminance as a stronger drawing cue.
    if invert:
        gray = 255.0 - gray

    height, width = gray.shape
    importance = _prepare_guidance_map(importance_map, gray.shape)
    allowed = _prepare_guidance_map(allowed_mask, gray.shape, binary=True)
    orientation = _prepare_orientation_map(orientation_map, gray.shape, cell_size)
    if orientation is not None:
        return _generate_oriented_hatching(
            gray, orientation, cell_size, min_spacing, max_spacing,
            dark_threshold, importance, allowed, angle_deg)

    # Rotate the image so the hatch direction is roughly horizontal in the rotated space.
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        gray,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=255.0,
    )
    rotated_importance = None
    if importance is not None:
        rotated_importance = cv2.warpAffine(
            importance,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=0.0,
        )
    rotated_allowed = None
    if allowed is not None:
        rotated_allowed = cv2.warpAffine(
            allowed,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        ) > 0

    # Use a local average to create a smooth darkness map. This provides a spatially stable
    # estimate of how dense the hatch should be in nearby regions.
    kernel_size = max(1, cell_size)
    if kernel_size > 1:
        local_mean = cv2.boxFilter(
            rotated, ddepth=-1, ksize=(kernel_size, kernel_size), normalize=True)
    else:
        local_mean = rotated.astype(np.float32)

    # The spacing is inversely proportional to darkness. We map darkness from [0, 255] to
    # [max_spacing, min_spacing]. A pixel value near 0 (pure black) becomes very tight lines,
    # while a pixel near 255 (pure white) remains sparse or even empty.
    # Compute density from actual drawable tone, not the full row. A small dark
    # subject should not become sparse merely because most of the row is white.
    tone_support = local_mean < dark_threshold
    if rotated_allowed is not None:
        tone_support &= rotated_allowed
    support_count = tone_support.sum(axis=1)
    tone_sum = np.where(tone_support, local_mean, 0.0).sum(axis=1)
    supported_mean = np.divide(
        tone_sum, np.maximum(support_count, 1), dtype=np.float32)
    min_row_support = max(3, int(round(width * 0.004)))
    row_darkness = np.where(
        support_count >= min_row_support, 255.0 - supported_mean, 0.0)
    spacing_map = np.clip(
        max_spacing - (max_spacing - min_spacing) * (row_darkness / 255.0),
        min_spacing,
        max_spacing,
    )

    rows_for_ordering: list[tuple[int, list[tuple[int, int]]]] = []
    tile_counts: dict[tuple[int, int], int] = {}
    min_segment_px = max(
        HATCH_MIN_SEGMENT_PX, min(8.0, min(height, width) * 0.0035))

    # Walk the rotated image using the local spacing. This keeps the computation efficient and
    # avoids a full per-pixel Python loop while still producing a realistic hatch density.
    y_cursor = 0.0
    while y_cursor < height:
        y_index = int(np.clip(round(y_cursor), 0, height - 1))
        local_spacing = float(spacing_map[y_index])

        # Clip hatch vao vung toi that su: local_mean giu do mem, rotated giu bien shadow sac hon.
        # Nho do hatch khong lan ra ngoai contour/shadow region khi anh co bien sang-toi ro.
        row_mask = ((local_mean[y_index, :] < dark_threshold) &
                    (rotated[y_index, :] < dark_threshold + HATCH_MASK_EDGE_MARGIN))
        if rotated_allowed is not None:
            row_mask &= rotated_allowed[y_index, :]
        if not row_mask.any():
            y_cursor += max(1.0, local_spacing)
            continue

        binary = row_mask.astype(np.int16)
        diffs = np.diff(binary)
        starts = np.nonzero(diffs == 1)[0] + 1
        ends = np.nonzero(diffs == -1)[0] + 1

        if binary[0] == 1:
            starts = np.insert(starts, 0, 0)
        if binary[-1] == 1:
            ends = np.append(ends, width)

        merged_runs = _merge_nearby_runs(
            starts,
            ends,
            HATCH_JOIN_GAP_PX,
            row_tone=rotated[y_index, :],
            gap_brightness_limit=dark_threshold + HATCH_MASK_EDGE_MARGIN,
        )
        kept_runs: list[tuple[int, int]] = []
        for start_x, end_x in merged_runs:
            if end_x <= start_x:
                continue
            if float(end_x - start_x) < min_segment_px:
                continue
            segment_mean = float(local_mean[y_index, start_x:end_x].mean())
            darkness_margin = max(0.0, dark_threshold - segment_mean)
            darkness_strength = np.clip(darkness_margin / max(1.0, dark_threshold), 0.0, 1.0)
            keep_prob = HATCH_MIN_KEEP_PROB + (1.0 - HATCH_MIN_KEEP_PROB) * darkness_strength
            keep_prob = 1.0 - HATCH_THINNING_STRENGTH * (1.0 - keep_prob)
            if rotated_importance is not None:
                segment_importance = float(
                    rotated_importance[y_index, start_x:end_x].mean())
                keep_prob *= 0.70 + 0.30 * segment_importance
            # Adaptive hatch thinning: midtone/texture duoc lam thua bot, shadow dam van giu day.
            if _stable_keep_value(y_index, int(start_x), int(end_x)) > keep_prob:
                continue
            tile_key = (int(((start_x + end_x) * 0.5) // HATCH_TILE_SIZE_PX),
                        int(y_index // HATCH_TILE_SIZE_PX))
            if not _tile_density_keep(
                    tile_counts, tile_key, float(darkness_strength),
                    y_index, int(start_x), int(end_x)):
                continue
            kept_runs.append((int(start_x), int(end_x)))

        if kept_runs:
            rows_for_ordering.append((y_index, kept_runs))

        y_cursor += max(1.0, local_spacing)

    ordered_runs = _serpentine_hatch_order(rows_for_ordering)
    max_row_gap_px = HATCH_ZIGZAG_ROW_GAP_FACTOR * float(max_spacing)
    max_connect_gap_px = HATCH_ZIGZAG_CONNECT_GAP_FACTOR * float(max_spacing)
    hatch_mask = (local_mean < dark_threshold) & (rotated < dark_threshold + HATCH_MASK_EDGE_MARGIN)
    if rotated_allowed is not None:
        hatch_mask &= rotated_allowed
    stitched_runs = _zigzag_stitch_runs(
        ordered_runs, max_row_gap_px, max_connect_gap_px, hatch_mask=hatch_mask)

    polylines: list[list[tuple[float, float]]] = []
    for points_rotated in stitched_runs:
        if len(points_rotated) < 2:
            continue
        # Sau khi stitch trong rotated space, moi inverse-rotate ca polyline ve toa do goc.
        points_original = _inverse_rotate_points(points_rotated, angle_deg, center)
        polylines.extend(_clip_polyline_to_image(points_original, width, height))

    return polylines


def generate_cross_hatching(
    image: np.ndarray,
    cell_size: int,
    min_spacing: int,
    max_spacing: int,
    angle_deg_1: float = 45.0,
    angle_deg_2: float = 135.0,
    threshold_dark: int = 180,
    invert: bool = False,
    importance_map: np.ndarray | None = None,
    allowed_mask: np.ndarray | None = None,
) -> list[list[tuple[float, float]]]:
    """Generate a cross-hatching pattern by combining two hatch directions.

    The first pass is executed on a dark-mask version of the image to emphasize the
    shadowed regions. A second pass is added at a different angle to create the classic
    cross-hatch effect.

    Parameters
    ----------
    image:
        Grayscale image in [0, 255].
    cell_size:
        Local neighborhood size for determining the hatch density.
    min_spacing, max_spacing:
        Minimum and maximum spacing between hatch lines.
    angle_deg_1, angle_deg_2:
        Two hatch directions to combine.
    threshold_dark:
        Brightness threshold used to keep only darker regions. The image is masked so that
        pixels with value < threshold_dark are considered shadow regions.
    invert:
        Whether the input should be inverted before generating the hatch map.

    Returns
    -------
    list[list[tuple[float, float]]]
        The combined cross-hatching set.
    """
    gray = _as_grayscale(image)
    if gray.size == 0:
        return []

    if threshold_dark < 0:
        threshold_dark = 0
    if threshold_dark > 255:
        threshold_dark = 255

    tone = 255.0 - gray if invert else gray
    first_mask = np.where(tone < threshold_dark, tone, 255.0)
    hatch_1 = generate_hatching(
        first_mask,
        cell_size,
        angle_deg_1,
        min_spacing,
        max_spacing,
        invert=False,
        dark_threshold=float(threshold_dark),
        importance_map=importance_map,
        allowed_mask=allowed_mask,
    )

    angle_delta = abs((float(angle_deg_1) - float(angle_deg_2)) % 180.0)
    if min(angle_delta, 180.0 - angle_delta) < 1e-6:
        return hatch_1

    deep_threshold = float(np.clip(threshold_dark * 0.72, 0.0, 255.0))
    second_mask = np.where(tone < deep_threshold, tone, 255.0)
    second_min_spacing = max(1, int(round(min_spacing * 1.2)))
    second_max_spacing = max(second_min_spacing, int(round(max_spacing * 1.25)))
    hatch_2 = generate_hatching(
        second_mask,
        cell_size,
        angle_deg_2,
        second_min_spacing,
        second_max_spacing,
        invert=False,
        dark_threshold=deep_threshold,
        importance_map=importance_map,
        allowed_mask=allowed_mask,
    )
    return hatch_1 + hatch_2


def _contour_segments(contour_lines: Iterable[Sequence[Tuple[float, float]]]) -> np.ndarray:
    """Chuyen contour polyline thanh mang segment de tinh khoang cach nhanh."""
    segments = []
    for poly in contour_lines:
        if len(poly) < 2:
            continue
        pts = np.asarray(poly, dtype=np.float32)
        for p0, p1 in zip(pts[:-1], pts[1:]):
            segments.append((p0, p1))
    if not segments:
        return np.empty((0, 2, 2), dtype=np.float32)
    return np.asarray(segments, dtype=np.float32)


def _point_near_contour(point: Tuple[float, float], contour_segments: np.ndarray,
                        threshold: float) -> bool:
    """Kiem tra mot diem hatch co nam gan bat ky segment contour nao khong."""
    if contour_segments.size == 0:
        return False
    p = np.asarray(point, dtype=np.float32)
    a = contour_segments[:, 0, :]
    b = contour_segments[:, 1, :]
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom <= 1e-9, 1.0, denom)
    t = np.sum((p - a) * ab, axis=1) / denom
    t = np.clip(t, 0.0, 1.0)
    closest = a + ab * t[:, None]
    dist = np.linalg.norm(closest - p, axis=1)
    return bool(np.any(dist < threshold))


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Tinh huong quay 2D cua bo ba diem, dung de phat hien segment giao nhau."""
    return (b[0] - a[0]) * (c[:, 1] - a[1]) - (b[1] - a[1]) * (c[:, 0] - a[0])


def _hatch_segment_near_contour(p0: Tuple[float, float], p1: Tuple[float, float],
                                contour_segments: np.ndarray,
                                threshold: float) -> bool:
    """
    Kiem tra toan bo hatch segment co trung/gan contour khong.
    Neu hai segment cat nhau o giua, khoang cach coi nhu 0 va hatch se bi bo.
    """
    if contour_segments.size == 0:
        return False
    a = np.asarray(p0, dtype=np.float32)
    b = np.asarray(p1, dtype=np.float32)
    c = contour_segments[:, 0, :]
    d = contour_segments[:, 1, :]

    # Bat giao cat that su giua hatch segment va contour segment.
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = (d[:, 0] - c[:, 0]) * (a[1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (a[0] - c[:, 0])
    o4 = (d[:, 0] - c[:, 0]) * (b[1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (b[0] - c[:, 0])
    bbox_overlap = (
        (np.maximum(np.minimum(a[0], b[0]), np.minimum(c[:, 0], d[:, 0])) <=
         np.minimum(np.maximum(a[0], b[0]), np.maximum(c[:, 0], d[:, 0]))) &
        (np.maximum(np.minimum(a[1], b[1]), np.minimum(c[:, 1], d[:, 1])) <=
         np.minimum(np.maximum(a[1], b[1]), np.maximum(c[:, 1], d[:, 1])))
    )
    intersects = (o1 * o2 <= 0.0) & (o3 * o4 <= 0.0) & bbox_overlap
    if bool(np.any(intersects)):
        return True

    # Neu khong cat nhau, lay min khoang cach endpoint-to-segment o hai chieu.
    if _point_near_contour(p0, contour_segments, threshold):
        return True
    if _point_near_contour(p1, contour_segments, threshold):
        return True

    ab = b - a
    denom = float(np.dot(ab, ab)) or 1.0
    for pts in (c, d):
        t = np.sum((pts - a) * ab, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        closest = a + ab * t[:, None]
        if bool(np.any(np.linalg.norm(closest - pts, axis=1) < threshold)):
            return True
    return False


def _remove_hatch_contour_overlap(
    hatch_lines: Iterable[Sequence[Tuple[float, float]]],
    contour_lines: Iterable[Sequence[Tuple[float, float]]],
    threshold: float,
) -> list[list[tuple[float, float]]]:
    """
    Cat bo hatch bi trung contour: kiem tra ca segment hatch, khong chi dau/cuoi,
    de tranh plotter ve de hai lan cung mot vi tri.
    """
    contour_segments = _contour_segments(contour_lines)
    if contour_segments.size == 0 or threshold <= 0:
        return [[(float(x), float(y)) for x, y in poly] for poly in hatch_lines if poly]

    filtered: list[list[tuple[float, float]]] = []
    for poly in hatch_lines:
        if len(poly) < 2:
            continue
        pts = [(float(x), float(y)) for x, y in poly]
        current: list[tuple[float, float]] = []
        for p0, p1 in zip(pts[:-1], pts[1:]):
            if _hatch_segment_near_contour(p0, p1, contour_segments, threshold):
                if len(current) >= 2:
                    filtered.append(current)
                current = []
                continue
            if not current:
                current = [p0]
            current.append(p1)
        if len(current) >= 2:
            filtered.append(current)
    return filtered


def merge_hatch_with_lineart(
    hatch_lines: Iterable[Sequence[Tuple[float, float]]],
    contour_lines: Iterable[Sequence[Tuple[float, float]]],
    overlap_threshold_mm: float = HATCH_CONTOUR_OVERLAP_THRESHOLD_MM,
) -> list[list[tuple[float, float]]]:
    """Combine hatch polylines with existing contour/edge polylines.

    This is meant to be a lightweight integration step before G-code generation. The
    returned combined list keeps both styles of geometry as independent polylines so the
    downstream planner can lift the pen between polylines without drawing a continuous
    line across unrelated stroke groups.
    """
    combined: list[list[tuple[float, float]]] = []
    contour_lines = list(contour_lines)
    hatch_lines = _remove_hatch_contour_overlap(
        hatch_lines, contour_lines, float(overlap_threshold_mm))

    for poly in hatch_lines:
        if poly:
            combined.append([(float(x), float(y)) for x, y in poly])

    for poly in contour_lines:
        if poly:
            combined.append([(float(x), float(y)) for x, y in poly])

    return combined


def _demo_hatching() -> None:
    """Create a small test image and visualize the generated hatching lines."""
    import matplotlib.pyplot as plt

    height, width = 800, 800
    yy, xx = np.mgrid[0:height, 0:width]

    # A soft gradient plus a few dark geometric shapes simulate a grayscale sketch.
    base = 220.0 - 45.0 * np.sin(xx / 70.0) - 30.0 * np.cos(yy / 60.0)
    ring = ((xx - 420) ** 2 + (yy - 430) ** 2) < 170 ** 2
    blob = ((xx - 600) ** 2 + (yy - 270) ** 2) < 150 ** 2
    base = np.where(ring | blob, base * 0.68, base)
    base = np.clip(base, 0.0, 255.0).astype(np.uint8)

    hatch_lines = generate_hatching(
        base,
        cell_size=18,
        angle_deg=35.0,
        min_spacing=3,
        max_spacing=18,
        invert=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(base, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Source grayscale image")
    axes[0].axis("off")

    axes[1].imshow(base, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Generated hatching")
    for polyline in hatch_lines:
        xs = [p[0] for p in polyline]
        ys = [p[1] for p in polyline]
        axes[1].plot(xs, ys, color="black", linewidth=0.8)
    axes[1].axis("off")

    plt.show()


if __name__ == "__main__":
    _demo_hatching()
