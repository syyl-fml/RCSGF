#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generalized Mask 10 generator for RCSGF self-supervised gap filling.

This script generates the artificial Input images used by the Input-PGT
self-supervised proxy task. It is intended as the single public Mask 10
implementation for SAXS, WAXD, AgBiO3, USAXS, and other 2-D synchrotron
scattering/diffraction patterns with horizontal and/or vertical detector gaps.

Key properties
--------------
1. Any 2-D TIFF size is accepted; the original image size and dtype are kept.
2. Horizontal and vertical detector-gap counts are not hard-coded.
3. Gap widths are measured from each input image.
4. For each detector gap, a three-band neighborhood is defined on both sides:
       point-candidate band | fixed strip | point-candidate band
   Each band has the width of the corresponding detector gap.
5. Fixed strips are shifted by one gap width from the physical detector gap.
6. Additional 1x1 point masks are sampled in the remaining valid candidate
   region according to a dataset-specific target ratio.
7. Physical detector-gap pixels are never used as artificial-mask pixels.
8. A stable per-image random seed makes repeated runs reproducible regardless
   of file traversal order.
9. A CSV manifest records the exact parameters and generated-mask statistics.
10. Resume mode is supported and guarded by a configuration fingerprint.

Paper presets
-------------
    SAXS   : 0.65-0.70
    WAXD   : 0.75-0.80
    AgBiO3 : 0.45-0.55
    USAXS  : 0.65-0.70

Examples
--------
    python generate_mask10_general.py \
        --input-dir /path/to/SAXS_PGT \
        --output-dir /path/to/SAXS_Input \
        --preset saxs

    python generate_mask10_general.py \
        --input-dir /path/to/USAXS_PGT \
        --output-dir /path/to/USAXS_Input \
        --preset usaxs

    python generate_mask10_general.py \
        --input-dir /path/to/custom_PGT \
        --output-dir /path/to/custom_Input \
        --ratio 0.60 0.70

Notes on the 1x1 point-mask spacing
-----------------------------------
The implementation first places point masks while avoiding orthogonally
adjacent occupied pixels. For target ratios above the density achievable under
that spacing constraint, the remaining required pixels are sampled from the
still-unmasked valid candidate region. The manifest records the number of such
spacing-relaxed pixels in ``relaxed_point_pixels``. This preserves the behavior
used to reach the target point-mask coverage ratios in the experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff
from tqdm import tqdm


SCRIPT_VERSION = "1.0.0"
ALGORITHM_NAME = "RCSGF_Mask10_generalized"

PRESET_RATIOS: Dict[str, Tuple[float, float]] = {
    "saxs": (0.65, 0.70),
    "waxd": (0.75, 0.80),
    "agbio3": (0.45, 0.55),
    "usaxs": (0.65, 0.70),
}

TIFF_SUFFIXES = (".tif", ".tiff")

MANIFEST_FIELDS = (
    "source_relpath",
    "output_relpath",
    "algorithm",
    "script_version",
    "config_hash",
    "dataset_preset",
    "height",
    "width",
    "dtype",
    "horizontal_gap_count",
    "vertical_gap_count",
    "horizontal_gap_widths",
    "vertical_gap_widths",
    "ratio_reference",
    "ratio_low",
    "ratio_high",
    "sampled_target_ratio",
    "actual_point_ratio",
    "ratio_denominator_pixels",
    "point_pixels",
    "spaced_point_pixels",
    "relaxed_point_pixels",
    "fixed_strip_pixels",
    "artificial_mask_pixels",
    "artificial_mask_ratio_full_image",
    "gap_pixels",
    "image_seed",
)


@dataclass(frozen=True)
class RunConfig:
    algorithm: str
    script_version: str
    preset: str
    ratio_low: float
    ratio_high: float
    seed: int
    gap_scan_margin: int
    gap_low_threshold: float
    gap_high_threshold: float
    candidate_band_count: int
    require_gap: bool
    recursive: bool
    preserve_relative_paths: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generalized RCSGF Mask 10 artificial inputs from "
            "2-D TIFF PGT images containing detector gaps."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PGT TIFF images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which masked Input TIFF images will be written.",
    )

    ratio_group = parser.add_mutually_exclusive_group(required=True)
    ratio_group.add_argument(
        "--preset",
        choices=sorted(PRESET_RATIOS.keys()),
        help=(
            "Dataset preset controlling the 1x1 point-mask ratio: "
            "saxs=0.65-0.70, waxd=0.75-0.80, "
            "agbio3=0.45-0.55, usaxs=0.65-0.70."
        ),
    )
    ratio_group.add_argument(
        "--ratio",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Custom point-mask ratio range, e.g. --ratio 0.60 0.70.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base random seed (default: 100).",
    )
    parser.add_argument(
        "--gap-scan-margin",
        type=int,
        default=25,
        help=(
            "Rows/columns ignored at each outer image edge during full-row/"
            "full-column gap detection (default: 25)."
        ),
    )
    parser.add_argument(
        "--gap-low-threshold",
        type=float,
        default=0.0,
        help=(
            "A pixel <= this value is considered invalid for gap detection "
            "and valid-observation filtering (default: 0)."
        ),
    )
    parser.add_argument(
        "--gap-high-threshold",
        type=float,
        default=60000.0,
        help=(
            "A pixel > this value is considered invalid for full-row/full-column "
            "gap detection (default: 60000). It is not used to reject isolated "
            "high-intensity observed pixels."
        ),
    )
    parser.add_argument(
        "--candidate-band-count",
        type=int,
        default=3,
        choices=(3,),
        help=(
            "Number of gap-width bands on each side of a gap. Mask 10 uses "
            "exactly 3; the option is exposed for provenance and fixed to 3."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input TIFF files recursively.",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help=(
            "Write all TIFFs directly into output-dir. By default, relative "
            "subdirectories are preserved when --recursive is used."
        ),
    )
    parser.add_argument(
        "--allow-no-gap",
        action="store_true",
        help=(
            "Do not fail immediately if an image contains no detected full-row/"
            "full-column detector gap. Such an image will still fail later if "
            "no candidate region can be constructed."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume a previous run with exactly the same configuration. "
            "Existing verified outputs listed in the manifest are skipped."
        ),
    )
    mode_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated TIFFs and start a fresh manifest.",
    )

    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable read-back verification of each saved TIFF.",
    )
    parser.add_argument(
        "--save-mask-npz",
        action="store_true",
        help=(
            "Also save gap/fixed/point/final masks as compressed NPZ files "
            "under output-dir/_mask_npz."
        ),
    )

    return parser.parse_args()


def resolve_ratio(args: argparse.Namespace) -> Tuple[str, float, float]:
    if args.preset is not None:
        low, high = PRESET_RATIOS[args.preset]
        preset = args.preset
    else:
        low, high = map(float, args.ratio)
        preset = "custom"

    if not (0.0 <= low <= high <= 1.0):
        raise ValueError(
            f"Invalid ratio range ({low}, {high}); expected 0 <= LOW <= HIGH <= 1."
        )

    return preset, low, high


def build_run_config(args: argparse.Namespace) -> RunConfig:
    preset, ratio_low, ratio_high = resolve_ratio(args)

    if args.gap_scan_margin < 0:
        raise ValueError("--gap-scan-margin must be >= 0.")
    if args.gap_low_threshold >= args.gap_high_threshold:
        raise ValueError(
            "--gap-low-threshold must be smaller than --gap-high-threshold."
        )

    return RunConfig(
        algorithm=ALGORITHM_NAME,
        script_version=SCRIPT_VERSION,
        preset=preset,
        ratio_low=ratio_low,
        ratio_high=ratio_high,
        seed=int(args.seed),
        gap_scan_margin=int(args.gap_scan_margin),
        gap_low_threshold=float(args.gap_low_threshold),
        gap_high_threshold=float(args.gap_high_threshold),
        candidate_band_count=int(args.candidate_band_count),
        require_gap=not bool(args.allow_no_gap),
        recursive=bool(args.recursive),
        preserve_relative_paths=not bool(args.flat_output),
    )


def config_hash(config: RunConfig) -> str:
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_image_seed(base_seed: int, relative_path: str, cfg_hash: str) -> int:
    payload = f"{base_seed}|{relative_path}|{cfg_hash}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little") % (2**32)


def list_source_images(input_dir: Path, recursive: bool) -> List[Path]:
    iterator: Iterable[Path]
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()

    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    ]
    return sorted(files, key=lambda p: p.as_posix().lower())


def find_contiguous_bounds(flags: np.ndarray) -> List[Tuple[int, int]]:
    """Return [(start, end), ...] for contiguous True runs; end is exclusive."""
    indices = np.flatnonzero(flags)
    if indices.size == 0:
        return []

    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, split_points)
    return [(int(g[0]), int(g[-1]) + 1) for g in groups]


def invalid_pixel_mask(image: np.ndarray, config: RunConfig) -> np.ndarray:
    return (
        ~np.isfinite(image)
        | (image <= config.gap_low_threshold)
        | (image > config.gap_high_threshold)
    )


def valid_observation_mask(image: np.ndarray, config: RunConfig) -> np.ndarray:
    # Match the experimental implementation: the high threshold is used to
    # recognize complete detector-gap rows/columns, not to discard isolated
    # high-intensity measured pixels. Non-positive/non-finite pixels are not
    # used as artificial-mask supervision targets.
    return np.isfinite(image) & (image > config.gap_low_threshold)


def detect_gap_bounds(
    image: np.ndarray,
    config: RunConfig,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Detect all full-row and full-column detector gaps in a 2-D image."""
    if image.ndim != 2:
        raise ValueError(f"Only 2-D TIFF images are supported; got shape={image.shape}.")

    height, width = image.shape
    invalid = invalid_pixel_mask(image, config)

    row_flags = np.all(invalid, axis=1)
    col_flags = np.all(invalid, axis=0)

    row_margin = min(config.gap_scan_margin, max((height - 1) // 2, 0))
    col_margin = min(config.gap_scan_margin, max((width - 1) // 2, 0))

    if row_margin:
        row_flags[:row_margin] = False
        row_flags[height - row_margin :] = False
    if col_margin:
        col_flags[:col_margin] = False
        col_flags[width - col_margin :] = False

    horizontal = find_contiguous_bounds(row_flags)
    vertical = find_contiguous_bounds(col_flags)

    if config.require_gap and not horizontal and not vertical:
        raise RuntimeError(
            "No horizontal or vertical detector gap was detected. "
            "Check the input PGT or the gap thresholds."
        )

    return horizontal, vertical


def build_gap_mask(
    shape: Tuple[int, int],
    horizontal_bounds: Sequence[Tuple[int, int]],
    vertical_bounds: Sequence[Tuple[int, int]],
) -> np.ndarray:
    height, width = shape
    gap_mask = np.zeros(shape, dtype=bool)

    for start, end in horizontal_bounds:
        gap_mask[max(0, start) : min(height, end), :] = True
    for start, end in vertical_bounds:
        gap_mask[:, max(0, start) : min(width, end)] = True

    return gap_mask


def build_candidate_region(
    shape: Tuple[int, int],
    horizontal_bounds: Sequence[Tuple[int, int]],
    vertical_bounds: Sequence[Tuple[int, int]],
    gap_mask: np.ndarray,
    band_count: int = 3,
) -> np.ndarray:
    """Build the union of three gap-width bands on both sides of each gap."""
    height, width = shape
    region = np.zeros(shape, dtype=bool)

    for start, end in horizontal_bounds:
        gap_width = end - start
        region[max(0, start - band_count * gap_width) : start, :] = True
        region[end : min(height, end + band_count * gap_width), :] = True

    for start, end in vertical_bounds:
        gap_width = end - start
        region[:, max(0, start - band_count * gap_width) : start] = True
        region[:, end : min(width, end + band_count * gap_width)] = True

    region[gap_mask] = False
    return region


def build_fixed_shifted_strips(
    shape: Tuple[int, int],
    horizontal_bounds: Sequence[Tuple[int, int]],
    vertical_bounds: Sequence[Tuple[int, int]],
    candidate_region: np.ndarray,
    gap_mask: np.ndarray,
    valid_observed: np.ndarray,
) -> np.ndarray:
    """Generate the fixed strip component of generalized Mask 10.

    For each physical detector gap, the strip width equals the gap width and one
    gap width of spacing is retained between the strip and the gap. A strip is
    clipped around orthogonal gaps together with one gap-width margin.
    """
    height, width = shape
    fixed_mask = np.zeros(shape, dtype=bool)

    # Horizontal gaps -> horizontal shifted strips.
    for h_start, h_end in horizontal_bounds:
        h_width = h_end - h_start
        valid_x = np.ones(width, dtype=bool)

        for v_start, v_end in vertical_bounds:
            v_width = v_end - v_start
            valid_x[
                max(0, v_start - v_width) : min(width, v_end + v_width)
            ] = False

        y_ranges = (
            (h_start - 2 * h_width, h_start - h_width),
            (h_end + h_width, h_end + 2 * h_width),
        )
        for y_start, y_end in y_ranges:
            y0 = max(0, y_start)
            y1 = min(height, y_end)
            if y0 < y1:
                fixed_mask[y0:y1, valid_x] = True

    # Vertical gaps -> vertical shifted strips.
    for v_start, v_end in vertical_bounds:
        v_width = v_end - v_start
        valid_y = np.ones(height, dtype=bool)

        for h_start, h_end in horizontal_bounds:
            h_width = h_end - h_start
            valid_y[
                max(0, h_start - h_width) : min(height, h_end + h_width)
            ] = False

        x_ranges = (
            (v_start - 2 * v_width, v_start - v_width),
            (v_end + v_width, v_end + 2 * v_width),
        )
        for x_start, x_end in x_ranges:
            x0 = max(0, x_start)
            x1 = min(width, x_end)
            if x0 < x1:
                fixed_mask[valid_y, x0:x1] = True

    fixed_mask &= candidate_region
    fixed_mask &= valid_observed
    fixed_mask[gap_mask] = False
    return fixed_mask


def can_place_spaced_point(
    occupied: np.ndarray,
    valid_region: np.ndarray,
    row: int,
    col: int,
) -> bool:
    """Check point placement with no horizontal/vertical adjacent occupancy.

    Diagonal neighbors are permitted, matching the previous implementation.
    """
    if not valid_region[row, col] or occupied[row, col]:
        return False

    height, width = occupied.shape
    if row > 0 and occupied[row - 1, col]:
        return False
    if row + 1 < height and occupied[row + 1, col]:
        return False
    if col > 0 and occupied[row, col - 1]:
        return False
    if col + 1 < width and occupied[row, col + 1]:
        return False

    return True


def generate_point_mask(
    valid_region: np.ndarray,
    occupied: np.ndarray,
    target_pixels: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, int]:
    """Generate 1x1 point masks.

    Spacing-aware placement is attempted first. If the requested ratio cannot
    be reached under that constraint, remaining pixels are sampled from the
    still-unoccupied valid region so that the requested experimental coverage
    can be reproduced exactly.
    """
    point_mask = np.zeros_like(valid_region, dtype=bool)
    coords = np.argwhere(valid_region & ~occupied)

    if len(coords) < target_pixels:
        raise RuntimeError(
            f"Insufficient point-mask candidate pixels: available={len(coords)}, "
            f"target={target_pixels}."
        )

    placed = 0
    for index in rng.permutation(len(coords)):
        if placed >= target_pixels:
            break

        row, col = map(int, coords[int(index)])
        if can_place_spaced_point(occupied, valid_region, row, col):
            point_mask[row, col] = True
            occupied[row, col] = True
            placed += 1

    spaced_pixels = placed
    relaxed_pixels = target_pixels - placed

    if relaxed_pixels > 0:
        remaining = np.argwhere(valid_region & ~occupied)
        if len(remaining) < relaxed_pixels:
            raise RuntimeError(
                "Could not complete the requested point-mask ratio: "
                f"remaining={len(remaining)}, needed={relaxed_pixels}."
            )

        chosen = rng.choice(len(remaining), size=relaxed_pixels, replace=False)
        chosen_coords = remaining[np.atleast_1d(chosen)]
        point_mask[chosen_coords[:, 0], chosen_coords[:, 1]] = True
        occupied[chosen_coords[:, 0], chosen_coords[:, 1]] = True

    if int(point_mask.sum()) != target_pixels:
        raise RuntimeError(
            "Internal error: generated point-mask pixel count does not match target."
        )

    return point_mask, spaced_pixels, relaxed_pixels


def generate_mask10(
    image: np.ndarray,
    config: RunConfig,
    image_seed: int,
) -> Tuple[np.ndarray, Dict[str, object], Dict[str, np.ndarray]]:
    """Generate generalized Mask 10 for one PGT image."""
    horizontal, vertical = detect_gap_bounds(image, config)
    gap_mask = build_gap_mask(image.shape, horizontal, vertical)
    valid_observed = valid_observation_mask(image, config) & ~gap_mask

    candidate_region = build_candidate_region(
        image.shape,
        horizontal,
        vertical,
        gap_mask,
        band_count=config.candidate_band_count,
    )
    candidate_region &= valid_observed

    fixed_mask = build_fixed_shifted_strips(
        image.shape,
        horizontal,
        vertical,
        candidate_region,
        gap_mask,
        valid_observed,
    )

    point_candidate_region = (
        candidate_region & ~gap_mask & ~fixed_mask & valid_observed
    )
    denominator = int(point_candidate_region.sum())
    if denominator <= 0:
        raise RuntimeError(
            "No valid point-mask candidate pixels remain after gap/strip exclusion."
        )

    rng = np.random.default_rng(image_seed)
    sampled_ratio = float(rng.uniform(config.ratio_low, config.ratio_high))
    target_pixels = int(round(denominator * sampled_ratio))
    target_pixels = max(1, min(target_pixels, denominator))

    occupied = (gap_mask | fixed_mask).copy()
    point_mask, spaced_pixels, relaxed_pixels = generate_point_mask(
        point_candidate_region,
        occupied,
        target_pixels,
        rng,
    )

    artificial_mask = fixed_mask | point_mask

    if np.any(artificial_mask & gap_mask):
        raise RuntimeError("Artificial mask overlaps a physical detector gap.")
    if np.any(artificial_mask & ~candidate_region):
        raise RuntimeError("Artificial mask extends outside the candidate region.")
    if np.any(artificial_mask & ~valid_observed):
        raise RuntimeError("Artificial mask includes invalid/non-observed pixels.")

    actual_ratio = float(point_mask.sum()) / denominator
    tolerance = 1.0 / denominator + 1e-12
    if actual_ratio < config.ratio_low - tolerance or actual_ratio > config.ratio_high + tolerance:
        raise RuntimeError(
            "Generated point-mask ratio is outside the requested range: "
            f"actual={actual_ratio:.8f}, requested="
            f"[{config.ratio_low:.8f}, {config.ratio_high:.8f}]."
        )

    stats: Dict[str, object] = {
        "horizontal_gap_count": len(horizontal),
        "vertical_gap_count": len(vertical),
        "horizontal_gap_widths": ";".join(str(e - s) for s, e in horizontal),
        "vertical_gap_widths": ";".join(str(e - s) for s, e in vertical),
        "ratio_reference": "remaining_valid_candidate_pixels_after_gap_and_fixed_strip_exclusion",
        "sampled_target_ratio": f"{sampled_ratio:.8f}",
        "actual_point_ratio": f"{actual_ratio:.8f}",
        "ratio_denominator_pixels": denominator,
        "point_pixels": int(point_mask.sum()),
        "spaced_point_pixels": int(spaced_pixels),
        "relaxed_point_pixels": int(relaxed_pixels),
        "fixed_strip_pixels": int(fixed_mask.sum()),
        "artificial_mask_pixels": int(artificial_mask.sum()),
        "artificial_mask_ratio_full_image": f"{artificial_mask.mean():.8f}",
        "gap_pixels": int(gap_mask.sum()),
    }

    components = {
        "gap_mask": gap_mask,
        "candidate_region": candidate_region,
        "fixed_strip_mask": fixed_mask,
        "point_mask": point_mask,
        "artificial_mask": artificial_mask,
    }
    return artificial_mask, stats, components


def masked_image_from_pgt(pgt: np.ndarray, artificial_mask: np.ndarray) -> np.ndarray:
    masked = pgt.copy()
    masked[artificial_mask] = 0
    if masked.dtype != pgt.dtype:
        raise RuntimeError("Image dtype changed while applying the artificial mask.")
    return masked


def arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:
        # Compatibility with older NumPy versions.
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if np.issubdtype(a.dtype, np.floating):
            return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))
        return bool(np.array_equal(a, b))


def save_and_verify_tiff(
    pgt: np.ndarray,
    masked: np.ndarray,
    artificial_mask: np.ndarray,
    output_path: Path,
    verify: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiff.imwrite(str(output_path), masked, compression=None)

    if not verify:
        return

    restored = tiff.imread(str(output_path))
    if restored.shape != pgt.shape or restored.dtype != pgt.dtype:
        raise RuntimeError(
            f"Saved TIFF shape/dtype mismatch for {output_path}: "
            f"saved={restored.shape}/{restored.dtype}, "
            f"expected={pgt.shape}/{pgt.dtype}."
        )

    if not arrays_equal(restored[~artificial_mask], pgt[~artificial_mask]):
        raise RuntimeError(
            f"Non-artificial-mask pixels changed after saving: {output_path}."
        )

    if not np.all(restored[artificial_mask] == 0):
        raise RuntimeError(
            f"Artificial-mask pixels are not all zero after saving: {output_path}."
        )


def save_mask_npz(
    components: Dict[str, np.ndarray],
    output_root: Path,
    relative_source: Path,
) -> None:
    target = output_root / "_mask_npz" / relative_source.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        **{name: array.astype(np.uint8) for name, array in components.items()},
    )


def output_relative_path(
    source_path: Path,
    input_root: Path,
    preserve_relative_paths: bool,
) -> Path:
    rel = source_path.relative_to(input_root)
    return rel if preserve_relative_paths else Path(source_path.name)


def read_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(MANIFEST_FIELDS).issubset(reader.fieldnames):
            raise RuntimeError(
                f"Existing manifest has incompatible fields: {path}. "
                "Use a new output directory or --overwrite."
            )
        return {
            row["source_relpath"]: row
            for row in reader
            if row.get("source_relpath")
        }


def rewrite_manifest(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    temp.replace(path)


def append_manifest_row(
    manifest_file,
    writer: csv.DictWriter,
    row: Dict[str, object],
) -> None:
    writer.writerow(row)
    manifest_file.flush()
    os.fsync(manifest_file.fileno())


def write_or_validate_run_config(
    output_dir: Path,
    config: RunConfig,
    cfg_hash: str,
    resume: bool,
    overwrite: bool,
) -> None:
    config_path = output_dir / "run_config.json"
    payload = {
        **asdict(config),
        "config_hash": cfg_hash,
    }

    if config_path.exists() and resume:
        with config_path.open("r", encoding="utf-8") as f:
            previous = json.load(f)
        if previous.get("config_hash") != cfg_hash:
            raise RuntimeError(
                "Resume requested, but run_config.json does not match the current "
                "configuration. Use the original parameters, a new output directory, "
                "or --overwrite."
            )
        return

    if config_path.exists() and not overwrite and not resume:
        raise RuntimeError(
            f"Output directory already contains run_config.json: {config_path}. "
            "Use --resume, --overwrite, or a new output directory."
        )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def prepare_output_directory(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
) -> None:
    input_resolved = input_dir.resolve()
    output_resolved = output_dir.resolve()
    if input_resolved == output_resolved:
        raise ValueError("Input and output directories must be different.")

    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        # Only remove files created by this script; never recursively delete the
        # entire user-provided output directory.
        for name in ("dataset_manifest.csv", "run_config.json", "variant_summary.json"):
            path = output_dir / name
            if path.exists():
                path.unlink()


def make_manifest_row(
    source_rel: Path,
    output_rel: Path,
    image: np.ndarray,
    config: RunConfig,
    cfg_hash: str,
    image_seed: int,
    stats: Dict[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "source_relpath": source_rel.as_posix(),
        "output_relpath": output_rel.as_posix(),
        "algorithm": config.algorithm,
        "script_version": config.script_version,
        "config_hash": cfg_hash,
        "dataset_preset": config.preset,
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "dtype": str(image.dtype),
        "ratio_low": config.ratio_low,
        "ratio_high": config.ratio_high,
        "image_seed": int(image_seed),
        **stats,
    }
    return row


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not rows:
        return {"count": 0}

    def mean_of(key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    return {
        "count": len(rows),
        "mean_horizontal_gap_count": mean_of("horizontal_gap_count"),
        "mean_vertical_gap_count": mean_of("vertical_gap_count"),
        "mean_actual_point_ratio": mean_of("actual_point_ratio"),
        "mean_fixed_strip_pixels": mean_of("fixed_strip_pixels"),
        "mean_artificial_mask_ratio_full_image": mean_of(
            "artificial_mask_ratio_full_image"
        ),
        "total_relaxed_point_pixels": int(
            sum(int(row["relaxed_point_pixels"]) for row in rows)
        ),
    }


def main() -> int:
    args = parse_args()
    config = build_run_config(args)
    cfg_hash = config_hash(config)

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    prepare_output_directory(input_dir, output_dir, overwrite=args.overwrite)
    write_or_validate_run_config(
        output_dir,
        config,
        cfg_hash,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    source_files = list_source_images(input_dir, recursive=config.recursive)
    if not source_files:
        raise RuntimeError(f"No TIFF images found in {input_dir}.")

    manifest_path = output_dir / "dataset_manifest.csv"
    existing_rows = read_manifest(manifest_path) if args.resume else {}

    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    mode = "a" if args.resume and manifest_path.exists() else "w"

    all_rows: Dict[str, Dict[str, object]] = {
        key: dict(value) for key, value in existing_rows.items()
    }
    generated = 0
    skipped = 0

    print(f"RCSGF generalized Mask 10 generator v{SCRIPT_VERSION}")
    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Images          : {len(source_files)}")
    print(
        f"Point ratio     : {config.ratio_low:.2f}-{config.ratio_high:.2f} "
        f"(preset={config.preset})"
    )
    print(f"Base seed       : {config.seed}")
    print(f"Config hash     : {cfg_hash}")

    with manifest_path.open(mode, newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        if mode == "w":
            writer.writeheader()
            manifest_file.flush()

        for source_path in tqdm(source_files, desc="Generating Mask 10 inputs"):
            source_rel = source_path.relative_to(input_dir)
            output_rel = output_relative_path(
                source_path,
                input_dir,
                preserve_relative_paths=config.preserve_relative_paths,
            )
            output_path = output_dir / output_rel
            source_key = source_rel.as_posix()

            previous = all_rows.get(source_key)
            if args.resume and previous is not None:
                same_config = previous.get("config_hash") == cfg_hash
                same_output = previous.get("output_relpath") == output_rel.as_posix()
                if same_config and same_output and output_path.is_file():
                    skipped += 1
                    continue

            image = tiff.imread(str(source_path))
            if image.ndim != 2:
                raise ValueError(
                    f"Only 2-D TIFF images are supported: {source_path} has "
                    f"shape={image.shape}."
                )

            image_seed = stable_image_seed(config.seed, source_key, cfg_hash)
            artificial_mask, stats, components = generate_mask10(
                image,
                config,
                image_seed,
            )
            masked = masked_image_from_pgt(image, artificial_mask)

            if output_path.exists() and not (args.overwrite or args.resume):
                raise FileExistsError(
                    f"Output already exists: {output_path}. Use --overwrite or --resume."
                )

            save_and_verify_tiff(
                image,
                masked,
                artificial_mask,
                output_path,
                verify=not args.no_verify,
            )

            if args.save_mask_npz:
                save_mask_npz(components, output_dir, source_rel)

            row = make_manifest_row(
                source_rel,
                output_rel,
                image,
                config,
                cfg_hash,
                image_seed,
                stats,
            )
            all_rows[source_key] = row
            append_manifest_row(manifest_file, writer, row)
            generated += 1

    # Rebuild a clean, unique manifest after a successful pass.
    ordered_rows = [
        all_rows[path.relative_to(input_dir).as_posix()]
        for path in source_files
        if path.relative_to(input_dir).as_posix() in all_rows
    ]
    if len(ordered_rows) != len(source_files):
        raise RuntimeError(
            "Manifest sample count mismatch after processing: "
            f"expected={len(source_files)}, actual={len(ordered_rows)}."
        )

    rewrite_manifest(manifest_path, ordered_rows)

    # Validate that every expected generated TIFF exists.
    missing_outputs = []
    for source_path in source_files:
        output_rel = output_relative_path(
            source_path,
            input_dir,
            preserve_relative_paths=config.preserve_relative_paths,
        )
        if not (output_dir / output_rel).is_file():
            missing_outputs.append(output_rel.as_posix())

    if missing_outputs:
        preview = ", ".join(missing_outputs[:5])
        raise RuntimeError(
            f"Missing {len(missing_outputs)} expected output TIFF(s), e.g. {preview}."
        )

    summary = summarize_rows(ordered_rows)
    summary_path = output_dir / "variant_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print("\nCompleted successfully.")
    print(f"Generated: {generated}")
    print(f"Skipped  : {skipped}")
    print(f"Manifest : {manifest_path}")
    print(f"Summary  : {summary_path}")
    if args.save_mask_npz:
        print(f"Masks    : {output_dir / '_mask_npz'}")

    if summary.get("total_relaxed_point_pixels", 0):
        print(
            "Note: spacing-relaxed point placement was required to attain the "
            "requested point-mask coverage. See relaxed_point_pixels in the manifest."
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
