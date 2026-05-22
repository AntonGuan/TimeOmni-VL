#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
METRICS_ROOT = Path(__file__).resolve().parent
if str(METRICS_ROOT) not in sys.path:
    sys.path.insert(0, str(METRICS_ROOT))

from reconstruct import reconstruct_timeseries


@dataclass
class EvalItem:
    output_image: Path
    source_image: Optional[Path]
    target_image: Optional[Path]
    metadata: Path
    series: Path
    instruction: str = ""
    thinking: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate imputation edit images with metadata mask ranges. "
            "The MASE numerator is computed on masked points; the denominator "
            "is the in-sample seasonal naive MAE from observed, unmasked points."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "eval/outputs/imputation_demo",
        help="Directory that contains generated edit images and optionally metrics.csv.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Inference metrics.csv to read and update. Defaults to <output-root>/metrics.csv.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "data_pipeline/demo_level_samples",
        help="Root used to resolve sample folders.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional imputation JSONL. Used as an additional source-image lookup table.",
    )
    parser.add_argument(
        "--output-name",
        default="edit.png",
        help="Generated image filename to scan when metrics.csv is not available.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Raise immediately on the first failed sample instead of writing skip rows.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_path(value: Any) -> Optional[Path]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _resolve_path(path: Optional[Path], base: Path) -> Optional[Path]:
    if path is None:
        return None
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, base / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / path


def _load_jsonl_lookup(jsonl_path: Optional[Path], input_root: Path) -> Dict[str, Dict[str, str]]:
    if jsonl_path is None or not jsonl_path.exists():
        return {}

    lookup: Dict[str, Dict[str, str]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            source = _resolve_path(_as_path(row.get("source_image")), input_root)
            if source is None:
                continue
            metadata = _resolve_path(_as_path(row.get("metadata")), input_root)
            if metadata is None:
                metadata = source.parent / "metadata.json"
            target = _resolve_path(_as_path(row.get("target_image")), input_root)

            keys = {str(source)}
            try:
                keys.add(str(source.resolve()))
            except OSError:
                pass
            for key in keys:
                lookup[key] = {
                    "metadata": str(metadata),
                    "target_image": str(target) if target is not None else "",
                    "instruction": str(row.get("instruction", "")),
                    "thinking": str(row.get("thinking", "")),
                }
    return lookup


def _metadata_from_source(
    source_image: Optional[Path],
    input_root: Path,
    jsonl_lookup: Dict[str, Dict[str, str]],
) -> Optional[Path]:
    if source_image is None:
        return None

    keys = [str(source_image)]
    try:
        keys.append(str(source_image.resolve()))
    except OSError:
        pass
    for key in keys:
        if key in jsonl_lookup:
            return Path(jsonl_lookup[key]["metadata"])

    candidate = source_image.parent / "metadata.json"
    if candidate.exists():
        return candidate
    candidate = input_root / source_image.parent / "metadata.json"
    if candidate.exists():
        return candidate
    return None


def _series_from_metadata(metadata_path: Path) -> Path:
    return metadata_path.with_name("series.npz")


def _item_from_metrics_row(
    row: pd.Series,
    input_root: Path,
    jsonl_lookup: Dict[str, Dict[str, str]],
) -> EvalItem:
    output_image = _resolve_path(_as_path(row.get("output_image")), Path.cwd())
    if output_image is None:
        raise ValueError("metrics row is missing output_image")

    source_image = _resolve_path(_as_path(row.get("source_image")), input_root)
    target_image = _resolve_path(_as_path(row.get("target_image")), input_root)
    metadata = _resolve_path(_as_path(row.get("metadata")), input_root)
    if metadata is None:
        metadata = _metadata_from_source(source_image, input_root, jsonl_lookup)
    if metadata is None:
        raise ValueError(f"cannot resolve metadata for output image {output_image}")

    return EvalItem(
        output_image=output_image,
        source_image=source_image,
        target_image=target_image,
        metadata=metadata,
        series=_series_from_metadata(metadata),
        instruction=str(row.get("instruction", "")),
        thinking=str(row.get("thinking", "")),
    )


def _scan_edit_images(output_root: Path, input_root: Path, output_name: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    for output_image in sorted(output_root.rglob(output_name)):
        rel_parent = output_image.parent.relative_to(output_root)
        sample_dir = input_root / rel_parent
        items.append(
            EvalItem(
                output_image=output_image,
                source_image=sample_dir / "image_mask.png",
                target_image=sample_dir / "image_full.png",
                metadata=sample_dir / "metadata.json",
                series=sample_dir / "series.npz",
            )
        )
    return items


def _load_items(args: argparse.Namespace) -> List[EvalItem]:
    jsonl_lookup = _load_jsonl_lookup(args.jsonl, args.input_root)
    metrics_csv = args.metrics_csv
    if metrics_csv is None:
        default_metrics = args.output_root / "metrics.csv"
        metrics_csv = default_metrics if default_metrics.exists() else None

    if metrics_csv is not None:
        df = pd.read_csv(metrics_csv)
        return [_item_from_metrics_row(row, args.input_root, jsonl_lookup) for _, row in df.iterrows()]

    return _scan_edit_images(args.output_root, args.input_root, args.output_name)


def _ensure_2d_time_major(array: np.ndarray, expected_nvars: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"expected a 1D or 2D array, got shape {arr.shape}")

    if expected_nvars is not None and arr.shape[0] == expected_nvars and arr.shape[1] != expected_nvars:
        arr = arr.T
    return arr


def _load_true_series(series_path: Path, nvars: int) -> np.ndarray:
    npz = np.load(series_path)
    if "series" in npz:
        return _ensure_2d_time_major(npz["series"], expected_nvars=nvars)
    if "context" in npz and "pred" in npz:
        context = _ensure_2d_time_major(npz["context"], expected_nvars=nvars)
        pred = _ensure_2d_time_major(npz["pred"], expected_nvars=nvars)
        return np.concatenate([context, pred], axis=0)
    raise KeyError(f"{series_path} must contain 'series' or both 'context' and 'pred'")


def _metadata_for_full_reconstruction(metadata: Dict[str, Any], total_len: int) -> Dict[str, Any]:
    working = dict(metadata)
    working.setdefault("context_len", 0)
    working.setdefault("pred_len", int(working.get("total_len", total_len)))
    working.setdefault("total_len", int(working["context_len"]) + int(working["pred_len"]))
    return working


def _reconstruct_full_series(output_image: Path, metadata: Dict[str, Any], total_len: int) -> np.ndarray:
    reconstruction_metadata = _metadata_for_full_reconstruction(metadata, total_len)
    _ctx, _pred, full = reconstruct_timeseries(output_image, reconstruction_metadata, denormalize=True)
    return _ensure_2d_time_major(full, expected_nvars=int(metadata.get("nvars", 1)))


def _metadata_seasonality(metadata: Dict[str, Any]) -> int:
    for key in ("generated_period", "periodicity"):
        if key in metadata:
            value = int(metadata[key])
            if value > 0:
                return value
    return 1


def _normalize_range(raw_range: Sequence[Any], total_len: int) -> Optional[Tuple[int, int]]:
    if len(raw_range) != 2:
        return None
    start = max(0, int(raw_range[0]))
    end = min(total_len, int(raw_range[1]))
    if end <= start:
        return None
    return start, end


def _ranges_for_var(mask_ranges: Any, var_idx: int, nvars: int, total_len: int) -> List[Tuple[int, int]]:
    if not isinstance(mask_ranges, list) or not mask_ranges:
        return []

    if len(mask_ranges) == nvars and isinstance(mask_ranges[var_idx], list):
        var_ranges = mask_ranges[var_idx]
        if len(var_ranges) == 2 and all(isinstance(v, (int, float)) for v in var_ranges):
            normalized = _normalize_range(var_ranges, total_len)
            return [normalized] if normalized is not None else []
        if all(isinstance(r, list) for r in var_ranges):
            return [
                normalized
                for raw in var_ranges
                if (normalized := _normalize_range(raw, total_len)) is not None
            ]

    if all(isinstance(r, list) for r in mask_ranges):
        return [
            normalized
            for raw in mask_ranges
            if (normalized := _normalize_range(raw, total_len)) is not None
        ]
    return []


def _mask_for_ranges(ranges: Sequence[Tuple[int, int]], total_len: int) -> np.ndarray:
    mask = np.zeros(total_len, dtype=bool)
    for start, end in ranges:
        mask[start:end] = True
    return mask


def _seasonal_error(
    values: np.ndarray,
    seasonality: int,
    observed_mask: np.ndarray,
) -> float:
    """Return in-sample seasonal naive MAE from observed training points."""
    values = np.asarray(values, dtype=np.float64)
    observed_mask = np.asarray(observed_mask, dtype=bool)
    if values.size < 2:
        return float("nan")

    lag = max(1, int(seasonality))
    if lag > values.size:
        lag = 1

    curr = values[lag:]
    prev = values[:-lag]
    valid = observed_mask[lag:] & observed_mask[:-lag] & np.isfinite(curr) & np.isfinite(prev)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(curr[valid] - prev[valid])))


def _safe_dataset_label(metadata: Dict[str, Any], metadata_path: Path, input_root: Path) -> str:
    source_dataset = metadata.get("source_dataset")
    if source_dataset:
        return str(source_dataset)
    try:
        rel = metadata_path.parent.relative_to(input_root)
        if len(rel.parts) >= 3:
            return "/".join(rel.parts[:-2])
    except ValueError:
        pass
    return ""


def _safe_term(metadata_path: Path, input_root: Path) -> str:
    try:
        rel = metadata_path.parent.relative_to(input_root)
        if len(rel.parts) >= 3:
            return rel.parts[-2]
    except ValueError:
        pass
    return metadata_path.parent.parent.name if metadata_path.parent.parent.name else ""


def _evaluate_item(item: EvalItem, input_root: Path) -> List[Dict[str, Any]]:
    if not item.output_image.exists():
        raise FileNotFoundError(f"output image not found: {item.output_image}")
    if not item.metadata.exists():
        raise FileNotFoundError(f"metadata not found: {item.metadata}")
    if not item.series.exists():
        raise FileNotFoundError(f"series.npz not found: {item.series}")

    metadata = _read_json(item.metadata)
    nvars = int(metadata.get("nvars", 1))
    seasonality = _metadata_seasonality(metadata)
    true_full = _load_true_series(item.series, nvars)
    pred_full = _reconstruct_full_series(item.output_image, metadata, total_len=true_full.shape[0])

    total_len = min(true_full.shape[0], pred_full.shape[0], int(metadata.get("total_len", true_full.shape[0])))
    nvars_eval = min(true_full.shape[1], pred_full.shape[1], nvars)
    mask_ranges = metadata.get("mask_ranges")
    dataset = _safe_dataset_label(metadata, item.metadata, input_root)
    term = _safe_term(item.metadata, input_root)
    freq = str(metadata.get("freq", ""))

    rows: List[Dict[str, Any]] = []
    for var_idx in range(nvars_eval):
        ranges = _ranges_for_var(mask_ranges, var_idx, nvars_eval, total_len)
        if not ranges:
            continue

        y_true = true_full[:total_len, var_idx]
        y_pred = pred_full[:total_len, var_idx]
        var_mask = _mask_for_ranges(ranges, total_len)
        observed_mask = ~var_mask
        seasonal_error = _seasonal_error(y_true, seasonality, observed_mask)

        for range_idx, (start, end) in enumerate(ranges):
            eval_mask = np.zeros(total_len, dtype=bool)
            eval_mask[start:end] = True
            valid = eval_mask & np.isfinite(y_true) & np.isfinite(y_pred)

            abs_error = np.abs(y_true[valid] - y_pred[valid])
            if np.isfinite(seasonal_error) and not np.isclose(seasonal_error, 0.0):
                scaled_error = abs_error / seasonal_error
                mase_valid_count = int(valid.sum())
                sum_scaled_abs_error = float(np.sum(scaled_error)) if scaled_error.size else float("nan")
                sum_mase_abs_error = float(np.sum(abs_error)) if abs_error.size else float("nan")
            else:
                scaled_error = np.array([], dtype=np.float64)
                mase_valid_count = 0
                sum_scaled_abs_error = float("nan")
                sum_mase_abs_error = float("nan")

            mae = float(np.mean(abs_error)) if abs_error.size else float("nan")
            mase = float(np.mean(scaled_error)) if scaled_error.size else float("nan")

            rows.append(
                {
                    "sample": item.metadata.parent.name,
                    "dataset": dataset,
                    "freq": freq,
                    "term": term,
                    "var": var_idx,
                    "range_index": range_idx,
                    "range_start": start,
                    "range_end": end,
                    "output_image": str(item.output_image),
                    "source_image": str(item.source_image) if item.source_image else "",
                    "target_image": str(item.target_image) if item.target_image else "",
                    "metadata": str(item.metadata),
                    "series": str(item.series),
                    "seasonality": seasonality,
                    "valid_count": int(valid.sum()),
                    "mase_valid_count": mase_valid_count,
                    "sum_abs_error": float(np.sum(abs_error)) if abs_error.size else float("nan"),
                    "sum_mase_abs_error": sum_mase_abs_error,
                    "sum_scaled_abs_error": sum_scaled_abs_error,
                    "mae": mae,
                    "season_naive_mae": seasonal_error,
                    "mase": mase,
                }
            )

    if not rows:
        raise ValueError(f"no valid mask ranges found in metadata: {item.metadata}")
    return rows


def _skip_row(item: EvalItem, error: Exception) -> Dict[str, Any]:
    return {
        "sample": item.metadata.parent.name if item.metadata else "",
        "dataset": "",
        "freq": "",
        "term": "",
        "var": "",
        "range_index": "",
        "range_start": "",
        "range_end": "",
        "output_image": str(item.output_image),
        "source_image": str(item.source_image) if item.source_image else "",
        "target_image": str(item.target_image) if item.target_image else "",
        "metadata": str(item.metadata),
        "series": str(item.series),
        "seasonality": "",
        "valid_count": 0,
        "mase_valid_count": 0,
        "sum_abs_error": float("nan"),
        "sum_mase_abs_error": float("nan"),
        "sum_scaled_abs_error": float("nan"),
        "mae": float("nan"),
        "season_naive_mae": float("nan"),
        "mase": float("nan"),
        "error": str(error),
    }


def _weighted(values: pd.Series, counts: pd.Series) -> float:
    valid = values.notna() & counts.notna() & (counts > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=counts[valid]))


def _aggregate(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    valid_df = df[df["valid_count"].fillna(0) > 0].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "num_rows",
                "valid_count",
                "mase_valid_count",
                "mae",
                "season_naive_mae",
                "mase",
            ]
        )

    grouped = [((), valid_df)] if not group_cols else valid_df.groupby(list(group_cols), dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        sum_abs = group["sum_abs_error"].sum(skipna=True)
        sum_mase_abs = group["sum_mase_abs_error"].sum(skipna=True)
        sum_scaled = group["sum_scaled_abs_error"].sum(skipna=True)
        valid_count = int(group["valid_count"].sum())
        mase_valid_count = int(group["mase_valid_count"].sum())
        mase = float(sum_scaled / mase_valid_count) if mase_valid_count else float("nan")

        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "num_rows": int(len(group)),
                "valid_count": valid_count,
                "mase_valid_count": mase_valid_count,
                "mae": float(sum_abs / valid_count) if valid_count else float("nan"),
                "season_naive_mae": (
                    float(sum_mase_abs / sum_scaled)
                    if sum_scaled and not np.isclose(sum_scaled, 0.0)
                    else float("nan")
                ),
                "mase": mase,
                "mean_range_mase": float(group["mase"].mean(skipna=True)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _sample_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["sample", "output_image"]
    return _aggregate(detail_df, cols)


def _update_metrics_csv(metrics_csv: Path, sample_df: pd.DataFrame) -> None:
    metrics_df = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()
    if metrics_df.empty:
        sample_df[["output_image", "mae", "season_naive_mae", "mase"]].to_csv(metrics_csv, index=False)
        return

    for col in ("mae", "season_naive_mae", "mase"):
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan

    lookup = sample_df.set_index("output_image")[["mae", "season_naive_mae", "mase"]]
    for idx, row in metrics_df.iterrows():
        output_image = row.get("output_image")
        if output_image in lookup.index:
            metrics_df.loc[idx, ["mae", "season_naive_mae", "mase"]] = lookup.loc[output_image].values

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)


def main() -> None:
    args = _parse_args()
    metrics_csv = args.metrics_csv or (args.output_root / "metrics.csv")

    items = _load_items(args)
    if not items:
        raise FileNotFoundError(
            f"No evaluation items found under {args.output_root}. "
            f"Provide --metrics-csv or ensure {args.output_name} files exist."
        )

    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        try:
            rows.extend(_evaluate_item(item, args.input_root))
        except Exception as exc:  # noqa: BLE001
            if args.fail_fast:
                raise
            rows.append(_skip_row(item, exc))
            print(f"[skip] {idx}/{len(items)} {item.output_image}: {exc}")
            continue
        print(f"[ok] {idx}/{len(items)} {item.output_image}")

    detail_df = pd.DataFrame(rows)
    sample_df = _sample_summary(detail_df)

    _update_metrics_csv(metrics_csv, sample_df)
    print(f"Updated metrics -> {metrics_csv}")


if __name__ == "__main__":
    main()
