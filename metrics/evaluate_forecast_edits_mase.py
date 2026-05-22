#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    metadata: Path
    series: Path
    instruction: str = ""
    thinking: str = ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated forecast edit images with metadata-based MASE."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "eval/outputs/model_edits",
        help="Directory that contains generated edit images and optionally metrics.csv.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Inference metrics.csv. Defaults to <output-root>/metrics.csv if it exists.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=REPO_ROOT / "data_pipeline/forecast_benchmark_samples",
        help="Original benchmark sample root used to resolve metadata when scanning edit images.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Optional generation JSONL. Used as an additional metadata lookup table.",
    )
    parser.add_argument(
        "--output-name",
        default="edit.png",
        help="Generated image filename to scan when metrics.csv is not available.",
    )
    parser.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Per-variable output CSV. Defaults to <output-root>/mase_metrics.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Grouped summary CSV. Defaults to <output-root>/mase_summary.csv.",
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
            metadata = _resolve_path(_as_path(row.get("metadata")), input_root)
            if source is None or metadata is None:
                continue
            keys = {str(source)}
            try:
                keys.add(str(source.resolve()))
            except OSError:
                pass
            for key in keys:
                lookup[key] = {
                    "metadata": str(metadata),
                    "instruction": str(row.get("instruction", "")),
                    "thinking": str(row.get("thinking", "")),
                }
    return lookup


def _metadata_from_source(
    source_image: Optional[Path],
    input_root: Path,
    jsonl_lookup: Dict[str, Dict[str, str]],
) -> Optional[Path]:
    if source_image is not None:
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
    metadata = _resolve_path(_as_path(row.get("metadata")), input_root)
    if metadata is None:
        metadata = _metadata_from_source(source_image, input_root, jsonl_lookup)
    if metadata is None:
        raise ValueError(f"cannot resolve metadata for output image {output_image}")

    return EvalItem(
        output_image=output_image,
        source_image=source_image,
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
        metadata = sample_dir / "metadata.json"
        items.append(
            EvalItem(
                output_image=output_image,
                source_image=sample_dir / "image_mask.png",
                metadata=metadata,
                series=sample_dir / "series.npz",
            )
        )
    return items


def _load_items(args: argparse.Namespace) -> List[EvalItem]:
    input_root = args.input_root
    output_root = args.output_root
    jsonl_lookup = _load_jsonl_lookup(args.jsonl, input_root)

    metrics_csv = args.metrics_csv
    if metrics_csv is None:
        default_metrics = output_root / "metrics.csv"
        metrics_csv = default_metrics if default_metrics.exists() else None

    if metrics_csv is not None:
        df = pd.read_csv(metrics_csv)
        return [
            _item_from_metrics_row(row, input_root, jsonl_lookup)
            for _, row in df.iterrows()
        ]

    return _scan_edit_images(output_root, input_root, args.output_name)


def _ensure_2d_time_major(array: np.ndarray, expected_nvars: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim != 2:
        raise ValueError(f"expected a 1D or 2D array, got shape {arr.shape}")

    if expected_nvars is not None and arr.shape[0] == expected_nvars and arr.shape[1] != expected_nvars:
        arr = arr.T
    return arr


def _metadata_seasonality(metadata: Dict[str, Any]) -> int:
    for key in ("generated_period", "periodicity"):
        if key in metadata:
            value = int(metadata[key])
            if value > 0:
                return value
    return 1


def _seasonal_error(context: np.ndarray, seasonality: int) -> float:
    values = np.asarray(context, dtype=np.float64)
    if values.size < 2:
        return float("nan")

    lag = int(seasonality)
    if lag <= 0:
        lag = 1
    if lag > values.size:
        lag = 1

    curr = values[lag:]
    prev = values[:-lag]
    valid = np.isfinite(curr) & np.isfinite(prev)
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(curr[valid] - prev[valid])))


def _safe_term(metadata_path: Path, input_root: Path) -> str:
    try:
        rel = metadata_path.parent.relative_to(input_root)
        if len(rel.parts) >= 3:
            return rel.parts[-2]
    except ValueError:
        pass
    return metadata_path.parent.parent.name if metadata_path.parent.parent.name else ""


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


def _evaluate_item(
    item: EvalItem,
    input_root: Path,
) -> List[Dict[str, Any]]:
    if not item.output_image.exists():
        raise FileNotFoundError(f"output image not found: {item.output_image}")
    if not item.metadata.exists():
        raise FileNotFoundError(f"metadata not found: {item.metadata}")
    if not item.series.exists():
        raise FileNotFoundError(f"series.npz not found: {item.series}")

    metadata = _read_json(item.metadata)
    nvars = int(metadata.get("nvars", 1))
    context_len = int(metadata.get("context_len", 0))
    seasonality = _metadata_seasonality(metadata)

    _, pred_hat, _ = reconstruct_timeseries(item.output_image, metadata, denormalize=True)

    npz = np.load(item.series)
    context = _ensure_2d_time_major(npz["context"], expected_nvars=nvars)
    pred_true = _ensure_2d_time_major(npz["pred"], expected_nvars=nvars)
    pred_hat = _ensure_2d_time_major(pred_hat, expected_nvars=nvars)

    nvars_eval = min(context.shape[1], pred_true.shape[1], pred_hat.shape[1], nvars)
    horizon = min(pred_true.shape[0], pred_hat.shape[0])
    dataset = _safe_dataset_label(metadata, item.metadata, input_root)
    term = _safe_term(item.metadata, input_root)
    freq = str(metadata.get("freq", ""))
    var_indices = metadata.get("var_indices")
    if not isinstance(var_indices, Iterable) or isinstance(var_indices, (str, bytes)):
        var_indices = list(range(nvars_eval))
    var_indices = list(var_indices)

    rows: List[Dict[str, Any]] = []
    for var_idx in range(nvars_eval):
        y_context = context[:, var_idx]
        y_true = pred_true[:horizon, var_idx]
        y_pred = pred_hat[:horizon, var_idx]
        valid = np.isfinite(y_true) & np.isfinite(y_pred)

        abs_error = np.abs(y_true[valid] - y_pred[valid])
        seasonal_error = _seasonal_error(y_context, seasonality)
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_error = abs_error / seasonal_error

        mae = float(np.mean(abs_error)) if abs_error.size else float("nan")
        mase = float(np.mean(scaled_error)) if scaled_error.size else float("nan")
        original_var = var_indices[var_idx] if var_idx < len(var_indices) else var_idx

        rows.append(
            {
                "sample": item.metadata.parent.name,
                "dataset": dataset,
                "freq": freq,
                "term": term,
                "source_index": metadata.get("source_index"),
                "part_index": metadata.get("part_index", 0),
                "part_count": metadata.get("part_count", 1),
                "var": var_idx,
                "original_var": original_var,
                "output_image": str(item.output_image),
                "source_image": str(item.source_image) if item.source_image else "",
                "metadata": str(item.metadata),
                "series": str(item.series),
                "seasonality": seasonality,
                "context_len": context_len,
                "pred_len_true": int(pred_true.shape[0]),
                "pred_len_reconstructed": int(pred_hat.shape[0]),
                "eval_horizon": int(horizon),
                "valid_count": int(valid.sum()),
                "sum_abs_error": float(np.sum(abs_error)) if abs_error.size else float("nan"),
                "sum_scaled_abs_error": (
                    float(np.sum(scaled_error)) if scaled_error.size else float("nan")
                ),
                "mae": mae,
                "seasonal_error": seasonal_error,
                "mase": mase,
            }
        )

    return rows


def _skip_row(item: EvalItem, error: Exception) -> Dict[str, Any]:
    return {
        "sample": item.metadata.parent.name if item.metadata else "",
        "dataset": "",
        "freq": "",
        "term": "",
        "source_index": "",
        "part_index": "",
        "part_count": "",
        "var": "",
        "original_var": "",
        "output_image": str(item.output_image),
        "source_image": str(item.source_image) if item.source_image else "",
        "metadata": str(item.metadata),
        "series": str(item.series),
        "seasonality": "",
        "context_len": "",
        "pred_len_true": "",
        "pred_len_reconstructed": "",
        "eval_horizon": "",
        "valid_count": 0,
        "sum_abs_error": float("nan"),
        "sum_scaled_abs_error": float("nan"),
        "mae": float("nan"),
        "seasonal_error": float("nan"),
        "mase": float("nan"),
        "error": str(error),
    }


def _weighted_mase(df: pd.DataFrame) -> float:
    working = df.dropna(subset=["sum_scaled_abs_error", "valid_count"])
    working = working[working["valid_count"] > 0]
    if working.empty:
        return float("nan")
    return float(working["sum_scaled_abs_error"].sum() / working["valid_count"].sum())


def _weighted_mae(df: pd.DataFrame) -> float:
    working = df.dropna(subset=["sum_abs_error", "valid_count"])
    working = working[working["valid_count"] > 0]
    if working.empty:
        return float("nan")
    return float(working["sum_abs_error"].sum() / working["valid_count"].sum())


def _summarize(detail_df: pd.DataFrame) -> pd.DataFrame:
    valid_df = detail_df[detail_df["valid_count"].fillna(0) > 0].copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "dataset",
                "freq",
                "term",
                "num_rows",
                "num_samples",
                "valid_count",
                "mae",
                "mase",
                "mean_var_mase",
            ]
        )

    rows: List[Dict[str, Any]] = []

    def add_group(group_name: str, group_df: pd.DataFrame, labels: Dict[str, Any]) -> None:
        rows.append(
            {
                "group": group_name,
                "dataset": labels.get("dataset", ""),
                "freq": labels.get("freq", ""),
                "term": labels.get("term", ""),
                "num_rows": int(len(group_df)),
                "num_samples": int(group_df["metadata"].nunique()),
                "valid_count": int(group_df["valid_count"].sum()),
                "mae": _weighted_mae(group_df),
                "mase": _weighted_mase(group_df),
                "mean_var_mase": float(group_df["mase"].mean()),
            }
        )

    add_group("overall", valid_df, {})

    for term, group_df in valid_df.groupby("term", dropna=False):
        add_group("term", group_df, {"term": term})

    for (dataset, freq, term), group_df in valid_df.groupby(["dataset", "freq", "term"], dropna=False):
        add_group(
            "dataset_freq_term",
            group_df,
            {"dataset": dataset, "freq": freq, "term": term},
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = _parse_args()
    detail_csv = args.detail_csv or (args.output_root / "mase_metrics.csv")
    summary_csv = args.summary_csv or (args.output_root / "mase_summary.csv")

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
    summary_df = _summarize(detail_df)

    detail_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Wrote detail metrics -> {detail_csv}")
    print(f"Wrote summary metrics -> {summary_csv}")


if __name__ == "__main__":
    main()
