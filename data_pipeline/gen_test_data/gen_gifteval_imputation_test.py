# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GIFT_EVAL_DATA_ROOT = DATA_PIPELINE_ROOT / "GiftEval"

for gift_eval_src in (
    DATA_PIPELINE_ROOT / "gift-eval" / "src",
    PROJECT_ROOT / "gift-eval" / "src",
    PROJECT_ROOT.parent / "gift-eval" / "src",
):
    if gift_eval_src.exists() and str(gift_eval_src) not in sys.path:
        sys.path.insert(0, str(gift_eval_src))
        break

import json
import math
import random
import re
import itertools
import csv
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*'.+' is deprecated and will be removed in a future version, please use '.+' instead\.",
    category=FutureWarning,
)

import numpy as np
import pandas as pd
import torch
from PIL import Image

from gift_eval.data import Dataset
from data_pipeline.bi_tsi.imputation_adapter import VisionTSImputationConverter


POSSIBLE_SEASONALITIES = {
    "S": [3600],
    "T": [1440],
    "H": [24, 168],
    "D": [7],
    "B": [5],
    "W": [52],
    "M": [12],
    "Q": [4],
}

MAX_VARS_PER_IMAGE = 7

FREQ_NAME_ALIASES = {
    "s": "S",
    "sec": "S",
    "second": "S",
    "seconds": "S",
    "secondly": "S",
    "t": "T",
    "min": "T",
    "mins": "T",
    "minute": "T",
    "minutes": "T",
    "minutely": "T",
    "h": "H",
    "hr": "H",
    "hrs": "H",
    "hour": "H",
    "hours": "H",
    "hourly": "H",
    "d": "D",
    "day": "D",
    "days": "D",
    "daily": "D",
    "b": "B",
    "business": "B",
    "businessday": "B",
    "businessdays": "B",
    "w": "W",
    "week": "W",
    "weeks": "W",
    "weekly": "W",
    "m": "M",
    "month": "M",
    "months": "M",
    "monthly": "M",
    "q": "Q",
    "quarter": "Q",
    "quarters": "Q",
    "quarterly": "Q",
}

DATASET_NAME_ALIASES = {
    "kdd_cup_2018": "kdd_cup_2018_with_missing",
    "car_parts": "car_parts_with_missing",
    "loop_seattle": "LOOP_SEATTLE",
    "m_dense": "M_DENSE",
    "sz_taxi": "SZ_TAXI",
    "temperature_rain": "temperature_rain_with_missing",
    "saugeen": "saugeenday",
    "restaurant/D": "restaurant",
}


def _norm_freq_str(freq_str: str) -> str:
    base_freq = str(freq_str).split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def _normalize_freq_for_period(freq_str: str) -> str:
    raw = str(freq_str).strip()
    if not raw:
        raise ValueError("Empty freq string.")
    try:
        offset = pd.tseries.frequencies.to_offset(raw)
        base = _norm_freq_str(offset.name)
        if base.lower() == "min":
            base = "T"
        if offset.n and offset.n != 1:
            return f"{offset.n}{base}"
        return base
    except Exception:
        pass

    lower = raw.lower()
    if lower in FREQ_NAME_ALIASES:
        return FREQ_NAME_ALIASES[lower]

    match = re.match(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$", raw)
    if match:
        multiplier = match.group(1)
        unit = match.group(2).lower()
        base = FREQ_NAME_ALIASES.get(unit)
        if base:
            return f"{multiplier}{base}"

    raise ValueError(f"Unrecognized frequency string: {freq_str}")


def _split_freq(freq_str: str):
    base_freq_char = "".join([c for c in str(freq_str) if c.isalpha()]).upper()
    if base_freq_char == "MIN":
        base_freq_char = "T"
    if base_freq_char == "Y":
        base_freq_char = "A"
    match = re.match(r"(\d+)", str(freq_str))
    multiplier = int(match.group(1)) if match else 1
    return base_freq_char, multiplier


def _candidate_periods(freq_str: str, total_len: int):
    if not freq_str or total_len <= 0:
        return []
    base_freq_char, multiplier = _split_freq(freq_str)
    suggestions = POSSIBLE_SEASONALITIES.get(base_freq_char)
    if not suggestions:
        return []
    periods = []
    for base_period_steps in suggestions:
        if multiplier <= 0:
            continue
        period = int(base_period_steps // multiplier)
        if period > 0:
            periods.append(period)
    return periods


def _max_pred_cycles_for_period(freq_str: str, period: int) -> int:
    base_freq_char, _ = _split_freq(freq_str)
    max_horizon_map = {
        "S": 900,
        "T": 720,
        "H": 720,
        "D": 60,
        "B": 40,
        "W": 26,
        "M": 36,
        "Q": 16,
    }
    max_horizon = max_horizon_map.get(base_freq_char, 1)
    return max(1, int(math.ceil(max_horizon / max(1, period))))


def build_dataset_names(gift_eval_root: Path):
    names = []
    for dataset_dir in gift_eval_root.iterdir():
        if dataset_dir.name.startswith("."):
            continue
        if dataset_dir.is_dir():
            freq_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if freq_dirs:
                for freq_dir in freq_dirs:
                    names.append(f"{dataset_dir.name}/{freq_dir.name}")
            else:
                names.append(dataset_dir.name)
    return sorted(names)


def _normalize_dataset_key(name: str) -> str:
    return str(name).strip().lower()


def load_allowed_terms_from_csv(csv_path: str):
    if not csv_path:
        return None
    allowed = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row.get("dataset")
            if not dataset:
                continue
            parts = dataset.strip().split("/")
            if len(parts) < 2:
                continue
            term = parts[-1]
            if len(parts) >= 3:
                base_name = "/".join(parts[:-2])
                base_name = DATASET_NAME_ALIASES.get(base_name, base_name)
                freq = parts[-2]
                ds_name = f"{base_name}/{freq}"
            else:
                freq = None
                ds_name = "/".join(parts[:-1])
                ds_name = DATASET_NAME_ALIASES.get(ds_name, ds_name)
            key = _normalize_dataset_key(ds_name)
            entry = allowed.setdefault(key, {"name": ds_name, "terms": set()})
            entry["terms"].add(term)
            if freq:
                freqs_by_term = entry.setdefault("freqs_by_term", {})
                freqs_by_term.setdefault(term, set()).add(freq)
    return allowed


def resolve_dataset_storage_info(ds_name: str, gift_eval_root: Path) -> tuple[str | None, str, str | None]:
    if "/" in ds_name:
        dataset_name, freq_name = ds_name.split("/", 1)
    else:
        dataset_name, freq_name = ds_name, None
    direct = gift_eval_root / ds_name
    if direct.exists():
        return ds_name, dataset_name, freq_name
    if "/" in ds_name:
        base = ds_name.rsplit("/", 1)[0]
        if (gift_eval_root / base).exists():
            return base, dataset_name, freq_name
    if (gift_eval_root / ds_name).exists():
        return ds_name, dataset_name, freq_name
    return None, dataset_name, freq_name


def _infer_single_freq_from_allowed(ds_name: str, term: str, allowed_terms_map: dict | None):
    if not allowed_terms_map:
        return None
    key = _normalize_dataset_key(ds_name)
    entry = allowed_terms_map.get(key)
    if not entry:
        return None
    freqs_by_term = entry.get("freqs_by_term", {})
    freqs = freqs_by_term.get(term)
    if not freqs or len(freqs) != 1:
        return None
    return next(iter(freqs))


def _as_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _missing_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isnan(arr).sum() / arr.size)


def _interpolate_nan_2d(arr: np.ndarray) -> np.ndarray:
    filled = np.asarray(arr, dtype=np.float32).copy()
    for i in range(filled.shape[0]):
        series = pd.Series(filled[i])
        filled[i] = series.interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    return filled


def show_image(image: torch.Tensor, nvars: int, color_list, save_path: Path) -> None:
    imagenet_mean = np.array([0.5, 0.5, 0.5])
    imagenet_std = np.array([0.5, 0.5, 0.5])
    cur_image = torch.zeros_like(image).cpu()
    height_per_var = image.shape[0] // max(nvars, 1)
    for i in range(nvars):
        cur_color = color_list[i]
        cur_image[i * height_per_var : (i + 1) * height_per_var, :, cur_color] = (
            image[i * height_per_var : (i + 1) * height_per_var, :, cur_color].cpu()
            * imagenet_std[cur_color]
            + imagenet_mean[cur_color]
        ) * 255
    cur_image = torch.clip(cur_image, 0, 255).to(torch.uint8).numpy()
    Image.fromarray(cur_image).save(save_path)


def _to_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def build_imputation_instruction_and_thinking(meta):
    total_len = int(meta.get("total_len", meta.get("total_len_real", 0)))
    periodicity = int(meta.get("periodicity", 1))
    image_size = int(meta.get("image_size", 0))
    n_vars = int(meta.get("nvars", 1))
    image_size_per_var = int(meta.get("image_size_per_var", image_size // max(1, n_vars)))
    exact_width = int(meta.get("exact_width", image_size))
    pad_left = int(meta.get("pad_left", 0))
    pad_right = int(meta.get("pad_right", 0))
    mask_ranges = meta.get("mask_ranges", [])

    cycles_total = (pad_left + total_len + pad_right) // periodicity if periodicity > 0 else 0

    var_infos = []
    for var_idx in range(n_vars):
        rng = mask_ranges[var_idx] if var_idx < len(mask_ranges) else None
        if not rng or len(rng) < 2:
            continue
        start, end = int(rng[0]), int(rng[1])
        start_cycle = (start + pad_left) // periodicity + 1
        end_cycle = (end + pad_left - 1) // periodicity + 1

        y0 = var_idx * image_size_per_var
        y1 = y0 + image_size_per_var - 1

        if cycles_total > 0:
            x0 = int((start_cycle - 1) / cycles_total * exact_width)
            x1 = int(end_cycle / cycles_total * exact_width) - 1
        else:
            x0, x1 = 0, exact_width - 1

        var_infos.append(
            f"Var{var_idx+1}: missing cycles {start_cycle}-{end_cycle}, "
            f"bbox=[({x0}, {y0}), ({x1}, {y1})]"
        )

    var_info_text = " ; ".join(var_infos) if var_infos else "No masked regions detected."

    instruction = (
        f"<image>You are given a {image_size}x{image_size} image that encodes {n_vars} time-series variable(s) "
        f"as horizontal bands stacked from top to bottom. "
        f"Each series contains {cycles_total} cycles, "
        f"where each cycle has {periodicity} time steps (totaling {cycles_total}*{periodicity}={total_len} observations). "
        f"Some regions in these series are masked (shown as black areas) and need to be imputed. "
        f"Different series may have masked regions at different positions, "
        f"and the number of masked cycles can vary across series. "
        f"Based on the observable parts in the time-series image, please restore all the masked black regions for each series."
    )

    thinking = (
        f"This image contains {n_vars} independent time-series encoded as horizontal bands, "
        f"where brighter pixels indicate larger values and darker pixels indicate smaller values. "
        f"Each series spans {cycles_total} cycles with {periodicity} time steps per cycle. "
        f"Each variable band has height image_size_per_var = image_size / n_vars = {image_size}/{n_vars} = {image_size_per_var} pixels. "
        f"Cycle width is computed as exact_width divided by the total number of cycles. "
        f"Each cycle occupies approximately image_size/cycles_total = {exact_width}/{cycles_total} pixels in width. "
        f"Each series has its own masked black region(s). "
        f"Per-series missing-cycle and bounding-box summary: {var_info_text}. "
        f"First, analyze each series separately: identify its specific masked region and examine the observable "
        f"patterns before and after the gap (trend, seasonality, value range). "
        f"Then, impute each series independently using its own context, maintaining consistent brightness encoding. "
        f"Finally, output the complete full image with all series restored."
    )

    return instruction, thinking


def _safe_test_data_iter(ds, ds_name, term):
    if getattr(ds, "windows", 0) <= 0 or ds.hf_dataset.num_rows <= 0:
        return None
    test_data = ds.test_data
    try:
        it = iter(test_data)
        first = next(it)
    except StopIteration:
        return None
    except AssertionError:
        print(f"[skip] {ds_name} term={term}: series too short for test split")
        return None
    return itertools.chain([first], it)


def _process_window(
    output_root,
    ds_name,
    output_dataset_name,
    output_freq_name,
    term,
    idx,
    test_input,
    test_label,
    freq_raw,
    period,
    period_candidates,
    converter,
):
    freq = _normalize_freq_for_period(freq_raw)
    context_arr = _as_2d(test_input["target"])
    pred_arr = _as_2d(test_label["target"])
    context_len = context_arr.shape[-1]
    pred_len = pred_arr.shape[-1]
    if context_len <= 0 or pred_len <= 0:
        return None

    full_arr = np.concatenate([context_arr, pred_arr], axis=1)
    total_len = full_arr.shape[1]

    picked = None
    for period_try in period_candidates:
        if total_len < 3 * period_try:
            continue
        pred_cycles = max(1, int(math.ceil(pred_len / period_try)))
        pred_len_real = pred_cycles * period_try
        context_len_real = 2 * pred_len_real
        total_len_real = context_len_real + pred_len_real
        if total_len < total_len_real:
            continue

        valid = ~np.isnan(full_arr).any(axis=0)
        start_idx = None
        i = 0
        while i < total_len:
            if not valid[i]:
                i += 1
                continue
            j = i
            while j < total_len and valid[j]:
                j += 1
            run_len = j - i
            if run_len >= total_len_real:
                start_idx = i
                break
            i = j
        if start_idx is None:
            continue

        picked = (period_try, pred_cycles, pred_len_real, context_len_real, total_len_real, start_idx)
        break

    if picked is None:
        return None

    period, pred_cycles, pred_len_real, context_len_real, total_len_real, start_idx = picked
    series_arr = full_arr[:, start_idx : start_idx + total_len_real]
    context_arr = series_arr[:, :context_len_real]
    pred_arr = series_arr[:, context_len_real:]

    nvars_total = series_arr.shape[0]
    var_indices = list(range(nvars_total))
    chunks = [
        var_indices[i : i + MAX_VARS_PER_IMAGE]
        for i in range(0, nvars_total, MAX_VARS_PER_IMAGE)
    ]

    rows = []
    part_count = len(chunks)
    for part_idx, chunk_indices in enumerate(chunks):
        series_slice = series_arr[chunk_indices]
        context_slice = context_arr[chunk_indices]
        pred_slice = pred_arr[chunk_indices]

        converted = converter.convert(
            series_slice.T,
            freq=freq,
            period=period,
            mask_ratio=(0.05, 0.5),
            mask_prob=1.0,
            mask_mode="contiguous",
            apply_mask=True,
        )

        dataset_name = output_dataset_name
        freq_name = output_freq_name or "unknown"
        sample_suffix = f"_part{part_idx:02d}" if part_count > 1 else ""
        sample_dir = (
            output_root
            / dataset_name
            / freq_name
            / term
            / f"sample_{idx:06d}{sample_suffix}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        image_tensor = torch.as_tensor(converted.image)
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.permute(1, 2, 0)

        color_list = converted.metadata.get("color_list")
        nvars = int(converted.metadata.get("nvars", len(chunk_indices)))
        if color_list is None:
            color_list = [0] * nvars
        else:
            color_list = list(np.asarray(color_list, dtype=int))

        show_image(image_tensor, nvars, color_list, sample_dir / "image_mask.png")

        meta = dict(converted.metadata)
        meta["freq"] = freq
        meta["total_len"] = int(total_len_real)
        meta["generated_period"] = period
        meta["period_candidates"] = period_candidates
        meta["source_dataset"] = ds_name
        meta["source_index"] = idx
        meta["selection_type"] = "imputation_benchmark"
        meta["pred_cycles"] = pred_cycles
        meta["pred_len_real"] = pred_len_real
        meta["context_len_real"] = context_len_real
        meta["nvars_total"] = nvars_total
        meta["part_index"] = part_idx
        meta["part_count"] = part_count
        meta["var_indices"] = list(chunk_indices)

        instruction, thinking = build_imputation_instruction_and_thinking(meta)
        meta["instruction"] = instruction
        meta["thinking"] = thinking
        (sample_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=_to_json_safe),
            encoding="utf-8",
        )

        full = converter.convert_full_with_stats(
            series_slice.T,
            freq=freq,
            period=period,
            stats_metadata=meta,
        )
        full_tensor = torch.as_tensor(full.image)
        if full_tensor.dim() == 4:
            full_tensor = full_tensor[0]
        full_tensor = full_tensor.permute(1, 2, 0)
        show_image(full_tensor, nvars, color_list, sample_dir / "image_full.png")

        np.savez(
            sample_dir / "series.npz",
            context=context_slice.T,
            pred=pred_slice.T,
            series=series_slice.T,
        )

        rows.append(
            {
                "dataset": f"{ds_name}/{term}",
                "instruction": instruction,
                "thinking": thinking,
                "source_image": str(sample_dir / "image_mask.png"),
                "target_image": str(sample_dir / "image_full.png"),
                "metadata": str(sample_dir / "metadata.json"),
            }
        )

    return rows


def _build_converter(image_size, patch_size, norm_const, align_const):
    return VisionTSImputationConverter(
        image_size=image_size,
        patch_size=patch_size,
        norm_const=norm_const,
        align_const=align_const,
        color=True,
        seed=None,
        mask_ratio=(0.05, 0.5),
        mask_prob=1.0,
        mask_mode="contiguous",
    )


def _process_sample_items(items, output_root, converter):
    rows = []
    for item in items:
        row_list = _process_window(
            output_root,
            item["ds_name"],
            item["output_dataset_name"],
            item["output_freq_name"],
            item["term"],
            item["idx"],
            item["test_input"],
            item["test_label"],
            item["freq_raw"],
            item["period"],
            item["period_candidates"],
            converter,
        )
        if row_list:
            rows.extend(row_list)
    return rows


def _write_jsonl(jsonl_path: Path, rows):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {jsonl_path}")


def generate_gifteval_samples_global(
    gift_eval_root: str,
    output_root: str,
    terms,
    image_size: int = 896,
    patch_size: int = 16,
    norm_const: float = 0.4,
    align_const: float = 1.0,
    max_total_samples: int = 500,
    max_samples_per_term: int = None,
    max_samples_per_dataset: int = None,
    samples_per_dataset_term: int = None,
    seed: int = 42,
    allowed_terms_map: dict | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    gift_eval_root = Path(gift_eval_root)
    os.environ["GIFT_EVAL"] = str(gift_eval_root.resolve())
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if allowed_terms_map is None:
        dataset_names = build_dataset_names(gift_eval_root)
    else:
        dataset_names = [entry["name"] for entry in allowed_terms_map.values()]

    converter = _build_converter(image_size, patch_size, norm_const, align_const)
    jsonl_path = output_root / f"gift_eval_imputation_samples_all_{max_total_samples}.jsonl"

    reservoir = []
    seen = 0
    term_reservoirs = {term: [] for term in terms} if max_samples_per_term else None
    term_seen = {term: 0 for term in terms} if max_samples_per_term else None

    dataset_term_reservoirs: dict = {}

    for ds_name in dataset_names:
        allowed_for_ds = None
        if allowed_terms_map is not None:
            key = _normalize_dataset_key(ds_name)
            entry = allowed_terms_map.get(key)
            allowed_for_ds = entry["terms"] if entry is not None else set()
            if not allowed_for_ds:
                continue
        for term in terms:
            if allowed_for_ds is not None and term not in allowed_for_ds:
                continue
            storage_name, output_dataset_name, output_freq_name = resolve_dataset_storage_info(
                ds_name, gift_eval_root
            )
            if storage_name is None:
                print(f"[skip] {ds_name} term={term}: dataset path not found")
                continue
            if output_freq_name is None:
                inferred = _infer_single_freq_from_allowed(
                    output_dataset_name, term, allowed_terms_map
                )
                if inferred:
                    output_freq_name = inferred
            ds = Dataset(name=storage_name, term=term)
            freq_raw = ds.freq
            test_iter = _safe_test_data_iter(ds, ds_name, term)
            if test_iter is None:
                continue

            try:
                for idx, (test_input, test_label) in enumerate(test_iter):
                    context_arr = _as_2d(test_input["target"])
                    pred_arr = _as_2d(test_label["target"])
                    total_len = context_arr.shape[-1] + pred_arr.shape[-1]
                    period_candidates = _candidate_periods(
                        _normalize_freq_for_period(freq_raw), total_len
                    )
                    if not period_candidates:
                        continue
                    period = random.choice(period_candidates)

                    item = {
                        "ds_name": ds_name,
                        "term": term,
                        "idx": idx,
                        "test_input": test_input,
                        "test_label": test_label,
                        "freq_raw": freq_raw,
                        "period": period,
                        "period_candidates": period_candidates,
                        "output_dataset_name": output_dataset_name,
                        "output_freq_name": output_freq_name,
                    }

                    if samples_per_dataset_term:
                        if ds_name not in dataset_term_reservoirs:
                            dataset_term_reservoirs[ds_name] = {}
                        if term not in dataset_term_reservoirs[ds_name]:
                            dataset_term_reservoirs[ds_name][term] = {
                                "items": [],
                                "seen": 0,
                            }
                        term_bucket = dataset_term_reservoirs[ds_name][term]
                        term_bucket["seen"] += 1
                        if len(term_bucket["items"]) < samples_per_dataset_term:
                            term_bucket["items"].append(item)
                        else:
                            j = random.randint(0, term_bucket["seen"] - 1)
                            if j < samples_per_dataset_term:
                                term_bucket["items"][j] = item
                    elif max_samples_per_dataset:
                        if ds_name not in dataset_term_reservoirs:
                            dataset_term_reservoirs[ds_name] = {}
                        if term not in dataset_term_reservoirs[ds_name]:
                            dataset_term_reservoirs[ds_name][term] = {
                                "items": [],
                                "seen": 0,
                            }
                        term_bucket = dataset_term_reservoirs[ds_name][term]
                        term_bucket["seen"] += 1
                        if len(term_bucket["items"]) < max_samples_per_dataset:
                            term_bucket["items"].append(item)
                        else:
                            j = random.randint(0, term_bucket["seen"] - 1)
                            if j < max_samples_per_dataset:
                                term_bucket["items"][j] = item
                    elif max_samples_per_term:
                        term_seen[term] += 1
                        if len(term_reservoirs[term]) < max_samples_per_term:
                            term_reservoirs[term].append(item)
                        else:
                            j = random.randint(0, term_seen[term] - 1)
                            if j < max_samples_per_term:
                                term_reservoirs[term][j] = item
                    else:
                        seen += 1
                        if len(reservoir) < max_total_samples:
                            reservoir.append(item)
                        else:
                            j = random.randint(0, seen - 1)
                            if j < max_total_samples:
                                reservoir[j] = item
            except AssertionError:
                print(f"[skip] {ds_name} term={term}: series too short for test split")
                continue

    if samples_per_dataset_term:
        selected_items = [
            item
            for term_buckets in dataset_term_reservoirs.values()
            for bucket in term_buckets.values()
            for item in bucket["items"]
        ]
        rows = _process_sample_items(selected_items, output_root, converter)
        jsonl_path = (
            output_root
            / f"gift_eval_imputation_samples_per_dataset_term_{samples_per_dataset_term}.jsonl"
        )
        _write_jsonl(jsonl_path, rows)
    elif max_samples_per_dataset:
        selected_items = []
        for ds_name, term_buckets in dataset_term_reservoirs.items():
            available_terms = [
                term for term, bucket in term_buckets.items() if bucket["items"]
            ]
            if not available_terms:
                continue
            base = max_samples_per_dataset // len(available_terms)
            random.shuffle(available_terms)
            counts = {term: 0 for term in available_terms}
            for term in available_terms:
                counts[term] = min(base, len(term_buckets[term]["items"]))
            remaining = max_samples_per_dataset - sum(counts.values())
            if remaining > 0:
                extra_terms = [
                    term
                    for term in available_terms
                    if len(term_buckets[term]["items"]) > counts[term]
                ]
                random.shuffle(extra_terms)
                while remaining > 0 and extra_terms:
                    term = extra_terms.pop(0)
                    counts[term] += 1
                    remaining -= 1
                    if len(term_buckets[term]["items"]) > counts[term]:
                        extra_terms.append(term)

            for term, count in counts.items():
                if count <= 0:
                    continue
                items = term_buckets[term]["items"]
                sampled_items = (
                    random.sample(items, count) if len(items) > count else items
                )
                selected_items.extend(sampled_items)

        rows = _process_sample_items(selected_items, output_root, converter)
        jsonl_path = output_root / f"gift_eval_imputation_samples_per_dataset_{max_samples_per_dataset}.jsonl"
        _write_jsonl(jsonl_path, rows)
    elif max_samples_per_term:
        for term in terms:
            rows = _process_sample_items(term_reservoirs[term], output_root, converter)
            jsonl_path = output_root / f"gift_eval_imputation_samples_{term}_{max_samples_per_term}.jsonl"
            _write_jsonl(jsonl_path, rows)
    else:
        rows = _process_sample_items(reservoir, output_root, converter)
        _write_jsonl(jsonl_path, rows)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gift-eval-root",
        default=str(DEFAULT_GIFT_EVAL_DATA_ROOT),
        help="Gift-Eval benchmark root directory",
    )
    parser.add_argument("--output-root", required=True, help="Output directory")
    parser.add_argument(
        "--term",
        nargs="+",
        default=["short", "medium", "long"],
        choices=["short", "medium", "long"],
        help="One or more terms to run.",
    )
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--norm-const", type=float, default=0.4)
    parser.add_argument("--align-const", type=float, default=1.0)
    parser.add_argument("--max-samples-per-dataset", type=int, default=None)
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=500,
        help="Global reservoir size used when no more specific sampling option is set.",
    )
    parser.add_argument("--max-samples-per-term", type=int, default=None)
    parser.add_argument("--samples-per-dataset-term", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allowed-datasets-csv",
        type=str,
        default=None,
        help="CSV path. Only dataset/term pairs in the CSV 'dataset' column are sampled.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    allowed_terms_map = load_allowed_terms_from_csv(args.allowed_datasets_csv)
    generate_gifteval_samples_global(
        gift_eval_root=args.gift_eval_root,
        output_root=args.output_root,
        terms=args.term,
        image_size=args.image_size,
        patch_size=args.patch_size,
        norm_const=args.norm_const,
        align_const=args.align_const,
        max_total_samples=args.max_total_samples,
        max_samples_per_term=args.max_samples_per_term,
        max_samples_per_dataset=args.max_samples_per_dataset,
        samples_per_dataset_term=args.samples_per_dataset_term,
        seed=args.seed,
        allowed_terms_map=allowed_terms_map,
    )
