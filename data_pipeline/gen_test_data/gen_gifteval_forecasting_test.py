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
import einops
from PIL import Image
from timm.models.vision_transformer import PatchEmbed

from gift_eval.data import Dataset
from data_pipeline.bi_tsi.forecasting_adapter import VisionTSImageConverter

POSSIBLE_SEASONALITIES = {
    "S": [3600],         # Second: Day
    "T": [1440],         # Minute: Day
    "H": [24, 168],      # Hour: Day, Week
    "D": [7],            # Day: Week
    "B": [5],            # Business Day: Week
    "W": [52],           # Week: Year
    "M": [12],           # Month: Year
    "Q": [4],            # Quarter: Year
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

def analyze_capacity_constraints(
    context_len,
    pred_len,
    image_size,
    patch_size,
    period,
    align_const,
    nvars=None,
    verbose=False,
):
    total_len = context_len + pred_len
    if total_len <= 0 or period <= 0:
        return False

    height_ok = True
    height_per_var = None
    if nvars is not None and nvars > 0:
        height_per_var = image_size / nvars
        if height_per_var < period:
            height_ok = False

    input_ratio = context_len / total_len
    num_total_patch = image_size // patch_size

    num_patch_in = int(input_ratio * num_total_patch * align_const)
    if num_patch_in < 1:
        num_patch_in = 1
    num_patch_out = num_total_patch - num_patch_in
    if num_patch_out < 1:
        num_patch_out = 1

    global_capacity = image_size * period
    capacity_ctx = num_patch_in * patch_size * period
    capacity_pred = num_patch_out * patch_size * period

    is_global_safe = total_len <= global_capacity
    is_ctx_safe = context_len <= capacity_ctx
    is_pred_safe = pred_len <= capacity_pred
    is_all_safe = is_global_safe and is_ctx_safe and is_pred_safe and height_ok

    if verbose:
        status = "PASS" if is_all_safe else "FAIL"
        print(f"[capacity] {status} total={total_len} ctx={context_len} pred={pred_len}")
        print(f"  global {total_len} <= {global_capacity}")
        print(f"  ctx    {context_len} <= {capacity_ctx}")
        print(f"  pred   {pred_len} <= {capacity_pred}")
        if nvars is not None and nvars > 0:
            print(f"  height_per_var={height_per_var} nvars={nvars} patch={patch_size}")

    return is_all_safe

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
                ds_name = f"{base_name}/{parts[-2]}"
            else:
                ds_name = "/".join(parts[:-1])
                ds_name = DATASET_NAME_ALIASES.get(ds_name, ds_name)
            key = _normalize_dataset_key(ds_name)
            entry = allowed.setdefault(key, {"name": ds_name, "terms": set()})
            entry["terms"].add(term)
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

def _as_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr

def _pad_series_to_length(series: torch.Tensor, target_len: int, pad_left: bool) -> torch.Tensor:
    if series.shape[0] >= target_len:
        return series
    pad_len = target_len - series.shape[0]
    if pad_left:
        pad_value = series[:1].repeat(pad_len, 1)
        return torch.cat([pad_value, series], dim=0)
    pad_value = series[-1:].repeat(pad_len, 1)
    return torch.cat([series, pad_value], dim=0)

def _missing_ratio(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.isnan(arr).sum() / arr.size)

def _interpolate_nan_2d(arr: np.ndarray) -> np.ndarray:
    filled = np.asarray(arr, dtype=np.float32).copy()
    for i in range(filled.shape[0]):
        series = pd.Series(filled[i])
        filled[i] = (
            series.interpolate(limit_direction="both")
            .to_numpy(dtype=np.float32)
        )
    return filled

def random_masking(x, mask_ratio, noise):
    n_batch, n_tokens, n_dim = x.shape
    len_keep = int(round(n_tokens * (1 - mask_ratio)))
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1, 1, n_dim))
    mask = torch.ones([n_batch, n_tokens], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, 1, ids_restore)
    return x_masked, mask, ids_restore

def unpatchify(x, patch_size=16):
    batch, length, dim = x.shape
    p = patch_size
    h = w = int(length**0.5)
    x = x.reshape(batch, h, w, p, p, 3)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(batch, 3, h * p, w * p)

def show_image(image, cur_nvars, cur_color_list, save_path):
    imagenet_mean = np.array([0.5, 0.5, 0.5])
    imagenet_std = np.array([0.5, 0.5, 0.5])
    cur_image = torch.zeros_like(image).cpu()
    height_per_var = image.shape[0] // max(cur_nvars, 1)

    for i in range(cur_nvars):
        cur_color = cur_color_list[i]
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

def build_instruction_and_thinking(meta):
    context_len = int(meta.get("context_len", 0))
    pred_len = int(meta.get("pred_len", 0))
    periodicity = int(meta.get("periodicity", 1))
    image_size = int(meta.get("image_size", 0))
    n_vars = int(meta.get("nvars", 1))
    image_size_per_var = int(meta.get("image_size_per_var", image_size // max(1, n_vars)))
    exact_width = int(meta.get("exact_width", image_size))

    context_cycles = context_len // periodicity if periodicity > 0 else 0
    pred_cycles = pred_len // periodicity if periodicity > 0 else 0
    cycles_total = context_cycles + pred_cycles

    pred_bbox_lines = []
    pred_start_cycle = context_cycles + 1
    pred_end_cycle = context_cycles + pred_cycles

    for var_idx in range(n_vars):
        y0 = var_idx * image_size_per_var
        y1 = y0 + image_size_per_var - 1

        if cycles_total > 0:
            x0 = int((pred_start_cycle - 1) / cycles_total * exact_width)
            x1 = int(pred_end_cycle / cycles_total * exact_width) - 1
        else:
            x0, x1 = 0, exact_width - 1

        pred_bbox_lines.append(
            f"Var{var_idx+1}: pred bbox=[({x0}, {y0}), ({x1}, {y1})]."
        )
    pred_bbox_text = " ".join(pred_bbox_lines)

    instruction = (
        f"<image>You are given a {image_size}x{image_size} image that encodes {n_vars} time-series variable(s) "
        f"as horizontal bands stacked from top to bottom. "
        f"Each series contains {context_cycles} cycles, "
        f"where each cycle has {periodicity} time steps (totaling {context_cycles}*{periodicity}={context_len} observations). "
        f"The right side is masked and needs to be predicted for the next {pred_cycles} cycles "
        f"(totaling {pred_cycles}*{periodicity}={pred_len} observations). "
        f"Based on the observable parts in the time-series image, please restore the masked right side."
    )

    thinking = (
        f"This image contains {n_vars} independent time-series encoded as horizontal bands, "
        f"where brighter pixels indicate larger values and darker pixels indicate smaller values. "
        f"Each series spans {cycles_total} cycles with {periodicity} time steps per cycle. "
        f"Each variable band has height image_size_per_var = image_size / n_vars = {image_size}/{n_vars} = {image_size_per_var} pixels. "
        f"Cycle width is computed as exact_width divided by the total number of cycles. "
        f"Each cycle occupies approximately image_size/cycles_total = {exact_width}/{cycles_total} pixels in width. "
        f"Prediction-region bounding-box per variable: {pred_bbox_text} "
        f"First, analyze each variable's historical pattern in the left {context_len} time steps "
        f"({context_cycles} cycles): identify trend, seasonality, value range, and anomalies. "
        f"Then, for each variable independently, extrapolate the future {pred_len} time steps ({pred_cycles} cycles) "
        f"by continuing its specific patterns and maintaining consistent brightness encoding. "
        f"Finally, output the complete full image with all series restored."
    )

    return instruction, thinking

def _select_period(freq, context_len, pred_len, image_size, patch_size, align_const, nvars):
    total_len = context_len + pred_len
    period_candidates = _candidate_periods(freq, total_len)
    if not period_candidates:
        return None, None
    period = random.choice(period_candidates)
    if not analyze_capacity_constraints(
        context_len,
        pred_len,
        image_size,
        patch_size,
        period,
        align_const,
        nvars=nvars,
        verbose=False,
    ):
        return None, None
    return period, period_candidates

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
    patch_embed_layer,
    image_size,
    patch_size,
    align_const,
):
    freq = _normalize_freq_for_period(freq_raw)
    context_arr = _as_2d(test_input["target"])
    pred_arr = _as_2d(test_label["target"])
    context_len = context_arr.shape[-1]
    pred_len = pred_arr.shape[-1]
    if context_len <= 0 or pred_len <= 0:
        print(
            f"[skip] {ds_name} term={term} idx={idx}: "
            f"empty context/pred (context_len={context_len}, pred_len={pred_len})"
        )
        return None

    pred_len_rounded = max(int(period), int(math.ceil(pred_len / period) * period))
    pred_cycles = pred_len_rounded // period
    max_context_cycles = context_len // period
    if max_context_cycles < pred_cycles:
        print(
            f"[skip] {ds_name} term={term} idx={idx}: "
            f"context shorter than pred cycles (context_cycles={max_context_cycles}, pred_cycles={pred_cycles})"
        )
        return None
    context_cycles = 2 * pred_cycles if max_context_cycles >= 2 * pred_cycles else pred_cycles
    context_len_real = context_cycles * period
    pred_len_real = pred_len_rounded

    context_arr = context_arr[:, -context_len_real:]
    pred_arr = pred_arr[:, :pred_len]

    missing_ratio = _missing_ratio(context_arr)
    if missing_ratio > 0.05:
        print(
            f"[skip] {ds_name} term={term} idx={idx}: "
            f"context missing ratio {missing_ratio:.4f} > 0.05"
        )
        return None
    if missing_ratio > 0:
        context_arr = _interpolate_nan_2d(context_arr)

    if context_len_real < period:
        print(
            f"[skip] {ds_name} term={term} idx={idx}: "
            f"context shorter than one period (context_len={context_len_real}, period={period})"
        )
        return None
    pred_len_padded = pred_len_rounded

    nvars_total = context_arr.shape[0]
    var_indices = list(range(nvars_total))
    chunks = [
        var_indices[i : i + MAX_VARS_PER_IMAGE]
        for i in range(0, nvars_total, MAX_VARS_PER_IMAGE)
    ]

    def _build_row_for_chunk(chunk_indices, part_idx, part_count):
        context_slice = context_arr[chunk_indices]
        pred_slice = pred_arr[chunk_indices]

        # convert expects [T, nvars]
        context_series = torch.from_numpy(context_slice.T).float()
        pred_series_raw = torch.from_numpy(pred_slice.T).float()
        pred_series = pred_series_raw
        nvars = context_series.shape[1]
        if not analyze_capacity_constraints(
            context_len_real,
            pred_len_padded,
            image_size,
            patch_size,
            period,
            align_const,
            nvars=nvars,
            verbose=False,
        ):
            print(
                f"[skip] {ds_name} term={term} idx={idx}: "
                f"capacity fail (context_len={context_len_real}, pred_len={pred_len_padded}, period={period})"
            )
            return None

        pred_series = _pad_series_to_length(
            pred_series, pred_len_padded, pad_left=False
        )

        converted = converter.convert(
            context_len=context_len_real,
            pred_len=pred_len_padded,
            series=context_series,
            freq=freq,
            period=period,
        )

        full_series = torch.cat([context_series, pred_series], dim=0)
        full_series_raw = torch.cat([context_series, pred_series_raw], dim=0)
        full_image = converter.render_full_groundtruth_image(full_series, converted)

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

        # Mask the right-side prediction region.
        image_tensor = torch.as_tensor(converted.image).float()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        patches = patch_embed_layer(image_tensor)

        if "mask" in converted.metadata:
            noise = einops.repeat(
                torch.as_tensor(converted.metadata["mask"]),
                "1 l -> n l",
                n=1,
            )
        else:
            noise = torch.rand(1, patches.shape[1])

        mask_ratio = float(converted.metadata.get("mask_ratio", 0.0))
        _, mask, _ = random_masking(patches, mask_ratio, noise)
        mask_expand = mask.unsqueeze(-1).repeat(
            1, 1, patch_size * patch_size * 3
        )
        mask_img = unpatchify(mask_expand, patch_size)

        green_bg = -torch.ones_like(image_tensor) * 2
        image_vis = image_tensor * (1 - mask_img) + green_bg * mask_img
        image_vis_hwc = image_vis[0].permute(1, 2, 0)

        nvars = int(converted.metadata.get("nvars", 1))
        color_list = converted.metadata.get("color_list")
        if color_list is None:
            color_list = [0] * nvars
        else:
            color_list = list(np.asarray(color_list, dtype=int))

        show_image(
            image_vis_hwc,
            nvars,
            color_list,
            sample_dir / "image_mask.png",
        )

        full_tensor = torch.as_tensor(full_image).float()
        if full_tensor.dim() == 3:
            full_tensor = full_tensor.unsqueeze(0)
        full_hwc = full_tensor[0].permute(1, 2, 0)
        show_image(
            full_hwc,
            nvars,
            color_list,
            sample_dir / "image_full.png",
        )

        meta = dict(converted.metadata)
        meta["freq"] = freq
        meta["exact_width"] = image_size  # Keep consistent with show_edit_data.ipynb.
        meta["generated_period"] = period
        meta["period_candidates"] = period_candidates
        meta["source_dataset"] = ds_name
        meta["source_index"] = idx
        meta["selection_type"] = "benchmark"
        meta["pred_len_real"] = pred_len_real
        meta["pred_len_padded"] = pred_len_padded
        meta["context_len_real"] = context_len_real
        meta["context_len_padded"] = context_len_real
        meta["nvars_total"] = nvars_total
        meta["part_index"] = part_idx
        meta["part_count"] = part_count
        meta["var_indices"] = list(chunk_indices)

        instruction, thinking = build_instruction_and_thinking(meta)
        meta["instruction"] = instruction
        meta["thinking"] = thinking
        (sample_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False, default=_to_json_safe),
            encoding="utf-8",
        )
        np.savez(
            sample_dir / "series.npz",
            context=context_series.numpy(),
            pred=pred_series_raw.numpy(),
            series=full_series_raw.numpy(),
        )

        row = {
            "dataset": f"{ds_name}/{term}",
            "instruction": instruction,
            "thinking": thinking,
            "source_image": str(sample_dir / "image_mask.png"),
            "target_image": str(sample_dir / "image_full.png"),
            "metadata": str(sample_dir / "metadata.json"),
        }
        return row

    rows = []
    part_count = len(chunks)
    for part_idx, chunk_indices in enumerate(chunks):
        row = _build_row_for_chunk(chunk_indices, part_idx, part_count)
        if row is not None:
            rows.append(row)
    return rows

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

def _process_sample_items(
    items,
    output_root,
    converter,
    patch_embed_layer,
    image_size,
    patch_size,
    align_const,
):
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
            patch_embed_layer,
            image_size,
            patch_size,
            align_const,
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

    converter = VisionTSImageConverter(
        image_size=image_size,
        patch_size=patch_size,
        num_patch_input=None,
        norm_const=norm_const,
        align_const=align_const,
        color=True,
    )
    patch_embed_layer = PatchEmbed(image_size, patch_size, 3, embed_dim=1024)

    jsonl_path = output_root / f"gift_eval_samples_all_{max_total_samples}.jsonl"
    reservoir = []
    seen = 0

    term_reservoirs = {term: [] for term in terms} if max_samples_per_term else None
    term_seen = {term: 0 for term in terms} if max_samples_per_term else None

    dataset_term_reservoirs = {}

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
            ds = Dataset(name=storage_name, term=term)
            freq_raw = ds.freq
            test_iter = _safe_test_data_iter(ds, ds_name, term)
            if test_iter is None:
                continue

            try:
                for idx, (test_input, test_label) in enumerate(test_iter):
                    context_arr = _as_2d(test_input["target"])
                    pred_arr = _as_2d(test_label["target"])
                    context_len = context_arr.shape[-1]
                    pred_len = pred_arr.shape[-1]
                    nvars = context_arr.shape[0]

                    period, period_candidates = _select_period(
                        _normalize_freq_for_period(freq_raw),
                        context_len,
                        pred_len,
                        image_size,
                        patch_size,
                        align_const,
                        nvars,
                    )
                    if period is None:
                        continue

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
        rows = _process_sample_items(
            selected_items,
            output_root,
            converter,
            patch_embed_layer,
            image_size,
            patch_size,
            align_const,
        )
        dataset_jsonl_path = output_root / f"gift_eval_samples_per_dataset_term_{samples_per_dataset_term}.jsonl"
        _write_jsonl(dataset_jsonl_path, rows)
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

        rows = _process_sample_items(
            selected_items,
            output_root,
            converter,
            patch_embed_layer,
            image_size,
            patch_size,
            align_const,
        )
        dataset_jsonl_path = output_root / f"gift_eval_samples_per_dataset_{max_samples_per_dataset}.jsonl"
        _write_jsonl(dataset_jsonl_path, rows)
    elif max_samples_per_term:
        for term in terms:
            rows = _process_sample_items(
                term_reservoirs[term],
                output_root,
                converter,
                patch_embed_layer,
                image_size,
                patch_size,
                align_const,
            )
            term_jsonl_path = output_root / f"gift_eval_samples_{term}_{max_samples_per_term}.jsonl"
            _write_jsonl(term_jsonl_path, rows)
    else:
        rows = _process_sample_items(
            reservoir,
            output_root,
            converter,
            patch_embed_layer,
            image_size,
            patch_size,
            align_const,
        )
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
    parser.add_argument(
        "--samples-per-dataset-term",
        type=int,
        default=None,
        help="Sample N windows per dataset/term (e.g. bitbrains_fast_storage/5T/long).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allowed-datasets-csv",
        type=str,
        default=None,
        help="CSV path (e.g. chronos_bolt_base/all_results.csv). "
             "Only dataset/term pairs in the CSV 'dataset' column are sampled.",
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
