from __future__ import annotations

import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from torchvision import transforms
from torchvision.transforms import Resize



POSSIBLE_SEASONALITIES = {
    "S": [3600],
    "T": [1440, 10080],
    "H": [24, 168],
    "D": [7, 30, 365],
    "W": [52, 4],
    "M": [12, 6, 3],
    "B": [5],
    "Q": [4, 2],
}


def safe_resize(size: Union[int, Tuple[int, int]], interpolation: int) -> Resize:
    signature = inspect.signature(Resize)
    params = signature.parameters
    if "antialias" in params:
        return Resize(size, interpolation=interpolation, antialias=False)
    return Resize(size, interpolation=interpolation)


def norm_freq_str(freq_str: str) -> str:
    base_freq = freq_str.split("-")[0]
    if len(base_freq) >= 2 and base_freq.endswith("S"):
        return base_freq[:-1]
    return base_freq


def freq_to_seasonality_list(freq: str, mapping_dict=None) -> List[int]:
    if mapping_dict is None:
        mapping_dict = POSSIBLE_SEASONALITIES
    offset = pd.tseries.frequencies.to_offset(freq)
    base = mapping_dict.get(norm_freq_str(offset.name), [])
    seasonality_list = []
    for base_seasonality in base:
        seasonality, remainder = divmod(base_seasonality, offset.n)
        if not remainder:
            seasonality_list.append(seasonality)
    seasonality_list.append(1)
    return seasonality_list


# -----------------------------------------------------------------------------
#Imputation Converter / Reconstructor
# -----------------------------------------------------------------------------
@dataclass
class VisionTSImageSample:
    image: np.ndarray          # shape: (C, H, W)
    metadata: Dict[str, Any]
    normalized_context: Optional[np.ndarray] = None
    normalized_future: Optional[np.ndarray] = None


class VisionTSImputationConverter:
    """
    Convert a full sequence into a VisionTS image with random masking per var.
    Masking happens in normalized space, so statistics remain consistent.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_patch_input: int = 7,
        norm_const: float = 0.4,
        align_const: float = 1,
        padding_mode: str = "replicate",
        color: bool = True,
        seed: Optional[int] = None,
        mask_ratio: Union[float, Tuple[float, float], List[float]] = (0.05, 0.5),
        mask_prob: float = 0.5,
        mask_mode: str = "contiguous",
        mask_value: float = -1.0,
        store_mask: bool = False,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch_input = num_patch_input
        self.norm_const = norm_const
        self.padding_mode = padding_mode
        self.align_const = align_const
        self.color = color
        self.mask_ratio = mask_ratio
        self.mask_prob = mask_prob
        self.mask_mode = mask_mode
        self.mask_value = mask_value
        self.store_mask = store_mask


        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    def update_config(
        self,
        total_len: int,
        periodicity: int = 1,
        interpolation: str = "bilinear",
        padding_mode: str = "replicate",
    ) -> None:
        self.image_size_cfg = self.image_size
        self.patch_size_cfg = self.patch_size
        self.periodicity = periodicity
        self.padding_mode = padding_mode
        self.total_len = total_len

        self.num_patch = self.image_size_cfg // self.patch_size_cfg
        self.exact_width = self.num_patch * self.patch_size_cfg

        self.pad_left = 0
        if self.total_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.total_len % self.periodicity

        self.pad_right = 0
        self.full_padded_len = self.pad_left + self.total_len + self.pad_right

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]

        cycles = self.full_padded_len // periodicity
        self.scale_x = cycles / self.exact_width

    def convert(
        self,
        series: np.ndarray,
        freq: str = "H",
        period: Optional[int] = None,
        mask_ratio: Optional[Union[float, Tuple[float, float], List[float]]] = None,
        mask_prob: Optional[float] = None,
        mask_mode: Optional[str] = None,
        apply_mask: bool = True,
    ) -> VisionTSImageSample:
        arr = np.asarray(series, dtype=np.float32).transpose(1, 0)
        if arr.ndim == 1:
            arr = arr[None, :]

        nvars, total_len = arr.shape

        if not period:
            periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < total_len]
            if len(periodicity_list) >= 2:
                periodicity = random.choice(periodicity_list[:-1])
            else:
                periodicity = periodicity_list[-1] if periodicity_list else 1
        else:
            periodicity = period

        self.update_config(
            total_len=total_len,
            periodicity=periodicity,
            padding_mode=self.padding_mode,
        )

        entry = self._render_full_sequence(
            arr,
            freq=freq,
            mask_ratio=self.mask_ratio if mask_ratio is None else mask_ratio,
            mask_prob=self.mask_prob if mask_prob is None else mask_prob,
            mask_mode=self.mask_mode if mask_mode is None else mask_mode,
            apply_mask=apply_mask,
        )

        return VisionTSImageSample(
            image=entry["target"],
            metadata=entry["metadata"],
        )

    def convert_full_with_stats(
        self,
        series: np.ndarray,
        freq: str = "H",
        period: Optional[int] = None,
        stats_metadata: Optional[Dict[str, Any]] = None,
    ) -> VisionTSImageSample:
        if stats_metadata is None:
            raise ValueError("stats_metadata is required for convert_full_with_stats.")

        means = stats_metadata.get("means")
        stdev = stats_metadata.get("stdev")
        if means is None or stdev is None:
            raise ValueError("stats_metadata must include 'means' and 'stdev'.")

        arr = np.asarray(series, dtype=np.float32).transpose(1, 0)
        if arr.ndim == 1:
            arr = arr[None, :]

        nvars, _total_len = arr.shape
        if "periodicity" not in stats_metadata:
            raise ValueError("stats_metadata must include 'periodicity'.")

        entry = self._render_full_sequence_with_stats(
            arr,
            freq=freq,
            means=means,
            stdev=stdev,
            stats_metadata=stats_metadata,
        )

        return VisionTSImageSample(
            image=entry["target"],
            metadata=entry["metadata"],
        )

    def _resolve_mask_ratio(
        self,
        mask_ratio: Union[float, Tuple[float, float], List[float]],
    ) -> Tuple[float, float]:
        if isinstance(mask_ratio, (tuple, list, np.ndarray)):
            if len(mask_ratio) != 2:
                raise ValueError("mask_ratio range must have exactly 2 values.")
            min_ratio = float(mask_ratio[0])
            max_ratio = float(mask_ratio[1])
        else:
            min_ratio = float(mask_ratio)
            max_ratio = float(mask_ratio)
        if min_ratio > max_ratio:
            min_ratio, max_ratio = max_ratio, min_ratio

        min_ratio = min(max(min_ratio, 0.0), 1.0)
        max_ratio = min(max(max_ratio, 0.0), 1.0)
        return min_ratio, max_ratio

    def _build_mask(
        self,
        nvars: int,
        total_len: int,
        min_ratio: float,
        max_ratio: float,
        mask_prob: float,
        mask_mode: str,
        period: int,
    ) -> Tuple[Optional[np.ndarray], List[Optional[List[int]]], List[Optional[float]]]:
        if total_len <= 0 or max_ratio <= 0 or mask_prob <= 0:
            return None, [None] * nvars, [None] * nvars

        if mask_mode != "contiguous":
            raise ValueError(f"Unknown mask_mode '{mask_mode}'.")

        mask_prob = min(max(mask_prob, 0.0), 1.0)
        period = int(period) if period is not None else 1
        if period <= 0:
            period = 1
        periods_total = total_len // period if period > 0 else 0

        mask = np.zeros((nvars, total_len), dtype=np.uint8)
        mask_ranges: List[Optional[List[int]]] = []
        mask_ratios: List[Optional[float]] = []

        for var_idx in range(nvars):
            if random.random() > mask_prob:
                mask_ranges.append(None)
                mask_ratios.append(None)
                continue

            mask_ratio_var = random.uniform(min_ratio, max_ratio)
            if mask_ratio_var <= 0:
                mask_ranges.append(None)
                mask_ratios.append(0.0)
                continue

            target_len = max(1, int(round(total_len * mask_ratio_var)))

            if period > 1 and periods_total > 0:
                mask_periods = (target_len + period - 1) // period
                if mask_periods > periods_total:
                    mask_periods = periods_total
                mask_len = mask_periods * period
                if mask_len >= total_len:
                    start = 0
                    end = total_len
                else:
                    max_start_period = periods_total - mask_periods
                    start_period = random.randint(0, max_start_period)
                    start = start_period * period
                    end = start + mask_len
            else:
                mask_len = min(target_len, total_len)
                if mask_len >= total_len:
                    start = 0
                    end = total_len
                else:
                    start = random.randint(0, total_len - mask_len)
                    end = start + mask_len

            mask[var_idx, start:end] = 1
            mask_ranges.append([int(start), int(end)])
            mask_ratios.append(float(mask_ratio_var))

        return mask.astype(bool), mask_ranges, mask_ratios

    def _compute_stats(
        self,
        x: torch.Tensor,
        mask: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            median_val = x.median(dim=1, keepdim=True).values.detach()
            abs_deviation = (x - median_val).abs()
            mad_val = abs_deviation.median(dim=1, keepdim=True).values.detach()
            robust_scale = mad_val / 0.6745
            std_val = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-18
            ).detach()
        else:
            mask_tensor = torch.as_tensor(mask, device=x.device).bool()
            nvars = x.shape[2]
            medians = []
            mads = []
            stds = []
            for var_idx in range(nvars):
                series = x[0, :, var_idx]
                valid = series[~mask_tensor[var_idx]]
                if valid.numel() == 0:
                    valid = series
                median = valid.median()
                abs_deviation = (valid - median).abs()
                mad = abs_deviation.median()
                std = torch.sqrt(torch.var(valid, unbiased=False) + 1e-18)
                medians.append(median)
                mads.append(mad)
                stds.append(std)
            median_val = torch.stack(medians).view(1, 1, nvars).detach()
            mad_val = torch.stack(mads).view(1, 1, nvars).detach()
            std_val = torch.stack(stds).view(1, 1, nvars).detach()
            robust_scale = mad_val / 0.6745

        scale = 0.5 * robust_scale + 0.5 * std_val
        scale = torch.maximum(scale, torch.tensor(1e-18, device=x.device))
        return median_val, scale

    def _render_full_sequence(
        self,
        data: np.ndarray,
        freq: str,
        mask_ratio: Union[float, Tuple[float, float], List[float]],
        mask_prob: float,
        mask_mode: str,
        apply_mask: bool,
    ) -> Dict[str, Any]:
        nvars, total_len = data.shape
        periodicity = self.periodicity
        pad_left = self.pad_left
        pad_right = self.pad_right
        image_size_per_var = max(1, self.image_size // max(1, nvars))

        x = torch.from_numpy(data).float().transpose(0, 1).unsqueeze(0)

        min_ratio, max_ratio = self._resolve_mask_ratio(mask_ratio)
        mask, mask_ranges, mask_ratios = self._build_mask(
            nvars=nvars,
            total_len=total_len,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            mask_prob=mask_prob,
            mask_mode=mask_mode,
            period=periodicity,
        )

        means, stdev = self._compute_stats(x, mask)

        x_enc = (x - means) / stdev
        tanh_factor = 4.0
        x_enc = torch.tanh(x_enc / tanh_factor)

        x_enc = rearrange(x_enc, "b s n -> b n s")

        if apply_mask and mask is not None:
            mask_tensor = torch.as_tensor(mask, device=x_enc.device)
            x_enc = x_enc.masked_fill(mask_tensor.unsqueeze(0), self.mask_value)

        if pad_left > 0 or pad_right > 0:
            x_pad = F.pad(x_enc, (pad_left, pad_right), mode=self.padding_mode)
        else:
            x_pad = x_enc

        x_2d = rearrange(x_pad, "b n (c p) -> b n p c", p=periodicity)

        resizer = safe_resize(
            (image_size_per_var, self.exact_width),
            interpolation=self.interpolation,
        )
        x_resize = resizer(x_2d)

        x_resize = rearrange(x_resize, "b n h w -> b 1 (n h) w")

        pad_down = self.image_size - x_resize.shape[2]
        pad_right_img = self.image_size - x_resize.shape[3]

        if pad_down > 0 or pad_right_img > 0:
            x_resize = F.pad(x_resize, (0, pad_right_img, 0, pad_down))

        assert x_resize.shape[2] == self.image_size, f"Height mismatch: {x_resize.shape[2]}"
        assert x_resize.shape[3] == self.image_size, f"Width mismatch: {x_resize.shape[3]}"

        image_input = torch.zeros(
            (x_resize.shape[0], 3, x_resize.shape[2], x_resize.shape[3]),
            device=x_resize.device,
            dtype=x_resize.dtype,
        )

        color_list = None
        if color_list is None:
            color_list = [i % 3 for i in range(nvars)]

        for i in range(nvars):
            color = color_list[i]
            h_start = i * image_size_per_var
            h_end = (i + 1) * image_size_per_var
            image_input[:, color, h_start:h_end, :] = x_resize[:, 0, h_start:h_end, :]

        mask_ratio_actual = float(mask.mean()) if mask is not None else 0.0
        mask_payload = mask.astype(np.uint8) if (mask is not None and self.store_mask) else None

        metadata = {
            "mode": "imputation",
            "freq": freq,
            "total_len": int(self.total_len),
            "pad_left": int(pad_left),
            "pad_right": int(pad_right),
            "periodicity": int(periodicity),
            "scale_x": float(self.scale_x),
            "means": means.cpu().numpy()[0],
            "stdev": stdev.cpu().numpy()[0],
            "norm_const": float(self.norm_const),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "num_patch": int(self.num_patch),
            "exact_width": int(self.exact_width),
            "nvars": int(nvars),
            "pad_down": int(pad_down),
            "pad_right_img": int(pad_right_img),
            "color_list": np.array(color_list, dtype=int),
            "image_size_per_var": int(image_size_per_var),
            "padding_mode": self.padding_mode,
            "align_const": float(self.align_const),
            "color_mode": self.color,
            "mask_mode": mask_mode,
            "mask_ratio": float(max_ratio),
            "mask_prob": float(mask_prob),
            "mask_ratio_actual": mask_ratio_actual,
            "mask_value": float(self.mask_value),
            "mask_ranges": mask_ranges,
            "mask_applied": bool(apply_mask),
        }
        if min_ratio != max_ratio:
            metadata["mask_ratio_range"] = [float(min_ratio), float(max_ratio)]
            metadata["mask_ratio_per_var"] = mask_ratios

        if mask_payload is not None:
            metadata["mask"] = mask_payload

        return {
            "target": image_input.cpu(),
            "metadata": metadata,
        }

    def _render_full_sequence_with_stats(
        self,
        data: np.ndarray,
        freq: str,
        means: Union[np.ndarray, List[float], torch.Tensor],
        stdev: Union[np.ndarray, List[float], torch.Tensor],
        stats_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        nvars, total_len = data.shape
        if stats_metadata is None:
            raise ValueError("stats_metadata is required for _render_full_sequence_with_stats.")

        periodicity = int(stats_metadata["periodicity"])
        pad_left = int(stats_metadata.get("pad_left", 0))
        pad_right = int(stats_metadata.get("pad_right", 0))
        image_size = int(stats_metadata.get("image_size", self.image_size))
        exact_width = int(stats_metadata.get("exact_width", self.exact_width))
        image_size_per_var = int(
            stats_metadata.get("image_size_per_var", max(1, image_size // max(1, nvars)))
        )

        x = torch.from_numpy(data).float().transpose(0, 1).unsqueeze(0)
        device = x.device

        means_t = torch.as_tensor(means, device=device).float().reshape(1, 1, -1)
        stdev_t = torch.as_tensor(stdev, device=device).float().reshape(1, 1, -1)

        if means_t.shape[-1] != nvars:
            raise ValueError(f"means shape mismatch: {means_t.shape[-1]} vs nvars={nvars}")
        if stdev_t.shape[-1] != nvars:
            raise ValueError(f"stdev shape mismatch: {stdev_t.shape[-1]} vs nvars={nvars}")

        stdev_t = torch.maximum(stdev_t, torch.tensor(1e-18, device=device))

        x_enc = (x - means_t) / stdev_t
        tanh_factor = 4.0
        x_enc = torch.tanh(x_enc / tanh_factor)
        x_enc = rearrange(x_enc, "b s n -> b n s")

        if pad_left > 0 or pad_right > 0:
            x_pad = F.pad(x_enc, (pad_left, pad_right), mode=self.padding_mode)
        else:
            x_pad = x_enc

        x_2d = rearrange(x_pad, "b n (c p) -> b n p c", p=periodicity)

        resizer = safe_resize((image_size_per_var, exact_width), interpolation=self.interpolation)
        x_resize = resizer(x_2d)

        x_resize = rearrange(x_resize, "b n h w -> b 1 (n h) w")

        pad_down = image_size - x_resize.shape[2]
        pad_right_img = image_size - x_resize.shape[3]

        if pad_down > 0 or pad_right_img > 0:
            x_resize = F.pad(x_resize, (0, pad_right_img, 0, pad_down))

        assert x_resize.shape[2] == image_size, f"Height mismatch: {x_resize.shape[2]}"
        assert x_resize.shape[3] == image_size, f"Width mismatch: {x_resize.shape[3]}"

        image_input = torch.zeros(
            (x_resize.shape[0], 3, x_resize.shape[2], x_resize.shape[3]),
            device=x_resize.device,
            dtype=x_resize.dtype,
        )

        color_list = None
        if "color_list" in stats_metadata:
            color_list = stats_metadata.get("color_list")
        if color_list is None:
            color_list = [i % 3 for i in range(nvars)]
        else:
            color_list = list(np.asarray(color_list, dtype=int))

        for i in range(nvars):
            color = color_list[i]
            h_start = i * image_size_per_var
            h_end = (i + 1) * image_size_per_var
            image_input[:, color, h_start:h_end, :] = x_resize[:, 0, h_start:h_end, :]

        metadata = {
            "mode": "imputation",
            "freq": freq,
            "total_len": int(stats_metadata.get("total_len", total_len)),
            "pad_left": int(pad_left),
            "pad_right": int(pad_right),
            "periodicity": int(periodicity),
            "scale_x": float(stats_metadata.get("scale_x", 0.0)),
            "means": means_t.cpu().numpy()[0],
            "stdev": stdev_t.cpu().numpy()[0],
            "norm_const": float(self.norm_const),
            "image_size": int(image_size),
            "patch_size": int(stats_metadata.get("patch_size", self.patch_size)),
            "num_patch": int(stats_metadata.get("num_patch", self.num_patch)),
            "exact_width": int(exact_width),
            "nvars": int(nvars),
            "pad_down": int(pad_down),
            "pad_right_img": int(pad_right_img),
            "color_list": np.array(color_list, dtype=int),
            "image_size_per_var": int(image_size_per_var),
            "padding_mode": stats_metadata.get("padding_mode", self.padding_mode),
            "align_const": float(stats_metadata.get("align_const", self.align_const)),
            "color_mode": bool(stats_metadata.get("color_mode", self.color)),
            "mask_applied": False,
        }

        if stats_metadata:
            for key in (
                "mask_mode",
                "mask_ratio",
                "mask_prob",
                "mask_ratio_actual",
                "mask_ratio_range",
                "mask_ratio_per_var",
                "mask_ranges",
                "mask_value",
            ):
                if key in stats_metadata:
                    metadata[key] = stats_metadata[key]

        return {
            "target": image_input.cpu(),
            "metadata": metadata,
        }


class VisionTSImputationReconstructor:
    """Reconstruct full sequence from a reconstructed image."""

    _IMAGENET_MEAN = [0.5, 0.5, 0.5]
    _IMAGENET_STD = [0.5, 0.5, 0.5]

    def __init__(self, interpolation: str = "bilinear") -> None:
        interp = {"bilinear": Image.BILINEAR, "nearest": Image.NEAREST, "bicubic": Image.BICUBIC}
        if interpolation not in interp:
            raise ValueError(f"Unknown interpolation '{interpolation}'.")
        self._interpolation = interp[interpolation]

    def reconstruct(
        self,
        image: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
        metadata: Dict[str, Any],
        denormalize: bool = True,
        patch_size: Optional[int] = None,
    ) -> np.ndarray:
        tensor = self._load_image_to_tensor(image, int(metadata.get("image_size", 224)))
        nvars = max(1, int(metadata.get("nvars", 1)))
        color_list = metadata.get("color_list")

        color_flag = bool(metadata.get("color_mode", False))
        if color_flag or color_list is not None:
            y_grey = self._process_images(tensor, nvars, color_list)
        else:
            y_grey = torch.mean(tensor, dim=1, keepdim=True)

        seq = self._extract_full_ts_from_image(y_grey, metadata)

        if not denormalize:
            return seq.squeeze(0).cpu().numpy()

        means = torch.as_tensor(metadata["means"], dtype=torch.float32).view(1, 1, -1)
        stdev = torch.as_tensor(metadata["stdev"], dtype=torch.float32).view(1, 1, -1)

        seq = torch.clamp(seq, -0.9999, 0.9999)
        seq = torch.atanh(seq) * 4.0

        seq = seq * stdev + means
        return seq.squeeze(0).cpu().numpy()

    def _load_image_to_tensor(self, img, size) -> torch.Tensor:
        if isinstance(img, (str, Path)):
            pil_img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img).convert("RGB")
        elif isinstance(img, torch.Tensor):
            return img.unsqueeze(0) if img.ndim == 3 else img
        else:
            pil_img = img.convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
            ]
        )
        return transform(pil_img).unsqueeze(0)

    def _process_images(self, image_reconstructed, nvars, color_list) -> torch.Tensor:
        batch_size = image_reconstructed.shape[0]
        height = image_reconstructed.shape[2]
        width = image_reconstructed.shape[3]
        output = torch.zeros(
            (batch_size, 1, height, width),
            device=image_reconstructed.device,
            dtype=image_reconstructed.dtype,
        )

        for i in range(batch_size):
            h_per_var = height // nvars
            remainder = height % nvars
            for k in range(nvars):
                start_h = k * h_per_var
                end_h = (k + 1) * h_per_var
                if k == nvars - 1 and remainder != 0:
                    end_h = height

                c_idx = color_list[k] if isinstance(color_list, (list, np.ndarray)) else 0
                output[i, 0, start_h:end_h, :] = image_reconstructed[i, c_idx, start_h:end_h, :]
        return output

    def _extract_full_ts_from_image(self, y_grey: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        nvars = max(1, int(metadata.get("nvars", 1)))
        image_size_per_var = int(metadata.get("image_size_per_var", 1))

        valid_height = nvars * image_size_per_var
        if y_grey.shape[2] > valid_height:
            y_grey = y_grey[:, :, :valid_height, :]

        y_grey = rearrange(y_grey, "b 1 (n h) w -> b n h w", n=nvars)

        exact_width = int(metadata.get("exact_width", y_grey.shape[3]))
        if y_grey.shape[3] > exact_width:
            y_grey = y_grey[..., :exact_width]

        periodicity = int(metadata.get("periodicity"))
        total_len = int(metadata.get("total_len"))
        pad_left = int(metadata.get("pad_left"))
        pad_right = int(metadata.get("pad_right"))

        full_padded_len = pad_left + total_len + pad_right

        target_cycles = full_padded_len // periodicity
        if target_cycles == 0:
            target_cycles = 1

        resizer = safe_resize((periodicity, target_cycles), interpolation=self._interpolation)
        y_seg = resizer(y_grey)

        y_flat = rearrange(y_seg, "b n p c -> b n (c p)")
        y_flat = y_flat.transpose(1, 2)

        start_idx = pad_left
        end_idx = y_flat.shape[1] - pad_right

        if end_idx > start_idx:
            y_flat = y_flat[:, start_idx:end_idx, :]
        else:
            y_flat = y_flat[:, :total_len, :]

        return y_flat

