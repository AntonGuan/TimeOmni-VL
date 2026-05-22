from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange


ImageLike = Union[str, Path, np.ndarray, Image.Image, torch.Tensor]


_IMAGENET_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_IMAGENET_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _as_plain_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    plain = {}
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        plain[key] = value
    return plain


def _get_int(metadata: Dict[str, Any], key: str, default: int | None = None) -> int:
    if key not in metadata:
        if default is None:
            raise KeyError(f"metadata is missing required key '{key}'")
        return default
    return int(metadata[key])


def _to_color_list(color_list: Any, nvars: int) -> list[int]:
    if color_list is None:
        return [i % 3 for i in range(nvars)]
    if isinstance(color_list, torch.Tensor):
        color_list = color_list.detach().cpu().numpy()
    if isinstance(color_list, np.ndarray):
        color_list = color_list.tolist()
    if not isinstance(color_list, Iterable):
        raise TypeError("metadata['color_list'] must be an iterable of channel ids")

    colors = [int(c) for c in color_list]
    if len(colors) < nvars:
        colors.extend(i % 3 for i in range(len(colors), nvars))
    return [max(0, min(2, c)) for c in colors[:nvars]]


def _pil_from_image(image: ImageLike) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            if arr.min() < 0:
                arr = (arr * _IMAGENET_STD + _IMAGENET_MEAN) * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


def _load_image_to_tensor(image: ImageLike, image_size: int) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu().float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"image tensor must be CHW or BCHW, got {tuple(tensor.shape)}")
        if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.shape[1] != 3:
            raise ValueError(f"image tensor must have 3 channels, got {tuple(tensor.shape)}")
        if tensor.shape[-2:] != (image_size, image_size):
            tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
        if tensor.min() >= 0.0 and tensor.max() > 1.5:
            tensor = tensor / 255.0
        if tensor.min() >= 0.0:
            tensor = (tensor - 0.5) / 0.5
        return tensor

    pil_img = _pil_from_image(image)
    if pil_img.size != (image_size, image_size):
        pil_img = pil_img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def _select_variable_channels(tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    nvars = max(1, _get_int(metadata, "nvars", 1))
    color_mode = bool(metadata.get("color_mode", True))
    color_list = _to_color_list(metadata.get("color_list"), nvars)

    if not color_mode and metadata.get("color_list") is None:
        return tensor.mean(dim=1, keepdim=True)

    batch, _channels, height, width = tensor.shape
    output = torch.zeros((batch, 1, height, width), dtype=tensor.dtype, device=tensor.device)
    h_per_var = height // nvars
    remainder = height % nvars

    for var_idx in range(nvars):
        y0 = var_idx * h_per_var
        y1 = (var_idx + 1) * h_per_var
        if var_idx == nvars - 1 and remainder:
            y1 = height
        channel = color_list[var_idx]
        output[:, 0, y0:y1, :] = tensor[:, channel, y0:y1, :]

    return output


def _resize_sequence_image(y_grey: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    nvars = max(1, _get_int(metadata, "nvars", 1))
    image_size_per_var = _get_int(metadata, "image_size_per_var", max(1, y_grey.shape[2] // nvars))
    valid_height = nvars * image_size_per_var
    if y_grey.shape[2] > valid_height:
        y_grey = y_grey[:, :, :valid_height, :]

    y_grey = rearrange(y_grey, "b 1 (n h) w -> b n h w", n=nvars)

    image_size = _get_int(metadata, "image_size", y_grey.shape[3])
    if y_grey.shape[3] > image_size:
        y_grey = y_grey[..., :image_size]

    periodicity = max(1, _get_int(metadata, "periodicity"))
    context_len = _get_int(metadata, "context_len")
    pred_len = _get_int(metadata, "pred_len")
    pad_left = _get_int(metadata, "pad_left", 0)
    pad_right = _get_int(metadata, "pad_right", 0)

    total_len = context_len + pred_len
    full_padded_len = pad_left + total_len + pad_right
    target_cycles = max(1, full_padded_len // periodicity)
    if target_cycles * periodicity < full_padded_len:
        target_cycles += 1

    return F.interpolate(
        y_grey,
        size=(periodicity, target_cycles),
        mode="bilinear",
        align_corners=False,
    )


def _extract_forecast_sequence(y_grey: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    y_seg = _resize_sequence_image(y_grey, metadata)
    y_flat = rearrange(y_seg, "b n f p -> b (p f) n")

    context_len = _get_int(metadata, "context_len")
    pred_len = _get_int(metadata, "pred_len")
    total_len = context_len + pred_len
    pad_left = _get_int(metadata, "pad_left", 0)

    start = max(0, pad_left)
    end = start + total_len
    seq = y_flat[:, start:end, :]

    if seq.shape[1] < total_len:
        pad_len = total_len - seq.shape[1]
        seq = F.pad(seq.transpose(1, 2), (0, pad_len), mode="replicate").transpose(1, 2)
    elif seq.shape[1] > total_len:
        seq = seq[:, :total_len, :]

    return seq


def _denormalize_sequence(seq: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
    if "means" not in metadata or "stdev" not in metadata:
        raise KeyError("metadata must include 'means' and 'stdev' when denormalize=True")

    means = torch.as_tensor(metadata["means"], dtype=torch.float32).view(1, 1, -1)
    stdev = torch.as_tensor(metadata["stdev"], dtype=torch.float32).view(1, 1, -1)

    seq = torch.clamp(seq, -0.9999, 0.9999)
    seq = torch.atanh(seq) * 4.0
    return seq * stdev + means


def reconstruct_timeseries(
    image: ImageLike,
    metadata: Dict[str, Any],
    denormalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover context, prediction, and full forecast arrays from a VisionTS image.

    The returned arrays use shape [time, nvars]. This function matches the call
    site in deploy/app.py:
        ctx_hat, pred_hat, full = reconstruct_timeseries(...)
    """

    metadata = _as_plain_metadata(metadata)
    image_size = _get_int(metadata, "image_size", 224)
    context_len = _get_int(metadata, "context_len")
    pred_len = _get_int(metadata, "pred_len")

    tensor = _load_image_to_tensor(image, image_size)
    y_grey = _select_variable_channels(tensor, metadata)
    seq = _extract_forecast_sequence(y_grey, metadata)

    if denormalize:
        seq = _denormalize_sequence(seq, metadata)

    full = seq.squeeze(0).detach().cpu().numpy()
    ctx = full[:context_len]
    pred = full[context_len : context_len + pred_len]
    return ctx, pred, full


def _main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct time series from a VisionTS forecast image.")
    parser.add_argument("image", type=Path, help="Path to the generated/restored RGB image.")
    parser.add_argument("metadata", type=Path, help="Path to the metadata JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npy path for the full reconstructed series.")
    parser.add_argument("--no-denormalize", action="store_true", help="Return normalized tanh-space values.")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    ctx, pred, full = reconstruct_timeseries(args.image, metadata, denormalize=not args.no_denormalize)
    print(f"context shape: {ctx.shape}, pred shape: {pred.shape}, full shape: {full.shape}")

    if args.output is not None:
        np.save(args.output, full)


if __name__ == "__main__":
    _main()
