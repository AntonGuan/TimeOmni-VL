from __future__ import annotations

import inspect
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import einops
from timm.models.vision_transformer import PatchEmbed

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from einops import rearrange, repeat
from torchvision import transforms
from torchvision.transforms import Resize


# -----------------------------------------------------------------------------
# utility functions & constants
# -----------------------------------------------------------------------------
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
# converter / reconstructor
# -----------------------------------------------------------------------------
@dataclass
class TSImageSample:
    image: np.ndarray           # shape: (C, H, W)
    metadata: Dict[str, Any]
    normalized_context: Optional[np.ndarray] = None
    normalized_future: Optional[np.ndarray] = None


class VisionTSImageConverter:
    """Converts time series windows into VisionTS forward rendering results consistently."""

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
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patch_input = num_patch_input
        self.norm_const = norm_const
        self.padding_mode = padding_mode
        self.align_const = align_const
        self.color = color
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)


    def update_config(
        self,
        context_len: int,
        pred_len: int,
        num_patch_input: Optional[int] = None,
        periodicity: int = 1,
        norm_const: Optional[float] = None,
        interpolation: str = "bilinear",
        padding_mode: str = "replicate",
        ) -> None:
        self.image_size_cfg = self.image_size
        self.patch_size_cfg = self.patch_size
        self.num_patch = self.image_size_cfg // self.patch_size_cfg

        self.context_len = context_len
        self.pred_len = pred_len
        self.periodicity = periodicity
        self.padding_mode = padding_mode
        
        if num_patch_input is not None:
            # extra padding
            extra_padding = pred_len / (self.num_patch - num_patch_input) * num_patch_input - self.context_len
            if extra_padding > 0:
                self.context_len += int(np.ceil(extra_padding))

        self.pad_left = 0
        self.pad_right = 0
        if self.context_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.context_len % self.periodicity

        if self.pred_len % self.periodicity != 0:
            self.pad_right = self.periodicity - self.pred_len % self.periodicity
        
        input_ratio = (self.pad_left + self.context_len) / (self.pad_left + self.context_len + self.pad_right + self.pred_len)

        if num_patch_input is None:
            self.num_patch_input = int(input_ratio * self.num_patch * self.align_const)
            if self.num_patch_input == 0:
                self.num_patch_input = 1
        else:
            self.num_patch_input = num_patch_input

        self.num_patch_output = self.num_patch - self.num_patch_input
        self.adjust_input_ratio = self.num_patch_input / self.num_patch

        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        
        exact_input_width = self.num_patch_input * self.patch_size
        
        self.scale_x = ((self.pad_left + self.context_len) // self.periodicity) / exact_input_width

        self.norm_const = norm_const
        
        mask = torch.ones((self.num_patch, self.num_patch))
        mask[:, :self.num_patch_input] = torch.zeros((self.num_patch, self.num_patch_input))
        self.mask = mask.reshape(1, -1)      # Normal attribute
        self.mask_ratio = self.mask.mean().item()


    def convert(self, context_len, pred_len, series: np.ndarray, freq: str = "H" , period = None) -> TSImageSample:
        # Input: length * nvar, swap dimensions
        arr = np.asarray(series, dtype=np.float32).transpose(1, 0)
        if arr.ndim == 1:
            arr = arr[None, :]

        nvars, total_len = arr.shape

        if not period:
            periodicity_list = [x for x in freq_to_seasonality_list(freq) if x * 2 < total_len]
            
            print(periodicity_list)

            if len(periodicity_list) >=2:
                periodicity = random.choice(periodicity_list[:-1])
            else:
                periodicity = periodicity_list[-1]

            print(f"Selected periodicity: {periodicity}")
        else:
            periodicity = period
            print(f"Selected periodicity: {periodicity}")

        self.update_config(
            context_len=context_len,
            pred_len=pred_len,
            num_patch_input=None,
            periodicity=periodicity,
            norm_const=self.norm_const,
            padding_mode=self.padding_mode,
        )

        entry = self._render_like_model(arr, freq=freq)
        return TSImageSample(
            image=entry["target"],
            metadata=entry["metadata"]
        )



    def _render_like_model(self, data: np.ndarray, freq: str) -> Dict[str, Any]:

        nvars, _ = data.shape
        periodicity = self.periodicity
        context_len = self.context_len
        pred_len = self.pred_len
        pad_left = self.pad_left
        pad_right = self.pad_right
        num_patch_output = self.num_patch_output
        adjust_input_ratio = self.adjust_input_ratio
        image_size_per_var = max(1, self.image_size // max(1, nvars))
        scale_x = self.scale_x

        x_tensor = torch.from_numpy(data).transpose(0, 1).float()

        x = x_tensor.unsqueeze(0) 


        # Robust Norm + Tanh
        
        median_val = x.median(dim=1, keepdim=True).values.detach()
        abs_deviation = (x - median_val).abs()
        mad_val = abs_deviation.median(dim=1, keepdim=True).values.detach()
        robust_scale = mad_val / 0.6745
        
        std_val = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-18).detach()
        

        scale = 0.5 * robust_scale + 0.5 * std_val
        
        scale = torch.maximum(scale, torch.tensor(1e-18, device=x.device))
        
        # 3. Normalization
        means = median_val
        stdev = scale 
        
        x_enc = (x - means) / stdev
        
        tanh_factor = 4.0 
        x_enc = torch.tanh(x_enc / tanh_factor)

        x_enc = einops.rearrange(x_enc, 'b s n -> b n s')


        if x_enc.shape[-1] < self.context_len:
            extra_padding = self.context_len - x_enc.shape[-1]
        else:
            extra_padding = 0

        # 2. Segmentation
        x_pad = F.pad(x_enc, (pad_left + extra_padding, 0), mode=self.padding_mode) # [b n s]

        x_2d = rearrange(x_pad, "b n (p f) -> b n f p", f=periodicity)

        exact_input_width = self.num_patch_input * self.patch_size

        input_resize = safe_resize(
            (image_size_per_var, exact_input_width), # <--- Modified here
            interpolation=self.interpolation,
        )
        
        x_resize = input_resize(x_2d)
        x_resize = rearrange(x_resize, "b n h w -> b 1 (n h) w")
        pad_down = self.image_size - x_resize.shape[2]
        if pad_down > 0:
            x_resize = torch.cat(
                [
                    x_resize,
                    torch.zeros(
                        (x_resize.shape[0], 1, pad_down, x_resize.shape[3]),
                        device=x_resize.device,
                        dtype=x_resize.dtype,
                    ),
                ],
                dim=2,
            )
        assert x_resize.shape[2] == self.image_size, f"image size mismatch: {x_resize.shape[2]} vs {self.image_size}"

        masked = torch.zeros(
            (x_2d.shape[0], 1, self.image_size, num_patch_output * self.patch_size),
            device=x_2d.device,
            dtype=x_2d.dtype,
        )
        x_concat_with_masked = torch.cat([x_resize, masked], dim=-1)

        image_input = torch.zeros((x_concat_with_masked.shape[0], 3, x_concat_with_masked.shape[2], x_concat_with_masked.shape[3]), 
                                  device=x_concat_with_masked.device, 
                                  dtype=x_concat_with_masked.dtype)  # [bs, 3, 224, 224]


        # # RGB sequence rotation
        color_list = None
        if color_list is None:
            color_list = [i % 3 for i in range(nvars)]

        for i in range(nvars):
            color = color_list[i]
            image_input[:, color, i*image_size_per_var:(i+1)*image_size_per_var, :] = \
                x_concat_with_masked[:, 0, i*image_size_per_var:(i+1)*image_size_per_var, :]

        metadata = {
            "freq": freq,
            "pad_left": int(pad_left),
            "pad_right": int(pad_right),
            "context_len": int(context_len),
            "pred_len": int(pred_len),
            "periodicity": int(periodicity),
            "scale_x": float(scale_x),
            "means": means.cpu().numpy()[0],
            "stdev": stdev.cpu().numpy()[0],
            "norm_const": float(self.norm_const),
            "image_size": int(self.image_size),
            "patch_size": int(self.patch_size),
            "num_patch_input": int(self.num_patch_input),
            "num_patch_output": int(num_patch_output),
            "adjust_input_ratio": float(adjust_input_ratio),
            "nvars": int(nvars),
            "pad_down": int(pad_down),
            "color_list": None if color_list is None else np.array(color_list, dtype=int),
            "image_size_per_var": int(image_size_per_var),
            "padding_mode": self.padding_mode,
            "color_mode": self.color,
            "mask_ratio": float(self.mask_ratio),
            "mask": self.mask.cpu(),
        }

        return {
            "target": image_input.cpu(),
            "metadata": metadata,
        }


    def render_full_groundtruth_image(
        self,
        full_seq: torch.Tensor,        # [total_len, nvars]
        sample: TSImageSample,   # Return from convert()
    ) -> torch.Tensor:
        """
        Construct "complete sequence" image, reusing _render_like_model metadata processing.
        The full image contains only the complete sequence, no longer concatenated left and right.
        """

        if full_seq.ndim != 2:
            raise ValueError(f"full_seq must be [total_len, nvars], got {full_seq.shape}")

        # =====================================================
        # 0. Extract metadata
        # =====================================================
        meta = sample.metadata

        image_size = int(meta["image_size"])
        image_size_per_var = int(meta["image_size_per_var"])
        periodicity = int(meta["periodicity"])
        pad_left = int(meta["pad_left"])
        pad_right = int(meta["pad_right"])
        color_list = meta["color_list"]
        padding_mode = meta.get("padding_mode", self.padding_mode)

        expected_len = int(meta["context_len"]) + int(meta["pred_len"])
        nvars = int(meta["nvars"])
        if full_seq.shape[0] != expected_len:
            raise ValueError(
                f"full_seq length {full_seq.shape[0]} != expected {expected_len}"
            )
        if full_seq.shape[1] != nvars:
            raise ValueError(
                f"full_seq nvars {full_seq.shape[1]} != expected {nvars}"
            )

        device = full_seq.device
        means = torch.as_tensor(meta["means"], device=device).float().view(1, 1, -1)
        stdev = torch.as_tensor(meta["stdev"], device=device).float().view(1, 1, -1)

        # =====================================================
        # 1. Process full sequence
        # =====================================================
        full_seq = full_seq.to(device).float().unsqueeze(0)  # [1, total_len, nvars]

        # Use normalization statistics from _render_like_model
        full_seq = (full_seq - means) / stdev
        full_seq = torch.tanh(full_seq / 4.0)
        full_seq = rearrange(full_seq, "b s n -> b n s")  # [1, nvars, total_len]

        if pad_left > 0 or pad_right > 0:
            full_seq = F.pad(full_seq, (pad_left, pad_right), mode=padding_mode)

        if full_seq.shape[-1] % periodicity != 0:
            raise RuntimeError(
                f"Full sequence length {full_seq.shape[-1]} not divisible by periodicity={periodicity}"
            )

        # segmentation
        full_2d = rearrange(full_seq, "b n (p f) -> b n f p", f=periodicity)

        # =====================================================
        # 2. Resize to full image width
        # =====================================================
        full_width = image_size
        resize_full = safe_resize(
            (image_size_per_var, full_width),
            interpolation=self.interpolation,
        )

        full_resize = resize_full(full_2d)
        full_resize = rearrange(full_resize, "b n h w -> b 1 (n h) w")

        pad_down = image_size - full_resize.shape[2]
        pad_right_img = image_size - full_resize.shape[3]
        if pad_down > 0 or pad_right_img > 0:
            full_resize = F.pad(full_resize, (0, pad_right_img, 0, pad_down))

        # =====================================================
        # 3. Write color channels
        # =====================================================
        image_input = torch.zeros(
            (full_resize.shape[0], 3, full_resize.shape[2], full_resize.shape[3]),
            device=full_resize.device,
            dtype=full_resize.dtype,
        )

        if color_list is None:
            color_list = [i % 3 for i in range(nvars)]
        else:
            color_list = [int(c) for c in color_list]

        for i, color in enumerate(color_list):
            h0 = i * image_size_per_var
            h1 = (i + 1) * image_size_per_var
            image_input[:, color, h0:h1, :] = full_resize[:, 0, h0:h1, :]

        return image_input


class VisionTSImageReconstructor:
    """Reads the image and reconstructs the time series."""

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
    ) -> np.ndarray:
        tensor = self._load_image_to_tensor(image, int(metadata.get("image_size", 224)))
        nvars = max(1, int(metadata.get("nvars", 1)))
        color_list = metadata.get("color_list")
        
        color_flag = bool(metadata.get("color_mode", True))
        if color_flag:
            y_grey = self._process_images(tensor, nvars, color_list)
        else:
            y_grey = torch.mean(tensor, dim=1, keepdim=True)
                
        seq = self._extract_ts_from_image(y_grey, metadata)
        context_len = int(metadata.get("context_len", 0))
        pred_len = int(metadata.get("pred_len", 0))
        total_len = context_len + pred_len

        if isinstance(seq, tuple):
            seq_full = torch.cat(seq, dim=1)
        else:
            seq_full = seq

        if not denormalize:
            ctx = seq_full[:, :context_len, :]
            pred = seq_full[:, context_len:context_len + pred_len, :]
            return ctx.squeeze(0).cpu().numpy(), pred.squeeze(0).cpu().numpy()
        means = torch.as_tensor(metadata["means"], dtype=torch.float32).view(1, 1, -1)
        stdev = torch.as_tensor(metadata["stdev"], dtype=torch.float32).view(1, 1, -1)


        # 2. ArcTanh + Linear Denorms
        seq_full = torch.clamp(seq_full, -0.9999, 0.9999)
        seq_full = torch.atanh(seq_full) * 4.0
        seq_full = seq_full * stdev + means

        ctx = seq_full[:, :context_len, :]
        pred = seq_full[:, context_len:context_len + pred_len, :]
        return ctx.squeeze(0).cpu().numpy(), pred.squeeze(0).cpu().numpy()

    def _load_image_to_tensor(
        self,
        img: Union[str, Path, np.ndarray, Image.Image, torch.Tensor],
        size: int,
    ) -> torch.Tensor:

        pil_img = Image.open(img).convert("RGB")

        print(pil_img.size)

        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])


        return transform(pil_img).unsqueeze(0)

    def _process_images(
        self,
        image_reconstructed: torch.Tensor,
        nvars: int,
        color_list: List[int],
    ) -> torch.Tensor:
        batch_size = image_reconstructed.shape[0]
        height = image_reconstructed.shape[2]
        width = image_reconstructed.shape[3]
        output = torch.zeros((batch_size, 1, height, width), device=image_reconstructed.device, dtype=image_reconstructed.dtype)

        for i in range(batch_size):
            h_per_var = height // nvars
            remainder = height % nvars
            for k in range(nvars):
                start_h = k * h_per_var
                end_h = (k + 1) * h_per_var
                if k == nvars - 1 and remainder != 0:
                    end_h = height
                color_channel = color_list[k]
                output[i, 0, start_h:end_h, :] = image_reconstructed[i, color_channel, start_h:end_h, :]
        return output



    def _extract_ts_from_image(
        self,
        y_grey: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract complete sequence from image, then slice into (context, pred) based on pred_len.
        """
        nvars = max(1, int(metadata.get("nvars", 1)))
        image_size_per_var = int(metadata.get("image_size_per_var", 1))

        # 1. Handle bottom Padding (Pad Down)
        valid_height = nvars * image_size_per_var
        if y_grey.shape[2] > valid_height:
            y_grey = y_grey[:, :, :valid_height, :]

        # 2. Restore variable stacking (Stacking)
        y_grey = rearrange(y_grey, "b 1 (n h) w -> b n h w", n=nvars)

        # 3. Truncate to image width
        image_size = int(metadata.get("image_size", y_grey.shape[3]))
        if y_grey.shape[3] > image_size:
            y_grey = y_grey[..., :image_size]

        periodicity = int(metadata.get("periodicity"))
        pad_left = int(metadata.get("pad_left"))
        pad_right = int(metadata.get("pad_right"))
        context_len = int(metadata.get("context_len"))
        pred_len = int(metadata.get("pred_len"))

        total_len = context_len + pred_len
        full_padded_len = pad_left + total_len + pad_right

        target_cycles = full_padded_len // periodicity
        if target_cycles == 0:
            target_cycles = 1

        resize_full = safe_resize((periodicity, target_cycles), interpolation=self._interpolation)
        y_seg = resize_full(y_grey)  # [B, N, Periodicity, Cycles]

        y_flat = rearrange(y_seg, "b n f p -> b (p f) n")

        start_idx = pad_left
        end_idx = start_idx + total_len
        if end_idx <= y_flat.shape[1]:
            ts_full = y_flat[:, start_idx:end_idx, :]
        else:
            ts_full = y_flat[:, start_idx:, :]

        ts_context = ts_full[:, :context_len, :]
        ts_pred = ts_full[:, context_len:context_len + pred_len, :]

        return ts_context, ts_pred