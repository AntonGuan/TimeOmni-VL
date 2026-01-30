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
#Full Sequence Converter / Reconstructor
# -----------------------------------------------------------------------------
@dataclass
class TSImageSample:
    image: np.ndarray           # shape: (C, H, W)
    metadata: Dict[str, Any]
    normalized_context: Optional[np.ndarray] = None
    normalized_future: Optional[np.ndarray] = None


class VisionTSFullSequenceConverter:
    """
    Converts a [Complete Time Series] into an image.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_patch_input: int = 7, 
        norm_const: float = 0.4,
        align_const: float = 0.4,
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
        total_len: int,
        periodicity: int = 1,
        interpolation: str = "bilinear",
        padding_mode: str = "replicate",
    ) -> None:
        """
        Configuration update for the full sequence version.
        """
        # 1. Basic parameters
        self.image_size_cfg = self.image_size
        self.patch_size_cfg = self.patch_size
        self.periodicity = periodicity
        self.padding_mode = padding_mode
        self.total_len = total_len
        
        # 2. Calculate valid width related to Patch

        self.num_patch = self.image_size_cfg // self.patch_size_cfg
        self.exact_width = self.num_patch * self.patch_size_cfg

        # 3. Calculate Padding (Replicate original pad_left logic)

        self.pad_left = 0
        if self.total_len % self.periodicity != 0:
            self.pad_left = self.periodicity - self.total_len % self.periodicity
        
        self.pad_right = 0

        self.full_padded_len = self.pad_left + self.total_len + self.pad_right

        # 4. Other configurations
        self.interpolation = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }[interpolation]
        
        # Calculate scaling ratio
        cycles = self.full_padded_len // periodicity
        self.scale_x = cycles / self.exact_width

    def convert(self, series: np.ndarray, freq: str = "H", period=None) -> TSImageSample:
        # Input processing: [L, N] -> [N, L]
        arr = np.asarray(series, dtype=np.float32).transpose(1, 0)
        if arr.ndim == 1:
            arr = arr[None, :]

        nvars, total_len = arr.shape

        # Determine periodicity
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

        entry = self._render_full_sequence(arr, freq=freq)
        
        return TSImageSample(
            image=entry["target"],
            metadata=entry["metadata"]
        )

    def _render_full_sequence(self, data: np.ndarray, freq: str) -> Dict[str, Any]:
        nvars, _ = data.shape
        periodicity = self.periodicity
        pad_left = self.pad_left
        pad_right = self.pad_right
        image_size_per_var = max(1, self.image_size // max(1, nvars))
        
        # 1. Convert to Tensor & add Batch dim: [1, L, N]
        x = torch.from_numpy(data).float().transpose(0, 1).unsqueeze(0)

        # updated normalization with Soft MAD
        median_val = x.median(dim=1, keepdim=True).values.detach()
        abs_deviation = (x - median_val).abs()
        mad_val = abs_deviation.median(dim=1, keepdim=True).values.detach()
        robust_scale = mad_val / 0.6745
        
        # Calculate Standard Deviation as a fallback
        std_val = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-18).detach()
        
        # 2. Hybrid Scaling Strategy

        scale = 0.5 * robust_scale + 0.5 * std_val
        
        # Prevent division by very small values (Fallback mechanism)
        scale = torch.maximum(scale, torch.tensor(1e-18, device=x.device))
        
        # 3. Normalization
        means = median_val
        stdev = scale 
        
        x_enc = (x - means) / stdev
        
        tanh_factor = 4.0 
        x_enc = torch.tanh(x_enc / tanh_factor)

        # [B, L, N] -> [B, N, L]
        x_enc = einops.rearrange(x_enc, "b s n -> b n s")
        
        # 3. Padding (Apply Pad Left & Pad Right)
        if pad_left > 0 or pad_right > 0:
            x_pad = F.pad(x_enc, (pad_left, pad_right), mode=self.padding_mode)
        else:
            x_pad = x_enc

        # 4. Segmentation: [B, N, L_pad] -> [B, N, Period, Cycles]

        x_2d = rearrange(x_pad, "b n (c p) -> b n p c", p=periodicity)

        # 5. Resize to Exact Width (based on patch_size)
        resizer = safe_resize(
            (image_size_per_var, self.exact_width), 
            interpolation=self.interpolation
        )
        
        x_resize = resizer(x_2d) # -> [B, N, H_var, W_exact]
        
        # 6. Stack Variables
        x_resize = rearrange(x_resize, "b n h w -> b 1 (n h) w")
        
        # 7. Image Padding (Handle Image Size misalignment)

        pad_down = self.image_size - x_resize.shape[2]

        pad_right_img = self.image_size - x_resize.shape[3]
        
        if pad_down > 0 or pad_right_img > 0:
            x_resize = F.pad(x_resize, (0, pad_right_img, 0, pad_down))
        
        assert x_resize.shape[2] == self.image_size, f"Height mismatch: {x_resize.shape[2]}"
        assert x_resize.shape[3] == self.image_size, f"Width mismatch: {x_resize.shape[3]}"

        # 8. Colorization
        image_input = torch.zeros(
            (x_resize.shape[0], 3, x_resize.shape[2], x_resize.shape[3]), 
            device=x_resize.device, 
            dtype=x_resize.dtype
        )
        
        color_list = None
        if color_list is None:
            color_list = [i % 3 for i in range(nvars)]

        for i in range(nvars):
            color = color_list[i]
            
            h_start = i * image_size_per_var
            h_end = (i + 1) * image_size_per_var
            
            image_input[:, color, h_start:h_end, :] = x_resize[:, 0, h_start:h_end, :]

        metadata = {
            "mode": "full_sequence",
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
            
            # Patch related parameters
            "patch_size": int(self.patch_size), 
            "num_patch": int(self.num_patch),
            "exact_width": int(self.exact_width),
            
            "nvars": int(nvars),
            "pad_down": int(pad_down),
            "pad_right_img": int(pad_right_img),
            "color_list": np.array(color_list, dtype=int),
            "image_size_per_var": int(image_size_per_var),
            "padding_mode": self.padding_mode,
            "color_mode": self.color,
        }

        return {
            "target": image_input.cpu(),
            "metadata": metadata,
        }


class VisionTSFullSequenceReconstructor:
    """Reads the image and reconstructs the [Complete Time Series]."""

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
        patch_size: Optional[int] = None 
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

        # 2. ArcTanh + Linear Denorm
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
             
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._IMAGENET_MEAN, std=self._IMAGENET_STD),
        ])
        return transform(pil_img).unsqueeze(0)

    def _process_images(self, image_reconstructed, nvars, color_list) -> torch.Tensor:
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
                
                c_idx = color_list[k] if isinstance(color_list, (list, np.ndarray)) else 0
                output[i, 0, start_h:end_h, :] = image_reconstructed[i, c_idx, start_h:end_h, :]
        return output

    def _extract_full_ts_from_image(self, y_grey: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """
        Extract complete sequence from image.
        """
        nvars = max(1, int(metadata.get("nvars", 1)))
        image_size_per_var = int(metadata.get("image_size_per_var", 1))
        
        # 1. Remove Image Pad Down
        valid_height = nvars * image_size_per_var
        if y_grey.shape[2] > valid_height:
            y_grey = y_grey[:, :, :valid_height, :]

        # 2. Unstack Variables
        y_grey = rearrange(y_grey, "b 1 (n h) w -> b n h w", n=nvars)
        
        # 3. Remove Image Pad Right - Key to restoring patch_size logic
        exact_width = int(metadata.get("exact_width", y_grey.shape[3]))
        if y_grey.shape[3] > exact_width:
             y_grey = y_grey[..., :exact_width]

        # 4. Calculate Target Dimensions
        periodicity = int(metadata.get("periodicity"))
        total_len = int(metadata.get("total_len"))
        pad_left = int(metadata.get("pad_left"))
        pad_right = int(metadata.get("pad_right"))
        
        full_padded_len = pad_left + total_len + pad_right
        
        # Target Width (Cycles)
        target_cycles = full_padded_len // periodicity
        if target_cycles == 0: target_cycles = 1
        
        # 5. Resize: [Height=Periodicity, Width=Cycles]
        resizer = safe_resize((periodicity, target_cycles), interpolation=self._interpolation)
        y_seg = resizer(y_grey) # -> [B, N, Period, Cycles]

        # 6. Flatten (Patching Logic)
        # Reverse: "b n p c -> b n (c p)"
        y_flat = rearrange(y_seg, "b n p c -> b n (c p)")

        # 7. Transpose to standard [B, L, N]
        y_flat = y_flat.transpose(1, 2)

        # 8. Remove Sequence Padding
        # Pad Left is at the head, Pad Right is at the tail
        start_idx = pad_left
        end_idx = y_flat.shape[1] - pad_right
        
        # Safe slicing
        if end_idx > start_idx:
            y_flat = y_flat[:, start_idx:end_idx, :]
        else:
            # Fallback if something went wrong, though exact_width logic should prevent this
            y_flat = y_flat[:, :total_len, :]

        return y_flat