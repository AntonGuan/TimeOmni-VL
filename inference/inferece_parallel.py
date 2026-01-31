from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import multiprocessing as mp
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data.data_utils import add_special_tokens, pil_img2rgb
from training.data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from training.modeling.autoencoder import load_ae
from training.modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from training.modeling.qwen2 import Qwen2Tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument(
        "--jsonl",
        default=str(REPO_ROOT / "data_pipeline/demo_level_samples/forecast_samples_thinking_gen.jsonl"),
        help="JSONL containing instruction/source_image/target_image.",
    )
    parser.add_argument(
        "--input-root",
        default=str(REPO_ROOT / "data_pipeline/demo_level_samples"),
        help="Root used to build relative output paths.",
    )
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "inference/outputs/model_edits"),
    )
    parser.add_argument("--output-name", default="edit.png")
    parser.add_argument(
        "--metrics-xlsx",
        default=None,
        help="Optional XLSX output path for metrics.",
    )
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--cfg_text_scale", type=float, default=4.0)
    parser.add_argument("--cfg_img_scale", type=float, default=2.0)
    parser.add_argument("--cfg_interval", type=float, default=0.0)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--timestep_shift", type=float, default=3.0)
    parser.add_argument("--cfg_renorm_min", type=float, default=0.0)
    parser.add_argument("--cfg_renorm_type", type=str, default="text_channel")
    parser.add_argument("--think", action="store_true", help="Enable think mode.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling samples.")
    parser.add_argument(
        "--device-ids",
        type=str,
        default="0",
        help="Comma-separated CUDA device ids for model sharding.",
    )
    parser.add_argument(
        "--data-parallel",
        action="store_true",
        help="Split samples across device-ids and run in parallel (one process per device).",
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Output CSV for metrics. Defaults to <output-root>/metrics.csv",
    )
    return parser


def load_model(args):
    llm_config = Qwen2Config.from_json_file(os.path.join(args.base_model, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(args.base_model, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(args.base_model, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(args.base_model)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    device_ids = [int(x) for x in args.device_ids.split(",") if x.strip()]
    if not device_ids:
        raise ValueError("--device-ids must contain at least one CUDA device id.")
    max_memory = {i: "75GiB" for i in device_ids}
    print(f"DEBUG: Using max_memory allocation: {max_memory}")

    primary_device = f"cuda:{device_ids[0]}"
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )

    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]

    first_device = primary_device
    for k in same_device_modules:
        device_map[k] = first_device

    for module_name, device in device_map.items():
        if device in ("cpu", "disk"):
            device_map[module_name] = first_device

    if args.mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=args.ckpt,
            device_map=device_map,
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
    elif args.mode == 2:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
        model = load_and_quantize_model(
            model,
            weights_location=args.ckpt,
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
        ).eval()
    elif args.mode == 3:
        bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, torch_dtype=torch.float32)
        model = load_and_quantize_model(
            model,
            weights_location=args.ckpt,
            bnb_quantization_config=bnb_quantization_config,
            device_map=device_map,
        ).eval()
    else:
        raise NotImplementedError

    vae_model = vae_model.to(device=first_device, dtype=torch.bfloat16).eval()

    original_forward_cache_update_vae = model.forward_cache_update_vae

    def _forward_cache_update_vae_with_cast(
        self,
        vae_model,
        past_key_values,
        padded_images,
        patchified_vae_latent_shapes,
        packed_vae_position_ids,
        packed_timesteps,
        packed_vae_token_indexes,
        packed_text_ids,
        packed_text_indexes,
        packed_position_ids,
        packed_seqlens,
        packed_indexes,
        key_values_lens,
        packed_key_value_indexes,
    ):
        vae_param = next(vae_model.parameters())
        if padded_images.device != vae_param.device or padded_images.dtype != vae_param.dtype:
            padded_images = padded_images.to(device=vae_param.device, dtype=vae_param.dtype)
        return original_forward_cache_update_vae(
            vae_model,
            past_key_values,
            padded_images,
            patchified_vae_latent_shapes,
            packed_vae_position_ids,
            packed_timesteps,
            packed_vae_token_indexes,
            packed_text_ids,
            packed_text_indexes,
            packed_position_ids,
            packed_seqlens,
            packed_indexes,
            key_values_lens,
            packed_key_value_indexes,
        )

    model.forward_cache_update_vae = types.MethodType(_forward_cache_update_vae_with_cast, model)

    vae_transform = ImageTransform(896, 896, 16)
    vit_transform = ImageTransform(896, 896, 14)

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def _resolve_ckpt(base_model: str, ckpt: Optional[str]) -> str:
    if ckpt:
        return ckpt
    candidates = [
        os.path.join(base_model, "ema.safetensors"),
        os.path.join(base_model, "model.safetensors"),
        os.path.join(base_model, "model.safetensors.index.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "ckpt not provided and no default checkpoint found under base_model. "
        "Tried: " + ", ".join(candidates)
    )


def load_jsonl(jsonl_path: Path) -> List[Dict[str, str]]:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _image_to_array(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def _compute_metrics(pred: Image.Image, target: Image.Image) -> Dict[str, float]:
    if pred.size != target.size:
        target = target.resize(pred.size, resample=Image.BILINEAR)
    pred_arr = _image_to_array(pred)
    target_arr = _image_to_array(target)
    diff = pred_arr - target_arr
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 20 * math.log10(255.0 / math.sqrt(mse))
    return {"mae": mae, "mse": mse, "psnr": psnr}


def run_inference(
    records: List[Dict[str, str]],
    args: argparse.Namespace,
    worker_id: Optional[int] = None,
    total_workers: int = 1,
) -> Optional[Path]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output_root / "metrics.csv"

    inferencer = load_model(args)

    metrics_rows: List[Dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        src_path = Path(record["source_image"])
        tgt_path = Path(record["target_image"])
        instruction = record["instruction"]

        image = pil_img2rgb(Image.open(src_path))
        result = inferencer(
            image=image,
            text=instruction,
            think=args.think,
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=args.cfg_img_scale,
            cfg_interval=[args.cfg_interval, 1.0],
            timestep_shift=args.timestep_shift,
            num_timesteps=args.num_timesteps,
            cfg_renorm_min=args.cfg_renorm_min,
            cfg_renorm_type=args.cfg_renorm_type,
        )

        thinking_text = result.get("text") if args.think else None
        out = result["image"]
        if not isinstance(out, Image.Image):
            out = Image.fromarray(out)

        try:
            rel = src_path.relative_to(input_root)
            out_dir = output_root / rel.parent
        except ValueError:
            out_dir = output_root / src_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.output_name
        out.save(out_path)

        metrics = {}
        if tgt_path.exists():
            metrics = _compute_metrics(out, Image.open(tgt_path))
        metrics_rows.append(
            {
                "index": idx,
                "source_image": str(src_path),
                "target_image": str(tgt_path),
                "output_image": str(out_path),
                "instruction": instruction,
                "thinking": thinking_text,
                **metrics,
            }
        )

        prefix = f"[{worker_id + 1}/{total_workers}] " if worker_id is not None else ""
        print(f"{prefix}[{idx}/{len(records)}] {src_path} -> {out_path}")

    if worker_id is None:
        result_df = pd.DataFrame(metrics_rows)
        result_df.to_csv(metrics_csv, index=False)
        print(f"Wrote metrics -> {metrics_csv}")
        if args.metrics_xlsx:
            xlsx_path = Path(args.metrics_xlsx)
            xlsx_path.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_excel(xlsx_path, index=False)
            print(f"Wrote metrics -> {xlsx_path}")
        return None

    worker_csv = output_root / f"metrics_worker_{worker_id}.csv"
    pd.DataFrame(metrics_rows).to_csv(worker_csv, index=False)
    return worker_csv


def _worker_entry(
    records: List[Dict[str, str]],
    args: argparse.Namespace,
    worker_id: int,
    total_workers: int,
) -> Optional[str]:
    args = copy.deepcopy(args)
    args.device_ids = str(args.device_id)
    return str(run_inference(records, args, worker_id=worker_id, total_workers=total_workers))


def main() -> None:
    args = build_parser().parse_args()
    args.ckpt = _resolve_ckpt(args.base_model, args.ckpt)
    records = load_jsonl(Path(args.jsonl))
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(records)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    device_ids = [int(x) for x in args.device_ids.split(",") if x.strip()]
    if args.data_parallel and len(device_ids) > 1:
        chunks = [records[i::len(device_ids)] for i in range(len(device_ids))]
        worker_args = []
        for worker_id, (device_id, chunk) in enumerate(zip(device_ids, chunks)):
            worker_ns = copy.deepcopy(args)
            worker_ns.device_id = device_id
            worker_args.append((chunk, worker_ns, worker_id, len(device_ids)))

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(device_ids)) as pool:
            worker_csvs = pool.starmap(_worker_entry, worker_args)

        output_root = Path(args.output_root)
        metrics_csv = Path(args.metrics_csv) if args.metrics_csv else output_root / "metrics.csv"
        worker_csv_paths = [Path(p) for p in worker_csvs if p]
        if worker_csv_paths:
            merged = pd.concat([pd.read_csv(p) for p in worker_csv_paths], ignore_index=True)
            merged.to_csv(metrics_csv, index=False)
            print(f"Wrote metrics -> {metrics_csv}")
            if args.metrics_xlsx:
                xlsx_path = Path(args.metrics_xlsx)
                xlsx_path.parent.mkdir(parents=True, exist_ok=True)
                merged.to_excel(xlsx_path, index=False)
                print(f"Wrote metrics -> {xlsx_path}")
    else:
        run_inference(records, args)


if __name__ == "__main__":
    main()
