from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from generation_inference import REPO_ROOT, _resolve_ckpt, load_model
from training.data.data_utils import pil_img2rgb


DEFAULT_SAMPLE_DIR = REPO_ROOT / "data_pipeline/demo_level_samples/sample_understanding"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TimeOmni-VL understanding inference on an image QA sample.")
    parser.add_argument(
        "--base_model",
        default=str(REPO_ROOT / "checkpoint/model-checkpoint"),
        help="Model directory containing configs/tokenizer/checkpoint files.",
    )
    parser.add_argument("--ckpt", default=None, help="Checkpoint path. Defaults to model.safetensors under base_model.")
    parser.add_argument("--image", default=str(DEFAULT_SAMPLE_DIR / "image_full.png"), help="Input image path.")
    parser.add_argument(
        "--qa-json",
        default=str(DEFAULT_SAMPLE_DIR / "qa_pairs.json"),
        help="QA JSON containing conversations. The first human message is used by default.",
    )
    parser.add_argument(
        "--qa-index",
        type=int,
        default=0,
        help="Zero-based index among human QA prompts in --qa-json.",
    )
    parser.add_argument("--prompt", default=None, help="Override prompt text instead of reading --qa-json.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "eval/outputs/understanding_demo"),
        help="Directory for answer.txt and result.json.",
    )
    parser.add_argument("--device-ids", default="0", help="Comma-separated CUDA device ids for model sharding.")
    parser.add_argument("--mode", type=int, default=1, help="1=bf16, 2=4bit NF4, 3=8bit.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--text-temperature", type=float, default=0.3)
    parser.add_argument("--think", action="store_true", help="Enable <think> output mode.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def set_seed(seed: int) -> None:
    if seed <= 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _load_qa_prompt(qa_json: Path, qa_index: int) -> Tuple[str, Optional[str], Dict[str, Any]]:
    data = json.loads(qa_json.read_text(encoding="utf-8"))
    conversations: List[Dict[str, Any]] = data.get("conversations", [])
    human_turns = [(idx, turn) for idx, turn in enumerate(conversations) if turn.get("from") == "human"]
    if not human_turns:
        raise ValueError(f"No human QA turns found in {qa_json}")
    if qa_index < 0 or qa_index >= len(human_turns):
        raise IndexError(f"--qa-index {qa_index} out of range; found {len(human_turns)} human QA turns.")

    turn_idx, human_turn = human_turns[qa_index]
    prompt = str(human_turn.get("value", ""))
    reference_answer: Optional[str] = None
    for next_turn in conversations[turn_idx + 1 :]:
        if next_turn.get("from") == "gpt":
            reference_answer = str(next_turn.get("value", ""))
            break
    return prompt, reference_answer, data


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    args.ckpt = _resolve_ckpt(args.base_model, args.ckpt)

    image_path = Path(args.image)
    qa_json = Path(args.qa_json)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.prompt is None:
        prompt, reference_answer, qa_data = _load_qa_prompt(qa_json, args.qa_index)
    else:
        prompt = args.prompt
        reference_answer = None
        qa_data = {}

    inferencer = load_model(args)
    image = pil_img2rgb(Image.open(image_path))
    result = inferencer(
        image=image,
        text=prompt,
        think=args.think,
        understanding_output=True,
        max_think_token_n=args.max_new_tokens,
        do_sample=args.do_sample,
        text_temperature=args.text_temperature,
    )
    answer = result.get("text") or ""

    result_payload = {
        "image": str(image_path),
        "qa_json": str(qa_json),
        "qa_index": args.qa_index,
        "prompt": prompt,
        "answer": answer,
        "reference_answer": reference_answer,
        "sample_id": qa_data.get("id"),
    }
    (output_root / "answer.txt").write_text(answer + "\n", encoding="utf-8")
    (output_root / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(answer)
    print(f"Wrote answer -> {output_root / 'answer.txt'}")
    print(f"Wrote result -> {output_root / 'result.json'}")


if __name__ == "__main__":
    main()
