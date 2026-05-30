"""Task 2 — image description generation with DeepSeek-VL2.

Local GPU inference (open-weights MLLM); descriptions feed the text-only classifier.
Needs the non-PyPI package and a CUDA GPU:
``pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git``.
Output CSV (descriptions.csv): [id, text].

    python inference/deepseek_task2_caption.py --dataset_csv <csv> \\
        --save_path data/deepseek --gpu_id 0
"""

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DESCRIPTION_PROMPT,
    add_io_args,
    build_output_path,
    run_over_dataset,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Task 2 — image description generation with DeepSeek-VL2"
    )
    add_io_args(parser, require_p_version=False)
    parser.add_argument("--model", default="deepseek-ai/deepseek-vl2-small")
    parser.add_argument(
        "--cache_dir",
        default=os.environ.get("HF_HOME"),
        help="HuggingFace cache directory for model weights",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    # Imported lazily so `--help` works without the deepseek_vl2 package installed.
    from deepseek_runtime import DeepSeekVL2

    print(f"Model: {args.model}\nPrompt: {DESCRIPTION_PROMPT}")
    runtime = DeepSeekVL2(model_path=args.model, cache_dir=args.cache_dir, gpu_id=args.gpu_id)

    df = pd.read_csv(args.dataset_csv)
    print(f"Loaded {len(df)} samples from {args.dataset_csv}")
    output_file = build_output_path(
        args.save_path, args.alpha_version, args.p_version, kind="caption"
    )

    def infer(image_path: str) -> str:
        return runtime.generate(image_path, DESCRIPTION_PROMPT, max_new_tokens=args.max_new_tokens)

    def build_record(image_id: str, raw: str) -> dict:
        return {"id": image_id, "text": raw}

    run_over_dataset(
        df,
        args.image_dir,
        output_file,
        infer,
        build_record,
        resume=args.resume,
        limit=args.limit,
        desc="DeepSeek-VL2 Task 2",
    )


if __name__ == "__main__":
    main()
