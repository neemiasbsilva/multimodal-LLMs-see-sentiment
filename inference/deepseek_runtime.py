"""
DeepSeek-VL2 runtime wrapper (local GPU inference).

Isolates the non-PyPI ``deepseek_vl2`` dependency so it is only imported by the
DeepSeek task scripts — never by ``common.py`` or the OpenAI/Gemini scripts.
Adapted from the reference implementation at ``../deepseek/src/deepseek_inference.py``.

Requires the DeepSeek-VL2 package and a CUDA GPU:
    pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images


class DeepSeekVL2:
    """Loads DeepSeek-VL2 once and generates text for an (image, prompt) pair."""

    def __init__(
        self,
        model_path: str = "deepseek-ai/deepseek-vl2-small",
        cache_dir: str | None = None,
        gpu_id: int = 0,
    ):
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            device = f"cuda:{gpu_id}"
        else:
            device = "cpu"

        self.processor = DeepseekVLV2Processor.from_pretrained(model_path, cache_dir=cache_dir)
        self.tokenizer = self.processor.tokenizer

        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, cache_dir=cache_dir, trust_remote_code=True
        )
        self.model = model.to(torch.bfloat16).to(device).eval()
        self.device = device

    @torch.no_grad()
    def generate(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        conversation = [
            {
                "role": "<|User|>",
                "content": f"This is an image: <image>\n{prompt}",
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt="",
        ).to(self.model.device)

        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        inputs_embeds, past_key_values = self.model.incremental_prefilling(
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            chunk_size=512,
        )
        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=prepare_inputs.input_ids,
            images=prepare_inputs.images,
            images_seq_mask=prepare_inputs.images_seq_mask,
            images_spatial_crop=prepare_inputs.images_spatial_crop,
            attention_mask=prepare_inputs.attention_mask,
            past_key_values=past_key_values,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

        answer = self.tokenizer.decode(
            outputs[0][len(prepare_inputs.input_ids[0]):].cpu().tolist(),
            skip_special_tokens=False,
        )
        return answer.replace("<｜end▁of▁sentence｜>", "").strip()
