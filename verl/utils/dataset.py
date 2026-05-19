# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import re
import ast
import random
from glob import glob
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
import base64
from io import BytesIO

DEBUG_BATCH_PROMPT_SAMPLES = int(os.getenv("DEBUG_BATCH_PROMPT_SAMPLES", "0"))
QUESTIONER_MASK_SOURCE_QA = os.getenv("QUESTIONER_MASK_SOURCE_QA", "0") == "1"
RISE_TRAINING_ROLE = os.getenv("RISE_TRAINING_ROLE", "").strip().lower()

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    debug_prompts = [feature.pop("_debug_prompt", None) for feature in features]
    tensors = {}
    non_tensors = {}
    batch_size = len(features)

    tensor_keys = set()
    all_keys = set()
    for feature in features:
        all_keys.update(feature.keys())
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensor_keys.add(key)

    # Tensor keys must exist for every sample in the batch.
    for key in tensor_keys:
        values = []
        for feature in features:
            if key not in feature:
                raise KeyError(f"Missing tensor key in collate_fn: {key}")
            values.append(feature[key])
        tensors[key] = torch.stack(values, dim=0)

    # Non-tensor keys are aligned to batch size by padding missing entries with None.
    for key in sorted(all_keys - tensor_keys):
        values = [feature.get(key, None) for feature in features]
        if len(values) != batch_size:
            raise RuntimeError(f"Non-tensor key {key} length {len(values)} != batch size {batch_size}")
        non_tensors[key] = np.array(values, dtype=object)

    valid_debug_prompts = [prompt for prompt in debug_prompts if prompt]
    if DEBUG_BATCH_PROMPT_SAMPLES > 0 and valid_debug_prompts:
        sample_count = min(DEBUG_BATCH_PROMPT_SAMPLES, len(valid_debug_prompts))
        sampled_prompts = random.sample(valid_debug_prompts, sample_count)
        for prompt_idx, sampled_prompt in enumerate(sampled_prompts, 1):
            print(f"\n[DEBUG batch prompt {prompt_idx}/{sample_count}]\n{sampled_prompt}\n")

    return {**tensors, **non_tensors}

def b64_to_image(b64_str):
    try:
        img_bytes = base64.b64decode(b64_str)
        return Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None

class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        image = b64_to_image(image) if isinstance(image, str) else image
        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        include_options_in_prompt: bool = False,
        force_choice_letter_output: bool = False,
        force_yes_no_output: bool = False,
        option_keys: Tuple[str, ...] = ("options", "choices"),
        eval_normalize_image_placeholders: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.include_options_in_prompt = include_options_in_prompt
        self.force_choice_letter_output = force_choice_letter_output
        self.force_yes_no_output = force_yes_no_output
        self.option_keys = option_keys
        self.eval_normalize_image_placeholders = eval_normalize_image_placeholders
        self.indexed_image_cols: List[str] = []
        SUBSAMPLE_DATASET_KEY = "Vision-SR1-47K"
        SUBSAMPLE_RATIO = 0.5
        SUBSAMPLE_SEED = 1

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            arrow_files = sorted(glob(os.path.join(data_path, "*.arrow")))
            parquet_files = sorted(glob(os.path.join(data_path, "*.parquet")))
            if arrow_files:
                self.dataset = load_dataset("arrow", data_files=arrow_files, split="train")
            elif parquet_files:
                self.dataset = load_dataset("parquet", data_files=parquet_files, split="train")
            else:
                # fallback to builder-style directory
                self.dataset = load_dataset("parquet", data_dir=data_path, split="train")
        elif os.path.isfile(data_path):
            if data_path.endswith(".arrow"):
                self.dataset = load_dataset("arrow", data_files=data_path, split="train")
            else:
                self.dataset = load_dataset("parquet", data_files=data_path, split="train")
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        # Only when path includes Vision-SR1-47K, keep a fixed random 25% subset to speed up preprocessing.
        if SUBSAMPLE_DATASET_KEY in data_path:
            raw_n = len(self.dataset)
            keep_n = max(1, int(raw_n * SUBSAMPLE_RATIO))
            self.dataset = self.dataset.shuffle(seed=SUBSAMPLE_SEED).select(range(keep_n))
            print(
                "[dataset] Vision-SR1-47K subsample: "
                f"before={raw_n}, after={keep_n}, ratio={SUBSAMPLE_RATIO}, "
                f"seed={SUBSAMPLE_SEED}, path={data_path}"
            )

        # Preserve stable row ids from the original source dataset, so downstream
        # outputs can be joined back to ground truth after filtering/reordering.
        if "orig_row_index" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column("orig_row_index", list(range(len(self.dataset))))

        # Auto-resolve common schema mismatches across datasets.
        self.prompt_key = self._resolve_key(self.prompt_key, ["prompt", "question", "problem"])
        self.answer_key = self._resolve_key(self.answer_key, ["answer", "solution", "final_answer", "response"])
        self.image_key = self._resolve_key(self.image_key, ["images", "image"])
        if self.image_key and self.image_key not in self.dataset.column_names:
            self.indexed_image_cols = self._discover_indexed_image_columns()

        # Early-drop multi-image samples for stable single-image evaluation.
        # This avoids vLLM placeholder/image-count mismatches on MMMU-style rows.
        if self.image_key and (self.image_key in self.dataset.column_names or self.indexed_image_cols):
            raw_count = len(self.dataset)
            self.dataset = self.dataset.filter(
                self._filter_real_multi_images,
                desc="Filtering multi-image samples early (image_count <= 1)",
            )
            kept_count = len(self.dataset)
            dropped_count = raw_count - kept_count
            print(
                "[dataset] Early multi-image filter: "
                f"before={raw_count}, after={kept_count}, dropped_multi={dropped_count}"
            )
        
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()
        if self.image_key and (self.image_key in self.dataset.column_names or self.indexed_image_cols):
            # Drop malformed multimodal rows before vLLM:
            # - image rows must have exactly one <image> placeholder
            # - non-image rows must have zero <image> placeholders
            placeholder_raw_count = len(self.dataset)
            self.dataset = self.dataset.filter(
                self._filter_image_placeholder_consistency,
                desc="Filtering placeholder/image mismatch rows",
            )
            print(
                "[dataset] Placeholder consistency filter: "
                f"before={placeholder_raw_count}, after={len(self.dataset)}, "
                f"dropped={placeholder_raw_count - len(self.dataset)}"
            )
            # Filter out examples with multiple images for MMMU
            self.dataset = self.dataset.filter(self._filter_multiple_images, desc="Filtering multiple images")

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

        self.dataset = self.dataset.add_column("dataset_index", list(range(len(self.dataset))))

    def _is_valid_image_item(self, image: Any) -> bool:
        if image is None:
            return False
        if isinstance(image, dict):
            return image.get("bytes") is not None or bool(image.get("path"))
        return True

    def _has_images(self, example: Dict[str, Any]) -> bool:
        if not self.image_key:
            return False
        if self.image_key in example:
            raw_images = example[self.image_key]
            if raw_images is None:
                return False
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            return any(self._is_valid_image_item(image) for image in raw_images)
        if self.indexed_image_cols:
            return any(self._is_valid_image_item(example.get(col)) for col in self.indexed_image_cols)
        return False

    def _extract_raw_images(self, example: Dict[str, Any]) -> List[Any]:
        if self.image_key in example:
            raw_images = example[self.image_key]
            if raw_images is None:
                return []
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            return [image for image in raw_images if self._is_valid_image_item(image)]
        if self.indexed_image_cols:
            return [example.get(col) for col in self.indexed_image_cols if self._is_valid_image_item(example.get(col))]
        return []

    def _count_valid_images(self, example: Dict[str, Any]) -> int:
        return len(self._extract_raw_images(example))

    def _filter_real_multi_images(self, example: Dict[str, Any]) -> bool:
        # Keep text-only and single-image rows.
        return self._count_valid_images(example) <= 1

    def _discover_indexed_image_columns(self) -> List[str]:
        return sorted(
            [col for col in self.dataset.column_names if re.fullmatch(r"image_\d+", col)],
            key=lambda x: int(x.split("_")[1]),
        )

    def _normalize_prompt_image_tokens(self, prompt: str, has_images: bool) -> str:
        img_token = "<image>"
        # Keep existing normalization for "<image 1>" style placeholders.
        prompt = re.sub(r"<image\s+\d+>", img_token, prompt)
        # Optional eval-only extension to cover "<image1>" / "<image_1>" datasets.
        if self.eval_normalize_image_placeholders:
            prompt = re.sub(r"<image[_\s]*\d+>", img_token, prompt)

        if has_images and (prompt.count(img_token) == 1):
            prompt = prompt.replace(img_token, "").rstrip()
            prompt = prompt + img_token
        elif has_images and (prompt.count(img_token) == 0):
            prompt = prompt.replace(img_token, "")
            if not prompt.lstrip().startswith(img_token):
                prompt = prompt + img_token

        return prompt

    def _resolve_key(self, preferred: Optional[str], candidates: List[str]) -> Optional[str]:
        if not preferred:
            return preferred
        if preferred in self.dataset.column_names:
            return preferred
        for key in candidates:
            if key in self.dataset.column_names:
                return key
        return preferred

    def _mask_source_qa_for_questioner(self) -> bool:
        return QUESTIONER_MASK_SOURCE_QA and RISE_TRAINING_ROLE == "questioner"

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = self._normalize_prompt_image_tokens(example[self.prompt_key], self._has_images(example))
        mask_source_qa = self._mask_source_qa_for_questioner()
        if (
            mask_source_qa
            and self._has_images(example)
        ):
            prompt_str = "<image>"
        options_text = "" if mask_source_qa else self._extract_options_text(example)
        # Val-only switch from config/registry:
        # when enabled, add a soft MCQ constraint without forcing all tasks to be letter-only.
        use_choice_constraint = False if mask_source_qa else self.force_choice_letter_output
        use_yes_no_constraint = (
            (not mask_source_qa)
            and self.force_yes_no_output
            and not use_choice_constraint
            and self._is_binary_answer(example)
        )

        if self.include_options_in_prompt and options_text:
            options_block = f"\n\nOptions:\n{options_text}\n"
            # Keep options in the text segment before the <image> placeholder.
            # The downstream multimodal conversion returns after the first image token.
            if "<image>" in prompt_str:
                prompt_str = prompt_str.replace("<image>", options_block + "<image>")
            else:
                prompt_str = f"{prompt_str}{options_block}"

        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        if self._has_images(example):
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})
                    return [{"role": "user", "content": content_list}]
                if self.format_prompt and "solver_format" in self.format_prompt:
                    if content:
                        content_list.append({
                            "type": "text",
                            "text": (
                                "Please reason step by step carefully based on the image for the following question: " + content + " "
                                "After completing your reasoning, you MUST output the final, clean, and concise answer "
                                "strictly inside " + r"\\boxed{ }." +
                                "The final answer MUST appear inside \\boxed{}, and nowhere else. "
                                "If there is no boxed answer, your response is considered incorrect. "
                            )
                        })
                else:
                    content_list.append({"type": "text", "text": """
                        You are an intelligent Question Generator. Your task is to create a **difficult** visual reasoning question based on the given image.

                        **Requirements (must follow exactly):**

                        1. Analyze the image carefully and understand all details.
                        2. Generate **exactly one question** that is directly related to the image.
                        3. Choose the skill from **only one** of the following:
                        - `coarse perception`
                        - `fine-grained perception`
                        - `instance reasoning`
                        - `logical reasoning`
                        - `math & counting`
                        - `science & technology`
                        4. Use the following skill definitions and restrictions when choosing the skill:
                        - `coarse perception`
                          Allowed: overall scene type, main objects, global layout, salient entity presence, broad visual category recognition.
                        - `fine-grained perception`
                          Allowed: local details, subtle visual attributes, textures, small text, fine-grained category differences, small part recognition.
                        - `instance reasoning`
                          Allowed: comparing instances, identifying relations between instances, attribute binding, matching an attribute to the correct instance.
                        - `logical reasoning`
                          Allowed: multi-step visual deduction, elimination, conditional reasoning, combining multiple visual cues to infer a conclusion.
                        - `math & counting`
                          Allowed: counting, estimation, arithmetic, geometric or numerical reasoning, approximate quantity judgment.
                        - `science & technology`
                          Allowed: diagrams, charts, scientific illustrations, technical structures, instrument/figure understanding.
                        5. Choose the question type from **only one** of the following:
                        - `multiple choice` (**must** have four options labeled A, B, C, D; only one correct answer)
                        - `numerical` (requires a specific numeric answer)
                        - `regression` (requires predicting a continuous value, such as a measurement, quantity, or coordinate)
                        6. The question must require analysis or reasoning, not just description.
                        7. The answer should be short, unique, and verifiable.
                        8. **Output must be strictly in this format, with nothing else:**
                        9. Skill must be one of the six classes above.
                        10. Question type must be **only one** of: `multiple choice`, `numerical`, `regression`.
                        11. Do **not** add commentary, explanations, or extra text.
                        The following FOUR blocks:
                        <skill>S</skill>
                        <type>X</type>
                        <question>Y</question>
                        <answer>Z</answer>

                        **Example of correct output:**
                        <skill>logical reasoning</skill>
                        <type>multiple choice</type>
                        <question>Which option best explains why the child in the foreground is likely studying rather than relaxing? Options: A. The child is holding a toy. B. The child is surrounded by books and writing materials. C. The child is running outdoors. D. The child is sleeping on the desk.</question>
                        <answer>B</answer>
                        """})
            return [{"role": "user", "content": content_list}]
        else:
            if use_choice_constraint:
                prompt_str += (
                    "\n\nIf this is a multiple-choice question, output only the option letter "
                    "(e.g., A/B/C/D) inside \\boxed{}."
                )
            if use_yes_no_constraint:
                prompt_str += (
                    "\n\nFinal answer must be exactly one word yes or no inside \\boxed{}. "
                    "Do not output anything else in the final boxed answer."
                )
            return [{"role": "user", "content": prompt_str}]

    def _has_choice_markers(self, text: str) -> bool:
        if not text:
            return False
        lower_text = text.lower()
        has_option_header = bool(re.search(r"\boptions?\b|\bchoices?\b", lower_text))

        # Prefer line-style option labels to avoid false positives like "Plan A. ..."
        # Require multiple labels (or explicit options header + one label).
        line_labels = re.findall(r"(?:^|\n)\s*[\(\[]?([a-f])[\)\].:\-]\s+", lower_text)
        inline_labels = re.findall(r"\(([a-f])\)\s+", lower_text)
        labels = line_labels + inline_labels
        unique_labels = set(labels)

        if len(unique_labels) >= 2:
            return True
        if has_option_header and len(unique_labels) >= 1:
            return True
        return False

    def _extract_options_text(self, example: Dict[str, Any]) -> str:
        raw_options = None
        for key in self.option_keys:
            if key in example and example[key] is not None:
                raw_options = example[key]
                break
        if raw_options is None:
            return ""

        option_list: List[str] = []
        if isinstance(raw_options, list):
            option_list = [str(item).strip() for item in raw_options if str(item).strip()]
        elif isinstance(raw_options, str):
            text = raw_options.strip()
            if not text:
                return ""
            parsed = None
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                option_list = [str(item).strip() for item in parsed if str(item).strip()]
            else:
                # Fallback for plain text options, split by line if available.
                if "\n" in text:
                    option_list = [line.strip() for line in text.split("\n") if line.strip()]
                else:
                    option_list = [text]
        else:
            text = str(raw_options).strip()
            if text:
                option_list = [text]

        if not option_list:
            return ""

        formatted = []
        for idx, option in enumerate(option_list):
            label = chr(ord("A") + idx) if idx < 26 else f"Opt{idx + 1}"
            formatted.append(f"{label}. {option}")
        return "\n".join(formatted)

    def _is_binary_answer(self, example: Dict[str, Any]) -> bool:
        answer = example.get(self.answer_key)
        if answer is None:
            return False
        ans = str(answer).strip().lower()
        return ans in {"0", "1", "yes", "no", "true", "false"}

    def _filter_multiple_images(self, example: Dict[str, Any]) -> bool:
        """Filter out examples with more than one <image> tag for MMMU"""
        prompt = self._normalize_prompt_image_tokens(example[self.prompt_key], self._has_images(example))
        img_token = "<image>"
        # Keep only examples with 0 or 1 image tags
        return prompt.count(img_token) <= 1

    def _filter_image_placeholder_consistency(self, example: Dict[str, Any]) -> bool:
        has_images = self._has_images(example)
        prompt = self._normalize_prompt_image_tokens(example[self.prompt_key], has_images)
        image_tag_count = prompt.count("<image>")
        if has_images:
            return image_tag_count == 1
        return image_tag_count == 0

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        prompt = processing_class.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # Drop samples that will require truncation. For image samples, this avoids
        # cases where right truncation removes the trailing <image> placeholder.
        return len(token_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        example["dataset_index"] = int(example["dataset_index"])
        mask_source_qa = self._mask_source_qa_for_questioner()
        if mask_source_qa:
            example["question"] = ""
        else:
            example["question"] = self._normalize_prompt_image_tokens(example[self.prompt_key], self._has_images(example))
        messages = self._build_messages(example)
        if self._has_images(example):
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            raw_images = self._extract_raw_images(example)
            images = [self.process_image(image) for image in raw_images]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images}
            example["multi_modal_inputs"] = dict(model_inputs)
            # ensure images are passed through dataloader collate for reward stage
            example["images"] = images
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # print(f"[DEBUG] Text-only prompt: {prompt}")
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
        
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # For raw_prompt_ids, we need the tokenized prompt with placeholders (not expanded image tokens)
        # vLLM will handle the image token expansion itself based on multi_modal_data
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # DEBUG: Check prompt and raw_prompt_ids length
        # if self.image_key in example and index < 3:  # Only print first 3 samples
        #     print(f"[DEBUG] Sample {index}:")
        #     print(f"[DEBUG] Prompt string (first 200 chars): {prompt[:200]}...")
        #     print(f"[DEBUG] Prompt string (last 200 chars): ...{prompt[-200:]}")
        #     print(f"[DEBUG] raw_prompt_ids length: {len(raw_prompt_ids)}")
        #     print(f"[DEBUG] input_ids length (after processor): {len(input_ids)}")
        
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["_debug_prompt"] = prompt
        source_answer = example.pop(self.answer_key)
        example["ground_truth"] = "" if mask_source_qa else source_answer
        return example
