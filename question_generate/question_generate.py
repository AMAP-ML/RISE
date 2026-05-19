import vllm
import torch
from transformers import AutoTokenizer
import argparse
from typing import List
from vllm.outputs import RequestOutput
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Evaluation.datasets_loader import get_dataset_handler
from verl.utils.vllm_utils import VLLMHijack
import json
import regex as re
import os
import random
from datasets import Dataset
import base64
from io import BytesIO
from PIL import Image
STORAGE_PATH = os.getenv("STORAGE_PATH")
QUESTION_GENERATE_SHUFFLE = os.getenv("QUESTION_GENERATE_SHUFFLE", "1") == "1"
QUESTION_GENERATE_SHUFFLE_SEED = int(os.getenv("QUESTION_GENERATE_SHUFFLE_SEED", "42"))

ALLOWED_SKILLS = [
    "coarse perception",
    "fine-grained perception",
    "instance reasoning",
    "logical reasoning",
    "math & counting",
    "science & technology",
]
SKILL_ALIASES = {
    "coarse perception": "coarse perception",
    "fine grained perception": "fine-grained perception",
    "fine-grained perception": "fine-grained perception",
    "instance reasoning": "instance reasoning",
    "logical reasoning": "logical reasoning",
    "math": "math & counting",
    "math & counting": "math & counting",
    "math and counting": "math & counting",
    "science & technology": "science & technology",
    "science and technology": "science & technology",
}

SKILL_GUIDELINES_TEXT = """
Skill definitions (use these as the decision boundary):
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

Important constraint:
- For all non-`math & counting` skills, do NOT generate a question whose primary solution depends on counting, estimating quantities, summing values, or approximating the number of objects, regions, or parts.
"""


def normalize_skill_label(skill):
    if skill is None:
        return None
    normalized = str(skill).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized)


def resolve_datasets_cache_dir():
    datasets_cache_dir = os.environ.get("HF_DATASETS_CACHE")
    if not datasets_cache_dir:
        user_name = os.environ.get("USER") or os.environ.get("LOGNAME") or "user"
        datasets_cache_dir = f"/tmp/{user_name}/hf_datasets_cache"
        os.environ["HF_DATASETS_CACHE"] = datasets_cache_dir
    os.makedirs(datasets_cache_dir, exist_ok=True)
    print(f"Using HF_DATASETS_CACHE: {datasets_cache_dir}")
    return datasets_cache_dir

def load_vqa_dataset(data_path, max_samples=None):
    """Load VQA dataset from parquet or arrow files"""
    from datasets import load_from_disk, concatenate_datasets
    datasets_cache_dir = resolve_datasets_cache_dir()
    
    # First try to load as a HuggingFace dataset directory (with arrow files)
    try:
        combined_dataset = load_from_disk(data_path)
        print(f"Loaded dataset from disk: {len(combined_dataset)} samples")
    except Exception:
        # Fall back to loading individual parquet/arrow files
        datasets = []
        data_files = [f for f in os.listdir(data_path) if f.endswith('.parquet') or f.endswith('.arrow')]
        data_files.sort()  # Sort to ensure consistent order
        
        if not data_files:
            raise FileNotFoundError(f"No .parquet or .arrow files found in {data_path}")
        
        for data_file in data_files:
            print(f"Loading {data_file}...")
            file_path = os.path.join(data_path, data_file)
            if data_file.endswith('.parquet'):
                dataset = Dataset.from_parquet(file_path, cache_dir=datasets_cache_dir)
            else:  # .arrow
                dataset = Dataset.from_file(file_path)
            datasets.append(dataset)
        
        # Concatenate all datasets
        if len(datasets) > 1:
            combined_dataset = concatenate_datasets(datasets)
        else:
            combined_dataset = datasets[0]
    
    if max_samples:
        combined_dataset = combined_dataset.select(range(min(max_samples, len(combined_dataset))))

    if QUESTION_GENERATE_SHUFFLE and len(combined_dataset) > 0:
        print(f"Shuffling dataset with seed={QUESTION_GENERATE_SHUFFLE_SEED}")
        combined_dataset = combined_dataset.shuffle(seed=QUESTION_GENERATE_SHUFFLE_SEED)
    
    print(f"Total samples loaded: {len(combined_dataset)}")
    return combined_dataset

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

import math
from PIL import Image
from io import BytesIO

def process_image_for_vllm(image, max_pixels: int = 4194304, min_pixels: int = 262144):
    """
    Process image for vLLM multi-modal input, following ImageProcessMixin pattern.

    Args:
        image: PIL.Image.Image or dict with key "bytes" or raw bytes.
        max_pixels: maximum allowed total pixels (width * height). If exceeded, image will be downscaled.
        min_pixels: minimum allowed total pixels. If smaller, image will be upscaled.

    Returns:
        PIL.Image.Image in RGB mode, resized to be within [min_pixels, max_pixels].
    """
    # Accept dict / bytes / PIL.Image
    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Ensure image is loaded
    image.load()

    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.width, image.height
    total_pixels = width * height

    # Downscale if too large
    if total_pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / float(total_pixels))
        new_w = max(1, int(width * resize_factor))
        new_h = max(1, int(height * resize_factor))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # Upscale if too small
    elif total_pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / float(total_pixels))
        new_w = max(1, int(width * resize_factor))
        new_h = max(1, int(height * resize_factor))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    return image


def extract_boxed(text):
    results, i = [], 0
    prefix = r'\boxed{'
    plen = len(prefix)

    while True:
        start = text.find(prefix, i)
        if start == -1:
            break   # no more \boxed{…}

        j = start + plen
        depth = 1
        while j < len(text) and depth:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1

        results.append(text[start + plen : j - 1])
        i = j

    return results

def get_response_mask(response_ids, eos_token_id, dtype):
    batch_size, seq_len = response_ids.shape
    mask = torch.ones((batch_size, seq_len), dtype=dtype)
    for i in range(batch_size):
        for j in range(seq_len):
            if response_ids[i][j] == eos_token_id:
                mask[i][j:] = 0
                break
    return mask


def engine_is_dead(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "engine core" in message
        or "engine died" in message
        or "enginecore encountered an issue" in message
        or "shutting down" in message
    )


def is_addr_in_use_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "eaddrinuse" in message or "address already in use" in message


def build_vllm_model(args):
    last_exc = None
    for attempt in range(3):
        try:
            return vllm.LLM(
                model=args.model,
                tokenizer=args.model,
                gpu_memory_utilization=args.gpu_mem_util,
                max_model_len=args.max_model_len,
                seed=int(args.suffix),
            )
        except Exception as exc:
            last_exc = exc
            if not is_addr_in_use_error(exc) or attempt == 2:
                raise
            wait_s = 2 + attempt
            print(f"[vllm-init-retry] address already in use, retrying in {wait_s}s (attempt {attempt + 1}/3)")
            import time
            time.sleep(wait_s)
    raise last_exc


def build_sample_indices(total_samples, start_index, num_samples, suffix):
    requested = max(0, int(num_samples))
    if total_samples <= 0 or requested <= 0:
        return [], "empty"

    if start_index < total_samples:
        end_index = min(total_samples, start_index + requested)
        print(f"Using contiguous slice [{start_index}, {end_index}) from dataset of size {total_samples}")
        return list(range(start_index, end_index)), "slice"

    seed_material = f"{QUESTION_GENERATE_SHUFFLE_SEED}:{start_index}:{suffix}"
    rng = random.Random(seed_material)
    all_indices = list(range(total_samples))
    if total_samples >= requested:
        sampled_indices = rng.sample(all_indices, requested)
        replacement = False
    else:
        sampled_indices = [rng.choice(all_indices) for _ in range(requested)]
        replacement = True

    print(
        f"start_index ({start_index}) is out of range for dataset size {total_samples}. "
        f"Fallback to random sampling {len(sampled_indices)} items from the dataset "
        f"(replacement={replacement}, seed='{seed_material}')."
    )
    return sampled_indices, "random"

def main(args):
    # breakpoint()
    VLLMHijack.hijack()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = build_vllm_model(args)
    
    # Load VQA dataset
    print(f"Loading VQA dataset from {args.data_path}...")
    vqa_dataset = load_vqa_dataset(args.data_path, max_samples=args.max_samples)
    total_samples = len(vqa_dataset)
    start_index = max(0, int(args.start_index))
    sample_indices, sampling_mode = build_sample_indices(
        total_samples=total_samples,
        start_index=start_index,
        num_samples=args.num_samples,
        suffix=args.suffix,
    )
    
    # Process each sample in the dataset
    results = []
    target_count = len(sample_indices)
    print(f"Question generation mode: {sampling_mode}, target_count={target_count}")
    for offset, i in enumerate(sample_indices):
        sample = vqa_dataset[i]
            
        print(f"Processing sample {offset+1}/{target_count} (dataset_idx={i})")
        
        # Process image for vLLM and convert to base64 for storage
        processed_image = process_image_for_vllm(sample['images'])
        image_base64 = image_to_base64(processed_image)
        
        # Create prompt for qwen2.5-VL model
        system_prompt = f"""You are an intelligent Question Generator. Your task is to create a **difficult** visual reasoning question based on the given image.

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
                        {SKILL_GUIDELINES_TEXT}
                        5. Choose the question type from **only one** of the following:
                        - `multiple choice` (**must** have four options labeled A, B, C, D; only one correct answer)
                        - `numerical` (requires a specific numeric answer)
                        - `regression` (requires predicting a continuous value, such as a measurement, quantity, or coordinate)
                        6. The question must require analysis or reasoning, not just description.
                        7. The answer should be short, unique, and verifiable.
                        8. **Output must be strictly in this format, with nothing else:**
                        9. Skill must be **only one** of the six classes listed above.
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
                        <answer>B</answer>"""

        user_question = "Generate one new, challenging reasoning question based on this image. Remember to format the output exactly as instructed."
        
        # Create prompt in qwen2.5-VL format
        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{user_question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        sample_params = vllm.SamplingParams(
            max_tokens=2048,
            temperature=1.0,
            top_p=0.95,
            n=1,
            stop_token_ids=[tokenizer.eos_token_id],
        )

        # Generate response for this sample
        # Prepare valid chat with prompt and processed image
        valid_chat = {
            "prompt": prompt,
            "multi_modal_data": {"image": processed_image}
        }
        
        try:
            completions: List[RequestOutput] = model.generate(
                [valid_chat],
                sampling_params=sample_params
            )
        except Exception as e:
            error_payload = {
                "dataset_idx": i,
                "declared_skill": "error",
                "question_type": "generation_error",
                "question": "",
                "answer": "",
                "image": image_base64,
                "error": str(e),
                "image_size": {
                    "width": processed_image.width,
                    "height": processed_image.height,
                },
            }
            print(f"Generation failed for dataset_idx={i}: {e}")
            results.append(error_payload)
            if engine_is_dead(e):
                print("vLLM engine died during question generation. Saving partial results and stopping early.")
                break
            continue
        
        for completion in completions:
            response = completion.outputs[0].text
            try:
                # Extract question type, question, and answer
                skills = re.findall(r"<skill>(.*?)</skill>", response, re.DOTALL)
                question_types = re.findall(r"<type>(.*?)</type>", response, re.DOTALL)
                questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
                answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)
                declared_skill = normalize_skill_label(skills[-1]) if skills else None

                if questions and answers:
                    question_type = question_types[-1].strip() if question_types else "unknown"
                    question = questions[-1].strip()
                    answer = answers[-1].strip()
                    results.append({
                        "declared_skill": declared_skill or "unknown",
                        "question_type": question_type,
                        "question": question,
                        "answer": answer,
                        "image": image_base64
                    })
                else:
                    results.append({
                        "declared_skill": declared_skill or "unknown",
                        "question_type": "unknown",
                        "question": response,
                        "answer": "",
                        "image": image_base64
                    })
            except Exception as e:
                print(f"Error processing response: {e}")
                results.append({
                    "declared_skill": "error",
                    "question_type": "error",
                    "question": response,
                    "answer": "",
                    "image": image_base64
                })
    
    # Save results
    output_file = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Generated {len(results)} questions and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
    parser.add_argument("--data_path", type=str, default="../datasets/parquet/LMMs-Lab-Turtle__Vision-SR1-47K", help="Path to VQA dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--start_index", type=int, default=0, help="Start index in dataset for slicing")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to load from dataset")
    parser.add_argument("--suffix", type=str, default="", help="Suffix to add to the output file")
    parser.add_argument("--save_name", type=str, default="vqa_generated", help="Base name for output file")
    parser.add_argument("--gpu_mem_util", type=float, default=0.8, help="GPU memory utilization passed to vLLM")
    parser.add_argument("--max_model_len", type=int, default=12288, help="Maximum model length passed to vLLM")
    args = parser.parse_args()

    main(args) 
