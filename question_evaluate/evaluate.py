#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description:
    This script evaluates generated answers against golden answers for a set of questions.
    It uses vLLM for efficient generation and a robust, timed grading mechanism to score the results.
    The script is designed to run as a batch job, often in parallel across multiple GPUs.

Refactoring Notes:
    - Replaced 'timeout-decorator' with the thread-safe 'stopit' library to provide robust
      timeout protection for the grading function without causing errors.
    - Optimized the answer comparison logic to perform cheap checks first, only calling the
      expensive grading function when necessary.
    - Improved error handling and code structure for better readability and stability.

Setup:
    pip install stopit transformers torch vllm

Example Usage (in a shell script):
    # This would run the script for GPU 0, with a specific model and save name.
    CUDA_VISIBLE_DEVICES=0 python evaluate.py --model "Qwen/Qwen3-4B-Base" --suffix 0 --save_name "my_experiment" &
'''

import json
import vllm
from transformers import AutoTokenizer
import argparse
import os
import time
import re
from datetime import datetime
import stopit  # Use the robust, thread-safe stopit library for timeouts
from mathruler.grader import extract_boxed_content, grade_answer
import base64
from io import BytesIO
from PIL import Image
import math
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verl.utils.vllm_utils import VLLMHijack

SUPERVISOR_VALIDITY_ENABLED = os.getenv("SUPERVISOR_VALIDITY_ENABLED", "1") == "1"
SUPERVISOR_ANSWER_ENABLED = os.getenv("SUPERVISOR_ANSWER_ENABLED", "1") == "1"
SUPERVISOR_MIN_SCORE = float(os.getenv("SUPERVISOR_MIN_SCORE", "0.3"))
SUPERVISOR_MAX_SCORE = float(os.getenv("SUPERVISOR_MAX_SCORE", "0.8"))

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
SKILL_CONTEXTS = {
    "coarse perception": (
        "Skill definition for `coarse perception`:\n"
        "Allowed: overall scene type, main objects, global layout, salient entity presence, broad visual category recognition.\n"
        "Forbidden: any question whose primary solution relies on counting, estimating quantity, totaling, or approximating numbers.\n"
    ),
    "fine-grained perception": (
        "Skill definition for `fine-grained perception`:\n"
        "Allowed: local details, subtle visual attributes, textures, small text, fine-grained category differences, small part recognition.\n"
        "Forbidden: counting small parts, estimating the number of segments/pieces/regions, or any quantity-focused question.\n"
    ),
    "instance reasoning": (
        "Skill definition for `instance reasoning`:\n"
        "Allowed: comparing instances, identifying relations between instances, attribute binding, matching an attribute to the correct instance.\n"
        "Forbidden: solving mainly by counting instances or estimating how many instances satisfy a condition.\n"
    ),
    "logical reasoning": (
        "Skill definition for `logical reasoning`:\n"
        "Allowed: multi-step visual deduction, elimination, conditional reasoning, combining multiple visual cues to infer a conclusion.\n"
        "Forbidden: questions whose main reasoning path is counting, estimation, arithmetic, or quantity aggregation.\n"
    ),
    "math & counting": (
        "Skill definition for `math & counting`:\n"
        "Allowed: counting, estimation, arithmetic, geometric or numerical reasoning, approximate quantity judgment.\n"
    ),
    "science & technology": (
        "Skill definition for `science & technology`:\n"
        "Allowed: diagrams, charts, scientific illustrations, technical structures, instrument or figure understanding.\n"
        "Forbidden: if the question is mainly about counting parts or estimating quantities, it should not be labeled as this skill.\n"
    ),
}

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Evaluate generated questions using vLLM.")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model in Hugging Face format.")
parser.add_argument("--num_samples", type=int, default=9, help="Number of candidate answers to generate per question (n).")
parser.add_argument("--suffix", type=str, default="0", help="A unique suffix for file naming, often the GPU index.")
parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
parser.add_argument("--gpu_mem_util", type=float, default=0.85, help="GPU memory utilization passed to vLLM.")
parser.add_argument("--max_model_len", type=int, default=12288, help="Maximum model length passed to vLLM.")
parser.add_argument("--batch_size", type=int, default=256, help="Maximum number of prompts per vLLM generate batch.")
parser.add_argument("--max_pixels", type=int, default=2097152, help="Maximum pixels for each image before feeding vLLM.")
parser.add_argument("--min_pixels", type=int, default=262144, help="Minimum pixels for each image before feeding vLLM.")
args = parser.parse_args()

# --- Constants and Paths ---
STORAGE_PATH = os.getenv("STORAGE_PATH", "../storage_RISE_Qwen3-VL-8B")
INPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}.json"
OUTPUT_FILE = f"{STORAGE_PATH}/generated_question/{args.save_name}_{args.suffix}_results.json"

# --- Timeout-Protected Grading Function ---
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    Wraps the mathruler 'grade_answer' function with a timeout.
    If the function takes too long, it returns 'TIMED_OUT' instead of hanging.
    """
    # The actual timeout value is passed as a keyword argument on each call.
    return grade_answer(res1, res2)


def process_image_for_vllm(image, max_pixels: int, min_pixels: int):
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    image.load()
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.width, image.height
    total_pixels = width * height

    if total_pixels > max_pixels:
        resize_factor = math.sqrt(max_pixels / float(total_pixels))
        new_w = max(1, int(width * resize_factor))
        new_h = max(1, int(height * resize_factor))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    elif total_pixels < min_pixels:
        resize_factor = math.sqrt(min_pixels / float(total_pixels))
        new_w = max(1, int(width * resize_factor))
        new_h = max(1, int(height * resize_factor))
        image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    return image


def get_image_size(image):
    if image is None:
        return None
    return {"width": int(image.width), "height": int(image.height)}


def normalize_skill_label(skill):
    if skill is None:
        return None
    normalized = str(skill).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized)


def get_skill_context(declared_skill):
    normalized = normalize_skill_label(declared_skill)
    if normalized in SKILL_CONTEXTS:
        return SKILL_CONTEXTS[normalized]
    return (
        "Skill definition unavailable because the declared skill is not one of the six valid classes.\n"
    )


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
            time.sleep(wait_s)
    raise last_exc


def extract_boxed_binary(text):
    boxed = extract_boxed_content(text or "")
    if boxed is None:
        return 0
    normalized = str(boxed).strip().lower()
    if normalized in {"1", "yes", "true", "correct"}:
        return 1
    if normalized in {"0", "no", "false", "incorrect"}:
        return 0
    return 0


def extract_skill_match(text, final_valid):
    if not text:
        return final_valid
    patterns = [
        r"skill\s*match\s*[:：]\s*([01])",
        r"declared\s*skill\s*correct\s*[:：]\s*([01])",
        r"skill\s*is\s*correct\s*[:：]\s*([01])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return final_valid


def build_validity_prompt(question, declared_skill):
    skill_context = get_skill_context(declared_skill)
    return (
        "<|im_start|>system\n"
        "You are a strict visual question validity judge. "
        "Only decide whether the question can be answered solely from the provided image and whether the declared skill matches the question. "
        "Do not solve the question. "
        "Use the following skill definition and restriction for the declared skill when judging skill correctness:\n"
        f"{skill_context}"
        "You may first give a brief reason, then you must output a line in the form 'Skill Match: 1' or 'Skill Match: 0'. "
        "Finally, you must put the final decision inside \\boxed{} exactly once at the end. "
        "Output \\boxed{1} only if the question is image-grounded, well-posed, answerable from the image alone, and the declared skill is correct. "
        "Output \\boxed{0} otherwise.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"Question: {question}\n"
        f"Declared Skill: {declared_skill}\n"
        "The only valid skill classes are: coarse perception; fine-grained perception; instance reasoning; logical reasoning; math & counting; science & technology.\n"
        "Judge whether this question is image-grounded, well-posed, and answerable from the image alone, and whether the declared skill matches the question. "
        "If either condition fails, output \\boxed{0}. Do not solve the question. "
        "You may briefly explain why, then output 'Skill Match: 1' or 'Skill Match: 0', and end with \\boxed{1} or \\boxed{0}.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# --- Main Script Logic ---

# 1. Load and Prepare Data
print(f"[{args.suffix}] Loading data from: {INPUT_FILE}")
try:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    # Clean up the input file immediately after loading to save space
    os.remove(INPUT_FILE)
except FileNotFoundError:
    print(f"[{args.suffix}] ERROR: Input file not found. Exiting.")
    exit()

# Adapt to input format: each item provides question metadata plus image (base64)
questions = [item.get("question", "") for item in data]
question_types = [item.get("question_type", "") for item in data]
declared_skills = [normalize_skill_label(item.get("declared_skill")) or item.get("declared_skill", "unknown") for item in data]
skill_matches = [int(item.get("skill_match", 0)) for item in data]
validities = [int(item.get("valid", 0)) for item in data]
images_base64 = [item.get("image", "") for item in data]

# Filter out empty questions to avoid unnecessary generation
filtered = [
    (q, qt, ds, sm, vd, img)
    for q, qt, ds, sm, vd, img in zip(
        questions,
        question_types,
        declared_skills,
        skill_matches,
        validities,
        images_base64,
    )
    if q
]
if not filtered:
    print(f"[{args.suffix}] No valid questions found. Exiting.")
    with open(OUTPUT_FILE, "w") as f:
        json.dump([], f)
    exit()

questions, question_types, declared_skills, skill_matches, validities, images_base64 = zip(*filtered)
print(f"[{args.suffix}] Found {len(questions)} questions to process.")

# 2. Initialize Model and Tokenizer
print(f"[{now()}][{args.suffix}] Initializing vLLM for model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
VLLMHijack.hijack()
model = build_vllm_model(args)
sample_params = vllm.SamplingParams(
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.num_samples,
)

judge_sample_params = vllm.SamplingParams(
    max_tokens=512,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    stop_token_ids=[tokenizer.eos_token_id],
    n=1,
)

# 3. Prepare images
print(f"[{now()}][{args.suffix}] Model loaded. Preparing generated questions for evaluation...")

def b64_to_image(b64_str):
    try:
        img_bytes = base64.b64decode(b64_str)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        return process_image_for_vllm(
            image,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
        )
    except Exception:
        return None

images_pil = [b64_to_image(b64) for b64 in images_base64]

# Prepare candidate items with usable images.
candidate_items = []
for img, q, qt, ds, sm, vd, image_b64 in zip(
    images_pil,
    questions,
    question_types,
    declared_skills,
    skill_matches,
    validities,
    images_base64,
):
    if img is not None:
        candidate_items.append({
            "question": q,
            "question_type": qt,
            "declared_skill": ds,
            "skill_match": int(sm),
            "valid": int(vd),
            "image_b64": image_b64,
            "image_pil": img,
            "image_size": get_image_size(img),
        })
print(
    f"[{now()}][{args.suffix}] Image preprocessing kept "
    f"{len(candidate_items)}/{len(questions)} questions with usable images."
)

# 4. Joint validity + skill verification for solver data construction
if SUPERVISOR_VALIDITY_ENABLED:
    validity_indices = []
    validity_chats = []
    for idx, item in enumerate(candidate_items):
        normalized_skill = normalize_skill_label(item["declared_skill"])
        item["declared_skill"] = normalized_skill or str(item["declared_skill"] or "unknown")
        item["skill_match"] = 0
        item["valid"] = 0
        item["validity_reason"] = "skipped_invalid_skill_or_image"
        if item["question"] and item["image_pil"] is not None and normalized_skill in ALLOWED_SKILLS:
            validity_indices.append(idx)
            validity_chats.append({
                "prompt": build_validity_prompt(item["question"], normalized_skill),
                "multi_modal_data": {"image": item["image_pil"]},
            })

    if validity_chats:
        validity_start = time.time()
        print(
            f"[{now()}][{args.suffix}] Running joint validity+skill verification for "
            f"{len(validity_chats)}/{len(candidate_items)} generated questions..."
        )
        validity_responses = model.generate(validity_chats, sampling_params=judge_sample_params, use_tqdm=True)
        for debug_idx, response in enumerate(validity_responses[:3]):
            raw_text = response.outputs[0].text if response.outputs else ""
            print(f"[{args.suffix}] [validity-debug-{debug_idx}] {raw_text}")
        for idx, response in zip(validity_indices, validity_responses):
            raw_text = response.outputs[0].text if response.outputs else ""
            valid = extract_boxed_binary(raw_text)
            skill_match = extract_skill_match(raw_text, final_valid=valid)
            candidate_items[idx]["skill_match"] = skill_match
            candidate_items[idx]["valid"] = valid
            candidate_items[idx]["validity_reason"] = raw_text.strip()
        validity_elapsed = time.time() - validity_start
        validity_pass_count = sum(int(item.get("valid", 0)) == 1 for item in candidate_items)
        print(
            f"[{now()}][{args.suffix}] Joint validity+skill verification kept "
            f"{validity_pass_count}/{len(candidate_items)} questions in {validity_elapsed:.1f}s."
        )
    else:
        print(f"[{now()}][{args.suffix}] No samples eligible for joint validity+skill verification.")

    solver_items = [item for item in candidate_items if int(item.get("valid", 0)) == 1]
    print(
        f"[{now()}][{args.suffix}] Solver answering stage input: "
        f"{len(solver_items)} questions after validity filtering."
    )

    if not solver_items:
        print(f"[{now()}][{args.suffix}] No valid questions remain after joint validity+skill verification.")
        with open(OUTPUT_FILE, "w") as f:
            json.dump([], f, indent=4)
        print(f"[{now()}][{args.suffix}] Saved empty results to: {OUTPUT_FILE}")
        print(f"[{now()}][{args.suffix}] Script finished.")
        exit()
else:
    for item in candidate_items:
        normalized_skill = normalize_skill_label(item["declared_skill"])
        item["declared_skill"] = normalized_skill or str(item["declared_skill"] or "unknown")
        item["skill_match"] = int(item.get("skill_match", 1)) if normalized_skill in ALLOWED_SKILLS else 0
        item["valid"] = 1
        item["validity_reason"] = "skipped_validity_filter_disabled"
    solver_items = candidate_items
    print(
        f"[{now()}][{args.suffix}] Joint validity+skill verification disabled; "
        f"passing {len(solver_items)} questions directly to solver answering."
    )

# 5. Generate solver responses for valid questions only.
solver_gen_start = time.time()
placeholder = "<|image_pad|>"
solver_chats = []
for item in solver_items:
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
        "Please reason step by step carefully based on the image for the following question: "
        f"{item['question']} "
        "After completing your reasoning, you MUST output the final, clean, and concise answer "
        "strictly inside \\boxed{}. "
        "The final answer MUST appear inside \\boxed{}, and nowhere else. "
        "If there is no boxed answer, your response is considered incorrect.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    solver_chats.append({
        "prompt": prompt,
        "multi_modal_data": {"image": item["image_pil"]},
    })

BATCH_SIZE = args.batch_size
total_prompts = len(solver_chats)
print(
    f"[{now()}][{args.suffix}] Starting solver answer generation "
    f"({total_prompts} prompts × {args.num_samples} samples = {total_prompts * args.num_samples} sequences, "
    f"batch_size={BATCH_SIZE})..."
)
responses = []
for batch_start in range(0, total_prompts, BATCH_SIZE):
    batch = solver_chats[batch_start:batch_start + BATCH_SIZE]
    batch_items = solver_items[batch_start:batch_start + BATCH_SIZE]
    batch_end = min(batch_start + BATCH_SIZE, total_prompts)
    print(
        f"[{now()}][{args.suffix}] Generating solver batch "
        f"{batch_start//BATCH_SIZE + 1}/{(total_prompts + BATCH_SIZE - 1)//BATCH_SIZE} "
        f"(prompts {batch_start}-{batch_end-1})..."
    )
    try:
        batch_responses = model.generate(batch, sampling_params=sample_params, use_tqdm=True)
    except Exception as e:
        debug_path = OUTPUT_FILE.replace(
            "_results.json",
            f"_failed_batch_{batch_start}_{batch_end - 1}.json",
        )
        debug_payload = {
            "suffix": args.suffix,
            "model": args.model,
            "batch_start": batch_start,
            "batch_end": batch_end,
            "batch_size": len(batch),
            "items": [
                {
                    "prompt_index": batch_start + offset,
                    "question": item["question"],
                    "question_type": item["question_type"],
                    "image_size": item["image_size"],
                }
                for offset, item in enumerate(batch_items)
            ],
        }
        with open(debug_path, "w") as f:
            json.dump(debug_payload, f, indent=2, ensure_ascii=False)
        print(f"[{now()}][{args.suffix}] Saved failed batch metadata to: {debug_path}")
        if engine_is_dead(e):
            raise
        recovered_responses = []
        for local_idx, single_prompt in enumerate(batch):
            single_item = batch_items[local_idx]
            single_index = batch_start + local_idx
            try:
                single_output = model.generate([single_prompt], sampling_params=sample_params, use_tqdm=False)
                recovered_responses.extend(single_output)
            except Exception as single_exc:
                single_debug_path = OUTPUT_FILE.replace(
                    "_results.json",
                    f"_failed_item_{single_index}.json",
                )
                with open(single_debug_path, "w") as f:
                    json.dump(
                        {
                            "suffix": args.suffix,
                            "model": args.model,
                            "prompt_index": single_index,
                            "question": single_item["question"],
                            "question_type": single_item["question_type"],
                            "image_size": single_item["image_size"],
                            "error": str(single_exc),
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(f"[{now()}][{args.suffix}] Saved failed item metadata to: {single_debug_path}")
                if engine_is_dead(single_exc):
                    raise
                print(f"[{now()}][{args.suffix}] Skipping prompt {single_index}: {single_exc}")
                recovered_responses.append(None)
        batch_responses = recovered_responses
    responses.extend(batch_responses)
solver_gen_elapsed = time.time() - solver_gen_start
print(
    f"[{now()}][{args.suffix}] Solver answer generation complete for "
    f"{len(responses)}/{len(solver_items)} questions in {solver_gen_elapsed:.1f}s ({solver_gen_elapsed/60:.1f}min)."
)

# 6. Process and Grade Responses
results_all = []
grade_start = time.time()
print(f"[{now()}][{args.suffix}] Grading solver responses...")
for response, item in zip(responses, solver_items):
    try:
        if response is None:
            print(f"[{args.suffix}] WARNING: Skipping failed prompt: '{item['question'][:50]}...'")
            continue
        question = item["question"]
        image_b64 = item["image_b64"]
        question_type = item["question_type"]
        # Extract the boxed content from all generated samples
        results = [extract_boxed_content(output.text) for output in response.outputs]
        results = [res for res in results if res] # Filter out None/empty results

        if not results:
            print(f"[{args.suffix}] WARNING: No valid boxed answers found for question: '{question[:50]}...'")
            continue

        answer_counts = {}
        for result in results:
            matched = False
            for existing_answer in answer_counts:
                # OPTIMIZATION: Perform cheap string comparisons first.
                if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                    answer_counts[existing_answer] += 1
                    matched = True
                    break
                
                # If cheap checks fail, use the expensive, timed grader.
                # Check both directions (A vs B and B vs A).
                match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                if match_1 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{result[:30]}...' vs '{existing_answer[:30]}...'")
                    continue # Skip to the next existing_answer
                
                if match_1:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

                match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                if match_2 == 'TIMED_OUT':
                    print(f"[{args.suffix}] GRADER TIMEOUT on: '{existing_answer[:30]}...' vs '{result[:30]}...'")
                    continue

                if match_2:
                    answer_counts[existing_answer] += 1
                    matched = True
                    break

            if not matched:
                answer_counts[result] = 1

        if not answer_counts:
            continue

        # Determine the majority answer and its score
        majority_answer = max(answer_counts, key=answer_counts.get)
        max_count = answer_counts[majority_answer]
        score = max_count / len(results)

        # Skip certain question types that are hard to grade automatically
        if "proof" in question.lower() or 'box' in question.lower() or 'text' in majority_answer.lower():
            continue

        results_all.append({
            "question": question,
            "answer": majority_answer,
            "score": score,
            "image": image_b64,
            "question_type": question_type,
            "declared_skill": item["declared_skill"],
            "skill_match": item["skill_match"],
            "valid": item["valid"],
            "validity_reason": item.get("validity_reason", ""),
            "supervisor_correct": 1,
            "supervisor_reason": "",
            "_image_pil": item["image_pil"],
            'results': results
        })

    except Exception as e:
        print(f"[{args.suffix}] CRITICAL ERROR processing question '{question[:50]}...': {e}")
        continue

print(
    f"[{now()}][{args.suffix}] Majority-vote grading kept "
    f"{len(results_all)}/{len(solver_items)} questions before supervisor filtering."
)

if SUPERVISOR_ANSWER_ENABLED and results_all:
    eligible_indices = [
        idx for idx, result_item in enumerate(results_all)
        if result_item["answer"]
        and result_item["score"] >= SUPERVISOR_MIN_SCORE
        and result_item["score"] <= SUPERVISOR_MAX_SCORE
    ]
    eligible_index_set = set(eligible_indices)
    print(
        f"[{now()}][{args.suffix}] Running supervisor answer verification for "
        f"{len(eligible_indices)}/{len(results_all)} pre-filtered samples..."
    )
    judge_chats = []
    for idx in eligible_indices:
        result_item = results_all[idx]
        prompt = (
            "<|im_start|>system\n"
            "You are a strict visual question answering verifier. "
            "Given an image, a question, and a candidate answer, decide whether the candidate answer is correct. "
            "If the question is invalid, ambiguous, or cannot be answered from the image, output \\boxed{0}. "
            "You may first give a brief reason, then you must put the final decision inside \\boxed{} exactly once at the end. "
            "Output \\boxed{1} if the candidate answer is correct, otherwise output \\boxed{0}.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"Question: {result_item['question']}\n"
            f"Candidate Answer: {result_item['answer']}\n"
            "You may briefly explain why, then end with \\boxed{1} or \\boxed{0}.\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        judge_chats.append({
            "prompt": prompt,
            "multi_modal_data": {"image": result_item["_image_pil"]},
        })

    judge_responses = model.generate(judge_chats, sampling_params=judge_sample_params, use_tqdm=True) if judge_chats else []
    supervisor_correct_count = 0
    for debug_idx, response in enumerate(judge_responses[:3]):
        raw_text = response.outputs[0].text if response.outputs else ""
        print(f"[{args.suffix}] [supervisor-debug-{debug_idx}] {raw_text}")
    for idx, response in zip(eligible_indices, judge_responses):
        raw_text = response.outputs[0].text if response.outputs else ""
        correct = extract_boxed_binary(raw_text)
        results_all[idx]["supervisor_correct"] = correct
        results_all[idx]["supervisor_reason"] = raw_text.strip()
        supervisor_correct_count += correct
    for idx, result_item in enumerate(results_all):
        if idx not in eligible_index_set:
            result_item["supervisor_reason"] = "skipped_pre_filter"
    print(
        f"[{now()}][{args.suffix}] Supervisor answer verification complete: "
        f"{supervisor_correct_count}/{len(eligible_indices)} passed among pre-filtered samples."
    )
else:
    print(f"[{now()}][{args.suffix}] Supervisor answer verification disabled.")

for item in results_all:
    item.pop("_image_pil", None)

# 5. Save Final Results
grade_elapsed = time.time() - grade_start
print(f"[{now()}][{args.suffix}] Grading complete in {grade_elapsed:.1f}s ({grade_elapsed/60:.1f}min).")
print(f"[{now()}][{args.suffix}] Processed {len(results_all)} questions. Saving results to: {OUTPUT_FILE}")
with open(OUTPUT_FILE, "w") as f:
    json.dump(results_all, f, indent=4)

print(f"[{now()}][{args.suffix}] Script finished.")
