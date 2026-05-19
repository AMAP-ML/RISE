#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import sys
import threading
import time
import torch
import re
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit  # 1. Import the thread-safe 'stopit' library
import base64
import io
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from verl.utils.vllm_utils import VLLMHijack
# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument(
    '--max_model_len',
    type=int,
    default=12288,
    help='Cap vLLM max model length to keep KV cache requirements bounded.',
)
parser.add_argument(
    '--hello_batch_size',
    type=int,
    default=256,
    help='Maximum number of /hello prompts submitted to one vLLM generate call.',
)
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print('[init] Loading model...')

VLLMHijack.hijack()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=args.max_model_len,
    disable_mm_preprocessor_cache=True,
    enable_prefix_caching=False,
)

sample_params = vllm.SamplingParams(
    max_tokens=2048,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10, # Generate 10 candidate answers for each question
)

judge_sample_params = vllm.SamplingParams(
    max_tokens=256,
    temperature=0.0,
    top_p=1.0,
    top_k=-1,
    stop_token_ids=[tokenizer.eos_token_id],
    n=1,
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()
generation_lock = threading.Lock()

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

# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)


@app.route('/healthz', methods=['GET'])
def healthz():
    return jsonify({"status": "ok", "port": int(args.port)})


def base64_to_pil(b64_string):
    if not b64_string:
        return None
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def extract_boxed_binary(text):
    boxed = extract_boxed_content(text or "")
    if boxed is None:
        return 0
    normalized = str(boxed).strip().lower()
    if normalized in {"1", "yes", "true", "valid", "correct"}:
        return 1
    if normalized in {"0", "no", "false", "invalid", "incorrect"}:
        return 0
    return 0


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


def build_validity_chat(question, img, declared_skill):
    skill_context = get_skill_context(declared_skill)
    prompt = (
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
    return {"prompt": prompt, "multi_modal_data": {"image": img}}


def process_validity_single(question, declared_skill, response):
    raw_text = response.outputs[0].text if response.outputs else ""
    valid = extract_boxed_binary(raw_text)
    skill_match = extract_skill_match(raw_text, final_valid=valid)
    return {
        "question": question,
        "declared_skill": declared_skill,
        "skill_match": skill_match,
        "valid": valid,
        "reason": raw_text.strip(),
    }


def engine_is_dead(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "engine core" in message
        or "engine died" in message
        or "enginecore encountered an issue" in message
        or "shutting down" in message
    )


def dump_failed_batch(name, batch_name, batch_start, batch_end, items):
    debug_path = name.replace(
        ".json",
        f"_{batch_name}_failed_batch_{batch_start}_{batch_end - 1}.json",
    )
    with open(debug_path, "w") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"[server] Saved failed batch metadata to {debug_path}")


def generate_with_fallback(name, batch_name, prompts, sampling_params, batch_items, use_tqdm=True):
    if not prompts:
        return []

    outputs = []
    total = len(prompts)
    chunk_size = args.hello_batch_size if batch_name == "hello" else len(prompts)
    total_batches = (total + chunk_size - 1) // chunk_size

    for batch_idx, batch_start in enumerate(range(0, total, chunk_size), start=1):
        batch = prompts[batch_start:batch_start + chunk_size]
        items = batch_items[batch_start:batch_start + chunk_size]
        batch_end = batch_start + len(batch)
        print(
            f"[server] /{batch_name} generating batch {batch_idx}/{total_batches} "
            f"(prompts {batch_start}-{batch_end - 1}, size={len(batch)})"
        )
        try:
            outputs.extend(model.generate(batch, sampling_params=sampling_params, use_tqdm=use_tqdm))
        except Exception as e:
            dump_failed_batch(name, batch_name, batch_start, batch_end, items)
            if engine_is_dead(e):
                raise
            print(f"[server] /{batch_name} batch failed, retrying one-by-one: {e}")
            for local_idx, single_prompt in enumerate(batch):
                single_item = items[local_idx]
                single_index = batch_start + local_idx
                try:
                    single_output = model.generate([single_prompt], sampling_params=sampling_params, use_tqdm=False)
                    outputs.extend(single_output)
                except Exception as single_exc:
                    dump_failed_batch(
                        name,
                        batch_name,
                        single_index,
                        single_index + 1,
                        [single_item],
                    )
                    if engine_is_dead(single_exc):
                        raise
                    print(f"[server] /{batch_name} skipping prompt {single_index}: {single_exc}")
                    outputs.append(None)
    return outputs

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]
    types     = [item.get('types',    '') for item in data]
    image     = [item.get('image',    '') for item in data]

    # Convert image list.
    pil_images = []
    for img_b64 in image:
        if img_b64:
            try:
                pil_images.append(base64_to_pil(img_b64))
            except Exception as e:
                print(f"[warning] Image decode failed: {e}")
                pil_images.append(None)
        else:
            pil_images.append(None)

    # (Data preparation logic remains unchanged)
    valid_chats = []
    valid_chat_items = []
    valid_indices = []
    for i, (q, a, t, img) in enumerate(zip(questions, answers, types, pil_images)):
        if q and a and t and img:
            prompt = (
                "<|im_start|>system\n"
                "You are an AI visual question answering assistant. "
                "Answer questions based only on the visual content provided. "
                "You **must only output your final answer inside \\boxed{}**. "
                "Do not write explanations or any other text.\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"This is a question: {q}.\n"
                "**IMPORTANT:** Only output your answer in the form \\boxed{{answer}}.Do NOT include any units; provide only the numeric value or option.\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            valid_chats.append({
                "prompt": prompt,
                "multi_modal_data": {"image": img}
            })
            valid_chat_items.append({
                "prompt_index": i,
                "question": q,
                "answer": a,
                "type": t,
                "image_size": {"width": img.width, "height": img.height},
            })
            valid_indices.append(i)
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # (vLLM generation logic remains unchanged)

    responses = []
    with generation_lock:
        responses = generate_with_fallback(
            name,
            "hello",
            valid_chats,
            sample_params,
            valid_chat_items,
            use_tqdm=True,
        )


    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    def process_single(question, golden_answer, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        # print(f"[process_single] Processing question: '{question[:70]}...'")

        answer_counts = {}
        for res in results:
            if not res: continue # Skip empty results
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop over exist_ans
                
                # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
                try:
                    is_match = False
                    # First direction: res vs exist_ans
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    # Second direction (only if first failed): exist_ans vs res
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                             # Log timeout for the second direction as well
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop

                except Exception as e:
                    # Catch any other potential errors from the grader function itself.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue # Continue to the next comparison in the inner loop
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,
            'results':  results
        }

    results_all = [
        {
            'question': q,
            'answer': a,
            'score': -1,
            'results': [],
            'reason': 'missing question, answer, type, or image',
        }
        for q, a in zip(questions, answers)
    ]
    for idx, response in zip(valid_indices, responses):
        q = questions[idx]
        a = answers[idx]
        try:
            if response is None:
                raise RuntimeError("generation failed for this prompt")
            item = process_single(q, a, response)
            item['reason'] = ''
            results_all[idx] = item
        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all[idx] = {
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'unhandled exception in process_single: {str(e)}'
            }
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})


@app.route('/judge_validity', methods=['GET'])
def judge_validity():
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received validity request for task file: {name}')

    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    declared_skills = [normalize_skill_label(item.get('declared_skill')) or item.get('declared_skill', 'unknown') for item in data]
    images = [item.get('image', '') for item in data]

    pil_images = []
    for img_b64 in images:
        if img_b64:
            try:
                pil_images.append(base64_to_pil(img_b64))
            except Exception as e:
                print(f"[warning] Image decode failed in validity judge: {e}")
                pil_images.append(None)
        else:
            pil_images.append(None)

    valid_chats = []
    valid_chat_items = []
    valid_indices = []
    results_all = [
        {
            'question': q,
            'declared_skill': declared_skill,
            'skill_match': 0,
            'valid': 0,
            'reason': 'missing question, image, or declared skill',
        }
        for q, declared_skill in zip(questions, declared_skills)
    ]

    for idx, (q, img, declared_skill) in enumerate(zip(questions, pil_images, declared_skills)):
        if q and img and normalize_skill_label(declared_skill):
            valid_chats.append(build_validity_chat(q, img, declared_skill))
            valid_chat_items.append({
                "prompt_index": idx,
                "question": q,
                "declared_skill": declared_skill,
                "image_size": {"width": img.width, "height": img.height},
            })
            valid_indices.append(idx)

    if valid_chats:
        with generation_lock:
            responses = generate_with_fallback(
                name,
                "judge_validity",
                valid_chats,
                judge_sample_params,
                valid_chat_items,
                use_tqdm=True,
            )
        debug_responses = [response for response in responses if response is not None]
        for debug_idx, response in enumerate(debug_responses[:3]):
            raw_text = response.outputs[0].text if response.outputs else ""
            print(f"[server][validity-debug-{debug_idx}] {raw_text}")
        for idx, response in zip(valid_indices, responses):
            if response is not None:
                results_all[idx] = process_validity_single(questions[idx], declared_skills[idx], response)
            else:
                results_all[idx] = {
                    "question": questions[idx],
                    "declared_skill": declared_skills[idx],
                    "skill_match": 0,
                    "valid": 0,
                    "reason": "generation failed",
                }

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    pause_event.clear()
    print(f'[server] Processed validity {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed validity {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=False)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
