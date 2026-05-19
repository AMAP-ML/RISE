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

'''
This reward function is for regular [CoT] -> [Answer] GRPO finetuning
'''
import base64
from io import BytesIO
import re, os, json, glob
from typing import Dict, List, Optional
import time
import random
from mathruler.grader import extract_boxed_content, grade_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
import numpy as np
STORAGE_PATH = os.getenv("STORAGE_PATH")
if STORAGE_PATH is None:
    STORAGE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
SUPERVISOR_VALIDITY_ENABLED = os.getenv("SUPERVISOR_VALIDITY_ENABLED", "1") == "1"
QUESTIONER_DEBUG_SAMPLES = int(os.getenv("QUESTIONER_DEBUG_SAMPLES", "3"))
SKILL_AWARE_ENABLED = os.getenv("SKILL_AWARE_ENABLED", "1") == "1"
SKILL_BALANCE_ENABLED = os.getenv("SKILL_BALANCE_ENABLED", "1") == "1"
SKILL_BALANCE_WEIGHT = float(os.getenv("SKILL_BALANCE_WEIGHT", "0.2"))
REWARD_REQUEST_RETRIES = int(os.getenv("REWARD_REQUEST_RETRIES", "12"))
REWARD_REQUEST_RETRY_SLEEP_SEC = float(os.getenv("REWARD_REQUEST_RETRY_SLEEP_SEC", "5"))

TEMP_RESULTS_DIR = os.path.join(STORAGE_PATH, "temp_results")
os.makedirs(TEMP_RESULTS_DIR, exist_ok=True)

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
_SKILL_COUNTS_CACHE = {"signature": None, "counts": Counter()}


def normalize_skill_label(skill: Optional[str]) -> Optional[str]:
    if skill is None:
        return None
    normalized = str(skill).strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return SKILL_ALIASES.get(normalized)


def get_recent_skill_counts() -> Counter:
    summary_dir = os.path.join(STORAGE_PATH, "local_parquet")
    if not os.path.isdir(summary_dir):
        return Counter()

    summary_files = sorted(glob.glob(os.path.join(summary_dir, "*_train_summary.json")))
    signature = tuple(
        (path, os.path.getmtime(path), os.path.getsize(path))
        for path in summary_files
        if os.path.exists(path)
    )
    if signature == _SKILL_COUNTS_CACHE["signature"]:
        return _SKILL_COUNTS_CACHE["counts"]

    counts = Counter()
    for path, _, _ in signature:
        try:
            with open(path, "r") as f:
                payload = json.load(f)
        except Exception as exc:
            print(f"[skill-balance] failed to load summary {path}: {exc}")
            continue
        skill_counts = payload.get("declared_skill_counts_after_balance", {}) or payload.get("declared_skill_counts_after_filter", {})
        for skill, value in skill_counts.items():
            normalized = normalize_skill_label(skill)
            if normalized:
                try:
                    counts[normalized] += int(value)
                except Exception:
                    continue

    _SKILL_COUNTS_CACHE["signature"] = signature
    _SKILL_COUNTS_CACHE["counts"] = counts
    return counts


def compute_skill_balance_bonus(skill: Optional[str], counts: Counter) -> float:
    normalized = normalize_skill_label(skill)
    if not normalized or normalized not in ALLOWED_SKILLS:
        return 0.0
    total = sum(counts.get(label, 0) for label in ALLOWED_SKILLS)
    if total <= 0:
        return 0.0
    target = total / len(ALLOWED_SKILLS)
    current = counts.get(normalized, 0)
    if current >= target:
        return 0.0
    return (target - current) / max(target, 1.0)

def encode_image_to_base64(image):
    if image is None:
        return None
        
    if isinstance(image, np.ndarray) and image.dtype == object and image.size >= 1:
        img_obj = image.item(0) if image.ndim > 0 else image.item()
    else:
        img_obj = image
        

    if 'Image' not in str(type(img_obj)):
        print(f"Warning: Cannot encode unhandled object type: {type(img_obj)}")
        return None

    try:
        buffered = BytesIO()
        img_obj.save(buffered, format="PNG") 
        base64_data = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"
        
    except Exception as e:
        print(f"Error during Base64 encoding: {e}")
        return None

def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = [sentences[j].split()]
                hyp = sentences[i].split()
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist

def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
    start_time = time.time()
    dist_mat = _bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions

def format_reward(predict: str) -> float:
    pattern = re.compile(
        r"^\s*<skill>.+?</skill>\s*"
        r"<type>(multiple choice|numerical|regression)</type>\s*"
        r"<question>.+?</question>\s*"
        r"<answer>.+?</answer>\s*$",
        re.DOTALL
    )
    return 1.0 if pattern.fullmatch(predict.strip()) else 0.0


def has_multiple_choice_options_in_question(question: str) -> bool:
    if not question:
        return False
    detected = {char for char in question if char in {"A", "B", "C", "D"}}
    return all(label in detected for label in {"A", "B", "C", "D"})


def has_single_letter_answer(answer: str) -> bool:
    if not answer:
        return False
    return re.fullmatch(r"[A-Za-z]", answer.strip()) is not None


def match(generation):
    pattern = r"<skill>(.*?)</skill>.*?<type>(.*?)</type>.*?<question>(.*?)</question>.*?<answer>(.*?)</answer>"
    match_obj = re.search(pattern, generation, re.DOTALL)

    if match_obj:
        return {
            "declared_skill": normalize_skill_label(match_obj.group(1)),
            "question": match_obj.group(3).strip(),
            "answer": match_obj.group(4).strip(),
            "types": match_obj.group(2).strip()
        }
    return None


def compute_format_components(predict: str) -> Dict[str, float]:
    normalized_predict = predict.strip()
    structure_ok = format_reward(normalized_predict)
    parsed = match(normalized_predict) if structure_ok else None

    skill_ok = 1.0 if parsed and parsed.get("declared_skill") in ALLOWED_SKILLS else 0.0
    type_ok = 1.0 if parsed and parsed.get("types") in {"multiple choice", "numerical", "regression"} else 0.0
    question_ok = 1.0 if parsed and parsed.get("question") else 0.0
    answer_ok = 1.0 if parsed and parsed.get("answer") else 0.0

    choice_answer_ok = 1.0
    if parsed and parsed.get("types") == "multiple choice":
        question_has_choices = has_multiple_choice_options_in_question(parsed.get("question", ""))
        answer_is_single_letter = has_single_letter_answer(parsed.get("answer", ""))
        choice_answer_ok = 1.0 if (question_has_choices or answer_is_single_letter) else 0.0
    elif not parsed:
        choice_answer_ok = 0.0

    overall_ok = 1.0 if structure_ok and skill_ok and type_ok and question_ok and answer_ok else 0.0
    if parsed and parsed.get("types") == "multiple choice" and choice_answer_ok != 1.0:
        overall_ok = 0.0

    return {
        "format": overall_ok,
        "format_structure": structure_ok,
        "format_skill": skill_ok,
        "format_type": type_ok,
        "format_question": question_ok,
        "format_answer": answer_ok,
        "format_choice_answer": choice_answer_ok,
    }

def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def fetch(index, path, endpoint):
    url = f"http://127.0.0.1:{6000+index}/{endpoint}"
    last_exc = None
    for attempt in range(1, REWARD_REQUEST_RETRIES + 1):
        try:
            response = requests.get(url, params={"name": path}, timeout=1800)
            response.raise_for_status()
            return True
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= REWARD_REQUEST_RETRIES:
                break
            print(
                f"[reward-fetch] retrying endpoint={endpoint} port={6000+index} "
                f"attempt={attempt}/{REWARD_REQUEST_RETRIES} after error: {exc}"
            )
            time.sleep(REWARD_REQUEST_RETRY_SLEEP_SEC)
    raise last_exc

def generate_results(data, endpoint="hello"):
    datas = split_list(data,4)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(4)]
    for i in range(4):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch, i, random_names[i], endpoint) for i in range(4)]

        for future in as_completed(futures):
            print(future.result())

    for i in range(4):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(4):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results


def _shorten_text(text: str, limit: int = 160) -> str:
    text = "" if text is None else str(text).strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compute_score(predicts: List[str], ground_truths: List[str], questions: List[str], description_answers: List[str], format_weight: float = 0.1, images: Optional[List[str]] = None) -> List[Dict[str, float]]:
    print("Computing rewards")
    results = []
    format_components = []
    skill_counts = get_recent_skill_counts() if (SKILL_AWARE_ENABLED and SKILL_BALANCE_ENABLED) else Counter()
    for idx, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_info = compute_format_components(predict)
        dirty_results = match(predict)
        if dirty_results == None:
            item = {"question": "", "declared_skill": None}
        else:
            item = dirty_results
        if images is not None and idx < len(images):
            encoded_image = encode_image_to_base64(images[idx]) 
            if encoded_image:
                item["image"] = encoded_image
            else:
                item["image"] = None 
        results.append(item)
        format_components.append(format_info)

    validity_results = [
        {
            "question": item.get("question", ""),
            "declared_skill": item.get("declared_skill"),
            "skill_match": 0,
            "valid": 0,
            "reason": "skipped_format_gate",
        }
        for item in results
    ]
    validity_indices = [
        idx for idx, item in enumerate(results)
        if format_components[idx]["format"] == 1.0 and item.get("question")
    ]
    if SUPERVISOR_VALIDITY_ENABLED and validity_indices:
        validity_inputs = [
            {
                "question": results[idx]["question"],
                "image": results[idx].get("image"),
                "declared_skill": results[idx].get("declared_skill"),
            }
            for idx in validity_indices
        ]
        fetched_validity_results = generate_results(validity_inputs, endpoint="judge_validity")
        for idx, item in zip(validity_indices, fetched_validity_results):
            validity_results[idx] = item
        invalid_examples = [item.get("question", "") for item in validity_results if item.get("valid", 0) != 1][:3]
        print(
            f"[reward] validity enabled: valid={sum(1 for item in validity_results if item.get('valid', 0) == 1)}/"
            f"{len(validity_results)}, invalid_examples={invalid_examples}"
        )
    elif SUPERVISOR_VALIDITY_ENABLED:
        print("[reward] validity enabled: valid=0/0, invalid_examples=[]")

    difficulty_results = [{"question": "", "answer": "", "score": 0.0, "results": []} for _ in results]
    difficulty_indices = [
        idx for idx, item in enumerate(results)
        if format_components[idx]["format"] == 1.0 and item.get("question")
    ]
    if difficulty_indices:
        difficulty_inputs = [results[idx] for idx in difficulty_indices]
        fetched_difficulty_results = generate_results(difficulty_inputs)
        for idx, item in zip(difficulty_indices, fetched_difficulty_results):
            difficulty_results[idx] = item

    penalties = [0.0 for _ in results]
    if difficulty_indices:
        difficulty_questions = [difficulty_results[idx]["question"] for idx in difficulty_indices]
        penalty_values = cluster_share_per_problem(difficulty_questions, distance_threshold=0.5)
        assert len(penalty_values) == len(difficulty_indices)
        for idx, penalty in zip(difficulty_indices, penalty_values):
            penalties[idx] = penalty

    scores = []
    for i in range(len(difficulty_results)):
        format_info = format_components[i]
        valid = 1 if validity_results[i].get("valid", 0) == 1 else 0
        validity_bonus = 0.1 if valid == 1 else 0.0
        declared_skill = results[i].get("declared_skill")
        skill_match = 1 if validity_results[i].get("skill_match", 0) == 1 else 0
        skill_balance_bonus = 0.0
        if valid == 1 and SKILL_AWARE_ENABLED and SKILL_BALANCE_ENABLED:
            skill_balance_bonus = compute_skill_balance_bonus(declared_skill, skill_counts)
        penalty = penalties[i]
        skill_indicator_metrics = {
            f"skill_{skill.replace(' & ', '_').replace('-', '_').replace(' ', '_')}": 1.0 if declared_skill == skill else 0.0
            for skill in ALLOWED_SKILLS
        }

        if format_info["format"] != 1.0:
            difficulty_score = -1.0
            final_score = -1.0
        else:
            difficulty_score = min(difficulty_results[i]["score"], 1 - difficulty_results[i]["score"]) - penalty
            final_score = difficulty_score + validity_bonus + SKILL_BALANCE_WEIGHT * skill_balance_bonus
        score_item = {
            "overall": final_score,
            "format": format_info["format"],
            "format_skill": format_info["format_skill"],
            "format_choice_answer": format_info["format_choice_answer"],
            "validity": valid,
            "skill_match": skill_match,
            "skill_balance_bonus": skill_balance_bonus,
            "difficulty": difficulty_score,
            "penalty": penalty,
        }
        score_item.update(skill_indicator_metrics)
        scores.append(score_item)
    if QUESTIONER_DEBUG_SAMPLES > 0:
        debug_count = min(QUESTIONER_DEBUG_SAMPLES, len(scores))
        print(f"[questioner-debug] showing {debug_count}/{len(scores)} samples")
        for i in range(debug_count):
            item = results[i] if i < len(results) else {}
            score_item = scores[i]
            print(
                f"[questioner-debug-{i}] "
                f"skill={item.get('declared_skill') or 'unknown'} | "
                f"type={item.get('types', '')} | "
                f"question={_shorten_text(item.get('question', ''))} | "
                f"answer={_shorten_text(item.get('answer', ''))} | "
                f"overall={score_item['overall']:.4f} | "
                f"format={score_item['format']:.1f} | "
                f"validity={score_item['validity']} | "
                f"skill_match={score_item['skill_match']} | "
                f"skill_bonus={score_item['skill_balance_bonus']:.4f} | "
                f"difficulty={score_item['difficulty']:.4f}"
            )
    return scores
