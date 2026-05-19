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

from typing import Dict, List, Optional
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(predict: str) -> float:
    answer = extract_boxed_content(predict)
    return 0.0 if answer else -1

def extract_description(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<description>([\s\S]*?)</description>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()

def extract_answer(predict: str) -> Optional[str]:
    """
    Extracts the content of the <answer>…</answer> block from `predict`.
    Returns the inner text (with leading/trailing whitespace stripped),
    or None if no <answer> tag is found.
    """
    match = re.search(r"<answer>([\s\S]*?)</answer>", predict, re.DOTALL)
    if not match:
        return predict
    return match.group(1).strip()


def extract_boxed_answer(predict: str) -> Optional[str]:
    """
    Extracts the content inside \\boxed{...} from the predict string.
    Returns the content inside the boxed command, or None if not found.
    
    Example:
        Input: "...\\boxed{1000, 1000}"
        Output: "1000, 1000"
    """
    match = re.search(r"\\boxed\{([^}]*)\}", predict)
    if not match:
        return None
    return match.group(1).strip()


def accuracy_reward(predict: str, ground_truth: str) -> float:
    answer = extract_boxed_content(predict)
    if not answer:
        return 0.0
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predicts: List[str], ground_truths: List[str], questions: List[str], description_answers: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []

    for predict, ground_truth in zip(predicts, ground_truths):
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        overall_score = format_score + accuracy_score
        scores.append(
            {
                "overall": overall_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )
    return scores
