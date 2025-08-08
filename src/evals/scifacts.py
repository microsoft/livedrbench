import re
from utils import retry_on_exception
from openai import OpenAI

from typing import List

prompts = {
    "EM": (
        "You are an evaluator. Given a Golden Answer and a Predicted Answer, "
        "check whether the Predicted Answer is correct. The prediction is correct if it fully aligns "
        "with and contains all the key information present in the Golden Answer. "
        "Respond with True if the prediction is correct and False otherwise."
    ),
    "SCORE": (
        "You are an evaluator. Given a Golden Answer and a Predicted Answer, assign a score from 0 to 100 "
        "based on how closely the Predicted Answer matches the Golden Answer. Use the following rubric:\n\n"
        "Scoring Rubric:\n"
        "- 0-25: Mostly incorrect, off-topic, or irrelevant.\n"
        "- 26-50: Partially correct but misses or misrepresents most key information.\n"
        "- 51-75: Mostly accurate with some omissions or distortions.\n"
        "- 76-100: Closely aligned with the Golden Answer.\n\n"
        "Return only an integer score (0-100). Do not provide any explanation."
    ),
}


def compute_metrics(actual, prediction, key, client=None, judge_name=None):
    if prediction and len(prediction) > 0:
        precision, recall = compute_precision_recall_llm(actual, prediction, key=key, client=client, judge_name=judge_name)
    else:
        precision, recall = 0, 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def compute_precision_recall_llm(actual, prediction, client=None, judge_name=None, key='paper_title'):
    actual_sets = [set(entry[key]) for entry in actual]
    predicted_values = [entry.get(key, "") for entry in prediction if isinstance(entry, dict)]

    # Recall: each actual entry is correct if any prediction matches any of its equivalence class values
    correct_recall = 0
    for actual_set in actual_sets:
        if any(any(is_match_func(pred, actual_item, client=client, judge_name=judge_name) for actual_item in actual_set) for pred in predicted_values):
            correct_recall += 1
    recall = correct_recall / len(actual_sets) if actual_sets else 0.0

    # Precision: each prediction is correct if it matches any value from any equivalence class
    all_actual_items = {item for actual_set in actual_sets for item in actual_set}
    correct_precision = sum(
        any(is_match_func(pred, actual_item, client=client, judge_name=judge_name) for actual_item in all_actual_items)
        for pred in predicted_values
    )
    precision = correct_precision / len(predicted_values) if predicted_values else 0.0

    return precision, recall

@retry_on_exception(max_retries=5, delay=1)
def is_match_func(pred, actual_item, client=None, judge_name=None, prompt_type="EM"):
    if not pred:
        return 0
    if pred.lower().strip() == actual_item.lower().strip():
        return 1
    
    if client is not None:
        decoded_outputs = client.chat.completions.create(
                model=judge_name,
                messages=[
                    {"role": "system", "content": prompts[prompt_type]},
                    {
                        "role": "user",
                        "content": f"Golden Answer: {actual_item}\nPredicted Answer: {pred}",
                    },
                ],
            )
    
    else:
        # assume some global model
        decoded_outputs = client.chat.completions.create(
                model=judge_name,
                messages=[
                    {"role": "system", "content": prompts[prompt_type]},
                    {
                        "role": "user",
                        "content": f"Golden Answer: {actual_item}\nPredicted Answer: {pred}",
                    },
                ],
            )
            
    if prompt_type == "EM":
        return 1 if decoded_outputs.choices[0].message.content.lower() == "true" else 0
    elif prompt_type == "SCORE":
        return parse_output(decoded_outputs.choices[0].message.content)
    
def parse_output(output):
    match = re.search(r'\b(100|[1-9]?[0-9])\b', output)
    if match:
        return int(match.group(0))
    raise ValueError("No valid score (0-100) found in the model output.")

def grade(judge_name: str, key: str, ground_truths: List[List[dict]], preds: List[List[dict]], eval_info: dict | None) -> dict:
    client = OpenAI()
    
    # compute metrics
    if "material" in ground_truths[0][0].keys():
        m_precision, m_recall, m_f1 = compute_metrics(ground_truths[0], preds, key="material", client=client, judge_name=judge_name)
        p_precision, p_recall, p_f1 = compute_metrics(ground_truths[0], preds, key="paper_title", client=client, judge_name=judge_name)
        
        return {"precision": m_precision*p_precision, "recall": m_recall*p_recall, "f1": m_f1*p_f1}
        
    else:
        precision, recall, f1 = compute_metrics(ground_truths[0], preds, key="paper_title", client=client, judge_name=judge_name)

        return {"precision": precision, "recall": recall, "f1": f1}
