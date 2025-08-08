import json
from openai import OpenAI
from utils import retry_on_exception

from typing import List

prompts = {
    'str_search': """
You are evaluating whether a predicted string is present in a ground-truth list of strings. 

- Allow for minor differences in formatting, such as casing, whitespace, or common abbreviations.
- If found, return the matching string from the ground truth list in JSON format: {{ "match": <matched_string> }}
- If not, return an empty dictionary.

Now compare the following:

Ground Truth List:
{gt_list}

Predicted String:
{pred_str}

IMPORTANT:
- Do NOT explain your reasoning.
- Do NOT include any extra text.
- Only output the JSON object. Do not wrap it in markdown or say anything else.
"""
}

def find_and_remove_str(str_to_remove, list_of_strs):
    if str_to_remove is None:
        return list_of_strs
    
    for i, d in enumerate(list_of_strs):
        if d.lower() == str_to_remove.lower():
            return list_of_strs[:i] + list_of_strs[i+1:]
    return list_of_strs

@retry_on_exception(max_retries=5, delay=1)
def evaluate_list_of_str(client, judge_name, gt_list, pred_value):
    prompt = prompts['str_search'].format(gt_list=json.dumps(gt_list), pred_str=json.dumps(pred_value))
    response = client.chat.completions.create(
        model=judge_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.0
    )
    output = response.choices[0].message.content.strip()
    eval_output = json.loads(output)
    
    if 'match' not in eval_output:
        return None
    return eval_output

            
def grade(judge_name: str, key: str, ground_truths: List[List[dict]], preds: List[List[dict]], eval_info: dict | None) -> dict:
    client = OpenAI()
    
    evaluations = []
    for gt, pred in zip(ground_truths, preds):
        gt = [row['title'] for row in gt]
        pred = [row['title'] for row in pred]
        
        for row in pred:
            eval_output = evaluate_list_of_str(client, judge_name, gt, row)
            if eval_output is None:
                evaluations.append({ 'type': 'list_of_dicts_match', 'extracted_gt': None, 'pred': row, 'eval_output': None })
                continue
            
            extracted_gt = eval_output['match']
            gt = find_and_remove_str(extracted_gt, gt)
            evaluations.append({ 'type': 'str_search', 'extracted_gt': extracted_gt, 'pred': row, 'eval_output': extracted_gt })

    gt_key_count = 0
    for gt in ground_truths:
        gt_key_count += len(gt)
    
    eval_key_count = len(evaluations)
    eval_score = sum([1 for evaluation in evaluations if evaluation['extracted_gt'] is not None])
    
    assert eval_score <= eval_key_count, f"Eval score {eval_score} exceeds eval key count {eval_key_count} for key {key}"
    assert eval_score <= gt_key_count, f"Eval score {eval_score} exceeds ground truth key count {gt_key_count} for key {key}"
    
    precision = eval_score / eval_key_count if eval_key_count > 0 else 0
    recall = eval_score / gt_key_count if gt_key_count > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # print(key, precision, recall, f1)

    return { 'precision': precision, 'recall': recall, 'f1': f1 }
 
