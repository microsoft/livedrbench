import json
from openai import OpenAI
from utils import retry_on_exception

from typing import List

prompts = {
    'dict': """
You are evaluating whether two dictionaries refer to the same information. Your task is to determine whether each corresponding value is equivalent.

For each key-value pair in the ground truth dictionary:
- If the key is absent in the model output dictionary, it is not equivalent.
- If the value is a number (integer or float), they are equivalent within a 1% margin of error.
- If the value is a string, consider them equivalent if they are similar in meaning, or if one is a subset of the other.
    - Acronyms or shortened names are equivalent to their expanded forms.
    - "BERT" and "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" -> equivalent
    - "ICLR" and "International Conference on Learning Representations" -> equivalent
    - "CNN" and "ResNet" -> equivalent (both are convolutional neural networks)
    - "Adam" and "SGD" -> not equivalent (different optimizers)
    - "0.8" and "0.80" -> equivalent numerically
- If the value is a URL, they must be exactly the same, or a shortened version of the same URL.
- If the value is a list, check if every element in the ground truth list is present in the model output list, regardless of order. Follow the same rules for strings as above.
- If the value is a dictionary, check if every key-value pair in the ground truth dictionary is present in the model output dictionary, regardless of order. Again, follow the same rules for strings as above.
- Some values may depend on information from other keys in the dictionary. Use such context to determine whether two values are equivalent, even if they are not identical on their own.
- Ignore trivial formatting differences such as casing, whitespace, or common abbreviations if the meaning is preserved.

Now compare the following dictionaries:

Ground Truth:
{dict_gt}

Model Output:
{dict_pred}

Your output should be a JSON object with the same keys as the input dictionaries. For each key, the value should be:
- 3: The predicted value closely matches the ground-truth, preserving almost all important information with minimal or no loss of meaning.
- 2: The predicted value captures the main idea of the ground-truth but omits key details or has minor inaccuracies.
- 1: The predicted value shows some overlap with the ground-truth but misses most of the essential meaning.
- 0: The predicted value is largely incorrect, unrelated, or does not correspond meaningfully to the ground-truth value.

IMPORTANT:
- Do NOT explain your reasoning.
- Do NOT include any extra text.
- Only output the final JSON object. Do not wrap it in markdown or say anything else.
""",
    'list_of_dicts_match': """
You are given a list of dictionaries (the ground truth) and a single predicted dictionary (the model output). Your task is to determine which dictionary in the ground truth list corresponds with the predicted dictionary, using a set of primary keys to identify the best match.

- You will be a given a list of primary keys that can be used to compare the dictionaries. If a primary key is not present in two dictionaries, make sure to use the other primary keys to determine the match.
- Do not consider ANY other fields in the dictionaries except for the primary keys.
- If the value is a string, consider them equivalent if they are similar in meaning, or if one is a subset of the other.
- Match based on semantic similarity or if one is a subset of the other, not exact string equality.
    - Acronyms, shortened names, or partial names are equivalent to their expanded forms.
    - "BERT" and "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" -> equivalent
    - "ICLR" and "International Conference on Learning Representations" -> equivalent
    - A prediction may include extra info (e.g., a year or qualifier). If the ground truth is a subset and meaning is preserved, treat them as equivalent. If the ground truth includes a qualifier, the prediction's qualifier must match if present.
- You may also be provided with a set of extra evaluation notes that provide more specific evaluation criteria. These extra rules may override the general rules above.
- If after all this no match exists, return an empty dictionary.

Now compare the following:

Ground Truth List:
{list_gt}

Model Output:
{dict_pred}

Primary Keys:
{primary_keys}

Extra Evaluation Notes:
{note}

Your output should be the dictionary from the ground truth list that best matches the model output, or an empty dictionary if no match is found.

IMPORTANT:
- Do NOT explain your reasoning.
- Do NOT include any extra text.
- Only output the final dictionary as a JSON object. Do not wrap it in markdown or say anything else.
""",
}

def find_and_remove_dict(dict_to_remove, list_of_dicts, primary_keys):
    if dict_to_remove is None:
        return list_of_dicts
    
    for i, d in enumerate(list_of_dicts):
        if all(d.get(pk) == dict_to_remove.get(pk) for pk in primary_keys):
            return list_of_dicts[:i] + list_of_dicts[i+1:]
    return list_of_dicts

def evaluate_dict_item(client, judge_name, key, gt_value, pred_value):
    prompt = prompts['dict_item'].format(key=key, gt_value=json.dumps(gt_value), pred_value=json.dumps(pred_value))
    response = client.chat.completions.create(
        model=judge_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0.0
    )
    eval_output = int(response.choices[0].message.content.strip())
    return eval_output
    
@retry_on_exception(max_retries=5, delay=1)
def evaluate_dict(client, judge_name, gt_dict, pred_dict):
    prompt = prompts['dict'].format(dict_gt=json.dumps(gt_dict), dict_pred=json.dumps(pred_dict))
    response = client.chat.completions.create(
        model=judge_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.0
    )
    output = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    eval_output = json.loads(output)
    return eval_output

@retry_on_exception(max_retries=5, delay=1)
def evaluate_list_of_dicts(client, judge_name, list_gt, dict_pred, eval_info):
    primary_keys = eval_info['primary_keys']
    note = eval_info.get('evaluation_note', {})
    match_prompt = prompts['list_of_dicts_match'].format(list_gt=json.dumps(list_gt), dict_pred=json.dumps(dict_pred), primary_keys=json.dumps(primary_keys), note=json.dumps(note))
    response = client.chat.completions.create(
        model=judge_name,
        messages=[{"role": "user", "content": match_prompt}],
        max_tokens=1024,
        temperature=0.0
    )
    output = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    extracted_gt_dict = json.loads(output)
    
    if len(extracted_gt_dict) == 0:
        return None, None

    eval_prompt = prompts['dict'].format(dict_gt=json.dumps(extracted_gt_dict), dict_pred=json.dumps(dict_pred))
    response = client.chat.completions.create(
        model=judge_name,
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=1024,
        temperature=0.0
    )
    
    output = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
    eval_output = json.loads(output)
    return extracted_gt_dict, eval_output

def grade(judge_name: str, key: str, ground_truths: List[List[dict] | dict], preds: List[List[dict] | dict], eval_info: dict | None) -> dict:
    client = OpenAI()
    
    evaluations = []
    for gt, pred in zip(ground_truths, preds):
        if isinstance(gt, dict):
            eval_output = evaluate_dict(client, judge_name, gt, pred)
            evaluations.append({ 'type': 'dict', 'extracted_gt': None, 'pred': pred, 'eval_output': eval_output })
        elif isinstance(gt, list) and isinstance(gt[0], dict):
            for row in pred:
                extracted_gt_dict, eval_output = evaluate_list_of_dicts(client, judge_name, gt, row, eval_info)
                if eval_output is None:
                    evaluations.append({ 'type': 'list_of_dicts_match', 'extracted_gt': None, 'pred': row, 'eval_output': None })
                    continue
                
                gt = find_and_remove_dict(extracted_gt_dict, gt, eval_info['primary_keys'])
                evaluations.append({ 'type': 'list_of_dicts_match', 'extracted_gt': extracted_gt_dict, 'pred': row, 'eval_output': eval_output })
        else:
            raise ValueError(f"Unsupported ground truth type: {type(gt)} for key {key}")
    
    main_claims = eval_info.get('main_claims', [])
    ignore_keys = set(eval_info.get('ignore_keys', []))
    
    gt_key_count = 0
    for gt in ground_truths:
        if type(gt) == dict:
            gt_key_count += len([k for k in gt if k not in ignore_keys])
        elif type(gt) == list and type(gt[0]) == dict:
            for item in gt:
                gt_key_count += len([k for k in item if k not in ignore_keys])
    
    eval_key_count = 0
    for evaluation in evaluations:
        pred = evaluation['pred']
        assert type(pred) is dict
        eval_key_count += len([k for k in pred if k not in ignore_keys])
    
    # search for a possible dataset/flight identification task. if this is not correct, there is no point in evaluating the rest of the claims
    is_identification_claim_correct = True
    for evaluation in evaluations: 
        evaluation_type = evaluation['type']
        if evaluation_type != 'dict':
            continue
        
        pred, eval_output = evaluation['pred'], evaluation['eval_output']
        
        for k, v in eval_output.items():
            if k in main_claims and v <= 1:
                is_identification_claim_correct = False 
                break        
            
        if not is_identification_claim_correct:
            break
    
    # now do the actual evaluation
    eval_score = 0
    for evaluation in evaluations:
        pred, eval_output = evaluation['pred'], evaluation['eval_output']
        if eval_output is None:
            continue
        
        # check if all main_extractive_claims, if any, are correct
        # is_identification_claim_correct will be False if the dataset or flight number identification task was not correct
        # leading to everything being considered incorrect
        are_main_extractive_claims_correct = is_identification_claim_correct 
        for k, v in eval_output.items():
            if k in main_claims and v <= 1:
                are_main_extractive_claims_correct = False
                break
        
        # if main claims are wrong, all claims for this paritular evaluation are not considered correct
        if not are_main_extractive_claims_correct:
            for k, v in eval_output.items():
                eval_output[k] = 0
        
        # binarize
        for k, v in eval_output.items():
            eval_output[k] = 0 if v <= 1 else 3
                
        eval_score += sum([v for k, v in eval_output.items() if k not in ignore_keys])
    
    eval_score /= 3 # normalize to [0, 1] range
    assert eval_score <= eval_key_count, f"Eval score {eval_score} exceeds eval key count {eval_key_count} for key {key}"
    assert eval_score <= gt_key_count, f"Eval score {eval_score} exceeds ground truth key count {gt_key_count} for key {key}"
    
    precision = eval_score / eval_key_count if eval_key_count > 0 else 0
    recall = eval_score / gt_key_count if gt_key_count > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # print(key, precision, recall, f1)
    
    return { 'precision': precision, 'recall': recall, 'f1': f1 }
