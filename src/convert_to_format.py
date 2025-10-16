import jsonlines
from tqdm import tqdm
import json
from datasets import load_dataset
from argparse import ArgumentParser
from openai import OpenAI

def parse_output(client, pred):
    PROMPT = "From the given trace, locate the JSON structure (the part enclosed in {} or []). Extract it and return only that JSON object exactly as it appears in the trace. Do not add or remove any characters outside the JSON."
    decoded_outputs = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": f"Provided text: {pred}",
                },
            ],
        )
    
    return decoded_outputs.choices[0].message.content
    

def convert_to_format(client, pred, format):
    PROMPT = """You are given a question that specifies an exact output format, along with a predicted output that may or may not follow that format. Your task is to convert the predicted JSON output to match the specified format as closely as possible.

**Instructions:**
- The final output must strictly follow the required JSON structure.  
- If any key from the specified format is missing in the predicted output, assign it an empty string (`""`).  
- Return **only** the final JSON object — no additional text, explanations, or formatting.
- DO NOT try to answer the question, only focus on the format.
- The final answer must be a parsable json, remove any additional characters like json```"""
    decoded_outputs = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PROMPT},
                {
                    "role": "user",
                    "content": f"Question which specifies the format: {format}\n\n Provided json: {json.dumps(pred)}",
                },
            ],
        )
    
    return decoded_outputs.choices[0].message.content

### change this according to your output format
def convert(args, client):
    livedrbench = load_dataset("microsoft/LiveDRBench", "v1-full")['test']
    rows = livedrbench.to_list()
    key_question_map = {r["question"]: r["key"] for r in rows}

    infer = list(jsonlines.open(args.preds_file))

    livedrbench_format = []

    correct_count = 0
    for example in tqdm(infer):
        key = key_question_map[example["question"]]
        if example["prediction"]:
            prediction = parse_output(client, example["prediction"])
            prediction = convert_to_format(client, prediction, example["question"])
            try:
                prediction = [json.loads(prediction)]
                correct_count += 1
            except:
                prediction = []
        
        else:
            prediction = []

        livedrbench_format.append({"key": key, "preds": prediction})
    
    print(f"Correct format: {correct_count}/{len(infer)}")
        
    with open(args.out_file, "w") as f:
        json.dump(livedrbench_format, f, indent=1)
    

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    args.add_argument("--openai_model_name", type=str, default="gpt-4o", help="OpenAI model name to use as judge")
    args.add_argument("--preds_file", type=str, required=True, help="Path to the JSON file containing predictions")
    args.add_argument("--out_file", type=str, required=True, help="Output file name")
    args = args.parse_args()
    
    client = OpenAI()
    convert(args, client)
    
    
