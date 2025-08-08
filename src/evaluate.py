from datasets import load_dataset
from utils import decrypt, map_with_progress
from evals import scifacts, datasets_flights, priorart, entities
import os
import json
from collections import defaultdict

from typing import List
from argparse import ArgumentParser

class LiveDRBenchEval():
    def __init__(self, judge_name: str, preds_file: str, num_threads: int):
        livedrbench = load_dataset("microsoft/LiveDRBench", "v1-full")['test']
        self.rows = livedrbench.to_list()
        
        self.judge_name = judge_name
        self.num_threads = num_threads
        
        self.all_preds = {}
        with open(preds_file, 'r') as f:
            all_preds = json.load(f)
        for preds in all_preds:
            key = preds['key']
            self.all_preds[key] = preds
            
    def verify_types(self, row: dict, preds: List[List[dict | str] | dict]) -> None:
        key = row['key']
        category = row.get("category", "")
        ground_truths = json.loads(decrypt(row.get("ground_truths", ""), row.get("canary", "")))
        
        assert len(ground_truths) == len(preds), f"Ground truths and predictions not the same length for key {key}: {len(ground_truths)} vs {len(preds)}"
        
        if 'scifacts-' in category:
            for gt, pred in zip(ground_truths, preds):
                assert type(gt) == type(pred), f"Type mismatch for {key}: {type(gt)} vs {type(pred)}"
                for row in pred:
                    assert type(row) == dict, f"Type mismatch for {key}: {type(row)}, expected dict"
        elif 'novel-datasets-' in category:
            for gt, pred in zip(ground_truths, preds):
                assert type(gt) == type(pred), f"Type mismatch for {key}: {type(gt)} vs {type(pred)}"
        elif category == "prior-art":
            for gt, pred in zip(ground_truths, preds):
                assert type(gt) == type(pred) == list, f"Type mismatch for {key}: {type(gt)} vs {type(pred)}"
                for row in pred:
                    assert type(row) == dict, f"Type mismatch for {key}: {type(row)}, expected dict"
        elif category == "entities":
            for gt, pred in zip(ground_truths, preds):
                assert type(gt) == type(pred) == list, f"Type mismatch for {key}: {type(gt)} vs {type(pred)}"
                for row in pred:
                    assert type(row) == str, f"Type mismatch for {key}: {type(row)}, expected str"
        elif category == "flights":
            for gt, pred in zip(ground_truths, preds):
                assert type(gt) == type(pred), f"Type mismatch for {key}: {type(gt)} vs {type(pred)}"

    def grade_row(self, row: dict, preds: List[List[dict | str] | dict]) -> dict:
        key = row['key']
        category = row.get("category", "")
        ground_truths = json.loads(decrypt(row.get("ground_truths", ""), row.get("canary", "")))
        eval_info = json.loads(decrypt(row.get("misc", ""), row.get("canary", "")))['eval_info']
        
        if "scifacts-" in category:
            grade_fn = scifacts.grade
        elif "novel-datasets-" in category:
            grade_fn = datasets_flights.grade
        elif category == "prior-art":
            grade_fn = priorart.grade
        elif category == "entities":
            grade_fn = entities.grade
        elif category == "flights":
            grade_fn = datasets_flights.grade
        else:
            raise ValueError(f"Unknown category: {category}")
        
        grade_results = grade_fn(
            judge_name=self.judge_name,
            key=key,
            ground_truths=ground_truths,
            preds=preds,
            eval_info=eval_info,
        )
        
        grade_results = {
            'key': key,
            'category': category,
            **grade_results
        }
        
        return grade_results
    
    def aggregate_metrics(self, all_grade_results: dict) -> dict:
        category_grouping = defaultdict(list)
        for result in all_grade_results:
            category_grouping[result['category']].append(result)
        
        category_results = {}
        for category, results in category_grouping.items():
            precisions = [r['precision'] for r in results]
            recalls = [r['recall'] for r in results]
            f1s = [r['f1'] for r in results]

            category_results[category] = {
                'precision': sum(precisions) / len(precisions) if precisions else 0,
                'recall': sum(recalls) / len(recalls) if recalls else 0,
                'f1': sum(f1s) / len(f1s) if f1s else 0,
            }
            
        # Compute overall average metrics across categories
        all_precisions = [m['precision'] for m in all_grade_results]
        all_recalls = [m['recall'] for m in all_grade_results]
        all_f1s = [m['f1'] for m in all_grade_results]

        category_results['overall'] = {
            'precision': sum(all_precisions) / len(all_precisions) if all_precisions else 0,
            'recall': sum(all_recalls) / len(all_recalls) if all_recalls else 0,
            'f1': sum(all_f1s) / len(all_f1s) if all_f1s else 0,
        }
        
        return category_results

    def __call__(self):
        jobs = []
        for row in self.rows:
            key = row['key']
            preds = self.all_preds[key]['preds']
            self.verify_types(row, preds)
            jobs.append({'row': row, 'preds': preds})
        all_grade_results = map_with_progress(self.grade_row, jobs, num_threads=self.num_threads, pbar=True)
            
        # Aggregate metrics
        aggregate_metrics = self.aggregate_metrics(all_grade_results)
        printing_order = ['scifacts-materials', 'scifacts-geo', 'novel-datasets-identification', 'novel-datasets-identi-extraction', 'novel-datasets-peer', 'prior-art', 'entities', 'flights', 'overall']
        print("\n\n##################")
        print("AGGREGATE METRICS") 
        for category in printing_order:
            metrics = aggregate_metrics[category]
            print(f"{category}: Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
        print("##################")

        return all_grade_results, aggregate_metrics

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    args.add_argument("--openai_model_name", type=str, default="gpt-4o", help="OpenAI model name to use as judge")
    args.add_argument("--preds_file", type=str, required=True, help="Path to the JSON file containing predictions")
    args.add_argument("--num_threads", type=int, default=8, help="Number of threads to use for evaluation")
    args.add_argument("--debug", action='store_true', help="Enable debug mode, without multithreading")
    args = args.parse_args()
    
    os.environ['OPENAI_API_KEY'] = args.openai_model_name
    
    if args.debug:
        print("Running in debug mode")
        os.environ['debug'] = '1'
    
    LiveDRBenchEval(        
        judge_name=args.openai_model_name,
        preds_file=args.preds_file,
        num_threads=args.num_threads
    )()
