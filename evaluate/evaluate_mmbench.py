import json
import re
import os
from typing import List, Dict, Any, Optional
from math_verify import parse, verify
from datetime import datetime
from pathlib import Path
from mathruler.grader import extract_boxed_content, grade_answer
import argparse
    
class MmbenchEvaluator():
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def normalize_answer(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        return text

    def compute_accuracy(self, answer_content, gt, options):

        # judging the option-only answer
        only_option_patterns = [
            r"^\(\s*([A-Z])\s*\)$",    # (A)
            r"^\s*([A-Z])\s*\)$",       # A) 
            r"^\s*([A-Z])\s*\.$",       # A.
            r"^\s*([A-Z])\s*-$",        # A-
            r"^\s*([A-Z])\s*:$",        # A:
            r"^\s*([A-Z])\s*$",         # A
        ]
        for pattern in only_option_patterns:
            match = re.match(pattern, answer_content.strip(), re.IGNORECASE)
            if match:
                extracted_option = match.group(1).upper()
                if extracted_option == gt.strip().upper():
                    return 1.0

        # judging the option+content answer
        option_patterns = [
            r"^\(\s*([A-Z])\s*\)\s*(.+)",    # (A) answer
            r"^\s*([A-Z])\s*\)\s*(.+)",       # A) answer 
            r"^\s*([A-Z])\s*\.\s*(.+)",       # A. answer
            r"^\s*([A-Z])\s*-\s*(.+)",        # A- answer
            r"^\s*([A-Z])\s*:\s*(.+)",        # A: answer
            r"^\s*([A-Z])\s*\s*(.+)",         # A answer
        ]
        for pattern in option_patterns:
            match = re.match(pattern, answer_content.strip(), re.IGNORECASE)
            if match:
                extracted_option = match.group(1).upper()
                if extracted_option == gt.strip().upper():
                    return 1.0

        # judging the content-only answer
        normalized_student = self.normalize_answer(answer_content)
        for option, content in options.items():
            if not content:
                continue
            normalized_content = self.normalize_answer(content)
            if normalized_student == normalized_content and option == gt:
                return 1.0
            if grade_answer(answer_content, content) and option == gt: 
                return 1.0
            if float(verify(parse(answer_content), parse(content))) > 0 and option == gt:
                return 1.0
        return 0.0

    def evaluate(self, results: Dict) -> float:
        total = 0
        correct = 0

        total_length = 0
        short_response_length=0
        long_response_length=0

        short_num = 0
        long_num = 0

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B") # load your tokenizer here

        for key, item in results.items():
            gt = item['answer']
            response = item['response']
            if response.startswith("Short Thinking:"):
                short_num += 1
                short_response_length += len(tokenizer.tokenize(response))
            elif response.startswith("Long Thinking:"):
                long_num += 1
                long_response_length += len(tokenizer.tokenize(response))
            total_length += len(tokenizer.tokenize(response))
            if mode == "answer":
                answer_content = self.extract_answer(response)
            else:
                answer_content = extract_boxed_content(response)
            options = {k: item.get(k) for k in ["A", "B", "C", "D"] if item.get(k)}
            total += 1
            correct += self.compute_accuracy(answer_content, gt, options)
        accuracy = correct / total if total > 0 else 0
        avg_length = total_length / total
        avg_short_length = short_response_length / short_num if short_num > 0 else 0
        avg_long_length = long_response_length / long_num if long_num > 0 else 0

        print(f"Dataset: {self.dataset_name}, Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}, Avg_length:{avg_length}")
        print(f"Short Thinking: {short_num}, Long Thinking: {long_num}")
        print(f"Average Short Thinking Length: {avg_short_length}, Average Long Thinking Length: {avg_long_length}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', type=str, required=True, 
                       help='Path to the JSON file to evaluate')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found!")
        exit(1)
    
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)

    evaluator = MmbenchEvaluator(dataset_name='mmbench')
    evaluator.evaluate(data)
