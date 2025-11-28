import json
import re
import os
from typing import List, Dict, Any, Optional
from math_verify import parse, verify
from datetime import datetime
from pathlib import Path
from mathruler.grader import extract_boxed_content, grade_answer
import argparse


class MathEvaluator():
    def __init__(self, dataset_name: str):
        self.num_to_word = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
            "10": "ten",
            "11": "eleven",
            "12": "twelve",
            "13": "thirteen",
            "14": "fourteen",
            "15": "fifteen",
            "16": "sixteen",
            "17": "seventeen",
            "18": "eighteen",
            "19": "nineteen",
            "20": "twenty",
            "30": "thirty",
            "40": "forty",
            "50": "fifty",
            "60": "sixty",
            "70": "seventy",
            "80": "eighty",
            "90": "ninety",
            "100": "one hundred"
        }
        self.dataset_name = dataset_name

    def compute_accuracy(self, answer_content, gt, options, question_type):
        if question_type == "multi_choice":
            if self.dataset_name == "mathvista":
                index = options.index(gt)
                option = chr(ord("A") + index)
                content = gt
            else:
                option = gt.strip().upper()
                content = options[ord(option)- ord('A')]

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
                    if extracted_option == option.strip().upper():
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
                    if extracted_option == option.strip().upper():
                        return 1.0

            # judging the content-only answer
            if grade_answer(answer_content, content):
                return 1.0
            answer = parse(answer_content)
            gt_content = parse(content)
            if float(verify(answer, gt_content)) > 0:
                return 1.0
            if gt.lower().strip() == answer_content.lower().strip():
                return 1.0
            if self.num_to_word.get(gt.strip().lower(), "xxxx") == answer_content.strip().lower():
                return 1.0
        else:
            if grade_answer(answer_content, gt):
                return 1.0
            math_answer = parse(answer_content)
            if float(verify(math_answer, parse(gt))) > 0:
                return 1.0
            if gt.lower().strip() == answer_content.lower().strip():
                return 1.0 
            if self.num_to_word.get(gt.strip().lower(), "xxxx") == answer_content.strip().lower():
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
            if response.startswith("Short Thinking"):
                short_num += 1
                short_response_length += len(tokenizer.tokenize(response))
            elif response.startswith("Long Thinking"):
                long_num += 1
                long_response_length += len(tokenizer.tokenize(response))
            total_length += len(tokenizer.tokenize(response))
            options = item['options']
            question_type = item['question_type']

            answer_content = extract_boxed_content(response)
                
            total += 1
            correct += self.compute_accuracy(answer_content, gt, options, question_type)

        accuracy = correct / total if total > 0 else 0
        avg_length = total_length / total
        avg_short_length = short_response_length / short_num if short_num > 0 else 0
        avg_long_length = long_response_length / long_num if long_num > 0 else 0
        print(f"Dataset: {self.dataset_name}, Total: {total}, Correct: {correct}, Accuracy: {accuracy:.2%}, avg_length:{avg_length}")
        print(f"Short Thinking Responses: {short_num}, Long Thinking Responses: {long_num}")
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

    evaluator = MathEvaluator(dataset_name='mathvista')
    evaluator.evaluate(data)
