import argparse
import json
import os
import random
import base64
from io import BytesIO

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
import numpy as np
from urllib.parse import urlparse
from datasets import Dataset
import re

def inference_chat_model(dataset_name, out_dir, seed, system_prompt):
    random.seed(seed)
    system_prompt = system_prompt
    output_path = os.path.join(out_dir, f"{dataset_name}.json")

    dataset = load_dataset("MathLLMs/MathVision", split='test')
    inputs = []
    print(f"Processing dataset {dataset_name}")
    from PIL import Image
    for idx, data_item in tqdm(enumerate(dataset)):
        ## PIL image
        base64_image = data_item['decoded_image'].convert('RGB')
        buffer = BytesIO()
        base64_image.save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        base64_string = base64_bytes.decode("utf-8")

        question = data_item["question"].removesuffix("\n<image1>")
        choices = data_item["options"]
        if len(choices) == 0:
            ques = question
        else:
            options = [chr(ord("A") + i) for i in range(len(choices))]
            choices_str = "\n".join([f"{option}. {choice}" for option, choice in zip(options, choices)])
            ques = question + f'\n Options:{choices_str}'
    
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{base64_string}"
                    },
                    {
                        "type": "text",
                        "text": ques
                    },
                ],
            }
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_data, _ = process_vision_info(messages)

        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_data
            },
        })

    sampling_params = SamplingParams(temperature=0.01, top_p=0.001, top_k=1, max_tokens=2048,
                                    stop_token_ids=None, skip_special_tokens=False,
                                    repetition_penalty=1.0)
    
    print("Generating responses...")
    model_outputs = llm.generate(inputs, sampling_params=sampling_params)

    outputs = {}
    for idx, (data_item, model_output) in enumerate(zip(dataset, model_outputs)):
        outputs[idx] = {
            "question": data_item['question'],
            "response": model_output.outputs[0].text
        }

        outputs[idx]['answer'] = data_item['answer']
        outputs[idx]['question_type'] = "multi_choice" if len(data_item['options']) > 0 else "free_form"
        outputs[idx]['options'] = data_item['options']
    

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    print('Results saved to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--out-dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    llm = LLM(
        model=args.checkpoint,
        trust_remote_code=True,
        tensor_parallel_size=1,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.8
    )
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)

    SYSTEM_PROMPT = """You are a Vision-Language Model answering questions about images. 
Follow these rules strictly:  
1. Judge the length of reasoning needed.
- Short: start with "Short Thinking:".
- Long: start with "Long Thinking:".
2. Short Thinking: give a concise thinking process which is sufficient to answer the question, then provide the final answer.
3. Long Thinking: give a structured reasoning process of the question and the image, including question analysis, visual details description, self-verification and then provide the final answer.
4. The final answer MUST BE put in \\boxed{}."""

    inference_chat_model("mathvision", args.out_dir, args.seed, SYSTEM_PROMPT)
