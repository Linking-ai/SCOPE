import re
import argparse
from datasets import load_dataset
import os
import json

def process_question(args, q_list, a_list, prompt_original):
    prompt_q = 'Examples: \n' + prompt_original + '\n\nFollowing Question: \n'
    for i, q in enumerate(q_list, start=9):
        prompt_q += f'Question_{i}:\n{q}\n'
    prompt_q += '\n'
    
    # Save JSON Lines
    json_data = {
        "prompt": prompt_q,
        "questions": q_list,
        "answers": a_list
    }
    jsonl_output_dir = args.output_dir
    os.makedirs(jsonl_output_dir, exist_ok=True)
    jsonl_filename = os.path.join(jsonl_output_dir, f'gsm8k_{len(q_list)}.jsonl')
    with open(jsonl_filename, 'a') as jsonl_file:
        jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

def main(args):
    prompt_original = open(args.prompt_path).read()

    # Load GSM8K dataset
    gsm8k = load_dataset('gsm8k', 'main')
    questions = gsm8k['test']['question'][:args.question_limit]
    answers = gsm8k['test']['answer'][:args.question_limit]

    # Split questions and answers into batches of size k
    batched_questions = [questions[i:i+args.k] for i in range(0, len(questions), args.k)]
    batched_answers = [answers[i:i+args.k] for i in range(0, len(answers), args.k)]

    # Process each batch of questions in main
    for q_list, a_list in zip(batched_questions, batched_answers):
        process_question(args, q_list, a_list, prompt_original)
    
    print("*" * 50)
    print("Running configuration:")
    print("Number of questions to process in parallel: ", args.k)
    print("Prompt file path: ", args.prompt_path)
    print("Output file path: ", args.output_dir)
    print("Dataset: GSM8K")
    print("*" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create parallel GSM8K questions.')
    parser.add_argument('--k', type=int, default=4, help='Number of questions')
    parser.add_argument('--prompt_path', type=str, default='data/LongGenBench_GSM8K_prompt/LongGenBench_prompt.txt', help='Path to the prompt file')
    parser.add_argument('--output_dir', type=str, default='data/LongGenBench', help='Path to the output file')
    parser.add_argument('--question_limit', type=int, default=1319, help='Limit to the number of questions to process')
    args = parser.parse_args()
    main(args)