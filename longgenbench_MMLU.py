import re
import argparse
from datasets import load_dataset
import pandas as pd
import os
import json

TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies', 
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions'
]

choices = ["A", "B", "C", "D"]

def format_batch_example(df, k, include_answer=True, include_question_mark=True):
    prompt = ""
    for i in range(k):
        if include_question_mark:
            prompt += "Question_{}:\n".format(i+1)
        prompt += df.iloc[i, 0]
        prompt += "\n"
        for j in range(df.shape[1] - 2):
            prompt += "({}) {}".format(choices[j], df.iloc[i, j+1])
        prompt += "\n\n"
    prompt += "\n\n"
    if include_answer:
        for i in range(k):
            prompt += "Answer_{}:\n".format(i+1)
            prompt += "{}\n\n".format(df.iloc[i, df.shape[1] - 1])
    return prompt

def format_questions_answers(prompt_original, task):
    questions = re.findall(r'Q:.*?\(D\)', prompt_original, re.DOTALL)
    answers = re.findall(r'A:.*?The answer is.*?\)', prompt_original, re.DOTALL)

    sys_str = f"The following are multiple choice questions (with answers) about {task}.\n\n"
    output_str = ""
    
    for i, question in enumerate(questions, 1):
        output_str += f"Question_{i}:\n"
        output_str += re.sub(r'^Q: ', '', question.strip()) + "\n\n"

    for i, answer in enumerate(answers, 1):
        output_str += f"Answer_{i}:\n"
        output_str += re.sub(r'^A: ', '', answer.strip()) + "\n\n"

    return output_str, sys_str

def process_question(args, q_list, a_list, prompt_original, sys_str, task):
    prompt_q = 'Examples: \n' + prompt_original + '\n\nFollowing Question: \n'
    for i, q in enumerate(q_list, start=6):
        prompt_q += f'Question_{i}:\n{q}\n'
    prompt_q += '\n'
    
    # Save JSON Lines
    json_data = {
        "task": task,
        "prompt": prompt_q,
        "questions": q_list,
        "answers": a_list
    }
    jsonl_output_dir = args.output_dir
    os.makedirs(jsonl_output_dir, exist_ok=True)
    jsonl_filename = os.path.join(jsonl_output_dir, f'mmlu_{len(q_list)}.jsonl')
    with open(jsonl_filename, 'a') as jsonl_file:
        jsonl_file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

def main(args):
    for task in TASKS:
        # Load MMLU dataset
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", task + "_test.csv"), header=None)
        questions = format_batch_example(test_df, test_df.shape[0], False, False).split('\n\n')
        answers = list(test_df.iloc[:, test_df.shape[1] - 1])

        # Split questions and answers into batches of size k
        prompt_original = json.load(open(args.prompt_path))[task]
        prompt_original, sys_str = format_questions_answers(prompt_original, task)

        batched_questions = [questions[i:i+args.k] for i in range(0, len(questions), args.k)]
        batched_answers = [answers[i:i+args.k] for i in range(0, len(answers), args.k)]

        # Process each batch of questions in main
        for q_list, a_list in zip(batched_questions, batched_answers):
            process_question(args, q_list, a_list, prompt_original, sys_str, task)
            break
        
        
        print(f"Processed task: {task}")
    
    print("*" * 50)
    print("Running configuration:")
    print("Number of questions to process in parallel: ", args.k)
    print("Prompt file path: ", args.prompt_path)
    print("Output file path: ", args.output_dir)
    print("Dataset: MMLU")
    print("*" * 50)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create parallel MMLU questions.')
    parser.add_argument('--k', type=int, default=4, help='Number of questions')
    parser.add_argument('--prompt_path', type=str, default='data/LongGenBench_MMLU_prompt/LongGenBench_prompt.json', help='Path to the prompt file')
    parser.add_argument('--data_dir', type=str, default='data/MMLU', help='Path to the input file')
    parser.add_argument('--output_dir', type=str, default='data/LongGenBench/mmlu/', help='Path to the output file')
    parser.add_argument('--ntrain', type=int, default=5)
    args = parser.parse_args()
    main(args)