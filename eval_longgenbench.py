import os
import json
import argparse
import numpy as np
import re

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default=None)
    return parser.parse_args(args)

def extract_final_answer(answer):
    """Extracts the final answer from the answer string."""
    match = re.search(r'####\s*(\d+)', answer)
    return match.group(1) if match else None

def extract_predicted_answers(pred):
    """Extracts the predicted answers from the pred string."""
    matches = re.findall(r'Answer_\d+:\s*.*?answer is (\d+)', pred, re.DOTALL)
    return matches

def compare_answers(pred, answers):
    # Extract final answers from the answers list
    expected_answers = [extract_final_answer(ans) for ans in answers]
    
    # Extract predicted answers from the pred string
    predicted_answers = extract_predicted_answers(pred)

    # print(f'Expected: {expected_answers}')
    # print(f'Predicted: {predicted_answers}')
    
    # Compare the two lists and calculate accuracy
    results = {}
    correct_count = 0
    for i, (expected, predicted) in enumerate(zip(expected_answers, predicted_answers)):
        is_correct = expected == predicted
        results[f'Answer_{i+9}'] = {
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        }
        if is_correct:
            correct_count += 1

    # Calculate accuracy
    total_questions = len(expected_answers)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    # print(f'Accuracy: {accuracy:.4f}')
    return accuracy


def extract_predicted_choices(pred):
    """Extracts the predicted answers (A, B, C, D, E, etc.) from the pred string, considering Answer_... patterns."""
    matches = re.findall(r'Answer_\d+:\s*.*?answer is \((.*?)\)', pred)
    return matches


def compare_choices(pred, answers):
    # Extract final answers from the answers list
    expected_answers = answers
    
    # Extract predicted answers from the pred string
    predicted_answers = extract_predicted_choices(pred)

    # print(f'Expected: {expected_answers}')
    # print(f'Predicted: {predicted_answers}')
    
    # Compare the two lists and calculate accuracy
    results = {}
    correct_count = 0
    for i, (expected, predicted) in enumerate(zip(expected_answers, predicted_answers)):
        is_correct = expected == predicted
        results[f'Answer_{i+6}'] = {
            'expected': expected,
            'predicted': predicted,
            'correct': is_correct
        }
        if is_correct:
            correct_count += 1

    # Calculate accuracy
    total_questions = len(expected_answers)
    accuracy = correct_count / total_questions if total_questions > 0 else 0.0
    # print(f'Accuracy: {accuracy:.4f}')
    return accuracy

def scorer(dataset, predictions, answers):
    scores = []
    for (prediction, ground_truths) in zip(predictions, answers):
        if dataset in ["gsm8k"]:
            scores.append(compare_answers(prediction, ground_truths))
        elif dataset in ["mmlu","csqa"]:
            scores.append(compare_choices(prediction, ground_truths))

    return round(100 * np.mean(scores), 4)

# if __name__ == '__main__':
#     args = parse_args()
    
#     dataset_list = [
#         "gsm8k",
#         "mmlu",
#         "csqa",
#         ]
    
#     results_list = [
#         ["dataset"],
#         ["SnapKV"],
#         ["StreamingLLM"],
#         ["H2O"],
#         ["PyramidKV"],
#         ["PyramidInfer"],
#         ["ALLKV"],
#     ]
    
#     for dataset in dataset_list:
        
#         results_list[0].append(dataset)
        
#         for idx, method in enumerate(["SnapKV", "StreamingLLM", "H2O", "PyramidKV", "PyramidInfer" ,"ALLKV"]):
#             try:
#                 args.method = method
#                 args.dataset = dataset
#                 args.eval_file = os.path.join(args.results_dir,dataset,f"{method}.json")
                
#                 scores = dict()

#                 predictions, answers, lengths = [], [], []
#                 acc = []
#                 # dataset = filename.split('.')[0]
#                 with open(args.eval_file, "r", encoding="utf-8") as f:
#                     for line in f:
#                         try:
#                             data = json.loads(line)
#                             predictions.append(data["pred"])
#                             answers.append(data["answers"])
#                             if "length" in data:
#                                 lengths.append(data["length"])
#                         except:
#                             print("error")

#                 score = scorer(args.dataset, predictions, answers)

#                 avg_length = round(np.mean(lengths), 2)
#                 print(avg_length)

#                 scores[args.dataset] = score

#                 output_dir = os.path.dirname(args.eval_file)
                
#                 results_list[idx+1].append(score)
                
#                 with open(os.path.join(output_dir, "metrics.json"), "w") as f:
#                     json.dump(scores, f, ensure_ascii=False, indent=4)
            
#                 print(f"dataset {args.dataset} method {args.method} scores {scores}")
#             except:
                
#                 results_list[idx+1].append(-1)
                
#                 print(f"dataset {args.dataset} method {args.method} scores {None}")
                
#     import csv
#     with open(os.path.join(args.results_dir,f"results.csv"), 'w') as fp:
#         writer = csv.writer(fp)
#         writer.writerows(results_list)

if __name__ == '__main__':
    args = parse_args()
    
    dataset_list = ["gsm8k", "mmlu", "csqa"]
    method_map = {
        "H2O": ["None", "h2o", "slide", "adaptive", "discontinuous"],
        "PyramidKV": ["None", "slide", "adaptive", "discontinuous"],
        "PyramidInfer": ["None", "pyramidinfer"],
        "StreamingLLM": ["None", "slm"],
        "SnapKV": ["None", "slide", "adaptive", "discontinuous"],
        "ALLKV": ["None", "slide", "adaptive", "discontinuous"]
    }
    
    results_row = ["prefill+decoding"]

    for dataset in dataset_list:
        score_found = False
        for prefill in method_map.keys():
            for decoding in method_map[prefill]:
                method = f"{prefill}+{decoding}"
                if method in args.results_dir:
                    results_row[0]=method
                    eval_file = os.path.join(args.results_dir, dataset, f"{prefill}.json")
                    if os.path.exists(eval_file):
                        
                        # 读取文件并计算准确率
                        predictions, answers = [], []
                        with open(eval_file, "r", encoding="utf-8") as f:
                            for line in f:
                                data = json.loads(line)
                                predictions.append(data["pred"])
                                answers.append(data["answers"])
                        
                        score = scorer(dataset, predictions, answers)
                        results_row.append(score)
                        score_found = True
                        break  # 找到一个匹配的文件后跳出
            if score_found:
                break
        if not score_found:
            results_row.append(-1)  # 如果没有找到匹配的文件，填充 -1

    # 保存结果
    import csv
    output_csv_path = os.path.join(args.results_dir, "results.csv")
    with open(output_csv_path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["dataset"] + dataset_list)
        writer.writerow(results_row)
