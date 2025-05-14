import os
import json
import argparse
import numpy as np
from metrics_infinitebench import rouge_score  # 假设您已实现 rouge_score 函数

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True, help="Path to the directory containing result files")
    return parser.parse_args()

def evaluate_rouge(dataset, predictions, answers):
    """计算 ROUGE 分数"""
    scores = []
    for prediction, ground_truths in zip(predictions, answers):
        # 使用第一个答案进行评估
        ground_truth = ground_truths[0]
        score = rouge_score(prediction, ground_truth)
        scores.append(score)
    return round(100 * np.mean(scores), 2)

if __name__ == '__main__':
    args = parse_args()

    dataset_list = ["longbook_sum_eng"]
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
                        try:
                            with open(eval_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    data = json.loads(line)
                                    predictions.append(data["pred"])
                                    answers.append(data["answers"])
                        except:
                            print("error")

                        score = evaluate_rouge(dataset, predictions, answers)

                        avg_length = round(np.mean([len(pred) for pred in predictions]), 2)
                        print(avg_length)

                        results_row.append(score)
                        print(f"dataset {dataset} method {method} scores {score}")
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
