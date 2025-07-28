import json
from pathlib import Path

import numpy as np
import pandas as pd

print("TRAIN")
model_eval_files = [
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/evaluation_results.json",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_01/distilbert/default/evaluation_results.json",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_001/distilbert/default/evaluation_results.json",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_0001/distilbert/default/evaluation_results.json",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_00001/distilbert/batch128/evaluation_results.json",
]

results = []
for file in model_eval_files:
    with open(file) as f:
        data = json.load(f)
    proc_data = {}
    for key, values in data.items():
        if isinstance(values, list):
            for i, v in enumerate(values):
                proc_data[f"{key}_{i}"] = v
        else:
            proc_data[key] = values
    results.append(proc_data)

df = pd.DataFrame(results)
df.columns = df.columns.str.replace("eval_", "")
df.to_csv("train.csv", index=False)
print(df)

print("\nTEST (w/balanced train)")
model_eval_dirs = [
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/05/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/01/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/001/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/0001/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/00001/accuracy_no_pretrain",
]

results = []
for path in model_eval_dirs:
    with open(Path(path) / "metrics_full.json") as f:
        data = json.load(f)
    results.append(data)

df = pd.DataFrame(results)

pred_0 = []
pred_1 = []
for path in model_eval_dirs:
    with open(Path(path) / "stats_full.json") as f:
        data = json.load(f)
    preds = np.array(data["softmax"]) > 0.5
    pred_0.append(preds[:, 0].sum())
    pred_1.append(preds[:, 1].sum())
df["pred_0"] = pred_0
df["pred_1"] = pred_1

print(df)
df.to_csv("test_balanced_train.csv")

print("\nTEST (w/imbalanced train)")
model_eval_dirs = [
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/05/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_01/distilbert/default/eval_outputs/01/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_001/distilbert/default/eval_outputs/001/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_0001/distilbert/default/eval_outputs/0001/accuracy_no_pretrain",
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_00001/distilbert/batch128/eval_outputs/00001/accuracy_no_pretrain",
]

results = []
for path in model_eval_dirs:
    with open(Path(path) / "metrics_full.json") as f:
        data = json.load(f)
    results.append(data)

df = pd.DataFrame(results)

pred_0 = []
pred_1 = []
for path in model_eval_dirs:
    with open(Path(path) / "stats_full.json") as f:
        data = json.load(f)
    preds = np.array(data["softmax"]) > 0.5
    pred_0.append(preds[:, 0].sum())
    pred_1.append(preds[:, 1].sum())
df["pred_0"] = pred_0
df["pred_1"] = pred_1

print(df)
df.to_csv("test_imbalanced_train.csv")

print("\nTEST (1/100, batch size 256)")
model_eval_dirs = [
    "/bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_001/distilbert/batch256/eval_outputs/001/random",
]

results = []
for path in model_eval_dirs:
    with open(Path(path) / "metrics_full.json") as f:
        data = json.load(f)
    results.append(data)

df = pd.DataFrame(results)

pred_0 = []
pred_1 = []
for path in model_eval_dirs:
    with open(Path(path) / "stats_full.json") as f:
        data = json.load(f)
    preds = np.array(data["softmax"]) > 0.5
    pred_0.append(preds[:, 0].sum())
    pred_1.append(preds[:, 1].sum())
df["pred_0"] = pred_0
df["pred_1"] = pred_1

print(df)
df.to_csv("test_imbalanced_train.csv")
