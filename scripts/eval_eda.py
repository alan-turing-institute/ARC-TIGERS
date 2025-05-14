import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from arc_tigers.data.reddit_data import get_reddit_data
from arc_tigers.utils import load_yaml


def main(args):
    counter_dict = {
        "all": Counter(),
        "correct": {0: Counter(), 1: Counter()},
        "incorrect": {0: Counter(), 1: Counter()},
    }
    experiment_dir = args.experiment_dir

    eval_stats = os.path.join(experiment_dir, "stats_full.json")
    with open(eval_stats) as f:
        eval_data = json.load(f)

    accuracy_vector = np.array(eval_data["accuracy"])
    entropy_vector = np.array(eval_data["entropy"])
    softmax_vector = np.array(eval_data["softmax"])
    n_classes = softmax_vector.shape[1]

    softmax_probs = np.max(softmax_vector, axis=1)
    normalised_entropy = entropy_vector / np.log(n_classes)

    correct_indices = np.concatenate(np.argwhere(accuracy_vector == 1))
    incorrect_indices = np.concatenate(np.argwhere(accuracy_vector == 0))

    correct_softmax = softmax_probs[correct_indices]
    incorrect_softmax = softmax_probs[incorrect_indices]
    correct_entropy = normalised_entropy[correct_indices]
    incorrect_entropy = normalised_entropy[incorrect_indices]

    plt.hist(correct_softmax, density=True, bins=100, alpha=0.5, label="correct")
    plt.hist(incorrect_softmax, density=True, bins=100, alpha=0.5, label="incorrect")
    plt.yscale("log")
    plt.legend()
    plt.show()

    plt.hist(correct_entropy, density=True, bins=100, alpha=0.5, label="correct")
    plt.hist(incorrect_entropy, density=True, bins=100, alpha=0.5, label="incorrect")
    plt.yscale("log")
    plt.legend()
    plt.show()

    exp_config = os.path.join(experiment_dir, "../experiment_config.json")
    with open(exp_config) as f:
        exp_config = json.load(f)
    data_config = load_yaml(exp_config["data_config_pth"])
    _, _, test_data, meta_data = get_reddit_data(
        **data_config["data_args"], random_seed=exp_config["seed"], tokenizer=None
    )
    subreddit_label_map = meta_data["test_target_map"]
    label_subreddit_map = {v: k for k, v in subreddit_label_map.items()}

    for input in test_data["text"]:
        counter_dict["all"].update(input.split())

    correct_inputs = test_data[correct_indices]["text"]
    correct_labels = test_data[correct_indices]["label"]

    incorrect_inputs = test_data[incorrect_indices]["text"]
    incorrect_labels = test_data[incorrect_indices]["label"]

    for label, inp in tqdm(zip(incorrect_labels, incorrect_inputs, strict=True)):
        counter_dict["incorrect"][label].update(inp.split())

    for label, inp in tqdm(zip(correct_labels, correct_inputs, strict=True)):
        counter_dict["correct"][label].update(inp.split())

    # remove the top 50 frequent words from the all counter
    most_common_words = counter_dict["all"].most_common(100)
    for word, _ in most_common_words:
        for counter in counter_dict["correct"].values():
            if word in counter:
                counter.pop(word)
        for counter in counter_dict["incorrect"].values():
            if word in counter:
                counter.pop(word)

    for acc in ["incorrect", "correct"]:
        print(f"Most common words in {acc} inputs:")
        print(label_subreddit_map[0])
        print(counter_dict[acc][0].most_common(50))
        print(label_subreddit_map[1])
        print(counter_dict[acc][1].most_common(50))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an experiment and save outputs."
    )
    parser.add_argument(
        "experiment_dir", type=str, help="Path to the experiment directory."
    )
    args = parser.parse_args()
    main(args)
