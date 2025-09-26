import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm

from arc_tigers.data.config import HFDataConfig
from arc_tigers.utils import load_yaml


def main(args):
    stop_words = set(stopwords.words("english"))

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

    plt.hist(correct_softmax, bins=100, alpha=0.5, label="correct")
    plt.hist(incorrect_softmax, bins=100, alpha=0.5, label="incorrect")
    plt.yscale("log")
    plt.xlabel("Predicted Softmax")
    plt.ylabel("Counts")
    plt.legend(title="Prediction type")
    plt.savefig(os.path.join(experiment_dir, "softmax_histogram.pdf"))
    plt.clf()

    plt.hist(correct_entropy, bins=100, alpha=0.5, label="correct")
    plt.hist(incorrect_entropy, bins=100, alpha=0.5, label="incorrect")
    plt.yscale("log")
    plt.xlabel("Normalised Predicted Entropy")
    plt.ylabel("Counts")
    plt.legend(title="Prediction type")
    plt.savefig(os.path.join(experiment_dir, "entropy_histogram.pdf"))
    plt.clf()

    data_config_spec = load_yaml(os.path.join(experiment_dir, "data_config.yaml"))
    data_config = HFDataConfig.from_name(data_config_spec["config_name"])
    test_data = data_config.get_test_split()

    for input in test_data["text"]:
        words = [w for w in input.split() if w.lower() not in stop_words]
        counter_dict["all"].update(words)

    correct_inputs = test_data[correct_indices]["text"]
    correct_labels = test_data[correct_indices]["label"]

    incorrect_inputs = test_data[incorrect_indices]["text"]
    incorrect_labels = test_data[incorrect_indices]["label"]

    for label, inp in tqdm(zip(incorrect_labels, incorrect_inputs, strict=True)):
        words = [w for w in inp.split() if w.lower() not in stop_words]
        counter_dict["incorrect"][label].update(words)

    for label, inp in tqdm(zip(correct_labels, correct_inputs, strict=True)):
        words = [w for w in inp.split() if w.lower() not in stop_words]
        counter_dict["correct"][label].update(words)

    # remove the top 50 frequent words from the all counter
    most_common_words = counter_dict["all"].most_common(50)
    for word, _ in most_common_words:
        for counter in counter_dict["correct"].values():
            if word in counter:
                counter.pop(word)
        for counter in counter_dict["incorrect"].values():
            if word in counter:
                counter.pop(word)

    for class_label in range(n_classes):
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        for ax_idx, acc in enumerate(["incorrect", "correct"]):
            top_50 = counter_dict[acc][class_label].most_common(50)
            words, counts = zip(*top_50, strict=True)
            x_pos = np.arange(len(words))
            axes[ax_idx].set_title(f"{acc} inputs")
            axes[ax_idx].bar(x_pos, counts, align="center")
            axes[ax_idx].set_xticks(x_pos, words, rotation=70)
            axes[ax_idx].set_ylabel("Counts")
            axes[ax_idx].set_xlabel("Words")
        fig.tight_layout()
        fig.savefig(
            os.path.join(experiment_dir, f"class_{class_label}_word_counts.pdf")
        )
        fig.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Produce some stats and EDA plots on model performance for an experiment, "
            "such as words present in correct and incorrect predictions."
        )
    )
    parser.add_argument(
        "experiment_dir", type=str, help="Path to the experiment directory."
    )
    args = parser.parse_args()
    main(args)
