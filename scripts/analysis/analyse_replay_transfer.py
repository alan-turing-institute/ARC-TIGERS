import argparse
import os

from arc_tigers.eval.plotting import plot_replay_results
from arc_tigers.eval.utils import get_replay_exp_results

CONFIGS = ["distilbert_default", "gpt2_default", "ModernBERT_fp16", "zero-shot_default"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse replay transfer results")
    parser.add_argument(
        "source_model_dir",
        type=str,
        help="Path to the source model's results directory",
    )
    args = parser.parse_args()
    source_model_dir: str = args.source_model_dir
    clean_source_model_dir = source_model_dir.rstrip(os.sep)

    replay_results, save_dir = get_replay_exp_results(
        clean_source_model_dir, configs=CONFIGS
    )
    plot_replay_results(replay_results, save_dir)
