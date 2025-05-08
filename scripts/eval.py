import argparse
import os

# from transformers import AutoModel, AutoTokenizer

# from arc_tigers.constants import DATA_CONFIG_DIR
# from arc_tigers.eval.utils import compute_metrics
# from arc_tigers.training.utils import get_reddit_data
# from arc_tigers.utils import load_yaml


def main(args):
    experiment_dir = args.experiment_dir
    output_dir = os.path.join(experiment_dir, "eval_outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load the model and tokenizer from the experiment directory
    # model_path = os.path.join(experiment_dir, "model")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path)

    print("Model and tokenizer loaded successfully.")

    # _, eval_dataset = get_reddit_data(**data_config["data_args"], tokenizer=tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate an experiment and save outputs."
    )
    parser.add_argument(
        "experiment_dir", type=str, help="Path to the experiment directory."
    )
    args = parser.parse_args()
    main(args)
