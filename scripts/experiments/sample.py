import argparse
import logging
import os

import numpy as np

from arc_tigers.data.config import load_data_config
from arc_tigers.sampling.run import sampling_loop
from arc_tigers.sampling.utils import get_eval_outputs_dir
from arc_tigers.training.config import TrainConfig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Perform test set sampling using one \
        of the implemented acquisition strategies."
    )
    parser.add_argument(
        "data_config",
        help="path to the data config file specifying the test set to use",
    )
    parser.add_argument(
        "train_config",
        help="path to the training config file specifying the trained model to use",
    )
    parser.add_argument("strategy", type=str, help="Sampling strategy")

    parser.add_argument("--n_repeats", type=int, required=True)
    parser.add_argument("--max_labels", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="Re-compute metrics every eval_every samples",
    )
    parser.add_argument(
        "--min_labels",
        type=int,
        default=10,
        help="Minimum number of labels to sample before computing metrics",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if the output directory already exists",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation outputs, populated with a default if not set",
    )
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    train_config = TrainConfig.from_path(args.train_config)

    if args.output_dir is None:
        output_dir = get_eval_outputs_dir(train_config, data_config, args.strategy)
    else:
        output_dir = args.output_dir

    if os.path.exists(output_dir) and not args.force:
        msg = (
            f"Output directory {output_dir} already exists. Either remove it, set "
            "--force, or change the output directory."
        )
        raise FileExistsError(msg)

    evaluate_steps = np.arange(
        args.min_labels, args.max_labels + args.eval_every, args.eval_every
    ).tolist()
    if evaluate_steps[-1] > args.max_labels:
        evaluate_steps[-1] = args.max_labels

    sampling_loop(
        data_config,
        train_config,
        n_repeats=args.n_repeats,
        sampling_strategy=args.strategy,
        init_seed=args.seed,
        evaluate_steps=evaluate_steps,
        output_dir=output_dir,
    )
