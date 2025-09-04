import argparse
import logging

from arc_tigers.replay.run import replay_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay sampling sequence(s) from "
        "one experiment on a different model"
    )
    parser.add_argument(
        "original_experiment_dir", help="Path to original experiment output directory"
    )
    parser.add_argument(
        "new_train_config", help="Path to training config for new model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specific seed to replay from original experiment. "
        "If not provided, replays ALL available seeds",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (per seed if replaying all seeds)",
    )
    parser.add_argument(
        "--eval_every", type=int, default=50, help="Evaluate metrics every N samples"
    )
    parser.add_argument(
        "--shared_output_dir",
        type=str,
        default=None,
        help="Custom shared output directory (only used when replaying all seeds)",
    )
    parser.add_argument(
        "--eval_data_config",
        type=str,
        default=None,
        help="Evaluation data config to use (e.g., '05', '01'). "
        "If not provided, infers from experiment path",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.seed is None:
        print("No seed specified. Will replay ALL available seeds from the experiment.")
    else:
        print(f"Will replay specific seed: {args.seed}")

    if args.eval_data_config:
        print(f"Using provided eval data config: {args.eval_data_config}")
    else:
        print("Will infer eval data config from experiment path")

    replay_experiment(
        original_experiment_dir=args.original_experiment_dir,
        new_train_config=args.new_train_config,
        seed_to_replay=args.seed,
        max_samples=args.max_samples,
        eval_every=args.eval_every,
        shared_output_dir=args.shared_output_dir,
        eval_data_config=args.eval_data_config,
    )
