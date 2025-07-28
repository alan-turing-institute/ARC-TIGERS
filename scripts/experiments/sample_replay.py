import argparse
import logging

from arc_tigers.sampling.replay import replay_experiment_with_new_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay sampling sequence from one experiment on a different model"
    )
    parser.add_argument(
        "original_experiment_dir", help="Path to original experiment output directory"
    )
    parser.add_argument(
        "new_train_config", help="Path to training config for new model"
    )
    parser.add_argument(
        "output_dir", help="Directory to save replayed experiment results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specific seed to replay from original experiment",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--eval_every", type=int, default=50, help="Evaluate metrics every N samples"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    replay_experiment_with_new_model(
        original_experiment_dir=args.original_experiment_dir,
        new_train_config=args.new_train_config,
        output_dir=args.output_dir,
        seed_to_replay=args.seed,
        max_samples=args.max_samples,
        eval_every=args.eval_every,
    )
