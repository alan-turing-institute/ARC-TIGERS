import logging
from pathlib import Path

import yaml

from arc_tigers.data.config import load_data_config
from arc_tigers.replay.utils import (
    create_fixed_sampler_from_experiment,
    create_replay_output_dir,
    extract_eval_data_config_from_path,
    load_sampling_data,
)
from arc_tigers.sampling.run import sampling_loop
from arc_tigers.training.config import TrainConfig

logger = logging.getLogger(__name__)


def replay_experiment_with_new_model(
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig | str | Path,
    seed_to_replay: int,
    max_samples: int | None = None,
    eval_every: int = 50,
    eval_data_config: str | None = None,
) -> None:
    """
    Replay a previous experiment's sampling sequence using a different trained model.

    This function loads the sampling data from a previous experiment and applies
    the exact same sampling sequence to a new model, enabling direct comparison
    of model performance with identical sampling strategies.

    Args:
        original_experiment_dir: Path to the original experiment's output directory
        new_train_config: TrainConfig or path to config for the new model to evaluate
        seed_to_replay: Specific seed from original experiment to replay
        max_samples: Maximum number of samples to evaluate
        eval_every: Evaluate metrics every N samples
        eval_data_config: Evaluation data config to use. If None, infers from experiment
        path
    """
    original_dir = Path(original_experiment_dir)

    # Load original experiment configuration
    with open(original_dir / "config.yaml") as f:
        original_config = yaml.safe_load(f)

    # Load the new train config
    if isinstance(new_train_config, str | Path):
        new_train_config = TrainConfig.from_path(new_train_config)

    # Extract evaluation data config from path if not provided
    if eval_data_config is None:
        eval_data_config = extract_eval_data_config_from_path(original_dir)
        logger.info("Inferred eval data config from path: %s", eval_data_config)
    else:
        logger.info("Using provided eval data config: %s", eval_data_config)

    # Create structured output directory
    output_dir = create_replay_output_dir(
        original_experiment_dir=original_dir,
        new_train_config=new_train_config,
        seed_to_replay=seed_to_replay,
    )
    logger.info("Using structured output directory: %s", output_dir)

    # Load the base data config and get the evaluation dataset
    # Use the original training data config as the base,
    # but apply eval data config for evaluation
    base_data_config_path = f"configs/data/{original_config['data_config']}.yaml"
    base_data_config = load_data_config(base_data_config_path)

    # Get the evaluation dataset using the eval data config
    # This should load the test split with the appropriate imbalance/configuration
    eval_data = base_data_config.get_test_split()

    logger.info(
        "Using base data config '%s' with eval data config '%s'",
        original_config["data_config"],
        eval_data_config,
    )

    # Create fixed sampler from original experiment
    fixed_sampler = create_fixed_sampler_from_experiment(
        eval_data=eval_data,
        experiment_output_dir=original_dir,
        seed_to_replay=seed_to_replay,
        max_samples=max_samples,
    )

    # Determine evaluation steps
    n_samples = len(fixed_sampler._sample_order)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    evaluate_steps = list(range(eval_every, n_samples + 1, eval_every))
    if not evaluate_steps or evaluate_steps[-1] != n_samples:
        evaluate_steps.append(n_samples)

    logger.info(
        "Replaying experiment with %d samples, evaluating at steps: %s",
        n_samples,
        evaluate_steps,
    )

    # Run the sampling loop with replay parameters
    sample_indices = fixed_sampler._sample_order
    sample_probabilities = fixed_sampler._sample_prob

    # Suppress verbose metrics logging during replay
    metrics_logger = logging.getLogger("arc_tigers.sampling.metrics")
    original_level = metrics_logger.level
    metrics_logger.setLevel(logging.WARNING)

    try:
        # Run the sampling loop
        sampling_loop(
            data_config=base_data_config,
            train_config=new_train_config,
            n_repeats=1,  # Only one repeat since we're replaying specific sequence
            sampling_strategy="fixed",
            init_seed=seed_to_replay,
            evaluate_steps=evaluate_steps,
            output_dir=output_dir,
            retrain_every=1,
            surrogate_pretrain=False,
            replay_sample_indices=sample_indices,
            replay_sample_probabilities=sample_probabilities,
        )
    finally:
        # Restore original logging level
        metrics_logger.setLevel(original_level)

    logger.info("Replay experiment completed. Results saved to %s", output_dir)


def replay_all_seeds_with_new_model(
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig | str | Path,
    max_samples: int | None = None,
    eval_every: int = 50,
    shared_output_dir: str | Path | None = None,
    eval_data_config: str | None = None,
) -> None:
    """
    Replay a previous experiment's sampling sequences for ALL available seeds using a
    different trained model.

    This function loads all available seeds from a previous experiment and replays each
    one with the new model, saving all results under a shared directory structure.

    Args:
        original_experiment_dir: Path to the original experiment's output directory
        new_train_config: TrainConfig or path to config for the new model to evaluate
        max_samples: Maximum number of samples to evaluate per seed
        eval_every: Evaluate metrics every N samples
        shared_output_dir: Custom shared output directory. If None, uses structured
        naming
        eval_data_config: Evaluation data config to use. If None, infers from experiment
        path
    """
    original_dir = Path(original_experiment_dir)

    # Load original experiment configuration
    with open(original_dir / "config.yaml") as f:
        original_config = yaml.safe_load(f)

    # Load the new train config
    if isinstance(new_train_config, str | Path):
        new_train_config = TrainConfig.from_path(new_train_config)

    # Extract evaluation data config from path if not provided
    if eval_data_config is None:
        eval_data_config = extract_eval_data_config_from_path(original_dir)
        logger.info("Inferred eval data config from path: %s", eval_data_config)
    else:
        logger.info("Using provided eval data config: %s", eval_data_config)

    # Load sampling data to get all available seeds
    sampling_data = load_sampling_data(original_dir)
    available_seeds = sampling_data["seeds"]

    logger.info("Found %d seeds to replay: %s", len(available_seeds), available_seeds)

    # Create shared output directory
    if shared_output_dir is None:
        shared_output_dir = create_replay_output_dir(
            original_experiment_dir=original_dir,
            new_train_config=new_train_config,
            seed_to_replay=None,  # Indicates shared directory for all seeds
        )
    else:
        shared_output_dir = Path(shared_output_dir)
        shared_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Using shared output directory: %s", shared_output_dir)

    # Load the base data config and get the evaluation dataset
    # Use the original training data config as the base,
    # but apply eval data config for evaluation
    base_data_config_path = f"configs/data/{original_config['data_config']}.yaml"
    base_data_config = load_data_config(base_data_config_path)

    # Get the evaluation dataset using the eval data config
    eval_data = base_data_config.get_test_split()

    logger.info(
        "Using base data config '%s' with eval data config '%s'",
        original_config["data_config"],
        eval_data_config,
    )

    # Replay each seed
    for seed in available_seeds:
        try:
            # Create fixed sampler from original experiment
            fixed_sampler = create_fixed_sampler_from_experiment(
                eval_data=eval_data,
                experiment_output_dir=original_dir,
                seed_to_replay=seed,
                max_samples=max_samples,
            )

            # Determine evaluation steps
            n_samples = len(fixed_sampler._sample_order)
            if max_samples:
                n_samples = min(n_samples, max_samples)

            evaluate_steps = list(range(eval_every, n_samples + 1, eval_every))
            if not evaluate_steps or evaluate_steps[-1] != n_samples:
                evaluate_steps.append(n_samples)

            # Run the sampling loop with replay parameters
            sample_indices = fixed_sampler._sample_order
            sample_probabilities = fixed_sampler._sample_prob

            # Suppress verbose metrics logging during replay
            metrics_logger = logging.getLogger("arc_tigers.sampling.metrics")
            original_level = metrics_logger.level
            metrics_logger.setLevel(logging.WARNING)

            try:
                # Run the sampling loop directly in the shared output directory
                # The sampling_loop will create metrics files
                # with the seed in the filename
                sampling_loop(
                    data_config=base_data_config,
                    train_config=new_train_config,
                    n_repeats=1,
                    sampling_strategy="fixed",
                    init_seed=seed,
                    evaluate_steps=evaluate_steps,
                    output_dir=shared_output_dir,
                    retrain_every=1,
                    surrogate_pretrain=False,
                    replay_sample_indices=sample_indices,
                    replay_sample_probabilities=sample_probabilities,
                )
            finally:
                # Restore original logging level
                metrics_logger.setLevel(original_level)

        except Exception as e:
            logger.error("Failed to replay seed %d: %s", seed, e)
            # Continue with other seeds instead of failing completely
            continue

    logger.info("All results saved under: %s", shared_output_dir)


def replay_experiment(
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig | str | Path,
    seed_to_replay: int | None = None,
    max_samples: int | None = None,
    eval_every: int = 50,
    shared_output_dir: str | Path | None = None,
    eval_data_config: str | None = None,
) -> None:
    """
    Replay a previous experiment's sampling sequence(s) using a different trained model.

    This is a convenience function that can either
    replay a specific seed or all available seeds.

    Args:
        original_experiment_dir: Path to the original experiment's output directory
        new_train_config: TrainConfig or path to config for the new model to evaluate
        seed_to_replay: Specific seed to replay. If None, replays ALL available seeds
        max_samples: Maximum number of samples to evaluate (per seed if replaying all)
        eval_every: Evaluate metrics every N samples
        shared_output_dir: Custom output directory (only used when replaying all seeds)
        eval_data_config: Evaluation data config to use.
        If None, infers from experiment path
    """
    if seed_to_replay is None:
        logger.info("No specific seed provided. Replaying ALL available seeds.")
        replay_all_seeds_with_new_model(
            original_experiment_dir=original_experiment_dir,
            new_train_config=new_train_config,
            max_samples=max_samples,
            eval_every=eval_every,
            shared_output_dir=shared_output_dir,
            eval_data_config=eval_data_config,
        )
    else:
        logger.info("Replaying specific seed: %d", seed_to_replay)
        replay_experiment_with_new_model(
            original_experiment_dir=original_experiment_dir,
            new_train_config=new_train_config,
            seed_to_replay=seed_to_replay,
            max_samples=max_samples,
            eval_every=eval_every,
            eval_data_config=eval_data_config,
        )
