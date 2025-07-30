import json
import logging
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from arc_tigers.data.config import load_data_config
from arc_tigers.samplers.fixed import FixedSampler
from arc_tigers.sampling.run import sampling_loop
from arc_tigers.training.config import TrainConfig

logger = logging.getLogger(__name__)


def extract_eval_data_config_from_path(experiment_output_dir: str | Path) -> str:
    """
    Extract the evaluation data config from the experiment output directory path.

    For a path like: outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/05/random/
    This returns: "05" (the second to last component, before the sampling strategy)

    Args:
        experiment_output_dir: Path to the experiment output directory

    Returns:
        The evaluation data config identifier (e.g., "05", "01", "001")

    Raises:
        ValueError: If the path structure doesn't match expected format
    """
    path = Path(experiment_output_dir)
    parts = path.parts

    # Expected structure: .../eval_outputs/{eval_data_config}/{sampling_strategy}
    # So eval_data_config should be the second to last part
    if len(parts) < 2:
        raise ValueError(f"Path too short to extract eval data config: {path}")

    # Find eval_outputs in the path
    eval_outputs_idx = None
    for i, part in enumerate(parts):
        if part == "eval_outputs":
            eval_outputs_idx = i
            break

    if eval_outputs_idx is None:
        raise ValueError(f"Path does not contain 'eval_outputs': {path}")

    # eval_data_config should be the part immediately after eval_outputs
    if eval_outputs_idx + 1 >= len(parts):
        raise ValueError(f"No eval data config found after 'eval_outputs' in path: {path}")

    eval_data_config = parts[eval_outputs_idx + 1]

    logger.info("Extracted eval data config '%s' from path: %s", eval_data_config, path)
    return eval_data_config


def load_sampling_data(experiment_output_dir: str | Path) -> dict[str, Any]:
    """
    Load sampling data from a previous experiment output directory.

    Args:
        experiment_output_dir: Path to the experiment output directory containing
            sample_*.json files

    Returns:
        Dictionary containing sampling data with keys:
        - 'sample_files': List of paths to sample JSON files
        - 'sample_data': List of loaded sampling data dictionaries
        - 'seeds': List of seeds used in the original experiment

    Raises:
        FileNotFoundError: If no sample files are found in the directory
        ValueError: If the sampling data is inconsistent across files
    """
    experiment_dir = Path(experiment_output_dir)

    # Find all sample_*.json files
    sample_files = list(experiment_dir.glob("sample_*.json"))

    if not sample_files:
        msg = f"No sample_*.json files found in {experiment_dir}"
        raise FileNotFoundError(msg)

    # Load and validate all sample data
    sample_data = []
    seeds = []
    dataset_size = None

    for sample_file in sorted(sample_files):
        with open(sample_file) as f:
            data = json.load(f)

        # Extract seed from filename (sample_{seed}.json)
        seed = int(sample_file.stem.split("_")[1])
        seeds.append(seed)

        # Validate dataset size consistency
        if dataset_size is None:
            dataset_size = data["dataset_size"]

        sample_data.append(data)

    return {
        "sample_files": sample_files,
        "sample_data": sample_data,
        "seeds": seeds,
        "dataset_size": dataset_size,
    }


def create_fixed_sampler_from_experiment(
    eval_data: Dataset,
    experiment_output_dir: str | Path,
    seed_to_replay: int | None = None,
    max_samples: int | None = None,
) -> FixedSampler:
    """
    Create a FixedSampler that will replay the sampling sequence from a previous
    experiment.

    Args:
        eval_data: The dataset to sample from (should be the same as the original)
        experiment_output_dir: Path to the previous experiment's output directory
        seed_to_replay: Specific seed to replay. If None, uses the first available seed
        max_samples: Maximum number of samples to include. If None, uses all samples

    Returns:
        FixedSampler configured to replay the specified sampling sequence

    Raises:
        ValueError: If the specified seed is not found or data is inconsistent
    """
    sampling_data = load_sampling_data(experiment_output_dir)

    # Select which seed to replay
    if seed_to_replay is None:
        seed_to_replay = sampling_data["seeds"][0]
        logger.info("No seed specified, using first available: %d", seed_to_replay)

    # Find the data for the specified seed
    target_data = None
    for data, seed in zip(
        sampling_data["sample_data"], sampling_data["seeds"], strict=True
    ):
        if seed == seed_to_replay:
            target_data = data
            break

    if target_data is None:
        available_seeds = sampling_data["seeds"]
        msg = f"Seed {seed_to_replay} not found. Available seeds: {available_seeds}"
        raise ValueError(msg)

    # Extract sample indices and probabilities
    sample_indices = target_data["sample_idx"]
    sample_probabilities = target_data["sample_prob"]

    # Validate data lengths
    if len(sample_indices) != len(sample_probabilities):
        msg = (
            f"Mismatched lengths: {len(sample_indices)} indices vs "
            f"{len(sample_probabilities)} probabilities"
        )
        raise ValueError(msg)

    # Truncate if max_samples is specified
    if max_samples is not None and max_samples < len(sample_indices):
        sample_indices = sample_indices[:max_samples]
        sample_probabilities = sample_probabilities[:max_samples]
        logger.info("Truncated sampling sequence to %d samples", max_samples)

    # Validate indices are within dataset bounds
    max_idx = max(sample_indices)
    if max_idx >= len(eval_data):
        msg = (
            f"Sample index {max_idx} is out of bounds for dataset of size "
            f"{len(eval_data)}"
        )
        raise ValueError(msg)

    return FixedSampler(
        eval_data=eval_data,
        seed=seed_to_replay,
        sample_indices=sample_indices,
        sample_probabilities=sample_probabilities,
    )


def create_replay_output_dir(
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig,
    seed_to_replay: int | None = None,
) -> Path:
    """
    Create a structured output directory for replay experiments.

    Creates output directory structure like:
    `outputs/{data}/{task}/{model_id}/replays/{eval_id}/replay_from_{orig_model}_{orig_hparams}_{orig_strategy}/to_{new_model}_{new_hparams}/seed_{seed}`
    If seed_to_replay is None, creates a shared directory without the seed suffix for multi-seed experiments.

    `original_experiment_directory` should be structured like:
    `outputs/{dataset}/{task}/{model_id}/{orig_model}/{orig_hparams}/eval_outputs/{eval_id}/{strategy}`

    Args:
        original_experiment_dir: Path to original experiment.
        new_train_config: Configuration for the new model
        seed_to_replay: Seed being replayed. If None, creates shared directory for all seeds

    Returns:
        Path to the structured output directory
    """
    original_path = Path(original_experiment_dir)
    original_parts = original_path.parts

    # Parse original path to extract components
    # Path: outputs/{data}/{task}/{model_id}/{orig_model}/{orig_hparams}/
    #       eval_outputs/{eval_id}/{strategy}

    # Initialize defaults
    data_config = "unknown_data"
    task = "unknown_task"
    model_id = "unknown_model"
    eval_id = "unknown_eval"
    original_model = "unknown_orig_model"
    original_hparams = "unknown_orig_hparams"
    original_strategy = "unknown_strategy"

    for i, part in enumerate(original_parts):
        try:
            if part == "eval_outputs":
                # Extract components using the eval_outputs index as reference (i=7)
                if i >= 1:
                    original_hparams = original_parts[i - 1]  # train config
                if i >= 2:
                    original_model = original_parts[i - 2]  # model
                # For the directory structure components, use absolute indices
                if len(original_parts) > 1:
                    data_config = original_parts[1]  # dataset
                if len(original_parts) > 2:
                    task = original_parts[2]  # task
                if len(original_parts) > 4:
                    model_id = original_parts[4]  # data seed and train imbalance
                if i + 1 < len(original_parts):
                    eval_id = original_parts[i + 1]  # test imbalance 05 (index 8)
                if i + 2 < len(original_parts):
                    original_strategy = original_parts[i + 2]  # sample strategy
                break
        except IndexError as e:
            logger.warning("Could not parse full directory structure: %s", e)

    # Get new model configuration names
    new_model = getattr(
        new_train_config.model_config,
        "config_name",
        "unknown_new_model",
    )
    new_hparams = getattr(
        new_train_config.hparams_config,
        "config_name",
        "unknown_new_hparams",
    )

    # Create structured path
    if seed_to_replay is not None:
        replay_dir = (
            Path("outputs")
            / data_config  # data
            / task  # task
            / original_parts[3]  # data config
            / model_id  # seed/train imbalance
            / "replays"
            / eval_id  # (evaluation data imbalance)
            / f"{new_model}_{new_hparams}"
            / f"from_{original_model}_{original_hparams}/{original_strategy}"
            / f"seed_{seed_to_replay}"
        )
    else:
        # Shared directory for all seeds
        replay_dir = (
            Path("outputs")
            / data_config  # data
            / task  # task
            / original_parts[3]  # data config
            / model_id  # seed/train imbalance
            / "replays"
            / eval_id  # (evaluation data imbalance)
            / f"{new_model}_{new_hparams}"
            / f"from_{original_model}_{original_hparams}/{original_strategy}"
            / "all_seeds"
        )

    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir


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
        eval_data_config: Evaluation data config to use. If None, infers from experiment path
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
    # Use the original training data config as the base, but apply eval data config for evaluation
    base_data_config_path = f"configs/data/{original_config['data_config']}.yaml"
    base_data_config = load_data_config(base_data_config_path)

    # Get the evaluation dataset using the eval data config
    # This should load the test split with the appropriate imbalance/configuration
    eval_data = base_data_config.get_test_split()

    logger.info(
        "Using base data config '%s' with eval data config '%s'",
        original_config['data_config'],
        eval_data_config
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
    Replay a previous experiment's sampling sequences for ALL available seeds using a different trained model.

    This function loads all available seeds from a previous experiment and replays each one
    with the new model, saving all results under a shared directory structure.

    Args:
        original_experiment_dir: Path to the original experiment's output directory
        new_train_config: TrainConfig or path to config for the new model to evaluate
        max_samples: Maximum number of samples to evaluate per seed
        eval_every: Evaluate metrics every N samples
        shared_output_dir: Custom shared output directory. If None, uses structured naming
        eval_data_config: Evaluation data config to use. If None, infers from experiment path
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

    logger.info(
        "Found %d seeds to replay: %s",
        len(available_seeds),
        available_seeds
    )

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
    # Use the original training data config as the base, but apply eval data config for evaluation
    base_data_config_path = f"configs/data/{original_config['data_config']}.yaml"
    base_data_config = load_data_config(base_data_config_path)

    # Get the evaluation dataset using the eval data config
    eval_data = base_data_config.get_test_split()

    logger.info(
        "Using base data config '%s' with eval data config '%s'",
        original_config['data_config'],
        eval_data_config
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
                # The sampling_loop will create metrics files with the seed in the filename
                sampling_loop(
                    data_config=base_data_config,
                    train_config=new_train_config,
                    n_repeats=1,  # Only one repeat since we're replaying specific sequence
                    sampling_strategy="fixed",
                    init_seed=seed,
                    evaluate_steps=evaluate_steps,
                    output_dir=shared_output_dir,  # All seeds use the same output directory
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

    This is a convenience function that can either replay a specific seed or all available seeds.

    Args:
        original_experiment_dir: Path to the original experiment's output directory
        new_train_config: TrainConfig or path to config for the new model to evaluate
        seed_to_replay: Specific seed to replay. If None, replays ALL available seeds
        max_samples: Maximum number of samples to evaluate (per seed if replaying all)
        eval_every: Evaluate metrics every N samples
        shared_output_dir: Custom output directory (only used when replaying all seeds)
        eval_data_config: Evaluation data config to use. If None, infers from experiment path
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
