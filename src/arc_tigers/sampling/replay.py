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

    logger.info("Found %d sample files in %s", len(sample_files), experiment_dir)

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
        log_msg = (
            f"Loaded sampling data from {sample_file.name}: "
            f"{len(data['sample_idx'])} samples, seed={seed}"
        )
        logger.info(log_msg)

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

    logger.info(
        "Creating FixedSampler with %d samples from seed %d",
        len(sample_indices),
        seed_to_replay,
    )

    return FixedSampler(
        eval_data=eval_data,
        seed=seed_to_replay,
        sample_indices=sample_indices,
        sample_probabilities=sample_probabilities,
    )


def create_replay_output_dir(
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig,
    seed_to_replay: int,
) -> Path:
    """
    Create a structured output directory for replay experiments.

    Creates output directory structure like:
    `outputs/{data}/{task}/{model_id}/replays/{eval_id}/replay_from_{orig_model}_{orig_hparams}_{orig_strategy}/to_{new_model}_{new_hparams}/seed_{seed}`

    `original_experiment_directory` should be structured like:
    `outputs/{dataset}/{task}/{model_id}/{orig_model}/{orig_hparams}/eval_outputs/{eval_id}/{strategy}`

    Args:
        original_experiment_dir: Path to original experiment.
        new_train_config: Configuration for the new model
        seed_to_replay: Seed being replayed

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
    replay_dir = (
        Path("outputs")
        / data_config  # data
        / task  # task
        / original_parts[3]  # data config
        / model_id  # seed/train imbalance
        / "replays"
        / eval_id  # (evaluation data imbalance)
        / f"from_{original_model}_{original_hparams}_{original_strategy}"
        / f"to_{new_model}_{new_hparams}"
        / f"seed_{seed_to_replay}"
    )

    replay_dir.mkdir(parents=True, exist_ok=True)
    return replay_dir


def save_replay_metadata(
    output_dir: Path,
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig,
    seed_to_replay: int,
    max_samples: int | None,
    n_samples_replayed: int,
) -> None:
    """
    Save metadata about the replay experiment.

    Args:
        output_dir: Directory where replay results are saved
        original_experiment_dir: Path to original experiment
        new_train_config: Configuration for the new model
        seed_to_replay: Seed that was replayed
        max_samples: Maximum samples specified (if any)
        n_samples_replayed: Actual number of samples replayed
    """
    metadata = {
        "replay_info": {
            "original_experiment_dir": str(original_experiment_dir),
            "new_model_config": new_train_config.model_config,
            "new_hparams_config": new_train_config.hparams_config,
            "data_config": new_train_config.data_config,
            "seed_replayed": seed_to_replay,
            "max_samples_requested": max_samples,
            "actual_samples_replayed": n_samples_replayed,
        },
        "original_config_path": str(Path(original_experiment_dir) / "config.yaml"),
        "replay_strategy": "fixed_sampler",
        "creation_timestamp": str(Path(output_dir).stat().st_mtime),
    }

    metadata_file = output_dir / "replay_metadata.yaml"
    with open(metadata_file, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, indent=2)

    logger.info("Saved replay metadata to %s", metadata_file)


def replay_experiment_with_new_model(
    original_experiment_dir: str | Path,
    new_train_config: TrainConfig | str | Path,
    seed_to_replay: int,
    max_samples: int | None = None,
    eval_every: int = 50,
) -> None:
    """
    Replay a previous experiment's sampling sequence using a different trained model.

    This function loads the sampling data from a previous experiment and applies
    the exact same sampling sequence to a new model, enabling direct comparison
    of model performance with identical sampling strategies.

    Args:
        original_experiment_dir: Path to the original experiment's output directory
        new_train_config: TrainConfig or path to config for the new model to evaluate
        output_dir: Directory to save the new experiment results
        seed_to_replay: Specific seed from original experiment to replay
        max_samples: Maximum number of samples to evaluate
        eval_every: Evaluate metrics every N samples
    """
    original_dir = Path(original_experiment_dir)

    # Load original experiment configuration
    with open(original_dir / "config.yaml") as f:
        original_config = yaml.safe_load(f)

    # Load the new train config
    if isinstance(new_train_config, str | Path):
        new_train_config = TrainConfig.from_path(new_train_config)

    # Create structured output directory if not provided as absolute path
    output_dir = create_replay_output_dir(
        original_experiment_dir=original_dir,
        new_train_config=new_train_config,
        seed_to_replay=seed_to_replay,
    )
    logger.info("Using structured output directory: %s", output_dir)

    # Load data config (should be the same as original)
    data_config_path = f"configs/data/{original_config['data_config']}.yaml"
    data_config = load_data_config(data_config_path)

    # Get the test dataset
    eval_data = data_config.get_test_split()

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

    # Run the sampling loop
    sampling_loop(
        data_config=data_config,
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

    # Save replay metadata
    save_replay_metadata(
        output_dir=Path(output_dir),
        original_experiment_dir=original_dir,
        new_train_config=new_train_config,
        seed_to_replay=seed_to_replay,
        max_samples=max_samples,
        n_samples_replayed=n_samples,
    )

    logger.info("Replay experiment completed. Results saved to %s", output_dir)
