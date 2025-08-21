import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from arc_tigers.samplers.fixed import FixedSampler
from arc_tigers.training.config import TrainConfig

logger = logging.getLogger(__name__)


def extract_eval_data_config_from_path(experiment_output_dir: str | Path) -> str:
    """
    Extract the evaluation data config from the experiment output directory path.

    For a path like: outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/
    default/eval_outputs/05/random/
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
        msg = f"Path too short to extract eval data config: {path}"
        raise ValueError(msg)

    # Find eval_outputs in the path
    eval_outputs_idx = None
    for i, part in enumerate(parts):
        if part == "eval_outputs":
            eval_outputs_idx = i
            break

    if eval_outputs_idx is None:
        msg = f"Path does not contain 'eval_outputs': {path}"
        raise ValueError(msg)

    # eval_data_config should be the part immediately after eval_outputs
    if eval_outputs_idx + 1 >= len(parts):
        msg = f"No eval data config found after 'eval_outputs' in path: {path}"
        raise ValueError(msg)

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

        if type(data) is list:
            data = data[-1]

        # Validate dataset size consistency
        if dataset_size is None:
            dataset_size = data["dataset_size"]

        sample_data.append(data)

    msg = (
        "Lists found in sample files, expected dicts of final sampling runs. "
        "Taken final index assuming they contain final sampling run."
    )
    logger.warning(msg)

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
    If seed_to_replay is None, creates a shared directory without the seed suffix for
    multi-seed experiments.

    `original_experiment_directory` should be structured like:
    `outputs/{dataset}/{task}/{model_id}/{orig_model}/{orig_hparams}/eval_outputs/{eval_id}/{strategy}`

    Args:
        original_experiment_dir: Path to original experiment.
        new_train_config: Configuration for the new model
        seed_to_replay: Seed being replayed. If None, creates shared directory for all
        seeds

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
