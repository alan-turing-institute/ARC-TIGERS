#!/bin/bash
#SBATCH --qos turing
#SBATCH --job-name run_sampling
#SBATCH --time 0-10:00:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /bask/projects/v/vjgo8416-tigers/ARC-TIGERS/slurm_logs/run_sampling-%j.out
#SBATCH --cpus-per-gpu 18

# Load required modules here
module purge
module load baskerville
module load bask-apps/live/live
module load Python/3.10.8-GCCcore-12.2.0

PROJECT_ROOT="/bask/projects/v/vjgo8416-tigers/ARC-TIGERS"

source "$PROJECT_ROOT/.venv/bin/activate"

# change working directory
cd $PROJECT_ROOT

# change huggingface cache to be in project dir rather than user home
export HF_HOME="/bask/projects/v/vjgo8416-tigers/hf_cache"
# Usage: ./run_resampling.sh <base_path> <target_directory>
# Example: ./run_resampling.sh outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/05 /path/to/output

if [ $# -ne 2 ]; then
    echo "Usage: $0 <base_path> <target_directory>"
    echo "Example: $0 outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs/05 /configs/training/balanced_train_gpt2.yaml"
    exit 1
fi

BASE_PATH=$1
TARGET_CONFIG=$2

echo "Running resampling experiments from base path: $BASE_PATH"
echo "Target config: $TARGET_CONFIG"
echo

python scripts/experiments/sample_replay.py ${BASE_PATH}/accuracy_lightgbm/ $TARGET_CONFIG
python scripts/experiments/sample_replay.py ${BASE_PATH}/random/ $TARGET_CONFIG
python scripts/experiments/sample_replay.py ${BASE_PATH}/isolation/ $TARGET_CONFIG
python scripts/experiments/sample_replay.py ${BASE_PATH}/minority/ $TARGET_CONFIG
python scripts/experiments/sample_replay.py ${BASE_PATH}/info_gain_lightgbm/ $TARGET_CONFIG
