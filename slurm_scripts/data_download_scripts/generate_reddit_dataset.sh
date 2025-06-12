#!/bin/bash
#SBATCH --qos turing
#SBATCH --job-name generate_reddit_dataset
#SBATCH --time 0-10:00:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /bask/projects/v/vjgo8416-tigers/ARC-TIGERS/slurm_logs/generate_reddit_dataset-%j.out
#SBATCH --cpus-per-gpu 18

# Load required modules here
module purge
module load baskerville
module load bask-apps/live/live
module load Python/3.10.8-GCCcore-12.2.0


# change working directory
cd /bask/projects/v/vjgo8416-tigers/ARC-TIGERS

source /bask/projects/v/vjgo8416-tigers/ARC-TIGERS/.venv/bin/activate

# change huggingface cache to be in project dir rather than user home
export HF_HOME="/bask/projects/v/vjgo8416-tigers/hf_cache"

# script uses relative path to project home so must be run from home, fix
SHARD_SIZE_ARG=""
if [ "$1" -gt 10000000 ]; then
    SHARD_SIZE_ARG="--shard_size 5000000"
fi

echo number of rows: $1

DATA_DIR="data/reddit_dataset_12/${1}_rows"
SINGLE_FILE="${DATA_DIR}/filtered_rows.json"
SHARD_PATTERN="${DATA_DIR}/filtered_rows_shard_"*

# Check for single file or any shard files
if [ -f "$SINGLE_FILE" ] || ls $SHARD_PATTERN 1> /dev/null 2>&1; then
    echo "$DATA_DIR already contains filtered data. Skipping download."
else
    python scripts/data_processing/dataset_download.py --max_rows $1 $SHARD_SIZE_ARG
fi

if [ $# -eq 2 ]; then
    python scripts/data_processing/dataset_generation.py "data/reddit_dataset_12/"$1"_rows" $2
fi