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

echo number of rows: $1

python scripts/data_processing/dataset_download.py --max_rows $1
