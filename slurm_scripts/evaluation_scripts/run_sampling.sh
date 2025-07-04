#!/bin/bash
#SBATCH --qos turing
#SBATCH --job-name run_sampling
#SBATCH --time 0-5:00:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /bask/projects/v/vjgo8416-tigers/ARC-TIGERS/slurm_logs/run_sampling-%j.out
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

# $1 = Data config
# $2 = Model config
# $3 = Save dir
# $4 = Acquisition strategy
python scripts/experiments/sample.py $1 $2 $3 distance --n_repeats 10 --max_labels 500 --seed 321 --class_balance 0.0001
