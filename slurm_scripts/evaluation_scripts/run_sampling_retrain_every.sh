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
module load CUDA/12.6.0
module load Python/3.11.3-GCCcore-12.3.0

PROJECT_ROOT="/bask/projects/v/vjgo8416-tigers/ARC-TIGERS"

source "$PROJECT_ROOT/.venv/bin/activate"

# change working directory
cd $PROJECT_ROOT

# change huggingface cache to be in project dir rather than user home
export HF_HOME="/bask/projects/v/vjgo8416-tigers/hf_cache"

# $1 = Data config
# $2 = Training config
# $3 = Acquisition strategy
# $4 = Retrain frequency
# $5 = Test imbalance
python scripts/experiments/sample.py $1 $2 $3 --retrain_every $4 --n_repeats 10 --max_labels 1000 --seed 321 --output_dir /bask/projects/v/vjgo8416-tigers/ARC-TIGERS/outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/default/eval_outputs_retrain_every/$5/eval_outputs_retrain_$4
