# $1 target model

if [[ "$1" != *"zero_shot"* ]]; then
    echo "Submitting resampling jobs for configs/training/zero_shot.yaml"
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/05/ configs/training/zero_shot.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/01/ configs/training/zero_shot.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/001/ configs/training/zero_shot.yaml
fi

if [[ "$1" != *"distilbert"* ]]; then
    echo "Submitting resampling jobs for configs/training/balanced_train_distilbert.yaml"
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/05/ configs/training/balanced_train_distilbert.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/01/ configs/training/balanced_train_distilbert.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/001/ configs/training/balanced_train_distilbert.yaml
fi

if [[ "$1" != *"ModernBERT"* ]]; then
    echo "Submitting resampling jobs for configs/training/balanced_train_ModernBERT.yaml"
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/05/ configs/training/balanced_train_ModernBERT.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/01/ configs/training/balanced_train_ModernBERT.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/001/ configs/training/balanced_train_ModernBERT.yaml
fi

if [[ "$1" != *"gpt2"* ]]; then
    echo "Submitting resampling jobs for configs/training/balanced_train_gpt2.yaml"
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/05/ configs/training/balanced_train_gpt2.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/01/ configs/training/balanced_train_gpt2.yaml
    sbatch slurm_scripts/evaluation_scripts/run_resampling.sh $1/001/ configs/training/balanced_train_gpt2.yaml
fi
