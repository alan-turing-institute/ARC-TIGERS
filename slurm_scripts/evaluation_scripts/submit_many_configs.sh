RETRAIN_FREQUENCY=50

# # ZERO-SHOT EXPERIMENTS
# DATA_CONFIG="configs/data/football_one_vs_all_balanced.yaml"
# TRAINING_CONFIG="configs/training/zero_shot.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_10.yaml"
# TRAINING_CONFIG="configs/training/zero_shot.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_100.yaml"
# TRAINING_CONFIG="configs/training/zero_shot.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# # BALANCED TRAINING EXPERIMENTS
# DATA_CONFIG="configs/data/football_one_vs_all_balanced.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_distilbert.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_10.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_distilbert.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_100.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_distilbert.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# # IMBALANCED TRAINING EXPERIMENTS
# DATA_CONFIG="configs/data/imb_train_1_in_10.yaml"
# TRAINING_CONFIG="configs/training/imb_train_1_in_10_distilbert.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/imb_train_1_in_100.yaml"
# TRAINING_CONFIG="configs/training/imb_train_1_in_100_distilbert.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# # GPT2
# DATA_CONFIG="configs/data/football_one_vs_all_balanced.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_gpt2.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_10.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_gpt2.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_100.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_gpt2.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# ModernBERT EXPERIMENTS - 2k train set
# DATA_CONFIG="configs/data/football_one_vs_all_balanced.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_ModernBERT.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_10.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_ModernBERT.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# DATA_CONFIG="configs/data/football_one_vs_all_1_in_100.yaml"
# TRAINING_CONFIG="configs/training/balanced_train_ModernBERT.yaml"
# bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

# ModernBERT EXPERIMENTS - 200k train set
DATA_CONFIG="configs/data/football_200k_balanced.yaml"
TRAINING_CONFIG="configs/training/balanced_200k_train_ModernBERT.yaml"
bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

DATA_CONFIG="configs/data/football_200k_1_in_10.yaml"
TRAINING_CONFIG="configs/training/balanced_200k_train_ModernBERT.yaml"
bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY

DATA_CONFIG="configs/data/football_200k_1_in_100.yaml"
TRAINING_CONFIG="configs/training/balanced_200k_train_ModernBERT.yaml"
bash slurm_scripts/evaluation_scripts/submit_all.sh $DATA_CONFIG $TRAINING_CONFIG $RETRAIN_FREQUENCY
