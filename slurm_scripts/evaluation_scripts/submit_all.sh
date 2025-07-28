# $1 = Data config
# $2 = Training config
# $3 = Retrain frequency

sbatch slurm_scripts/evaluation_scripts/run_sampling.sh $1 $2 random $3
# sbatch slurm_scripts/evaluation_scripts/run_sampling.sh $1 $2 distance $3
sbatch slurm_scripts/evaluation_scripts/run_sampling.sh $1 $2 isolation $3
sbatch slurm_scripts/evaluation_scripts/run_sampling.sh $1 $2 minority $3

sbatch slurm_scripts/evaluation_scripts/run_sampling.sh $1 $2 accuracy $3
sbatch slurm_scripts/evaluation_scripts/run_sampling.sh $1 $2 info_gain $3

sbatch slurm_scripts/evaluation_scripts/run_sampling_nopt.sh $1 $2 accuracy $3
sbatch slurm_scripts/evaluation_scripts/run_sampling_nopt.sh $1 $2 info_gain $3
