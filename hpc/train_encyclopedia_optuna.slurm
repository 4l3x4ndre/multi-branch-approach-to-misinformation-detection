#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-48:00:00     # 0 days, 20 hours 00 minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=6G              # 5 gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="TRAIN_ENCYC"
#SBATCH --output=logs/train_enc_optuna_adam.log    # Log file

echo "Starting job with ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python src/train_encyclopedia_optuna.py --config-filename config_encyclopedia --device gpu

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"

