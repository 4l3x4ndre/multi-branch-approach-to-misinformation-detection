#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-60:00:00     # days, hours, minute limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --cpus-per-task=1     # CPU cores
#SBATCH --mem=15G              # gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="TRAIN_GLOB"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/train_global_optuna_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

# --- Optional study name argument ---
STUDY_NAME=$1   # first argument to sbatch
if [ -n "$STUDY_NAME" ]; then
	STUDY_ARGS=(--force-study-name --study-name "$STUDY_NAME")
	echo "Using custom study name: $STUDY_NAME"
else
	STUDY_ARGS=()
	echo "No study name provided — running without --study-name"
fi


module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.train_mmmd --pretrained-encyclopedia models/encyclopedia_tuning_2025-10-14-10-17_trial_23_best.pth --device gpu --split train --wandblog "${STUDY_ARGS[@]}"

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
