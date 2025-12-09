#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-10:00:00     # days, hours, minute limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --cpus-per-task=1     # CPU cores
#SBATCH --mem=15G              # gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="ABLAT"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder
#SBATCH --array=0-14

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/train_ablation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID, task ID: $SLURM_ARRAY_TASK_ID"
echo "Logging to: $LOGFILE"

# --- Ablation Settings ---
declare -a ABLATION_ARGS
ABLATION_ARGS[0]="--no-clip-use-image-features --no-clip-use-text-features --no-use-encyclopedic-knowledge"
ABLATION_ARGS[1]="--no-use-deepfake-detector --no-clip-use-text-features --no-use-encyclopedic-knowledge"
ABLATION_ARGS[2]="--no-use-deepfake-detector --no-clip-use-image-features --no-use-encyclopedic-knowledge"
ABLATION_ARGS[3]="--no-use-deepfake-detector --no-clip-use-image-features --no-clip-use-text-features"
ABLATION_ARGS[4]="--no-clip-use-text-features --no-use-encyclopedic-knowledge"
ABLATION_ARGS[5]="--no-clip-use-image-features --no-use-encyclopedic-knowledge"
ABLATION_ARGS[6]="--no-clip-use-image-features --no-clip-use-text-features"
ABLATION_ARGS[7]="--no-use-deepfake-detector --no-use-encyclopedic-knowledge"
ABLATION_ARGS[8]="--no-use-deepfake-detector --no-clip-use-text-features"
ABLATION_ARGS[9]="--no-use-deepfake-detector --no-clip-use-image-features"
ABLATION_ARGS[10]="--no-use-encyclopedic-knowledge"
ABLATION_ARGS[11]="--no-clip-use-text-features"
ABLATION_ARGS[12]="--no-clip-use-image-features"
ABLATION_ARGS[13]="--no-use-deepfake-detector"
ABLATION_ARGS[14]=""

STUDY_NAME="ablation-fixed_${SLURM_ARRAY_TASK_ID}"
STUDY_ARGS=(--force-study-name --study-name "$STUDY_NAME")

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.train_mmmd --n-trials 1 --device gpu --split train --wandblog --no-use-optuna-params ${ABLATION_ARGS[$SLURM_ARRAY_TASK_ID]} "${STUDY_ARGS[@]}" --jobid "$SLURM_JOB_ID" "$@"

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
