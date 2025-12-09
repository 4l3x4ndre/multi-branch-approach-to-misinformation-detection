#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-02:00:00     # days, hours, minute limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --cpus-per-task=1     # CPU cores
#SBATCH --mem=15G              # gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="TEST_ABLAT"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder
#SBATCH --array=0-14

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/test_ablation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID, task ID: $SLURM_ARRAY_TASK_ID"
echo "Logging to: $LOGFILE"

# --- Ablation Settings ---
declare -a ABLATION_NAMES
ABLATION_NAMES[0]="deepfake_only"
ABLATION_NAMES[1]="clip_image_only"
ABLATION_NAMES[2]="clip_text_only"
ABLATION_NAMES[3]="encyclo_only"
ABLATION_NAMES[4]="deepfake_clip_image"
ABLATION_NAMES[5]="deepfake_clip_text"
ABLATION_NAMES[6]="deepfake_encyclo"
ABLATION_NAMES[7]="clip_image_clip_text"
ABLATION_NAMES[8]="clip_image_encyclo"
ABLATION_NAMES[9]="clip_text_encyclo"
ABLATION_NAMES[10]="deepfake_clip_image_clip_text"
ABLATION_NAMES[11]="deepfake_clip_image_encyclo"
ABLATION_NAMES[12]="deepfake_clip_text_encyclo"
ABLATION_NAMES[13]="clip_image_clip_text_encyclo"
ABLATION_NAMES[14]="all_features"

STUDY_NAME="ablation-fixed-predres"
TRAIN_STUDY_NAME_PREFIX="ablation-fixed"
ABLATION_NAME=${ABLATION_NAMES[$SLURM_ARRAY_TASK_ID]}
CHECKPOINT_PATH="${TRAIN_STUDY_NAME_PREFIX}_${SLURM_ARRAY_TASK_ID}"

module load Python/3.12.3-GCCcore-13.3.0

echo "Testing with checkpoint: $CHECKPOINT_PATH"

poetry run python -m src.test_mmmd \
    --checkpoint "$CHECKPOINT_PATH" \
    --studyname "$STUDY_NAME" \
    --ablationname "$ABLATION_NAME" \
    --jobid "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" \
    --device gpu \
    "$@"

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
