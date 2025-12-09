#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-30:00:00     # days, hours, minute limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --cpus-per-task=1     # CPU cores
#SBATCH --mem=10G              # gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="TEST_GLOB"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/test_global_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.test_mmmd "$@"

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
