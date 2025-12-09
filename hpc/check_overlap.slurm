#!/bin/sh
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00     # 0 days, 1 hours 00 minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=10G             # 10 gigabytes system memory
#SBATCH --job-name="CHECK_OVERLAP"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGFILE="logs/check_overlap_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

module load Python/3.12.3-GCCcore-13.3.0

# Run the script
poetry run python check_dataset_overlap.py

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
