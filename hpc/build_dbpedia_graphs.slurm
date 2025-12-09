#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-06:00:00     # 0 days, 6 hours 00 minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=5G              # 5 gigabytes system memory (not gpu mem)
#SBATCH --job-name="BUILD_DB"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M") # Create a unique timestamp
LOGFILE="logs/build_dbpedia_graphs_${TIMESTAMP}.log"
exec > "$LOGFILE" 2>&1 # Redirect all future output to the unique log file

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.data.build_dbpedia_database "$@"

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"
