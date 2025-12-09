#!/bin/sh
#SBATCH --partition=short
#SBATCH --time=0-00:20:00     # 0 days, hours minute limit
#SBATCH --nodes=1             # compute nodes
#SBATCH --cpus-per-task=1     # CPU cores
#SBATCH --mem=5G              # gigabytes system memory (not gpu mem)
#SBATCH --job-name="BUILD_SETS"
#SBATCH --output=logs/build_sets.log    # Log file

echo "Starting job with ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m src.data.create_stratified_split

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"

