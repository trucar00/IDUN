#!/bin/bash
#SBATCH --account=trulsc
#SBATCH --partition=short
#SBATCH --job-name=python-test
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%j.out

# Go to submission folder
# cd 

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Run your script
python run.py