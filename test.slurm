#!/bin/bash
#SBATCH --job-name=python-test
#SBATCH --partition=short
#SBATCH --account=<account>
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm-%j.out

# Go to submission folder
# cd 

# Load Python
module load Python/3.11.5-GCCcore-13.2.0

# Run your script
python run.py