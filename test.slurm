#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=short
#SBATCH --job-name=python-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3

# Go to submission folder
# cd 

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Run your script
python run.py