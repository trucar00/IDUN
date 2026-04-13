#!/bin/bash
#SBATCH --account=share-ie-itk
#SBATCH --partition=short
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=2     # 2 CPU cores
#SBATCH --mem=5G              # 5 GB memory
#SBATCH --output=hello.txt    # Log file

# Go to submission folder
# cd 

# Load Python
module purge
module load Python/3.11.5-GCCcore-13.2.0

# Activate venv
source ~/venv/myenv/bin/activate

# Run your script
python run.py