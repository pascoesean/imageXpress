#!/bin/bash
#SBATCH -p mit_normal
#SBATCH --job-name=Classification
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdweiss1@mit.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

EXP_ID=$1

module load miniforge
mamba activate cellpose2

python3 -u classification.py "$EXP_ID"
