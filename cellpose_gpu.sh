#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=Cellpose
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdweiss1@mit.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#BASE_PATH="/home/jdweiss1/orcd/scratch/15/test"
BASE_PATH=$1

module load miniforge
mamba activate cellpose2

python3 -u process_3dimages.py "$BASE_PATH" --use_gpu
