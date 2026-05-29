#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=YOUR JOB NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=spascoe@mit.edu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

BASE_PATH="../orcd/scratch/imageXpress/05142026-TRIDONOR-TRICULTURE/coculture-284"

module load miniforge
mamba activate cellpose2
python3 -u process_3dimages.py "$BASE_PATH"