#!/bin/bash
#SBATCH --job-name=train_k2d_models
#SBATCH --output=logs/train.out
#SBATCH --error=logs/train.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# --- Email notifications ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il

# Activate environment
source /home/eyal.amdur/K2D/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
python src/models/train_models.py --config src/config/ppo_config.json

