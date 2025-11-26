#!/bin/bash
#SBATCH --job-name=eval_k2d
#SBATCH --output=logs/eval.out
#SBATCH --error=logs/eval.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# --- Email notifications ---
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il

# Activate environment
source /home/eyal.amdur/K2D/.venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run evaluation
python src/evaluate.py --config src/config/evaluate_config.json

