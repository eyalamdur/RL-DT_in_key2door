#!/bin/bash
#SBATCH --job-name=context_exp
#SBATCH --partition=clair
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il
#SBATCH --output=logs/context/context.out
#SBATCH --error=logs/context/context.err

set -euo pipefail

# Determine project root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

cd "${PROJECT_ROOT}"

# Activate virtual environment
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    echo "Activating virtual environment: ${PROJECT_ROOT}/.venv"
    source "${PROJECT_ROOT}/.venv/bin/activate"
elif [[ -f "${PROJECT_ROOT}/venv/bin/activate" ]]; then
    echo "Activating virtual environment: ${PROJECT_ROOT}/venv"
    source "${PROJECT_ROOT}/venv/bin/activate"
else
    echo "ERROR: No virtual environment found at ${PROJECT_ROOT}/.venv or ${PROJECT_ROOT}/venv" >&2
    exit 1
fi

# Add project root and src to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "Starting Context Experiment Pipeline..."
echo "Job ID: ${SLURM_JOB_ID:-LOCAL}"
echo "Project Root: ${PROJECT_ROOT}"

# Run the orchestration script
python "${PROJECT_ROOT}/src/experiments/context/run_context.py"
