#!/bin/bash
#SBATCH --job-name=eval_k2d
#SBATCH --partition=clair
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Check if running under SLURM
if [ -z "${SLURM_JOB_ID:-}" ]; then
    # Not running under SLURM, submit self
    DATE=$(date +%d_%m_%Y)
    TIME=$(date +%H_%M_%S)
    LOG_DIR="logs/eval/${DATE}/${TIME}"
    mkdir -p "${LOG_DIR}"
    echo "Submitting evaluation job to SLURM. Logs will be in ${LOG_DIR}"
    sbatch -p clair --output="${LOG_DIR}/eval.out" --error="${LOG_DIR}/eval.err" "$0" "$@"
    exit 0
fi

# Use the directory where the script is located as the project root
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

# Verify Python is from the venv
echo "Using Python: $(which python)"

# Set PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# Default config path
CONFIG_PATH="${PROJECT_ROOT}/src/config/evaluate_config.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="${EXTRA_ARGS} $1"
            shift
            ;;
    esac
done

echo "Project root: ${PROJECT_ROOT}"
echo "Config: ${CONFIG_PATH}"

# Run evaluation
"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/src/evaluate.py" --config "${CONFIG_PATH}" ${EXTRA_ARGS}
