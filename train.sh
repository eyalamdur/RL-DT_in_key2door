#!/bin/bash
#SBATCH --job-name=train_k2d
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=eyal.amdur@campus.technion.ac.il

# Define default log path just in case, though usually overridden
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Self-Submission Logic for Dynamic Logs
# -----------------------------------------------------------------------------
if [ -z "${SLURM_JOB_ID:-}" ]; then
    # We are NOT in a SLURM job. Calculate date/time and submit.
    
    # Get current date and time
    DATE=$(date +%d_%m_%Y)
    TIME=$(date +%H:%M:%S)
    
    # Create log directory
    LOG_DIR="logs/train/${DATE}/${TIME}"
    mkdir -p "${LOG_DIR}"
    
    echo "Submitting job with logs in ${LOG_DIR}..."
    
    # Submit this script itself to sbatch, overriding the output/error paths
    sbatch --output="${LOG_DIR}/train.out" --error="${LOG_DIR}/train.err" "$0" "$@"
    
    exit 0
fi

# -----------------------------------------------------------------------------
# Main Job Logic (runs on compute node)
# -----------------------------------------------------------------------------

# Determine project root
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

CONFIG_PATH="${PROJECT_ROOT}/src/config/train_config.json"
EXTRA_ARGS=""

print_help() {
    cat <<'EOF'
Usage: ./train.sh [options]

This script self-submits to SLURM with timestamped logs in logs/DD_MM_YYYY/HH_MM_SS/.

Options:
--config PATH       Path to config file (default: src/config/train_config.json)
--extra "ARGS"      Extra arguments to pass to the python script
--help, -h          Show this help message and exit
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                CONFIG_PATH="$2"
                shift 2
                ;;
            --extra)
                EXTRA_ARGS="$2"
                shift 2
                ;;
            --help|-h)
                print_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1" >&2
                exit 1
                ;;
        esac
    done
}

run_training() {
    # Activate virtual environment
    if [[ -d "${PROJECT_ROOT}/.venv" ]]; then
        source "${PROJECT_ROOT}/.venv/bin/activate"
    elif [[ -d "${PROJECT_ROOT}/venv" ]]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
    fi
    
    # Add project root to PYTHONPATH
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

    echo "Starting training pipeline..."
    echo "Job ID: ${SLURM_JOB_ID}"
    echo "Project Root: ${PROJECT_ROOT}"
    echo "Config: ${CONFIG_PATH}"
    
    python "${PROJECT_ROOT}/src/models/train_models.py" \
        --config "${CONFIG_PATH}" \
        ${EXTRA_ARGS}
}

main() {
    parse_args "$@"
    run_training
}

main "$@"
