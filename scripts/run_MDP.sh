#!/bin/bash
# =====================================================
# run_MDP.sh - submit test.py to SLURM
# =====================================================

# Exit on errors
set -euo pipefail

echo "Starting the submission of test.py to SLURM..."

# ----------------------------
# Determine directories
# ----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # scripts/
BASE_DIR="$(dirname "$SCRIPT_DIR")"                          # project root
SRC_DIR="${BASE_DIR}/src"
VENV_PATH="${BASE_DIR}/env"
LOG_DIR="${BASE_DIR}/outputs/slurm_logs"

echo "Base directory: $BASE_DIR"
echo "Source directory: $SRC_DIR"
echo "Virtual env: $VENV_PATH"
echo "Log directory: $LOG_DIR"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

# ----------------------------
# SLURM job submission
# ----------------------------
# Customize SBATCH options here
SBATCH_SCRIPT=$(mktemp)

cat <<EOT > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH -J MDP
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -o ${LOG_DIR}/MDP_%j.out
#SBATCH -e ${LOG_DIR}/MDP_%j.err

# Load virtual environment
source "${VENV_PATH}/bin/activate"

# Go to source directory
cd "${SRC_DIR}"

# Run Python script
python test.py

# Deactivate virtualenv
deactivate
EOT

# Submit the job
JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
echo "Submitted SLURM job with ID: $JOB_ID"

# Remove temporary script
rm "$SBATCH_SCRIPT"