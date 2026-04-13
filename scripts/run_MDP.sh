#!/bin/bash
# =====================================================
# run_MDP.sh - submit test.py to SLURM
# =====================================================

# Exit on errors
set -euo pipefail

# Resolve project root relative to this script's location
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"





# Create log directory if it doesn't exist

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
#SBATCH -o ${WORK_DIR}/outputs/slurm_logs/MDP_%j.out
#SBATCH -e ${WORK_DIR}/outputs/slurm_logs/MDP_%j.err



# Load virtual environment
source ${WORK_DIR}/venv/bin/activate

# Go to source directory
cd ${WORK_DIR}/src

# Run Python script
python MDP_solver.py




# Deactivate virtualenv
deactivate
EOT

# Submit the job
JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
echo "Submitted SLURM job with ID: $JOB_ID"

# Remove temporary script
rm "$SBATCH_SCRIPT"