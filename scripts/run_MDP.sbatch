#!/bin/bash

#SBATCH -J MDP
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH -o /dev/null
#SBATCH -e /dev/null



# Activate virtual environment
# Resolve project root (works whether submitted from project root or scripts/)
if [ -d "${SLURM_SUBMIT_DIR}/venv" ]; then
    WORK_DIR="${SLURM_SUBMIT_DIR}"
elif [ -d "${SLURM_SUBMIT_DIR}/../venv" ]; then
    WORK_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: Cannot find project root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi

# Redirect logs
mkdir -p "${WORK_DIR}/outputs/slurm_logs"
exec > "${WORK_DIR}/outputs/slurm_logs/MDP_${SLURM_JOB_ID}.out" 2> "${WORK_DIR}/outputs/slurm_logs/MDP_${SLURM_JOB_ID}.err"
source "${WORK_DIR}/venv/bin/activate"

echo "Active Python: $(which python)"
echo "Python version: $(python --version)"




# Go to source directory
cd "${WORK_DIR}/src"

# Run Python script
python -u MDP_solver.py --config ../config/orphan.yaml

#python test.py 
# Deactivate virtual environment
deactivate