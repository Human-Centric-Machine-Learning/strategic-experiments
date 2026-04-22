#!/bin/bash

#SBATCH -J ss_greedy
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH -o /dev/null
#SBATCH -e /dev/null

if [ -d "${SLURM_SUBMIT_DIR}/venv" ]; then
    WORK_DIR="${SLURM_SUBMIT_DIR}"
elif [ -d "${SLURM_SUBMIT_DIR}/../venv" ]; then
    WORK_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
else
    echo "ERROR: Cannot find project root from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
    exit 1
fi

mkdir -p "${WORK_DIR}/outputs/slurm_logs"
exec > "${WORK_DIR}/outputs/slurm_logs/ss_greedy_${SLURM_JOB_ID}.out" 2> "${WORK_DIR}/outputs/slurm_logs/ss_greedy_${SLURM_JOB_ID}.err"
source "${WORK_DIR}/venv/bin/activate"

echo "Active Python: $(which python)"
echo "Python version: $(python --version)"

cd "${WORK_DIR}/src"

python -u single_shot_baseline.py --config ../config/single_shot_greedy.yaml

deactivate
