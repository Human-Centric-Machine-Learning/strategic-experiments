#!/bin/bash
# =====================================================================
# run_sensitivity_runtime.sh
#
# Submits one SLURM job per (T, n_max) combination to measure how
# runtime scales with these parameters.
#
# All jobs use the fiducial rho_S (2000) and theta_star (0.65) only,
# NOT the full sensitivity ranges.
# =====================================================================

set -euo pipefail

# Resolve project root relative to this script's location
WORK_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Parameter grid (mirrors range_T / range_n_max in config/fiducial.yaml)
# ---------------------------------------------------------------------------
T_VALUES=(0 1 2 3 4)
NMAX_VALUES=(50 100 150 200 250 300)

# Fiducial single values (not ranges)
RHO_S_FIDUCIAL=2000
THETA_STAR_FIDUCIAL=0.65

# Directories
CONFIG_DIR="${WORK_DIR}/config/sensitivity_runtime"
OUTPUT_BASE="${WORK_DIR}/outputs/sensitivity_runtime"
LOG_DIR="${WORK_DIR}/outputs/slurm_logs"

mkdir -p "$CONFIG_DIR" "$LOG_DIR"

# ---------------------------------------------------------------------------
# Submit one job per (T, n_max) combination
# ---------------------------------------------------------------------------
TOTAL=0

for T in "${T_VALUES[@]}"; do
  for NMAX in "${NMAX_VALUES[@]}"; do

    JOB_TAG="sens_T${T}_nmax${NMAX}"
    JOB_CONFIG="${CONFIG_DIR}/${JOB_TAG}.yaml"
    SAVE_DIR="${OUTPUT_BASE}/T${T}_nmax${NMAX}/mdp_output"

    # ------------------------------------------------------------------
    # Write per-job config: fiducial values, single-element rho_S_range
    # and theta_star_range so sensitivity_analysis.py only runs one pair
    # ------------------------------------------------------------------
    cat > "$JOB_CONFIG" <<YAML
# Auto-generated config for runtime sensitivity: T=${T}, n_max=${NMAX}
# Uses fiducial rho_S and theta_star only.

rho_S: ${RHO_S_FIDUCIAL}
rho_S_range: [${RHO_S_FIDUCIAL}]
rho_A: 240

c0: 48.9
c1: 0.066

T: ${T}
n_max: ${NMAX}

epsilon: 0.3
theta_star: ${THETA_STAR_FIDUCIAL}
theta_star_range: [${THETA_STAR_FIDUCIAL}]

theta_b: 0.5
kappa: 0.05
alpha_0: 1.0
beta_0: 1.0
epsilon_max: 0.9

device: cuda
tol: 1e-6
save_dir: ${SAVE_DIR}
YAML

    # ------------------------------------------------------------------
    # Build and submit SLURM script
    # ------------------------------------------------------------------
    SBATCH_SCRIPT=$(mktemp)

    cat > "$SBATCH_SCRIPT" <<EOT
#!/bin/bash
#SBATCH -J ${JOB_TAG}
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --partition=h200,h100,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH -o ${LOG_DIR}/${JOB_TAG}_%j.out
#SBATCH -e ${LOG_DIR}/${JOB_TAG}_%j.err

source ${WORK_DIR}/venv/bin/activate

echo "Active Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "Job: T=${T}, n_max=${NMAX}"

cd ${WORK_DIR}/src

python -u sensitivity_analysis.py --config ${JOB_CONFIG}

deactivate
EOT

    JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
    echo "Submitted T=${T}, n_max=${NMAX} -> job ${JOB_ID}  (config: ${JOB_CONFIG})"
    rm "$SBATCH_SCRIPT"

    TOTAL=$((TOTAL + 1))
  done
done

echo ""
echo "Done. Submitted ${TOTAL} jobs (${#T_VALUES[@]} T values x ${#NMAX_VALUES[@]} n_max values)."
echo "Configs saved under: ${CONFIG_DIR}"
echo "Outputs will be written under: ${OUTPUT_BASE}"
