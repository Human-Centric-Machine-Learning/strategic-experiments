# Optimizing Social Utility in Sequential Experiments


## Paper abstract

Regulatory approval of products in high-stakes domains such as drug development requires statistical evidence of safety and efficacy through large-scale randomized controlled trials.
However, the high financial cost of these trials may deter developers who lack absolute certainty in their product's efficacy, ultimately stifling the development of "moonshot" products that could offer high social utility.
To address this inefficiency, in this paper, we introduce a statistical protocol for experimentation where the product developer (the agent) conducts a randomized controlled trial sequentially and the regulator (the principal) partially subsidizes its cost.
By modeling the protocol using a belief Markov decision process, we show that the agent's optimal strategy can be found efficiently using dynamic programming.
Further, we show that the social utility is a piecewise linear and convex function over the subsidy level the principal selects, and thus the socially optimal subsidy can also be found efficiently using divide-and-conquer.
Simulation experiments using publicly available data on drug approvals demonstrate that our statistical protocol can be used to increase the social utility compared to non-sequential alternatives. 


## Dependencies

All experiments were performed using Python 3.11.2. The main computational dependencies are:

- **PyTorch** (2.6.0) -- GPU-accelerated backward induction and Beta-Binomial PMF computation
- **NumPy** (2.2.4) -- array operations and Monte Carlo simulations
- **PyYAML** (6.0.2) -- configuration file parsing
- **Matplotlib** (3.10.1) / **Seaborn** (0.13.2) -- figure generation
- **Jupyter** (1.1.1) -- interactive analysis notebooks

To create a virtual environment and install all dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended. The MDP solver automatically falls back to CPU if no GPU is available, but large configurations (e.g., `T=4, n_max=300`) may require significant memory and time on CPU.


## Repository structure

```
├── config/                              # YAML configuration files for all experiments
│   ├── fiducial.yaml                    #   Fiducial (baseline) parameterization
│   ├── costly.yaml                      #   High-cost scenario (c0=100, c1=0.1)
│   ├── greedy.yaml                      #   Greedy agent (rho_A=5000)
│   ├── optimist.yaml                    #   Optimistic prior (alpha_0=4, beta_0=1)
│   ├── optimist_concentrated.yaml       #   Concentrated optimist (alpha_0=130, beta_0=70)
│   ├── optimist_concentrated_false.yaml #   False concentrated optimist (alpha_0=130, beta_0=30)
│   ├── pessimist.yaml                   #   Pessimistic prior (alpha_0=1, beta_0=1.5)
│   ├── single_shot.yaml                 #   Single-shot baseline (T=0, n_max=500)
│   └── sensitivity_runtime/             #   Auto-generated configs for runtime scaling
├── figures/                             # All figures (PDF/PNG), organized by scenario
│   ├── fiducial/
│   ├── costly/
│   ├── greedy/
│   ├── optimist/
│   ├── optimist_concentrated/
│   ├── optimist_concentrated_false/
│   ├── pessimist/
│   └── sensitivity_runtime/
├── notebooks/                           # Jupyter notebooks for analysis and figure generation
│   ├── fiducial/
│   │   ├── plot_mdp.ipynb               #   MDP value functions, policies, belief trajectories
│   │   ├── plot_optimal_subsidy.ipynb   #   Optimal subsidy analysis (Algorithm 1 output)
│   │   └── plot_sensitivity.ipynb       #   Sensitivity over rho_S and theta*
│   ├── costly/
│   │   ├── plot_mdp.ipynb
│   │   ├── plot_optimal_subsidy.ipynb
│   │   ├── plot_optimal_subsidy_new.ipynb
│   │   └── plot_sensitivity.ipynb
│   ├── greedy/
│   ├── optimist/
│   ├── optimist_concentrated/
│   ├── optimist_concentrated_false/
│   ├── pessimist/
│   └── sensitivity_runtime/
│       └── plot_runtime_sensitivity.ipynb  # Runtime scaling analysis (T x n_max grid)
├── outputs/                             # Intermediate results (.pt files), organized by scenario
│   ├── fiducial/
│   ├── costly/
│   ├── greedy/
│   ├── optimist/
│   ├── optimist_concentrated/
│   ├── optimist_concentrated_false/
│   ├── pessimist/
│   ├── sensitivity_runtime/
│   ├── single_shot/
│   └── slurm_logs/
├── scripts/                             # SLURM submission scripts for cluster execution
│   ├── run_MDP.sbatch                   #   Single MDP solve
│   ├── run_MDP.sh                       #   Shell wrapper for MDP submission
│   ├── run_MDP_cum.sbatch               #   MDP solve with C-indexed output
│   ├── run_optimal_subsidy_*.sbatch     #   Algorithm 1 per scenario
│   ├── run_sensitivity_*.sbatch         #   Sensitivity analysis per scenario
│   ├── run_sensitivity_runtime.sh       #   Runtime scaling grid (T x n_max)
│   └── run_single_shot_baseline.sbatch  #   Single-shot baseline
├── src/                                 # Source code
│   ├── MDP_solver.py                    #   Core MDP solver (backward induction, Algorithm 2)
│   ├── MDP_solver_cum.py                #   MDP solver variant with cost-indexed output
│   ├── optimal_subsidy.py               #   Optimal subsidy search (Algorithm 1)
│   ├── sensitivity_analysis.py          #   Sensitivity sweep over rho_S and theta*
│   ├── single_shot_baseline.py          #   Single-shot (T=0) baseline comparison
│   ├── deploy_policy.py                 #   Monte Carlo policy evaluation under true theta*
│   └── utils.py                         #   LaTeX-compatible plotting utilities
├── requirements.txt
├── LICENSE                              # MIT License
└── README.md
```


## Model overview

The code implements a Stackelberg game between a **principal** (regulator) and an **agent** (firm) in a sequential RCT setting:

- **State**: `(alpha, beta, C)` -- Beta-distribution belief parameters and accumulated cost. In the code, states are encoded as `(N, X)` at time step `l`, where `alpha = alpha_0 + X`, `beta = beta_0 + N - X`, and `C = l * c0 + N * c1`.
- **Actions**: at each stage `l = 0, 1, ..., T`, the agent chooses to enroll `n` subjects (`n in {1, ..., n_max}`) or opt out (`n = 0`).
- **Transitions**: outcomes follow a Beta-Binomial distribution (Bayesian updates under Beta prior).
- **Approval condition**: the null hypothesis is rejected (treatment approved) when an e-value threshold is crossed: `f(alpha, beta) >= 1/kappa` (Eq. 30 in the paper).
- **Agent payoff**: `rho_A + epsilon * C` upon approval minus accumulated trial costs `C`, or `-C` if the agent opts out.
- **Social utility**: `U^S(epsilon; pi) = rho_S * P(approval) - epsilon * E[cost | approval]` (Eq. 18).


## Key parameters

| Parameter | Symbol | Description | Fiducial value |
|-----------|--------|-------------|----------------|
| `rho_A` | rho_A | Agent's private reward from approval ($M) | 240 |
| `rho_S` | rho_S | Social value of approval ($M) | 2000 |
| `c0` | c_0 | Fixed cost per trial stage ($M) | 48.9 |
| `c1` | c_1 | Per-patient cost ($M) | 0.066 |
| `T` | T | Maximum number of trial stages (T+1 actions total) | 3 |
| `n_max` | n_max | Maximum patients per stage | 200 |
| `epsilon` | epsilon | Subsidy rate (fraction of costs reimbursed upon approval) | varies |
| `kappa` | kappa | Significance level for approval (e-value threshold) | 0.05 |
| `theta_b` | theta_b | Null hypothesis treatment effect | 0.5 |
| `alpha_0, beta_0` | alpha_0, beta_0 | Prior Beta distribution parameters | 1.0, 1.0 |
| `theta_star` | theta* | True treatment efficacy (for Monte Carlo evaluation) | 0.65 |

Cost estimates are derived from Moore et al. (2018) for `c0` and Stergiopoulos et al. (2017) for `c1`. The agent reward `rho_A` is based on Rahman et al. (2020).


## Source code description

### `MDP_solver.py` -- Core MDP solver (Algorithm 2)

Solves the agent's subsidized MDP via backward induction over the time horizon `l = T, ..., 0`. For each step, it computes:

- **V^epsilon**: the agent's optimal value function under subsidy rate epsilon.
- **V^0**: the agent's value with no subsidy (epsilon = 0), under the *same* policy.
- **A**: the expected cost conditional on approval (used by the linear decomposition, Proposition 7: `V^epsilon = V^0 + epsilon * A`).
- **P_approval**: the probability of approval under the optimal policy.
- **Policy**: the optimal action `n*(N, X, l)` at each state.

The computation is GPU-accelerated and vectorized over the `(N, X)` state grid. Large state spaces are processed in configurable chunks (`chunk_size` parameter) to manage GPU memory.

**Usage:**
```bash
python src/MDP_solver.py --config config/fiducial.yaml
```

### `MDP_solver_cum.py` -- Cost-indexed MDP solver

Identical backward induction to `MDP_solver.py`, but additionally builds 2D `(alpha, beta)` grids indexed by accumulated cost `C`. For each achievable cost level, the value function is stored on the anti-diagonal `alpha + beta = alpha_0 + beta_0 + N`. This format is used for verifying the monotonicity properties in Proposition 5.

**Usage:**
```bash
python src/MDP_solver_cum.py --config config/fiducial.yaml
```

### `optimal_subsidy.py` -- Optimal subsidy search (Algorithm 1)

Implements the divide-and-conquer algorithm to find the principal's Stackelberg-optimal subsidy `epsilon*`. Exploits the piecewise-linear structure of the agent's value function (Proposition 7):

1. Solves the MDP at `epsilon = 0` and `epsilon = epsilon_max`.
2. Finds the intersection of the two linear value functions.
3. Solves the MDP at the intersection point. If the optimal value matches the left-endpoint extrapolation (TRUE branch), the intersection is a genuine policy transition breakpoint. Otherwise (ELSE branch), a new policy has been discovered and the algorithm recurses into both sub-intervals.
4. Evaluates social utility `U^S = rho_S * P - epsilon * A` at each breakpoint and returns the optimum.

**Usage:**
```bash
python src/optimal_subsidy.py --config config/fiducial.yaml
```

### `sensitivity_analysis.py` -- Sensitivity sweep

For each `rho_S` in `rho_S_range`:
1. Runs Algorithm 1 to find `epsilon*(rho_S)` and the corresponding MDP-optimal policy.
2. For each `theta*` in `theta_star_range`, evaluates the policy via Monte Carlo rollouts under true Binomial(`n`, `theta*`) dynamics to obtain the true approval probability, expected cost, opt-out probability, and social utility.

**Usage:**
```bash
python src/sensitivity_analysis.py --config config/fiducial.yaml [--n_episodes 200000] [--seed 42]
```

### `single_shot_baseline.py` -- Single-shot (T=0) baseline

Computes the single-shot (non-sequential) baseline for decomposing the value of sequentiality and subsidies. For each `rho_S`, it evaluates three settings:

- **(a)** `epsilon = 0`: no subsidy, single trial stage.
- **(b)** `epsilon = epsilon*_fiducial(rho_S)`: the fiducial MDP-optimal subsidy applied to a single-shot trial.
- **(c)** `epsilon = epsilon*_single_shot(rho_S)`: the single-shot's own optimal subsidy (recomputed via Algorithm 1 on the T=0 MDP).

Comparing with the fiducial sensitivity results decomposes the total social utility gain into contributions from sequentiality and from subsidies.

**Usage:**
```bash
python src/single_shot_baseline.py --config config/single_shot.yaml [--n_episodes 200000] [--seed 42]
```

### `deploy_policy.py` -- Monte Carlo policy evaluation

Evaluates a pre-computed MDP policy under the real efficacy `theta*` via Monte Carlo simulation. Reports approval probability, expected cost conditional on approval, and agent utility. Optionally tracks Bayesian belief trajectories `(alpha_t, beta_t)` across episodes for visualization.

**Usage:**
```bash
python src/deploy_policy.py --config config/fiducial.yaml --policy_path outputs/fiducial/mdp_output/mdp_results_eps_0.300_T_3_nmax_200_alpha_1.0_beta_1.0_thetab_0.5.pt [--n_episodes 200000] [--seed 42]
```

### `utils.py` -- Plotting utilities

Provides `latexify()` for setting LaTeX-compatible matplotlib RC parameters and `get_fig_dim()` for computing figure dimensions that avoid scaling artifacts in LaTeX documents.


## Configuration

All experiment parameters are specified in YAML files under `config/`. Each file defines:

- **Model parameters**: `rho_A`, `rho_S`, `c0`, `c1`, `T`, `n_max`, `epsilon`, `kappa`, `theta_b`, `alpha_0`, `beta_0`
- **Sensitivity ranges**: `rho_S_range`, `theta_star_range` (used by `sensitivity_analysis.py`)
- **Runtime settings**: `device` (`cuda`/`cpu`), `tol` (numerical tolerance), `save_dir` (output path)

The provided scenarios differ from the fiducial parameterization as follows:

| Scenario | Key difference |
|----------|---------------|
| `fiducial` | Baseline: `alpha_0=1, beta_0=1, rho_A=240, c0=48.9, c1=0.066` |
| `costly` | Higher costs: `c0=100, c1=0.1` |
| `greedy` | Higher agent reward: `rho_A=5000` |
| `optimist` | Optimistic prior: `alpha_0=4, beta_0=1` |
| `optimist_concentrated` | Concentrated optimist: `alpha_0=130, beta_0=70` |
| `optimist_concentrated_false` | Falsely optimistic concentrated: `alpha_0=130, beta_0=30` |
| `pessimist` | Pessimistic prior: `alpha_0=1, beta_0=1.5` |
| `single_shot` | Non-sequential baseline: `T=0, n_max=500` |


## Instructions

### Running a single MDP solve

To solve the agent's MDP for a specific configuration:

```bash
cd src
python MDP_solver.py --config ../config/fiducial.yaml
```

Results are saved as a `.pt` file in the directory specified by `save_dir` in the config. The output contains dictionaries keyed by time step `l`, with 2D NumPy arrays indexed by `[N, X]` for each quantity (`V_eps`, `V_0`, `A`, `P_approval`, `Policy`).

### Running Algorithm 1 (optimal subsidy)

```bash
cd src
python optimal_subsidy.py --config ../config/fiducial.yaml
```

This runs the divide-and-conquer search over `[0, epsilon_max]` and saves the partition breakpoints, social utilities, and per-breakpoint policies to `optimal_subsidy_results.pt`.

### Running the full sensitivity analysis

```bash
cd src
python sensitivity_analysis.py --config ../config/fiducial.yaml --n_episodes 200000
```

This first runs Algorithm 1 for each `rho_S` in `rho_S_range`, then evaluates each resulting policy under every `theta*` in `theta_star_range` via Monte Carlo. Results are saved to `sensitivity_results.pt`.

### Running on a SLURM cluster

Each script in `scripts/` is a ready-to-use SLURM batch file. They auto-detect the project root, so they can be submitted from any directory:

```bash
# Single optimal subsidy run
sbatch scripts/run_optimal_subsidy_fiducial.sbatch

# Full sensitivity analysis
sbatch scripts/run_sensitivity_fiducial.sbatch

# Single-shot baseline
sbatch scripts/run_single_shot_baseline.sbatch

# Runtime scaling grid (submits one job per (T, n_max) pair)
bash scripts/run_sensitivity_runtime.sh
```

The SLURM scripts request a single GPU (H200/H100/A100 partition), 70 GB RAM, and up to 48 hours of wall time for sensitivity analyses. Adjust the `#SBATCH` directives to match your cluster's configuration.

### Generating figures

All figures in the paper are generated from the Jupyter notebooks in `notebooks/`. Each scenario has its own subdirectory with up to three notebooks:

1. **`plot_mdp.ipynb`** -- Visualizes MDP value functions, optimal policies, and belief trajectories for a fixed epsilon.
2. **`plot_optimal_subsidy.ipynb`** -- Plots the agent value function `V^epsilon` and social utility `U^S` as functions of epsilon, showing the piecewise-linear structure and the optimal subsidy `epsilon*`.
3. **`plot_sensitivity.ipynb`** -- Plots sensitivity results: `epsilon*`, `P(approval)`, `P(opt-out)`, and `U^S` as functions of `rho_S`, with curves for different `theta*` values.

The runtime scaling analysis is in `notebooks/sensitivity_runtime/plot_runtime_sensitivity.ipynb`.

To run all notebooks:

```bash
source venv/bin/activate
jupyter notebook
```

Then navigate to the desired notebook in the browser.


## Output format

All results are saved as PyTorch `.pt` files (loaded via `torch.load(path, weights_only=False)`).

### MDP solver output (`mdp_results_eps_*.pt`)

```python
{
    'V_eps':      {l: np.ndarray (max_N_l+1, max_N_l+1)},   # Agent value under epsilon
    'V_0':        {l: np.ndarray (max_N_l+1, max_N_l+1)},   # Agent value with no subsidy
    'A':          {l: np.ndarray (max_N_l+1, max_N_l+1)},   # Expected cost | approval
    'P_approval': {l: np.ndarray (max_N_l+1, max_N_l+1)},   # Approval probability
    'Policy':     {l: np.ndarray (max_N_l+1, max_N_l+1)},   # Optimal action n*(N,X,l)
    'params':     dict                                        # All model parameters
}
```

### Optimal subsidy output (`optimal_subsidy_results.pt`)

```python
{
    'epsilons':              [float],           # Partition breakpoints
    'social_utilities':      [float],           # U^S at each breakpoint
    'eps_star':              float,             # Optimal subsidy
    'us_star':               float,             # Optimal social utility
    'V0_per_breakpoint':     [float],           # V^0 at each breakpoint
    'A_per_breakpoint':      [float],           # A at each breakpoint
    'P_per_breakpoint':      [float],           # P(approval) at each breakpoint
    'policy_per_breakpoint': [dict],            # Policy at each breakpoint
    'params':                dict
}
```

### Sensitivity output (`sensitivity_results.pt`)

```python
{
    'rho_S_range':      [float],                      # (n_rho,)
    'theta_star_range': [float],                      # (n_theta,)
    'eps_star':         [float],                      # (n_rho,) MDP-optimal subsidy per rho_S
    'us_mdp':           [float],                      # (n_rho,) MDP social utility
    'P_mdp':            [float],                      # (n_rho,) MDP approval probability
    'A_mdp':            [float],                      # (n_rho,) MDP E[cost | approval]
    'P_true':           [[float]],                    # (n_rho, n_theta) true P(approval)
    'A_true':           [[float]],                    # (n_rho, n_theta) true E[cost * 1{appr}]
    'p_optout_true':    [[float]],                    # (n_rho, n_theta) true P(opt-out)
    'us_true':          [[float]],                    # (n_rho, n_theta) true social utility
    'params':           dict
}
```


## Contact & attribution

In case you have questions about the code, you identify potential bugs, or you would like us to include additional functionalities, feel free to open an issue.

If you use parts of the code in this repository for your own research, please consider citing:
```

```
