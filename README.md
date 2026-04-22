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
- **SciPy** (via `scipy.special`) -- log-Beta, regularized incomplete Beta functions for E_mix computation
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
│   ├── fiducial_mix.yaml                #   Fiducial with E_mix rejection test
│   ├── costly.yaml                      #   High-cost scenario (c0=100, c1=0.1)
│   ├── greedy.yaml                      #   Greedy agent (rho_A=5000)
│   ├── optimist.yaml                    #   Optimistic prior (alpha_0=4, beta_0=1)
│   ├── optimist_concentrated.yaml       #   Concentrated optimist (alpha_0=130, beta_0=70)
│   ├── optimist_concentrated_false.yaml #   False concentrated optimist (alpha_0=130, beta_0=30)
│   ├── pessimist.yaml                   #   Pessimistic prior (alpha_0=1, beta_0=1.5)
│   ├── single_shot.yaml                 #   Single-shot baseline for fiducial (T=0, n_max=800)
│   ├── single_shot_mix.yaml             #   Single-shot baseline for fiducial_mix (T=0, n_max=800)
│   ├── single_shot_costly.yaml          #   Single-shot baseline for costly
│   ├── single_shot_greedy.yaml          #   Single-shot baseline for greedy
│   ├── single_shot_optimist.yaml        #   Single-shot baseline for optimist
│   ├── single_shot_optimist_concentrated.yaml        #   Single-shot baseline for optimist_concentrated
│   ├── single_shot_optimist_concentrated_false.yaml  #   Single-shot baseline for optimist_concentrated_false
│   ├── single_shot_pessimist.yaml       #   Single-shot baseline for pessimist
│   └── sensitivity_runtime/             #   Auto-generated configs for runtime scaling
├── figures/                             # All figures (PDF/PNG), organized by scenario
│   ├── fiducial/
│   ├── fiducial_mix/
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
│   ├── fiducial_mix/
│   │   ├── plot_mdp.ipynb               #   E_mix MDP value functions, policies, belief trajectories
│   │   ├── plot_optimal_subsidy.ipynb   #   E_mix optimal subsidy analysis
│   │   └── plot_sensitivity.ipynb       #   E_mix sensitivity over rho_S and theta*
│   ├── costly/
│   │   ├── plot_mdp.ipynb
│   │   ├── plot_optimal_subsidy.ipynb
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
│   ├── fiducial_mix/
│   ├── costly/
│   ├── greedy/
│   ├── optimist/
│   ├── optimist_concentrated/
│   ├── optimist_concentrated_false/
│   ├── pessimist/
│   ├── sensitivity_runtime/
│   ├── single_shot/
│   ├── single_shot_mix/
│   └── slurm_logs/
├── scripts/                             # SLURM submission scripts for cluster execution
│   ├── run_MDP.sbatch                   #   Single MDP solve (log-linear)
│   ├── run_MDP.sh                       #   Shell wrapper for MDP submission
│   ├── run_MDP_fiducial_mix.sbatch      #   Single MDP solve (E_mix)
│   ├── run_optimal_subsidy_*.sbatch     #   Algorithm 1 per scenario
│   ├── run_sensitivity_*.sbatch         #   Sensitivity analysis per scenario
│   ├── run_sensitivity_runtime.sh       #   Runtime scaling grid (T x n_max)
│   ├── run_single_shot_baseline.sbatch  #   Single-shot baseline (fiducial)
│   ├── run_single_shot_baseline_mix.sbatch  # Single-shot baseline (fiducial_mix)
│   └── run_single_shot_*.sbatch         #   Single-shot baseline per scenario
├── src/                                 # Source code
│   ├── MDP_solver.py                    #   Core MDP solver (backward induction, Algorithm 2)
│   ├── MDP_solver_mix.py                #   MDP solver using E_mix rejection test
│   ├── rejection_mix.py                 #   Shared helpers for E_mix rejection table
│   ├── optimal_subsidy.py               #   Optimal subsidy search (Algorithm 1)
│   ├── optimal_subsidy_mix.py           #   Optimal subsidy search (Algorithm 1, E_mix variant)
│   ├── sensitivity_analysis.py          #   Sensitivity sweep over rho_S and theta*
│   ├── sensitivity_analysis_mix.py      #   Sensitivity sweep (E_mix variant)
│   ├── single_shot_baseline.py          #   Single-shot (T=0) baseline comparison
│   ├── single_shot_baseline_mix.py      #   Single-shot (T=0) baseline (E_mix variant)
│   ├── deploy_policy.py                 #   Monte Carlo policy evaluation under true theta*
│   ├── deploy_policy_mix.py             #   Monte Carlo policy evaluation (E_mix variant)
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
- **Approval condition**: the null hypothesis is rejected (treatment approved) when an e-value threshold is crossed. Two rejection tests are implemented:
  - **Log-linear** (default): `f(alpha, beta) >= 1/kappa` (Eq. 30 in the paper).
  - **E_mix** (mixture e-value): `E_mix(N, X) >= 1/kappa`, where `E_mix` integrates the likelihood ratio uniformly over `[theta_b, 1]`. This provides uniform power -- for any `theta* > theta_b`, `E_mix(N, X) -> infinity` as `N` grows.
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
| `action_stride` | -- | Action grid stride (E_mix solver only): `{stride, 2*stride, ..., n_max}` | 1 |
| `chunk_size` | -- | Backward-induction batch size over `(N,X)` states (E_mix solver only) | 1024 |

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

### `MDP_solver_mix.py` -- MDP solver using E_mix rejection test

Identical backward induction to `MDP_solver.py`, but uses the method-of-mixtures e-value `E_mix` for the rejection test instead of the log-linear e-value. The mixture e-value integrates the likelihood ratio uniformly over `[theta_b, 1]`:

```
E_mix(N, X) = [int_{theta_b}^1 theta^X (1-theta)^(N-X) d theta]
              / [(1-theta_b) * theta_b^X * (1-theta_b)^(N-X)]
```

Unlike the log-linear e-value of Eq. 5, `E_mix` is uniformly powerful: for any `theta* > theta_b`, `E_mix(N, X) -> infinity` as `N` grows. The MDP state `(N, X)` and backward-induction structure of Algorithm 2 are preserved because `E_mix` depends only on cumulative `(N, X)`.

Additional parameters: `action_stride` (controls the action grid granularity) and `chunk_size` (controls GPU memory usage during backward induction).

**Usage:**
```bash
python src/MDP_solver_mix.py --config config/fiducial_mix.yaml
```

### `rejection_mix.py` -- E_mix rejection table helpers

Provides shared functions for the mixture e-value rejection test:
- `build_rejection_table_np(max_N, kappa, theta_b)`: precomputes a boolean `(max_N+1) x (max_N+1)` lookup table where `reject[N, X] = (E_mix(N, X) >= 1/kappa)`. Computation is performed in log-space throughout for numerical stability using `scipy.special.betaln` and `betaincc`.
- `log_E_mix(N, X, theta_b)`: elementwise log of the mixture e-value over numpy arrays.

Used by `MDP_solver_mix.py`, `deploy_policy_mix.py`, `sensitivity_analysis_mix.py`, and the E_mix notebooks.

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

### `optimal_subsidy_mix.py` -- Optimal subsidy search (Algorithm 1, E_mix variant)

Mirrors `optimal_subsidy.py` but uses `SubsidizedMDPSolverMix` instead of the standard solver. The divide-and-conquer logic is identical: Proposition 7, Proposition 8, and the social utility formula all hold for any valid sequential e-value test, as they depend only on the MDP reward structure, not on the specific rejection rule.

**Usage:**
```bash
python src/optimal_subsidy_mix.py --config config/fiducial_mix.yaml
```

### `sensitivity_analysis.py` -- Sensitivity sweep

For each `rho_S` in `rho_S_range`:
1. Runs Algorithm 1 to find `epsilon*(rho_S)` and the corresponding MDP-optimal policy.
2. For each `theta*` in `theta_star_range`, evaluates the policy via Monte Carlo rollouts under true Binomial(`n`, `theta*`) dynamics to obtain the true approval probability, expected cost, opt-out probability, and social utility.

**Usage:**
```bash
python src/sensitivity_analysis.py --config config/fiducial.yaml [--n_episodes 200000] [--seed 42]
```

### `sensitivity_analysis_mix.py` -- Sensitivity sweep (E_mix variant)

Mirrors `sensitivity_analysis.py`; calls `find_optimal_subsidy_mix` and uses the precomputed E_mix rejection table for the true-dynamics Monte Carlo evaluation. Results are saved to `<save_dir>/sensitivity_results_mix.pt`.

**Usage:**
```bash
python src/sensitivity_analysis_mix.py --config config/fiducial_mix.yaml [--n_episodes 200000] [--seed 42]
```

### `single_shot_baseline.py` -- Single-shot (T=0) baseline

Computes the single-shot (non-sequential) baseline for decomposing the value of sequentiality and subsidies. For each `rho_S`, it evaluates three settings:

- **(a)** `epsilon = 0`: no subsidy, single trial stage.
- **(b)** `epsilon = epsilon*_sequential(rho_S)`: the sequential MDP-optimal subsidy (recovered from the corresponding scenario's `sensitivity_results.pt`) applied to a single-shot trial.
- **(c)** `epsilon = epsilon*_single_shot(rho_S)`: the single-shot's own optimal subsidy (recomputed via Algorithm 1 on the T=0 MDP).

Comparing with the scenario's sequential sensitivity results decomposes the total social utility gain into contributions from sequentiality and from subsidies. A single-shot config is provided for every sequential scenario (`single_shot.yaml`, `single_shot_costly.yaml`, ..., `single_shot_pessimist.yaml`); each points its `fiducial_sensitivity_path` at the matching scenario's sensitivity results.

**Usage:**
```bash
# Fiducial scenario
python src/single_shot_baseline.py --config config/single_shot.yaml [--n_episodes 200000] [--seed 42]
# Any other scenario, e.g. costly
python src/single_shot_baseline.py --config config/single_shot_costly.yaml
```

### `single_shot_baseline_mix.py` -- Single-shot (T=0) baseline (E_mix variant)

Mirrors `single_shot_baseline.py` for the E_mix rejection test. Uses `SubsidizedMDPSolverMix` and the E_mix rejection table. Results are saved to `<save_dir>/single_shot_baseline_mix.pt`.

**Usage:**
```bash
python src/single_shot_baseline_mix.py --config config/single_shot_mix.yaml [--n_episodes 200000] [--seed 42]
```

### `deploy_policy.py` -- Monte Carlo policy evaluation

Evaluates a pre-computed MDP policy under the real efficacy `theta*` via Monte Carlo simulation. Reports approval probability, expected cost conditional on approval, and agent utility. Optionally tracks Bayesian belief trajectories `(alpha_t, beta_t)` across episodes for visualization.

**Usage:**
```bash
python src/deploy_policy.py --config config/fiducial.yaml --policy_path outputs/fiducial/mdp_output/mdp_results_eps_0.300_T_3_nmax_200_alpha_1.0_beta_1.0_thetab_0.5.pt [--n_episodes 200000] [--seed 42]
```

### `deploy_policy_mix.py` -- Monte Carlo policy evaluation (E_mix variant)

Mirrors `deploy_policy.py` but uses a precomputed E_mix rejection table (via `rejection_mix.build_rejection_table_np`) instead of the log-linear formula for the approval test.

**Usage:**
```bash
python src/deploy_policy_mix.py --config config/fiducial_mix.yaml --policy_path outputs/fiducial_mix/mdp_output/mdp_results_mix_eps_*.pt [--n_episodes 200000] [--seed 42]
```

### `utils.py` -- Plotting utilities

Provides `latexify()` for setting LaTeX-compatible matplotlib RC parameters and `get_fig_dim()` for computing figure dimensions that avoid scaling artifacts in LaTeX documents.


## Configuration

All experiment parameters are specified in YAML files under `config/`. Each file defines:

- **Model parameters**: `rho_A`, `rho_S`, `c0`, `c1`, `T`, `n_max`, `epsilon`, `kappa`, `theta_b`, `alpha_0`, `beta_0`
- **Sensitivity ranges**: `rho_S_range`, `theta_star_range` (used by `sensitivity_analysis.py`)
- **E_mix-specific**: `action_stride` (action grid granularity), `chunk_size` (backward-induction batch size)
- **Runtime settings**: `device` (`cuda`/`cpu`), `tol` (numerical tolerance), `save_dir` (output path)

The provided scenarios differ from the fiducial parameterization as follows:

| Scenario | Key difference |
|----------|---------------|
| `fiducial` | Baseline: `alpha_0=1, beta_0=1, rho_A=240, c0=48.9, c1=0.066` |
| `fiducial_mix` | Fiducial with E_mix rejection test (+ `action_stride=1`, `chunk_size=1024`) |
| `costly` | Higher costs: `c0=100, c1=0.1` |
| `greedy` | Higher agent reward: `rho_A=5000` |
| `optimist` | Optimistic prior: `alpha_0=4, beta_0=1` |
| `optimist_concentrated` | Concentrated optimist: `alpha_0=130, beta_0=70` |
| `optimist_concentrated_false` | Falsely optimistic concentrated: `alpha_0=130, beta_0=30` |
| `pessimist` | Pessimistic prior: `alpha_0=1, beta_0=1.5` |
| `single_shot` | Non-sequential baseline for fiducial: `T=0, n_max=800` |
| `single_shot_mix` | Non-sequential baseline for fiducial_mix: `T=0, n_max=800` |
| `single_shot_<scenario>` | Non-sequential (`T=0, n_max=800`) counterpart of each scenario above, used for the sequential-vs.-non-sequential comparison |


## Instructions

### Running a single MDP solve

To solve the agent's MDP for a specific configuration:

```bash
# Log-linear e-value
python src/MDP_solver.py --config config/fiducial.yaml

# E_mix (mixture e-value)
python src/MDP_solver_mix.py --config config/fiducial_mix.yaml
```

Results are saved as a `.pt` file in the directory specified by `save_dir` in the config. The output contains dictionaries keyed by time step `l`, with 2D NumPy arrays indexed by `[N, X]` for each quantity (`V_eps`, `V_0`, `A`, `P_approval`, `Policy`).

### Running Algorithm 1 (optimal subsidy)

```bash
# Log-linear
python src/optimal_subsidy.py --config config/fiducial.yaml

# E_mix
python src/optimal_subsidy_mix.py --config config/fiducial_mix.yaml
```

This runs the divide-and-conquer search over `[0, epsilon_max]` and saves the partition breakpoints, social utilities, and per-breakpoint policies to `optimal_subsidy_results.pt`.

### Running the full sensitivity analysis

```bash
# Log-linear
python src/sensitivity_analysis.py --config config/fiducial.yaml --n_episodes 200000

# E_mix
python src/sensitivity_analysis_mix.py --config config/fiducial_mix.yaml --n_episodes 200000
```

This first runs Algorithm 1 for each `rho_S` in `rho_S_range`, then evaluates each resulting policy under every `theta*` in `theta_star_range` via Monte Carlo. Results are saved to `sensitivity_results.pt` (or `sensitivity_results_mix.pt`). Per-episode Monte Carlo samples at `theta_fid` (default `0.65`) are also stored (`samples_fid_approved`, `samples_fid_cost`) whenever `theta_fid` is in `theta_star_range`; these enable bootstrap confidence intervals on the sequential-vs.-non-sequential comparison in the plotting notebooks.

### Reproducing the sequential-vs.-non-sequential comparison

For every scenario, the plotting notebook's final cell compares the sequential protocol to its non-sequential counterpart. The comparison requires two ingredients:

1. Sequential sensitivity results with per-episode samples (produced by `sensitivity_analysis.py` or `sensitivity_analysis_mix.py` -- run or re-run the corresponding `run_sensitivity_*.sbatch`).
2. A single-shot baseline (produced by `single_shot_baseline.py` or `single_shot_baseline_mix.py` -- run the corresponding `run_single_shot_*.sbatch`).

Once both artefacts exist, re-executing the scenario's `plot_sensitivity.ipynb` produces the two-panel figure with 95% bootstrap confidence intervals.

### Running on a SLURM cluster

Each script in `scripts/` is a ready-to-use SLURM batch file. They auto-detect the project root, so they can be submitted from any directory:

```bash
# Single optimal subsidy run
sbatch scripts/run_optimal_subsidy_fiducial.sbatch

# Full sensitivity analysis (log-linear)
sbatch scripts/run_sensitivity_fiducial.sbatch

# Full sensitivity analysis (E_mix)
sbatch scripts/run_sensitivity_fiducial_mix.sbatch

# Single-shot baseline (fiducial scenario)
sbatch scripts/run_single_shot_baseline.sbatch
# Single-shot baseline (E_mix)
sbatch scripts/run_single_shot_baseline_mix.sbatch
# Single-shot baseline for any other scenario
sbatch scripts/run_single_shot_costly.sbatch
# (similarly for greedy / pessimist / optimist / optimist_concentrated / optimist_concentrated_false)

# Runtime scaling grid (submits one job per (T, n_max) pair)
bash scripts/run_sensitivity_runtime.sh
```

The SLURM scripts request a single GPU, 70 GB RAM, and up to 48 hours of wall time for sensitivity analyses. Adjust the `#SBATCH` directives to match your cluster's configuration.

### Generating figures

All figures in the paper are generated from the Jupyter notebooks in `notebooks/`. Each scenario has its own subdirectory with up to three notebooks:

1. **`plot_mdp.ipynb`** -- Visualizes MDP value functions, optimal policies, and belief trajectories for a fixed epsilon.
2. **`plot_optimal_subsidy.ipynb`** -- Plots the agent value function `V^epsilon` and social utility `U^S` as functions of epsilon, showing the piecewise-linear structure and the optimal subsidy `epsilon*`.
3. **`plot_sensitivity.ipynb`** -- Plots sensitivity results: `epsilon*`, `P(approval)`, `P(opt-out)`, and `U^S` as functions of `rho_S`, with curves for different `theta*` values. The final cell additionally produces a two-panel figure with the optimal subsidy `epsilon*` on top and the percentage social-utility gain of the sequential protocol over the single-shot baseline (both without subsidy and at the single-shot's own optimal subsidy) on the bottom, plotted as a function of the social-to-agent approval utility ratio `rho_S / rho_A`. 95% bootstrap confidence intervals are rendered automatically when per-episode Monte Carlo samples are available in both `sensitivity_results.pt` and `single_shot_baseline.pt` (controlled by `theta_fid` being in `theta_star_range`, which is the default). The figure is saved to `figures/<scenario>/sensitivity/eps_star_vs_rhoS_vs_ss.pdf` (fiducial) or `figures/<scenario>/mdp_output/sensitivity/eps_star_vs_rhoS_vs_ss.pdf` (variants).

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
    'theta_fid':        float,                        # (optional) theta* at which samples are stored
    'samples_fid_approved': [[int8]],                 # (optional, n_rho, n_ep) per-episode approval indicator
    'samples_fid_cost':     [[float]],                # (optional, n_rho, n_ep) per-episode cost
    'params':           dict
}
```

### Single-shot baseline output (`single_shot_baseline.pt`)

```python
{
    'rho_S_range':       [float],                     # (n_rho,)
    'theta_star_range':  [float],                     # (n_theta,)
    'eps_fid_used':      [float],                     # (n_rho,) sequential MDP optimal subsidy used in case (b)
    'eps_ss_opt':        [float],                     # (n_rho,) single-shot's own optimal subsidy (case c)
    'P_true_ss0':        [[float]],                   # (n_rho, n_theta) case (a): epsilon=0
    'A_true_ss0':        [[float]],
    'us_true_ss0':       [[float]],
    'P_true_ss_epsfid':  [[float]],                   # (n_rho, n_theta) case (b): eps*_sequential
    'A_true_ss_epsfid':  [[float]],
    'us_true_ss_epsfid': [[float]],
    'P_true_ss_epsopt':  [[float]],                   # (n_rho, n_theta) case (c): eps*_single_shot
    'A_true_ss_epsopt':  [[float]],
    'us_true_ss_epsopt': [[float]],
    'theta_fid':         float,                       # (optional) theta* at which samples are stored
    'samples_ss0_approved':   [int8],                 # (optional, n_ep,) case (a) per-episode approval
    'samples_ss0_cost':       [float],                # (optional, n_ep,)
    'samples_ssopt_approved': [[int8]],               # (optional, n_rho, n_ep) case (c) per-episode approval
    'samples_ssopt_cost':     [[float]],              # (optional, n_rho, n_ep)
    'params':            dict
}
```


## Contact & attribution

*Author and citation information have been omitted to preserve double-blind review.
They will be added to the camera-ready version.*
