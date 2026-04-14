"""
Sensitivity analysis over rho_S_range and theta_star_range.

For each rho_S in rho_S_range (from the YAML config):
  - Runs find_optimal_subsidy (Algorithm 1) to obtain:
      eps_star   : MDP-optimal subsidy epsilon*
      us_mdp     : MDP social utility U^S at eps*  (= rho_S * P_mdp - eps* * A_mdp)
      P_mdp      : MDP approval probability at eps* under the Beta-Binomial dynamics
      A_mdp      : MDP expected cost | approval at eps*

For each (rho_S, theta_star) pair:
  - Monte Carlo rollout under true Binomial(n, theta_star) dynamics with the
    policy pi*(eps_star(rho_S)) to obtain:
      P_true     : true approval probability
      A_true     : true E[cost | approved]
      p_optout   : true P(opt-out before approval)
      us_true    : rho_S * P_true - eps* * A_true  (true social utility, Eq. 18)

Saves all results to <save_dir>/sensitivity_results.pt.

Usage:
    python sensitivity_analysis.py --config ../config/yourfavoriteconfig.yaml
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from optimal_subsidy import find_optimal_subsidy


# ---------------------------------------------------------------------------
# Monte Carlo true-dynamics evaluation
# ---------------------------------------------------------------------------

def _mc_true(policy, theta_star, params, n_episodes=200_000, seed=42,
             return_samples=False):
    """
    Evaluate policy under true Binomial(n, theta_star) dynamics.

    Returns:
      p_approval : P(approved | theta_star, policy)
      a_true     : E[cost | approved, theta_star, policy]  -- NaN if never approved
      p_optout   : P(agent chooses n=0 at some stage before approval | theta_star, policy)
      samples    : (only if return_samples) dict with per-episode arrays
                   'approved' (int8, n_episodes): 1 if episode ended in approval else 0
                   'cost'     (float32, n_episodes): episode cost if approved else 0
                   Satisfies mean(approved)=p_approval and mean(cost)=a_true.
    """
    rng           = np.random.default_rng(seed)
    T             = int(params['T'])
    alpha_0       = float(params['alpha_0'])
    beta_0        = float(params['beta_0'])
    c0            = float(params['c0'])
    c1            = float(params['c1'])
    log_kappa_inv = np.log(1.0 / float(params['kappa']))
    log_term      = np.log(1.0 + float(params['theta_b']) * (np.e - 1.0))

    def _approved(N, X):
        alpha = alpha_0 + X
        beta  = beta_0  + N - X
        return ((alpha - alpha_0)
                - (alpha + beta - alpha_0 - beta_0) * log_term) >= log_kappa_inv

    n_approved        = 0
    n_optout          = 0
    sum_cost_approved = 0.0

    if return_samples:
        approved_arr = np.zeros(n_episodes, dtype=np.int8)
        cost_arr     = np.zeros(n_episodes, dtype=np.float32)

    for ep in range(n_episodes):
        N, X    = 0, 0
        C       = 0.0
        outcome = 'horizon'  # 'approved' | 'optout' | 'horizon' (exhausted T)

        for l in range(T + 1):
            n_star = int(policy[l][N, X])
            if n_star == 0:
                outcome = 'optout'
                break
            x  = int(rng.binomial(n_star, theta_star))
            C += c0 + c1 * n_star
            N += n_star
            X += x
            if _approved(N, X):
                outcome = 'approved'
                break

        if outcome == 'approved':
            n_approved        += 1
            sum_cost_approved += C
            if return_samples:
                approved_arr[ep] = 1
                cost_arr[ep]     = C
        elif outcome == 'optout':
            n_optout += 1

    out = {
        'p_approval': n_approved / n_episodes,
        # E[C_τ · 1{approved}] — unconditional, consistent with MDP solver's A definition.
        # (MDP sets A=cost at approval, A=0 at opt-out, so A[0,0,0] = P * E[C|approved].)
        'a_true':     sum_cost_approved / n_episodes,
        # Conditional version, stored separately for reporting.
        'e_cost_cond_approval': sum_cost_approved / n_approved if n_approved > 0 else float('nan'),
        'p_optout':   n_optout / n_episodes,
    }
    if return_samples:
        out['samples'] = {'approved': approved_arr, 'cost': cost_arr}
    return out


# ---------------------------------------------------------------------------
# Main sensitivity sweep
# ---------------------------------------------------------------------------

def run_sensitivity(config_path, n_episodes=200_000, seed=42, verbose=True,
                    theta_fid=0.65):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    solver_keys = ['rho_A', 'T', 'n_max', 'c0', 'c1', 'kappa', 'theta_b',
                   'alpha_0', 'beta_0', 'device']
    base_config = {k: config[k] for k in solver_keys}

    rho_S_range      = [float(v) for v in config['rho_S_range']]
    theta_star_range = [float(v) for v in config['theta_star_range']]
    epsilon_max      = float(config.get('epsilon_max', 1.0))
    tol              = float(config.get('tol', 1e-6))
    save_dir         = config.get('save_dir', './mdp_output')
    theta_fid        = float(config.get('theta_fid', theta_fid))

    # Subdirectory for per-epsilon MDP solve files written by find_optimal_subsidy
    mdp_scratch_dir = os.path.join(save_dir, 'sensitivity_mdp_solves')
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Sweep over rho_S: run Algorithm 1 for each value
    # ------------------------------------------------------------------
    eps_star_list    = []   # (n_rho_S,)
    us_mdp_list      = []
    P_mdp_list       = []
    A_mdp_list       = []
    policy_star_list = []   # optimal policy dict {l: 2-D numpy array}

    total_t0 = time.time()

    for rho_S in rho_S_range:
        if verbose:
            print(f"\n{'='*60}")
            print(f"rho_S = {rho_S}")
            print('='*60)

        t0 = time.time()
        epsilons, social_utilities, eps_star, us_star, breakpoint_stats, policy_per_bp = \
            find_optimal_subsidy(
                base_config,
                rho_S=rho_S,
                epsilon_max=epsilon_max,
                tol=tol,
                save_dir=mdp_scratch_dir,
                verbose=verbose,
            )

        best_idx = int(np.argmax(social_utilities))
        _V0_star, A_star, P_star = breakpoint_stats[best_idx]

        eps_star_list.append(eps_star)
        us_mdp_list.append(us_star)
        P_mdp_list.append(P_star)
        A_mdp_list.append(A_star)
        policy_star_list.append(policy_per_bp[best_idx])

        if verbose:
            print(f"  -> eps* = {eps_star:.6f}  U^S_mdp = {us_star:.4f}  "
                  f"P_mdp = {P_star:.4f}  A_mdp = {A_star:.4f}  "
                  f"(elapsed {time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------
    # True MC evaluation for each (rho_S, theta_star) pair
    # ------------------------------------------------------------------
    n_rho = len(rho_S_range)
    n_th  = len(theta_star_range)

    P_true_mat    = np.full((n_rho, n_th), np.nan)
    A_true_mat    = np.full((n_rho, n_th), np.nan)
    p_optout_mat  = np.full((n_rho, n_th), np.nan)
    us_true_mat   = np.full((n_rho, n_th), np.nan)

    # Per-episode samples at theta_fid for downstream bootstrap CIs.
    j_fid = next((jj for jj, th in enumerate(theta_star_range)
                  if abs(th - theta_fid) < 1e-9), None)
    if j_fid is None and verbose:
        print(f"  [info] theta_fid={theta_fid} not in theta_star_range; "
              f"skipping per-episode sample collection.")
    samples_fid_approved = (np.zeros((n_rho, n_episodes), dtype=np.int8)
                            if j_fid is not None else None)
    samples_fid_cost     = (np.zeros((n_rho, n_episodes), dtype=np.float32)
                            if j_fid is not None else None)

    mc_params = {k: config[k]
                 for k in ['T', 'alpha_0', 'beta_0', 'kappa', 'theta_b', 'c0', 'c1']}

    if verbose:
        print(f"\n{'='*60}")
        print("Monte Carlo true-dynamics evaluation")
        print(f"  n_episodes = {n_episodes:,}  |  "
              f"{n_rho} rho_S values x {n_th} theta_star values = {n_rho*n_th} runs")
        print('='*60)

    for i, (rho_S, eps_star, policy) in enumerate(
            zip(rho_S_range, eps_star_list, policy_star_list)):
        for j, theta_star in enumerate(theta_star_range):
            if verbose:
                print(f"  rho_S={rho_S:8.1f}  theta_star={theta_star:.2f} ...",
                      end='  ', flush=True)

            mc = _mc_true(
                policy, theta_star, mc_params,
                n_episodes=n_episodes,
                seed=seed + i * n_th + j,
                return_samples=(j == j_fid),
            )

            P_true_mat[i, j]   = mc['p_approval']
            A_true_mat[i, j]   = mc['a_true']   # E[C · 1{approved}]
            p_optout_mat[i, j] = mc['p_optout']
            # a_true = E[C · 1{approved}] is already 0 when P_true=0, so no nan risk.
            us_true_mat[i, j]  = rho_S * mc['p_approval'] - eps_star * mc['a_true']

            if j == j_fid:
                samples_fid_approved[i] = mc['samples']['approved']
                samples_fid_cost[i]     = mc['samples']['cost']

            if verbose:
                print(f"P_true={mc['p_approval']:.4f}  "
                      f"p_optout={mc['p_optout']:.4f}  "
                      f"U^S_true={us_true_mat[i,j]:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out = {
        'rho_S_range':      rho_S_range,
        'theta_star_range': theta_star_range,
        # Per-rho_S MDP quantities — shape (n_rho_S,)
        'eps_star':         eps_star_list,
        'us_mdp':           us_mdp_list,
        'P_mdp':            P_mdp_list,
        'A_mdp':            A_mdp_list,
        # True MC quantities — shape (n_rho_S, n_theta_star) stored as nested lists
        'P_true':           P_true_mat.tolist(),
        'A_true':           A_true_mat.tolist(),
        'p_optout_true':    p_optout_mat.tolist(),
        'us_true':          us_true_mat.tolist(),
        'params': {
            **base_config,
            'epsilon_max': epsilon_max,
            'tol':         tol,
            'n_episodes':  n_episodes,
            'theta_fid':   theta_fid,
        },
    }
    if j_fid is not None:
        out['theta_fid']             = theta_fid
        out['samples_fid_approved']  = samples_fid_approved
        out['samples_fid_cost']      = samples_fid_cost

    out_path = os.path.join(save_dir, 'sensitivity_results.pt')
    torch.save(out, out_path)

    elapsed_total = time.time() - total_t0
    print(f"\nTotal elapsed: {elapsed_total:.1f}s")
    print(f"Sensitivity results saved to: {out_path}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sensitivity analysis over rho_S_range and theta_star_range.')
    parser.add_argument('--config',     type=str, required=True,
                        help='Path to YAML config file.')
    parser.add_argument('--n_episodes', type=int, default=200_000,
                        help='MC episodes per (rho_S, theta_star) pair (default: 200000).')
    parser.add_argument('--seed',       type=int, default=42,
                        help='Base random seed (default: 42).')
    parser.add_argument('--no_verbose', dest='verbose', action='store_false',
                        help='Suppress per-solve output.')
    parser.add_argument('--theta_fid',  type=float, default=0.65,
                        help='theta* at which per-episode samples are stored (default: 0.65).')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    run_sensitivity(
        args.config,
        n_episodes=args.n_episodes,
        seed=args.seed,
        verbose=args.verbose,
        theta_fid=args.theta_fid,
    )
