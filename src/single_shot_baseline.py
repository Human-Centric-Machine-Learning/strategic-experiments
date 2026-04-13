"""
Single-shot baseline for comparison against the fiducial sensitivity sweep.

For each rho_S in the config's rho_S_range, we solve a single-action MDP
(T=0, n_max large) under three subsidy settings:
  (a) epsilon = 0                          -> policy pi_ss_0
  (b) epsilon = eps*_fid(rho_S)            -> policy pi_ss_eps
  (c) epsilon = eps*_ss(rho_S)             -> policy pi_ss_epsopt
where eps*_fid(rho_S) is the MDP-optimal subsidy under the fiducial setting
(loaded from a prior fiducial sensitivity_results.pt), and eps*_ss(rho_S) is
the single-shot setting's *own* MDP-optimal subsidy, recomputed via Algorithm 1
(divide-and-conquer) on the T=0 MDP.

Each resulting single-shot policy is evaluated via Monte Carlo rollouts under
the true Binomial(n, theta_star) dynamics for every theta_star in the config,
yielding the true social utility
    U^S_true(eps; pi_ss, theta*) = rho_S * P_true - eps * A_true    (Eq. 18).

Together with the fiducial us_true matrix, this lets us decompose:
  - (a) fiducial_us_true - ss0_us_true     : joint effect of sequentiality + subsidies.
  - (b) fiducial_us_true - sseps_us_true   : marginal effect of sequentiality alone,
                                             holding the subsidy fixed at eps*_fid.

Usage:
    python single_shot_baseline.py --config ../config/single_shot.yaml
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from MDP_solver import SubsidizedMDPSolver
from sensitivity_analysis import _mc_true
from optimal_subsidy import find_optimal_subsidy


SOLVER_KEYS = ["rho_A", "T", "n_max", "c0", "c1", "epsilon",
               "kappa", "theta_b", "alpha_0", "beta_0", "device"]


def _solve_single_shot(base_solver_cfg: dict, epsilon: float, save_dir: str):
    """Solve the single-shot MDP once and return (policy, V0, A, P) at (N=0, X=0)."""
    cfg = {k: base_solver_cfg[k] for k in SOLVER_KEYS if k in base_solver_cfg}
    cfg["epsilon"] = float(epsilon)
    solver = SubsidizedMDPSolver(**cfg)
    results = solver.solve(save_dir=save_dir)
    policy = {l: results["Policy"][l] for l in range(solver.T + 1)}
    V0 = float(results["V_0"][0][0, 0])
    A  = float(results["A"][0][0, 0])
    P  = float(results["P_approval"][0][0, 0])
    return policy, V0, A, P


def run(config_path: str, n_episodes: int = 200_000, seed: int = 42, verbose: bool = True):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_solver_cfg = {k: config[k] for k in SOLVER_KEYS}
    rho_S_range      = [float(v) for v in config["rho_S_range"]]
    theta_star_range = [float(v) for v in config["theta_star_range"]]
    save_dir         = config.get("save_dir", "./single_shot_output")
    fid_path         = config["fiducial_sensitivity_path"]

    os.makedirs(save_dir, exist_ok=True)
    mdp_scratch = os.path.join(save_dir, "mdp_solves")

    # ---- load fiducial eps*(rho_S) --------------------------------------
    fid = torch.load(fid_path, weights_only=False)
    fid_rho    = [float(v) for v in fid["rho_S_range"]]
    fid_eps    = [float(v) for v in fid["eps_star"]]
    rho_to_eps = dict(zip(fid_rho, fid_eps))
    missing = [r for r in rho_S_range if r not in rho_to_eps]
    if missing:
        raise ValueError(
            f"rho_S values {missing} not found in fiducial sensitivity results at {fid_path}"
        )

    # MC uses the single-shot parameters (T=0, etc.)
    mc_params = {k: config[k] for k in ["T", "alpha_0", "beta_0", "kappa",
                                        "theta_b", "c0", "c1"]}

    n_rho = len(rho_S_range)
    n_th  = len(theta_star_range)

    # Case (a): epsilon = 0  -> single policy, reused across all rho_S
    # Case (b): epsilon = eps*_fid(rho_S) -> one policy per rho_S
    P_ss0    = np.full((n_rho, n_th), np.nan)
    A_ss0    = np.full((n_rho, n_th), np.nan)
    us_ss0   = np.full((n_rho, n_th), np.nan)

    P_sseps  = np.full((n_rho, n_th), np.nan)
    A_sseps  = np.full((n_rho, n_th), np.nan)
    us_sseps = np.full((n_rho, n_th), np.nan)

    # Case (c): epsilon = eps*_ss(rho_S), the single-shot's OWN optimal subsidy
    P_ssopt    = np.full((n_rho, n_th), np.nan)
    A_ssopt    = np.full((n_rho, n_th), np.nan)
    us_ssopt   = np.full((n_rho, n_th), np.nan)
    eps_ss_opt = np.full(n_rho, np.nan)

    eps_fid_used = np.array([rho_to_eps[r] for r in rho_S_range])

    # Base config (without epsilon) used by Algorithm 1 for the single-shot setting
    ss_alg1_base = {k: base_solver_cfg[k] for k in SOLVER_KEYS if k != "epsilon"}
    epsilon_max  = float(config.get("epsilon_max", 1.0))
    tol          = float(config.get("tol", 1e-6))

    t0 = time.time()

    # ---- (a) epsilon = 0 solved once ------------------------------------
    if verbose:
        print("\n=== Single-shot baseline (a): epsilon = 0 ===")
    pol0, V0_0, A0_0, P0_0 = _solve_single_shot(base_solver_cfg, 0.0, mdp_scratch)
    n_star0 = int(pol0[0][0, 0])
    if verbose:
        print(f"  policy n*(0,0) = {n_star0}   V0={V0_0:.4f}  A={A0_0:.4f}  P={P0_0:.4f}")

    for j, theta_star in enumerate(theta_star_range):
        mc = _mc_true(pol0, theta_star, mc_params,
                      n_episodes=n_episodes, seed=seed + j)
        p, a = mc["p_approval"], mc["a_true"]
        for i, rho_S in enumerate(rho_S_range):
            P_ss0[i, j]  = p
            A_ss0[i, j]  = a
            us_ss0[i, j] = rho_S * p - 0.0 * a
        if verbose:
            print(f"  theta*={theta_star:.2f}  P={p:.4f}  A={a:.4f}")

    # ---- (b) epsilon = eps*_fid(rho_S) ----------------------------------
    if verbose:
        print("\n=== Single-shot baseline (b): epsilon = eps*_fid(rho_S) ===")

    # Cache MC by unique epsilon value (different rho_S often share eps*_fid).
    mc_cache: dict[tuple[float, float], dict] = {}

    for i, rho_S in enumerate(rho_S_range):
        eps_i = rho_to_eps[rho_S]
        if verbose:
            print(f"\n  rho_S={rho_S:8.1f}  eps*_fid={eps_i:.6f}")

        pol_i, V0_i, A_i, P_i = _solve_single_shot(base_solver_cfg, eps_i, mdp_scratch)
        n_star_i = int(pol_i[0][0, 0])
        if verbose:
            print(f"    policy n*(0,0) = {n_star_i}   V0={V0_i:.4f}  A={A_i:.4f}  P={P_i:.4f}")

        for j, theta_star in enumerate(theta_star_range):
            ckey = (round(eps_i, 12), theta_star)
            if ckey not in mc_cache:
                mc_cache[ckey] = _mc_true(
                    pol_i, theta_star, mc_params,
                    n_episodes=n_episodes,
                    seed=seed + 10_000 + i * n_th + j,
                )
            mc = mc_cache[ckey]
            p, a = mc["p_approval"], mc["a_true"]
            P_sseps[i, j]  = p
            A_sseps[i, j]  = a
            us_sseps[i, j] = rho_S * p - eps_i * a

    # ---- (c) epsilon = eps*_ss(rho_S): single-shot's own optimal subsidy --
    if verbose:
        print("\n=== Single-shot baseline (c): epsilon = eps*_ss(rho_S) (Algorithm 1 on T=0) ===")

    for i, rho_S in enumerate(rho_S_range):
        if verbose:
            print(f"\n  rho_S={rho_S:8.1f}  running Algorithm 1 (single-shot) ...")

        epsilons_ss, us_ss_list, eps_star_ss, _us_star_ss, bp_stats_ss, pol_per_bp_ss = \
            find_optimal_subsidy(
                ss_alg1_base,
                rho_S=rho_S,
                epsilon_max=epsilon_max,
                tol=tol,
                save_dir=mdp_scratch,
                verbose=False,
            )
        best_idx = int(np.argmax(us_ss_list))
        eps_ss_opt[i] = eps_star_ss
        pol_ss = pol_per_bp_ss[best_idx]
        n_star_c = int(pol_ss[0][0, 0])
        if verbose:
            print(f"    eps*_ss = {eps_star_ss:.6f}   n*(0,0) = {n_star_c}")

        for j, theta_star in enumerate(theta_star_range):
            ckey = (round(eps_star_ss, 12), theta_star)
            if ckey not in mc_cache:
                mc_cache[ckey] = _mc_true(
                    pol_ss, theta_star, mc_params,
                    n_episodes=n_episodes,
                    seed=seed + 20_000 + i * n_th + j,
                )
            mc = mc_cache[ckey]
            p, a = mc["p_approval"], mc["a_true"]
            P_ssopt[i, j]  = p
            A_ssopt[i, j]  = a
            us_ssopt[i, j] = rho_S * p - eps_star_ss * a

    out = {
        "rho_S_range":       rho_S_range,
        "theta_star_range":  theta_star_range,
        "eps_fid_used":      eps_fid_used.tolist(),
        "eps_ss_opt":        eps_ss_opt.tolist(),
        # Case (a): eps = 0
        "P_true_ss0":        P_ss0.tolist(),
        "A_true_ss0":        A_ss0.tolist(),
        "us_true_ss0":       us_ss0.tolist(),
        # Case (b): eps = eps*_fid(rho_S)
        "P_true_ss_epsfid":  P_sseps.tolist(),
        "A_true_ss_epsfid":  A_sseps.tolist(),
        "us_true_ss_epsfid": us_sseps.tolist(),
        # Case (c): eps = eps*_ss(rho_S)  (single-shot's own optimal subsidy)
        "P_true_ss_epsopt":  P_ssopt.tolist(),
        "A_true_ss_epsopt":  A_ssopt.tolist(),
        "us_true_ss_epsopt": us_ssopt.tolist(),
        "params": {
            **{k: config[k] for k in SOLVER_KEYS},
            "n_episodes":               n_episodes,
            "fiducial_sensitivity_path": fid_path,
        },
    }
    out_path = os.path.join(save_dir, "single_shot_baseline.pt")
    torch.save(out, out_path)
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print(f"Single-shot baseline saved to: {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-shot (T=0) baseline for sensitivity comparison.")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=200_000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    run(args.config, n_episodes=args.n_episodes, seed=args.seed, verbose=args.verbose)
