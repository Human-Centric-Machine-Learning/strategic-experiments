"""
Single-shot baseline (E_mix variant) for comparison against the fiducial_mix
sensitivity sweep.

Mirrors single_shot_baseline.py exactly; the only differences:
  - calls SubsidizedMDPSolverMix / find_optimal_subsidy_mix for the T=0 MDP
  - the true-dynamics approval test in MC uses a precomputed E_mix rejection
    table (rejection_mix.build_rejection_table_np) instead of the log-linear
    formula

Three subsidy settings per rho_S:
  (a) epsilon = 0
  (b) epsilon = eps*_fid_mix(rho_S)   (loaded from sensitivity_results_mix.pt)
  (c) epsilon = eps*_ss_mix(rho_S)    (Algorithm 1 on the T=0 E_mix MDP)

Saves results to <save_dir>/single_shot_baseline_mix.pt.

Usage:
    python single_shot_baseline_mix.py --config ../config/single_shot_mix.yaml
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from MDP_solver_mix import SubsidizedMDPSolverMix
from optimal_subsidy_mix import find_optimal_subsidy_mix
from sensitivity_analysis_mix import _mc_true
from rejection_mix import build_rejection_table_np


SOLVER_KEYS = ["rho_A", "T", "n_max", "c0", "c1", "epsilon",
               "kappa", "theta_b", "alpha_0", "beta_0", "device",
               "action_stride"]


def _solve_single_shot_mix(base_solver_cfg: dict, epsilon: float, save_dir: str,
                           chunk_size: int = 4096):
    """Solve the single-shot E_mix MDP once; return (policy, V0, A, P) at (N=0, X=0)."""
    cfg = {k: base_solver_cfg[k] for k in SOLVER_KEYS if k in base_solver_cfg}
    cfg["epsilon"] = float(epsilon)
    solver = SubsidizedMDPSolverMix(**cfg)
    results = solver.solve(save_dir=save_dir, chunk_size=chunk_size)
    policy = {l: results["Policy"][l] for l in range(solver.T + 1)}
    V0 = float(results["V_0"][0][0, 0])
    A  = float(results["A"][0][0, 0])
    P  = float(results["P_approval"][0][0, 0])
    return policy, V0, A, P


def run(config_path: str, n_episodes: int = 200_000, seed: int = 42, verbose: bool = True,
        theta_fid: float = 0.65):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    base_solver_cfg = {k: config[k] for k in SOLVER_KEYS if k in config}
    rho_S_range      = [float(v) for v in config["rho_S_range"]]
    theta_star_range = [float(v) for v in config["theta_star_range"]]
    save_dir         = config.get("save_dir", "./single_shot_output_mix")
    fid_path         = config["fiducial_sensitivity_path"]
    theta_fid        = float(config.get("theta_fid", theta_fid))

    os.makedirs(save_dir, exist_ok=True)
    mdp_scratch = os.path.join(save_dir, "mdp_solves")

    # ---- load fiducial_mix eps*(rho_S) ----------------------------------
    fid = torch.load(fid_path, weights_only=False)
    fid_rho    = [float(v) for v in fid["rho_S_range"]]
    fid_eps    = [float(v) for v in fid["eps_star"]]
    rho_to_eps = dict(zip(fid_rho, fid_eps))
    missing = [r for r in rho_S_range if r not in rho_to_eps]
    if missing:
        raise ValueError(
            f"rho_S values {missing} not found in fiducial_mix sensitivity results at {fid_path}"
        )

    # MC parameters (single-shot: T=0)
    mc_params = {k: config[k] for k in ["T", "alpha_0", "beta_0", "kappa",
                                        "theta_b", "c0", "c1"]}

    # E_mix rejection table for MC rollouts.
    # Horizon is T+1 actions of up to n_max trials, so cumulative N can reach
    # (T+1)*n_max — covers the T=0 single-shot case (n_max) and any T>=1.
    max_N = (int(config['T']) + 1) * int(config['n_max'])
    rejection_table = build_rejection_table_np(
        max_N, float(config['kappa']), float(config['theta_b'])
    )

    n_rho = len(rho_S_range)
    n_th  = len(theta_star_range)

    # Case (a): epsilon = 0
    P_ss0    = np.full((n_rho, n_th), np.nan)
    A_ss0    = np.full((n_rho, n_th), np.nan)
    us_ss0   = np.full((n_rho, n_th), np.nan)

    # Case (b): epsilon = eps*_fid_mix(rho_S)
    P_sseps  = np.full((n_rho, n_th), np.nan)
    A_sseps  = np.full((n_rho, n_th), np.nan)
    us_sseps = np.full((n_rho, n_th), np.nan)

    # Case (c): epsilon = eps*_ss_mix(rho_S)
    P_ssopt    = np.full((n_rho, n_th), np.nan)
    A_ssopt    = np.full((n_rho, n_th), np.nan)
    us_ssopt   = np.full((n_rho, n_th), np.nan)
    eps_ss_opt = np.full(n_rho, np.nan)

    eps_fid_used = np.array([rho_to_eps[r] for r in rho_S_range])

    j_fid = next((jj for jj, th in enumerate(theta_star_range)
                  if abs(th - theta_fid) < 1e-9), None)
    if j_fid is None and verbose:
        print(f"  [info] theta_fid={theta_fid} not in theta_star_range; "
              f"skipping per-episode sample collection.")
    samples_ss0_approved   = (np.zeros(n_episodes, dtype=np.int8)
                              if j_fid is not None else None)
    samples_ss0_cost       = (np.zeros(n_episodes, dtype=np.float32)
                              if j_fid is not None else None)
    samples_ssopt_approved = (np.zeros((n_rho, n_episodes), dtype=np.int8)
                              if j_fid is not None else None)
    samples_ssopt_cost     = (np.zeros((n_rho, n_episodes), dtype=np.float32)
                              if j_fid is not None else None)

    ss_alg1_base = {k: base_solver_cfg[k] for k in SOLVER_KEYS if k != "epsilon"}
    epsilon_max  = float(config.get("epsilon_max", 1.0))
    tol          = float(config.get("tol", 1e-6))
    chunk_size   = int(config.get("chunk_size", 4096))

    t0 = time.time()

    # ---- (a) epsilon = 0 ------------------------------------------------
    if verbose:
        print("\n=== Single-shot baseline (mix, a): epsilon = 0 ===")
    pol0, V0_0, A0_0, P0_0 = _solve_single_shot_mix(base_solver_cfg, 0.0, mdp_scratch,
                                                    chunk_size=chunk_size)
    n_star0 = int(pol0[0][0, 0])
    if verbose:
        print(f"  policy n*(0,0) = {n_star0}   V0={V0_0:.4f}  A={A0_0:.4f}  P={P0_0:.4f}")

    for j, theta_star in enumerate(theta_star_range):
        mc = _mc_true(pol0, theta_star, mc_params, rejection_table,
                      n_episodes=n_episodes, seed=seed + j,
                      return_samples=(j == j_fid))
        p, a = mc["p_approval"], mc["a_true"]
        for i, rho_S in enumerate(rho_S_range):
            P_ss0[i, j]  = p
            A_ss0[i, j]  = a
            us_ss0[i, j] = rho_S * p - 0.0 * a
        if j == j_fid:
            samples_ss0_approved[:] = mc["samples"]["approved"]
            samples_ss0_cost[:]     = mc["samples"]["cost"]
        if verbose:
            print(f"  theta*={theta_star:.2f}  P={p:.4f}  A={a:.4f}")

    # ---- (b) epsilon = eps*_fid_mix(rho_S) ------------------------------
    if verbose:
        print("\n=== Single-shot baseline (mix, b): epsilon = eps*_fid_mix(rho_S) ===")

    mc_cache: dict[tuple[float, float], dict] = {}

    for i, rho_S in enumerate(rho_S_range):
        eps_i = rho_to_eps[rho_S]
        if verbose:
            print(f"\n  rho_S={rho_S:8.1f}  eps*_fid_mix={eps_i:.6f}")

        pol_i, V0_i, A_i, P_i = _solve_single_shot_mix(base_solver_cfg, eps_i, mdp_scratch,
                                                       chunk_size=chunk_size)
        n_star_i = int(pol_i[0][0, 0])
        if verbose:
            print(f"    policy n*(0,0) = {n_star_i}   V0={V0_i:.4f}  A={A_i:.4f}  P={P_i:.4f}")

        for j, theta_star in enumerate(theta_star_range):
            ckey = (round(eps_i, 12), theta_star)
            if ckey not in mc_cache:
                mc_cache[ckey] = _mc_true(
                    pol_i, theta_star, mc_params, rejection_table,
                    n_episodes=n_episodes,
                    seed=seed + 10_000 + i * n_th + j,
                )
            mc = mc_cache[ckey]
            p, a = mc["p_approval"], mc["a_true"]
            P_sseps[i, j]  = p
            A_sseps[i, j]  = a
            us_sseps[i, j] = rho_S * p - eps_i * a

    # ---- (c) eps = eps*_ss_mix(rho_S): single-shot's own optimum --------
    if verbose:
        print("\n=== Single-shot baseline (mix, c): eps = eps*_ss_mix(rho_S) (Algorithm 1 on T=0 E_mix) ===")

    for i, rho_S in enumerate(rho_S_range):
        if verbose:
            print(f"\n  rho_S={rho_S:8.1f}  running Algorithm 1 (single-shot, mix) ...")

        epsilons_ss, us_ss_list, eps_star_ss, _us_star_ss, bp_stats_ss, pol_per_bp_ss = \
            find_optimal_subsidy_mix(
                ss_alg1_base,
                rho_S=rho_S,
                epsilon_max=epsilon_max,
                tol=tol,
                save_dir=mdp_scratch,
                verbose=False,
                chunk_size=chunk_size,
            )
        best_idx = int(np.argmax(us_ss_list))
        eps_ss_opt[i] = eps_star_ss
        pol_ss = pol_per_bp_ss[best_idx]
        n_star_c = int(pol_ss[0][0, 0])
        if verbose:
            print(f"    eps*_ss_mix = {eps_star_ss:.6f}   n*(0,0) = {n_star_c}")

        for j, theta_star in enumerate(theta_star_range):
            ckey = (round(eps_star_ss, 12), theta_star)
            want_samples = (j == j_fid)
            cached = mc_cache.get(ckey)
            if cached is None or (want_samples and "samples" not in cached):
                cached = _mc_true(
                    pol_ss, theta_star, mc_params, rejection_table,
                    n_episodes=n_episodes,
                    seed=seed + 20_000 + i * n_th + j,
                    return_samples=want_samples,
                )
                mc_cache[ckey] = cached
            mc = cached
            p, a = mc["p_approval"], mc["a_true"]
            P_ssopt[i, j]  = p
            A_ssopt[i, j]  = a
            us_ssopt[i, j] = rho_S * p - eps_star_ss * a
            if want_samples:
                samples_ssopt_approved[i] = mc["samples"]["approved"]
                samples_ssopt_cost[i]     = mc["samples"]["cost"]

    out = {
        "rho_S_range":       rho_S_range,
        "theta_star_range":  theta_star_range,
        "eps_fid_used":      eps_fid_used.tolist(),
        "eps_ss_opt":        eps_ss_opt.tolist(),
        "P_true_ss0":        P_ss0.tolist(),
        "A_true_ss0":        A_ss0.tolist(),
        "us_true_ss0":       us_ss0.tolist(),
        "P_true_ss_epsfid":  P_sseps.tolist(),
        "A_true_ss_epsfid":  A_sseps.tolist(),
        "us_true_ss_epsfid": us_sseps.tolist(),
        "P_true_ss_epsopt":  P_ssopt.tolist(),
        "A_true_ss_epsopt":  A_ssopt.tolist(),
        "us_true_ss_epsopt": us_ssopt.tolist(),
        "params": {
            **{k: config[k] for k in SOLVER_KEYS if k in config},
            "n_episodes":               n_episodes,
            "fiducial_sensitivity_path": fid_path,
            "theta_fid":                theta_fid,
            "e_value":                  "mixture_uniform",
        },
    }
    if j_fid is not None:
        out["theta_fid"]              = theta_fid
        out["samples_ss0_approved"]   = samples_ss0_approved
        out["samples_ss0_cost"]       = samples_ss0_cost
        out["samples_ssopt_approved"] = samples_ssopt_approved
        out["samples_ssopt_cost"]     = samples_ssopt_cost
    out_path = os.path.join(save_dir, "single_shot_baseline_mix.pt")
    torch.save(out, out_path)
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
    print(f"Single-shot (E_mix) baseline saved to: {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-shot (T=0) baseline using the mixture e-value E_mix.")
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=200_000)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--no_verbose", dest="verbose", action="store_false")
    parser.add_argument("--theta_fid",  type=float, default=0.65,
                        help="theta* at which per-episode samples are stored (default: 0.65).")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    run(args.config, n_episodes=args.n_episodes, seed=args.seed, verbose=args.verbose,
        theta_fid=args.theta_fid)
