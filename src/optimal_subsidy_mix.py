"""
Algorithm 1 using the mixture e-value E_mix (uniformly powerful variant).

Mirrors optimal_subsidy.py but swaps the MDP solver for SubsidizedMDPSolverMix,
which uses the method-of-mixtures e-value (uniform mixing over [theta_b, 1])
instead of the log-linear e-value of Eq. 5.

The divide-and-conquer logic is identical: Proposition 7 (linear decomposition
V^eps = V^0 + eps * A), Proposition 8 (piecewise-linear convex value), and
Eq. 18/19 (U^S = rho_S * P - eps * A, maximized at the left endpoint of each
partition interval) all hold for any valid sequential e-value test, because
they only depend on the MDP reward structure, not on the specific rejection
rule. Only the SolveMDP subroutine changes.
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


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _make_solver_config(base_config: dict, epsilon: float) -> dict:
    cfg = base_config.copy()
    cfg["epsilon"] = epsilon
    return cfg


def _solve_mdp(epsilon: float, base_config: dict, save_dir: str | None, verbose: bool,
               chunk_size: int = 4096):
    """Solve the E_mix MDP at a given epsilon; return (V0, A, P, policy) at the initial state."""
    cfg = _make_solver_config(base_config, epsilon)
    solver = SubsidizedMDPSolverMix(**cfg)
    sd = save_dir if save_dir is not None else "./mdp_scratch_subsidy_mix"
    results = solver.solve(save_dir=sd, chunk_size=chunk_size)

    V0     = float(results["V_0"][0][0, 0])
    A      = float(results["A"][0][0, 0])
    P      = float(results["P_approval"][0][0, 0])
    policy = {l: results["Policy"][l] for l in range(solver.T + 1)}

    if verbose:
        Veps = float(results["V_eps"][0][0, 0])
        print(f"    V_eps={Veps:.6f}  V0={V0:.6f}  A={A:.6f}  P={P:.6f}")

    return V0, A, P, policy


def _social_utility(rho_S: float, P: float, epsilon: float, A: float) -> float:
    """U^S(eps; pi) = rho_S * P_approval - eps * A  (Eq. 18, point-mass Q)."""
    return rho_S * P - epsilon * A


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def find_optimal_subsidy_mix(
    base_config: dict,
    rho_S: float,
    epsilon_max: float = 1.0,
    tol: float = 1e-6,
    save_dir: str | None = None,
    verbose: bool = True,
    chunk_size: int = 4096,
) -> tuple[list[float], list[float], float, float, list, list]:
    """Divide-and-conquer search for the principal's optimal subsidy, using E_mix.

    Signature and semantics mirror optimal_subsidy.find_optimal_subsidy.
    See that file for the full docstring; only the MDP solver differs.
    """
    n_solves = 0
    cache: dict[float, tuple[float, float, float, dict]] = {}
    right_key: dict[float, float] = {}

    def solve(epsilon: float) -> tuple[float, float, float]:
        nonlocal n_solves
        key = round(epsilon, 14)
        if key not in cache:
            n_solves += 1
            if verbose:
                print(f"  [Solve #{n_solves}]  eps = {epsilon:.10f}")
            cache[key] = _solve_mdp(epsilon, base_config, save_dir, verbose, chunk_size=chunk_size)
        V0, A, P, _ = cache[key]
        return V0, A, P

    V0_0, A_0, P_0 = solve(0.0)
    V0_max, A_max, P_max = solve(epsilon_max)

    U: dict[float, float] = {0.0: _social_utility(rho_S, P_0, 0.0, A_0)}

    stack = [(0.0, V0_0, A_0, P_0, epsilon_max, V0_max, A_max, P_max)]

    while stack:
        eps_L, V0_L, A_L, P_L, eps_R, V0_R, A_R, P_R = stack.pop()

        if abs(A_R - A_L) < tol:
            continue

        eps_int = (V0_L - V0_R) / (A_R - A_L)

        if eps_int <= eps_L + tol or eps_int >= eps_R - tol:
            continue

        V0_int, A_int, P_int = solve(eps_int)

        val_opt      = V0_int + eps_int * A_int
        val_L_at_int = V0_L   + eps_int * A_L

        if val_opt <= val_L_at_int + tol:
            U[eps_int] = _social_utility(rho_S, P_R, eps_int, A_R)
            right_key[round(eps_int, 14)] = round(eps_R, 14)
        else:
            stack.append((eps_int, V0_int, A_int, P_int, eps_R,  V0_R,  A_R,  P_R))
            stack.append((eps_L,  V0_L,  A_L,  P_L,  eps_int, V0_int, A_int, P_int))

    sorted_items = sorted(U.items())
    epsilons: list[float]         = [item[0] for item in sorted_items]
    social_utilities: list[float] = [item[1] for item in sorted_items]

    breakpoint_stats: list[tuple[float, float, float]] = []
    policy_per_breakpoint: list[dict] = []
    for eps in epsilons:
        key = round(eps, 14)
        rk = right_key.get(key, key)
        v0, a, p, pol = cache[rk]
        breakpoint_stats.append((v0, a, p))
        policy_per_breakpoint.append(pol)

    best_idx = int(np.argmax(social_utilities))
    eps_star = epsilons[best_idx]
    us_star  = social_utilities[best_idx]

    if verbose:
        print(f"\n  Total MDP solves : {n_solves}")
        print(f"  Partition size   : {len(epsilons)}")
        print(f"  Optimal subsidy  : eps* = {eps_star:.8f}")
        print(f"  Optimal U^S      : {us_star:.8f}")

    return epsilons, social_utilities, eps_star, us_star, breakpoint_stats, policy_per_breakpoint


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Algorithm 1 (E_mix variant): find the principal's optimal subsidy."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    solver_keys = ["rho_A", "T", "n_max", "c0", "c1", "kappa", "theta_b",
                   "alpha_0", "beta_0", "device", "action_stride"]
    base_config = {k: config[k] for k in solver_keys if k in config}

    rho_S       = float(config["rho_S"])
    epsilon_max = float(config.get("epsilon_max", 1.0))
    tol         = float(config.get("tol", 1e-6))
    save_dir    = config.get("save_dir", "./mdp_output_mix")
    verbose     = bool(config.get("verbose", True))
    chunk_size  = int(config.get("chunk_size", 4096))

    print("Running Algorithm 1 (E_mix) with parameters:")
    print(f"  rho_S       = {rho_S}")
    print(f"  epsilon_max = {epsilon_max}")
    print(f"  tol         = {tol}")
    print(f"  save_dir    = {save_dir}")
    for k, v in base_config.items():
        print(f"  {k} = {v}")
    print()

    t0 = time.time()
    epsilons, social_utilities, eps_star, us_star, breakpoint_stats, policy_per_breakpoint = find_optimal_subsidy_mix(
        base_config,
        rho_S=rho_S,
        epsilon_max=epsilon_max,
        tol=tol,
        save_dir=save_dir,
        verbose=verbose,
        chunk_size=chunk_size,
    )
    elapsed = time.time() - t0

    print(f"\nElapsed: {elapsed:.2f}s")
    print("\nPartition (eps_i, U^S(eps_i), P_i, A_i):")
    for eps, us, (v0, a, p) in zip(epsilons, social_utilities, breakpoint_stats):
        marker = " <-- OPTIMAL" if abs(eps - eps_star) < 1e-12 else ""
        print(f"  eps = {eps:.8f}   U^S = {us:.8f}   P = {p:.6f}   A = {a:.4f}{marker}")

    os.makedirs(save_dir, exist_ok=True)
    out = {
        "epsilons":         epsilons,
        "social_utilities": social_utilities,
        "eps_star":         eps_star,
        "us_star":          us_star,
        "elapsed_s":        elapsed,
        "V0_per_breakpoint":     [s[0] for s in breakpoint_stats],
        "A_per_breakpoint":      [s[1] for s in breakpoint_stats],
        "P_per_breakpoint":      [s[2] for s in breakpoint_stats],
        "policy_per_breakpoint": policy_per_breakpoint,
        "params": {
            **base_config,
            "rho_S":       rho_S,
            "epsilon_max": epsilon_max,
            "tol":         tol,
            "e_value":     "mixture_uniform",
        },
    }
    out_path = os.path.join(save_dir, "optimal_subsidy_mix_results.pt")
    torch.save(out, out_path)
    print(f"\nResults saved to {out_path}")
