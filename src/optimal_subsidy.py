"""
Algorithm 1: Divide-and-conquer to find the principal's optimal subsidy.

Finds the partition of [0, epsilon_max] where the agent's optimal policy is
constant (Proposition 8), evaluates the social utility U^S at each breakpoint
(Eq. 18/19), and returns the Stackelberg-optimal subsidy epsilon*.

Key identities used:
  - Proposition 7:  V^eps_pi(S, l) = V^0_pi(S, l) + eps * A_pi(S, l)
  - Eq. 19:         max_{eps in [eps_i, eps_{i+1})} U^S(eps; pi_i) = U^S(eps_i; pi_i)
  - Eq. 18/point Q: U^S(eps; pi) = rho_S * P_approval - eps * A   at initial state

Algorithm 1 logic (Section 5):
  For an interval [eps_L, eps_R] with policies pi_L (optimal at eps_L) and pi_R
  (optimal at eps_R), their linear value functions intersect at
      eps_int = (V0_L - V0_R) / (A_R - A_L).
  Solving the MDP at eps_int gives V*(eps_int) = V0_int + eps_int * A_int.

  TRUE branch  (V*(eps_int) ≈ f_L(eps_int)):  by convexity of V*, this means V* is
    exactly linear on [eps_L, eps_int] (= f_L) and on [eps_int, eps_R] (= f_R).
    eps_int is a genuine partition breakpoint (pi_L → pi_R transition).
    Social utility at eps_int is U^S(eps_int; pi_R) = rho_S*P_R - eps_int*A_R,
    using pi_R's data from the RIGHT endpoint (paper, Algorithm 1 line 14).

  ELSE branch  (V*(eps_int) > f_L(eps_int)):  a new policy pi_int strictly improves
    at eps_int; split into [eps_L, eps_int] and [eps_int, eps_R] and recurse.

The MDP solver (MDP_solver.py) already computes V_0 and A as part of the backward
induction, so no extra passes are needed — each SolveMDP call returns everything.
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
import torch

# Allow importing MDP_solver from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from MDP_solver import SubsidizedMDPSolver


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _make_solver_config(base_config: dict, epsilon: float) -> dict:
    cfg = base_config.copy()
    cfg["epsilon"] = epsilon
    return cfg


def _solve_mdp(epsilon: float, base_config: dict, save_dir: str | None, verbose: bool):
    """
    Solve the MDP at a given epsilon.

    Returns (V0, A, P, policy) where V0/A/P are scalar floats at the initial
    state (N=0, X=0), l=0, and policy is a dict {l: 2-D numpy array} of actions.
      V0     : V^0_{pi}(alpha_0, beta_0, 0, 0)   — agent utility with no subsidy
      A      : A_{pi}(alpha_0, beta_0, 0, 0)       — expected cost conditional on approval
      P      : P_{pi}(approval | initial state)    — approval probability
      policy : {l: Policy[l]} numpy arrays indexed by [N, X]
    """
    cfg = _make_solver_config(base_config, epsilon)
    solver = SubsidizedMDPSolver(**cfg)
    sd = save_dir if save_dir is not None else "./mdp_scratch_subsidy"
    results = solver.solve(save_dir=sd)

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

def find_optimal_subsidy(
    base_config: dict,
    rho_S: float,
    epsilon_max: float = 1.0,
    tol: float = 1e-6,
    save_dir: str | None = None,
    verbose: bool = True,
) -> tuple[list[float], list[float], float, float]:
    """
    Algorithm 1: Divide-and-conquer search for the principal's optimal subsidy.

    Parameters
    ----------
    base_config   : MDP solver config WITHOUT 'epsilon' (it is varied internally).
                    Must contain: rho_A, T, n_max, c0, c1, kappa, theta_b,
                                  alpha_0, beta_0, device.
    rho_S         : Principal's (social) utility from approval.
    epsilon_max   : Maximum subsidy fraction (paper: eps^max <= 1). Default 1.0.
    tol           : Tolerance for detecting identical policies (value difference).
    save_dir      : Where to write per-epsilon MDP result files (None = temp dir).
    verbose       : Print progress.

    Returns
    -------
    epsilons          : Sorted list of partition breakpoints (including 0).
    social_utilities  : U^S evaluated at each breakpoint (same order).
    eps_star          : Optimal subsidy eps* = argmax U^S.
    us_star           : Optimal social utility U^S(eps*).
    breakpoint_stats      : List of (V0, A, P) at each breakpoint (same order).
                            V0 and A decompose the agent value via Proposition 7;
                            P is the approval probability under the optimal policy
                            at that breakpoint.
    policy_per_breakpoint : List of policy dicts {l: 2-D numpy array} at each
                            breakpoint. policy[l][N, X] gives the optimal action
                            (number of trials) at time l with accumulated counts
                            (N, X); 0 means opt-out.
    """
    n_solves = 0
    cache: dict[float, tuple[float, float, float, dict]] = {}  # eps -> (V0, A, P, policy)
    # For each breakpoint eps_int discovered in the TRUE branch, record the cache
    # key of the RIGHT endpoint policy (pi_R), which is optimal on [eps_int, next).
    # Needed because cache[eps_int] stores pi_int ≈ pi_L (left policy), not pi_R.
    right_key: dict[float, float] = {}  # round(eps_int, 14) -> round(eps_R, 14)

    def solve(epsilon: float) -> tuple[float, float, float]:
        nonlocal n_solves
        key = round(epsilon, 14)
        if key not in cache:
            n_solves += 1
            if verbose:
                print(f"  [Solve #{n_solves}]  eps = {epsilon:.10f}")
            cache[key] = _solve_mdp(epsilon, base_config, save_dir, verbose)
        V0, A, P, _ = cache[key]
        return V0, A, P

    # ------------------------------------------------------------------
    # Initialise: solve at both endpoints, seed U with eps=0 (Algorithm 1, line 2)
    # ------------------------------------------------------------------
    V0_0, A_0, P_0 = solve(0.0)
    V0_max, A_max, P_max = solve(epsilon_max)

    # U maps each partition breakpoint -> social utility U^S(eps_i; pi_i).
    # Seeded with eps=0 under pi_0 (the policy optimal at eps=0).
    U: dict[float, float] = {0.0: _social_utility(rho_S, P_0, 0.0, A_0)}

    # Stack entries: (eps_L, V0_L, A_L, P_L, eps_R, V0_R, A_R, P_R)
    stack = [(0.0, V0_0, A_0, P_0, epsilon_max, V0_max, A_max, P_max)]

    # ------------------------------------------------------------------
    # Divide-and-conquer main loop (Algorithm 1, lines 7–19)
    # ------------------------------------------------------------------
    while stack:
        eps_L, V0_L, A_L, P_L, eps_R, V0_R, A_R, P_R = stack.pop()

        # If the two endpoint policies respond identically to eps, no breakpoint inside.
        if abs(A_R - A_L) < tol:
            continue

        # Intersection of the two linear agent value functions (line 10):
        #   V0_L + eps * A_L  =  V0_R + eps * A_R
        #   => eps_int = (V0_L - V0_R) / (A_R - A_L)
        eps_int = (V0_L - V0_R) / (A_R - A_L)

        # Must be strictly inside the open interval to be meaningful.
        if eps_int <= eps_L + tol or eps_int >= eps_R - tol:
            continue

        V0_int, A_int, P_int = solve(eps_int)  # line 11

        # V*(eps_int): optimal agent value at eps_int under the returned policy pi_int.
        # f_L(eps_int): what pi_L (optimal at eps_L) would give at eps_int.
        # By convexity, V*(eps_int) >= f_L(eps_int) always.
        val_opt      = V0_int + eps_int * A_int
        val_L_at_int = V0_L   + eps_int * A_L

        if val_opt <= val_L_at_int + tol:
            # TRUE branch (line 13-14): V* coincides with f_L at eps_int.
            # By convexity of V*, this proves V* = f_L on [eps_L, eps_int] and
            # V* = f_R on [eps_int, eps_R] — no further breakpoints in either piece.
            # eps_int is the genuine pi_L -> pi_R transition point.
            # Social utility uses pi_R's data (right endpoint policy), per line 14.
            U[eps_int] = _social_utility(rho_S, P_R, eps_int, A_R)
            # Record that the policy for interval [eps_int, ...) is pi_R = cache[eps_R].
            right_key[round(eps_int, 14)] = round(eps_R, 14)
        else:
            # ELSE branch (line 15-16): pi_int is a genuinely new policy.
            # Split [eps_L, eps_R] at eps_int and recurse into both halves.
            stack.append((eps_int, V0_int, A_int, P_int, eps_R,  V0_R,  A_R,  P_R))
            stack.append((eps_L,  V0_L,  A_L,  P_L,  eps_int, V0_int, A_int, P_int))

    # ------------------------------------------------------------------
    # Build sorted output (line 20)
    # ------------------------------------------------------------------
    sorted_items = sorted(U.items())
    epsilons: list[float]         = [item[0] for item in sorted_items]
    social_utilities: list[float] = [item[1] for item in sorted_items]

    # Pull (V0, A, P, policy) for each breakpoint using the RIGHT endpoint's cache.
    # For breakpoint eps_i (i > 0), the policy on [eps_i, eps_{i+1}) is pi_R —
    # the policy from the right endpoint of the interval in which eps_i was found.
    # cache[eps_i] stores pi_int ≈ pi_L (torch.max ties go to smaller actions),
    # so we must look up right_key[eps_i] to get pi_R.
    # For eps_0 = 0 (seeded directly), cache[0] = pi_0 is already correct.
    breakpoint_stats: list[tuple[float, float, float]] = []
    policy_per_breakpoint: list[dict] = []
    for eps in epsilons:
        key = round(eps, 14)
        rk = right_key.get(key, key)   # fall back to cache[eps] for eps=0
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
        description="Algorithm 1: Find the principal's optimal subsidy via divide-and-conquer."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # MDP solver parameters (epsilon is varied internally)
    solver_keys = ["rho_A", "T", "n_max", "c0", "c1", "kappa", "theta_b", "alpha_0", "beta_0", "device"]
    base_config = {k: config[k] for k in solver_keys}

    rho_S       = float(config["rho_S"])
    epsilon_max = float(config.get("epsilon_max", 1.0))
    tol         = float(config.get("tol", 1e-6))
    save_dir    = config.get("save_dir", "./mdp_output")
    verbose     = bool(config.get("verbose", True))

    print("Running Algorithm 1 with parameters:")
    print(f"  rho_S       = {rho_S}")
    print(f"  epsilon_max = {epsilon_max}")
    print(f"  tol         = {tol}")
    print(f"  save_dir    = {save_dir}")
    for k, v in base_config.items():
        print(f"  {k} = {v}")
    print()

    t0 = time.time()
    epsilons, social_utilities, eps_star, us_star, breakpoint_stats, policy_per_breakpoint = find_optimal_subsidy(
        base_config,
        rho_S=rho_S,
        epsilon_max=epsilon_max,
        tol=tol,
        save_dir=save_dir,
        verbose=verbose,
    )
    elapsed = time.time() - t0

    print(f"\nElapsed: {elapsed:.2f}s")
    print("\nPartition (eps_i, U^S(eps_i), P_i, A_i):")
    for eps, us, (v0, a, p) in zip(epsilons, social_utilities, breakpoint_stats):
        marker = " <-- OPTIMAL" if abs(eps - eps_star) < 1e-12 else ""
        print(f"  eps = {eps:.8f}   U^S = {us:.8f}   P = {p:.6f}   A = {a:.4f}{marker}")

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    out = {
        "epsilons":         epsilons,
        "social_utilities": social_utilities,
        "eps_star":         eps_star,
        "us_star":          us_star,
        "elapsed_s":        elapsed,
        # Per-breakpoint (V0, A, P) for the policy optimal at each eps_i.
        # Within interval [eps_i, eps_{i+1}), U^S(eps) = rho_S*P_i - eps*A_i (Eq. 18).
        "V0_per_breakpoint":     [s[0] for s in breakpoint_stats],
        "A_per_breakpoint":      [s[1] for s in breakpoint_stats],
        "P_per_breakpoint":      [s[2] for s in breakpoint_stats],
        "policy_per_breakpoint": policy_per_breakpoint,
        "params": {
            **base_config,
            "rho_S":       rho_S,
            "epsilon_max": epsilon_max,
            "tol":         tol,
        },
    }
    out_path = os.path.join(save_dir, "optimal_subsidy_results.pt")
    torch.save(out, out_path)
    print(f"\nResults saved to {out_path}")
