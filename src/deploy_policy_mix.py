"""
PolicyEvaluator for the E_mix approval rule.

Mirrors deploy_policy.PolicyEvaluator exactly; only _check_rejection changes:
instead of the log-linear formula, it consults a precomputed boolean table
encoding E_mix(N, X) >= 1/kappa (see rejection_mix.build_rejection_table_np).
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from rejection_mix import build_rejection_table_np


class PolicyEvaluatorMix:
    """Evaluate a pre-computed E_mix MDP policy under the true efficacy theta_star.

    Same semantics as deploy_policy.PolicyEvaluator: outcomes are drawn from
    Bin(n_t, theta_star) while the policy uses the agent's Beta-Binomial belief.
    Only the approval test differs.
    """

    def __init__(self, rho_A, T, n_max, c0, c1, epsilon, kappa, theta_b,
                 alpha_0, beta_0, theta_star, policy_path,
                 n_episodes=200_000, seed=42):
        self.rho_A      = rho_A
        self.T          = T
        self.n_max      = n_max
        self.c0         = c0
        self.c1         = c1
        self.epsilon    = epsilon
        self.kappa      = kappa
        self.theta_b    = theta_b
        self.alpha_0    = alpha_0
        self.beta_0     = beta_0
        self.theta_star = theta_star
        self.n_episodes = n_episodes
        self.seed       = seed

        print(f"Loading policy from: {policy_path}")
        data = torch.load(policy_path, weights_only=False)
        self.policy      = {l: data['Policy'][l] for l in range(T + 1)}
        self.policy_data = data
        saved_params = data.get('params', {})
        if saved_params:
            for key in ('rho_A', 'T', 'n_max', 'c0', 'c1', 'epsilon',
                        'kappa', 'theta_b', 'alpha_0', 'beta_0'):
                saved = saved_params.get(key)
                given = getattr(self, key)
                if saved is not None and not np.isclose(float(saved), float(given)):
                    print(f"  WARNING: param '{key}' in policy file ({saved}) "
                          f"differs from config ({given})")

        # Horizon is T+1 actions of up to n_max trials, so cumulative N reaches
        # (T+1)*n_max. The solver sizes its own table at T*n_max and clamps,
        # which is safe internally; the evaluator must look up real terminal
        # states, so the table needs the full upper bound.
        max_N = int((T + 1) * n_max)
        self.rejection_table = build_rejection_table_np(max_N, kappa, theta_b)

    def _check_rejection(self, N, X):
        """Boolean approval test via table lookup. N, X are ints in [0, T*n_max]."""
        return bool(self.rejection_table[N, X])

    def evaluate(self, track_beliefs=True):
        """Run Monte Carlo evaluation of the policy under theta_star."""
        rng = np.random.default_rng(self.seed)

        n_approved        = 0
        sum_cost_approved = 0.0
        sum_agent_utility = 0.0

        belief_paths = [] if track_beliefs else None

        for _ in range(self.n_episodes):
            N, X     = 0, 0
            C        = 0.0
            approved = False

            if track_beliefs:
                path = [(self.alpha_0, self.beta_0)]

            for l in range(self.T + 1):
                n_star = int(self.policy[l][N, X])

                if n_star == 0:
                    break

                cost_step = self.c0 + self.c1 * n_star
                C        += cost_step
                x         = int(rng.binomial(n_star, self.theta_star))
                N        += n_star
                X        += x

                if track_beliefs:
                    alpha = self.alpha_0 + X
                    beta  = self.beta_0 + N - X
                    path.append((alpha, beta))

                if self._check_rejection(N, X):
                    approved = True
                    break

            if track_beliefs and len(belief_paths) < 20000:
                belief_paths.append(path)

            if approved:
                n_approved        += 1
                sum_cost_approved += C
                sum_agent_utility += self.rho_A + self.epsilon * C - C
            else:
                sum_agent_utility += -C

        n = self.n_episodes
        p_approval            = n_approved / n
        e_cost_given_approval = (sum_cost_approved / n_approved
                                 if n_approved > 0 else float('nan'))
        agent_utility         = sum_agent_utility / n

        return {
            'theta_star':            self.theta_star,
            'epsilon':               self.epsilon,
            'p_approval':            p_approval,
            'e_cost_given_approval': e_cost_given_approval,
            'agent_utility':         agent_utility,
            'n_episodes':            n,
            'n_approved':            n_approved,
            'belief_paths':          belief_paths,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a pre-computed E_mix MDP policy under the real efficacy theta_star.'
    )
    parser.add_argument('--config',      type=str, required=True,
                        help='Path to YAML config file (must include theta_star).')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to the .pt file produced by MDP_solver_mix.py.')
    parser.add_argument('--n_episodes',  type=int, default=200_000,
                        help='Number of Monte Carlo trajectories (default: 200000).')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    evaluator = PolicyEvaluatorMix(
        rho_A       = config['rho_A'],
        T           = config['T'],
        n_max       = config['n_max'],
        c0          = config['c0'],
        c1          = config['c1'],
        epsilon     = config['epsilon'],
        kappa       = config['kappa'],
        theta_b     = config['theta_b'],
        alpha_0     = config['alpha_0'],
        beta_0      = config['beta_0'],
        theta_star  = config['theta_star'],
        policy_path = args.policy_path,
        n_episodes  = args.n_episodes,
        seed        = args.seed,
    )

    print(f"\nEvaluating E_mix policy under theta* = {config['theta_star']} "
          f"with epsilon = {config['epsilon']} "
          f"over {args.n_episodes:,} episodes ...\n")

    start = time.time()
    results = evaluator.evaluate()
    elapsed = time.time() - start

    print(f"Results (theta* = {results['theta_star']}, epsilon = {results['epsilon']}):")
    print(f"  P(approval)                   = {results['p_approval']:.4f}  "
          f"({results['n_approved']:,} / {results['n_episodes']:,})")
    print(f"  E[total cost | approved]      = {results['e_cost_given_approval']:.4f}")
    print(f"  Agent utility                 = {results['agent_utility']:.4f}")
    print(f"\nEvaluation time: {elapsed:.2f} s")
