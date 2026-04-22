import argparse
import time

import numpy as np
import torch
import yaml


class PolicyEvaluator:
    """
    Evaluates a pre-computed MDP policy under the real efficacy parameter theta_star.

    In the MDP solver, outcomes X_t are averaged over theta ~ Beta(alpha_t, beta_t)
    (the agent's belief), enabling tractable planning. Here, we instead draw
    X_t ~ Bin(n_t, theta_star) to measure how the policy actually performs in the
    real world for a given true efficacy.

    Computed quantities (all under theta_star and the given policy):
      - P(approval)
      - E[total cost | approved]
      - Agent utility: E[(rho_A + epsilon * C_tau) * 1{approved} - C_tau]
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

        # Load policy from saved MDP results
        print(f"Loading policy from: {policy_path}")
        data = torch.load(policy_path, weights_only=False)
        # policy[l] is a 2-D numpy array indexed by [N, X]
        self.policy = {l: data['Policy'][l] for l in range(T + 1)}
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

        # Rejection threshold constants (same formula as solver)
        self.log_kappa_inv = np.log(1.0 / kappa)
        self.log_term      = np.log(1.0 + theta_b * (np.e - 1.0))

    # ------------------------------------------------------------------
    def _check_rejection(self, N, X):
        """Returns True if the accumulated evidence (N, X) triggers approval."""
        alpha = self.alpha_0 + X
        beta  = self.beta_0  + N - X
        lhs   = (alpha - self.alpha_0) - (alpha + beta - self.alpha_0 - self.beta_0) * self.log_term
        return lhs >= self.log_kappa_inv

    # ------------------------------------------------------------------
    def evaluate(self, track_beliefs=True):
        """Run Monte Carlo evaluation of the policy under theta_star.

        If ``track_beliefs=True``, the returned dictionary additionally contains
        ``belief_paths``: a list of per-episode belief trajectories, each entry
        of the form ``[(alpha_0, beta_0), (alpha_1, beta_1), ...]``. The list is
        capped at the first 20000 episodes to bound memory usage.
        """
        rng = np.random.default_rng(self.seed)

        n_approved         = 0
        sum_cost_approved  = 0.0
        sum_agent_utility  = 0.0

        belief_paths = [] if track_beliefs else None

        for _ in range(self.n_episodes):
            N, X   = 0, 0
            C      = 0.0
            approved = False

            if track_beliefs:
                path = []
                # initial belief
                path.append((self.alpha_0, self.beta_0))

            for l in range(self.T + 1):
                n_star = int(self.policy[l][N, X])

                if n_star == 0:
                    break  # agent opts out

                # --- real-world outcome ---
                cost_step = self.c0 + self.c1 * n_star
                C        += cost_step
                x         = int(rng.binomial(n_star, self.theta_star))
                N        += n_star
                X        += x

                # update belief
                if track_beliefs:
                    alpha = self.alpha_0 + X
                    beta  = self.beta_0 + N - X
                    path.append((alpha, beta))

                if self._check_rejection(N, X):
                    approved = True
                    break

            # Cap the number of stored trajectories to bound memory usage.
            if track_beliefs and len(belief_paths) < 20000:
                belief_paths.append(path)

            # --- accumulate statistics ---
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
        description='Evaluate a pre-computed MDP policy under the real efficacy theta_star.'
    )
    parser.add_argument('--config',      type=str, required=True,
                        help='Path to YAML config file (must include theta_star).')
    parser.add_argument('--policy_path', type=str, required=True,
                        help='Path to the .pt file produced by MDP_solver.py.')
    parser.add_argument('--n_episodes',  type=int, default=200_000,
                        help='Number of Monte Carlo trajectories (default: 200000).')
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    evaluator = PolicyEvaluator(
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

    print(f"\nEvaluating policy under theta* = {config['theta_star']} "
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
