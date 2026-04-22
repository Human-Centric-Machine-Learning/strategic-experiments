"""
Subsidized MDP solver using the mixture e-value E_mix (uniformly powerful variant).

Identical to MDP_solver.py except the rejection test uses the method-of-mixtures
e-value with uniform mixing over [theta_b, 1]:

    E_mix(N, X) = [int_{theta_b}^1 theta^X (1-theta)^(N-X) d theta]
                  / [(1 - theta_b) * theta_b^X * (1 - theta_b)^(N-X)]

Unlike the log-linear e-value of Eq. 5 (Proposition 1), E_mix is uniformly
powerful: for any theta* > theta_b, E_mix(N, X) -> infinity as N grows.

Why the state (N, X) still suffices. For any fixed theta, the likelihood ratio
L_t(theta) = (theta/theta_b)^{X_t} * ((1-theta)/(1-theta_b))^{N_t - X_t}
is a P_{theta_b}-martingale, so its uniform mixture
M_t = int_{theta_b}^1 L_t(theta) / (1-theta_b) d theta = E_mix(N_t, X_t)
is also a non-negative martingale starting at 1. By Ville's inequality,
P_{theta_b}(exists t: M_t >= 1/kappa) <= kappa, giving an anytime-valid test
at level kappa. Because M_t depends only on cumulative (N_t, X_t), the MDP
state and backward-induction structure of Algorithm 2 are unchanged.
"""

import os
import time
import argparse

import numpy as np
import torch
import yaml
from scipy.special import betaln, betaincc


class SubsidizedMDPSolverMix:
    """Vectorized backward-induction solver for the subsidized sequential RCT
    MDP using the mixture e-value E_mix for the rejection test."""

    def __init__(self, rho_A, T, n_max, c0, c1, epsilon, kappa, theta_b,
                 alpha_0, beta_0, device='cuda', action_stride=3):
        self.rho_A         = rho_A
        self.T             = T
        self.n_max         = n_max
        self.c0            = c0
        self.c1            = c1
        self.epsilon       = epsilon
        self.kappa         = kappa
        self.theta_b       = theta_b
        self.alpha_0       = alpha_0
        self.beta_0        = beta_0
        self.action_stride = int(action_stride)
        if self.action_stride < 1:
            raise ValueError(f"action_stride must be >= 1 (got {action_stride})")
        self.device  = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}  (action_stride={self.action_stride})")

        # Precompute a boolean (max_N+1) x (max_N+1) table of whether E_mix >= 1/kappa.
        # scipy's special functions run on CPU once; the table is then moved to device
        # and indexed via integer (N, X) during backward induction.
        #
        # Size: the backward induction at layer l queries check_rejection at
        # next_N = N + n with N <= l*n_max and n <= n_max, so next_N can reach
        # (l+1)*n_max. At l = T (terminal decision) this is (T+1)*n_max, which
        # exceeds T*n_max. Sizing the table at T*n_max would silently clamp
        # those lookups and return *wrong* rejection answers at terminal states,
        # biasing V^eps upward. The table must cover all reachable next states.
        self.rejection_table = self._build_rejection_table((self.T + 1) * self.n_max)

    # ------------------------------------------------------------------ helpers
    def _build_rejection_table(self, max_N):
        """Precompute reject[N, X] = (E_mix(N, X) >= 1/kappa) for all 0 <= X <= N <= max_N."""
        log_kappa_inv = np.log(1.0 / self.kappa)
        tb            = self.theta_b
        log_tb        = np.log(tb)
        log_1mtb      = np.log(1.0 - tb)

        # Full (N, X) mesh; invalid entries (X > N) are masked to False at the end.
        N_idx, X_idx = np.meshgrid(
            np.arange(max_N + 1), np.arange(max_N + 1), indexing='ij'
        )
        valid = X_idx <= N_idx

        # a = X + 1, b = N - X + 1 are the Beta parameters of the mixture integrand.
        a = (X_idx + 1).astype(np.float64)
        b = (N_idx - X_idx + 1).astype(np.float64)
        # Clamp b >= 1 in invalid cells so betaincc does not receive b <= 0.
        b_safe = np.where(valid, b, 1.0)

        # log integral = log B(a, b) + log(1 - I_{tb}(a, b))
        # betaincc(a, b, x) = 1 - regularized_incomplete_beta(a, b, x) [scipy >= 1.11]
        bicc = betaincc(a, b_safe, tb)
        with np.errstate(divide='ignore'):
            log_bicc = np.log(np.clip(bicc, 1e-300, None))
        log_integral = betaln(a, b_safe) + log_bicc

        log_denom = log_1mtb + X_idx * log_tb + (N_idx - X_idx) * log_1mtb
        log_E_mix = log_integral - log_denom

        reject = (log_E_mix >= log_kappa_inv) & valid
        return torch.from_numpy(reject).to(self.device)

    def beta_binomial_pmf(self, x, n, alpha, beta):
        """Beta-Binomial PMF computed via log-gamma for numerical stability."""
        log_pmf = (
            torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1) +
            torch.lgamma(x + alpha) + torch.lgamma(n - x + beta) - torch.lgamma(n + alpha + beta) +
            torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        )
        return torch.exp(log_pmf)

    def check_rejection(self, N, X):
        """Look up precomputed E_mix >= 1/kappa at integer (N, X) states.

        The table is sized at (T+1)*n_max to cover every reachable next state
        across the horizon, so the clamp is a no-op in correct use and exists
        only to defend against out-of-range indices.
        """
        max_idx = self.rejection_table.shape[0] - 1
        Nc = N.long().clamp(0, max_idx)
        Xc = X.long().clamp(0, max_idx)
        return self.rejection_table[Nc, Xc]

    # ------------------------------------------------------------------ solver
    def solve(self, save_dir='./mdp_results', chunk_size=4096):
        """Run Algorithm 2: vectorized backward induction over (N, X) states.

        Identical to MDP_solver.SubsidizedMDPSolver.solve — the only difference
        is that self.check_rejection now queries the E_mix table.

        Exact state-space pruning: with action_stride s, every reachable N is a
        multiple of s (since N is a sum of multiples of s), so we iterate the
        N-axis over {0, s, 2s, ..., l*n_max} instead of {0, 1, ..., l*n_max}.
        This gives an additional ~s× speedup over action-stride alone, for a
        total ~s²× speedup vs the dense solver. V_*_next tensors stay full-
        dense so the inner-loop reads at next_N = N + n (also a multiple of s)
        are unchanged; only the row assignments on roll-over use fancy indexing.
        Emitted results (Policy[l], V_eps[l], etc.) remain full-dense 2-D arrays
        so downstream code that indexes Policy[l][N, X] keeps working.
        """
        os.makedirs(save_dir, exist_ok=True)

        max_N_global = self.T * self.n_max

        results = {'V_eps': {}, 'V_0': {}, 'A': {}, 'P_approval': {}, 'Policy': {}}

        V_eps_next = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)
        V_0_next   = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)
        A_next     = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)
        P_next     = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)

        # Non-uniform action grid: multiples of `action_stride` in
        # {stride, 2*stride, ..., <= n_max}. Opt-out (n=0) is handled separately
        # via the opt_out flag below, so it need not appear here. The action
        # axis size is ~n_max/stride; the marginal value of n vs n+1 is small
        # at large n, so the utility loss vs the dense {1,...,n_max} grid is
        # negligible in practice and gives a ~stride-fold inner-loop speedup.
        stride           = self.action_stride
        actions          = torch.arange(stride, self.n_max + 1, stride,
                                        device=self.device)
        num_actions      = actions.size(0)
        costs_per_action = self.c0 + self.c1 * actions
        x_outcomes       = torch.arange(self.n_max + 1, dtype=torch.float32,
                                        device=self.device).view(1, 1, -1)

        for l in range(self.T, -1, -1):
            max_N_t = l * self.n_max

            # State grid: N restricted to multiples of `stride` in [0, max_N_t].
            # X is unrestricted in [0, max_N_t].
            n_range_N = torch.arange(0, max_N_t + 1, stride, device=self.device)
            n_range_X = torch.arange(max_N_t + 1, device=self.device)
            num_N     = n_range_N.size(0)
            num_X     = n_range_X.size(0)

            N_grid = n_range_N.view(-1, 1).expand(num_N, num_X)
            X_grid = n_range_X.view(1, -1).expand(num_N, num_X)

            valid_mask    = (X_grid <= N_grid)
            rejected_mask = self.check_rejection(N_grid, X_grid)
            active_mask   = valid_mask & (~rejected_mask)

            V_eps_curr  = torch.zeros(num_N, num_X, dtype=torch.float32, device=self.device)
            V_0_curr    = torch.zeros(num_N, num_X, dtype=torch.float32, device=self.device)
            A_curr      = torch.zeros(num_N, num_X, dtype=torch.float32, device=self.device)
            P_curr      = torch.zeros(num_N, num_X, dtype=torch.float32, device=self.device)
            Policy_curr = torch.zeros(num_N, num_X, dtype=torch.int32,   device=self.device)

            if active_mask.any():
                N_active = N_grid[active_mask]
                X_active = X_grid[active_mask]
                K = N_active.size(0)

                V_eps_vals  = torch.zeros(K, dtype=torch.float32, device=self.device)
                V_0_vals    = torch.zeros(K, dtype=torch.float32, device=self.device)
                A_vals      = torch.zeros(K, dtype=torch.float32, device=self.device)
                P_vals      = torch.zeros(K, dtype=torch.float32, device=self.device)
                Policy_vals = torch.zeros(K, dtype=torch.int32,   device=self.device)

                for cs in range(0, K, chunk_size):
                    ce = min(cs + chunk_size, K)
                    N_chunk = N_active[cs:ce]
                    X_chunk = X_active[cs:ce]
                    k = ce - cs

                    N_exp = N_chunk.unsqueeze(1).expand(-1, num_actions)
                    X_exp = X_chunk.unsqueeze(1).expand(-1, num_actions)
                    A_exp = actions.unsqueeze(0).expand(k, -1)

                    alpha_ch = (self.alpha_0 + X_exp).unsqueeze(-1)
                    beta_ch  = (self.beta_0 + N_exp - X_exp).unsqueeze(-1)

                    A_ext        = A_exp.unsqueeze(-1).expand(-1, -1, self.n_max + 1)
                    valid_x_mask = x_outcomes <= A_ext

                    pmf = torch.zeros(k, num_actions, self.n_max + 1,
                                      dtype=torch.float32, device=self.device)
                    pmf[valid_x_mask] = self.beta_binomial_pmf(
                        x_outcomes.expand(k, num_actions, -1)[valid_x_mask],
                        A_ext[valid_x_mask],
                        alpha_ch.expand(k, num_actions, self.n_max + 1)[valid_x_mask],
                        beta_ch.expand(k, num_actions, self.n_max + 1)[valid_x_mask],
                    )

                    next_N_true  = (N_exp.unsqueeze(-1) + A_ext).long()
                    next_X_true  = (X_exp.unsqueeze(-1) + x_outcomes.long()).long()
                    next_rejected = self.check_rejection(next_N_true, next_X_true)

                    next_N = next_N_true.clamp(0, max_N_global)
                    next_X = next_X_true.clamp(0, max_N_global)

                    current_accum_cost = (l * self.c0 + N_chunk * self.c1).unsqueeze(1)
                    total_cost_at_next = current_accum_cost.unsqueeze(-1) + (self.c0 + self.c1 * A_ext)

                    v_eps_fut = V_eps_next[next_N, next_X]
                    v_0_fut   = V_0_next[next_N, next_X]
                    v_a_fut   = A_next[next_N, next_X]
                    v_p_fut   = P_next[next_N, next_X]

                    v_eps_fut = torch.where(next_rejected, self.rho_A + self.epsilon * total_cost_at_next, v_eps_fut)
                    v_0_fut   = torch.where(next_rejected, torch.tensor(self.rho_A, dtype=torch.float32, device=self.device), v_0_fut)
                    v_a_fut   = torch.where(next_rejected, total_cost_at_next, v_a_fut)
                    v_p_fut   = torch.where(next_rejected, torch.ones_like(v_p_fut), v_p_fut)

                    E_eps = torch.sum(pmf * v_eps_fut, dim=-1)
                    E_v0  = torch.sum(pmf * v_0_fut,  dim=-1)
                    E_a   = torch.sum(pmf * v_a_fut,  dim=-1)
                    E_p   = torch.sum(pmf * v_p_fut,  dim=-1)

                    Q_eps = -costs_per_action.unsqueeze(0) + E_eps

                    best_q, best_idx = torch.max(Q_eps, dim=1)
                    opt_out = best_q <= 1e-15

                    V_eps_vals[cs:ce]  = torch.where(opt_out, 0.0, best_q)
                    Policy_vals[cs:ce] = torch.where(opt_out, 0, actions[best_idx].int())

                    best_E_v0 = E_v0.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                    best_E_a  = E_a.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                    best_E_p  = E_p.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                    best_cost = costs_per_action[best_idx]

                    V_0_vals[cs:ce] = torch.where(opt_out, 0.0, -best_cost + best_E_v0)
                    A_vals[cs:ce]   = torch.where(opt_out, 0.0, best_E_a)
                    P_vals[cs:ce]   = torch.where(opt_out, 0.0, best_E_p)

                V_eps_curr[active_mask]  = V_eps_vals
                V_0_curr[active_mask]    = V_0_vals
                A_curr[active_mask]      = A_vals
                P_curr[active_mask]      = P_vals
                Policy_curr[active_mask] = Policy_vals

            # Scatter compact (num_N, num_X) layer tensors into full-dense
            # (max_N_global+1, max_N_global+1) next-buffers. Unreachable rows
            # (N not a multiple of `stride`) stay zero and are never queried,
            # since the inner-loop reads at next_N = N + n with both N and n
            # multiples of stride. Fancy indexing with broadcast shapes.
            rows = n_range_N.view(-1, 1)
            cols = n_range_X.view(1, -1)

            V_eps_next.zero_()
            V_0_next.zero_()
            A_next.zero_()
            P_next.zero_()
            V_eps_next[rows, cols] = V_eps_curr
            V_0_next[rows, cols]   = V_0_curr
            A_next[rows, cols]     = A_curr
            P_next[rows, cols]     = P_curr

            # Emit dense (max_N_t+1, max_N_t+1) results for downstream code that
            # indexes Policy[l][N, X] directly (mc rollouts, deploy_policy_mix,
            # notebooks). Policy_curr is int32 so we scatter into an int array.
            dense_shape = (max_N_t + 1, max_N_t + 1)
            V_eps_dense  = torch.zeros(dense_shape, dtype=torch.float32, device=self.device)
            V_0_dense    = torch.zeros(dense_shape, dtype=torch.float32, device=self.device)
            A_dense      = torch.zeros(dense_shape, dtype=torch.float32, device=self.device)
            P_dense      = torch.zeros(dense_shape, dtype=torch.float32, device=self.device)
            Policy_dense = torch.zeros(dense_shape, dtype=torch.int32,   device=self.device)
            V_eps_dense[rows, cols]  = V_eps_curr
            V_0_dense[rows, cols]    = V_0_curr
            A_dense[rows, cols]      = A_curr
            P_dense[rows, cols]      = P_curr
            Policy_dense[rows, cols] = Policy_curr

            results['V_eps'][l]      = V_eps_dense.cpu().numpy()
            results['V_0'][l]        = V_0_dense.cpu().numpy()
            results['A'][l]          = A_dense.cpu().numpy()
            results['P_approval'][l] = P_dense.cpu().numpy()
            results['Policy'][l]     = Policy_dense.cpu().numpy()

        results['params'] = {
            'alpha_0': self.alpha_0,
            'beta_0':  self.beta_0,
            'kappa':   self.kappa,
            'theta_b': self.theta_b,
            'rho_A':   self.rho_A,
            'epsilon': self.epsilon,
            'c0':      self.c0,
            'c1':      self.c1,
            'T':             self.T,
            'n_max':         self.n_max,
            'action_stride': self.action_stride,
            'e_value':       'mixture_uniform',
        }

        out_path = os.path.join(
            save_dir,
            f'mdp_mix_results_eps_{self.epsilon:.3f}_T_{self.T}_nmax_{self.n_max}'
            f'_alpha_{self.alpha_0}_beta_{self.beta_0}_thetab_{self.theta_b}.pt'
        )
        torch.save(results, out_path)
        print(f"MDP (E_mix) solved. V^eps(0,0): {results['V_eps'][0][0,0]:.4f}, "
              f"P_approval(0,0): {results['P_approval'][0][0,0]:.4f}, "
              f"n*(0,0): {int(results['Policy'][0][0,0])}")
        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the subsidized MDP solver with the E_mix e-value (YAML config)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    solver_keys = ["rho_A", "T", "n_max", "c0", "c1",
                   "epsilon", "kappa", "theta_b",
                   "alpha_0", "beta_0", "device", "action_stride"]
    solver_config = {k: config[k] for k in solver_keys if k in config}

    print("Running MDP solver (E_mix) with parameters:")
    for key, value in solver_config.items():
        print(f"  {key}: {value}")

    solver = SubsidizedMDPSolverMix(**solver_config)

    save_dir = config.get("save_dir") or "./mdp_output_mix"
    print(f"Results will be saved to: {save_dir}")

    chunk_size = int(config.get("chunk_size", 4096))

    start_time = time.time()
    solver.solve(save_dir=save_dir, chunk_size=chunk_size)
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")
