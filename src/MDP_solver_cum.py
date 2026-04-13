import os
import torch
import numpy as np
import yaml
import argparse
import time


class SubsidizedMDPSolverCum:
    """
    Identical backward induction to SubsidizedMDPSolver, but saves results
    indexed by accumulated cost C in addition to the standard (l, N, X) format.

    For each achievable C = l*c0 + N*c1 and each step l, a full 2D (alpha, beta)
    grid is built.  Only states on the anti-diagonal
        alpha + beta = alpha_0 + beta_0 + N,  N = (C - l*c0) / c1
    are compatible with that C; all other cells are NaN.

    This lets the user plot V(alpha, beta | C, l) as a 2D heatmap for a fixed C
    and verify the monotonicity claimed by Proposition 5.
    """

    def __init__(self, rho_A, T, n_max, c0, c1, epsilon, kappa, theta_b,
                 alpha_0, beta_0, device='cuda'):
        self.rho_A   = rho_A
        self.T       = T
        self.n_max   = n_max
        self.c0      = c0
        self.c1      = c1
        self.epsilon = epsilon
        self.kappa   = kappa
        self.theta_b = theta_b
        self.alpha_0 = alpha_0
        self.beta_0  = beta_0
        self.device  = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.log_kappa_inv = torch.tensor(np.log(1 / kappa),                   device=self.device)
        self.log_term      = torch.tensor(np.log(1 + theta_b * (np.e - 1)),    device=self.device)

    # ------------------------------------------------------------------ helpers
    def beta_binomial_pmf(self, x, n, alpha, beta):
        log_pmf = (
            torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1)
            + torch.lgamma(x + alpha) + torch.lgamma(n - x + beta)
            - torch.lgamma(n + alpha + beta)
            + torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        )
        return torch.exp(log_pmf)

    def check_rejection(self, N, X):
        alpha = self.alpha_0 + X
        beta  = self.beta_0  + N - X
        lhs   = (alpha - self.alpha_0) - (alpha + beta - self.alpha_0 - self.beta_0) * self.log_term
        return lhs >= self.log_kappa_inv

    # ---------------------------------------------------------- backward induction
    def solve(self, save_dir='./mdp_results_cum'):
        os.makedirs(save_dir, exist_ok=True)

        max_N_global = self.T * self.n_max

        # Standard (l, N, X)-indexed results — identical to MDP_solver.py
        results = {'V_eps': {}, 'V_0': {}, 'A': {}, 'Policy': {}}

        V_eps_next = torch.zeros((max_N_global + 1, max_N_global + 1), device=self.device)
        V_0_next   = torch.zeros((max_N_global + 1, max_N_global + 1), device=self.device)
        A_next     = torch.zeros((max_N_global + 1, max_N_global + 1), device=self.device)

        actions          = torch.arange(1, self.n_max + 1, device=self.device)
        costs_per_action = self.c0 + self.c1 * actions

        for l in range(self.T, -1, -1):
            max_N_t = l * self.n_max

            n_range = torch.arange(max_N_t + 1, device=self.device)
            N_grid  = n_range.view(-1, 1).expand(max_N_t + 1, max_N_t + 1)
            X_grid  = n_range.view(1, -1).expand(max_N_t + 1, max_N_t + 1)

            valid_mask    = (X_grid <= N_grid)
            rejected_mask = self.check_rejection(N_grid, X_grid)
            active_mask   = valid_mask & (~rejected_mask)

            V_eps_curr  = torch.zeros_like(N_grid, dtype=torch.float32)
            V_0_curr    = torch.zeros_like(N_grid, dtype=torch.float32)
            A_curr      = torch.zeros_like(N_grid, dtype=torch.float32)
            Policy_curr = torch.zeros_like(N_grid, dtype=torch.int32)

            if active_mask.any():
                N_active = N_grid[active_mask]
                X_active = X_grid[active_mask]
                K = N_active.size(0)

                N_exp = N_active.unsqueeze(1).expand(-1, self.n_max)
                X_exp = X_active.unsqueeze(1).expand(-1, self.n_max)
                A_exp = actions.unsqueeze(0).expand(K, -1)

                alpha_active = (self.alpha_0 + X_exp).unsqueeze(-1)
                beta_active  = (self.beta_0  + N_exp - X_exp).unsqueeze(-1)

                x_outcomes   = torch.arange(self.n_max + 1, device=self.device).view(1, 1, -1)
                A_ext        = A_exp.unsqueeze(-1).expand(-1, -1, self.n_max + 1)
                valid_x_mask = x_outcomes <= A_ext

                pmf = torch.zeros_like(A_ext, dtype=torch.float32)
                pmf[valid_x_mask] = self.beta_binomial_pmf(
                    x_outcomes.expand_as(A_ext)[valid_x_mask],
                    A_ext[valid_x_mask],
                    alpha_active.expand_as(A_ext)[valid_x_mask],
                    beta_active.expand_as(A_ext)[valid_x_mask],
                )

                next_N_true  = (N_exp.unsqueeze(-1) + A_ext).long()
                next_X_true  = (X_exp.unsqueeze(-1) + x_outcomes).long()
                next_rejected = self.check_rejection(next_N_true, next_X_true)

                next_N = next_N_true.clamp(0, max_N_global)
                next_X = next_X_true.clamp(0, max_N_global)

                current_accum_cost = (l * self.c0 + N_active * self.c1).unsqueeze(1)
                total_cost_at_next = current_accum_cost.unsqueeze(-1) + (self.c0 + self.c1 * A_ext)

                v_eps_fut = V_eps_next[next_N, next_X]
                v_0_fut   = V_0_next[next_N, next_X]
                v_a_fut   = A_next[next_N, next_X]

                v_eps_fut = torch.where(next_rejected, self.rho_A + self.epsilon * total_cost_at_next, v_eps_fut)
                v_0_fut   = torch.where(next_rejected, torch.tensor(self.rho_A, device=self.device), v_0_fut)
                v_a_fut   = torch.where(next_rejected, total_cost_at_next, v_a_fut)

                E_eps = torch.sum(pmf * v_eps_fut, dim=-1)
                E_v0  = torch.sum(pmf * v_0_fut,  dim=-1)
                E_a   = torch.sum(pmf * v_a_fut,  dim=-1)

                Q_eps = -costs_per_action.unsqueeze(0) + E_eps
                best_q, best_idx = torch.max(Q_eps, dim=1)
                opt_out = best_q <= 1e-15

                V_eps_curr[active_mask]  = torch.where(opt_out, 0.0, best_q)
                Policy_curr[active_mask] = torch.where(opt_out, 0, actions[best_idx].int())

                best_cost = costs_per_action[best_idx]
                best_E_v0 = E_v0.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                best_E_a  = E_a.gather(1, best_idx.unsqueeze(1)).squeeze(1)

                V_0_curr[active_mask] = torch.where(opt_out, 0.0, -best_cost + best_E_v0)
                A_curr[active_mask]   = torch.where(opt_out, 0.0, best_E_a)

            results['V_eps'][l]  = V_eps_curr.cpu().numpy()
            results['V_0'][l]    = V_0_curr.cpu().numpy()
            results['A'][l]      = A_curr.cpu().numpy()
            results['Policy'][l] = Policy_curr.cpu().numpy()

            V_eps_next.zero_(); V_0_next.zero_(); A_next.zero_()
            V_eps_next[:max_N_t+1, :max_N_t+1] = V_eps_curr
            V_0_next[:max_N_t+1,   :max_N_t+1] = V_0_curr
            A_next[:max_N_t+1,     :max_N_t+1] = A_curr

        results['params'] = {
            'alpha_0': self.alpha_0, 'beta_0': self.beta_0,
            'kappa':   self.kappa,   'theta_b': self.theta_b,
            'rho_A':   self.rho_A,   'epsilon': self.epsilon,
            'c0': self.c0, 'c1': self.c1, 'T': self.T, 'n_max': self.n_max,
        }

        # ----------------------------------------------------------------
        # Build C-indexed 2D (alpha, beta) grids
        # ----------------------------------------------------------------
        # For each step l and each achievable N at that step, C = l*c0 + N*c1.
        # We build a 2D grid of shape (dim, dim) where dim = l*n_max + 1, with
        # axes:  row index a_idx = alpha - alpha_0  (0 .. l*n_max)
        #        col index b_idx = beta  - beta_0   (0 .. l*n_max)
        # Only cells on the anti-diagonal a_idx + b_idx == N carry values;
        # all other cells are NaN.  This is intentional: for a fixed C the
        # compatible (alpha, beta) pairs ARE exactly that anti-diagonal, and
        # the user can plot the full grid without caring about NaN cells.
        #
        # Storage layout:
        #   results['V_by_C'][l]  — dict keyed by C (float, rounded to 10 dp)
        #       -> {'C': float, 'N': int,
        #           'V_eps': ndarray (dim, dim),
        #           'V_0':   ndarray (dim, dim),
        #           'Policy':ndarray (dim, dim)}
        # ----------------------------------------------------------------

        print("Building C-indexed grids …")
        V_by_C = {}

        for l in range(self.T + 1):
            max_N_l = l * self.n_max
            dim     = max_N_l + 1          # grid side length

            V_eps_l  = results['V_eps'][l]
            V_0_l    = results['V_0'][l]
            Pol_l    = results['Policy'][l]

            V_by_C[l] = {}

            for N in range(max_N_l + 1):
                C     = round(float(l * self.c0 + N * self.c1), 10)
                C_key = C

                # Full 2D grids — NaN everywhere, values on anti-diagonal
                g_eps = np.full((dim, dim), np.nan, dtype=np.float32)
                g_v0  = np.full((dim, dim), np.nan, dtype=np.float32)
                g_pol = np.full((dim, dim), np.nan, dtype=np.float32)

                # Anti-diagonal: a_idx + b_idx == N
                # a_idx = X  (successes, 0..N),  b_idx = N - X  (failures)
                for X in range(N + 1):
                    a_idx = X
                    b_idx = N - X
                    g_eps[a_idx, b_idx] = V_eps_l[N, X]
                    g_v0 [a_idx, b_idx] = V_0_l[N,  X]
                    g_pol[a_idx, b_idx] = Pol_l[N,   X]

                V_by_C[l][C_key] = {
                    'C':      C,
                    'N':      N,
                    'V_eps':  g_eps,
                    'V_0':    g_v0,
                    'Policy': g_pol,
                }

        results['V_by_C'] = V_by_C

        # ----------------------------------------------------------------
        # Also expose the sorted list of achievable C values per l for
        # easy lookup in the notebook.
        # ----------------------------------------------------------------
        results['C_values'] = {
            l: sorted(V_by_C[l].keys())
            for l in range(self.T + 1)
        }

        fname = f'mdp_results_cum_eps_{self.epsilon:.3f}_T_{self.T}_nmax_{self.n_max}.pt'
        torch.save(results, os.path.join(save_dir, fname))
        print(f"Saved → {fname}")
        print(f"V^eps(l=0, N=0, X=0): {results['V_eps'][0][0, 0]:.4f}")
        print(f"Achievable C values at l=1: {len(results['C_values'].get(1, []))} levels")
        return results


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Subsidized MDP Solver (C-indexed output) with YAML config"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    solver_keys = ["rho_A", "T", "n_max", "c0", "c1",
                   "epsilon", "kappa", "theta_b", "alpha_0", "beta_0", "device"]
    solver = SubsidizedMDPSolverCum(**{k: config[k] for k in solver_keys})

    save_dir = config.get("save_dir", "./mdp_output")

    start = time.time()
    solver.solve(save_dir=save_dir)
    print(f"Total execution time: {time.time() - start:.2f} s")
