import os
import torch
import numpy as np
import yaml
import argparse
import time, psutil

class SubsidizedMDPSolver:
    def __init__(self, rho_A, T, n_max, c0, c1, epsilon, kappa, theta_b, alpha_0, beta_0, device='cuda'):
        """
        Initializes the Subsidized MDP Solver with parameters from the paper.
        """
        self.rho_A = rho_A
        self.T = T
        self.n_max = n_max
        self.c0 = c0
        self.c1 = c1
        self.epsilon = epsilon
        self.kappa = kappa
        self.theta_b = theta_b
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        #print device being used
        print(f"Using device: {self.device}")
        
        # Precompute constants for the rejection condition (Eq 30)
        self.log_kappa_inv = torch.tensor(np.log(1 / kappa), device=self.device)
        self.log_term = torch.tensor(np.log(1 + theta_b * (np.e - 1)), device=self.device)
        
    def beta_binomial_pmf(self, x, n, alpha, beta):
        """
        Computes the Beta-Binomial PMF efficiently using log-gamma functions.
        x: successes, n: trials, alpha/beta: belief parameters.
        All inputs can be broadcastable PyTorch tensors.
        """
        log_pmf = (
            torch.lgamma(n + 1) - torch.lgamma(x + 1) - torch.lgamma(n - x + 1) +
            torch.lgamma(x + alpha) + torch.lgamma(n - x + beta) - torch.lgamma(n + alpha + beta) +
            torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        )
        return torch.exp(log_pmf)

    def check_rejection(self, N, X):
        """
        Checks if f(alpha, beta) >= 1/kappa directly using the linear condition (Eq 30).
        """
        alpha = self.alpha_0 + X
        beta = self.beta_0 + N - X
        
        # Condition: alpha - alpha_0 - (alpha + beta - alpha_0 - beta_0) * log(...) >= log(1/kappa)
        lhs = (alpha - self.alpha_0) - (alpha + beta - self.alpha_0 - self.beta_0) * self.log_term
        return lhs >= self.log_kappa_inv

    def solve(self, save_dir='./mdp_results', chunk_size=512):
        """
        Executes Algorithm 2 (Backward Induction) using dense tensor vectorization.
        Tracks V^epsilon, V^0, and A components exactly per Proposition 7.
        Active states are processed in chunks to avoid MAX_INT-sized tensors at
        large n_max (the pmf tensor is (K, n_max, n_max+1); chunking bounds K).
        """
        os.makedirs(save_dir, exist_ok=True)

        # Absolute maximum trials N possible across the horizon
        max_N_global = self.T * self.n_max

        results = {'V_eps': {}, 'V_0': {}, 'A': {}, 'P_approval': {}, 'Policy': {}}

        # Initialize T+1 step arrays at global maximum size to prevent indexing errors
        V_eps_next = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)
        V_0_next   = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)
        A_next     = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)
        P_next     = torch.zeros((max_N_global + 1, max_N_global + 1), dtype=torch.float32, device=self.device)

        actions = torch.arange(1, self.n_max + 1, device=self.device)
        # Precompute cost for each action n in {1, ..., n_max}
        costs_per_action = self.c0 + self.c1 * actions # Shape: (n_max,)
        # Precompute x_outcomes once: shape (1, 1, n_max+1)
        x_outcomes = torch.arange(self.n_max + 1, dtype=torch.float32, device=self.device).view(1, 1, -1)

        for l in range(self.T, -1, -1):
            max_N_t = l * self.n_max

            # Create grids for the current time step l
            n_range = torch.arange(max_N_t + 1, device=self.device)
            N_grid = n_range.view(-1, 1).expand(max_N_t + 1, max_N_t + 1)
            X_grid = n_range.view(1, -1).expand(max_N_t + 1, max_N_t + 1)

            valid_mask = (X_grid <= N_grid)
            rejected_mask = self.check_rejection(N_grid, X_grid)
            active_mask = valid_mask & (~rejected_mask)

            V_eps_curr = torch.zeros(max_N_t + 1, max_N_t + 1, dtype=torch.float32, device=self.device)
            V_0_curr   = torch.zeros(max_N_t + 1, max_N_t + 1, dtype=torch.float32, device=self.device)
            A_curr     = torch.zeros(max_N_t + 1, max_N_t + 1, dtype=torch.float32, device=self.device)
            P_curr     = torch.zeros(max_N_t + 1, max_N_t + 1, dtype=torch.float32, device=self.device)
            Policy_curr = torch.zeros(max_N_t + 1, max_N_t + 1, dtype=torch.int32, device=self.device)

            if active_mask.any():
                N_active = N_grid[active_mask]  # (K,)
                X_active = X_grid[active_mask]  # (K,)
                K = N_active.size(0)

                # Accumulators for the full K dimension
                V_eps_vals  = torch.zeros(K, dtype=torch.float32, device=self.device)
                V_0_vals    = torch.zeros(K, dtype=torch.float32, device=self.device)
                A_vals      = torch.zeros(K, dtype=torch.float32, device=self.device)
                P_vals      = torch.zeros(K, dtype=torch.float32, device=self.device)
                Policy_vals = torch.zeros(K, dtype=torch.int32,   device=self.device)

                for cs in range(0, K, chunk_size):
                    ce = min(cs + chunk_size, K)
                    N_chunk = N_active[cs:ce]  # (k,)
                    X_chunk = X_active[cs:ce]  # (k,)
                    k = ce - cs

                    # Expand states for all actions: (k, n_max)
                    N_exp = N_chunk.unsqueeze(1).expand(-1, self.n_max)
                    X_exp = X_chunk.unsqueeze(1).expand(-1, self.n_max)
                    A_exp = actions.unsqueeze(0).expand(k, -1)

                    # Belief parameters: (k, n_max, 1)
                    alpha_ch = (self.alpha_0 + X_exp).unsqueeze(-1)
                    beta_ch  = (self.beta_0 + N_exp - X_exp).unsqueeze(-1)

                    # (k, n_max, n_max+1)
                    A_ext = A_exp.unsqueeze(-1).expand(-1, -1, self.n_max + 1)
                    valid_x_mask = x_outcomes <= A_ext

                    # Vectorized Beta-Binomial PMF
                    pmf = torch.zeros(k, self.n_max, self.n_max + 1, dtype=torch.float32, device=self.device)
                    pmf[valid_x_mask] = self.beta_binomial_pmf(
                        x_outcomes.expand(k, self.n_max, -1)[valid_x_mask],
                        A_ext[valid_x_mask],
                        alpha_ch.expand(k, self.n_max, self.n_max + 1)[valid_x_mask],
                        beta_ch.expand(k, self.n_max, self.n_max + 1)[valid_x_mask],
                    )

                    # Next states (true values, used for rejection check)
                    next_N_true = (N_exp.unsqueeze(-1) + A_ext).long()
                    next_X_true = (X_exp.unsqueeze(-1) + x_outcomes.long()).long()

                    # Check rejection on true next state
                    next_rejected = self.check_rejection(next_N_true, next_X_true)

                    # Clamped indices for value-table lookup
                    next_N = next_N_true.clamp(0, max_N_global)
                    next_X = next_X_true.clamp(0, max_N_global)

                    # Accumulated cost
                    current_accum_cost = (l * self.c0 + N_chunk * self.c1).unsqueeze(1)
                    total_cost_at_next = current_accum_cost.unsqueeze(-1) + (self.c0 + self.c1 * A_ext)

                    # Look up future values
                    v_eps_fut = V_eps_next[next_N, next_X]
                    v_0_fut   = V_0_next[next_N, next_X]
                    v_a_fut   = A_next[next_N, next_X]
                    v_p_fut   = P_next[next_N, next_X]

                    # Apply boundary conditions (Approval)
                    v_eps_fut = torch.where(next_rejected, self.rho_A + self.epsilon * total_cost_at_next, v_eps_fut)
                    v_0_fut   = torch.where(next_rejected, torch.tensor(self.rho_A, dtype=torch.float32, device=self.device), v_0_fut)
                    v_a_fut   = torch.where(next_rejected, total_cost_at_next, v_a_fut)
                    v_p_fut   = torch.where(next_rejected, torch.ones_like(v_p_fut), v_p_fut)

                    # Expectations over x: (k, n_max)
                    E_eps = torch.sum(pmf * v_eps_fut, dim=-1)
                    E_v0  = torch.sum(pmf * v_0_fut,  dim=-1)
                    E_a   = torch.sum(pmf * v_a_fut,  dim=-1)
                    E_p   = torch.sum(pmf * v_p_fut,  dim=-1)

                    # Agent's Q-value: -cost(n) + E[V_next]
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

                # Scatter chunk results back to the 2-D grids
                V_eps_curr[active_mask]  = V_eps_vals
                V_0_curr[active_mask]    = V_0_vals
                A_curr[active_mask]      = A_vals
                P_curr[active_mask]      = P_vals
                Policy_curr[active_mask] = Policy_vals
            
            # Store results for this time step
            results['V_eps'][l]      = V_eps_curr.cpu().numpy()
            results['V_0'][l]        = V_0_curr.cpu().numpy()
            results['A'][l]          = A_curr.cpu().numpy()
            results['P_approval'][l] = P_curr.cpu().numpy()
            results['Policy'][l]     = Policy_curr.cpu().numpy()
            results['params'] = {
            'alpha_0': self.alpha_0,
            'beta_0': self.beta_0,
            'kappa': self.kappa,
            'theta_b': self.theta_b,
            'rho_A': self.rho_A,
            'epsilon': self.epsilon,
            'c0': self.c0,
            'c1': self.c1,
            'T': self.T,
            'n_max': self.n_max
        }
            
            # Update "next" tensors for the next iteration of backward induction
            V_eps_next.zero_()
            V_0_next.zero_()
            A_next.zero_()
            P_next.zero_()
            V_eps_next[:max_N_t+1, :max_N_t+1] = V_eps_curr
            V_0_next[:max_N_t+1, :max_N_t+1]   = V_0_curr
            A_next[:max_N_t+1, :max_N_t+1]     = A_curr
            P_next[:max_N_t+1, :max_N_t+1]     = P_curr

        # Save exact results
        torch.save(results, os.path.join(save_dir, f'mdp_results_eps_{self.epsilon:.3f}_T_{self.T}_nmax_{self.n_max}_alpha_{self.alpha_0}_beta_{self.beta_0}_thetab_{self.theta_b}.pt'))
        print(f"MDP Solved. V^eps(0,0): {results['V_eps'][0][0,0]:.4f}, P_approval(0,0): {results['P_approval'][0][0,0]:.4f}")
        return results

# --- Execution Setup ---
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run Subsidized MDP Solver with YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load YAML config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Extract solver args
    solver_keys = [
        "rho_A","T","n_max","c0","c1",
        "epsilon","kappa","theta_b",
        "alpha_0","beta_0","device"
    ]
    solver_config = {k: config[k] for k in solver_keys}

    # print("CPU memory (MB):", psutil.Process(os.getpid()).memory_info().rss / 1024**2, flush=True)
    # if torch.cuda.is_available():
    #     print("GPU is available. Initial GPU memory usage (MB):", torch.cuda.memory_allocated() / 1024**2)
    #     print("GPU memory reserved (MB):", torch.cuda.memory_reserved() / 1024**2)
        
    #print parameters being used
    print("Running MDP Solver with parameters:")
    for key, value in solver_config.items():
        print(f"  {key}: {value}")
    # Create solver
    solver = SubsidizedMDPSolver(**solver_config)

    # Use the save_dir from YAML
    save_dir = config.get("save_dir")  # None if missing
    print(f"Results will be saved to: {save_dir}")
    if save_dir is None:
        save_dir = "./mdp_output"  # fallback default

    # --- Measure execution time ---
    start_time = time.time()           # record start
    res = solver.solve(save_dir=save_dir)
    end_time = time.time()             # record end

    elapsed = end_time - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")