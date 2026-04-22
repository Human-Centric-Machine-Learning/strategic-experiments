"""
Shared helpers for the mixture e-value E_mix rejection test.

    E_mix(N, X) = [int_{theta_b}^1 theta^X (1 - theta)^(N - X) d theta]
                  / [(1 - theta_b) * theta_b^X * (1 - theta_b)^(N - X)]

The null is rejected as soon as E_mix >= 1 / kappa.

Used by deploy_policy_mix, sensitivity_analysis_mix, and the notebooks so that
a single numpy lookup table encodes the E_mix rejection rule consistently.
"""

import numpy as np
from scipy.special import betaln, betaincc


def build_rejection_table_np(max_N: int, kappa: float, theta_b: float) -> np.ndarray:
    """Boolean (max_N+1) x (max_N+1) array reject[N, X] = (E_mix(N, X) >= 1/kappa).

    Cells with X > N (invalid) are set to False. Log-space throughout.
    """
    log_kappa_inv = np.log(1.0 / kappa)
    tb            = float(theta_b)
    log_tb        = np.log(tb)
    log_1mtb      = np.log(1.0 - tb)

    N_idx, X_idx = np.meshgrid(
        np.arange(max_N + 1), np.arange(max_N + 1), indexing='ij'
    )
    valid = X_idx <= N_idx

    a = (X_idx + 1).astype(np.float64)
    b = (N_idx - X_idx + 1).astype(np.float64)
    b_safe = np.where(valid, b, 1.0)

    bicc = betaincc(a, b_safe, tb)
    with np.errstate(divide='ignore'):
        log_bicc = np.log(np.clip(bicc, 1e-300, None))
    log_integral = betaln(a, b_safe) + log_bicc

    log_denom = log_1mtb + X_idx * log_tb + (N_idx - X_idx) * log_1mtb
    log_E_mix = log_integral - log_denom

    return (log_E_mix >= log_kappa_inv) & valid


def log_E_mix(N: np.ndarray, X: np.ndarray, theta_b: float) -> np.ndarray:
    """Log of the mixture e-value E_mix(N, X); elementwise over numpy arrays."""
    N = np.asarray(N, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    tb       = float(theta_b)
    log_tb   = np.log(tb)
    log_1mtb = np.log(1.0 - tb)

    a = X + 1.0
    b = N - X + 1.0

    bicc = betaincc(a, b, tb)
    with np.errstate(divide='ignore'):
        log_bicc = np.log(np.clip(bicc, 1e-300, None))
    log_integral = betaln(a, b) + log_bicc

    log_denom = log_1mtb + X * log_tb + (N - X) * log_1mtb
    return log_integral - log_denom
