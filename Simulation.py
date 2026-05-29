import math
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# Helpers

def _build_log_factorials(n_max: int) -> np.ndarray:
    """logfac[i] = log(i!) for i=0..n_max."""
    logfac = np.zeros(n_max + 1, dtype=np.float64)
    if n_max >= 1:
        logfac[1:] = np.cumsum(np.log(np.arange(1, n_max + 1, dtype=np.float64)))
    return logfac


def _logsumexp(logw: np.ndarray) -> float:
    m = float(np.max(logw))
    return m + float(np.log(np.sum(np.exp(logw - m))))


def _log_choose(logfac: np.ndarray, n: int, k: np.ndarray) -> np.ndarray:
    """Vectorized log(n choose k), assumes k is valid (0<=k<=n)."""
    return logfac[n] - logfac[k] - logfac[n - k]


def _fisher_nch_pmf(
    mt: int,
    N: int,
    C: int,
    omega: float,
    logfac: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fisher's noncentral hypergeometric for r:
      P(r | N, mt, C, ω) ∝ comb(mt,r) * comb(N-mt, C-r) * ω^r
    with r in [max(0, C-(N-mt)), min(mt,C)].
    """
    if C > N:
        raise ValueError("C cannot exceed N.")
    if omega <= 0:
        raise ValueError("omega must be > 0.")
    r_min = max(0, C - (N - mt))
    r_max = min(mt, C)
    rs = np.arange(r_min, r_max + 1, dtype=np.int32)
    if rs.size == 0:
        return rs, np.array([], dtype=np.float64)

    logomega = math.log(omega)
    logw = (
        _log_choose(logfac, mt, rs) +
        _log_choose(logfac, N - mt, C - rs) +
        rs * logomega
    )
    logw -= _logsumexp(logw)
    probs = np.exp(logw)
    probs /= probs.sum()  # numerical cleanup
    return rs, probs


# 1) Simulation: r values for each N

def simulate_r_values(
    N_min: int,
    K: int,
    m: int,
    C: int,
    phi: float,
    omega: float,
    Simulations: int,
    Seed: Optional[int] = None,
) -> Dict[int, np.ndarray]:
    """
    Simulate recapture counts r for each candidate N in [N_min, K].

    Model:
      Mt ~ Binomial(m, phi)
      r | (N, Mt=mt, C, omega) ~ Fisher's noncentral hypergeometric (or standard hypergeometric if omega=1)

    Returns:
      dict: {N: np.ndarray of length Simulations containing simulated r values}
    """
    # Basic validation
    if N_min > K:
        raise ValueError("N_min must be <= K.")
    if N_min < max(m, C):
        raise ValueError("For feasibility, require N_min >= max(m, C).")
    if not (0.0 <= phi <= 1.0):
        raise ValueError("phi must be in [0, 1].")
    if omega <= 0.0:
        raise ValueError("omega must be > 0.")
    if Simulations <= 0:
        raise ValueError("Simulations must be positive.")

    rng = np.random.default_rng(Seed)

    # Precompute log-factorials up to K for fast log-choose
    logfac = _build_log_factorials(K)

    out: Dict[int, np.ndarray] = {}
    r_dtype = np.int16 if min(m, C) < 32767 else np.int32

    for N in range(N_min, K + 1):
        if m > N or C > N:
            continue  # infeasible N (shouldn't happen if N_min >= max(m,C))

        # Step 1: number of marked individuals still available at recapture
        if phi == 1.0:
            mt_vec = np.full(Simulations, m, dtype=np.int32)
        elif phi == 0.0:
            mt_vec = np.zeros(Simulations, dtype=np.int32)
        else:
            mt_vec = rng.binomial(m, phi, size=Simulations).astype(np.int32)

        # Group by mt to avoid recomputing pmfs many times
        unique_mt, counts = np.unique(mt_vec, return_counts=True)

        r_vals = np.empty(Simulations, dtype=r_dtype)
        pos = 0

        # Cache pmfs for this N (keyed by mt)
        pmf_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for mt, cnt in zip(unique_mt, counts):
            if cnt == 0:
                continue

            if mt == 0:
                draws = np.zeros(cnt, dtype=r_dtype)

            else:
                # Fast path: omega==1 => standard hypergeometric
                if omega == 1.0:
                    draws = rng.hypergeometric(
                        ngood=int(mt),
                        nbad=int(N - mt),
                        nsample=int(C),
                        size=int(cnt),
                    ).astype(r_dtype)

                else:
                    # Fisher noncentral hypergeometric via pmf + multinomial sampling of counts
                    if mt not in pmf_cache:
                        rs, probs = _fisher_nch_pmf(mt=int(mt), N=int(N), C=int(C), omega=float(omega), logfac=logfac)
                        pmf_cache[mt] = (rs, probs)
                    rs, probs = pmf_cache[mt]

                    # Sample cnt draws efficiently by sampling counts-per-r
                    r_counts = rng.multinomial(int(cnt), probs)
                    draws = np.repeat(rs, r_counts).astype(r_dtype)

            r_vals[pos:pos + cnt] = draws
            pos += cnt

        rng.shuffle(r_vals)  # optional: remove grouping artifacts
        out[N] = r_vals

    return out


# 2) Process: normalized probability of seeing a specific r across N

def normalized_probabilities_for_r(
    r_by_N: Dict[int, np.ndarray],
    r_target: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each N:
      p_hat(N) = (# times r_target appears in simulations for N) / Simulations

    Then normalize across all N:
      p_norm(N) = p_hat(N) / sum_N p_hat(N)

    Returns:
      Ns      : array of N values
      p_norm  : normalized probabilities (sum to 1 across Ns)
      p_hat   : raw Monte-Carlo estimates of P(r_target | N)
    """
    Ns = np.array(sorted(r_by_N.keys()), dtype=np.int32)
    if Ns.size == 0:
        return Ns, np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    sims = np.array([len(r_by_N[int(N)]) for N in Ns], dtype=np.int32)
    if not np.all(sims == sims[0]):
        raise ValueError("All N must have the same number of simulations.")
    S = int(sims[0])

    counts = np.array([(r_by_N[int(N)] == r_target).sum() for N in Ns], dtype=np.float64)
    p_hat = counts / S

    denom = p_hat.sum()
    p_norm = p_hat / denom if denom > 0 else np.zeros_like(p_hat)

    return Ns, p_norm, p_hat


# Example usage

if __name__ == "__main__":
    r_sims = simulate_r_values(
        N_min=120, K=300,
        m=20, C=20,
        phi=0.9, omega=1.2,
        Simulations=20000,
        Seed=42
    )

    Ns, p_norm, p_hat = normalized_probabilities_for_r(r_sims, r_target=5)
