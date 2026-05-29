"""
Established Mark–recapture (MR) estimators + confidence intervals (CIs)

Covers 6 “doable + essential” categories:

1) Single MR point estimates (m, C, r)
   - Lincoln–Petersen (LP) MLE
   - Chapman (bias-corrected LP)

2) Single MR CIs (m, C, r)
   - Chapman log-normal CI (variance-based)
   - Exact hypergeometric inversion CI (frequentist equal-tail)

3) Multiple MR / combined-PDF point estimates (arrays of m, C, r)
   - Schnabel (classical repeated MR; assumes sequential marking design)
   - Schumacher–Eschmeyer (classical repeated MR; sequential marking design)

4) Multiple MR / combined-PDF CIs
   - Schnabel Poisson (Garwood / chi-square) CI
   - Schumacher–Eschmeyer t-CI on 1/N

5) Single MR with heterogeneity point estimates (m, C, r, omega[, K])
   - Chao lower bound (2-source; no omega needed, but standard heterogeneity-robust comparator)
   - MLE for N under Fisher noncentral hypergeometric (omega)

6) Single MR with heterogeneity CIs
   - Chao normal CI (variance-based)
   - Profile likelihood CI for N under Fisher NCHG (omega)
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import chi2, hypergeom, norm, t


# Utilities

def _validate_single(m: int, C: int, r: int) -> None:
    for name, val in [("m", m), ("C", C), ("r", r)]:
        if not (isinstance(val, int) and val >= 0):
            raise ValueError(f"{name} must be a nonnegative integer.")
    if r > m or r > C:
        raise ValueError("Require r <= min(m, C).")


def _validate_multiple(ms: Sequence[int], Cs: Sequence[int], rs: Sequence[int]) -> None:
    if not (len(ms) == len(Cs) == len(rs)):
        raise ValueError("ms, Cs, rs must have the same length.")
    if len(ms) == 0:
        raise ValueError("Need at least one MR event / occasion.")
    for i, (m, C, r) in enumerate(zip(ms, Cs, rs)):
        try:
            _validate_single(int(m), int(C), int(r))
        except Exception as e:
            raise ValueError(f"Invalid counts at index {i}: {e}") from e


def _log_choose(n: int, k: int) -> float:
    """log(n choose k) using gammaln; returns -inf if invalid."""
    if k < 0 or k > n:
        return -math.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p < 0 or p > 1:
        return -math.inf
    if k < 0 or k > n:
        return -math.inf
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    return _log_choose(n, k) + k * math.log(p) + (n - k) * math.log(1.0 - p)


def _shortest_credible_interval(domain: np.ndarray, pmf: np.ndarray, alpha: float = 0.05) -> Tuple[int, int]:
    """
    Greedy “shortest contiguous” credible interval builder.
    """
    idx_mode = int(np.argmax(pmf))
    i = j = idx_mode
    cum = float(pmf[idx_mode])
    target = 1.0 - alpha

    while cum < target:
        p_left = pmf[i - 1] if i > 0 else -1.0
        p_right = pmf[j + 1] if j < len(pmf) - 1 else -1.0

        if p_left < 0 and p_right < 0:
            break

        if p_left >= p_right:
            i -= 1
            cum += float(pmf[i])
        else:
            j += 1
            cum += float(pmf[j])

    return int(domain[i]), int(domain[j])


def _pmf_summaries(domain: np.ndarray, pmf: np.ndarray) -> Tuple[int, float, int]:
    idx_mode = int(np.argmax(pmf))
    mode = int(domain[idx_mode])
    mean = float(np.sum(domain * pmf))
    cdf = np.cumsum(pmf)
    median = int(domain[np.searchsorted(cdf, 0.5)])
    return mode, mean, median

# Likelihoods for r | N

def loglik_hypergeom(N: int, m: int, C: int, r: int) -> float:
    """
    log P(r | N,m,C) under standard hypergeometric model.
    R ~ Hypergeom(pop=N, marked=m, draws=C)
    """
    if r > m or r > C:
        return -math.inf
    if N < m or N < C:
        return -math.inf
    if C - r > N - m:
        return -math.inf
    return _log_choose(m, r) + _log_choose(N - m, C - r) - _log_choose(N, C)


def logpmf_fisher_nchypergeom(N: int, m: int, C: int, r: int, omega: float) -> float:
    """
    Fisher's noncentral hypergeometric pmf:
      P(r | N,m,C,omega) ∝ choose(m,r) choose(N-m, C-r) omega^r
      normalized by sum over feasible r.
    """
    if omega <= 0:
        raise ValueError("omega must be > 0.")
    if r > m or r > C:
        return -math.inf
    if N < m or N < C:
        return -math.inf
    if C - r > N - m:
        return -math.inf

    jmin = max(0, C - (N - m))
    jmax = min(m, C)

    log_omega = math.log(omega)
    logw = []
    for j in range(jmin, jmax + 1):
        if C - j > N - m:
            continue
        logw.append(_log_choose(m, j) + _log_choose(N - m, C - j) + j * log_omega)

    if not logw:
        return -math.inf

    logZ = float(logsumexp(logw))
    logr = _log_choose(m, r) + _log_choose(N - m, C - r) + r * log_omega
    return logr - logZ


def loglik_r_given_N(
    N: int,
    m: int,
    C: int,
    r: int,
    omega: float = 1.0,
    phi: float = 1.0,
    integrate_phi: bool = False,
) -> float:
    """
    General likelihood that matches your section 3.1 “option 1” structure:

      - omega: marked/unmarked sampling odds ratio (Fisher NCHG)
      - phi: marked retention/availability probability
      - integrate_phi:
          False: plug-in m_eff = round(m*phi)
          True : marginalize over M_t ~ Binomial(m, phi) and then r | (N, M_t) via (N)CHG

    If omega==1 and phi==1 this reduces to the standard hypergeometric likelihood.
    """
    if integrate_phi:
        terms = []
        for m_t in range(r, m + 1):  # must have at least r marked available
            if N < m_t:
                continue
            lb = _log_binom_pmf(m_t, m, phi)
            if omega == 1.0:
                lh = loglik_hypergeom(N, m_t, C, r)
            else:
                lh = logpmf_fisher_nchypergeom(N, m_t, C, r, omega)
            if math.isfinite(lb) and math.isfinite(lh):
                terms.append(lb + lh)
        return float(logsumexp(terms)) if terms else -math.inf

    # plug-in m_eff
    m_eff = int(round(m * phi))
    m_eff = min(max(m_eff, 0), m)
    if m_eff < r:
        return -math.inf
    return loglik_hypergeom(N, m_eff, C, r) if omega == 1.0 else logpmf_fisher_nchypergeom(N, m_eff, C, r, omega)


# Posterior over N (bounded by K)
@dataclass(frozen=True)
class PosteriorSummary:
    mode: int
    mean: float
    median: int
    ci: Tuple[int, int]
    domain_min: int
    domain_max: int


def posterior_over_N_single(
    m: int,
    C: int,
    r: int,
    K: int,
    *,
    omega: float = 1.0,
    phi: float = 1.0,
    integrate_phi: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Posterior pmf for N on [N_min, K] with a uniform prior over that range:
      P(N | data) ∝ L(N)
    """
    _validate_single(m, C, r)
    N_min = m + C - r
    if K < N_min:
        raise ValueError(f"K must be >= N_min = m + C - r = {N_min}.")

    domain = np.arange(N_min, K + 1, dtype=int)
    logL = np.array(
        [loglik_r_given_N(int(N), m, C, r, omega=omega, phi=phi, integrate_phi=integrate_phi) for N in domain],
        dtype=float,
    )

    finite = np.isfinite(logL)
    if not np.any(finite):
        raise ValueError("All likelihood values are -inf; check inputs and omega/phi.")

    logL_max = float(np.max(logL[finite]))
    w = np.exp(logL - logL_max)
    w[~finite] = 0.0
    pmf = w / float(np.sum(w))
    return domain, pmf


def posterior_summary_single(
    m: int,
    C: int,
    r: int,
    K: int,
    *,
    alpha: float = 0.05,
    omega: float = 1.0,
    phi: float = 1.0,
    integrate_phi: bool = False,
) -> PosteriorSummary:
    domain, pmf = posterior_over_N_single(m, C, r, K, omega=omega, phi=phi, integrate_phi=integrate_phi)
    mode, mean, median = _pmf_summaries(domain, pmf)
    L, U = _shortest_credible_interval(domain, pmf, alpha=alpha)
    return PosteriorSummary(
        mode=mode, mean=mean, median=median, ci=(L, U),
        domain_min=int(domain[0]), domain_max=int(domain[-1])
    )



# (1) Single MR: point estimates
# 1.1
def lp_mle_point(m: int, C: int, r: int) -> float:
    """Lincoln–Petersen MLE: m*C/r (undefined when r=0)."""
    _validate_single(m, C, r)
    return math.inf if r == 0 else (m * C) / r

# 1.2
def chapman_point(m: int, C: int, r: int) -> float:
    """Chapman estimator: ((m+1)(C+1)/(r+1)) - 1."""
    _validate_single(m, C, r)
    return ((m + 1) * (C + 1) / (r + 1)) - 1


# (2) Single MR: CIs
# 2.1
def chapman_variance(m: int, C: int, r: int) -> float:
    """
    Approx variance of Chapman estimator:
      Var = ((m+1)(C+1)(m-r)(C-r)) / ((r+1)^2 (r+2))
    """
    _validate_single(m, C, r)
    return ((m + 1) * (C + 1) * (m - r) * (C - r)) / (((r + 1) ** 2) * (r + 2))


def chapman_log_ci(m: int, C: int, r: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Log-normal CI built from Chapman variance (keeps bounds positive)."""
    Nhat = chapman_point(m, C, r)
    var = chapman_variance(m, C, r)
    if not math.isfinite(var) or var < 0 or Nhat <= 0:
        return (math.nan, math.nan)

    z = float(norm.ppf(1 - alpha / 2))
    se = math.sqrt(var)
    se_log = se / Nhat
    lower = Nhat * math.exp(-z * se_log)
    upper = Nhat * math.exp(z * se_log)

    # enforce feasibility
    N_min = m + C - r
    lower = max(lower, float(N_min))
    return lower, upper

# 2.2
def exact_hypergeom_ci(m: int, C: int, r: int, alpha: float = 0.05, K_max: Optional[int] = None) -> Tuple[int, float]:
    """
    Two-sided equal-tail exact CI for N by inversion of the hypergeometric distribution:
      R ~ Hypergeom(N, m, C)

    Returns (N_L, N_U). If r=0 and K_max is None, N_U is +inf.
    """
    _validate_single(m, C, r)
    N_min = m + C - r

    if r == 0:
        return int(N_min), (float(K_max) if K_max is not None else math.inf)

    # choose search ceiling
    if K_max is None:
        Nhat = chapman_point(m, C, r)
        K = max(N_min + 10, int(math.ceil(Nhat * 20 + 1000)))
        while hypergeom.sf(r - 1, K, m, C) > alpha / 2:
            K *= 2
            if K > 10_000_000:
                break
    else:
        K = int(K_max)

    def cdfN(N: int) -> float:
        return float(hypergeom.cdf(r, N, m, C))

    def sfN(N: int) -> float:
        return float(hypergeom.sf(r - 1, N, m, C))  # P(R >= r)

    # Lower: smallest N with CDF(r) >= alpha/2
    lo, hi = int(N_min), int(K)
    if cdfN(lo) >= alpha / 2:
        N_L = lo
    else:
        while hi < 10_000_000 and cdfN(hi) < alpha / 2:
            hi *= 2
        lo = int(N_min)
        while lo < hi:
            mid = (lo + hi) // 2
            if cdfN(mid) >= alpha / 2:
                hi = mid
            else:
                lo = mid + 1
        N_L = lo

    # Upper: largest N with SF >= alpha/2
    if K_max is not None:
        if sfN(K) >= alpha / 2:
            N_U = K
        else:
            lo, hi = N_L, K
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if sfN(mid) >= alpha / 2:
                    lo = mid
                else:
                    hi = mid - 1
            N_U = lo
        return int(N_L), float(N_U)

    # unbounded case
    if sfN(K) >= alpha / 2:
        while K < 10_000_000 and sfN(K) >= alpha / 2:
            K *= 2
        if sfN(K) >= alpha / 2:
            return int(N_L), math.inf

    lo, hi = N_L, int(K)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sfN(mid) >= alpha / 2:
            lo = mid
        else:
            hi = mid - 1
    return int(N_L), float(lo)



# (3) Multiple MR / combined-PDF: point estimates
# 3.1

def schnabel_point(ms: Sequence[int], Cs: Sequence[int], rs: Sequence[int], *, bias_correction: bool = False) -> float:
    """
    Schnabel estimate (sequential marking design):
      N_hat = sum(C_t M_t) / sum(R_t)  (or / (sum(R_t)+1) if bias_correction)
    """
    _validate_multiple(ms, Cs, rs)
    ms = np.asarray(ms, dtype=float)
    Cs = np.asarray(Cs, dtype=float)
    rs = np.asarray(rs, dtype=float)
    A = float(np.sum(ms * Cs))
    R = float(np.sum(rs))
    denom = (R + 1.0) if bias_correction else R
    return math.inf if denom <= 0 else (A / denom)

# 3.2
def schumacher_eschmeyer_point(ms: Sequence[int], Cs: Sequence[int], rs: Sequence[int]) -> float:
    """
    Schumacher–Eschmeyer estimate (sequential marking design):
      N_hat = sum(C_t M_t^2) / sum(R_t M_t)
    """
    _validate_multiple(ms, Cs, rs)
    ms = np.asarray(ms, dtype=float)
    Cs = np.asarray(Cs, dtype=float)
    rs = np.asarray(rs, dtype=float)
    num = float(np.sum(Cs * ms**2))
    den = float(np.sum(rs * ms))
    return math.inf if den <= 0 else (num / den)


# (4) Multiple MR / combined-PDF: CIs
# 4.1
def schnabel_poisson_ci(ms: Sequence[int], Cs: Sequence[int], rs: Sequence[int], alpha: float = 0.05) -> Tuple[float, float]:
    """
    Approx CI for Schnabel using Poisson approximation for total recaptures:
      R = sum(R_t) ~ Poisson(lambda),  lambda = A/N,  A = sum(C_t M_t)
    Garwood CI for lambda, then transform to N.
    """
    _validate_multiple(ms, Cs, rs)
    ms = np.asarray(ms, dtype=float)
    Cs = np.asarray(Cs, dtype=float)
    rs = np.asarray(rs, dtype=float)

    A = float(np.sum(ms * Cs))
    R = int(np.sum(rs))
    if A <= 0:
        raise ValueError("sum(C_t * M_t) must be positive.")

    lam_low = 0.0 if R == 0 else 0.5 * float(chi2.ppf(alpha / 2, 2 * R))
    lam_up = 0.5 * float(chi2.ppf(1 - alpha / 2, 2 * (R + 1)))

    N_low = A / lam_up if lam_up > 0 else math.inf
    N_up = (A / lam_low) if lam_low > 0 else math.inf
    return N_low, N_up

# 4.2
def schumacher_eschmeyer_t_ci(ms, Cs, rs, alpha=0.05):
    """
    Approx CI for Schumacher–Eschmeyer via t-interval on 1/N.

    Variance estimator on 1/N:
      Var(1/N_hat) = [ sum(R_t^2 / C_t) - (sum(R_t M_t))^2 / sum(C_t M_t^2) ] / (s-2)

    Then:
      1/N_hat ± t_{1-alpha/2, df=s-2} * sqrt(Var)
    invert to N.
    """
    _validate_multiple(ms, Cs, rs)
    s = len(ms)
    if s < 3:
        raise ValueError("Need at least 3 occasions for Schumacher–Eschmeyer t CI (df=s-2).")

    ms = np.asarray(ms, dtype=float)
    Cs = np.asarray(Cs, dtype=float)
    rs = np.asarray(rs, dtype=float)

    sum_nM2 = float(np.sum(Cs * ms**2))
    sum_mM  = float(np.sum(rs * ms))
    if sum_nM2 <= 0 or sum_mM <= 0:
        return (math.nan, math.inf)

    Nhat = sum_nM2 / sum_mM
    invN = 1.0 / Nhat

    # Weighted SSE for regression through origin
    term1 = float(np.sum((rs**2) / Cs))                 # sum(m^2 / n)
    term2 = (sum_mM ** 2) / sum_nM2
    SSE = term1 - term2

    df = s - 2
    if SSE <= 0:
        # perfect fit (or numerical tie) => zero SE
        return (Nhat, Nhat)

    var_invN = (SSE / df) / sum_nM2
    se_invN = math.sqrt(var_invN)

    tcrit = float(t.ppf(1 - alpha / 2, df=df))
    lo_inv = invN - tcrit * se_invN
    hi_inv = invN + tcrit * se_invN
    lo_inv, hi_inv = min(lo_inv, hi_inv), max(lo_inv, hi_inv)

    if hi_inv <= 0:
        return (math.nan, math.inf)

    N_low = 1.0 / hi_inv
    N_up = math.inf if lo_inv <= 0 else 1.0 / lo_inv
    return N_low, N_up


# (5) Single MR with heterogeneity: point estimates
# 5.1
def chao_point_two_sample(m: int, C: int, r: int) -> float:
    """
    Chao lower bound (two-source) in terms of f1, f2:
      f1 = (m-r) + (C-r) = m + C - 2r
      f2 = r
      n  = observed = m + C - r
      N_hat = n + f1^2 / (4 f2)
    """
    _validate_single(m, C, r)
    if r == 0:
        return math.inf
    f1 = m + C - 2 * r
    n = m + C - r
    return n + (f1 * f1) / (4.0 * r)

# 5.2
def nchypergeom_mle_point(m: int, C: int, r: int, omega: float, K: int, *, phi: float = 1.0, integrate_phi: bool = False) -> int:
    """
    MLE for N under Fisher noncentral hypergeometric likelihood (omega),
    optionally with retention phi (plug-in or integrated).
    """
    _validate_single(m, C, r)
    N_min = m + C - r
    if K < N_min:
        raise ValueError(f"K must be >= N_min = {N_min}.")

    domain = np.arange(N_min, K + 1, dtype=int)
    logL = np.array([loglik_r_given_N(int(N), m, C, r, omega=omega, phi=phi, integrate_phi=integrate_phi) for N in domain])
    if not np.any(np.isfinite(logL)):
        raise ValueError("All likelihood values are -inf; check omega/phi.")
    return int(domain[int(np.argmax(logL))])

# 5.3
def posterior_point_hetero(m: int, C: int, r: int, omega: float, K: int, *, phi: float = 1.0, integrate_phi: bool = False) -> Tuple[int, float, int]:
    """
    Posterior (mode, mean, median) for N under Fisher NCHG (omega),
    optionally with retention phi.
    """
    summ = posterior_summary_single(m, C, r, K, alpha=0.05, omega=omega, phi=phi, integrate_phi=integrate_phi)
    return summ.mode, summ.mean, summ.median


# (6) Single MR with heterogeneity: CIs
# 6.1
def chao_normal_ci_two_sample(m: int, C: int, r: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Normal CI for Chao using variance estimator:
      Var(N_hat) = (f1^2/(4 f2)) * (f1/(2 f2) + 1)^2
    """
    _validate_single(m, C, r)
    if r == 0:
        return (float(m + C - r), math.inf)

    f1 = m + C - 2 * r
    f2 = r
    Nhat = chao_point_two_sample(m, C, r)
    var = (f1 * f1) / (4.0 * f2) * ((f1 / (2.0 * f2) + 1.0) ** 2)

    z = float(norm.ppf(1 - alpha / 2))
    se = math.sqrt(max(var, 0.0))
    lower = max(float(m + C - r), Nhat - z * se)
    upper = Nhat + z * se
    return lower, upper

# 6.2
def nchypergeom_profile_ci(m: int, C: int, r: int, omega: float, K: int, alpha: float = 0.05, *, phi: float = 1.0, integrate_phi: bool = False) -> Tuple[int, int]:
    """
    Likelihood-ratio CI for N under Fisher NCHG likelihood (omega):
      2*(logL_max - logL(N)) <= chi2_{1,1-alpha}.
    Domain is [N_min, K].
    """
    _validate_single(m, C, r)
    N_min = m + C - r
    if K < N_min:
        raise ValueError(f"K must be >= N_min = {N_min}.")

    domain = np.arange(N_min, K + 1, dtype=int)
    logL = np.array([loglik_r_given_N(int(N), m, C, r, omega=omega, phi=phi, integrate_phi=integrate_phi) for N in domain])
    finite = np.isfinite(logL)
    if not np.any(finite):
        raise ValueError("All likelihood values are -inf; check omega/phi.")

    logL_max = float(np.max(logL[finite]))
    cutoff = logL_max - 0.5 * float(chi2.ppf(1 - alpha, df=1))
    ok = logL >= cutoff
    Ns = domain[ok]
    return int(Ns[0]), int(Ns[-1])


def posterior_ci_hetero(m: int, C: int, r: int, omega: float, K: int, alpha: float = 0.05, *, phi: float = 1.0, integrate_phi: bool = False) -> Tuple[int, int]:
    """Shortest (1-alpha) posterior credible interval for N under Fisher NCHG likelihood (omega)."""
    summ = posterior_summary_single(m, C, r, K, alpha=alpha, omega=omega, phi=phi, integrate_phi=integrate_phi)
    return summ.ci



# Example quick run
if __name__ == "__main__":
    m, C, r = 50, 40, 10
    K = 2000

    print("Single MR points:")
    print("  LP MLE:", lp_mle_point(m, C, r))
    print("  Chapman:", chapman_point(m, C, r))

    print("\nSingle MR CIs:")
    print("  Chapman log CI:", chapman_log_ci(m, C, r))
    print("  Exact hypergeom CI:", exact_hypergeom_ci(m, C, r))

    ms, Cs, rs = [10, 20, 30, 40], [15, 15, 15, 15], [1, 2, 3, 4]
    print("\nMultiple MR points:")
    print("  Schnabel:", schnabel_point(ms, Cs, rs))
    print("  Schumacher–Eschmeyer:", schumacher_eschmeyer_point(ms, Cs, rs))

    print("\nMultiple MR CIs:")
    print("  Schnabel Poisson CI:", schnabel_poisson_ci(ms, Cs, rs))
    print("  Schumacher–Eschmeyer t CI:", schumacher_eschmeyer_t_ci(ms, Cs, rs))

    omega = 1.2
    print("\nSingle MR with heterogeneity:")
    print("  Chao:", chao_point_two_sample(m, C, r))
    print("  NCHG MLE:", nchypergeom_mle_point(m, C, r, omega, K))

    print("\nSingle MR with heterogeneity CIs:")
    print("  Chao normal CI:", chao_normal_ci_two_sample(m, C, r))
    print("  NCHG profile CI:", nchypergeom_profile_ci(m, C, r, omega, K))