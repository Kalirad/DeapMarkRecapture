'''
ehpmarkrecap.core
ehpmarkrecap v 0.1.0
Mirzaee et al., Feb 2026
'''

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp


# Helpers

def _log_comb_scalar(n: int, k: int) -> float:
    """log( n choose k ) for integers, returns -inf if invalid."""
    if k < 0 or k > n:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))


def _log_comb_vec(n: np.ndarray, k: int) -> np.ndarray:
    """
    Vectorized log( n choose k ) for integer n-array and scalar k.
    Returns -inf where invalid.
    """
    n = np.asarray(n)
    out = np.full(n.shape, -np.inf, dtype=float)
    if k < 0:
        return out
    mask = n >= k
    if np.any(mask):
        nn = n[mask]
        out[mask] = gammaln(nn + 1) - gammaln(k + 1) - gammaln(nn - k + 1)
    return out


# Core MR likelihoods / posteriors

def _log_likelihood_hypergeom(domain: np.ndarray, m: int, C: int, r: int) -> np.ndarray:
    """
    log P(r | N) under the classic hypergeometric model:
      P(r | N) = C(m,r)*C(N-m, C-r) / C(N,C)
    """
    domain = np.asarray(domain, dtype=int)
    log_const = _log_comb_scalar(m, r)
    log_part1 = _log_comb_vec(domain - m, C - r)
    log_part2 = _log_comb_vec(domain, C)
    return log_const + log_part1 - log_part2


def _log_hypergeom_r(domain: np.ndarray, mt: int, C: int, r: int) -> np.ndarray:
    """log P(r | N, mt) where mt is the number of available marked individuals."""
    domain = np.asarray(domain, dtype=int)
    if r < 0 or r > min(mt, C):
        return np.full(domain.shape, -np.inf, dtype=float)
    log_num = _log_comb_scalar(mt, r) + _log_comb_vec(domain - mt, C - r)
    log_den = _log_comb_vec(domain, C)
    return log_num - log_den


def _log_fisher_nchypergeom_r(domain: np.ndarray, mt: int, C: int, r: int, omega: float) -> np.ndarray:
    """
    Fisher's noncentral hypergeometric log pmf:
      P(r | N, mt, C, omega) =
        [C(mt,r)*C(N-mt, C-r)*omega^r] / sum_j [C(mt,j)*C(N-mt, C-j)*omega^j]
    """
    domain = np.asarray(domain, dtype=int)
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if r < 0 or r > min(mt, C):
        return np.full(domain.shape, -np.inf, dtype=float)

    logomega = math.log(omega)

    # numerator
    log_num = _log_comb_scalar(mt, r) + _log_comb_vec(domain - mt, C - r) + r * logomega

    # denominator via stable log-add
    jmax = min(mt, C)
    log_denom = np.full(domain.shape, -np.inf, dtype=float)

    # precompute log C(mt, j)
    logc_mt = [_log_comb_scalar(mt, j) for j in range(jmax + 1)]
    for j in range(jmax + 1):
        log_term = logc_mt[j] + _log_comb_vec(domain - mt, C - j) + j * logomega
        log_denom = np.logaddexp(log_denom, log_term)

    return log_num - log_denom


def _log_binom_pmf_all(m: int, phi: float):
    """Return mt array and log Binom(m,phi) pmf (already normalized)."""
    if not (0 <= phi <= 1):
        raise ValueError("phi must be between 0 and 1")

    mts = np.arange(0, m + 1, dtype=int)

    if phi == 0.0:
        logw = np.full(m + 1, -np.inf, dtype=float)
        logw[0] = 0.0
        return mts, logw

    if phi == 1.0:
        logw = np.full(m + 1, -np.inf, dtype=float)
        logw[m] = 0.0
        return mts, logw

    log_phi = math.log(phi)
    log_1mphi = math.log(1 - phi)

    logw = np.array(
        [_log_comb_scalar(m, mt) + mt * log_phi + (m - mt) * log_1mphi for mt in mts],
        dtype=float,
    )
    return mts, logw


def _log_likelihood_dataset(
    domain: np.ndarray,
    m: int, C: int, r: int,
    phi: float = 1.0,
    omega: float = 1.0,
    mt_tail_cutoff: float = 1e-12,
) -> np.ndarray:
    """
    log L(N; r, m, C, phi, omega):
      Mt ~ Binomial(m, phi)
      r | (N, Mt) ~ Fisher noncentral hypergeometric with odds omega
      L(N) = sum_mt Binom(m,mt;phi) * P(r | N, mt, C, omega)
    """
    domain = np.asarray(domain, dtype=int)

    # mt support
    if phi == 1.0:
        mt_values = [m]
        logw_values = [0.0]
    else:
        mts, logw_all = _log_binom_pmf_all(m, phi)

        if mt_tail_cutoff is None or mt_tail_cutoff <= 0:
            mask = np.ones_like(mts, dtype=bool)
        else:
            mask = logw_all >= math.log(mt_tail_cutoff)
            if not np.any(mask):
                mask[np.argmax(logw_all)] = True  # keep at least one term

        mt_values = mts[mask].tolist()
        logw_values = logw_all[mask].tolist()

    # mixture sum in log space
    logL = np.full(domain.shape, -np.inf, dtype=float)
    for mt, logw in zip(mt_values, logw_values):
        if omega == 1.0:
            logpr = _log_hypergeom_r(domain, mt, C, r)
        else:
            logpr = _log_fisher_nchypergeom_r(domain, mt, C, r, omega)
        logL = np.logaddexp(logL, logw + logpr)

    return logL


# Unbounded (K = infinity) exact posterior

def _log_pmf_unbounded(domain: np.ndarray, m: int, C: int, r: int) -> np.ndarray:
    """
    Unbounded posterior P(N | r) using the closed-form normalization:
      Z = sum_{N=Nmin..inf} P(r|N) = m*C / (r*(r-1)), valid for r>1
    """
    if r <= 1:
        raise ValueError("Unbounded model requires r>1 for proper normalization.")
    domain = np.asarray(domain, dtype=int)
    log_unnorm = _log_likelihood_hypergeom(domain, m, C, r)
    Z = (m * C) / (r * (r - 1))
    return log_unnorm - math.log(Z)


def _find_mode_unbounded(m: int, C: int, r: int) -> int:
    """Find a mode by scanning from Nmin upwards until log-pmf stops increasing."""
    Nmin = m + C - r
    curN = Nmin
    cur_logp = _log_pmf_unbounded(np.array([curN]), m, C, r)[0]
    while True:
        nxtN = curN + 1
        nxt_logp = _log_pmf_unbounded(np.array([nxtN]), m, C, r)[0]
        if nxt_logp > cur_logp + 1e-15:
            curN, cur_logp = nxtN, nxt_logp
        else:
            break
    return curN


def _ci_mode_expand_unbounded(m: int, C: int, r: int, alpha: float = 0.05):
    """
    Greedy CI expansion starting at the mode (contiguous), in log space.
    """
    Nmin = m + C - r
    mode = _find_mode_unbounded(m, C, r)
    L = U = mode

    log_cum = _log_pmf_unbounded(np.array([mode]), m, C, r)[0]
    target = math.log(1 - alpha)

    while log_cum < target - 1e-15:
        log_pl = -np.inf
        if L > Nmin:
            log_pl = _log_pmf_unbounded(np.array([L - 1]), m, C, r)[0]
        log_pr = _log_pmf_unbounded(np.array([U + 1]), m, C, r)[0]

        if log_pl > log_pr:
            L -= 1
            log_cum = np.logaddexp(log_cum, log_pl)
        else:
            U += 1
            log_cum = np.logaddexp(log_cum, log_pr)

    return L, U, mode


def _median_unbounded(m: int, C: int, r: int) -> int:
    """Median from CDF accumulation in log space."""
    Nmin = m + C - r
    target = math.log(0.5)
    log_cum = -np.inf
    N = Nmin
    while True:
        logp = _log_pmf_unbounded(np.array([N]), m, C, r)[0]
        log_cum = np.logaddexp(log_cum, logp)
        if log_cum >= target - 1e-15:
            return N
        N += 1


# CI from a bounded pmf (mode expansion / HPD)

def _ci_mode_expand_from_pmf(domain: np.ndarray, pmf: np.ndarray, alpha: float = 0.05):
    """Contiguous, greedy CI expansion from the pmf array."""
    domain = np.asarray(domain)
    pmf = np.asarray(pmf)
    mode_idx = int(np.argmax(pmf))

    L = U = mode_idx
    cum = float(pmf[mode_idx])
    target = 1 - alpha

    while cum < target - 1e-15:
        pl = pmf[L - 1] if L > 0 else -1.0
        pr = pmf[U + 1] if U < len(pmf) - 1 else -1.0

        if pl > pr:
            L -= 1
            cum += float(pl)
        else:
            U += 1
            cum += float(pr)

        if L == 0 and U == len(pmf) - 1:
            break

    return int(domain[L]), int(domain[U])


def _ci_hpd_interval(domain: np.ndarray, pmf: np.ndarray, alpha: float = 0.05):
    """
    HPD-set interval: take highest pmf values until coverage (1-alpha),
    then return [min(selected), max(selected)].
    """
    idx = np.argsort(pmf)[::-1]
    cum = 0.0
    selected = []
    for i in idx:
        selected.append(int(domain[i]))
        cum += float(pmf[i])
        if cum >= 1 - alpha:
            break
    return min(selected), max(selected)



# Main

def ehp(
    data,
    K=False,
    phi=False,
    omega=False,
    alpha=0.05,
    ci_method="mode_expand",
    mt_tail_cutoff=1e-12,
):
    """
    Comprehensive MR inference function.

    Parameters
    ----------
    data : (m, C, r) OR list of (m, C, r)
    K : int or False/None
        Environmental capacity / upper bound on N.
        REQUIRED if:
          - multiple datasets are provided, OR
          - phi != 1 or omega != 1 (heterogeneity / loss model).
        If K is False/None, only the single-dataset classic model (phi=1, omega=1)
        is allowed, and uses the exact unbounded normalization.
    phi : float or list[float] or False/None
        Retention/availability probability (0..1). False/None => 1.0
    omega : float or list[float] or False/None
        Relative catchability odds (>0). False/None => 1.0
    alpha : float
        CI tail probability (coverage is 1-alpha).
    ci_method : {"mode_expand", "hpd"}
        CI construction method for bounded distributions:
          - "mode_expand": contiguous greedy expansion around the mode
          - "hpd": HPD-set interval (min/max of top-mass set)
    mt_tail_cutoff : float
        Speed/accuracy knob for phi<1 models. Binomial(mt) terms with probability below
        this cutoff are dropped. Set to 0 for maximum accuracy.

    Returns
    -------
    dict with:
      mode, mean, median, ci_low, ci_high,
      plus: domain, pmf, np_values for plotting.
    """

    # --- parse datasets ---
    if isinstance(data, tuple) and len(data) == 3:
        data_sets = [tuple(int(x) for x in data)]
    else:
        data_sets = [tuple(int(x) for x in d) for d in data]

    for (m, C, r) in data_sets:
        if m < 0 or C < 0 or r < 0:
            raise ValueError("m, C, r must be non-negative integers.")
        if r > min(m, C):
            raise ValueError(f"r cannot exceed min(m,C). Got r={r}, m={m}, C={C}.")

    n = len(data_sets)

    # --- normalize phi/omega to per-dataset lists ---
    def _as_list(x, default):
        if x is False or x is None:
            return [default] * n
        if isinstance(x, (int, float, np.number)):
            return [float(x)] * n
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) != n:
                raise ValueError(f"Expected a scalar or a list of length {n}.")
            return [float(v) for v in x]
        raise ValueError("phi/omega must be False/None, a scalar, or a list/tuple.")

    phi_list = _as_list(phi, 1.0)
    omega_list = _as_list(omega, 1.0)

    heterogeneity = any(abs(p - 1.0) > 0 for p in phi_list) or any(abs(w - 1.0) > 0 for w in omega_list)

    # --- normalize K ---
    K_val = None if (K is False or K is None) else int(K)

    # Rule
    if (n > 1 or heterogeneity) and K_val is None:
        raise ValueError("K must be provided (finite) for multiple datasets and/or when phi or omega are used.")

    # Unbounded exact case
    if K_val is None:
        (m, C, r) = data_sets[0]
        if r <= 1:
            raise ValueError("Without K, r must be > 1 for CI/PMF normalization.")

        ci_low, ci_high, mode = _ci_mode_expand_unbounded(m, C, r, alpha=alpha)
        median = _median_unbounded(m, C, r)

        # expected value rule you requested (and matches Eq. 8 behavior)
        if r <= 2:
            mean = float("inf")
        else:
            mean = (m - 1) * (C - 1) / (r - 2)

        Nmin = m + C - r
        plot_max = int(2 * ci_high)
        domain = np.arange(Nmin, plot_max + 1, dtype=int)
        pmf = np.exp(_log_pmf_unbounded(domain, m, C, r))
        np_values = list(zip(domain.tolist(), pmf.tolist()))

        return {
            "mode": int(mode),
            "mean": float(mean),
            "median": int(median),
            "ci_low": int(ci_low),
            "ci_high": int(ci_high),
            "alpha": float(alpha),
            "K": None,
            "datasets": data_sets,
            "phi": phi_list,
            "omega": omega_list,
            "ci_method": ci_method,
            "domain": domain,
            "pmf": pmf,
            "np_values": np_values,
        }

    # Bounded (K provided) case
    Nmin = max(m + C - r for (m, C, r) in data_sets)
    if K_val < Nmin:
        raise ValueError(f"K must be >= Nmin={Nmin}. Got K={K_val}.")

    domain = np.arange(Nmin, K_val + 1, dtype=int)
    total_loglik = np.zeros(domain.shape, dtype=float)

    for (m, C, r), ph, om in zip(data_sets, phi_list, omega_list):
        if not (0 <= ph <= 1):
            raise ValueError(f"phi must be in [0,1]. Got {ph}.")
        if om <= 0:
            raise ValueError(f"omega must be > 0. Got {om}.")

        total_loglik += _log_likelihood_dataset(
            domain, m, C, r, phi=ph, omega=om, mt_tail_cutoff=mt_tail_cutoff
        )

    logZ = logsumexp(total_loglik)
    logpost = total_loglik - logZ
    pmf = np.exp(logpost)

    # summary stats
    mode = int(domain[int(np.argmax(pmf))])
    mean = float(np.dot(domain, pmf))
    median = int(domain[np.searchsorted(np.cumsum(pmf), 0.5)])

    ci_method_l = ci_method.lower()
    if ci_method_l in ("mode_expand", "expand", "greedy"):
        ci_low, ci_high = _ci_mode_expand_from_pmf(domain, pmf, alpha=alpha)
    elif ci_method_l in ("hpd", "hpd_set"):
        ci_low, ci_high = _ci_hpd_interval(domain, pmf, alpha=alpha)
    else:
        raise ValueError("ci_method must be 'mode_expand' or 'hpd'.")

    np_values = list(zip(domain.tolist(), pmf.tolist()))
    return {
        "mode": int(mode),
        "mean": float(mean),
        "median": int(median),
        "ci_low": int(ci_low),
        "ci_high": int(ci_high),
        "alpha": float(alpha),
        "K": int(K_val),
        "datasets": data_sets,
        "phi": phi_list,
        "omega": omega_list,
        "ci_method": ci_method,
        "domain": domain,
        "pmf": pmf,
        "np_values": np_values,
    }


def ehp_plot(results, ax=None, title=None, figsize=(10, 5), show=True):
    """
    Minimal professional plot for MR posterior values + Mode/Mean/Median/CI.

    Parameters
    ----------
    results : dict
        Output of Calc_MR(...)
    ax : matplotlib.axes.Axes or None
    title : str or None
    figsize : tuple
    show : bool

    Returns
    -------
    (fig, ax)
    """
    domain = np.asarray(results["domain"])
    pmf = np.asarray(results["pmf"])

    mode = results["mode"]
    mean = results["mean"]
    median = results["median"]
    ci_low = results["ci_low"]
    ci_high = results["ci_high"]
    alpha = results.get("alpha", None)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(domain, pmf, linewidth=2, label="Posterior PMF")
    ax.fill_between(domain, pmf, alpha=0.15)

    # CI shading
    if alpha is not None:
        ax.axvspan(ci_low, ci_high, alpha=0.15, label=f"{int(round((1-alpha)*100))}% CI")
    else:
        ax.axvspan(ci_low, ci_high, alpha=0.15, label="CI")

    # markers
    ax.axvline(mode, linestyle="-", linewidth=2, label=f"Mode = {mode}")
    ax.axvline(median, linestyle="--", linewidth=2, label=f"Median = {median}")
    if math.isfinite(mean):
        ax.axvline(mean, linestyle=":", linewidth=2, label=f"Mean = {mean:.2f}")
    else:
        ax.axvline(mode, linestyle=":", linewidth=2, label="Mean = ∞")

    ax.axvline(ci_low, linestyle="--", linewidth=1)
    ax.axvline(ci_high, linestyle="--", linewidth=1)

    ax.set_xlabel("Population size (N)")
    ax.set_ylabel("Probability")

    if title is None:
        if results.get("K", None) is None:
            title = "Mark–recapture EHP (unbounded)"
        else:
            title = f"Mark–recapture EHP (K = {results['K']})"
    ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ax.set_xlim(domain[0], domain[-1])
    ax.set_ylim(bottom=0)

    if show:
        plt.show()

    return fig, ax

'''
# Example Usage ------------------------------
#Single"
m, C, r = 100, 100, 10
res = ehp((m, C, r), K=False, phi=False, omega=False, alpha=0.05)
print(res["mode"], res["mean"], res["median"], res["ci_low"], res["ci_high"])

#Single with K
res = ehp((100, 100, 10), K=10000, alpha=0.05)
print(res["mode"], res["mean"], res["median"], res["ci_low"], res["ci_high"])

#Multiple
data_sets = [(100, 100, 10), (100, 100, 10)]
res = ehp(data_sets, K=10000, alpha=0.05)
print(res["mode"], res["mean"], res["median"], res["ci_low"], res["ci_high"])

# With Hetrogeneity
# phi only
res_phi = ehp((20, 20, 5), K=3000, phi=0.5, omega=False, alpha=0.05)
# omega only
res_om = ehp((20, 20, 5), K=3000, phi=False, omega=2.0, alpha=0.05)
# both
res_both = ehp((20, 20, 5), K=3000, phi=0.7, omega=1.5, alpha=0.05)
# multiple
res_multi = ehp(data_sets, K=3000, phi=0.9, omega=1.1, alpha=0.05)
#Per-dataset phi/omega"
res = ehp(data_sets, K=2000, phi=[0.9, 0.8], omega=[1.0, 1.2], alpha=0.05)

# Adjusting Other Inputs
res = ehp(data_sets, K=2000, phi=0.8, omega=1.1, alpha=0.05, ci_method="hpd", mt_tail_cutoff=0)

# Visulaizing
ehp_plot(res)
'''
if __name__ == "__main__":
    m, C, r = 50000, 50000, 1000
    res = ehp((m, C, r), K=False, phi=False, omega=False, alpha=0.05)
    print(res["mode"], res["mean"], res["median"], res["ci_low"], res["ci_high"])