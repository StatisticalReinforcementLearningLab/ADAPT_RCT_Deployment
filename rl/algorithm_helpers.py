"""RL feature, reward-redistribution (V6), and RLSVI helpers for ``RL_agents``.

This module mirrors the project-level ``algorithm_helpers`` used by the
validated simulation arm ``rl_v6_invariant_redistributed``:

* 27-dimensional walking Q feature map (within-week mediator EWMAs,
  5-dimensional decision context, ``A x [1, E_w, b_hat, b_tilde, C,
  weekday, slot_pm]``);
* discount 1 within the week and ``gamma_bar`` only on the week-terminal
  slot;
* two-stage within-week reward redistribution (Stage 1 daily AA/FW/PJ
  decompositions, Stage 1b slot-level step count, Stage 2 learned map with
  realized page view) of the potential-shaped weekly target
  ``b_hat_{w+1} + gamma_bar * e_hat_{w+1} - e_hat_w``;
* empirical-Bayes TD noise variance on the ensemble-mean target;
* explicit numerical diagnostics instead of silent repairs.

Missing values are represented as NaN everywhere. Mediator EWMAs skip NaN
entries while keeping their calendar position; any non-finite feature that
would reach a regression or an action decision is reported, never coerced to
zero.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize_scalar

_DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def _rl_dimensions_from_config(
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> tuple[int, int]:
    path = config_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"RL config not found at {path}; n_rl_days and n_rl_slots are required"
        )
    with path.open(encoding="utf-8") as f:
        config = json.load(f)
    for key in ("n_rl_days", "n_rl_slots"):
        if key not in config or config[key] is None:
            raise KeyError(f"RL config must include {key!r}")
    return int(config["n_rl_days"]), int(config["n_rl_slots"])


N_RL_DAYS, N_RL_SLOTS = _rl_dimensions_from_config()
FIRST_D = 0
FIRST_T = 0
TERMINAL_D = N_RL_DAYS - 1
TERMINAL_T = N_RL_SLOTS - 1

EB_SIGMA2_MIN = 1e-4
EB_SIGMA2_LOG_HALFWIDTH = 8.0
EB_MAXITER = 200

# Decision-time context sent by the server: 5 entries, raw scale.
#   0 yesterdayStepCount, 1 prior2HourStepCount, 2 activeDaysLast7Days,
#   3 activitySuggestionsSentLast7Days (recent burden),
#   4 activitySuggestionInteractLast7Days.
# The PF fourSC model and the RLSVI context (testbed
# ``build_rl_context_vector``) both use all five.
SERVER_C_NAMES = (
    "yesterdayStepCount",
    "prior2HourStepCount",
    "activeDaysLast7Days",
    "activitySuggestionsSentLast7Days",
    "activitySuggestionInteractLast7Days",
)
N_SERVER_C = len(SERVER_C_NAMES)
RL_C_FROM_SERVER = (0, 1, 2, 3, 4)
N_RL_CONTEXT = len(RL_C_FROM_SERVER)
RL_CONTEXT_NAMES = (
    "yesterday_step_count",
    "prior2hour_step_count",
    "active_status_fraction_7days",
    "recent_burden",
    "walk_interaction_7d",
)
MEDIATOR_SUMMARY_NAMES = ("M_AA", "M_SC", "M_PV", "M_FW", "M_PJ")
PHI_STATE_BASE_NAMES = (
    "intercept",
    "weekday_vs_weekend",
    "slot_pm",
    "E_w",
    "b_hat",
    "b_tilde",
)
ACTION_TIME_NAMES = ("weekday_vs_weekend", "slot_pm")
# Stage-2 share columns (AA/FW/PJ daily + SC slot-level); PV enters Stage 2
# through its realized value.
STAGE1_NAMES = ("AA", "FW", "PJ", "SC")
_DAILY_TARGETS = {"AA": (0, 2), "FW": (1, 2), "PJ": (1, 3)}


class NonFiniteFeatures(ValueError):
    """Raised when a feature vector that must drive a decision is non-finite."""


# ---------------------------------------------------------------------------
# Context and mediator summaries
# ---------------------------------------------------------------------------
def rl_context_from_server(c: Any) -> np.ndarray:
    """Select the 5-d RLSVI context from the 5-d server context (same order).

    Entries not provided are NaN (never zero-filled).
    """
    arr = np.asarray(c, dtype=float).ravel()
    out = np.full(N_RL_CONTEXT, np.nan, dtype=float)
    for j, idx in enumerate(RL_C_FROM_SERVER):
        if idx < arr.size:
            out[j] = arr[idx]
    return out


def build_rl_context_vector(
    *,
    yesterdayStepCount,
    prior2HourStepCount,
    activeDaysLast7Days,
    activitySuggestionsSentLast7Days,
    activitySuggestionInteractLast7Days,
) -> np.ndarray:
    """Per-decision 5-d context ``C`` for RLSVI (model scale)."""
    return np.array(
        [
            float(yesterdayStepCount),
            float(prior2HourStepCount),
            float(activeDaysLast7Days),
            float(activitySuggestionsSentLast7Days),
            float(activitySuggestionInteractLast7Days),
        ],
        dtype=float,
    )


def gamma_from_n(n: int) -> float:
    n = int(n)
    if n <= 1:
        return 0.0
    return (n - 1) / n


def ewma_gamma(values, gamma: float | None = None, *, empty: float = 0.0) -> float:
    """Normalized discounted average (most recent last); NaN keeps its slot."""
    vals = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(vals)
    n = int(vals.size)
    if n == 0 or not bool(finite.any()):
        return float(empty)
    g = gamma_from_n(n) if gamma is None else float(gamma)
    weights = g ** np.arange(n - 1, -1, -1, dtype=float)
    w = weights[finite]
    return float(np.dot(w, vals[finite]) / w.sum())


def _past_slot_stream(M, d, t):
    M = np.asarray(M, dtype=float).reshape(N_RL_DAYS, -1)
    return [
        float(M[i, j])
        for i in range(N_RL_DAYS)
        for j in range(N_RL_SLOTS)
        if (i, j) < (d, t)
    ]


def _past_day_stream(M, d, col):
    M = np.asarray(M, dtype=float).reshape(N_RL_DAYS, -1)
    return [float(M[i, col]) for i in range(N_RL_DAYS) if i < d]


def summarize_mediators_ewma(M_Y, M_E, d, t) -> np.ndarray:
    """Length-5 within-week EWMA summaries visible strictly before ``(d, t)``.

    Order: anticipated affect (day-level), 4-h step count (slot-level),
    page view (slot-level), Fitbit wear (day-level), daily check-in
    (day-level). A stream with no finite past value summarizes to 0, the
    model-scale centre, exactly as in the validated simulation.
    """
    return np.array(
        [
            ewma_gamma(_past_day_stream(M_Y, d, 2)),
            ewma_gamma(_past_slot_stream(M_Y, d, t)),
            ewma_gamma(_past_slot_stream(M_E, d, t)),
            ewma_gamma(_past_day_stream(M_E, d, 2)),
            ewma_gamma(_past_day_stream(M_E, d, 3)),
        ],
        dtype=float,
    )


def _time_features(d, t):
    """Saturday indicator and afternoon indicator for zero-based ``(d, t)``."""
    weekday_vs_weekend = 1.0 if int(d) >= N_RL_DAYS - 1 else 0.0
    return weekday_vs_weekend, float(t)


# ---------------------------------------------------------------------------
# Walking Q feature map (27-d)
# ---------------------------------------------------------------------------
def build_phi_state(state, d, t, *, b_hat=0.0, b_tilde=0.0) -> np.ndarray:
    E_w = float(state["E_w"])
    d_feat, t_feat = _time_features(d, t)
    M_ewma = summarize_mediators_ewma(state["M_Y"], state["M_E"], d, t)
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    if C_dt.size != N_RL_CONTEXT:
        raise ValueError(f"RL context C must have length {N_RL_CONTEXT}, got {C_dt.size}")
    base = np.array([1.0, d_feat, t_feat, E_w, float(b_hat), float(b_tilde)])
    return np.concatenate([base, M_ewma, C_dt])


def action_interact_vec(E_w, b_hat, b_tilde, C_dt, d, t) -> np.ndarray:
    d_feat, t_feat = _time_features(d, t)
    return np.concatenate(
        [
            [1.0, float(E_w), float(b_hat), float(b_tilde)],
            np.asarray(C_dt, dtype=float).ravel(),
            [d_feat, t_feat],
        ]
    )


def build_phi_action(b_hat, b_tilde, state, d, t, action) -> np.ndarray:
    """``phi(S_{w,d,t}, A)`` of length 27."""
    state_part = build_phi_state(state, d, t, b_hat=b_hat, b_tilde=b_tilde)
    action_block = float(action) * action_interact_vec(
        state["E_w"], b_hat, b_tilde, state["C"], d, t
    )
    return np.concatenate([state_part, action_block])


def phi_state_feature_names() -> list[str]:
    return list(PHI_STATE_BASE_NAMES) + list(MEDIATOR_SUMMARY_NAMES) + list(RL_CONTEXT_NAMES)


def phi_action_feature_names() -> list[str]:
    action = ["A", "A*E_w", "A*b_hat", "A*b_tilde"]
    action += [f"A*{name}" for name in RL_CONTEXT_NAMES]
    action += [f"A*{name}" for name in ACTION_TIME_NAMES]
    return phi_state_feature_names() + action


P_RL = len(phi_action_feature_names())


def gamma_dt_matrix(gamma_bar: float) -> np.ndarray:
    """Per-slot discounts: 1 within week, ``gamma_bar`` at the terminal slot."""
    gamma_dt = np.ones((N_RL_DAYS, N_RL_SLOTS), dtype=float)
    gamma_dt[TERMINAL_D, TERMINAL_T] = float(gamma_bar)
    return gamma_dt


# ---------------------------------------------------------------------------
# Action probability
# ---------------------------------------------------------------------------
def ensemble_action_prob(phi_1, phi_0, betas) -> float:
    """Share of ensemble members preferring action 1 (validated 'fraction')."""
    phi_1 = np.asarray(phi_1, dtype=float)
    phi_0 = np.asarray(phi_0, dtype=float)
    if not (np.all(np.isfinite(phi_1)) and np.all(np.isfinite(phi_0))):
        raise NonFiniteFeatures("non-finite walking features at decision time")
    if not betas:
        raise ValueError("ensemble_action_prob requires at least one draw")
    B = np.stack([np.asarray(beta, dtype=float) for beta in betas], axis=0)
    if not np.all(np.isfinite(B)):
        raise ValueError("non-finite RLSVI ensemble parameters")
    adv = B @ (phi_1 - phi_0)
    return float(np.mean(adv > 0.0))


def clip_prob(pi_hat, epsilon_0) -> float:
    pi_hat = float(pi_hat)
    if not np.isfinite(pi_hat):
        raise ValueError(f"non-finite randomization probability {pi_hat!r}")
    return float(np.clip(pi_hat, epsilon_0, 1.0 - epsilon_0))


# ---------------------------------------------------------------------------
# Linear algebra with explicit diagnostics
# ---------------------------------------------------------------------------
def spd_inverse(A, info: dict | None = None):
    """SPD inverse. Records the path taken in ``info`` (cholesky/jitter/pinv)."""
    A = np.asarray(A, dtype=float)
    if not np.all(np.isfinite(A)):
        raise np.linalg.LinAlgError("non-finite matrix passed to spd_inverse")
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    if info is not None:
        try:
            info["cond"] = float(np.linalg.cond(A))
        except np.linalg.LinAlgError:
            info["cond"] = float("inf")
    try:
        L = np.linalg.cholesky(A)
        Linv = np.linalg.inv(L)
        out = Linv.T @ Linv
        if info is not None:
            info["path"] = "cholesky"
        return 0.5 * (out + out.T)
    except np.linalg.LinAlgError:
        pass
    diag = np.diagonal(A)
    base = float(np.mean(np.abs(diag))) if diag.size else 1.0
    if not np.isfinite(base) or base <= 0.0:
        base = 1.0
    jitter = base * 1e-10
    I = np.eye(n)
    for _ in range(10):
        try:
            L = np.linalg.cholesky(A + jitter * I)
            Linv = np.linalg.inv(L)
            out = Linv.T @ Linv
            if info is not None:
                info["path"] = "jitter"
                info["jitter"] = float(jitter)
            return 0.5 * (out + out.T)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    out = np.linalg.pinv(A)
    if info is not None:
        info["path"] = "pinv"
    return 0.5 * (out + out.T)


def _eb_neg_marginal_loglik(log_s2, XtX, Xty, yty, n, Sigma0_inv, Sigma0_inv_mu0):
    rho = float(np.exp(-log_s2))
    A = Sigma0_inv + rho * XtX
    sign, logdetA = np.linalg.slogdet(A)
    if sign <= 0 or not np.isfinite(logdetA):
        return np.inf
    m = Sigma0_inv_mu0 + rho * Xty
    try:
        Ainv_m = np.linalg.solve(A, m)
    except np.linalg.LinAlgError:
        return np.inf
    ll = -0.5 * logdetA + 0.5 * n * np.log(rho) - 0.5 * rho * yty + 0.5 * float(m @ Ainv_m)
    return -ll


def empirical_bayes_sigma2(X, y, mu_0, Sigma_0, fallback=1.0, info: dict | None = None):
    """Marginal-likelihood scalar noise variance; ``info`` records the reason."""
    info = {} if info is None else info
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] == 0 or y.size != X.shape[0]:
        info["reason"] = "no_rows"
        return float(fallback)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    info["n_dropped_nonfinite"] = int((~mask).sum())
    X, y = X[mask], y[mask]
    n = X.shape[0]
    if n == 0:
        info["reason"] = "no_finite_rows"
        return float(fallback)
    mu_0 = np.asarray(mu_0, dtype=float).ravel()
    Sigma_0 = np.asarray(Sigma_0, dtype=float)
    try:
        Sigma0_inv = np.linalg.inv(Sigma_0)
    except np.linalg.LinAlgError:
        info["reason"] = "singular_prior"
        return float(fallback)
    XtX, Xty, yty = X.T @ X, X.T @ y, float(y @ y)
    Sigma0_inv_mu0 = Sigma0_inv @ mu_0
    resid0 = y - X @ mu_0
    scale = max(float(np.mean(resid0**2)), 1e-12)
    log_c = np.log(scale)
    lo = max(log_c - EB_SIGMA2_LOG_HALFWIDTH, np.log(EB_SIGMA2_MIN))
    hi = log_c + EB_SIGMA2_LOG_HALFWIDTH
    if hi <= lo:
        hi = lo + EB_SIGMA2_LOG_HALFWIDTH
    res = minimize_scalar(
        _eb_neg_marginal_loglik,
        bounds=(lo, hi),
        method="bounded",
        args=(XtX, Xty, yty, n, Sigma0_inv, Sigma0_inv_mu0),
        options={"maxiter": EB_MAXITER},
    )
    info["optimizer_success"] = bool(getattr(res, "success", True))
    if not np.isfinite(res.fun):
        info["reason"] = "nonfinite_objective"
        return float(fallback)
    s2 = float(np.exp(res.x))
    if not np.isfinite(s2) or s2 <= 0.0:
        info["reason"] = "invalid_estimate"
        return float(fallback)
    info["at_lower_bound"] = bool(s2 <= EB_SIGMA2_MIN * 1.0001)
    info["reason"] = "ok"
    return max(s2, EB_SIGMA2_MIN)


def empirical_bayes_sigma2_ensemble(X, targets_per_b, mu_0, Sigma_0, fallback=1.0):
    """EB ``sigma2`` on the ensemble-mean TD target. Returns ``(sigma2, info)``."""
    info: dict[str, Any] = {"fallback": float(fallback)}
    if not targets_per_b:
        info.update(reason="no_targets", used_fallback=True)
        return float(fallback), info
    y_bar = np.mean(np.stack(targets_per_b), axis=0)
    s2 = empirical_bayes_sigma2(X, y_bar, mu_0, Sigma_0, fallback, info)
    info["used_fallback"] = info.get("reason") != "ok"
    info["sigma2"] = float(s2)
    return float(s2), info


def noise_cholesky(Sigma) -> np.ndarray:
    """Cholesky factor of a symmetrized, eigenvalue-floored covariance."""
    S = 0.5 * (np.asarray(Sigma, dtype=float) + np.asarray(Sigma, dtype=float).T)
    min_eig = float(np.linalg.eigvalsh(S).min())
    if min_eig < 1e-6:
        S = S + (1e-6 - min_eig) * np.eye(S.shape[0])
    return np.linalg.cholesky(S)


def compute_rlsvi_betas(Phi, targets_per_b, mu_0, Sigma_0, sigma2, rho, xi_prev, rng):
    """Closed-form RLSVI posterior with persistent, standardized ensemble noise.

    Each member keeps a standardized noise state ``xi ~ N(0, I)`` with
    ``xi_w = rho * xi_{w-1} + sqrt(1 - rho^2) * eps``; the perturbation is
    ``z_w = L_w xi_w`` with ``L_w L_w^T = Sigma_w``, so every week's
    perturbation has exactly the posterior covariance. ``rho`` is the
    persistence; the handler passes the weekly discount ``gamma_bar``.

    Returns ``(betas, xi_new, diagnostics)``. Raises on non-finite input rather
    than letting NaN propagate into the ensemble.
    """
    rho = float(rho)
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"noise persistence must be in [0, 1), got {rho!r}")
    Phi = np.asarray(Phi, dtype=float)
    if not np.all(np.isfinite(Phi)):
        raise ValueError("non-finite rows reached the RLSVI design")
    for y_b in targets_per_b:
        if not np.all(np.isfinite(y_b)):
            raise ValueError("non-finite TD targets reached the RLSVI update")
    if not (np.isfinite(sigma2) and sigma2 > 0):
        raise ValueError(f"invalid sigma2 {sigma2!r}")
    diag: dict[str, Any] = {"n_rows": int(Phi.shape[0])}
    B = len(targets_per_b)
    Sigma_0_inv = np.linalg.inv(Sigma_0)
    if Phi.shape[0] == 0:
        Sigma_post = np.asarray(Sigma_0, dtype=float).copy()
        diag["inverse"] = {"path": "prior_only"}
    else:
        inv_info: dict[str, Any] = {}
        Sigma_post = spd_inverse(Sigma_0_inv + (1.0 / sigma2) * (Phi.T @ Phi), inv_info)
        diag["inverse"] = inv_info
    min_eig = float(np.linalg.eigvalsh(0.5 * (Sigma_post + Sigma_post.T)).min())
    diag["noise_min_eig"] = min_eig
    if min_eig < 1e-6:
        diag["noise_eig_floor_applied"] = True
    precomp = Sigma_0_inv @ mu_0
    L = noise_cholesky(Sigma_post)
    scale = float(np.sqrt(1.0 - rho * rho))
    betas, z_new = [], []
    for b in range(B):
        if Phi.shape[0] == 0:
            mu_b = np.asarray(mu_0, dtype=float).copy()
        else:
            mu_b = Sigma_post @ (precomp + (1.0 / sigma2) * (Phi.T @ targets_per_b[b]))
        xi_b = rho * np.asarray(xi_prev[b], dtype=float) + scale * rng.standard_normal(L.shape[0])
        betas.append(mu_b + L @ xi_b)
        z_new.append(xi_b)
    diag["max_abs_beta"] = float(np.max(np.abs(np.asarray(betas))))
    return betas, z_new, diag


def ridge_posterior_mean(X, y, mu_0, Sigma_0, sigma2) -> np.ndarray:
    """Posterior mean of a Gaussian linear model (linear solve, no inverse of data)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    p = X.shape[1] if X.ndim == 2 else len(mu_0)
    mu_0 = np.asarray(mu_0, dtype=float).ravel()
    Sigma_0 = np.asarray(Sigma_0, dtype=float)
    if mu_0.shape != (p,) or Sigma_0.shape != (p, p) or not (np.isfinite(sigma2) and sigma2 > 0):
        raise ValueError("invalid Gaussian prior for ridge_posterior_mean")
    precision = np.linalg.inv(Sigma_0)
    if X.shape[0] == 0:
        return mu_0.copy()
    return np.linalg.solve(precision + X.T @ X / sigma2, precision @ mu_0 + X.T @ y / sigma2)


def default_gaussian_prior(p: int, sigma2: float = 1.0) -> dict:
    return {"mu_0": np.zeros(p), "Sigma_0": np.eye(p), "sigma2": float(sigma2)}


# ---------------------------------------------------------------------------
# Reward redistribution (V6)
# ---------------------------------------------------------------------------
def build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action) -> np.ndarray:
    """Stage-1 feature for daily mediators (no state-only slot column)."""
    E_w = float(state["E_w"])
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    _, slot_pm = _time_features(d, t)
    med_ctx = np.concatenate([summarize_mediators_ewma(state["M_Y"], state["M_E"], d, t), C_dt])
    action_ctx = float(action) * np.concatenate(
        [[1.0, E_w, float(b_hat), float(b_tilde)], C_dt, [slot_pm]]
    )
    return np.concatenate([[1.0, E_w, float(b_hat), float(b_tilde)], med_ctx, action_ctx])


def build_foursc_stage1_phi(b_hat, b_tilde, state, d, t, action) -> np.ndarray:
    """Stage-1b slot-level step-count feature: controls + A x [1, slot_pm]."""
    E_w = float(state["E_w"])
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    weekday, slot_pm = _time_features(d, t)
    controls = np.concatenate([[1.0, weekday, slot_pm, E_w, float(b_hat), float(b_tilde)], C_dt])
    return np.concatenate([controls, float(action) * np.array([1.0, slot_pm])])


def stage1_dims() -> dict[str, int]:
    probe = {
        "E_w": 0.0,
        "M_Y": np.zeros((N_RL_DAYS, 3)),
        "M_E": np.zeros((N_RL_DAYS, 4)),
        "C": np.zeros(N_RL_CONTEXT),
    }
    p_day = build_daily_mediator_phi(0.0, 0.0, probe, 0, 0, 0).size
    p_slot = build_foursc_stage1_phi(0.0, 0.0, probe, 0, 0, 0).size
    return {"AA": p_day, "FW": p_day, "PJ": p_day, "SC": p_slot}


def fit_stage1(
    weeks: list[int],
    A_hist,
    b_hat_hist,
    b_tilde_hist,
    get_state: Callable[[int, int, int], dict],
    get_full: Callable[[int], tuple[np.ndarray, np.ndarray]],
    priors: dict | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit Stage-1 daily (AA/FW/PJ) and Stage-1b slot SC decompositions.

    Rows whose target or features are non-finite are dropped and counted.
    """
    priors = priors or {}
    dims = stage1_dims()
    etas: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {}
    for name, (matrix, col) in _DAILY_TARGETS.items():
        rows, y, dropped = [], [], 0
        for k in weeks:
            full = get_full(k)
            target = np.asarray(full[matrix], dtype=float)
            for d in range(N_RL_DAYS):
                row = sum(
                    build_daily_mediator_phi(
                        b_hat_hist[k], b_tilde_hist[k], get_state(k, d, t), d, t, A_hist[k, d, t]
                    )
                    for t in range(N_RL_SLOTS)
                )
                yt = float(target[d, col])
                if np.isfinite(yt) and np.all(np.isfinite(row)):
                    rows.append(row)
                    y.append(yt)
                else:
                    dropped += 1
        prior = priors.get(name) or default_gaussian_prior(dims[name])
        X = np.asarray(rows, dtype=float) if rows else np.empty((0, dims[name]))
        etas[name] = ridge_posterior_mean(X, y, prior["mu_0"], prior["Sigma_0"], prior["sigma2"])
        info[name] = {"n_rows": len(rows), "n_dropped": dropped}
    rows, y, dropped = [], [], 0
    for k in weeks:
        M_Y = np.asarray(get_full(k)[0], dtype=float)
        for d in range(N_RL_DAYS):
            for t in range(N_RL_SLOTS):
                row = build_foursc_stage1_phi(
                    b_hat_hist[k], b_tilde_hist[k], get_state(k, d, t), d, t, A_hist[k, d, t]
                )
                yt = float(M_Y[d, t])
                if np.isfinite(yt) and np.all(np.isfinite(row)):
                    rows.append(row)
                    y.append(yt)
                else:
                    dropped += 1
    prior = priors.get("SC") or default_gaussian_prior(dims["SC"])
    X = np.asarray(rows, dtype=float) if rows else np.empty((0, dims["SC"]))
    etas["SC"] = ridge_posterior_mean(X, y, prior["mu_0"], prior["Sigma_0"], prior["sigma2"])
    info["SC"] = {"n_rows": len(rows), "n_dropped": dropped}
    for name, eta in etas.items():
        if not np.all(np.isfinite(eta)):
            raise ValueError(f"non-finite Stage-1 coefficients for {name}")
    return etas, info


def stage1_shares(etas, b_hat, b_tilde, state, d, t, action) -> np.ndarray:
    return np.array(
        [
            build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action) @ etas["AA"],
            build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action) @ etas["FW"],
            build_daily_mediator_phi(b_hat, b_tilde, state, d, t, action) @ etas["PJ"],
            build_foursc_stage1_phi(b_hat, b_tilde, state, d, t, action) @ etas["SC"],
        ],
        dtype=float,
    )


def build_redistribution_phi(b_hat, b_tilde, state, d, t, shares, M_E_full) -> np.ndarray:
    """Stage-2 feature (20-d): state, EWMAs, C, realized page view, 4 shares."""
    E_w = float(state["E_w"])
    C_dt = np.asarray(state["C"], dtype=float).ravel()
    d_feat, _ = _time_features(d, t)
    next_me = float(np.asarray(M_E_full, dtype=float)[d, t])
    return np.concatenate(
        [
            [1.0, d_feat, E_w, float(b_hat), float(b_tilde)],
            summarize_mediators_ewma(state["M_Y"], state["M_E"], d, t),
            C_dt,
            [next_me],
            np.asarray(shares, dtype=float).ravel(),
        ]
    )


P_STAGE2 = 5 + 5 + N_RL_CONTEXT + 1 + len(STAGE1_NAMES)


def build_v6_training_data(
    k_cur: int,
    A_hist,
    b_hat_hist,
    b_tilde_hist,
    e_hat_hist,
    betas_eval,
    betas_select,
    gamma_dt,
    gamma_bar: float,
    get_state: Callable[[int, int, int], dict],
    has_state: Callable[[int, int, int], bool],
    get_full: Callable[[int], tuple[np.ndarray, np.ndarray] | None],
    stage1_priors: dict | None = None,
    stage2_prior: dict | None = None,
) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """TD design and per-member targets for V6 on completed weeks ``0..k_cur-1``.

    A completed week is usable only if its belief, the next week's belief,
    both perceived-utility values, its full realized mediators, and all 12
    decision states exist. Unusable weeks are excluded (never zero-filled) and
    reported in ``info["excluded_weeks"]``.
    """
    info: dict[str, Any] = {"excluded_weeks": {}, "dropped_rows": 0}
    usable: list[int] = []
    for kp in range(int(k_cur)):
        reasons = []
        if not np.isfinite(b_hat_hist[kp]) or not np.isfinite(b_tilde_hist[kp]):
            reasons.append("belief_k")
        if not np.isfinite(b_hat_hist[kp + 1]) or not np.isfinite(b_tilde_hist[kp + 1]):
            reasons.append("belief_k+1")
        if not np.isfinite(e_hat_hist[kp]) or not np.isfinite(e_hat_hist[kp + 1]):
            reasons.append("e_hat")
        if get_full(kp) is None:
            reasons.append("full_mediators")
        if not all(has_state(kp, d, t) for d in range(N_RL_DAYS) for t in range(N_RL_SLOTS)):
            reasons.append("decision_states")
        if reasons:
            info["excluded_weeks"][int(kp)] = reasons
        else:
            usable.append(kp)
    info["usable_weeks"] = list(usable)
    p = len(betas_eval[0])
    B = len(betas_eval)
    if not usable:
        return np.empty((0, p)), [np.empty(0) for _ in range(B)], info

    etas, s1_info = fit_stage1(
        usable, A_hist, b_hat_hist, b_tilde_hist, get_state, lambda k: get_full(k), stage1_priors
    )
    info["stage1"] = s1_info

    def chi(kp, d, t):
        state = get_state(kp, d, t)
        a = A_hist[kp, d, t]
        shares = stage1_shares(etas, b_hat_hist[kp], b_tilde_hist[kp], state, d, t, a)
        return build_redistribution_phi(
            b_hat_hist[kp], b_tilde_hist[kp], state, d, t, shares, get_full(kp)[1]
        )

    chis: dict[int, np.ndarray] = {}
    rows2, y2 = [], []
    for kp in usable:
        week_chi = np.stack([chi(kp, d, t) for d in range(N_RL_DAYS) for t in range(N_RL_SLOTS)])
        if not np.all(np.isfinite(week_chi)):
            info["excluded_weeks"][int(kp)] = ["stage2_features"]
            continue
        chis[kp] = week_chi
        target = float(b_hat_hist[kp + 1]) + float(gamma_bar) * float(e_hat_hist[kp + 1]) - float(e_hat_hist[kp])
        rows2.append(week_chi.sum(axis=0))
        y2.append(target)
    info["stage2_n_weeks"] = len(rows2)
    if not rows2:
        return np.empty((0, p)), [np.empty(0) for _ in range(B)], info
    prior2 = stage2_prior or default_gaussian_prior(P_STAGE2)
    eta2 = ridge_posterior_mean(np.asarray(rows2), y2, prior2["mu_0"], prior2["Sigma_0"], prior2["sigma2"])
    if not np.all(np.isfinite(eta2)):
        raise ValueError("non-finite Stage-2 coefficients")
    info["stage2_eta"] = eta2.tolist()
    weekly_targets = np.asarray(y2)
    fitted_week = np.asarray(rows2) @ eta2
    info["stage2_rmse"] = float(np.sqrt(np.mean((weekly_targets - fitted_week) ** 2)))
    info["weekly_target_mean"] = float(np.mean(weekly_targets))

    Phi_rows: list[np.ndarray] = []
    targets: list[list[float]] = [[] for _ in range(B)]
    slot_rewards = []
    for kp, week_chi in chis.items():
        rewards = week_chi @ eta2
        slot_rewards.extend(rewards.tolist())
        for s, (d, t) in enumerate((d, t) for d in range(N_RL_DAYS) for t in range(N_RL_SLOTS)):
            nxt = _next_slot(d, t)
            if nxt is None:
                kn, dn, tn = kp + 1, FIRST_D, FIRST_T
                bh, bt = b_hat_hist[kp + 1], b_tilde_hist[kp + 1]
            else:
                kn, (dn, tn) = kp, nxt
                bh, bt = b_hat_hist[kp], b_tilde_hist[kp]
            if not has_state(kn, dn, tn):
                info["dropped_rows"] += 1
                continue
            phi = build_phi_action(b_hat_hist[kp], b_tilde_hist[kp], get_state(kp, d, t), d, t, A_hist[kp, d, t])
            nstate = get_state(kn, dn, tn)
            p1 = build_phi_action(bh, bt, nstate, dn, tn, 1)
            p0 = build_phi_action(bh, bt, nstate, dn, tn, 0)
            if not (np.all(np.isfinite(phi)) and np.all(np.isfinite(p1)) and np.all(np.isfinite(p0))):
                info["dropped_rows"] += 1
                continue
            Phi_rows.append(phi)
            g = float(gamma_dt[d, t])
            for b in range(B):
                a_star = 1 if (p1 @ betas_select[b] > p0 @ betas_select[b]) else 0
                q = (p1 if a_star else p0) @ betas_eval[b]
                targets[b].append(float(rewards[s]) + g * float(q))
    info["slot_reward_mean"] = float(np.mean(slot_rewards)) if slot_rewards else float("nan")
    info["slot_reward_sd"] = float(np.std(slot_rewards)) if slot_rewards else float("nan")
    Phi = np.asarray(Phi_rows, dtype=float) if Phi_rows else np.empty((0, p))
    return Phi, [np.asarray(tb, dtype=float) for tb in targets], info


def _next_slot(d, t):
    if t + 1 < N_RL_SLOTS:
        return (d, t + 1)
    if d + 1 < N_RL_DAYS:
        return (d + 1, 0)
    return None


# ---------------------------------------------------------------------------
# Policy selection and validation
# ---------------------------------------------------------------------------
def wanted_policy_key(k: int, d: int) -> int:
    """Validated protocol: Monday uses last week's fit, Tue–Sat this week's."""
    if k <= 0:
        return 0
    return int(k - 1) if int(d) == 0 else int(k)


def betas_for_action(model_params: dict, k: int, d: int) -> tuple[list[np.ndarray], int, int]:
    """Return ``(betas, key_used, staleness)`` from the validated-policy pointer.

    Never silently falls back to the week-0 prior: if no validated policy at or
    before the wanted key exists, raises ``KeyError``.
    """
    stores = model_params.get("betas_store", {})
    wanted = wanted_policy_key(k, d)
    active = int(model_params.get("active_policy_key", 0))
    key = min(wanted, active)
    if str(key) not in stores:
        raise KeyError(f"validated policy key {key} missing from betas_store")
    return [np.asarray(beta, dtype=float) for beta in stores[str(key)]], key, wanted - key


def validate_policy(
    betas: list[np.ndarray],
    decision_phis: list[tuple[np.ndarray, np.ndarray]],
    *,
    epsilon_0: float,
    previous_betas: list[np.ndarray] | None = None,
    max_saturated_fraction: float = 0.95,
    max_mean_prob_change: float = 0.5,
) -> tuple[bool, dict[str, Any]]:
    """Checks a candidate ensemble before it may be served."""
    report: dict[str, Any] = {}
    Bm = np.asarray(betas, dtype=float)
    if not np.all(np.isfinite(Bm)):
        return False, {"reason": "nonfinite_beta"}
    report["max_abs_beta"] = float(np.max(np.abs(Bm)))
    if not decision_phis:
        report["reason"] = "ok_no_probe_states"
        return True, report
    pis = np.array([ensemble_action_prob(p1, p0, betas) for p1, p0 in decision_phis])
    report["mean_pi_hat"] = float(np.mean(pis))
    report["saturated_fraction"] = float(np.mean((pis <= 0.0) | (pis >= 1.0)))
    if report["saturated_fraction"] > max_saturated_fraction:
        return False, {**report, "reason": "policy_saturated"}
    if previous_betas is not None:
        prev = np.array([ensemble_action_prob(p1, p0, previous_betas) for p1, p0 in decision_phis])
        report["mean_abs_prob_change"] = float(np.mean(np.abs(np.clip(pis, epsilon_0, 1 - epsilon_0) - np.clip(prev, epsilon_0, 1 - epsilon_0))))
        if report["mean_abs_prob_change"] > max_mean_prob_change:
            return False, {**report, "reason": "policy_jump"}
    report["reason"] = "ok"
    return True, report
