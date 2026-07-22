"""Self-contained RL feature and RLSVI helpers for ``RL_agents``.

This module contains the subset of the project-level ``algorithm_helpers`` used
by the deployment-facing RL handler. Keeping these helpers inside
``RL_agents`` lets the package run as a standalone codespace.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def gamma_dt_matrix(gamma_bar: float) -> np.ndarray:
    """Per-slot discount matrix ``gamma_{d,t}`` (mirrors ``experiment._gamma_dt_micro``).

    For ``gamma_bar = 0`` (myopic): non-terminal slots use ``gamma = 1``; the
    terminal slot uses ``gamma = 0``. Otherwise every slot uses
    ``gamma_bar ** (1 / (n_rl_days * n_rl_slots))``.
    """
    if float(gamma_bar) == 0.0:
        gamma_dt = np.ones((N_RL_DAYS, N_RL_SLOTS), dtype=float)
        gamma_dt[TERMINAL_D, TERMINAL_T] = 0.0
        return gamma_dt
    return (float(gamma_bar) ** (1.0 / (N_RL_DAYS * N_RL_SLOTS))) * np.ones(
        (N_RL_DAYS, N_RL_SLOTS), dtype=float
    )


def spd_inverse(A):
    """Symmetric positive-definite inverse robust to ill-conditioning."""
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    try:
        L = np.linalg.cholesky(A)
        Linv = np.linalg.inv(L)
        out = Linv.T @ Linv
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
            return 0.5 * (out + out.T)
        except np.linalg.LinAlgError:
            jitter *= 10.0

    out = np.linalg.pinv(A)
    return 0.5 * (out + out.T)


def ensemble_action_prob(phi_1, phi_0, betas):
    """Fraction of ensemble members preferring action 1 over action 0."""
    B = len(betas)
    votes = sum(1 for beta_b in betas if phi_1 @ beta_b > phi_0 @ beta_b)
    return votes / B


def clip_prob(pi_hat, epsilon_0):
    """Clip randomization probability to ``[epsilon_0, 1 - epsilon_0]``."""
    return np.clip(pi_hat, epsilon_0, 1.0 - epsilon_0)


def betas_for_action(model_params: dict, k: int, d: int) -> list[np.ndarray]:
    """Select the ``betas_store`` ensemble for week ``k`` and day ``d``."""
    stores = model_params.get("betas_store", {})
    if k <= 0:
        key = "0"
    elif d >= 1 and str(k) in stores:
        key = str(k)
    else:
        key = str(max(k - 1, 0))
    if key not in stores:
        key = "0"
    return [np.asarray(beta, dtype=float) for beta in stores[key]]


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
    ll = (
        -0.5 * logdetA
        + 0.5 * n * np.log(rho)
        - 0.5 * rho * yty
        + 0.5 * float(m @ Ainv_m)
    )
    return -ll


def empirical_bayes_sigma2(X, y, mu_0, Sigma_0, fallback=1.0):
    """Empirical-Bayes scalar noise variance for Bayesian linear regression."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] == 0 or y.size != X.shape[0]:
        return float(fallback)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    n = X.shape[0]
    if n == 0:
        return float(fallback)

    mu_0 = np.asarray(mu_0, dtype=float).ravel()
    Sigma_0 = np.asarray(Sigma_0, dtype=float)
    try:
        Sigma0_inv = np.linalg.inv(Sigma_0)
    except np.linalg.LinAlgError:
        return float(fallback)

    XtX = X.T @ X
    Xty = X.T @ y
    yty = float(y @ y)
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
    )
    if not np.isfinite(res.fun):
        return float(fallback)
    s2 = float(np.exp(res.x))
    if not np.isfinite(s2) or s2 <= 0.0:
        return float(fallback)
    return max(s2, EB_SIGMA2_MIN)


def empirical_bayes_sigma2_ensemble(X, targets_per_b, mu_0, Sigma_0, fallback=1.0):
    """Average empirical-Bayes ``sigma2`` across ensemble target vectors.

    Returns ``(sigma2, info)`` where ``info["used_fallback"]`` is True when no
    ensemble member produced a finite estimate and ``fallback`` was retained.
    """
    vals = []
    B = len(targets_per_b)
    for y_b in targets_per_b:
        s2 = empirical_bayes_sigma2(X, y_b, mu_0, Sigma_0, fallback)
        if np.isfinite(s2):
            vals.append(s2)
    used_fallback = len(vals) == 0
    sigma2_out = float(np.mean(vals)) if vals else float(fallback)
    info = {
        "used_fallback": used_fallback,
        "partial_fallback": (not used_fallback and len(vals) < B),
        "n_finite": len(vals),
        "n_ensemble": B,
        "sigma2": sigma2_out,
        "fallback": float(fallback),
    }
    return sigma2_out, info


def compute_rlsvi_betas(Phi, targets_per_b, mu_0, Sigma_0, sigma2, gamma_bar, z_prev, rng):
    """Closed-form RLSVI posterior update with discounted ensemble noise."""
    B = len(targets_per_b)
    Sigma_0_inv = np.linalg.inv(Sigma_0)

    if Phi.shape[0] == 0:
        Sigma_post = Sigma_0.copy()
    else:
        Sigma_post = spd_inverse(Sigma_0_inv + (1.0 / sigma2) * (Phi.T @ Phi))

    noise_cov = (1.0 - gamma_bar**2) * Sigma_post
    noise_cov = 0.5 * (noise_cov + noise_cov.T)
    min_eig = np.linalg.eigvalsh(noise_cov).min()
    if min_eig < 1e-6:
        noise_cov += (1e-6 - min_eig) * np.eye(noise_cov.shape[0])
    precomp = Sigma_0_inv @ mu_0

    betas, z_new = [], []
    for b in range(B):
        if Phi.shape[0] == 0:
            mu_b = mu_0.copy()
        else:
            mu_b = Sigma_post @ (
                precomp + (1.0 / sigma2) * (Phi.T @ targets_per_b[b])
            )
        try:
            z_b = rng.multivariate_normal(gamma_bar * z_prev[b], noise_cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(noise_cov + 1e-4 * np.eye(noise_cov.shape[0]))
            z_b = gamma_bar * z_prev[b] + L @ rng.standard_normal(noise_cov.shape[0])
        betas.append(mu_b + z_b)
        z_new.append(z_b)

    return betas, z_new


def build_rl_context_vector(
    *,
    yesterdayStepCount,
    stepCountLast7DaysEma,
    prior2HourStepCountAgent,
    activityCompletedLast7Days,
    activeDaysLast7Days,
    activitySuggestionsSentLast7Days,
    salienceMessageSentYesterday,
    activitySuggestionInteractLast7Days,
):
    """Per-decision context ``C`` for RLSVI."""
    return np.array(
        [
            float(yesterdayStepCount),
            float(stepCountLast7DaysEma),
            float(prior2HourStepCountAgent),
            float(activityCompletedLast7Days),
            float(activeDaysLast7Days),
            float(activitySuggestionsSentLast7Days),
            float(salienceMessageSentYesterday),
            float(activitySuggestionInteractLast7Days),
        ],
        dtype=float,
    )


def rl_context_feature_names() -> list[str]:
    """Names aligned with ``build_rl_context_vector`` output order."""
    return [
        "yesterday_step_count",
        "ema_step_count",
        "prior2hour_step_count",
        "previous7days_rpa",
        "active_status_fraction_7days",
        "recent_burden",
        "salience_yesterday",
        "walk_interaction_7d",
    ]


def rl_my_feature_names() -> list[str]:
    cols = ["fourSC_morning", "fourSC_afternoon", "anticipated_affect"]
    return [f"M_Y_day{d}_{name}" for d in range(1, N_RL_DAYS + 1) for name in cols]


def rl_me_feature_names() -> list[str]:
    cols = [
        "pageview_morning",
        "pageview_afternoon",
        "morning_fitbit_wear",
        "daily_survey_complete",
    ]
    return [f"M_E_day{d}_{name}" for d in range(1, N_RL_DAYS + 1) for name in cols]


def phi_state_feature_names() -> list[str]:
    """Names aligned with ``build_phi_state`` output order."""
    return [
        "intercept",
        "weekday_vs_weekend",
        "slot_pm",
        "E_w",
        "weekday_vs_weekend*E_w",
        "slot_pm*E_w",
        "b_hat",
        "weekday_vs_weekend*b_hat",
        "slot_pm*b_hat",
        "b_tilde",
    ] + rl_my_feature_names() + rl_me_feature_names() + rl_context_feature_names()


def phi_action_feature_names() -> list[str]:
    """Names aligned with ``build_phi_action`` output order."""
    action_context = [
        "A",
        "A*E_w",
        "A*b_hat",
        "A*weekday_vs_weekend",
        "A*slot_pm",
        "A*weekday_vs_weekend*E_w",
        "A*slot_pm*E_w",
        "A*weekday_vs_weekend*b_hat",
        "A*slot_pm*b_hat",
    ] + [f"A*{name}" for name in rl_context_feature_names()]
    return phi_state_feature_names() + action_context


def _iter_slots():
    for d in range(N_RL_DAYS):
        for t in range(N_RL_SLOTS):
            yield d, t


def _next_slot(d, t):
    """Successor of zero-based walking slot ``(d, t)``, or ``None`` at terminal."""
    if t + 1 < N_RL_SLOTS:
        return (d, t + 1)
    if d + 1 < N_RL_DAYS:
        return (d + 1, 0)
    return None


def _first_slot():
    return 0, 0


def _mask_mediators_for_slot(M_Y, M_E, d, t):
    """Zero mediators not yet observed at decision ``(d, t)``."""
    M_Y = np.asarray(M_Y, dtype=float).reshape(N_RL_DAYS, -1).copy()
    M_E = np.asarray(M_E, dtype=float).reshape(N_RL_DAYS, -1).copy()
    n_my, n_me = M_Y.shape[1], M_E.shape[1]
    for i in range(N_RL_DAYS):
        for j in range(n_my):
            if j < N_RL_SLOTS:
                if not ((i, j) < (d, t)):
                    M_Y[i, j] = 0.0
            elif not (i < d):
                M_Y[i, j] = 0.0
        for j in range(n_me):
            if j < N_RL_SLOTS:
                if not ((i, j) < (d, t)):
                    M_E[i, j] = 0.0
            elif not (i < d):
                M_E[i, j] = 0.0
    return M_Y, M_E


def _time_features(d, t):
    """Map zero-based walking slot ``(d, t)`` to weekend and slot features."""
    weekday_vs_weekend = 1.0 if int(d) >= N_RL_DAYS - 1 else 0.0
    slot_pm = float(t)
    return weekday_vs_weekend, slot_pm


def build_phi_state(state, d, t, *, b_hat=0.0, b_tilde=0.0):
    """State-only features shared by the walking-suggestion RLSVI model."""
    E_w = state["E_w"]
    d_feat, t_feat = _time_features(d, t)

    M_Y_m, M_E_m = _mask_mediators_for_slot(state["M_Y"], state["M_E"], d, t)
    C_dt = np.asarray(state["C"]).ravel()

    base = np.array(
        [
            1.0,
            d_feat,
            t_feat,
            E_w,
            d_feat * E_w,
            t_feat * E_w,
            b_hat,
            d_feat * b_hat,
            t_feat * b_hat,
            b_tilde,
        ]
    )
    med_ctx = np.concatenate([M_Y_m.ravel(), M_E_m.ravel(), C_dt])
    return np.concatenate([base, med_ctx])


def build_phi_action(b_hat, b_tilde, state, d, t, action):
    """Feature map ``phi(S_{w,d,t}, A_{w,d,t})`` for walking suggestions."""
    E_w = state["E_w"]
    b_w = b_hat

    d_feat, t_feat = _time_features(d, t)
    C_dt = np.asarray(state["C"]).ravel()
    state_part = build_phi_state(state, d, t, b_hat=b_hat, b_tilde=b_tilde)

    interact_vec = np.concatenate(
        [
            [
                1.0,
                E_w,
                b_w,
                d_feat,
                t_feat,
                d_feat * E_w,
                t_feat * E_w,
                d_feat * b_w,
                t_feat * b_w,
            ],
            C_dt,
        ]
    )
    action_block = float(action) * interact_vec

    return np.concatenate([state_part, action_block])


def build_rl_training_data(
    k_cur,
    A_hist,
    b_hat_hist,
    b_tilde_hist,
    betas_eval,
    betas_select,
    gamma_dt,
    get_state,
    phi_fn=None,
):
    """Build RLSVI design rows and per-ensemble TD targets for walking slots."""
    if phi_fn is None:
        phi_fn = build_phi_action

    B = len(betas_eval)
    Phi_rows = []
    targets = [[] for _ in range(B)]

    for kp in range(k_cur):
        bh_wp = b_hat_hist[kp]
        bt_wp = b_tilde_hist[kp]

        for d, t in _iter_slots():
            state_dt = get_state(kp, d, t)
            a_dt = A_hist[kp, d, t]
            Phi_rows.append(phi_fn(bh_wp, bt_wp, state_dt, d, t, a_dt))

            nxt = _next_slot(d, t)
            if nxt is not None:
                d_n, t_n = nxt
                state_n = get_state(kp, d_n, t_n)
                phi_1 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 1)
                phi_0 = phi_fn(bh_wp, bt_wp, state_n, d_n, t_n, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b] > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets[b].append(gamma_dt[d, t] * q_star)
            else:
                bh_n = b_hat_hist[kp + 1]
                bt_n = b_tilde_hist[kp + 1]
                R_next = float(bh_n) if np.isfinite(bh_n) else 0.0
                first_d, first_t = _first_slot()
                state_n = get_state(kp + 1, first_d, first_t)
                phi_1 = phi_fn(bh_n, bt_n, state_n, first_d, first_t, 1)
                phi_0 = phi_fn(bh_n, bt_n, state_n, first_d, first_t, 0)
                for b in range(B):
                    a_star = 1 if (phi_1 @ betas_select[b] > phi_0 @ betas_select[b]) else 0
                    q_star = (phi_1 if a_star else phi_0) @ betas_eval[b]
                    targets[b].append(
                        R_next + gamma_dt[TERMINAL_D, TERMINAL_T] * q_star
                    )

    if Phi_rows:
        Phi = np.array(Phi_rows)
    else:
        p = len(betas_eval[0])
        Phi = np.empty((0, p))

    targets_per_b = [np.array(t) for t in targets]
    return Phi, targets_per_b
