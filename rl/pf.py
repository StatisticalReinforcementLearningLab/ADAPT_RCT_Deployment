"""CAE belief-state particle filter (port of the testbed PF).

Working models, identical to the testbed (``algorithm_helpers`` /
``ParticleFilterRuntime.update_standard``):

* fourSC (18-d, ``PF_THETA_FOURSC_NAMES``), anticipated affect (13-d,
  ``PF_THETA_ANTIC_NAMES``): action-conditional mediator models whose
  ``CAE_avg_lastweek`` column is filled per particle;
* CAE transition ``[1, CAE_lag, fourSC_ewma, antic_ewma]`` (Mon-Sat EWMAs);
* short CAE ``[1, CAE]``.

Particle learning with per-particle sufficient statistics; parameters are
updated on weeks ``0..k-2`` and the week ``k-1`` likelihood is predictive.
The noise variances are refit by empirical Bayes each week on the pre-update
belief-mean trajectory.

Particle storage: ``y_hat[:, 0] = Y_1``; column ``k`` holds the belief draw for
``Y_k`` (CAE of week ``k-1``). A week can only be filtered when columns
``0..k-1`` exist (no gaps).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from RL_agents.rl.algorithm_helpers import N_RL_DAYS, N_RL_SLOTS, N_SERVER_C, empirical_bayes_sigma2, ewma_gamma, spd_inverse
from RL_agents.rl.week_data import WeekData

PF_THETA_FOURSC_NAMES = [
    "intercept", "stepCountNext4HourLag1", "yesterdayStepCount",
    "prior2HourStepCount", "activitySuggestionsSentLast7Days", "activitySuggestionInteractLast7Days",
    "activeDaysLast7Days", "isWeekend", "decisionTimeSlot", "perceivedUtilityLastWeek",
    "caeAverageLastWeek", "Ah", "Ah*yesterdayStepCount", "Ah*prior2HourStepCount",
    "Ah*perceivedUtilityLastWeek", "Ah*caeAverageLastWeek", "Ah*decisionTimeSlot",
]
PF_THETA_ANTIC_NAMES = [
    "intercept", "anticipated_affect_yesterday", "active_status_fraction_7days", "is_weekend",
    "perceived_utility_lastweek", "CAE_avg_lastweek", "recent_burden", "A0_morning", "A1_afternoon",
    "A0_morning_by_perceived_utility_lastweek", "A1_afternoon_by_perceived_utility_lastweek",
    "A0_morning_by_CAE_avg_lastweek", "A1_afternoon_by_CAE_avg_lastweek",
]
# The action does not interact with any 7-day summary in either mediator model.
PF_DIMS = {"fourSC": 17, "antic": 13, "CAE": 4, "CAE_short": 2}
_F = {n: i for i, n in enumerate(PF_THETA_FOURSC_NAMES)}
_A = {n: i for i, n in enumerate(PF_THETA_ANTIC_NAMES)}
_CAE_AR1_COL = 1


class PFStateMismatch(RuntimeError):
    """Stored particles do not line up with the requested week."""


# ---------------------------------------------------------------------------
# Design rows
# ---------------------------------------------------------------------------
def foursc_row(c_model: np.ndarray, *, lag1: float, d: int, t: int, pu: float, action: float) -> np.ndarray:
    """Testbed ``build_fourSC_features`` from the 5-entry model-scale context."""
    ysc, p2h, active, sug, inter = (c_model[i] for i in range(5))
    wkend = 1.0 if d >= N_RL_DAYS - 1 else 0.0
    a = float(action)
    return np.array([
        1.0, lag1, ysc, p2h, sug, inter, active, wkend, float(t), pu, 0.0,
        a, a * ysc, a * p2h, a * pu, 0.0, a * float(t),
    ], dtype=float)


def foursc_cae_delta(x: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(x)
    delta[_F["caeAverageLastWeek"]] = 1.0
    delta[_F["Ah*caeAverageLastWeek"]] = x[_F["Ah"]]
    return delta


def antic_row(*, antic_yesterday: float, active: float, d: int, pu: float, rb: float,
              ws_m: float, ws_a: float) -> np.ndarray:
    wkend = 1.0 if d >= N_RL_DAYS - 1 else 0.0
    return np.array([
        1.0, antic_yesterday, active, wkend, pu, 0.0, rb, ws_m, ws_a,
        ws_m * pu, ws_a * pu, 0.0, 0.0,
    ], dtype=float)


def antic_cae_delta(x: np.ndarray) -> np.ndarray:
    delta = np.zeros_like(x)
    delta[_A["CAE_avg_lastweek"]] = 1.0
    delta[_A["A0_morning_by_CAE_avg_lastweek"]] = x[_A["A0_morning"]]
    delta[_A["A1_afternoon_by_CAE_avg_lastweek"]] = x[_A["A1_afternoon"]]
    return delta


def cae_row(foursc12: np.ndarray, antic6: np.ndarray) -> np.ndarray:
    """``[1, 0, EWMA(fourSC Mon-Sat), EWMA(antic Mon-Sat)]`` (missing entries skipped; 0 if none)."""
    return np.array([1.0, 0.0, ewma_gamma(foursc12), ewma_gamma(antic6)])


class PFDesign:
    """Model-scale per-week PF inputs built from :class:`WeekData`."""

    def __init__(self, data: WeekData, std: dict, e_hat_hist: np.ndarray, y1_model: float):
        from RL_agents.rl.standardization import standardize_anticipated_affect, standardize_foursc_count

        self.data = data
        self.W = data.W
        self.e_hat = np.asarray(e_hat_hist, dtype=float)
        self.y1 = float(y1_model)
        W = self.W
        self.four = np.full((W, N_RL_DAYS, N_RL_SLOTS), np.nan)
        for idx in np.ndindex(self.four.shape):
            v = data.foursc[idx]
            if np.isfinite(v):
                self.four[idx] = standardize_foursc_count(v, std)
        locf = data.antic_locf()
        self.antic_obs = np.full((W, N_RL_DAYS), np.nan)
        self.antic_vis = np.full((W, N_RL_DAYS), np.nan)
        for idx in np.ndindex(self.antic_obs.shape):
            if np.isfinite(data.antic[idx]):
                self.antic_obs[idx] = standardize_anticipated_affect(data.antic[idx])
            if np.isfinite(locf[idx]):
                self.antic_vis[idx] = standardize_anticipated_affect(locf[idx])
        self.c = np.full((W, N_RL_DAYS, N_RL_SLOTS, N_SERVER_C), np.nan)
        self.c_imputed = 0
        for k in range(W):
            for d in range(N_RL_DAYS):
                for t in range(N_RL_SLOTS):
                    c, imp = data.model_c(k, d, t, std)
                    self.c[k, d, t] = c
                    self.c_imputed += len(imp)
        self.outcome = [data.model_outcome(i, std) for i in range(W + 1)]

    def four_row(self, k, d, t):
        # Lag = the preceding calendar slot. Monday morning's predecessor is
        # Sunday afternoon, which is not collected in deployment, so it takes
        # the testbed's value for an unavailable lag (0, the model-scale mean).
        if d == 0 and t == 0:
            lag = 0.0
        else:
            pd, pt = (d, t - 1) if t > 0 else (d - 1, N_RL_SLOTS - 1)
            v = self.four[k, pd, pt]
            lag = float(v) if np.isfinite(v) else 0.0
        return foursc_row(self.c[k, d, t], lag1=lag, d=d, t=t, pu=float(self.e_hat[k]),
                          action=self.data.A[k, d, t])

    def antic_row(self, k, d):
        if d == 0:
            yest = 0.0  # testbed zeroes the lag on the first day of the week
        else:
            yest = self.antic_vis[k, d - 1]
            yest = float(yest) if np.isfinite(yest) else 0.0
        c = self.c[k, d, 0]
        return antic_row(antic_yesterday=yest, active=c[2], d=d, pu=float(self.e_hat[k]), rb=c[3],
                         ws_m=self.data.A[k, d, 0], ws_a=self.data.A[k, d, 1])

    def cae_row(self, sw):
        # Answered check-ins only; missing entries skipped (as in the RL state).
        return cae_row(self.four[sw].ravel(), self.antic_obs[sw])

    def build(self, k: int) -> dict[str, Any]:
        """PF inputs for the belief update of ``Y_k`` (testbed ``build_pf_data``)."""
        sw_prev = k - 1
        X_four = np.array([self.four_row(sw_prev, d, t) for d in range(N_RL_DAYS) for t in range(N_RL_SLOTS)])
        X_antic = np.array([self.antic_row(sw_prev, d) for d in range(N_RL_DAYS)])
        data = {
            "X_MY_base": [X_four, X_antic],
            "cae_delta_MY": [np.array([foursc_cae_delta(x) for x in X_four]),
                             np.array([antic_cae_delta(x) for x in X_antic])],
            "M_Y_obs": [self.four[sw_prev].ravel().copy(), self.antic_obs[sw_prev].copy()],
            "X_Y_base": self.cae_row(sw_prev),
            "cae_delta_Y": np.eye(4)[_CAE_AR1_COL],
            "X_tY_base": np.array([1.0, 0.0]),
            "cae_delta_tY": np.array([0.0, 1.0]),
            "cae_all_0": self.y1,
            "sim_w_prev": sw_prev,
            "has_cumul": k >= 2,
        }
        if k >= 2:
            fx, fd, fw, fy = [], [], [], []
            ax, ad, aw, ay = [], [], [], []
            for sw in range(sw_prev):
                for d in range(N_RL_DAYS):
                    for t in range(N_RL_SLOTS):
                        y = self.four[sw, d, t]
                        x = self.four_row(sw, d, t)
                        if np.isfinite(y) and np.all(np.isfinite(x)):
                            fx.append(x); fd.append(foursc_cae_delta(x)); fw.append(sw); fy.append(y)
                    y = self.antic_obs[sw, d]
                    x = self.antic_row(sw, d)
                    if np.isfinite(y) and np.all(np.isfinite(x)):
                        ax.append(x); ad.append(antic_cae_delta(x)); aw.append(sw); ay.append(y)
            pf_, pa_ = PF_DIMS["fourSC"], PF_DIMS["antic"]
            data["X_cumul_MY_base"] = [np.array(fx).reshape(-1, pf_), np.array(ax).reshape(-1, pa_)]
            data["cae_delta_cumul_MY"] = [np.array(fd).reshape(-1, pf_), np.array(ad).reshape(-1, pa_)]
            data["week_idx_cumul_MY"] = [np.array(fw, dtype=int), np.array(aw, dtype=int)]
            data["y_cumul_MY"] = [np.array(fy), np.array(ay)]
            data["X_cumul_Y_base"] = np.array([self.cae_row(sw) for sw in range(sw_prev)])
            delta = np.zeros((sw_prev, 4))
            delta[:, _CAE_AR1_COL] = 1.0
            data["cae_delta_cumul_Y"] = delta
            t_rows = [i for i in range(1, sw_prev + 1)
                      if self.outcome[i]["J_w"] == 1 and self.outcome[i]["Y_tilde"] is not None]
            data["X_cumul_tY"] = np.tile([1.0, 0.0], (len(t_rows), 1))
            data["cae_delta_cumul_tY"] = np.tile([0.0, 1.0], (len(t_rows), 1))
            data["week_idx_cumul_tY"] = np.array([i - 1 for i in t_rows], dtype=int)
            data["y_cumul_tY"] = np.array([self.outcome[i]["Y_tilde"] for i in t_rows])
        return data


# ---------------------------------------------------------------------------
# Filter update
# ---------------------------------------------------------------------------
def bayesian_posterior_update(nu_0, Gamma_0, X, y, sigma2):
    nu_0 = np.asarray(nu_0, dtype=float).ravel()
    Gamma_0 = np.asarray(Gamma_0, dtype=float)
    X = np.asarray(X, dtype=float).reshape(-1, nu_0.size)
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        return nu_0.copy(), Gamma_0.copy()
    if not (np.isfinite(sigma2) and sigma2 > 0):
        raise ValueError(f"invalid PF sigma2 {sigma2!r}")
    G0i = np.linalg.inv(Gamma_0)
    Gp = spd_inverse(G0i + X.T @ X / sigma2)
    return Gp @ (G0i @ nu_0 + X.T @ y / sigma2), Gp


def _cae_by_sw(j, sw_prev, y_hat, cae0):
    out = np.empty(sw_prev + 1)
    out[0] = cae0
    if sw_prev > 0:
        out[1:] = y_hat[j, 1:sw_prev + 1]
    return out


def _check_cov(G, name):
    G = np.asarray(G, dtype=float)
    if not np.all(np.isfinite(G)) or float(np.linalg.eigvalsh(0.5 * (G + G.T)).min()) < -1e-8 * max(1.0, float(np.abs(G).max())):
        raise np.linalg.LinAlgError(f"PF covariance {name} is invalid")


def _refit_sigma2(pf, sw_prev, ybar, priors, s2):
    cae_sw = np.empty(sw_prev + 1)
    cae_sw[0] = pf["cae_all_0"]
    if sw_prev > 0:
        cae_sw[1:] = ybar[1:sw_prev + 1]
    out = {"sigma2_MY": list(s2["sigma2_MY"]), "sigma2_Y": s2["sigma2_Y"], "sigma2_tilde_Y": s2["sigma2_tilde_Y"]}
    for m in range(2):
        Xb = pf["X_cumul_MY_base"][m]
        if Xb.shape[0] == 0:
            continue
        X = Xb + cae_sw[pf["week_idx_cumul_MY"][m]][:, None] * pf["cae_delta_cumul_MY"][m]
        out["sigma2_MY"][m] = empirical_bayes_sigma2(X, pf["y_cumul_MY"][m], priors["nu_0_MY"][m],
                                                     priors["Gamma_0_MY"][m], out["sigma2_MY"][m])
    if sw_prev > 0:
        X = pf["X_cumul_Y_base"] + cae_sw[:sw_prev][:, None] * pf["cae_delta_cumul_Y"]
        out["sigma2_Y"] = empirical_bayes_sigma2(X, ybar[1:sw_prev + 1], priors["nu_0_Y"], priors["Gamma_0_Y"], out["sigma2_Y"])
    if len(pf["y_cumul_tY"]) > 0:
        X = pf["X_cumul_tY"] + ybar[pf["week_idx_cumul_tY"] + 1][:, None] * pf["cae_delta_cumul_tY"]
        out["sigma2_tilde_Y"] = empirical_bayes_sigma2(X, pf["y_cumul_tY"], priors["nu_0_tilde_Y"],
                                                       priors["Gamma_0_tilde_Y"], out["sigma2_tilde_Y"])
    return out


def update_week(*, k: int, y_hat: np.ndarray, v_hat: np.ndarray, sigma2: dict, pf: dict, outcome: dict,
                priors: dict, rng: np.random.Generator, refit_sigma2: bool) -> dict[str, Any]:
    """One belief update for ``Y_k``; returns new particles, weights, sigma2, belief."""
    J = y_hat.shape[0]
    if y_hat.shape[1] != k:
        raise PFStateMismatch(f"particles cover {y_hat.shape[1]} columns; week k={k} needs {k}")
    if not (np.all(np.isfinite(y_hat)) and np.all(np.isfinite(v_hat)) and v_hat.sum() > 0):
        raise PFStateMismatch("non-finite particles or weights")
    sw_prev = k - 1
    s2 = {"sigma2_MY": list(map(float, sigma2["sigma2_MY"])), "sigma2_Y": float(sigma2["sigma2_Y"]),
          "sigma2_tilde_Y": float(sigma2["sigma2_tilde_Y"])}
    if pf["has_cumul"] and refit_sigma2:
        s2 = _refit_sigma2(pf, sw_prev, (v_hat @ y_hat).ravel(), priors, s2)
    J_w, I_w = outcome.get("J_w"), outcome.get("I_w")
    Y_obs = outcome.get("Y") if (J_w == 1 and I_w == 1) else None
    tY_obs = outcome.get("Y_tilde") if (J_w == 1 and I_w == 0) else None

    y_w = np.zeros(J)
    log_lik = np.zeros(J)
    mu_Y = np.zeros(J)
    for j in range(J):
        if pf["has_cumul"]:
            cae_sw = _cae_by_sw(j, sw_prev, y_hat, pf["cae_all_0"])
            nu_MY, G_MY = [], []
            for m in range(2):
                X = pf["X_cumul_MY_base"][m] + cae_sw[pf["week_idx_cumul_MY"][m]][:, None] * pf["cae_delta_cumul_MY"][m]
                nu, G = bayesian_posterior_update(priors["nu_0_MY"][m], priors["Gamma_0_MY"][m], X,
                                                  pf["y_cumul_MY"][m], s2["sigma2_MY"][m])
                nu_MY.append(nu); G_MY.append(G)
            XY = pf["X_cumul_Y_base"] + cae_sw[:sw_prev][:, None] * pf["cae_delta_cumul_Y"]
            nu_Y, G_Y = bayesian_posterior_update(priors["nu_0_Y"], priors["Gamma_0_Y"], XY,
                                                  y_hat[j, 1:sw_prev + 1], s2["sigma2_Y"])
            XtY = pf["X_cumul_tY"] + y_hat[j, pf["week_idx_cumul_tY"] + 1][:, None] * pf["cae_delta_cumul_tY"]
            nu_t, G_t = bayesian_posterior_update(priors["nu_0_tilde_Y"], priors["Gamma_0_tilde_Y"], XtY,
                                                  pf["y_cumul_tY"], s2["sigma2_tilde_Y"])
        else:
            nu_MY = [np.asarray(v, dtype=float) for v in priors["nu_0_MY"]]
            G_MY = [np.asarray(v, dtype=float) for v in priors["Gamma_0_MY"]]
            nu_Y, G_Y = np.asarray(priors["nu_0_Y"]), np.asarray(priors["Gamma_0_Y"])
            nu_t, G_t = np.asarray(priors["nu_0_tilde_Y"]), np.asarray(priors["Gamma_0_tilde_Y"])
        for name, G in (("MY0", G_MY[0]), ("MY1", G_MY[1]), ("Y", G_Y), ("tY", G_t)):
            _check_cov(G, name)
        th_MY = [rng.multivariate_normal(nu_MY[m], G_MY[m]) for m in range(2)]
        th_Y = rng.multivariate_normal(nu_Y, G_Y)
        th_t = rng.multivariate_normal(nu_t, G_t)
        cae_curr = _cae_by_sw(j, sw_prev, y_hat, pf["cae_all_0"])[sw_prev]
        XY_j = pf["X_Y_base"] + cae_curr * pf["cae_delta_Y"]
        mu_Y[j] = float(th_Y @ XY_j)
        y_w[j] = rng.normal(mu_Y[j], np.sqrt(s2["sigma2_Y"]))
        for m in range(2):
            X_m = np.atleast_2d(pf["X_MY_base"][m]) + cae_curr * np.atleast_2d(pf["cae_delta_MY"][m])
            mu = X_m @ th_MY[m]
            y = np.asarray(pf["M_Y_obs"][m], dtype=float)
            mask = np.isfinite(y) & np.isfinite(mu)
            if np.any(mask):
                log_lik[j] += float(np.sum(-0.5 * np.log(2 * np.pi * s2["sigma2_MY"][m])
                                           - 0.5 * (y[mask] - mu[mask]) ** 2 / s2["sigma2_MY"][m]))
        if tY_obs is not None:
            mu_t = float(th_t @ (pf["X_tY_base"] + y_w[j] * pf["cae_delta_tY"]))
            log_lik[j] += -0.5 * np.log(2 * np.pi * s2["sigma2_tilde_Y"]) - 0.5 * (tY_obs - mu_t) ** 2 / s2["sigma2_tilde_Y"]
    log_v = np.log(np.maximum(v_hat, 1e-300)) + log_lik
    if Y_obs is not None:
        y_w[:] = Y_obs
        log_v += -0.5 * np.log(2 * np.pi * s2["sigma2_Y"]) - 0.5 * (Y_obs - mu_Y) ** 2 / s2["sigma2_Y"]
    if not np.all(np.isfinite(mu_Y)) or not np.any(np.isfinite(log_v)):
        raise FloatingPointError("non-finite PF likelihood or transition mean")
    log_v -= np.max(log_v)
    v = np.exp(log_v)
    v /= v.sum()
    ess = float(1.0 / np.sum(v**2))
    y_new = np.zeros((J, k + 1))
    resampled = ess < 0.5 * J
    if resampled:
        idx = rng.choice(J, size=J, replace=True, p=v)
        y_new[:, :k] = y_hat[idx]
        y_new[:, k] = y_w[idx]
        v_new = np.full(J, 1.0 / J)
    else:
        y_new[:, :k] = y_hat
        y_new[:, k] = y_w
        v_new = v
    b_hat = float(np.average(y_new[:, -1], weights=v_new))
    b_tilde = float(np.sqrt(np.average((y_new[:, -1] - b_hat) ** 2, weights=v_new)))
    return {"y_hat": y_new, "v_hat": v_new, "sigma2": s2, "b_hat": b_hat, "b_tilde": b_tilde,
            "ess": ess, "resampled": bool(resampled),
            "observed": "Y" if Y_obs is not None else ("Y_tilde" if tY_obs is not None else "none")}


def load_pf_priors_v2(payload: dict) -> dict:
    pf = payload.get("pf")
    if not isinstance(pf, dict):
        raise KeyError("prior file has no 'pf' block")
    for key, dim in PF_DIMS.items():
        if key not in pf or len(pf[key]["nu_0"]) != dim:
            raise ValueError(f"PF prior {key} must have dimension {dim} (testbed working models)")
    for key, names in (("fourSC", PF_THETA_FOURSC_NAMES), ("antic", PF_THETA_ANTIC_NAMES)):
        got = pf[key].get("names")
        if got is not None and list(got) != names:
            raise ValueError(f"PF prior {key} columns do not match the working model: {got}")
    arr = lambda v: np.asarray(v, dtype=float)  # noqa: E731
    return {
        "nu_0_MY": [arr(pf["fourSC"]["nu_0"]), arr(pf["antic"]["nu_0"])],
        "Gamma_0_MY": [arr(pf["fourSC"]["Gamma_0"]), arr(pf["antic"]["Gamma_0"])],
        "sigma2_MY": [float(pf["fourSC"]["sigma2"]), float(pf["antic"]["sigma2"])],
        "nu_0_Y": arr(pf["CAE"]["nu_0"]), "Gamma_0_Y": arr(pf["CAE"]["Gamma_0"]),
        "sigma2_Y": float(pf["CAE"]["sigma2"]),
        "nu_0_tilde_Y": arr(pf["CAE_short"]["nu_0"]), "Gamma_0_tilde_Y": arr(pf["CAE_short"]["Gamma_0"]),
        "sigma2_tilde_Y": float(pf["CAE_short"]["sigma2"]),
    }
