"""Particle-filter belief update helpers for the deployment RL handler."""

from __future__ import annotations

from typing import Any

import numpy as np

from RL_agents.rl.algorithm_helpers import empirical_bayes_sigma2, spd_inverse
from RL_agents.rl.other_helpers import jsonable_float_array, none_to_nan_array

FOURSC_SLOTS_PER_WEEK = 14

# PF regression model dimensions (fourSC, antic, weekly CAE, short CAE).
PF_FOURSC_DIM = 25
PF_ANTIC_DIM = 19
PF_CAE_DIM = 24
PF_CAE_SHORT_DIM = 2


def default_pf_priors(
    *,
    sigma2_MY: list[float] | None = None,
    sigma2_Y: float = 1.0,
    sigma2_tilde_Y: float = 1.0,
) -> dict:
    """Uninformative PF priors with canonical model dimensions.

    Mediator models: fourSC (``PF_FOURSC_DIM``) and antic (``PF_ANTIC_DIM``).
    Outcome models: weekly CAE (``PF_CAE_DIM``) and short CAE (``PF_CAE_SHORT_DIM``).
    """
    if sigma2_MY is None:
        sigma2_MY = [1.0, 1.0]
    return {
        "nu_0_MY": [
            np.zeros(PF_FOURSC_DIM, dtype=float),
            np.zeros(PF_ANTIC_DIM, dtype=float),
        ],
        "Gamma_0_MY": [
            np.eye(PF_FOURSC_DIM, dtype=float),
            np.eye(PF_ANTIC_DIM, dtype=float),
        ],
        "sigma2_MY": [float(s) for s in sigma2_MY],
        "nu_0_Y": np.zeros(PF_CAE_DIM, dtype=float),
        "Gamma_0_Y": np.eye(PF_CAE_DIM, dtype=float),
        "sigma2_Y": float(sigma2_Y),
        "nu_0_tilde_Y": np.zeros(PF_CAE_SHORT_DIM, dtype=float),
        "Gamma_0_tilde_Y": np.eye(PF_CAE_SHORT_DIM, dtype=float),
        "sigma2_tilde_Y": float(sigma2_tilde_Y),
    }


def resolve_pf_priors(
    loaded: tuple[dict, str] | None,
    *,
    use_default_priors: bool = False,
) -> tuple[dict, str | None]:
    """Return PF priors for the handler.

    Uses fitted priors from ``load_pf_priors`` when available and
    ``use_default_priors`` is false; otherwise returns ``default_pf_priors()``.
    """
    if use_default_priors or loaded is None:
        return default_pf_priors(), "default"
    pf_priors, source = loaded
    return pf_priors, source


def normalize_pf_packet(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one observation payload into the PF packet consumed by update."""
    if not payload:
        return None
    packet = payload.get("pf_packet", payload)
    if packet.get("type") not in (None, "rl_pf_packet", "pf_packet"):
        return None
    pf_data = packet.get("pf_data")
    if pf_data is None:
        return None
    y_prev = packet.get("Y_prev", packet.get("Y_w", packet.get("CAE")))
    ty_prev = packet.get(
        "tY_prev",
        packet.get("tilde_Y_w", packet.get("CAE_short", packet.get("cae_short"))),
    )
    # Week index contract: an external 1-indexed week ``w`` maps to the internal
    # 0-indexed RL week ``k = w - 1``; a raw ``k`` is assumed already internal.
    if packet.get("w") is not None:
        k_val = int(packet["w"]) - 1
    elif packet.get("k") is not None:
        k_val = int(packet["k"])
    else:
        k_val = None
    out = {
        "type": "rl_pf_packet",
        "k": k_val,
        "I_w": int(packet.get("I_w", packet.get("query_action", 1))),
        "J_w": None if packet.get("J_w") is None else int(packet["J_w"]),
        "Y_prev": None if y_prev is None else float(y_prev),
        "tY_prev": None if ty_prev is None else float(ty_prev),
        "pf_data": pf_data,
    }
    return out


def latest_pf_packet(observations: list[Any]) -> dict[str, Any] | None:
    packets: list[dict[str, Any]] = []
    for obs in observations:
        if getattr(obs, "observation_type", None) not in (None, "rl_pf_packet", "pf_packet"):
            payload = getattr(obs, "payload", {}) or {}
            if "pf_data" not in payload and "pf_packet" not in payload:
                continue
        packet = normalize_pf_packet(getattr(obs, "payload", {}) or {})
        if packet is not None:
            packets.append(packet)
    if not packets:
        return None
    return packets[-1]


def initial_pf_state(*, J: int, Y_1: float) -> dict[str, Any]:
    return {
        "y_hat": jsonable_float_array(np.full((int(J), 1), float(Y_1))),
        "v_hat": jsonable_float_array(np.full(int(J), 1.0 / int(J))),
    }


def pf_state_arrays(model_params: dict) -> tuple[np.ndarray, np.ndarray]:
    y_hat = none_to_nan_array(model_params.get("pf_y_hat"), shape=(int(model_params["J"]), 1))
    v_hat = none_to_nan_array(model_params.get("pf_v_hat"), shape=(int(model_params["J"]),))
    return y_hat, v_hat


def summarize_belief(y_hat: np.ndarray, v_hat: np.ndarray) -> tuple[float, float]:
    y_w = np.asarray(y_hat, dtype=float)[:, -1]
    v = np.asarray(v_hat, dtype=float).ravel()
    b_hat = float(np.average(y_w, weights=v))
    b_tilde = float(np.sqrt(np.average((y_w - b_hat) ** 2, weights=v)))
    return b_hat, b_tilde


def _posterior_update(nu_0, Gamma_0, X, y, sigma2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X = X[mask]
    y = y[mask]
    if X.shape[0] == 0:
        return np.asarray(nu_0, dtype=float).copy(), np.asarray(Gamma_0, dtype=float).copy()
    Gamma_0 = np.asarray(Gamma_0, dtype=float)
    nu_0 = np.asarray(nu_0, dtype=float).ravel()
    Gamma_0_inv = np.linalg.inv(Gamma_0)
    Gamma_post_inv = Gamma_0_inv + (1.0 / float(sigma2)) * (X.T @ X)
    Gamma_post = spd_inverse(Gamma_post_inv)
    nu_post = Gamma_post @ (
        Gamma_0_inv @ nu_0 + (1.0 / float(sigma2)) * (X.T @ y)
    )
    return nu_post, Gamma_post


def _cae_by_sw(j, sim_w_prev, y_hat_prev, cae_all_0):
    out = np.empty(sim_w_prev + 1)
    out[0] = float(cae_all_0)
    if sim_w_prev > 0:
        out[1:] = y_hat_prev[j, 1 : sim_w_prev + 1]
    return out


def _per_particle_posterior(
    j,
    sim_w_prev,
    y_hat_prev,
    pf_data,
    pf_priors,
    pf_sigma2,
):
    cae_sw = _cae_by_sw(j, sim_w_prev, y_hat_prev, pf_data["cae_all_0"])
    nu_MY_j, Gamma_MY_j = [], []
    for m in range(len(pf_priors["nu_0_MY"])):
        wk = np.asarray(pf_data["week_idx_cumul_MY"][m], dtype=int)
        X_j = np.asarray(pf_data["X_cumul_MY_base"][m], dtype=float) + (
            cae_sw[wk, None] * np.asarray(pf_data["cae_delta_cumul_MY"][m], dtype=float)
        )
        nu_m, G_m = _posterior_update(
            pf_priors["nu_0_MY"][m],
            pf_priors["Gamma_0_MY"][m],
            X_j,
            pf_data["y_cumul_MY"][m],
            pf_sigma2["sigma2_MY"][m],
        )
        nu_MY_j.append(nu_m)
        Gamma_MY_j.append(G_m)

    X_Y = np.asarray(pf_data["X_cumul_Y_base"], dtype=float) + (
        cae_sw[:sim_w_prev, None] * np.asarray(pf_data["cae_delta_cumul_Y"], dtype=float)
    )
    y_Y = y_hat_prev[j, 1 : sim_w_prev + 1]
    nu_Y_j, Gamma_Y_j = _posterior_update(
        pf_priors["nu_0_Y"],
        pf_priors["Gamma_0_Y"],
        X_Y,
        y_Y,
        pf_sigma2["sigma2_Y"],
    )

    wk_t = np.asarray(pf_data["week_idx_cumul_tY"], dtype=int)
    X_tY = np.asarray(pf_data["X_cumul_tY"], dtype=float) + (
        y_hat_prev[j, wk_t + 1, None]
        * np.asarray(pf_data["cae_delta_cumul_tY"], dtype=float)
    )
    nu_tY_j, Gamma_tY_j = _posterior_update(
        pf_priors["nu_0_tilde_Y"],
        pf_priors["Gamma_0_tilde_Y"],
        X_tY,
        pf_data["y_cumul_tY"],
        pf_sigma2["sigma2_tilde_Y"],
    )
    return nu_MY_j, Gamma_MY_j, nu_Y_j, Gamma_Y_j, nu_tY_j, Gamma_tY_j


def _transition_feature_with_particle_mediators(
    X_Y_base, cae_delta_Y, cae_value, X_MY_j, theta_MY_j, M_Y_obs
):
    X_Y_j = np.asarray(X_Y_base, dtype=float).ravel() + float(cae_value) * np.asarray(
        cae_delta_Y, dtype=float
    ).ravel()
    if len(M_Y_obs) >= 1 and len(X_MY_j) >= 1:
        y_four = np.asarray(M_Y_obs[0], dtype=float).ravel()
        mu_four = np.asarray(X_MY_j[0], dtype=float) @ np.asarray(theta_MY_j[0], dtype=float)
        n_four = min(y_four.size, mu_four.size, FOURSC_SLOTS_PER_WEEK, X_Y_j.size - 3)
        if n_four > 0:
            X_Y_j[3 : 3 + n_four] = np.where(
                np.isfinite(y_four[:n_four]), y_four[:n_four], mu_four[:n_four]
            )
    if len(M_Y_obs) >= 2 and len(X_MY_j) >= 2:
        y_antic = np.asarray(M_Y_obs[1], dtype=float).ravel()
        mu_antic = np.asarray(X_MY_j[1], dtype=float) @ np.asarray(theta_MY_j[1], dtype=float)
        start = 3 + FOURSC_SLOTS_PER_WEEK
        n_antic = min(y_antic.size, mu_antic.size, X_Y_j.size - start)
        if n_antic > 0:
            X_Y_j[start : start + n_antic] = np.where(
                np.isfinite(y_antic[:n_antic]), y_antic[:n_antic], mu_antic[:n_antic]
            )
    return X_Y_j


def _outcome_from_packet(packet: dict[str, Any]) -> tuple[int, float | None, float | None]:
    if packet.get("J_w") is not None and int(packet["J_w"]) != 1:
        return 0, None, None
    if int(packet.get("I_w", 1)) == 1:
        return 1, None if packet.get("Y_prev") is None else float(packet["Y_prev"]), None
    return 1, None, None if packet.get("tY_prev") is None else float(packet["tY_prev"])


def _update_sigma2_eb(
    pf_data: dict[str, Any],
    sim_w_prev: int,
    ybar: np.ndarray,
    pf_priors: dict[str, Any],
    sigma2: dict[str, Any],
) -> dict[str, Any]:
    """Empirical-Bayes refit of the PF mediator/CAE noise variances.

    Run once per RL week ("Monday night", before the posterior draws). Each
    working model's scalar variance is re-estimated by maximizing its marginal
    likelihood on the cumulative complete-case design, with the latent CAE
    columns filled by the particle-weighted-mean trajectory ``ybar``. Mirrors
    ``ParticleFilterRuntime._update_sigma2_eb`` in the top-level runtime.
    """
    cae_sw = np.empty(sim_w_prev + 1)
    cae_sw[0] = float(pf_data["cae_all_0"])
    if sim_w_prev > 0:
        cae_sw[1:] = ybar[1 : sim_w_prev + 1]

    sigma2_MY = list(map(float, sigma2["sigma2_MY"]))
    for m in range(len(pf_priors["nu_0_MY"])):
        X_base = pf_data["X_cumul_MY_base"][m]
        if X_base is None or len(X_base) == 0:
            continue
        wk = np.asarray(pf_data["week_idx_cumul_MY"][m], dtype=int)
        X_m = np.asarray(X_base, dtype=float) + (
            cae_sw[wk, None] * np.asarray(pf_data["cae_delta_cumul_MY"][m], dtype=float)
        )
        sigma2_MY[m] = float(
            empirical_bayes_sigma2(
                X_m,
                pf_data["y_cumul_MY"][m],
                pf_priors["nu_0_MY"][m],
                pf_priors["Gamma_0_MY"][m],
                sigma2_MY[m],
            )
        )

    sigma2_Y = float(sigma2["sigma2_Y"])
    X_Y_base = pf_data.get("X_cumul_Y_base")
    if X_Y_base is not None and sim_w_prev > 0:
        cae_cumul = cae_sw[:sim_w_prev]
        X_Y = np.asarray(X_Y_base, dtype=float) + (
            cae_cumul[:, None] * np.asarray(pf_data["cae_delta_cumul_Y"], dtype=float)
        )
        y_Y = ybar[1 : sim_w_prev + 1]
        sigma2_Y = float(
            empirical_bayes_sigma2(
                X_Y, y_Y, pf_priors["nu_0_Y"], pf_priors["Gamma_0_Y"], sigma2_Y
            )
        )

    sigma2_tilde_Y = float(sigma2["sigma2_tilde_Y"])
    X_tY = pf_data.get("X_cumul_tY")
    if X_tY is not None and len(pf_data.get("y_cumul_tY", [])) > 0:
        wk_t = np.asarray(pf_data["week_idx_cumul_tY"], dtype=int)
        X_tY_full = np.asarray(X_tY, dtype=float) + (
            ybar[wk_t + 1, None] * np.asarray(pf_data["cae_delta_cumul_tY"], dtype=float)
        )
        sigma2_tilde_Y = float(
            empirical_bayes_sigma2(
                X_tY_full,
                pf_data["y_cumul_tY"],
                pf_priors["nu_0_tilde_Y"],
                pf_priors["Gamma_0_tilde_Y"],
                sigma2_tilde_Y,
            )
        )

    return {
        "sigma2_MY": sigma2_MY,
        "sigma2_Y": sigma2_Y,
        "sigma2_tilde_Y": sigma2_tilde_Y,
    }


def update_pf_belief(
    *,
    model_params: dict,
    packet: dict[str, Any],
    pf_priors: dict[str, Any],
    rng: np.random.Generator,
    refit_sigma2: bool = True,
) -> tuple[dict[str, Any], float, float]:
    """Run one weekly particle-filter update from a normalized PF packet.

    When ``refit_sigma2`` is ``False`` the online empirical-Bayes refit of the
    mediator/CAE noise variances is skipped, and the ``sigma2_*`` values carried
    in ``model_params`` (i.e. the PF prior variances) are used as-is for this
    week's posterior draws and likelihoods.
    """
    pf_data = packet["pf_data"]
    k = int(packet["k"])
    if k == 0:
        y_hat, v_hat = pf_state_arrays(model_params)
        b_hat, b_tilde = summarize_belief(y_hat, v_hat)
        return {"pf_y_hat": jsonable_float_array(y_hat), "pf_v_hat": jsonable_float_array(v_hat)}, b_hat, b_tilde

    J = int(model_params["J"])
    y_hat, v_hat = pf_state_arrays(model_params)
    sim_w_prev = int(pf_data.get("sim_w_prev", k - 1))
    has_cumul = pf_data.get("X_cumul_MY_base") is not None
    J_w, Y_w, tilde_Y_w = _outcome_from_packet(packet)
    sigma2 = {
        "sigma2_MY": list(map(float, model_params["sigma2_MY"])),
        "sigma2_Y": float(model_params["sigma2_Y"]),
        "sigma2_tilde_Y": float(model_params["sigma2_tilde_Y"]),
    }

    # Online noise-variance (empirical-Bayes) refit FIRST, using the pre-update
    # belief-weighted-mean CAE trajectory. The refit sigma^2 are then consumed by
    # this week's posterior draws and likelihoods. Failures are surfaced (not
    # fatal): the prior sigma^2 are retained and a marker is returned for a
    # yellow "online noise-variance update failed" alert upstream.
    sigma2_eb_error = None
    if has_cumul and refit_sigma2:
        try:
            ybar = (
                np.asarray(v_hat, dtype=float).ravel() @ np.asarray(y_hat, dtype=float)
            ).ravel()
            sigma2 = _update_sigma2_eb(pf_data, sim_w_prev, ybar, pf_priors, sigma2)
        except Exception as exc:  # noqa: BLE001 - non-fatal; retain prior sigma^2
            sigma2_eb_error = str(exc)

    y_w = np.zeros(J)
    log_med_lik = np.zeros(J)
    mu_Y_arr = np.zeros(J)
    theta_sources = []
    for j in range(J):
        if has_cumul:
            posterior = _per_particle_posterior(j, sim_w_prev, y_hat, pf_data, pf_priors, sigma2)
            nu_MY_j, Gamma_MY_j, nu_Y_j, Gamma_Y_j, nu_tY_j, Gamma_tY_j = posterior
        else:
            nu_MY_j = [nu.copy() for nu in pf_priors["nu_0_MY"]]
            Gamma_MY_j = [G.copy() for G in pf_priors["Gamma_0_MY"]]
            nu_Y_j = pf_priors["nu_0_Y"].copy()
            Gamma_Y_j = pf_priors["Gamma_0_Y"].copy()
            nu_tY_j = pf_priors["nu_0_tilde_Y"].copy()
            Gamma_tY_j = pf_priors["Gamma_0_tilde_Y"].copy()

        theta_MY_j = [
            rng.multivariate_normal(nu_MY_j[m], Gamma_MY_j[m])
            for m in range(len(nu_MY_j))
        ]
        theta_Y_j = rng.multivariate_normal(nu_Y_j, Gamma_Y_j)
        theta_tY_j = rng.multivariate_normal(nu_tY_j, Gamma_tY_j)
        theta_sources.append(
            (nu_MY_j, Gamma_MY_j, theta_MY_j, nu_Y_j, Gamma_Y_j, theta_Y_j, nu_tY_j, Gamma_tY_j, theta_tY_j)
        )

        cae_curr = _cae_by_sw(j, sim_w_prev, y_hat, pf_data["cae_all_0"])[sim_w_prev]
        X_MY_j = [
            np.atleast_2d(pf_data["X_MY_base"][m])
            + cae_curr * np.atleast_2d(pf_data["cae_delta_MY"][m])
            for m in range(len(nu_MY_j))
        ]
        X_Y_j = _transition_feature_with_particle_mediators(
            pf_data["X_Y_base"],
            pf_data["cae_delta_Y"],
            cae_curr,
            X_MY_j,
            theta_MY_j,
            pf_data["M_Y_obs"],
        )
        mu_y_j = float(theta_Y_j @ X_Y_j)
        mu_Y_arr[j] = mu_y_j
        y_w[j] = rng.normal(mu_y_j, np.sqrt(sigma2["sigma2_Y"]))

        for m in range(len(nu_MY_j)):
            y_obs_m = np.asarray(pf_data["M_Y_obs"][m], dtype=float).reshape(-1)
            obs_mask = np.isfinite(y_obs_m)
            if np.any(obs_mask):
                mu_m = X_MY_j[m] @ theta_MY_j[m]
                log_med_lik[j] += float(
                    np.sum(
                        -0.5 * np.log(2.0 * np.pi * sigma2["sigma2_MY"][m])
                        -0.5 * ((y_obs_m[obs_mask] - mu_m[obs_mask]) ** 2)
                        / sigma2["sigma2_MY"][m]
                    )
                )

        if int(packet.get("I_w", 1)) == 0 and J_w == 1 and tilde_Y_w is not None:
            X_tY_j = np.asarray(pf_data["X_tY_base"], dtype=float) + (
                y_w[j] * np.asarray(pf_data["cae_delta_tY"], dtype=float)
            )
            mu_t = float(theta_tY_j @ X_tY_j)
            log_med_lik[j] += (
                -0.5 * np.log(2.0 * np.pi * sigma2["sigma2_tilde_Y"])
                -0.5 * ((tilde_Y_w - mu_t) ** 2) / sigma2["sigma2_tilde_Y"]
            )

    log_w_prev = np.log(np.maximum(v_hat, 1e-300))
    if int(packet.get("I_w", 1)) == 1 and J_w == 1 and Y_w is not None:
        y_w[:] = Y_w
        log_y_lik = (
            -0.5 * np.log(2.0 * np.pi * sigma2["sigma2_Y"])
            -0.5 * ((Y_w - mu_Y_arr) ** 2) / sigma2["sigma2_Y"]
        )
        log_v = log_w_prev + log_med_lik + log_y_lik
    else:
        log_v = log_w_prev + log_med_lik
    log_v = log_v - np.max(log_v)
    v_norm = np.exp(log_v)
    v_norm = v_norm / np.sum(v_norm)

    y_hat_new = np.zeros((J, k + 1))
    ess = 1.0 / np.sum(v_norm**2)
    if ess < 0.5 * J:
        idx = rng.choice(J, size=J, replace=True, p=v_norm)
        y_hat_new[:, :k] = y_hat[idx]
        y_hat_new[:, k] = y_w[idx]
        v_hat_new = np.full(J, 1.0 / J)
    else:
        y_hat_new[:, :k] = y_hat
        y_hat_new[:, k] = y_w
        v_hat_new = v_norm

    b_hat, b_tilde = summarize_belief(y_hat_new, v_hat_new)
    state_update = {
        "pf_y_hat": jsonable_float_array(y_hat_new),
        "pf_v_hat": jsonable_float_array(v_hat_new),
        "pf_ess": float(ess),
        "sigma2_MY": list(map(float, sigma2["sigma2_MY"])),
        "sigma2_Y": float(sigma2["sigma2_Y"]),
        "sigma2_tilde_Y": float(sigma2["sigma2_tilde_Y"]),
    }
    if sigma2_eb_error is not None:
        state_update["sigma2_eb_error"] = sigma2_eb_error
    return state_update, b_hat, b_tilde
