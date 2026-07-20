"""Prior loading and feature-order validation for the RL handler."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _read_json(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _q_block(payload: dict) -> dict:
    return payload.get("q_no_td_modify", payload)


def _legacy_placeholder_indices(n_features: int) -> tuple[int, int]:
    """Indices of the two legacy hardcoded-zero context columns."""
    n_context_legacy = 9
    n_action_block_legacy = 9 + n_context_legacy
    state_n = n_features - n_action_block_legacy
    return state_n - 1, n_features - 1


def _strip_legacy_placeholders(mu_0: np.ndarray, Sigma_0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    skip = _legacy_placeholder_indices(mu_0.size)
    mu_new = np.delete(mu_0, skip, axis=0)
    Sigma_new = np.delete(np.delete(Sigma_0, skip, axis=0), skip, axis=1)
    return mu_new, Sigma_new


def load_q_prior(
    *,
    q_prior_path: str | Path | None,
    combined_prior_path: str | Path | None,
    expected_feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, float, list[str], str] | None:
    """Load the base Q prior (``q_no_td_modify``) only if feature order is auditable.

    The single prior source is ``rl_priors.json`` (its ``q_no_td_modify`` block),
    read via ``combined_prior_path``. ``q_prior_path`` remains as an optional
    override for a standalone Q-prior file but is unset by default.

    Legacy blocks may have 79 coefficients but 77 runtime feature names because
    two hardcoded-zero placeholder columns were removed from ``RL_agents``.
    In that case we strip exactly those two columns before validating names.
    """
    source_path: str | Path | None = q_prior_path
    if source_path is not None and Path(source_path).expanduser().is_file():
        block = _q_block(_read_json(source_path))
    elif combined_prior_path is not None and Path(combined_prior_path).expanduser().is_file():
        source_path = combined_prior_path
        block = _q_block(_read_json(source_path))
    else:
        return None

    mu_0 = np.asarray(block["mu_0"], dtype=float).ravel()
    Sigma_0 = np.asarray(block["Sigma_0"], dtype=float)
    sigma2 = float(block.get("sigma2", 1.0))
    names = list(block.get("feature_names", []))
    expected_p = len(expected_feature_names)

    if Sigma_0.shape != (mu_0.size, mu_0.size):
        raise ValueError(
            f"Q prior covariance shape {Sigma_0.shape} does not match mean size {mu_0.size}"
        )

    if mu_0.size == expected_p + 2 and len(names) == expected_p:
        mu_0, Sigma_0 = _strip_legacy_placeholders(mu_0, Sigma_0)
    elif mu_0.size != expected_p:
        raise ValueError(
            f"Q prior dimension {mu_0.size} does not match runtime feature dimension {expected_p}"
        )

    if names:
        if names != expected_feature_names:
            raise ValueError("Q prior feature_names order does not match build_phi_action order")
    else:
        names = list(expected_feature_names)

    return mu_0, Sigma_0, sigma2, names, str(source_path)


def default_q_priors(
    feature_names: list[str],
    *,
    sigma2: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, list[str], str]:
    """Uninformative RLSVI Q prior: zero mean and identity covariance."""
    p = len(feature_names)
    mu_0 = np.zeros(p, dtype=float)
    Sigma_0 = np.eye(p, dtype=float)
    return mu_0, Sigma_0, float(sigma2), list(feature_names), "default"


def resolve_q_priors(
    loaded: tuple[np.ndarray, np.ndarray, float, list[str], str] | None,
    *,
    config: dict,
    feature_names: list[str],
    use_default_priors: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, list[str], str | None]:
    """Return RLSVI Q priors for the handler.

    Priority:
    1. ``use_default_priors`` → ``default_q_priors``
    2. ``config["mu_0_rl"]`` override
    3. fitted prior from ``load_q_prior``
    4. zero mean with ``config["p_rl"]`` (or runtime feature dimension)
    """
    sigma2 = float(config.get("sigma2_rl", 1.0))
    if use_default_priors:
        return default_q_priors(feature_names, sigma2=sigma2)

    mu_0_config = config.get("mu_0_rl")
    if mu_0_config is None:
        if loaded is not None:
            return loaded
        p_rl = config.get("p_rl")
        if p_rl is None:
            p_rl = len(feature_names)
        mu_0 = np.zeros(int(p_rl), dtype=float)
        prior_source = None
    else:
        mu_0 = np.asarray(mu_0_config, dtype=float).ravel()
        prior_source = "config"

    sigma_config = config.get("Sigma_0_rl")
    Sigma_0 = (
        np.eye(mu_0.size, dtype=float)
        if sigma_config is None
        else np.asarray(sigma_config, dtype=float)
    )
    if Sigma_0.shape != (mu_0.size, mu_0.size):
        raise ValueError(
            f"Sigma_0_rl shape {Sigma_0.shape} does not match mu_0_rl size {mu_0.size}"
        )
    if mu_0.size != len(feature_names):
        raise ValueError(
            f"RL prior dimension {mu_0.size} does not match runtime feature "
            f"dimension {len(feature_names)}"
        )
    return mu_0, Sigma_0, sigma2, list(feature_names), prior_source


def load_pf_priors(combined_prior_path: str | Path | None) -> tuple[dict, str] | None:
    if combined_prior_path is None or not Path(combined_prior_path).expanduser().is_file():
        return None
    payload = _read_json(combined_prior_path)
    pf = payload.get("pf")
    if not isinstance(pf, dict):
        return None
    required = ("fourSC", "antic", "CAE", "CAE_short")
    missing = [key for key in required if key not in pf]
    if missing:
        raise KeyError(f"PF priors missing blocks: {missing}")

    return (
        {
            "nu_0_MY": [
                np.asarray(pf["fourSC"]["nu_0"], dtype=float),
                np.asarray(pf["antic"]["nu_0"], dtype=float),
            ],
            "Gamma_0_MY": [
                np.asarray(pf["fourSC"]["Gamma_0"], dtype=float),
                np.asarray(pf["antic"]["Gamma_0"], dtype=float),
            ],
            "sigma2_MY": [
                float(pf["fourSC"]["sigma2"]),
                float(pf["antic"]["sigma2"]),
            ],
            "nu_0_Y": np.asarray(pf["CAE"]["nu_0"], dtype=float),
            "Gamma_0_Y": np.asarray(pf["CAE"]["Gamma_0"], dtype=float),
            "sigma2_Y": float(pf["CAE"]["sigma2"]),
            "nu_0_tilde_Y": np.asarray(pf["CAE_short"]["nu_0"], dtype=float),
            "Gamma_0_tilde_Y": np.asarray(pf["CAE_short"]["Gamma_0"], dtype=float),
            "sigma2_tilde_Y": float(pf["CAE_short"]["sigma2"]),
        },
        str(combined_prior_path),
    )
