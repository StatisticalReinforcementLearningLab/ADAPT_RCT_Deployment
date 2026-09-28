"""Agent-visible perceived-utility estimate ``e_hat_w`` for the deployment handler.

Mirrors the validated testbed (``agents/ew_hat.compute_Ew_hat_from_week`` with
the pooled coefficients written by ``6_est_Ew_weights.py``)::

    e_hat_k = intercept
              + c_E    * e_hat_{k-1}                 (AR term; e_hat_0 = E_lag0)
              + c_PV   * PV_sum_{k-1}                (Mon-Sat mean of 12 model-scale PV slots)
              + c_FW   * FW_sum_{k-1}                (Mon-Sat mean of 6 wear flags)
              + c_PJ   * PJ_sum_{k-1}                (Mon-Sat mean of 6 check-in flags)
              + c_J    * J_k                         (bundle opening week k)
              + c_half * J_k (U1_k + U2_k) / 14

``k`` is the 0-indexed week whose decisions use ``e_hat_k``; the mediators are
the completed week ``k-1``. Missing wear / check-in flags count as 0 and
missing page-view slots contribute 0 on the model scale (testbed convention).
When the bundle was opened (``J_k = 1``) but an item is missing, the item is
imputed by its pooled MRT mean from the coefficient file. The result is
reported with an explicit ``info`` dict; nothing falls back to other weeks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from RL_agents.logging_config import get_logger
from RL_agents.rl.algorithm_helpers import N_RL_DAYS, N_RL_SLOTS
from RL_agents.rl.standardization import standardize_pageview_count

logger = get_logger(__name__)

REQUIRED_EW_KEYS = ("intercept", "E_lag", "PV_sum", "FW_sum", "PJ_sum", "J_close", "half_J_close")
DEFAULT_EW_STORED_VALUE_TYPE = "rl_E_w"

W_DAYS = 7
ME_COL_PV_MORNING = 0
ME_COL_PV_AFTERNOON = 1
ME_COL_FW = 2
ME_COL_PJ = 3
N_ME_COLS = 4



def load_ew_coefs(path: str | Path | None) -> dict[str, float]:
    """Load pooled e_hat coefficients; raises if missing or in the legacy format."""
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(f"e_hat coefficient file not found: {path}")
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    missing = [k for k in REQUIRED_EW_KEYS if k not in raw]
    if missing:
        raise KeyError(
            f"e_hat coefficient file {path} lacks {missing}; regenerate it with "
            "6_est_Ew_weights.py (legacy no-intercept format is not supported)"
        )
    return {str(k): float(v) for k, v in raw.items()}


def ew_stored_value_type(config: dict) -> str:
    return str(config.get("ew_stored_value_type", DEFAULT_EW_STORED_VALUE_TYPE))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _as_me_matrix(value: Any) -> np.ndarray | None:
    """Coerce ``M_E`` to shape ``(<=7, 4)``; return ``None`` if unusable."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.size % N_ME_COLS == 0:
            arr = arr.reshape(-1, N_ME_COLS)
        else:
            return None
    if arr.ndim != 2 or arr.shape[1] < N_ME_COLS:
        return None
    return np.asarray(arr[:, :N_ME_COLS], dtype=float)


def ew_aggregates_from_me(M_E_raw: Any, std: dict) -> dict[str, float]:
    """Mon-Sat PV/FW/PJ means from a raw-scale ``M_E`` week (Sunday ignored)."""
    me = _as_me_matrix(M_E_raw)
    if me is None or not np.any(np.isfinite(me)):
        raise ValueError("M_E is missing or malformed")
    pv = np.full(N_RL_DAYS * N_RL_SLOTS, np.nan)
    fw = np.full(N_RL_DAYS, np.nan)
    pj = np.full(N_RL_DAYS, np.nan)
    for d in range(min(N_RL_DAYS, me.shape[0])):
        for t, col in enumerate((ME_COL_PV_MORNING, ME_COL_PV_AFTERNOON)):
            v = me[d, col]
            pv[d * N_RL_SLOTS + t] = standardize_pageview_count(v, std) if np.isfinite(v) else np.nan
        fw[d] = me[d, ME_COL_FW]
        pj[d] = me[d, ME_COL_PJ]
    return {
        "PV_sum": float(np.nansum(pv) / pv.size),
        "FW_sum": float(np.nansum(fw) / fw.size),
        "PJ_sum": float(np.nansum(pj) / pj.size),
        "n_pv_missing": int(np.sum(~np.isfinite(pv))),
        "n_fw_missing": int(np.sum(~np.isfinite(fw))),
        "n_pj_missing": int(np.sum(~np.isfinite(pj))),
    }


def compute_ew_for_week(
    k: int,
    *,
    e_lag: float,
    data,
    coefs: dict[str, float],
    std: dict,
) -> tuple[float | None, dict[str, Any]]:
    """``(e_hat_k, info)`` from week ``k-1`` mediators and the check-in labelled ``k``.

    ``data`` is a :class:`RL_agents.rl.week_data.WeekData`. The check-in
    labelled ``k`` is the Sunday survey at the end of week ``k-1``
    (``J_k``, ``U_k``); ``None`` is returned when week ``k-1`` has no
    engagement data at all.
    """
    info: dict[str, Any] = {"k": int(k)}
    if not np.isfinite(e_lag):
        info["reason"] = "e_lag_missing"
        return None, info
    kp = int(k) - 1
    M_E = np.column_stack([data.pv[kp, :, 0], data.pv[kp, :, 1], data.wear[kp], data.pj[kp]])
    try:
        agg = ew_aggregates_from_me(M_E, std)
    except ValueError:
        info["reason"] = "mediators_missing"
        return None, info
    info.update(agg)
    J = data.J_w[k]
    info["J_missing"] = not np.isfinite(J)
    J = 0.0 if not np.isfinite(J) else float(J)
    u1, u2 = data.U1[k], data.U2[k]
    info["U_imputed"] = bool(J == 1.0 and not (np.isfinite(u1) and np.isfinite(u2)))
    u1 = float(coefs.get("U1_impute_mean", 0.0)) if not np.isfinite(u1) else float(u1)
    u2 = float(coefs.get("U2_impute_mean", 0.0)) if not np.isfinite(u2) else float(u2)
    half = J * (u1 + u2) / 14.0
    e_hat = (
        coefs["intercept"]
        + coefs["E_lag"] * float(e_lag)
        + coefs["PV_sum"] * agg["PV_sum"]
        + coefs["FW_sum"] * agg["FW_sum"]
        + coefs["PJ_sum"] * agg["PJ_sum"]
        + coefs["J_close"] * J
        + coefs["half_J_close"] * half
    )
    if not np.isfinite(e_hat):
        info["reason"] = "nonfinite"
        return None, info
    info["reason"] = "ok"
    return float(e_hat), info
