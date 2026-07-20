"""Agent-visible perceived-utility (E_w) approximation for the deployment handler.

Mirrors ``agents/ew_hat.py`` in the simulation code: a pooled, no-intercept
linear map from weekly observable aggregates to the agent-visible perceived
utility ``E_w``. The coefficients are estimated offline and stored in
``Ew_pooled_linear_coefs.json`` (five predictors, no intercept)::

    E_w_hat =  c_Jw * J_w
             + c_half * half_J_tool8
             + c_PV * PV_sum
             + c_FW * FW_sum
             + c_PJ * PJ_sum

Predictor definitions (must match how the server aggregates the raw data):

- ``J_w``          : week-present / survey indicator for the week (0 or 1).
- ``half_J_tool8`` : ``0.5 * J_w * ((U1 + 1) + (U2 + 1)) / 8`` where ``U1``/``U2``
                     are the two raw tool ratings (0 if missing). May be supplied
                     directly, or derived here from ``U1``/``U2``.
- ``PV_sum``       : mean of the 14 per-slot standardized pageview counts.
- ``FW_sum``       : mean over 7 days of the ``nextday_wearing`` indicator.
- ``PJ_sum``       : mean over 7 days of the ``daily_present`` indicator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from RL_agents.base import StoredValue
from RL_agents.logging_config import YELLOW, get_logger, log_alert

logger = get_logger(__name__)

EW_PREDICTORS = ("J_w", "half_J_tool8", "PV_sum", "FW_sum", "PJ_sum")
DEFAULT_EW_HAT = 2.0
DEFAULT_EW_STORED_VALUE_TYPE = "rl_E_w"


def load_ew_coefs(path: str | Path | None) -> dict[str, float] | None:
    """Load pooled E_w coefficients, or ``None`` if the file is absent."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with p.open(encoding="utf-8") as f:
        return {str(k): float(v) for k, v in json.load(f).items()}


def apply_pooled_coefs(
    coefs: dict[str, float],
    J_w: float,
    half_J_tool8: float,
    PV_sum: float,
    FW_sum: float,
    PJ_sum: float,
) -> float:
    """Apply the pooled linear approximation for agent-visible E_w."""
    return float(
        coefs["J_w"] * float(J_w)
        + coefs["half_J_tool8"] * float(half_J_tool8)
        + coefs["PV_sum"] * float(PV_sum)
        + coefs["FW_sum"] * float(FW_sum)
        + coefs["PJ_sum"] * float(PJ_sum)
    )


def _coerce_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    val = float(value)
    return default if not np.isfinite(val) else val


def _half_j_tool8(J_w: float, u1: float, u2: float) -> float:
    return 0.5 * float(J_w) * ((float(u1) + 1.0) + (float(u2) + 1.0)) / 8.0


def ew_predictors_from_inputs(inputs: dict) -> dict[str, float]:
    """Normalize an E_w-inputs dict into the five pooled predictors.

    ``inputs`` (assembled by the server from the database for the just-finalized
    week) must provide ``PV_sum``, ``FW_sum``, ``PJ_sum`` and ``J_w``. It may
    provide ``half_J_tool8`` directly, or the raw tool ratings ``U1``/``U2``
    (aliases ``Exp-tool-1``/``Exp-tool-2``) from which it is derived.
    """
    if inputs is None:
        raise KeyError("E_w inputs are missing")
    J_w = _coerce_number(inputs.get("J_w"), 0.0)
    if inputs.get("half_J_tool8") is not None:
        half = float(inputs["half_J_tool8"])
    else:
        u1 = _coerce_number(inputs.get("U1", inputs.get("Exp-tool-1")), 0.0)
        u2 = _coerce_number(inputs.get("U2", inputs.get("Exp-tool-2")), 0.0)
        half = _half_j_tool8(J_w, u1, u2)
    missing = [key for key in ("PV_sum", "FW_sum", "PJ_sum") if inputs.get(key) is None]
    if missing:
        raise KeyError(f"E_w inputs must include {missing}")
    return {
        "J_w": J_w,
        "half_J_tool8": half,
        "PV_sum": float(inputs["PV_sum"]),
        "FW_sum": float(inputs["FW_sum"]),
        "PJ_sum": float(inputs["PJ_sum"]),
    }


def compute_ew_hat(coefs: dict[str, float], inputs: dict) -> float:
    """Compute the agent-visible ``E_w_hat`` from weekly observable aggregates."""
    p = ew_predictors_from_inputs(inputs)
    return apply_pooled_coefs(
        coefs, p["J_w"], p["half_J_tool8"], p["PV_sum"], p["FW_sum"], p["PJ_sum"]
    )


def ew_stored_value_type(config: dict) -> str:
    return str(config.get("ew_stored_value_type", DEFAULT_EW_STORED_VALUE_TYPE))


def ew_inputs(observations: list, packet: dict | None) -> dict | None:
    """Locate weekly E_w predictor aggregates for the perceived-utility update.

    The server assembles these from the database for the just-finalized week.
    Accepted locations: ``packet["ew_inputs"]``, an observation payload with an
    ``ew_inputs`` dict, or an observation of type ``rl_ew_inputs``.
    """
    if packet is not None and isinstance(packet.get("ew_inputs"), dict):
        return packet["ew_inputs"]
    for obs in observations:
        payload = getattr(obs, "payload", {}) or {}
        if isinstance(payload.get("ew_inputs"), dict):
            return payload["ew_inputs"]
        if getattr(obs, "observation_type", None) == "rl_ew_inputs" and payload:
            return payload
    return None


def prepare_ew_update(
    config: dict,
    user_id: str,
    observations: list,
    packet: dict | None,
    *,
    ew_coefs: dict[str, float] | None,
    action_type: str,
) -> list[StoredValue] | None:
    """Weekly perceived-utility (E_w) update from pooled linear coefficients.

    Returns a stored value for the updated E_w (consumed by ``make_state`` for
    the next week's decisions), or ``None`` if coefficients or inputs are
    unavailable. Failures emit a yellow alert and return ``None``.
    """
    if ew_coefs is None:
        return None
    inputs = ew_inputs(observations, packet)
    if inputs is None:
        return None
    try:
        ew_hat = compute_ew_hat(ew_coefs, inputs)
    except Exception as exc:
        log_alert(
            logger,
            YELLOW,
            "perceived-utility (E_w) update failed",
            action_type=action_type,
            user_id=str(user_id),
            error=str(exc),
        )
        return None
    return [
        StoredValue(
            user_id=str(user_id),
            type=ew_stored_value_type(config),
            value=float(ew_hat),
        )
    ]
