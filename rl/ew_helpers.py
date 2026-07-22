"""Agent-visible perceived-utility (E_w) approximation for the deployment handler.

Mirrors ``agents/ew_hat.py`` / the paper proxy ``\\widehat e_w``: a pooled,
no-intercept linear map from weekly observables to agent-visible perceived
utility. Coefficients live in ``Ew_pooled_linear_coefs.json``::

    E_w_hat =  c_Jw * J_w
             + c_half * half_J_tool8
             + c_PV * PV_sum
             + c_FW * FW_sum
             + c_PJ * PJ_sum

which matches

    \\widehat e_w
        = ω_1 J_w^{week}
        + ω_2 J_w^{week} (U_{w,1}+U_{w,2}) / 14
        + ω_3 \\overline M_{w-1}^{E,PJ}
        + ω_4 \\overline M_{w-1}^{E,PV}
        + ω_5 \\overline M_{w-1}^{E,FW}

with ``half_J_tool8 = J_w * (U1 + U2) / 14`` for RCT tool ratings on the
1–7 scale (no shift). The coef key name ``half_J_tool8`` is historical.

Predictor sources (built by the handler from ``db_reader`` when not pre-assembled):

- ``J_w``, ``U1``/``U2`` (aliases ``Exp-tool-1``/``Exp-tool-2``): weekly check-in
- ``PV_sum``, ``FW_sum``, ``PJ_sum``: aggregates of ``M_E`` from week ``w-1``
  (``M_E`` cols 0–1 = pageview slots, col 2 = Fitbit wear, col 3 = daily present)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from RL_agents.base import DBReader, StoredValue
from RL_agents.logging_config import YELLOW, get_logger, log_alert
from RL_agents.rl.algorithm_helpers import N_RL_DAYS, N_RL_SLOTS
from RL_agents.rl.other_helpers import (
    deserialize_state,
    internal_week_index,
    state_index,
)

logger = get_logger(__name__)

EW_PREDICTORS = ("J_w", "half_J_tool8", "PV_sum", "FW_sum", "PJ_sum")
DEFAULT_EW_HAT = 2.0
DEFAULT_EW_STORED_VALUE_TYPE = "rl_E_w"

# Calendar used by the offline E_w fit (7 days × 2 slots).
W_DAYS = 7
FOURSC_SLOTS_PER_WEEK = W_DAYS * N_RL_SLOTS  # 14

# RL ``M_E`` (n_days × 4): PV morning/afternoon, Fitbit wear, daily present.
ME_COL_PV_MORNING = 0
ME_COL_PV_AFTERNOON = 1
ME_COL_FW = 2
ME_COL_PJ = 3
N_ME_COLS = 4

RL_EW_OBSERVATION_TYPES = (
    "rl_ew_inputs",
    "rl_weekly_checkin",
    "weekly_checkin",
    "rl_pf_week",
    "pf_week",
)


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
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    return default if not np.isfinite(val) else val


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _half_j_tool8(J_w: float, u1: float, u2: float) -> float:
    """``J * (U1 + U2) / 14`` with RCT tool ratings on 1–7 (no shift)."""
    return float(J_w) * (float(u1) + float(u2)) / 14.0


def ew_predictors_from_inputs(inputs: dict) -> dict[str, float]:
    """Normalize an E_w-inputs dict into the five pooled predictors.

    ``inputs`` must provide ``PV_sum``, ``FW_sum``, ``PJ_sum`` and ``J_w``.
    It may provide ``half_J_tool8`` directly, or raw tool ratings ``U1``/``U2``
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


def weekly_pv_sum_for_ew(slot_pv: Any) -> float:
    """Mean pageview summary over the 14 calendar slots (``nansum / 14``)."""
    v = np.asarray(slot_pv, dtype=float).ravel()
    if v.size == 0 or not np.any(np.isfinite(v)):
        return float("nan")
    return float(np.nansum(v) / float(FOURSC_SLOTS_PER_WEEK))


def _as_me_matrix(value: Any) -> np.ndarray | None:
    """Coerce ``M_E`` to shape ``(<=7, 4)``; return ``None`` if unusable."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.size == N_ME_COLS:
            arr = arr.reshape(1, N_ME_COLS)
        elif arr.size % N_ME_COLS == 0:
            arr = arr.reshape(-1, N_ME_COLS)
        else:
            return None
    if arr.ndim != 2 or arr.shape[1] < N_ME_COLS:
        return None
    return np.asarray(arr[:, :N_ME_COLS], dtype=float)


def ew_aggregates_from_me(M_E: Any) -> dict[str, float]:
    """Build ``PV_sum``, ``FW_sum``, ``PJ_sum`` from an ``M_E`` week matrix.

    ``M_E[d, 0:2]`` = pageview slots, ``M_E[d, 2]`` = Fitbit wear (FW),
    ``M_E[d, 3]`` = daily present (PJ). Values are padded to a 7-day / 14-slot
    calendar (Sunday zeros when only Mon–Sat RL days are present) and averaged
    with the offline denominators 7 and 14.
    """
    me = _as_me_matrix(M_E)
    if me is None or me.size == 0:
        raise ValueError("M_E is missing or malformed")

    n_days = min(int(me.shape[0]), W_DAYS)
    fw = np.full(W_DAYS, np.nan, dtype=float)
    pj = np.full(W_DAYS, np.nan, dtype=float)
    pv = np.full(FOURSC_SLOTS_PER_WEEK, np.nan, dtype=float)
    for d in range(n_days):
        fw[d] = me[d, ME_COL_FW]
        pj[d] = me[d, ME_COL_PJ]
        pv[d * N_RL_SLOTS + 0] = me[d, ME_COL_PV_MORNING]
        pv[d * N_RL_SLOTS + 1] = me[d, ME_COL_PV_AFTERNOON]

    # Missing trailing calendar days (e.g. Sunday) contribute 0, matching the
    # offline ``nansum / 7`` and ``nansum / 14`` summaries when those cells are 0.
    fw = np.where(np.isfinite(fw), fw, 0.0)
    pj = np.where(np.isfinite(pj), pj, 0.0)
    pv = np.where(np.isfinite(pv), pv, 0.0)

    return {
        "PV_sum": weekly_pv_sum_for_ew(pv),
        "FW_sum": float(np.sum(fw) / float(W_DAYS)),
        "PJ_sum": float(np.sum(pj) / float(W_DAYS)),
    }


def _payload_week_index(payload: dict) -> int | None:
    if payload.get("w") is not None:
        return internal_week_index(int(payload["w"]))
    if payload.get("rl_week") is not None:
        return internal_week_index(int(payload["rl_week"]))
    if payload.get("week") is not None:
        return internal_week_index(int(payload["week"]))
    if payload.get("k") is not None:
        return int(payload["k"])
    return None


def _merge_me(dst: np.ndarray, src: Any) -> None:
    me = _as_me_matrix(src)
    if me is None:
        return
    n_days = min(int(me.shape[0]), int(dst.shape[0]))
    for d in range(n_days):
        for c in range(N_ME_COLS):
            val = me[d, c]
            if np.isfinite(val):
                dst[d, c] = val


def _extract_checkin(payload: dict) -> dict[str, float | None]:
    """Pull J_w / U1 / U2 from a weekly check-in (or week) payload."""
    j = payload.get("J_w", payload.get("J_w_week", payload.get("wp", payload.get("week_present"))))
    u1 = payload.get("U1", payload.get("Exp-tool-1", payload.get("U_w_1")))
    u2 = payload.get("U2", payload.get("Exp-tool-2", payload.get("U_w_2")))
    # Nested check-in block.
    nested = payload.get("weekly_checkin", payload.get("checkin"))
    if isinstance(nested, dict):
        if j is None:
            j = nested.get("J_w", nested.get("wp", nested.get("week_present")))
        if u1 is None:
            u1 = nested.get("U1", nested.get("Exp-tool-1"))
        if u2 is None:
            u2 = nested.get("U2", nested.get("Exp-tool-2"))
    out: dict[str, float | None] = {
        "J_w": _optional_float(j),
        "U1": _optional_float(u1),
        "U2": _optional_float(u2),
    }
    # Completing the survey with ratings implies week-present when J is omitted.
    if out["J_w"] is None and (out["U1"] is not None or out["U2"] is not None):
        out["J_w"] = 1.0
    return out


def collect_ew_week_sources(
    db_reader: DBReader | None,
    *,
    action_type: str,
) -> dict[int, dict[str, Any]]:
    """Collect per-week ``M_E`` and weekly-check-in fields from ``db_reader``.

    Returns ``{k: {"M_E", "J_w", "U1", "U2", ...}}`` keyed by internal week.
    """
    by_k: dict[int, dict[str, Any]] = {}
    if db_reader is None:
        return by_k

    def bucket(k: int) -> dict[str, Any]:
        return by_k.setdefault(
            k,
            {
                "M_E": np.full((W_DAYS, N_ME_COLS), np.nan, dtype=float),
                "J_w": None,
                "U1": None,
                "U2": None,
                "PV_sum": None,
                "FW_sum": None,
                "PJ_sum": None,
                "half_J_tool8": None,
            },
        )

    for action in db_reader.get_actions(action_type=action_type) or []:
        try:
            state = deserialize_state(action.state or action.context or {})
            w, d, t = state_index(state)
            k = internal_week_index(w)
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS):
            continue
        entry = bucket(k)
        if state.get("M_E") is not None or state.get("M^E") is not None:
            _merge_me(entry["M_E"], state.get("M_E", state.get("M^E")))

    try:
        observations = db_reader.get_observations() or []
    except TypeError:
        observations = db_reader.get_observations or []

    for obs in observations:
        payload = getattr(obs, "payload", {}) or {}
        obs_type = getattr(obs, "observation_type", None)
        series_payload = payload.get("weekly_series", payload)
        candidates = [payload]
        if isinstance(series_payload, dict) and series_payload is not payload:
            candidates.append(series_payload)

        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            k = _payload_week_index(cand)
            if k is None and obs_type not in RL_EW_OBSERVATION_TYPES:
                # Accept check-in / M_E fields only when week-indexed or typed.
                if not any(
                    key in cand
                    for key in (
                        "M_E",
                        "M^E",
                        "ew_inputs",
                        "Exp-tool-1",
                        "Exp-tool-2",
                        "U1",
                        "U2",
                        "J_w",
                        "weekly_checkin",
                    )
                ):
                    continue
            if k is None:
                continue
            entry = bucket(k)

            if isinstance(cand.get("ew_inputs"), dict):
                ew_in = cand["ew_inputs"]
                for key in ("PV_sum", "FW_sum", "PJ_sum", "half_J_tool8", "J_w", "U1", "U2"):
                    if ew_in.get(key) is not None and entry.get(key) is None:
                        entry[key] = _optional_float(ew_in[key])
                if ew_in.get("Exp-tool-1") is not None and entry["U1"] is None:
                    entry["U1"] = _optional_float(ew_in["Exp-tool-1"])
                if ew_in.get("Exp-tool-2") is not None and entry["U2"] is None:
                    entry["U2"] = _optional_float(ew_in["Exp-tool-2"])

            if cand.get("M_E") is not None or cand.get("M^E") is not None:
                _merge_me(entry["M_E"], cand.get("M_E", cand.get("M^E")))

            for key in ("PV_sum", "FW_sum", "PJ_sum", "half_J_tool8"):
                if cand.get(key) is not None and entry.get(key) is None:
                    entry[key] = _optional_float(cand[key])

            checkin = _extract_checkin(cand)
            for key in ("J_w", "U1", "U2"):
                if checkin[key] is not None:
                    entry[key] = checkin[key]

        if obs_type == "rl_ew_inputs" and isinstance(payload, dict) and payload:
            k = _payload_week_index(payload)
            if k is not None:
                entry = bucket(k)
                for key in ("PV_sum", "FW_sum", "PJ_sum", "half_J_tool8", "J_w", "U1", "U2"):
                    if payload.get(key) is not None:
                        entry[key] = _optional_float(payload[key])
                if payload.get("Exp-tool-1") is not None:
                    entry["U1"] = _optional_float(payload["Exp-tool-1"])
                if payload.get("Exp-tool-2") is not None:
                    entry["U2"] = _optional_float(payload["Exp-tool-2"])

    return by_k


def build_ew_inputs_for_week(sources: dict[str, Any]) -> dict[str, float] | None:
    """Assemble one week's ``ew_inputs`` dict from collected sources."""
    if sources is None:
        return None

    # Prefer explicit aggregates when the server already summarized M^E.
    pv = sources.get("PV_sum")
    fw = sources.get("FW_sum")
    pj = sources.get("PJ_sum")
    if pv is None or fw is None or pj is None:
        me = sources.get("M_E")
        if me is None or not np.any(np.isfinite(np.asarray(me, dtype=float))):
            return None
        try:
            agg = ew_aggregates_from_me(me)
        except ValueError:
            return None
        pv = agg["PV_sum"] if pv is None else pv
        fw = agg["FW_sum"] if fw is None else fw
        pj = agg["PJ_sum"] if pj is None else pj

    j_w = sources.get("J_w")
    u1 = sources.get("U1")
    u2 = sources.get("U2")
    if j_w is None:
        j_w = 1.0 if (u1 is not None or u2 is not None) else 0.0

    inputs: dict[str, float] = {
        "J_w": float(j_w),
        "PV_sum": float(pv),
        "FW_sum": float(fw),
        "PJ_sum": float(pj),
        "U1": float(u1) if u1 is not None else 0.0,
        "U2": float(u2) if u2 is not None else 0.0,
        "Exp-tool-1": float(u1) if u1 is not None else 0.0,
        "Exp-tool-2": float(u2) if u2 is not None else 0.0,
    }
    if sources.get("half_J_tool8") is not None:
        inputs["half_J_tool8"] = float(sources["half_J_tool8"])
    else:
        inputs["half_J_tool8"] = _half_j_tool8(
            inputs["J_w"], inputs["U1"], inputs["U2"]
        )
    return inputs


def build_ew_inputs_from_reader(
    db_reader: DBReader | None,
    *,
    action_type: str,
    k: int | None = None,
    me_week_k: int | None = None,
) -> dict[str, float] | None:
    """Build ``ew_inputs`` for ``\\widehat e_w`` from ``db_reader``.

    Uses engagement summaries ``M_E`` from week ``me_week_k`` (default ``k-1``
    when ``k`` is the belief / next-decision week) and weekly-check-in fields
    ``J_w``, ``U1``, ``U2`` from week ``k`` (falling back to the M_E week).
    """
    by_k = collect_ew_week_sources(db_reader, action_type=action_type)
    if not by_k:
        return None

    if k is None and me_week_k is None:
        me_week_k = max(by_k)
        k = me_week_k + 1
    elif me_week_k is None:
        me_week_k = int(k) - 1
    elif k is None:
        k = int(me_week_k) + 1

    me_week_k = int(me_week_k)
    k = int(k)
    if me_week_k < 0 or me_week_k not in by_k:
        # Fall back to the latest week that has any M_E / aggregates.
        candidates = sorted(by_k)
        me_week_k = candidates[-1]
        k = me_week_k + 1

    me_src = dict(by_k[me_week_k])
    checkin_src = by_k.get(k) or by_k.get(me_week_k) or {}
    for key in ("J_w", "U1", "U2", "half_J_tool8"):
        if checkin_src.get(key) is not None:
            me_src[key] = checkin_src[key]
    return build_ew_inputs_for_week(me_src)


def ew_inputs(observations: list, packet: dict | None) -> dict | None:
    """Locate pre-assembled weekly E_w predictor aggregates if the server sent them.

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
    db_reader: DBReader | None = None,
) -> list[StoredValue] | None:
    """Weekly perceived-utility (E_w) update from pooled linear coefficients.

    Prefers a pre-assembled ``ew_inputs`` dict; otherwise builds one from
    ``M_E`` summaries and weekly-check-in questions in ``db_reader``. Returns a
    stored value for the updated E_w, or ``None`` if coefficients or inputs are
    unavailable. Failures emit a yellow alert and return ``None``.
    """
    if ew_coefs is None:
        return None
    inputs = ew_inputs(observations, packet)
    if inputs is None and db_reader is not None:
        packet_k = None
        if packet is not None and packet.get("k") is not None:
            packet_k = int(packet["k"])
        try:
            inputs = build_ew_inputs_from_reader(
                db_reader,
                action_type=action_type,
                k=packet_k,
            )
        except Exception as exc:
            log_alert(
                logger,
                YELLOW,
                "E_w inputs build from db_reader failed",
                action_type=action_type,
                user_id=str(user_id),
                error=str(exc),
            )
            inputs = None
        if inputs is not None and packet is not None:
            packet["ew_inputs"] = inputs
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
