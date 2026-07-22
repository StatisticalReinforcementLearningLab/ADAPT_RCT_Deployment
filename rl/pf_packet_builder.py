"""Build PF packets from ``db_reader`` using general names (C, M_Y, A, E_w, Y, …).

Assembles the existing ``pf_data`` contract (fourSC dim 25, antic dim 19, CAE dim 24)
from actions and observation payloads so ``prepare_for_update`` does not require a
server-prebuilt design matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from RL_agents.base import DBReader
from RL_agents.logging_config import YELLOW, get_logger, log_alert
from RL_agents.rl.algorithm_helpers import N_RL_DAYS, N_RL_SLOTS
from RL_agents.rl.ew_helpers import ew_stored_value_type
from RL_agents.rl.other_helpers import (
    deserialize_state,
    internal_week_index,
    resolve_y1,
    state_index,
    stored_float,
)
from RL_agents.rl.pf_helpers import (
    PF_ANTIC_DIM,
    PF_CAE_DIM,
    PF_FOURSC_DIM,
    infer_pf_packet_k,
)
from RL_agents.rl.standardization import standardize_cae, standardize_pf_series

logger = get_logger(__name__)

# PF mediator grid: Mon–Sat × 2 slots (matches simulation N_MED_*).
N_MED_SLOT = N_RL_DAYS * N_RL_SLOTS  # 12
N_MED_ANTIC_DAY = N_RL_DAYS  # 6
# CAE transition feature uses a 7×2 / 7-day calendar (Sunday padded with 0).
FOURSC_SLOTS_PER_WEEK = 14
W_DAYS = 7
ANTIC_LAG1_COL = 1
_FOURSC_CAE_COL = 13
_FOURSC_AH_COL = 14
_FOURSC_AH_CAE_COL = 24
_ANTIC_CAE_COL = 7
_ANTIC_WS_M_COL = 8
_ANTIC_WS_A_COL = 9
_ANTIC_CAE_WS_M_COL = 17
_ANTIC_CAE_WS_A_COL = 18
_CAE_AR1_COL = 1

RL_PF_WEEK_OBSERVATION_TYPES = ("rl_pf_week", "pf_week")

# RLSVI C (length 9) → fourSC baseline columns.
# C[0]=yesterdaySC, C[1]=SC7dEma, C[2]=prior2h, C[3]=actCompleted7d,
# C[4]=activeDays7d, C[5]=suggestions7d, C[6]=salienceY, C[7]=interact7d, C[8]=pad/wear.
C_IDX_YESTERDAY_SC = 0
C_IDX_SC_EMA = 1
C_IDX_PRIOR2H = 2
C_IDX_ACT_COMPLETED = 3
C_IDX_ACTIVE_DAYS = 4
C_IDX_SUGGESTIONS = 5
C_IDX_SALIENCE_Y = 6
C_IDX_INTERACT = 7
C_IDX_WEAR_OR_PAD = 8


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(val) else val


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) else None


def _as_c_vector(value: Any, *, n: int = 9) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    out = np.zeros(n, dtype=float)
    m = min(int(arr.size), n)
    if m:
        out[:m] = arr[:m]
    return out


def day_of_week_norm(d: int) -> float:
    """Normalize Mon–Sat day index ``d`` in ``[0, 5]`` to ``[0, 1]``."""
    denom = max(N_RL_DAYS - 1, 1)
    return float(d) / float(denom)


def fourSC_cae_delta(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    d = np.zeros_like(x)
    d[_FOURSC_CAE_COL] = 1.0
    d[_FOURSC_AH_CAE_COL] = x[_FOURSC_AH_COL]
    return d


def antic_cae_delta(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    d = np.zeros_like(x)
    d[_ANTIC_CAE_COL] = 1.0
    d[_ANTIC_CAE_WS_M_COL] = x[_ANTIC_WS_M_COL]
    d[_ANTIC_CAE_WS_A_COL] = x[_ANTIC_WS_A_COL]
    return d


def foursc_row_from_general(
    C: Any,
    E_w: float,
    A: float,
    *,
    d: int,
    t: int,
    wear7: float | None = None,
    cae: float = 0.0,
) -> np.ndarray:
    """Build a length-25 fourSC base row from general-name covariates.

    ``C`` unpack map (RLSVI context vector)::

        C[0] yesterdayStepCount
        C[1] stepCountLast7DaysEma
        C[2] prior2HourStepCount
        C[3] activityCompletedLast7Days
        C[4] activeDaysLast7Days
        C[5] activitySuggestionsSentLast7Days
        C[6] salienceMessageSentYesterday
        C[7] activitySuggestionInteractLast7Days
        C[8] morningFitbitWearLast7Days when ``wear7`` is None

    ``dayOfWeekNorm`` / ``decisionTimeSlot`` come from ``(d, t)``.
    Pass ``cae=0`` for PF base rows; particles add CAE via ``fourSC_cae_delta``.
    """
    c = _as_c_vector(C)
    wear = (
        float(wear7)
        if wear7 is not None and np.isfinite(float(wear7))
        else float(c[C_IDX_WEAR_OR_PAD])
    )
    day_n = day_of_week_norm(int(d))
    slot = float(t)
    pu = float(E_w)
    ah = float(A)
    ysc = float(c[C_IDX_YESTERDAY_SC])
    ema = float(c[C_IDX_SC_EMA])
    p2h = float(c[C_IDX_PRIOR2H])
    act_c = float(c[C_IDX_ACT_COMPLETED])
    sug = float(c[C_IDX_SUGGESTIONS])
    sal = float(c[C_IDX_SALIENCE_Y])
    interact = float(c[C_IDX_INTERACT])
    active = float(c[C_IDX_ACTIVE_DAYS])
    _cae = float(cae)
    row = np.array(
        [
            1.0,
            ysc,
            ema,
            p2h,
            act_c,
            sug,
            wear,
            sal,
            interact,
            active,
            day_n,
            slot,
            pu,
            _cae,
            ah,
            ah * ysc,
            ah * p2h,
            ah * sug,
            ah * wear,
            ah * sal,
            ah * interact,
            ah * day_n,
            ah * slot,
            ah * pu,
            ah * _cae,
        ],
        dtype=float,
    )
    if row.size != PF_FOURSC_DIM:
        raise ValueError(f"fourSC row dim {row.size} != {PF_FOURSC_DIM}")
    return row


def antic_row_from_general(
    *,
    antic_lag: float,
    today_step_count: float,
    recorded_pa_today: float,
    activity_status_today: float,
    salience_today: float,
    d: int,
    E_w: float,
    ws_morning: float,
    ws_afternoon: float,
    cae: float = 0.0,
) -> np.ndarray:
    """Build a length-19 antic base row (CAE slot zeroed for PF base)."""
    day_n = day_of_week_norm(int(d))
    pu = float(E_w)
    _cae = float(cae)
    ws_m = float(ws_morning)
    ws_a = float(ws_afternoon)
    sal = float(salience_today)
    row = np.array(
        [
            1.0,
            float(antic_lag),
            float(today_step_count),
            float(recorded_pa_today),
            float(activity_status_today),
            sal,
            day_n,
            pu,
            _cae,
            ws_m,
            ws_a,
            ws_m * sal,
            ws_a * sal,
            ws_m * day_n,
            ws_a * day_n,
            ws_m * pu,
            ws_a * pu,
            ws_m * _cae,
            ws_a * _cae,
        ],
        dtype=float,
    )
    if row.size != PF_ANTIC_DIM:
        raise ValueError(f"antic row dim {row.size} != {PF_ANTIC_DIM}")
    return row


def cae_row_from_general(
    *,
    week_norm: float,
    foursc_wk: np.ndarray,
    antic_wk: np.ndarray,
    cae_lastweek: float = 0.0,
) -> np.ndarray:
    """Build length-24 CAE transition row (AR CAE slot zeroed for PF base)."""
    four = np.asarray(foursc_wk, dtype=float).ravel()
    antic = np.asarray(antic_wk, dtype=float).ravel()
    if four.size != FOURSC_SLOTS_PER_WEEK:
        padded = np.zeros(FOURSC_SLOTS_PER_WEEK, dtype=float)
        padded[: min(four.size, FOURSC_SLOTS_PER_WEEK)] = four[
            : min(four.size, FOURSC_SLOTS_PER_WEEK)
        ]
        four = padded
    if antic.size != W_DAYS:
        padded_a = np.zeros(W_DAYS, dtype=float)
        padded_a[: min(antic.size, W_DAYS)] = antic[: min(antic.size, W_DAYS)]
        antic = padded_a
    row = np.concatenate(
        [[1.0, float(cae_lastweek), float(week_norm)], four, antic]
    )
    if row.size != PF_CAE_DIM:
        raise ValueError(f"CAE row dim {row.size} != {PF_CAE_DIM}")
    return row


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


def _extract_my_slot_and_antic(M_Y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Split RL ``M_Y`` (n_days × 3) into fourSC slots (12,) and antic days (6,)."""
    my = np.asarray(M_Y, dtype=float)
    four = np.full(N_MED_SLOT, np.nan, dtype=float)
    antic = np.full(N_MED_ANTIC_DAY, np.nan, dtype=float)
    if my.ndim == 1:
        # Flat 12 fourSC values, or 12+6 concat.
        if my.size >= N_MED_SLOT:
            four[:] = my[:N_MED_SLOT]
        if my.size >= N_MED_SLOT + N_MED_ANTIC_DAY:
            antic[:] = my[N_MED_SLOT : N_MED_SLOT + N_MED_ANTIC_DAY]
        return four, antic
    if my.ndim >= 2:
        nrows = min(my.shape[0], N_RL_DAYS)
        ncols = my.shape[1]
        for d in range(nrows):
            if ncols >= 1:
                four[d * N_RL_SLOTS + 0] = my[d, 0]
            if ncols >= 2:
                four[d * N_RL_SLOTS + 1] = my[d, 1]
            if ncols >= 3:
                antic[d] = my[d, 2]
    return four, antic


def _payload_day_slot(payload: dict) -> tuple[int | None, int | None]:
    """Optional 0-indexed ``(d, t)`` from an observation payload."""
    d = payload.get("d", payload.get("day"))
    t = payload.get("t", payload.get("slot"))
    try:
        d_out = None if d is None else int(d)
        t_out = None if t is None else int(t)
    except (TypeError, ValueError):
        return None, None
    return d_out, t_out


def _ingest_c_into_slots(
    slot_data: dict[tuple[int, int, int], dict[str, Any]],
    *,
    k: int,
    C_value: Any,
    d: int | None = None,
    t: int | None = None,
) -> None:
    """Write server-provided ``C`` into ``slot_data`` (never from action state).

    Accepted shapes:
    - length-9 vector with ``(d, t)`` → one slot
    - ``(N_RL_DAYS, N_RL_SLOTS, 9)`` week grid
    - ``(N_MED_SLOT, 9)`` flattened Mon–Sat slots
    """
    if C_value is None:
        return
    C_arr = np.asarray(C_value, dtype=float)
    if (
        d is not None
        and t is not None
        and 0 <= int(d) < N_RL_DAYS
        and 0 <= int(t) < N_RL_SLOTS
        and C_arr.ndim == 1
    ):
        slot_data.setdefault((k, int(d), int(t)), {})["C"] = _as_c_vector(C_arr)
        return
    if C_arr.ndim == 3 and C_arr.shape[:2] == (N_RL_DAYS, N_RL_SLOTS):
        for dd in range(N_RL_DAYS):
            for tt in range(N_RL_SLOTS):
                slot_data.setdefault((k, dd, tt), {})["C"] = _as_c_vector(C_arr[dd, tt])
        return
    if C_arr.ndim == 2 and C_arr.shape[0] == N_MED_SLOT:
        for s in range(N_MED_SLOT):
            dd, tt = divmod(s, N_RL_SLOTS)
            slot_data.setdefault((k, dd, tt), {})["C"] = _as_c_vector(C_arr[s])


def collect_pf_week_series(
    db_reader: DBReader | None,
    *,
    action_type: str,
    config: dict | None = None,
) -> dict[str, Any]:
    """Collect general-name weekly series from actions and observations.

    ``A`` comes from action records (the realized decisions). ``C`` and ``M_Y``
    are read only from observation / retrieved-data payloads the server streams
    as covariates / outcomes become available — never from action-time ``state``
    (that copy is kept only for action-selection replicability).

    Returns a dict with arrays indexed by internal week ``k``::

        n_weeks, A, C, wear7, E_w, M_Y_fourSC, M_Y_antic,
        antic_cov (dict of day-level arrays), Y, Y_tilde, I_w, J_w
    """
    config = config or {}
    empty = {
        "n_weeks": 0,
        "A": np.zeros((0, N_RL_DAYS, N_RL_SLOTS), dtype=float),
        "C": np.zeros((0, N_RL_DAYS, N_RL_SLOTS, 9), dtype=float),
        "C_filled": np.zeros((0, N_RL_DAYS, N_RL_SLOTS), dtype=bool),
        "wear7": np.full((0, N_RL_DAYS, N_RL_SLOTS), np.nan, dtype=float),
        "E_w": np.full(0, np.nan, dtype=float),
        "M_Y_fourSC": np.full((0, N_MED_SLOT), np.nan, dtype=float),
        "M_Y_antic": np.full((0, N_MED_ANTIC_DAY), np.nan, dtype=float),
        "antic_cov": {
            "today_step_count": np.zeros((0, N_MED_ANTIC_DAY), dtype=float),
            "recorded_pa_today": np.zeros((0, N_MED_ANTIC_DAY), dtype=float),
            "activity_status_today": np.zeros((0, N_MED_ANTIC_DAY), dtype=float),
            "salience_today": np.zeros((0, N_MED_ANTIC_DAY), dtype=float),
        },
        "Y": np.full(0, np.nan, dtype=float),
        "Y_tilde": np.full(0, np.nan, dtype=float),
        "I_w": np.ones(0, dtype=float),
        "J_w": np.ones(0, dtype=float),
    }
    if db_reader is None:
        return empty

    max_k = -1
    slot_data: dict[tuple[int, int, int], dict[str, Any]] = {}
    week_data: dict[int, dict[str, Any]] = {}

    # --- actions: realized A (+ week span). Do NOT read C from action state. ---
    for action in db_reader.get_actions(action_type=action_type) or []:
        try:
            state = deserialize_state(action.state or action.context or {})
            w, d, t = state_index(state)
            k = internal_week_index(w)
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS):
            continue
        max_k = max(max_k, k)
        entry = slot_data.setdefault((k, d, t), {})
        entry["A"] = float(action.action)
        # E_w from action state is a soft fallback only. C and M_Y for the PF
        # packet come from observations (completed covariates / outcomes), not
        # action-time state (kept for action-selection replicability).
        if state.get("E_w") is not None or state.get("perceivedUtility") is not None:
            entry["E_w"] = _coerce_float(
                state.get("E_w", state.get("perceivedUtility")), np.nan
            )

    # --- observations: C as it becomes available, plus weekly PF fields ---
    try:
        observations = db_reader.get_observations() or []
    except TypeError:
        observations = db_reader.get_observations or []

    for obs in observations:
        payload = getattr(obs, "payload", {}) or {}
        obs_type = getattr(obs, "observation_type", None)
        # Nested series under retrieved-style keys.
        series_payload = payload.get("weekly_series", payload)
        src = series_payload if isinstance(series_payload, dict) else payload
        k = _payload_week_index(payload)
        if k is None:
            k = _payload_week_index(src)
        d_slot, t_slot = _payload_day_slot(payload)
        if d_slot is None and t_slot is None:
            d_slot, t_slot = _payload_day_slot(src)

        # Per-slot / streaming C: accept as soon as (week, day, slot) + C land.
        has_c = src.get("C") is not None or payload.get("C") is not None
        if k is not None and has_c:
            max_k = max(max_k, k)
            _ingest_c_into_slots(
                slot_data,
                k=k,
                C_value=src.get("C", payload.get("C")),
                d=d_slot,
                t=t_slot,
            )
            wear_src = src.get("wear7", src.get("morningFitbitWearLast7Days"))
            if wear_src is None:
                wear_src = payload.get("wear7", payload.get("morningFitbitWearLast7Days"))
            if wear_src is not None:
                wear_arr = np.asarray(wear_src, dtype=float)
                if (
                    d_slot is not None
                    and t_slot is not None
                    and wear_arr.ndim == 0
                    and 0 <= d_slot < N_RL_DAYS
                    and 0 <= t_slot < N_RL_SLOTS
                ):
                    slot_data.setdefault((k, d_slot, t_slot), {})["wear7"] = float(
                        wear_arr
                    )
                elif wear_arr.ndim == 0:
                    for dd in range(N_RL_DAYS):
                        for tt in range(N_RL_SLOTS):
                            slot_data.setdefault((k, dd, tt), {})["wear7"] = float(
                                wear_arr
                            )
                elif wear_arr.ndim == 2:
                    for dd in range(min(N_RL_DAYS, wear_arr.shape[0])):
                        for tt in range(min(N_RL_SLOTS, wear_arr.shape[1])):
                            slot_data.setdefault((k, dd, tt), {})["wear7"] = float(
                                wear_arr[dd, tt]
                            )

        if k is None and obs_type not in RL_PF_WEEK_OBSERVATION_TYPES:
            # Still accept outcome-only observations with action_id linkage later.
            if any(
                key in payload
                for key in (
                    "Y",
                    "CAE",
                    "Y_prev",
                    "Y_tilde",
                    "tilde_Y",
                    "CAE_short",
                    "tY_prev",
                    "I_w",
                    "J_w",
                    "M_Y",
                    "M^Y",
                )
            ):
                # Without week index, skip structured week fill.
                continue
            continue
        if k is None:
            continue
        max_k = max(max_k, k)
        wd = week_data.setdefault(k, {})

        for y_key in ("Y", "CAE", "Y_prev", "Y_w", "cae"):
            if src.get(y_key) is not None:
                wd["Y"] = _optional_float(src.get(y_key))
                break
        for ty_key in ("Y_tilde", "tilde_Y", "tilde_Y_w", "CAE_short", "cae_short", "tY_prev"):
            if src.get(ty_key) is not None:
                wd["Y_tilde"] = _optional_float(src.get(ty_key))
                break
        if src.get("I_w") is not None or src.get("query_action") is not None:
            wd["I_w"] = int(src.get("I_w", src.get("query_action", 1)))
        if src.get("J_w") is not None or src.get("wp") is not None:
            wd["J_w"] = int(src.get("J_w", src.get("wp", 1)))
        if src.get("E_w") is not None or src.get("perceivedUtility") is not None:
            wd["E_w"] = _optional_float(src.get("E_w", src.get("perceivedUtility")))

        my = src.get("M_Y", src.get("M^Y"))
        if my is not None:
            four, antic = _extract_my_slot_and_antic(my)
            wd["M_Y_fourSC"] = four
            wd["M_Y_antic"] = antic
        if src.get("stepCountNext4Hour") is not None:
            arr = np.asarray(src["stepCountNext4Hour"], dtype=float).ravel()
            four = np.full(N_MED_SLOT, np.nan, dtype=float)
            four[: min(arr.size, N_MED_SLOT)] = arr[: min(arr.size, N_MED_SLOT)]
            wd["M_Y_fourSC"] = four
        if src.get("dailyAnticipatedAffect") is not None:
            arr = np.asarray(src["dailyAnticipatedAffect"], dtype=float).ravel()
            antic = np.full(N_MED_ANTIC_DAY, np.nan, dtype=float)
            antic[: min(arr.size, N_MED_ANTIC_DAY)] = arr[
                : min(arr.size, N_MED_ANTIC_DAY)
            ]
            wd["M_Y_antic"] = antic

        if src.get("A") is not None:
            A_arr = np.asarray(src["A"], dtype=float)
            if A_arr.ndim == 2 and A_arr.shape[:2] == (N_RL_DAYS, N_RL_SLOTS):
                for d in range(N_RL_DAYS):
                    for t in range(N_RL_SLOTS):
                        slot_data.setdefault((k, d, t), {})["A"] = float(A_arr[d, t])
            elif A_arr.ndim == 1 and A_arr.size >= N_MED_SLOT:
                for s in range(N_MED_SLOT):
                    d, t = divmod(s, N_RL_SLOTS)
                    slot_data.setdefault((k, d, t), {})["A"] = float(A_arr[s])

        antic_cov = src.get("antic_covariates", src.get("antic_C", {})) or {}
        if isinstance(antic_cov, dict):
            wd.setdefault("antic_cov", {})
            for key, alias in (
                ("today_step_count", ("todayStepCount", "today_step_count")),
                (
                    "recorded_pa_today",
                    ("recordedPhysicalActivityToday", "recorded_pa_today"),
                ),
                ("activity_status_today", ("activityStatusToday", "activity_status_today")),
                ("salience_today", ("salienceMessageSentToday", "salience_today")),
            ):
                for name in alias:
                    if antic_cov.get(name) is not None or src.get(name) is not None:
                        arr = np.asarray(
                            antic_cov.get(name, src.get(name)), dtype=float
                        ).ravel()
                        out = np.zeros(N_MED_ANTIC_DAY, dtype=float)
                        out[: min(arr.size, N_MED_ANTIC_DAY)] = arr[
                            : min(arr.size, N_MED_ANTIC_DAY)
                        ]
                        wd["antic_cov"][key] = out
                        break

    # Optional batch C from prepare-time retrieved_data (server join).
    try:
        retrieved = db_reader.get_retrieved_data() or {}
    except Exception:  # noqa: BLE001 - retrieved_data is optional
        retrieved = {}
    for key in ("C", "rl_C", "pf_C"):
        batch_c = retrieved.get(key)
        if batch_c is None:
            continue
        # Expected: {w or k: array} or a single week array with retrieved week index.
        if isinstance(batch_c, dict):
            for week_key, C_value in batch_c.items():
                try:
                    wk = int(week_key)
                except (TypeError, ValueError):
                    continue
                # Keys may be 1-indexed study weeks or internal k; prefer w>=1 as study week.
                k_batch = internal_week_index(wk) if wk >= 1 else int(wk)
                max_k = max(max_k, k_batch)
                _ingest_c_into_slots(slot_data, k=k_batch, C_value=C_value)
        else:
            k_batch = _payload_week_index(retrieved)
            if k_batch is not None:
                max_k = max(max_k, k_batch)
                _ingest_c_into_slots(slot_data, k=k_batch, C_value=batch_c)
        break

    # Fallback E_w from stored value when weeks lack it.
    stored_ew = stored_float(db_reader, ew_stored_value_type(config))

    n_weeks = max_k + 1 if max_k >= 0 else 0
    if n_weeks == 0:
        return empty

    A = np.zeros((n_weeks, N_RL_DAYS, N_RL_SLOTS), dtype=float)
    C = np.zeros((n_weeks, N_RL_DAYS, N_RL_SLOTS, 9), dtype=float)
    C_filled = np.zeros((n_weeks, N_RL_DAYS, N_RL_SLOTS), dtype=bool)
    wear7 = np.full((n_weeks, N_RL_DAYS, N_RL_SLOTS), np.nan, dtype=float)
    E_w = np.full(n_weeks, np.nan, dtype=float)
    M_Y_fourSC = np.full((n_weeks, N_MED_SLOT), np.nan, dtype=float)
    M_Y_antic = np.full((n_weeks, N_MED_ANTIC_DAY), np.nan, dtype=float)
    antic_cov = {
        "today_step_count": np.zeros((n_weeks, N_MED_ANTIC_DAY), dtype=float),
        "recorded_pa_today": np.zeros((n_weeks, N_MED_ANTIC_DAY), dtype=float),
        "activity_status_today": np.zeros((n_weeks, N_MED_ANTIC_DAY), dtype=float),
        "salience_today": np.zeros((n_weeks, N_MED_ANTIC_DAY), dtype=float),
    }
    Y = np.full(n_weeks, np.nan, dtype=float)
    Y_tilde = np.full(n_weeks, np.nan, dtype=float)
    I_w = np.ones(n_weeks, dtype=float)
    J_w = np.ones(n_weeks, dtype=float)

    for (k, d, t), entry in slot_data.items():
        if k >= n_weeks:
            continue
        if "A" in entry:
            A[k, d, t] = entry["A"]
        if "C" in entry:
            C[k, d, t] = entry["C"]
            C_filled[k, d, t] = True
        if "wear7" in entry and np.isfinite(entry["wear7"]):
            wear7[k, d, t] = entry["wear7"]
        if "E_w" in entry and np.isfinite(entry["E_w"]):
            E_w[k] = entry["E_w"]

    for k, wd in week_data.items():
        if k >= n_weeks:
            continue
        if "E_w" in wd and wd["E_w"] is not None:
            E_w[k] = float(wd["E_w"])
        if "Y" in wd and wd["Y"] is not None:
            Y[k] = float(wd["Y"])
        if "Y_tilde" in wd and wd["Y_tilde"] is not None:
            Y_tilde[k] = float(wd["Y_tilde"])
        if "I_w" in wd:
            I_w[k] = float(wd["I_w"])
        if "J_w" in wd:
            J_w[k] = float(wd["J_w"])
        if "M_Y_fourSC" in wd:
            M_Y_fourSC[k] = wd["M_Y_fourSC"]
        if "M_Y_antic" in wd:
            M_Y_antic[k] = wd["M_Y_antic"]
        for key in antic_cov:
            if "antic_cov" in wd and key in wd["antic_cov"]:
                antic_cov[key][k] = wd["antic_cov"][key]

    if stored_ew is not None:
        for k in range(n_weeks):
            if not np.isfinite(E_w[k]):
                E_w[k] = float(stored_ew)

    return {
        "n_weeks": n_weeks,
        "A": A,
        "C": C,
        "C_filled": C_filled,
        "wear7": wear7,
        "E_w": E_w,
        "M_Y_fourSC": M_Y_fourSC,
        "M_Y_antic": M_Y_antic,
        "antic_cov": antic_cov,
        "Y": Y,
        "Y_tilde": Y_tilde,
        "I_w": I_w,
        "J_w": J_w,
    }


def _week_norm(sim_w: int, nweek: int) -> float:
    denom = max(int(nweek) - 1, 1)
    return float(sim_w) / float(denom)


def _pad_foursc_week(four_12: np.ndarray) -> np.ndarray:
    """Pad Mon–Sat (12,) mediator values to 7×2 (14,) with Sunday zeros."""
    out = np.zeros(FOURSC_SLOTS_PER_WEEK, dtype=float)
    arr = np.asarray(four_12, dtype=float).ravel()
    n = min(arr.size, N_MED_SLOT)
    finite = np.where(np.isfinite(arr[:n]), arr[:n], 0.0)
    out[:n] = finite
    return out


def _pad_antic_week(antic_6: np.ndarray) -> np.ndarray:
    out = np.zeros(W_DAYS, dtype=float)
    arr = np.asarray(antic_6, dtype=float).ravel()
    n = min(arr.size, N_MED_ANTIC_DAY)
    finite = np.where(np.isfinite(arr[:n]), arr[:n], 0.0)
    out[:n] = finite
    return out


def build_pf_data_from_series(
    k: int,
    series: dict[str, Any],
    *,
    cae_baseline: float,
    nweek: int | None = None,
) -> dict[str, Any]:
    """Assemble ``pf_data`` for belief week ``k`` from a general-name series."""
    sim_w_prev = int(k) - 1
    if sim_w_prev < 0:
        raise ValueError("build_pf_data_from_series requires k >= 1")
    n_weeks = int(series["n_weeks"])
    if sim_w_prev >= n_weeks:
        raise KeyError(
            f"series has n_weeks={n_weeks} but need finalized week sim_w_prev={sim_w_prev}"
        )
    nweek = int(nweek if nweek is not None else max(n_weeks, sim_w_prev + 1))

    A = series["A"]
    C = series["C"]
    wear7 = series["wear7"]
    E_w = series["E_w"]
    M_Y_fourSC = series["M_Y_fourSC"]
    M_Y_antic = series["M_Y_antic"]
    antic_cov = series["antic_cov"]
    Y = series["Y"]
    Y_tilde = series["Y_tilde"]
    J_w = series["J_w"]

    def fourSC_row_fn(sw: int, d: int, t: int) -> np.ndarray:
        wear = wear7[sw, d, t]
        wear_arg = float(wear) if np.isfinite(wear) else None
        ew = E_w[sw] if np.isfinite(E_w[sw]) else 0.0
        return foursc_row_from_general(
            C[sw, d, t],
            ew,
            A[sw, d, t],
            d=d,
            t=t,
            wear7=wear_arg,
            cae=0.0,
        )

    def antic_row_fn(sw: int, d: int) -> np.ndarray:
        lag = 0.0
        if d > 0 and np.isfinite(M_Y_antic[sw, d - 1]):
            lag = float(M_Y_antic[sw, d - 1])
        elif sw > 0 and d == 0 and np.isfinite(M_Y_antic[sw - 1, N_MED_ANTIC_DAY - 1]):
            lag = float(M_Y_antic[sw - 1, N_MED_ANTIC_DAY - 1])
        ew = E_w[sw] if np.isfinite(E_w[sw]) else 0.0
        return antic_row_from_general(
            antic_lag=lag,
            today_step_count=float(antic_cov["today_step_count"][sw, d]),
            recorded_pa_today=float(antic_cov["recorded_pa_today"][sw, d]),
            activity_status_today=float(antic_cov["activity_status_today"][sw, d]),
            salience_today=float(antic_cov["salience_today"][sw, d]),
            d=d,
            E_w=ew,
            ws_morning=float(A[sw, d, 0]),
            ws_afternoon=float(A[sw, d, 1]) if N_RL_SLOTS > 1 else 0.0,
            cae=0.0,
        )

    def cae_row_fn(sw: int) -> np.ndarray:
        four_agent = _pad_foursc_week(M_Y_fourSC[sw])
        antic_agent = _pad_antic_week(M_Y_antic[sw])
        return cae_row_from_general(
            week_norm=_week_norm(sw, nweek),
            foursc_wk=four_agent,
            antic_wk=antic_agent,
            cae_lastweek=0.0,
        )

    X_four = np.array(
        [
            fourSC_row_fn(sim_w_prev, d, t)
            for d in range(N_RL_DAYS)
            for t in range(N_RL_SLOTS)
        ],
        dtype=float,
    )
    delta_four = np.array([fourSC_cae_delta(X_four[i]) for i in range(X_four.shape[0])])
    y_four = np.asarray(M_Y_fourSC[sim_w_prev], dtype=float).copy()

    X_antic = np.array(
        [antic_row_fn(sim_w_prev, d) for d in range(N_MED_ANTIC_DAY)],
        dtype=float,
    )
    if N_MED_ANTIC_DAY > 0:
        X_antic[0, ANTIC_LAG1_COL] = 0.0
    delta_antic = np.array(
        [antic_cae_delta(X_antic[i]) for i in range(X_antic.shape[0])]
    )
    y_antic = np.asarray(M_Y_antic[sim_w_prev], dtype=float).copy()

    X_Y_base = cae_row_fn(sim_w_prev)
    cae_delta_Y = np.zeros(X_Y_base.shape[0], dtype=float)
    cae_delta_Y[_CAE_AR1_COL] = 1.0
    X_tY_base = np.array([1.0, 0.0], dtype=float)
    cae_delta_tY = np.array([0.0, 1.0], dtype=float)

    if k >= 2:
        four_pairs = [
            (sw, d, t)
            for sw in range(sim_w_prev)
            for d in range(N_RL_DAYS)
            for t in range(N_RL_SLOTS)
        ]
        four_sw_idx = np.array([sw for sw, _, _ in four_pairs], dtype=int)
        X_four_c = np.array(
            [fourSC_row_fn(sw, d, t) for sw, d, t in four_pairs], dtype=float
        )
        delta_four_c = np.array(
            [fourSC_cae_delta(X_four_c[i]) for i in range(X_four_c.shape[0])]
        )
        y_four_c = np.array(
            [
                M_Y_fourSC[sw, d * N_RL_SLOTS + t]
                for sw, d, t in four_pairs
            ],
            dtype=float,
        )
        obs_mask_four = np.isfinite(y_four_c)

        antic_pairs = [
            (sw, d) for sw in range(sim_w_prev) for d in range(N_MED_ANTIC_DAY)
        ]
        antic_sw_idx = np.array([sw for sw, _ in antic_pairs], dtype=int)
        antic_day_idx = np.array([d for _, d in antic_pairs], dtype=int)
        X_antic_c = np.array(
            [antic_row_fn(sw, d) for sw, d in antic_pairs], dtype=float
        )
        X_antic_c[antic_day_idx == 0, ANTIC_LAG1_COL] = 0.0
        y_antic_c = np.array(
            [M_Y_antic[sw, d] for sw, d in antic_pairs], dtype=float
        )
        obs_mask_antic = np.isfinite(y_antic_c)
        delta_antic_c = np.array(
            [antic_cae_delta(X_antic_c[i]) for i in range(X_antic_c.shape[0])]
        )

        X_cumul_MY_base = [X_four_c[obs_mask_four], X_antic_c[obs_mask_antic]]
        cae_delta_cumul_MY = [
            delta_four_c[obs_mask_four],
            delta_antic_c[obs_mask_antic],
        ]
        week_idx_cumul_MY = [
            four_sw_idx[obs_mask_four],
            antic_sw_idx[obs_mask_antic],
        ]
        y_cumul_MY = [y_four_c[obs_mask_four], y_antic_c[obs_mask_antic]]

        X_cumul_Y_base = np.array(
            [cae_row_fn(sw) for sw in range(sim_w_prev)], dtype=float
        )
        cae_delta_cumul_Y = np.zeros((sim_w_prev, X_cumul_Y_base.shape[1]))
        cae_delta_cumul_Y[:, _CAE_AR1_COL] = 1.0
        # Targets filled per-particle at update time; keep observed Y when present.
        y_cumul_Y = np.asarray(Y[:sim_w_prev], dtype=float).copy()

        X_tY_full = np.tile(np.array([1.0, 0.0]), (sim_w_prev, 1))
        cae_delta_full = np.zeros((sim_w_prev, 2))
        cae_delta_full[:, 1] = 1.0
        y_tY_full = np.asarray(Y_tilde[:sim_w_prev], dtype=float).copy()
        jw_tY_mask = np.asarray(J_w[:sim_w_prev], dtype=float) == 1.0
        X_cumul_tY = X_tY_full[jw_tY_mask]
        cae_delta_cumul_tY = cae_delta_full[jw_tY_mask]
        y_cumul_tY = y_tY_full[jw_tY_mask]
        week_idx_cumul_tY = np.arange(sim_w_prev)[jw_tY_mask]
    else:
        X_cumul_MY_base = cae_delta_cumul_MY = week_idx_cumul_MY = y_cumul_MY = None
        X_cumul_Y_base = cae_delta_cumul_Y = y_cumul_Y = None
        X_cumul_tY = cae_delta_cumul_tY = y_cumul_tY = None
        week_idx_cumul_tY = None

    return {
        "X_MY_base": [X_four, X_antic],
        "cae_delta_MY": [delta_four, delta_antic],
        "M_Y_obs": [y_four, y_antic],
        "X_Y_base": X_Y_base,
        "cae_delta_Y": cae_delta_Y,
        "X_tY_base": X_tY_base,
        "cae_delta_tY": cae_delta_tY,
        "X_cumul_MY_base": X_cumul_MY_base,
        "cae_delta_cumul_MY": cae_delta_cumul_MY,
        "week_idx_cumul_MY": week_idx_cumul_MY,
        "y_cumul_MY": y_cumul_MY,
        "X_cumul_Y_base": X_cumul_Y_base,
        "cae_delta_cumul_Y": cae_delta_cumul_Y,
        "y_cumul_Y": y_cumul_Y,
        "X_cumul_tY": X_cumul_tY,
        "cae_delta_cumul_tY": cae_delta_cumul_tY,
        "week_idx_cumul_tY": week_idx_cumul_tY,
        "y_cumul_tY": y_cumul_tY,
        "cae_all_0": float(cae_baseline),
        "sim_w_prev": sim_w_prev,
    }


def build_pf_packet_from_reader(
    db_reader: DBReader | None,
    *,
    action_type: str,
    config: dict | None = None,
    user_id: str | None = None,
    k: int | None = None,
    nweek: int | None = None,
    std_params: dict | None = None,
) -> dict[str, Any] | None:
    """Build a normalize-compatible PF packet from ``db_reader`` general-name data.

    When ``std_params`` is provided (``standardize_context: true``), raw
    observation fields in the collected series (``C``, mediators, CAE, …) are
    converted to model scale **before** design rows are built.

    Returns ``None`` when the series cannot support a PF update (no weeks / k<1).
    """
    config = config or {}
    if db_reader is None:
        return None

    series = collect_pf_week_series(
        db_reader, action_type=action_type, config=config
    )
    if series["n_weeks"] <= 0:
        return None

    if std_params is not None:
        series = standardize_pf_series(series, std_params)

    if k is None:
        k = infer_pf_packet_k(db_reader, action_type=action_type)
    if k is None:
        k = int(series["n_weeks"])  # belief week after last observed action week
    k = int(k)
    if k < 1:
        return None

    sim_w_prev = k - 1
    if sim_w_prev >= series["n_weeks"]:
        return None

    cae_baseline = 0.0
    if user_id is not None:
        cae_baseline = float(resolve_y1(db_reader, config, user_id))
    elif config.get("Y_1") is not None:
        cae_baseline = float(config.get("Y_1", 0.0))
    if std_params is not None:
        cae_baseline = float(standardize_cae(cae_baseline, std_params))

    try:
        pf_data = build_pf_data_from_series(
            k, series, cae_baseline=cae_baseline, nweek=nweek
        )
    except (KeyError, ValueError, IndexError) as exc:
        log_alert(
            logger,
            YELLOW,
            "PF packet build from db_reader failed",
            action_type=action_type,
            user_id=str(user_id) if user_id is not None else None,
            k=k,
            error=str(exc),
        )
        return None

    Y_prev = series["Y"][sim_w_prev]
    tY_prev = series["Y_tilde"][sim_w_prev]
    I_w = int(series["I_w"][sim_w_prev]) if np.isfinite(series["I_w"][sim_w_prev]) else 1
    J_w_val = series["J_w"][sim_w_prev]
    J_w = int(J_w_val) if np.isfinite(J_w_val) else None

    return {
        "type": "rl_pf_packet",
        "k": k,
        "I_w": I_w,
        "J_w": J_w,
        "Y_prev": None if not np.isfinite(Y_prev) else float(Y_prev),
        "tY_prev": None if not np.isfinite(tY_prev) else float(tY_prev),
        "pf_data": pf_data,
    }
