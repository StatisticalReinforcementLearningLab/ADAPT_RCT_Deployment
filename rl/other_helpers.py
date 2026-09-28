"""State, serialization, seeding, and validation helpers for ``RL.py``."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from RL_agents.base import DBReader, StoredValue
from RL_agents.logging_config import RED, YELLOW, get_logger, log_alert
from RL_agents.rl.algorithm_helpers import (
    N_RL_CONTEXT,
    N_RL_DAYS,
    N_SERVER_C,
    N_RL_SLOTS,
    P_RL,
    build_phi_action,
    build_rl_context_vector,
    rl_context_from_server,
)

logger = get_logger(__name__)


def ordered_observations(db_reader: DBReader | None) -> list:
    """All observations sorted by server time (then id), exact duplicates removed.

    "Later wins" merges downstream are only well defined on a deterministic
    order; ``DBReader.get_observations`` does not guarantee one.
    """
    if db_reader is None:
        return []
    obs = list(db_reader.get_observations() or [])
    seen: set = set()
    out = []
    for o in obs:
        oid = getattr(o, "observation_id", None)
        if oid is not None:
            if oid in seen:
                continue
            seen.add(oid)
        out.append(o)

    def key(o):
        ts = getattr(o, "server_timestamp", None) or getattr(o, "client_timestamp", None)
        ts_key = ts.timestamp() if hasattr(ts, "timestamp") else float("-inf")
        oid = getattr(o, "observation_id", None)
        return (ts_key, -1 if oid is None else int(oid))

    return sorted(out, key=key)


def required_config(config: dict, key: str) -> Any:
    if key not in config or config[key] is None:
        raise KeyError(f"RL config must include {key!r}")
    return config[key]


def stable_seed(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest(),
        "little",
    )


def decision_seed(
    base_seed: Any,
    user_id: str,
    w: int | None,
    d: int | None,
    t: int | None,
    *,
    fallback: bool = False,
) -> int:
    seed_material = json.dumps(
        [base_seed, str(user_id), w, d, t, "fallback" if fallback else "action"],
        separators=(",", ":"),
    )
    return stable_seed(seed_material)


def to_float_array(value: Any, *, default: Any = None) -> np.ndarray:
    if value is None:
        value = default
    return np.asarray(value, dtype=float)


def to_int_array(value: Any, *, default: Any = None) -> np.ndarray:
    if value is None:
        value = default
    return np.asarray(value, dtype=int)


def none_to_nan_array(value: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    if value is None:
        return np.full(shape, np.nan, dtype=float)
    arr = np.asarray(value, dtype=object)
    out = np.empty(arr.shape, dtype=float)
    it = np.nditer(arr, flags=["multi_index", "refs_ok", "zerosize_ok"])
    for x in it:
        v = x.item()
        out[it.multi_index] = np.nan if v is None else float(v)
    return out


def jsonable_float_array(arr: np.ndarray) -> list:
    vals = np.asarray(arr, dtype=float)
    if vals.ndim == 0:
        v = float(vals)
        return None if not np.isfinite(v) else v
    out = []
    for x in vals:
        if np.asarray(x).ndim:
            out.append(jsonable_float_array(x))
        else:
            v = float(x)
            out.append(None if not np.isfinite(v) else v)
    return out


def jsonable_int_array(arr: np.ndarray) -> list:
    return np.asarray(arr, dtype=int).tolist()


def history_arrays(
    model_params: dict,
    *,
    n_rl_days: int | None = None,
    n_rl_slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load persisted RL history arrays from ``model_params`` as NumPy arrays."""
    W = int(model_params["W"])
    n_days = int(model_params.get("n_rl_days", n_rl_days))
    n_slots = int(model_params.get("n_rl_slots", n_rl_slots))
    A_hist = to_int_array(
        model_params.get("A_hist"),
        default=np.zeros((W, n_days, n_slots), dtype=int),
    )
    pi_A_hist = none_to_nan_array(
        model_params.get("pi_A_hist"),
        shape=(W, n_days, n_slots),
    )
    b_hat_hist = none_to_nan_array(
        model_params.get("b_hat_hist"),
        shape=(W + 1,),
    )
    b_tilde_hist = none_to_nan_array(
        model_params.get("b_tilde_hist"),
        shape=(W + 1,),
    )
    return A_hist, pi_A_hist, b_hat_hist, b_tilde_hist


def require_key(state: dict, key: str) -> Any:
    if key not in state or state[key] is None:
        raise KeyError(f"state must include {key!r}")
    return state[key]


def coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    val = float(value)
    return val if np.isfinite(val) else None


def coerce_mediator_matrix(
    value: Any,
    *,
    name: str,
    n_cols: int,
) -> np.ndarray:
    if value is None:
        raise KeyError(f"state must include {name!r}")
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        if arr.size % n_cols != 0:
            raise ValueError(f"{name} length {arr.size} is not divisible by {n_cols}")
        arr = arr.reshape(arr.size // n_cols, n_cols)
    if arr.ndim != 2 or arr.shape[1] != n_cols:
        raise ValueError(f"{name} must have shape (<= {N_RL_DAYS}, {n_cols})")
    if arr.shape[0] > N_RL_DAYS:
        raise ValueError(f"{name} has {arr.shape[0]} rows; expected at most {N_RL_DAYS}")
    out = np.full((N_RL_DAYS, n_cols), np.nan, dtype=float)
    out[: arr.shape[0], :] = arr
    return out


def normalize_study_week(context: dict) -> dict:
    """Normalize context/state to a single 1-indexed study week field ``w``.

    Query and RL both speak 1-indexed study weeks at the API/state boundary
    (``w=1`` is enrollment/baseline). Accepted incoming aliases:

    - ``w``, ``rl_week``, ``week``: 1-indexed study week
    - legacy ``k`` only (no ``w``/aliases): treated as the old RL 0-indexed
      week and converted via ``w = k + 1``

    Aliases are dropped so ``state_index`` reads ``w`` unambiguously. Array
    indexing inside RL still uses ``k = w - 1`` via ``internal_week_index``.
    """
    ctx = dict(context or {})
    for key in ("w", "rl_week", "week"):
        if ctx.get(key) is not None:
            w = int(ctx[key])
            for alias in ("w", "rl_week", "week", "k"):
                ctx.pop(alias, None)
            if w < 1:
                raise ValueError(f"study week w must be >= 1, got {w}")
            ctx["w"] = w
            return ctx
    if ctx.get("k") is not None:
        k = int(ctx["k"])
        ctx.pop("k", None)
        ctx["w"] = k + 1
        return ctx
    return ctx


def internal_week_index(w: int) -> int:
    """Map 1-indexed study week ``w`` to 0-indexed RL array index ``k``."""
    return int(w) - 1


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def context_decision_datetime(context: dict) -> datetime | None:
    for key in ("decision_time", "decisionTime", "timestamp", "request_time"):
        if key in context and context[key] is not None:
            return parse_datetime(context[key])
    return None


def context_timezone(context: dict) -> str:
    for key in ("timezone", "time_zone", "tz"):
        value = context.get(key)
        if value:
            return str(value)
    raise KeyError("RL context must include timezone, time_zone, or tz")


def participant_info(db_reader: DBReader) -> dict:
    data = db_reader.get_retrieved_data() or {}
    info = (
        data.get("participantinfo")
        or data.get("participant_info")
        or data.get("ParticipantInfo")
    )
    if info is None:
        raise KeyError(
            "db_reader.get_retrieved_data() must include participantinfo"
        )
    if isinstance(info, list):
        if not info:
            raise KeyError("participantinfo list is empty")
        info = info[0]
    return dict(info)


def study_start_date(participant: dict) -> date:
    for key in (
        "study_start_date",
        "enroll_date",
        "start_date",
        "StudyStartDate",
        "date_min",
    ):
        value = participant.get(key)
        if value is None:
            continue
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        return parse_datetime(value).date()
    raise KeyError(
        "participantinfo must include study_start_date, enroll_date, or date_min"
    )


def sunday_on_or_before(d: date) -> date:
    """Return the Sunday on or before ``d`` (Python weekday: Sun=6)."""
    return d - timedelta(days=(d.weekday() + 1) % 7)


def study_week_from_dates(study_start: date, local_decision_date: date) -> int:
    """1-indexed study week; enrollment Sunday starts ``w=1``."""
    enrollment_sunday = sunday_on_or_before(study_start)
    day = (local_decision_date - enrollment_sunday).days
    if day < 0:
        raise ValueError(
            f"Decision date {local_decision_date} is before enrollment Sunday "
            f"{enrollment_sunday}"
        )
    return day // 7 + 1


def rl_day_from_dates(study_start: date, local_decision_date: date) -> int:
    """0-indexed Mon–Sat RL day ``d`` within the study week.

    Enrollment Sunday starts the week; Sunday itself is not an RL decision day.
    """
    enrollment_sunday = sunday_on_or_before(study_start)
    day = (local_decision_date - enrollment_sunday).days
    if day < 0:
        raise ValueError(
            f"Decision date {local_decision_date} is before enrollment Sunday "
            f"{enrollment_sunday}"
        )
    dow = day % 7  # 0=Sun ... 6=Sat
    if dow == 0:
        raise ValueError(
            f"Decision date {local_decision_date} is Sunday; RL days are Mon–Sat"
        )
    return int(dow - 1)


def _parse_clock_time(value: Any) -> time | None:
    if value is None:
        return None
    if isinstance(value, time):
        return value
    if isinstance(value, datetime):
        return value.time()
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%H:%M:%S", "%H:%M"):
            try:
                return datetime.strptime(text, fmt).time()
            except ValueError:
                continue
        try:
            return parse_datetime(text).time()
        except (TypeError, ValueError):
            return None
    return None


def _wakeup_time_for_date(participant: dict, local_date: date) -> time | None:
    is_weekend = local_date.weekday() >= 5
    keys = (
        ("WeekendWakeup", "weekend_wakeup", "weekendWakeup")
        if is_weekend
        else ("WeekdayWakeup", "weekday_wakeup", "weekdayWakeup")
    )
    for key in keys:
        parsed = _parse_clock_time(participant.get(key))
        if parsed is not None:
            return parsed
    for key in ("wakeup", "Wakeup", "wake_time"):
        parsed = _parse_clock_time(participant.get(key))
        if parsed is not None:
            return parsed
    return None


def rl_slot_from_local_datetime(
    local_dt: datetime,
    participant: dict | None = None,
) -> int:
    """Derive slot ``t`` (0=morning, 1=afternoon) from local decision time.

    Protocol default: slot 0 at wakeup+1h, slot 1 at wakeup+6h (nearest wins).
    Without wakeup metadata, use local hour ``< 14`` → 0 else 1.
    """
    wake = _wakeup_time_for_date(participant or {}, local_dt.date())
    if wake is not None:
        wake_dt = datetime.combine(local_dt.date(), wake, tzinfo=local_dt.tzinfo)
        slot0 = wake_dt + timedelta(hours=1)
        slot1 = wake_dt + timedelta(hours=6)
        d0 = abs((local_dt - slot0).total_seconds())
        d1 = abs((local_dt - slot1).total_seconds())
        return 0 if d0 <= d1 else 1
    return 0 if local_dt.hour < 14 else 1


def resolve_availability(context: dict) -> bool:
    """Require availability from request context (MRT eligibility)."""
    for key in ("availability", "available", "is_available", "Availability"):
        if key in context and context[key] is not None:
            value = context[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float, np.integer, np.floating)):
                return bool(int(value))
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"1", "true", "yes", "y"}:
                    return True
                if text in {"0", "false", "no", "n"}:
                    return False
            raise ValueError(f"Unrecognized availability value: {value!r}")
    raise KeyError(
        "RL context must include availability (or available / is_available)"
    )


def resolve_decision_index(
    context: dict,
    db_reader: DBReader | None,
) -> tuple[int, int, int]:
    """Resolve ``(w, d, t)`` from decision timestamp (+ enrollment) or explicit index.

    Preferred context fields: ``decision_time`` / ``timestamp`` and ``timezone``.
    Fallback for tests / legacy callers: explicit ``w``/``d``/``t`` (or aliases).
    """
    ctx = dict(context or {})
    decision_dt = context_decision_datetime(ctx)
    if decision_dt is not None:
        tz = ZoneInfo(context_timezone(ctx))
        if decision_dt.tzinfo is None:
            decision_dt = decision_dt.replace(tzinfo=tz)
        local_dt = decision_dt.astimezone(tz)
        if db_reader is None:
            raise KeyError(
                "db_reader is required to resolve study week from decision_time"
            )
        participant = participant_info(db_reader)
        study_start = study_start_date(participant)
        w = study_week_from_dates(study_start, local_dt.date())
        d = rl_day_from_dates(study_start, local_dt.date())
        # Explicit slot still wins when the server tags the decision window.
        t_raw = ctx.get("t", ctx.get("slot", ctx.get("decisionTimeSlot")))
        if t_raw is not None:
            t = int(t_raw)
        else:
            t = rl_slot_from_local_datetime(local_dt, participant)
        if not (0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS):
            raise ValueError(f"derived (d, t)=({d}, {t}) out of RL grid")
        return int(w), int(d), int(t)

    ctx = normalize_study_week(ctx)
    w = ctx.get("w")
    d = ctx.get("d", ctx.get("day"))
    t = ctx.get("t", ctx.get("slot", ctx.get("decisionTimeSlot")))
    if w is None or d is None or t is None:
        raise KeyError(
            "RL context must include decision_time (+ timezone) or explicit "
            "w, d/day, and t/slot"
        )
    return int(w), int(d), int(t)


def state_index(state: dict) -> tuple[int, int, int]:
    """Return ``(w, d, t)`` with 1-indexed study week ``w``."""
    state = normalize_study_week(state)
    w = state.get("w")
    d = state.get("d", state.get("day"))
    t = state.get("t", state.get("slot"))
    if w is None or d is None or t is None:
        raise KeyError("state must include w (or legacy k), d/day, and t/slot")
    return int(w), int(d), int(t)


def optional_state_index(state: dict | None) -> tuple[int | None, int | None, int | None]:
    if not state:
        return None, None, None
    try:
        return state_index(state)
    except (KeyError, TypeError, ValueError):
        return None, None, None


def coerce_state(raw: dict) -> dict:
    """Coerce a framework context/state dict for ``build_phi_action``."""
    if raw is None:
        raw = {}
    state = normalize_study_week(raw)
    w, d, t = state_index(state)

    if "C" in state:
        C = np.asarray(state["C"], dtype=float).ravel()
        if C.size != N_RL_CONTEXT:
            if C.size == N_SERVER_C:
                C = rl_context_from_server(C)
            else:
                raise ValueError(
                    f"state C must have length {N_RL_CONTEXT} (RL) or {N_SERVER_C} (server), got {C.size}"
                )
    else:
        context_fields = {
            "yesterdayStepCount": state.get("yesterdayStepCount"),
            "prior2HourStepCount": state.get(
                "prior2HourStepCount", state.get("prior2HourStepCountAgent")
            ),
            "activeDaysLast7Days": state.get("activeDaysLast7Days"),
            "activitySuggestionsSentLast7Days": state.get(
                "activitySuggestionsSentLast7Days"
            ),
            "activitySuggestionInteractLast7Days": state.get(
                "activitySuggestionInteractLast7Days"
            ),
        }
        missing = [key for key, value in context_fields.items() if value is None]
        if missing:
            raise KeyError(
                "state must include C or all raw context fields; "
                f"missing {missing}"
            )
        C = build_rl_context_vector(**context_fields)

    return {
        **state,
        "w": w,
        "d": d,
        "t": t,
        "E_w": float(
            state["E_w"] if "E_w" in state else require_key(state, "perceivedUtility")
        ),
        "M_Y": coerce_mediator_matrix(state.get("M_Y"), name="M_Y", n_cols=3),
        "M_E": coerce_mediator_matrix(state.get("M_E"), name="M_E", n_cols=4),
        "C": C,
    }


def feature_state(state: dict) -> dict:
    return {
        "E_w": float(state["E_w"]),
        "M_Y": np.asarray(state["M_Y"], dtype=float),
        "M_E": np.asarray(state["M_E"], dtype=float),
        "C": np.asarray(state["C"], dtype=float).ravel(),
    }


def stored_value_payload(stored: StoredValue | None) -> Any:
    if stored is None:
        return None
    return stored.value


def model_params_from_reader(db_reader: DBReader) -> dict | None:
    """Read current RL model parameters from ``db_reader.get_retrieved_data()``."""
    if db_reader is None:
        return None
    data = db_reader.get_retrieved_data() or {}
    for key in ("model_params", "current_model_params", "rl_model_params"):
        value = data.get(key)
        if isinstance(value, dict):
            return value
    return None


DEFAULT_B_HAT_STORED_VALUE_TYPE = "rl_b_hat"
DEFAULT_B_TILDE_STORED_VALUE_TYPE = "rl_b_tilde"
DEFAULT_PF_STATE_STORED_VALUE_TYPE = "rl_pf_state"
DEFAULT_PF_STATE_HISTORY_STORED_VALUE_TYPE_PREFIX = "rl_pf_state_week"


def stored_pf_state(config: dict, db_reader: DBReader) -> dict | None:
    if db_reader is None:
        return None
    stored = db_reader.get_stored_value(
        str(
            config.get(
                "pf_state_stored_value_type",
                DEFAULT_PF_STATE_STORED_VALUE_TYPE,
            )
        )
    )
    if stored is None or not isinstance(stored.value, dict):
        return None
    return stored.value


DEFAULT_Y1_OBSERVATION_TYPE = "rl_Y_1"


def y1_observation_type(config: dict) -> str:
    return str(config.get("y_1_observation_type", DEFAULT_Y1_OBSERVATION_TYPE))


def y1_from_reader(
    db_reader: DBReader | None,
    config: dict,
    user_id: str,
) -> float | None:
    """Read the per-participant baseline outcome ``Y_1`` from observations.

    The server records ``Y_1`` in an :class:`ObservationRecord`. It is found
    either in any observation payload under key ``Y_1``/``Y1``, or in the
    payload ``value`` of an observation whose ``observation_type`` matches
    ``y_1_observation_type`` (default ``"rl_Y_1"``). Only observations whose
    ``user_id`` matches ``user_id`` are considered. Returns ``None`` when no
    usable value is present so callers can fall back to config.
    """
    if db_reader is None:
        return None
    obs_type = y1_observation_type(config)
    uid = str(user_id)
    observations = ordered_observations(db_reader)
    values: list[float] = []
    for obs in observations:
        obs_uid = getattr(obs, "user_id", None)
        if obs_uid is not None and str(obs_uid) != uid:
            continue
        payload = getattr(obs, "payload", {}) or {}
        value = payload.get("Y_1", payload.get("Y1"))
        if value is None and getattr(obs, "observation_type", None) == obs_type:
            value = payload.get("value")
        if value is None:
            continue
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            values.append(val)
    if not values:
        return None
    if len(set(values)) > 1:
        log_alert(
            logger,
            YELLOW,
            "multiple distinct baseline Y_1 observations; using the latest",
            user_id=uid,
            values=values,
        )
    return values[-1]


def resolve_y1(db_reader: DBReader | None, config: dict, user_id: str) -> float:
    """Baseline CAE ``Y_1`` from observations, falling back to config."""
    observed = y1_from_reader(db_reader, config, user_id)
    if observed is None:
        raise KeyError(
            f"baseline CAE Y_1 ({y1_observation_type(config)}) is missing for "
            f"participant {user_id}; it is never defaulted"
        )
    return float(observed)


def infer_p_rl(state: dict) -> int:
    """Infer RLSVI feature dimension from the state produced by ``make_state``."""
    _, d, t = state_index(state)
    phi = build_phi_action(0.0, 0.0, feature_state(state), d, t, 0)
    return int(phi.size)


def state_key(k: int, d: int, t: int) -> str:
    """History key using 0-indexed internal week ``k`` (array index)."""
    return f"{int(k)}:{int(d)}:{int(t)}"


def serialize_state(state: dict) -> dict:
    w, d, t = state_index(state)
    return {
        "w": int(w),
        "d": int(d),
        "t": int(t),
        "availability": bool(state.get("availability", True)),
        "E_w": float(state["E_w"]),
        "M_Y": jsonable_float_array(np.asarray(state["M_Y"], dtype=float)),
        "M_E": jsonable_float_array(np.asarray(state["M_E"], dtype=float)),
        "C": jsonable_float_array(np.asarray(state["C"], dtype=float)),
        "b_hat": (
            None
            if state.get("b_hat") is None or not np.isfinite(float(state.get("b_hat")))
            else float(state["b_hat"])
        ),
        "b_tilde": (
            None
            if state.get("b_tilde") is None
            or not np.isfinite(float(state.get("b_tilde")))
            else float(state["b_tilde"])
        ),
        "p_rl": int(infer_p_rl(state)),
    }


def deserialize_state(state: dict) -> dict:
    availability = bool((state or {}).get("availability", True))
    out = coerce_state(state)
    out["availability"] = availability
    return out


def validate_feature_dimension(
    *,
    user_id: str,
    action_type: str,
    state: dict,
    model_params: dict,
) -> None:
    p_state = int(state["p_rl"]) if state.get("p_rl") is not None else infer_p_rl(state)
    p_model = (
        int(model_params["p_rl"])
        if model_params.get("p_rl") is not None
        else len(model_params["mu_0_rl"])
    )
    if p_model != P_RL:
        raise ValueError(
            f"model_params p_rl={p_model} is not this package's feature dimension {P_RL}"
        )
    if p_state == p_model:
        return
    w, d, t = state_index(state)
    log_alert(
        logger,
        RED,
        "RL state feature dimension does not match model parameters",
        action_type=action_type,
        user_id=str(user_id),
        w=w,
        d=d,
        t=t,
        p_state=p_state,
        p_model=p_model,
    )
    raise ValueError(f"state p_rl={p_state} does not match model p_rl={p_model}")


# ---------------------------------------------------------------------------
# Week-tagged stored values
# ---------------------------------------------------------------------------
def tagged_value(k: int, value: Any, **extra: Any) -> dict:
    return {"k": int(k), "value": value, **extra}


def stored_tagged(db_reader: DBReader | None, stored_value_type: str) -> tuple[float | None, int | None]:
    """Return ``(value, k)`` from a week-tagged stored value.

    Legacy untagged floats return ``k=None`` so callers can treat them as
    stale-of-unknown-age.
    """
    if db_reader is None:
        return None, None
    stored = stored_value_payload(db_reader.get_stored_value(stored_value_type))
    if isinstance(stored, dict):
        k = stored.get("k")
        return coerce_optional_float(stored.get("value")), (None if k is None else int(k))
    return coerce_optional_float(stored), None
