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
    N_RL_DAYS,
    N_RL_SLOTS,
    build_phi_action,
    build_rl_context_vector,
)
from RL_agents.rl.standardization import (
    standardize_c_vector,
    standardize_me_matrix,
    standardize_my_matrix,
)

logger = get_logger(__name__)


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


def apply_weekly_rewards_to_belief_hist(
    rewards: dict[int, float],
    action_id_to_k: dict[int, int],
    b_hat_hist: np.ndarray,
    b_tilde_hist: np.ndarray,
    *,
    latest_b_tilde: float | None,
    action_type: str | None = None,
    user_id: str | None = None,
) -> None:
    """Apply action_id-keyed rewards to ``b_hat_hist[k+1]`` once per completed week.

    The external API keeps ``rewards[action_id]`` for observation joins; this
    collapses duplicate weekly values from multiple slots in the same week.
    """
    by_k: dict[int, float] = {}
    for action_id, reward in rewards.items():
        k = action_id_to_k.get(int(action_id))
        if k is None or k + 1 >= b_hat_hist.size:
            continue
        val = float(reward)
        prev = by_k.get(k)
        if (
            prev is not None
            and prev != val
            and action_type is not None
            and user_id is not None
        ):
            log_alert(
                logger,
                YELLOW,
                "conflicting weekly rewards for same week; using last",
                action_type=action_type,
                user_id=str(user_id),
                k=k,
                previous=float(prev),
                reward=val,
            )
        by_k[k] = val

    b_tilde_carry = float(latest_b_tilde) if latest_b_tilde is not None else 0.0
    for k, reward in by_k.items():
        b_hat_hist[k + 1] = reward
        # Without PF, carry the last observed belief uncertainty forward.
        # PF (if prepared) overrides b_tilde_hist later in update().
        b_tilde_hist[k + 1] = b_tilde_carry


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
    out = np.zeros((N_RL_DAYS, n_cols), dtype=float)
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
    else:
        context_fields = {
            "yesterdayStepCount": state.get("yesterdayStepCount"),
            "stepCountLast7DaysEma": state.get("stepCountLast7DaysEma"),
            "prior2HourStepCountAgent": state.get("prior2HourStepCountAgent"),
            "activityCompletedLast7Days": state.get("activityCompletedLast7Days"),
            "activeDaysLast7Days": state.get("activeDaysLast7Days"),
            "activitySuggestionsSentLast7Days": state.get(
                "activitySuggestionsSentLast7Days"
            ),
            "salienceMessageSentYesterday": state.get("salienceMessageSentYesterday"),
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


def stored_float(db_reader: DBReader, stored_value_type: str) -> float | None:
    if db_reader is None:
        return None
    stored = stored_value_payload(db_reader.get_stored_value(stored_value_type))
    if isinstance(stored, dict):
        for key in ("value", "b_hat", "b_tilde"):
            if key in stored:
                return coerce_optional_float(stored[key])
        return None
    return coerce_optional_float(stored)


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


def belief_stored_values(
    config: dict,
    user_id: str,
    b_hat: float,
    b_tilde: float,
) -> list[StoredValue]:
    return [
        StoredValue(
            user_id=str(user_id),
            type=str(
                config.get(
                    "b_hat_stored_value_type",
                    DEFAULT_B_HAT_STORED_VALUE_TYPE,
                )
            ),
            value=float(b_hat),
        ),
        StoredValue(
            user_id=str(user_id),
            type=str(
                config.get(
                    "b_tilde_stored_value_type",
                    DEFAULT_B_TILDE_STORED_VALUE_TYPE,
                )
            ),
            value=float(b_tilde),
        ),
    ]


def pf_state_stored_values(
    config: dict,
    user_id: str,
    *,
    packet: dict,
    pf_update: dict,
    b_hat: float,
    b_tilde: float,
    rng_state: dict,
) -> list[StoredValue]:
    k = int(packet["k"])
    value = {
        "k": k,
        "b_hat": float(b_hat),
        "b_tilde": float(b_tilde),
        "pf_update": pf_update,
        "rng_state": rng_state,
    }
    latest_type = str(
        config.get(
            "pf_state_stored_value_type",
            DEFAULT_PF_STATE_STORED_VALUE_TYPE,
        )
    )
    history_prefix = str(
        config.get(
            "pf_state_history_stored_value_type_prefix",
            DEFAULT_PF_STATE_HISTORY_STORED_VALUE_TYPE_PREFIX,
        )
    )
    return [
        StoredValue(
            user_id=str(user_id),
            type=latest_type,
            value=value,
        ),
        StoredValue(
            user_id=str(user_id),
            type=f"{history_prefix}_{k}",
            value=value,
        ),
    ]


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


def stored_belief(
    config: dict, db_reader: DBReader
) -> tuple[float | None, float | None]:
    return (
        stored_float(
            db_reader,
            str(
                config.get(
                    "b_hat_stored_value_type",
                    DEFAULT_B_HAT_STORED_VALUE_TYPE,
                )
            ),
        ),
        stored_float(
            db_reader,
            str(
                config.get(
                    "b_tilde_stored_value_type",
                    DEFAULT_B_TILDE_STORED_VALUE_TYPE,
                )
            ),
        ),
    )


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
    try:
        observations = db_reader.get_observations() or []
    except TypeError:
        observations = db_reader.get_observations
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
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def resolve_y1(db_reader: DBReader | None, config: dict, user_id: str) -> float:
    """Baseline CAE ``Y_1`` from observations, falling back to config."""
    observed = y1_from_reader(db_reader, config, user_id)
    return (
        float(observed)
        if observed is not None
        else float(config.get("Y_1", 0.0))
    )


def infer_p_rl(state: dict) -> int:
    """Infer RLSVI feature dimension from the state produced by ``make_state``."""
    _, d, t = state_index(state)
    phi = build_phi_action(0.0, 0.0, feature_state(state), d, t, 0)
    return int(phi.size)


def state_key(k: int, d: int, t: int) -> str:
    """History key using 0-indexed internal week ``k`` (array index)."""
    return f"{int(k)}:{int(d)}:{int(t)}"


def _my_matrix_from_pf_series(series: dict[str, Any], k: int) -> np.ndarray | None:
    """Rebuild ``M_Y`` ``(N_RL_DAYS, 3)`` from PF week series, or ``None`` if empty."""
    if series is None or k < 0 or k >= int(series.get("n_weeks", 0)):
        return None
    four = np.asarray(series["M_Y_fourSC"][k], dtype=float).ravel()
    antic = np.asarray(series["M_Y_antic"][k], dtype=float).ravel()
    if not (np.any(np.isfinite(four)) or np.any(np.isfinite(antic))):
        return None
    out = np.full((N_RL_DAYS, 3), np.nan, dtype=float)
    for d in range(N_RL_DAYS):
        i0 = d * N_RL_SLOTS
        if i0 < four.size and np.isfinite(four[i0]):
            out[d, 0] = four[i0]
        if i0 + 1 < four.size and np.isfinite(four[i0 + 1]):
            out[d, 1] = four[i0 + 1]
        if d < antic.size and np.isfinite(antic[d]):
            out[d, 2] = antic[d]
    # Missing mediator cells are treated as 0 for RLSVI masking (same as coerce pad).
    return np.nan_to_num(out, nan=0.0)


def _obs_week_index(payload: dict) -> int | None:
    if payload.get("w") is not None:
        return internal_week_index(int(payload["w"]))
    if payload.get("rl_week") is not None:
        return internal_week_index(int(payload["rl_week"]))
    if payload.get("week") is not None:
        return internal_week_index(int(payload["week"]))
    if payload.get("k") is not None:
        return int(payload["k"])
    return None


def _obs_day_slot(payload: dict) -> tuple[int | None, int | None]:
    d = payload.get("d", payload.get("day"))
    t = payload.get("t", payload.get("slot"))
    try:
        d_out = None if d is None else int(d)
        t_out = None if t is None else int(t)
    except (TypeError, ValueError):
        return None, None
    return d_out, t_out


def _c_vector_for_slot(C_value: Any, d: int, t: int) -> np.ndarray | None:
    """Extract length-9 ``C`` for ``(d, t)`` from a vector or week grid."""
    if C_value is None:
        return None
    C_arr = np.asarray(C_value, dtype=float)
    n_med = N_RL_DAYS * N_RL_SLOTS
    if C_arr.ndim == 1:
        out = np.zeros(9, dtype=float)
        m = min(int(C_arr.size), 9)
        if m:
            out[:m] = C_arr[:m]
        return out
    if C_arr.ndim == 3 and C_arr.shape[:2] == (N_RL_DAYS, N_RL_SLOTS):
        if 0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS:
            out = np.zeros(9, dtype=float)
            row = np.asarray(C_arr[d, t], dtype=float).ravel()
            m = min(int(row.size), 9)
            if m:
                out[:m] = row[:m]
            return out
        return None
    if C_arr.ndim == 2 and C_arr.shape[0] == n_med:
        s = d * N_RL_SLOTS + t
        if 0 <= s < C_arr.shape[0]:
            out = np.zeros(9, dtype=float)
            row = np.asarray(C_arr[s], dtype=float).ravel()
            m = min(int(row.size), 9)
            if m:
                out[:m] = row[:m]
            return out
    return None


def _my_matrix_from_observation(src: dict) -> np.ndarray | None:
    """Build ``M_Y`` ``(N_RL_DAYS, 3)`` from an observation payload, or ``None``."""
    my = src.get("M_Y", src.get("M^Y"))
    if my is not None:
        arr = np.asarray(my, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            out = np.full((N_RL_DAYS, 3), np.nan, dtype=float)
            n = min(N_RL_DAYS, arr.shape[0])
            out[:n, :] = arr[:n, :3]
            if np.any(np.isfinite(out)):
                return np.nan_to_num(out, nan=0.0)
        if arr.ndim == 1 and arr.size >= N_RL_DAYS * N_RL_SLOTS:
            # Flat fourSC (+ optional antic) — same layout as PF collector.
            four = arr[: N_RL_DAYS * N_RL_SLOTS]
            antic = (
                arr[N_RL_DAYS * N_RL_SLOTS : N_RL_DAYS * N_RL_SLOTS + N_RL_DAYS]
                if arr.size >= N_RL_DAYS * N_RL_SLOTS + N_RL_DAYS
                else np.full(N_RL_DAYS, np.nan)
            )
            out = np.full((N_RL_DAYS, 3), np.nan, dtype=float)
            for dd in range(N_RL_DAYS):
                i0 = dd * N_RL_SLOTS
                if i0 < four.size:
                    out[dd, 0] = four[i0]
                if i0 + 1 < four.size:
                    out[dd, 1] = four[i0 + 1]
                if dd < antic.size:
                    out[dd, 2] = antic[dd]
            if np.any(np.isfinite(out)):
                return np.nan_to_num(out, nan=0.0)

    four_src = src.get("stepCountNext4Hour")
    antic_src = src.get("dailyAnticipatedAffect")
    if four_src is None and antic_src is None:
        return None
    out = np.full((N_RL_DAYS, 3), np.nan, dtype=float)
    if four_src is not None:
        four = np.asarray(four_src, dtype=float).ravel()
        for dd in range(N_RL_DAYS):
            i0 = dd * N_RL_SLOTS
            if i0 < four.size and np.isfinite(four[i0]):
                out[dd, 0] = four[i0]
            if i0 + 1 < four.size and np.isfinite(four[i0 + 1]):
                out[dd, 1] = four[i0 + 1]
    if antic_src is not None:
        antic = np.asarray(antic_src, dtype=float).ravel()
        for dd in range(min(N_RL_DAYS, antic.size)):
            if np.isfinite(antic[dd]):
                out[dd, 2] = antic[dd]
    if not np.any(np.isfinite(out)):
        return None
    return np.nan_to_num(out, nan=0.0)


def _me_matrix_from_observation(src: dict) -> np.ndarray | None:
    me = src.get("M_E", src.get("M^E"))
    if me is None:
        return None
    arr = np.asarray(me, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    if not np.any(np.isfinite(arr)):
        return None
    out = np.zeros((N_RL_DAYS, 4), dtype=float)
    n = min(N_RL_DAYS, arr.shape[0])
    out[:n, :] = np.nan_to_num(arr[:n, :4], nan=0.0)
    return out


def lookup_decision_covariates(
    db_reader: DBReader | None,
    *,
    k: int,
    d: int,
    t: int,
) -> dict[str, Any]:
    """Direct observation lookup for one decision: slot ``C``, week ``M_Y`` / ``M_E``.

    Does **not** scan action-time state or build a full-week PF series. Later
    matching observations overwrite earlier ones. ``retrieved_data`` keys
    ``C`` / ``rl_C`` / ``pf_C`` are also accepted for slot ``C``.

    Returns ``{"C": ndarray|None, "M_Y": ndarray|None, "M_E": ndarray|None}``.
    ``C`` is length 9 (RLSVI uses the first 8); mediators are ``(N_RL_DAYS, *)``.
    """
    out: dict[str, Any] = {"C": None, "M_Y": None, "M_E": None}
    if db_reader is None or not (0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS):
        return out

    try:
        observations = db_reader.get_observations() or []
    except TypeError:
        observations = db_reader.get_observations or []

    for obs in observations:
        payload = getattr(obs, "payload", {}) or {}
        if not isinstance(payload, dict):
            continue
        series_payload = payload.get("weekly_series", payload)
        candidates = [payload]
        if isinstance(series_payload, dict) and series_payload is not payload:
            candidates.append(series_payload)

        for src in candidates:
            if not isinstance(src, dict):
                continue
            wk = _obs_week_index(src)
            if wk is None:
                wk = _obs_week_index(payload)
            if wk is None or int(wk) != int(k):
                continue

            d_slot, t_slot = _obs_day_slot(src)
            if d_slot is None and t_slot is None:
                d_slot, t_slot = _obs_day_slot(payload)

            c_value = src.get("C", payload.get("C") if src is not payload else None)
            if c_value is not None:
                # Per-slot vector requires matching (d, t); week grids index internally.
                c_arr = np.asarray(c_value, dtype=float)
                if c_arr.ndim == 1:
                    if d_slot == d and t_slot == t:
                        extracted = _c_vector_for_slot(c_value, d, t)
                        if extracted is not None:
                            out["C"] = extracted
                else:
                    extracted = _c_vector_for_slot(c_value, d, t)
                    if extracted is not None:
                        out["C"] = extracted

            my = _my_matrix_from_observation(src)
            if my is not None:
                out["M_Y"] = my

            me = _me_matrix_from_observation(src)
            if me is not None:
                out["M_E"] = me

    try:
        retrieved = db_reader.get_retrieved_data() or {}
    except Exception:  # noqa: BLE001 - retrieved_data is optional
        retrieved = {}
    for key in ("C", "rl_C", "pf_C"):
        batch_c = retrieved.get(key)
        if batch_c is None:
            continue
        if isinstance(batch_c, dict):
            for week_key, C_value in batch_c.items():
                try:
                    wk = int(week_key)
                except (TypeError, ValueError):
                    continue
                k_batch = internal_week_index(wk) if wk >= 1 else int(wk)
                if k_batch != int(k):
                    continue
                extracted = _c_vector_for_slot(C_value, d, t)
                if extracted is not None:
                    out["C"] = extracted
        else:
            k_batch = _obs_week_index(retrieved)
            if k_batch == int(k):
                extracted = _c_vector_for_slot(batch_c, d, t)
                if extracted is not None:
                    out["C"] = extracted
        break

    return out


def apply_decision_covariates(
    state: dict,
    *,
    covariates: dict[str, Any],
    std_params: dict | None = None,
) -> dict:
    """Write looked-up ``C`` / ``M_Y`` / ``M_E`` onto a decision state (optional std)."""
    out = dict(state)
    c = covariates.get("C")
    if c is not None:
        c_raw = np.asarray(c, dtype=float).ravel()
        if std_params is not None:
            c_raw = standardize_c_vector(c_raw, std_params)
        out["C"] = c_raw[:8]

    my = covariates.get("M_Y")
    if my is not None:
        my_arr = np.asarray(my, dtype=float)
        if std_params is not None:
            my_arr = standardize_my_matrix(my_arr, std_params)
        out["M_Y"] = my_arr

    me = covariates.get("M_E")
    if me is not None:
        me_arr = np.asarray(me, dtype=float)
        if std_params is not None:
            me_arr = standardize_me_matrix(me_arr, std_params)
        out["M_E"] = me_arr

    return out


def overlay_observation_covariates_on_state(
    state: dict,
    *,
    k: int,
    d: int,
    t: int,
    series: dict[str, Any] | None,
    me_week: Any | None,
    std_params: dict | None = None,
) -> dict:
    """Replace delayed covariates from ``db_reader`` series; keep beliefs / ``E_w``.

    Kept from the action-time state (for RLSVI timing / replicability of beliefs):
    ``E_w``, ``b_hat``, ``b_tilde``, and the decision index ``(w, d, t)``.

    Overwritten when observation-sourced data exist:
    ``C`` (from PF series when ``C_filled[k,d,t]``), ``M_Y`` (week mediators),
    ``M_E`` (week engagement matrix). Raw values are model-scaled when
    ``std_params`` is provided.
    """
    out = dict(state)
    # Preserve belief fields explicitly even if absent on ``out``.
    for key in ("E_w", "b_hat", "b_tilde", "perceivedUtility"):
        if key in state:
            out[key] = state[key]

    if (
        series is not None
        and 0 <= k < int(series.get("n_weeks", 0))
        and 0 <= d < N_RL_DAYS
        and 0 <= t < N_RL_SLOTS
    ):
        filled = np.asarray(series.get("C_filled"), dtype=bool)
        if (
            filled.ndim == 3
            and k < filled.shape[0]
            and filled[k, d, t]
        ):
            c_raw = np.asarray(series["C"][k, d, t], dtype=float).ravel()
            if std_params is not None:
                c_raw = standardize_c_vector(c_raw, std_params)
            # RLSVI context is length 8; PF series packs optional wear in C[8].
            out["C"] = c_raw[:8]

        my = _my_matrix_from_pf_series(series, k)
        if my is not None:
            if std_params is not None:
                my = standardize_my_matrix(my, std_params)
            out["M_Y"] = my

    if me_week is not None:
        me = np.asarray(me_week, dtype=float)
        if me.ndim == 2 and me.shape[1] >= 4 and np.any(np.isfinite(me)):
            me6 = np.zeros((N_RL_DAYS, 4), dtype=float)
            n = min(N_RL_DAYS, me.shape[0])
            me6[:n, :] = np.nan_to_num(me[:n, :4], nan=0.0)
            if std_params is not None:
                me6 = standardize_me_matrix(me6, std_params)
            out["M_E"] = me6

    return coerce_state(out)


def refresh_state_hist_from_observations(
    state_hist: dict[str, dict],
    *,
    series: dict[str, Any] | None,
    me_by_k: dict[int, Any] | None,
    std_params: dict | None = None,
) -> dict[str, dict]:
    """Rebuild ``C`` / ``M_Y`` / ``M_E`` on every stored training state from DB series."""
    me_by_k = me_by_k or {}
    out: dict[str, dict] = {}
    for key, payload in state_hist.items():
        try:
            state = deserialize_state(payload)
            w, d, t = state_index(state)
            k = internal_week_index(w)
            rebuilt = overlay_observation_covariates_on_state(
                state,
                k=k,
                d=d,
                t=t,
                series=series,
                me_week=(me_by_k.get(k) or {}).get("M_E")
                if isinstance(me_by_k.get(k), dict)
                else me_by_k.get(k),
                std_params=std_params,
            )
            out[key] = serialize_state(rebuilt)
        except (KeyError, TypeError, ValueError):
            out[key] = payload
    return out


def state_hist_lookup(state_hist: dict[str, dict]):
    """Build ``get_state(k, d, t)`` for ``build_rl_training_data``."""

    def get_state(k: int, d: int, t: int) -> dict:
        key = state_key(k, d, t)
        if key not in state_hist:
            raise KeyError(f"missing RL state for {key}")
        return feature_state(deserialize_state(state_hist[key]))

    return get_state


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
