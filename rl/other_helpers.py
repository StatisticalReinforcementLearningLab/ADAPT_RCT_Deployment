"""State, serialization, seeding, and validation helpers for ``RL.py``."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from RL_agents.base import DBReader, StoredValue
from RL_agents.logging_config import RED, get_logger, log_alert
from RL_agents.rl.algorithm_helpers import (
    N_RL_DAYS,
    build_phi_action,
    build_rl_context_vector,
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
    return coerce_state(state)


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
