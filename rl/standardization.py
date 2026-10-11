"""Online standardization helpers for RL deployment contexts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from RL_agents.rl.algorithm_helpers import N_SERVER_C

DEFAULT_STD_PARAMS_PATH = Path(__file__).with_name("std_params.json")

# M_Y columns: fourSC_morning, fourSC_afternoon, anticipated_affect
# M_E columns: pageview_morning, pageview_afternoon, morning_fitbit_wear, daily_survey_complete
_MY_FOURSC_COLS = (0, 1)
_MY_ANTIC_COL = 2
_ME_PAGEVIEW_COLS = (0, 1)
# M_E columns 2–3 (morning_fitbit_wear, daily_survey_complete) are binary and left raw.


def load_std_params(path: str | Path = DEFAULT_STD_PARAMS_PATH) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _zscore_raw_count(value: Any, std: dict, shift_key: str, scale_key: str) -> float:
    logged = np.log(float(value) + 1.0)
    return float((logged - float(std[shift_key])) / float(std[scale_key]))


def _zscore_value(value: Any, std: dict, shift_key: str, scale_key: str) -> float:
    return float((float(value) - float(std[shift_key])) / float(std[scale_key]))


def _likert_1_7_to_unit(value: Any) -> float:
    """Map deployment Likert scores on 1–7 to approximately [-1, 1].

    Offline training data used a 0–7 coding (``2*x/7 - 1`` in
    ``3_standardization.py``). Deployment sends 1–7, so we use the matching
    endpoint-preserving map ``2*(x-1)/6 - 1`` (1 → -1, 7 → 1).
    """
    return float(2.0 * (float(value) - 1.0) / 6.0 - 1.0)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return not np.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _standardize_my_cell(value: Any, col: int, std: dict) -> Any:
    if _is_missing(value):
        return value
    if col in _MY_FOURSC_COLS:
        return _zscore_raw_count(
            value, std, "4hour_step_count_shift", "4hour_step_count_scale"
        )
    if col == _MY_ANTIC_COL:
        return _likert_1_7_to_unit(value)
    return float(value)


def standardize_pageview_count(value: Any, std: dict) -> Any:
    """Hourly page-view count -> model scale.

    Current artifacts (``HourlyPageviewCount_transform == "log1p"``) and old
    artifacts without either key use ``log(x + 1)``, then z-score. Artifacts
    written with ``HourlyPageviewCount_zero_count`` (Sept 2026 testbed) use
    ``log(x)`` for ``x > 0`` and ``log(zero_count)`` for zero.
    """
    if _is_missing(value):
        return value
    zc = std.get("HourlyPageviewCount_zero_count")
    if zc is None:
        return _zscore_raw_count(
            value, std, "HourlyPageviewCount_shift", "HourlyPageviewCount_scale"
        )
    x = float(value)
    if x < 0:
        raise ValueError(f"negative page-view count {x!r}")
    logged = np.log(x) if x > 0 else np.log(float(zc))
    return float(
        (logged - float(std["HourlyPageviewCount_shift"]))
        / float(std["HourlyPageviewCount_scale"])
    )


def uses_legacy_pageview_transform(std: dict) -> bool:
    """True for artifacts that predate the explicit page-view transform key."""
    if std.get("HourlyPageviewCount_transform") == "log1p":
        return False
    return std.get("HourlyPageviewCount_zero_count") is None


def _standardize_me_cell(value: Any, col: int, std: dict) -> Any:
    if _is_missing(value):
        return value
    if col in _ME_PAGEVIEW_COLS:
        return standardize_pageview_count(value, std)
    # Binary wear / survey indicators stay on the raw {0, 1} scale.
    return float(value)


def _standardize_mediator_matrix(
    matrix: Any,
    *,
    n_cols: int,
    cell_fn,
    std: dict,
) -> list[list[Any]]:
    arr = np.asarray(matrix, dtype=object)
    if arr.ndim == 1:
        if arr.size % n_cols != 0:
            raise ValueError(
                f"mediator length {arr.size} is not divisible by {n_cols}"
            )
        arr = arr.reshape(arr.size // n_cols, n_cols)
    if arr.ndim != 2 or arr.shape[1] != n_cols:
        raise ValueError(f"mediator matrix must have shape (*, {n_cols})")
    out: list[list[Any]] = []
    for row in arr:
        out.append([cell_fn(row[j], j, std) for j in range(n_cols)])
    return out


def standardize_foursc_count(value: Any, std: dict) -> Any:
    """``log(x+1)`` then z-score using ``4hour_step_count_*``."""
    if _is_missing(value):
        return value
    return _zscore_raw_count(
        value, std, "4hour_step_count_shift", "4hour_step_count_scale"
    )


def standardize_anticipated_affect(value: Any, std: dict | None = None) -> Any:
    """Deployment Likert 1–7 → [-1, 1]. ``std`` unused; kept for API symmetry."""
    del std  # unused
    if _is_missing(value):
        return value
    return _likert_1_7_to_unit(value)


def standardize_cae(value: Any, std: dict) -> Any:
    """Weekly full CAE / ``Y_1`` z-score using ``CAE_avg_*``."""
    if _is_missing(value):
        return value
    return _zscore_value(value, std, "CAE_avg_shift", "CAE_avg_scale")


def standardize_cae_short(value: Any, std: dict) -> Any:
    """Short CAE z-score using ``CAE_short_avg_*``."""
    if _is_missing(value):
        return value
    return _zscore_value(value, std, "CAE_short_avg_shift", "CAE_short_avg_scale")


def standardize_today_step_count(value: Any, std: dict) -> Any:
    """``log(x+1)`` then z-score using ``today_step_count_*``."""
    if _is_missing(value):
        return value
    return _zscore_raw_count(
        value, std, "today_step_count_shift", "today_step_count_scale"
    )


def standardize_my_matrix(M_Y: Any, std: dict) -> np.ndarray:
    """Standardize an ``M_Y`` ``(*, 3)`` matrix to model scale."""
    return np.asarray(
        _standardize_mediator_matrix(
            M_Y, n_cols=3, cell_fn=_standardize_my_cell, std=std
        ),
        dtype=float,
    )


def standardize_me_matrix(M_E: Any, std: dict) -> np.ndarray:
    """Standardize an ``M_E`` ``(*, 4)`` matrix to model scale."""
    return np.asarray(
        _standardize_mediator_matrix(
            M_E, n_cols=4, cell_fn=_standardize_me_cell, std=std
        ),
        dtype=float,
    )


def standardize_c_vector(C: Any, std: dict, *, pad_value: float = np.nan) -> np.ndarray:
    """Standardize one 5-entry server context vector (raw deployment scale).

    ``C[0]`` yesterday steps: ``log1p`` + z; ``C[1]`` prior-2h steps:
    ``log1p`` + z; ``C[3]`` recent burden: z.
    ``C[2]`` / ``C[4]`` (fractions) are left unchanged.
    """
    c = np.asarray(C, dtype=float).ravel()
    out = np.full(N_SERVER_C, float(pad_value), dtype=float)
    m = min(int(c.size), N_SERVER_C)
    if m:
        out[:m] = c[:m]
    if not _is_missing(out[0]):
        out[0] = _zscore_raw_count(
            out[0], std, "yesterday_step_count_shift", "yesterday_step_count_scale"
        )
    if not _is_missing(out[1]):
        out[1] = _zscore_raw_count(
            out[1], std, "prior2hour_step_count_shift", "prior2hour_step_count_scale"
        )
    if not _is_missing(out[3]):
        out[3] = _zscore_value(
            out[3], std, "recent_burden_shift", "recent_burden_scale"
        )
    return out


def _map_finite(arr: np.ndarray, fn) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    flat = out.ravel()
    for i in range(flat.size):
        if np.isfinite(flat[i]):
            flat[i] = float(fn(flat[i]))
    return out


def standardize_pf_series(series: dict[str, Any], std: dict) -> dict[str, Any]:
    """Return a copy of PF week series with raw observation fields model-scaled.

    Applied **before** ``build_pf_data_from_series`` so design-row interactions
    use standardized covariates. Assumes series ``C`` / mediators / CAE arrived
    from observations at raw deployment scale (not action-time state).

    Only slots marked in ``C_filled`` are transformed (unfilled slots stay 0).
    """
    out = dict(series)
    C = np.asarray(series["C"], dtype=float).copy()
    filled = np.asarray(
        series.get("C_filled", np.ones(C.shape[:3], dtype=bool)), dtype=bool
    )
    for idx in np.ndindex(C.shape[:3]):
        if filled[idx]:
            # PF design rows keep the legacy zero padding for absent entries.
            C[idx] = standardize_c_vector(C[idx], std, pad_value=0.0)
    out["C"] = C
    out["C_filled"] = filled

    out["M_Y_fourSC"] = _map_finite(
        series["M_Y_fourSC"], lambda v: standardize_foursc_count(v, std)
    )
    out["M_Y_antic"] = _map_finite(
        series["M_Y_antic"], lambda v: standardize_anticipated_affect(v, std)
    )
    out["Y"] = _map_finite(series["Y"], lambda v: standardize_cae(v, std))
    out["Y_tilde"] = _map_finite(
        series["Y_tilde"], lambda v: standardize_cae_short(v, std)
    )

    antic_cov = dict(series.get("antic_cov") or {})
    if "today_step_count" in antic_cov:
        antic_cov["today_step_count"] = _map_finite(
            antic_cov["today_step_count"],
            lambda v: standardize_today_step_count(v, std),
        )
    out["antic_cov"] = antic_cov
    return out


def standardize_context(context: dict, std: dict) -> dict:
    """Return a copy with raw deployment fields converted to model scale.

    Transforms applied when the corresponding raw fields are present:

    - Context counts (``YesterdayStepCount``, ``prior2HourStepCountRaw``, …):
      ``log(x+1)`` then z-score, or z-score only for already-continuous fields
      (``recentBurdenRaw``).
    - ``M_Y`` mediators: fourSC columns use ``log(x+1)`` + z-score
      (``4hour_step_count_*``); anticipated affect (deployment 1–7) maps to
      [-1, 1] via ``2*(x-1)/6 - 1``.
    - ``M_E`` mediators: pageview columns use ``log(x+1)`` + z-score
      (``HourlyPageviewCount_*``); morning wear / daily survey stay binary.

    Fields already present under model-scale context names are left unchanged.
    Non-finite mediator cells are preserved as-is. Use only when the API
    payload contains raw values; otherwise keep ``standardize_context``
    disabled in ``config.json``.
    """
    out = dict(context or {})

    raw_to_model = {
        "YesterdayStepCount": (
            "yesterdayStepCount",
            "yesterday_step_count_shift",
            "yesterday_step_count_scale",
            True,
        ),
        "yesterdayStepCountRaw": (
            "yesterdayStepCount",
            "yesterday_step_count_shift",
            "yesterday_step_count_scale",
            True,
        ),
        "prior2HourStepCountRaw": (
            "prior2HourStepCountAgent",
            "prior2hour_step_count_shift",
            "prior2hour_step_count_scale",
            True,
        ),
        "recentBurdenRaw": (
            "activitySuggestionsSentLast7Days",
            "recent_burden_shift",
            "recent_burden_scale",
            False,
        ),
    }

    for raw_key, (model_key, shift_key, scale_key, log_first) in raw_to_model.items():
        if model_key in out or raw_key not in out or out[raw_key] is None:
            continue
        if log_first:
            out[model_key] = _zscore_raw_count(out[raw_key], std, shift_key, scale_key)
        else:
            out[model_key] = _zscore_value(out[raw_key], std, shift_key, scale_key)

    if "M_Y" in out and out["M_Y"] is not None:
        out["M_Y"] = _standardize_mediator_matrix(
            out["M_Y"],
            n_cols=3,
            cell_fn=_standardize_my_cell,
            std=std,
        )
    if "M_E" in out and out["M_E"] is not None:
        out["M_E"] = _standardize_mediator_matrix(
            out["M_E"],
            n_cols=4,
            cell_fn=_standardize_me_cell,
            std=std,
        )

    return out
