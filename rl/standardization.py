"""Online standardization helpers for RL deployment contexts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_STD_PARAMS_PATH = Path(__file__).with_name("std_params.json")


def load_std_params(path: str | Path = DEFAULT_STD_PARAMS_PATH) -> dict:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _zscore_raw_count(value: Any, std: dict, shift_key: str, scale_key: str) -> float:
    logged = np.log(float(value) + 1.0)
    return float((logged - float(std[shift_key])) / float(std[scale_key]))


def _zscore_value(value: Any, std: dict, shift_key: str, scale_key: str) -> float:
    return float((float(value) - float(std[shift_key])) / float(std[scale_key]))


def standardize_context(context: dict, std: dict) -> dict:
    """Return a copy with raw deployment fields converted to model scale.

    This is intentionally conservative: fields already present in model-scale
    names are left unchanged. Use only when the API payload contains raw values;
    otherwise keep ``standardize_context`` disabled in ``config.json``.
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
        "EMA_StepCount": (
            "stepCountLast7DaysEma",
            "EMA_step_count_shift",
            "EMA_step_count_scale",
            False,
        ),
        "stepCountLast7DaysEmaRaw": (
            "stepCountLast7DaysEma",
            "EMA_step_count_shift",
            "EMA_step_count_scale",
            False,
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

    return out
