"""Framework-facing weekly query policy.

This handler implements the simple query rule used by the current experiment
baselines:

    I_2 = 1
    I_w ~ Bernoulli(0.5), for all other weeks

Policy knobs (``query_prob``, ``force_query_week``, ``seed``) live in
``config.json``.

Study weeks are 1-indexed and aligned to Sundays. The enrollment/start date
should fall on the enrollment Sunday (``w=1``); no query is issued that week.
The following Sunday is ``w=2`` (``I_2``). If the stored start date is not a
Sunday, it is snapped to the most recent preceding Sunday.

Each Bernoulli draw uses a deterministic seed derived from the configured
seed, participant ID, and study week.

``make_state`` derives the 1-indexed study week ``w`` from the request context
(decision datetime + timezone) and participant enrollment metadata from
``db_reader.get_retrieved_data()``. Query and RL both use ``w`` as the public
study-week field. Incoming aliases ``rl_week`` and ``week`` are also accepted
as 1-indexed study weeks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from RL_agents.base import (
    ActionHandlerBase,
    ActionRecord,
    ActionResult,
    DBReader,
    StateResult,
    UpdatePrepResult,
    UpdateResult,
)
from RL_agents.logging_config import GREEN, RED, YELLOW, get_logger, log_alert


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
logger = get_logger(__name__)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _stable_seed(value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(str(value).encode("utf-8"), digest_size=4).digest(),
        "little",
    )


def _decision_seed(base_seed: Any, user_id: str, week: int) -> int:
    seed_material = json.dumps(
        [base_seed, str(user_id), int(week)],
        separators=(",", ":"),
    )
    return _stable_seed(seed_material)


def _week_from_state(state: dict) -> int:
    """Return the 1-indexed query week from a pre-built state."""
    w = _optional_week_from_state(state)
    if w is None:
        raise KeyError("Query state must include w or calendar-derived week indices")
    return w


def _optional_week_from_state(state: dict) -> int | None:
    """Return 1-indexed study week from ``w`` or accepted aliases."""
    for key in ("w", "rl_week", "week"):
        if key in state and state[key] is not None:
            return int(state[key])
    return None


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        return datetime.fromisoformat(text)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _context_datetime(context: dict) -> datetime | None:
    for key in ("decision_time", "decisionTime", "timestamp", "request_time"):
        if key in context and context[key] is not None:
            return _parse_datetime(context[key])
    return None


def _context_timezone(context: dict) -> str:
    for key in ("timezone", "time_zone", "tz"):
        value = context.get(key)
        if value:
            return str(value)
    raise KeyError("Query context must include timezone, time_zone, or tz")


def _participant_info(db_reader: DBReader) -> dict:
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


def _study_start_date(participant: dict) -> date:
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
        return _parse_datetime(value).date()
    raise KeyError(
        "participantinfo must include study_start_date, enroll_date, or date_min"
    )


def _sunday_on_or_before(d: date) -> date:
    """Return the Sunday on or before ``d`` (Python weekday: Sun=6)."""
    return d - timedelta(days=(d.weekday() + 1) % 7)


def _study_week(
    study_start: date,
    local_decision_date: date,
) -> int:
    """Return the 1-indexed study week for ``local_decision_date``.

    ``study_start`` is snapped to the enrollment Sunday, which starts ``w=1``.
    The next Sunday is ``w=2``.
    """
    enrollment_sunday = _sunday_on_or_before(study_start)
    day = (local_decision_date - enrollment_sunday).days
    if day < 0:
        raise ValueError(
            f"Decision date {local_decision_date} is before enrollment Sunday "
            f"{enrollment_sunday}"
        )

    return day // 7 + 1


class Query(ActionHandlerBase):
    """Weekly query decision handler with the ``ActionHandlerBase`` API."""

    action_type = "weekly_query"

    def __init__(self, config: dict | None = None):
        self.config = load_config() if config is None else config

    def get_initial_model_params(self, user_id: str) -> dict:
        return {
            "version": 2,
            "user_id": str(user_id),
        }

    def make_state(
        self,
        user_id: str,
        context: dict,
        db_reader: DBReader,
    ) -> StateResult:
        context_copy = dict(context or {})
        try:
            decision_dt = _context_datetime(context_copy)
        except (TypeError, ValueError) as exc:
            log_alert(
                logger,
                YELLOW,
                "query decision timestamp parsing failed",
                action_type=self.action_type,
                user_id=str(user_id),
                error=str(exc),
            )
            raise

        if decision_dt is None:
            try:
                context_copy["w"] = _week_from_state(context_copy)
            except (KeyError, TypeError, ValueError) as exc:
                log_alert(
                    logger,
                    YELLOW,
                    "query state missing decision time and context week",
                    action_type=self.action_type,
                    user_id=str(user_id),
                    error=str(exc),
                )
                raise
            return StateResult(state=context_copy)

        try:
            tz = ZoneInfo(_context_timezone(context_copy))
        except KeyError as exc:
            log_alert(
                logger,
                YELLOW,
                "query state missing or invalid timezone",
                action_type=self.action_type,
                user_id=str(user_id),
                error=str(exc),
            )
            raise

        if decision_dt.tzinfo is None:
            decision_dt = decision_dt.replace(tzinfo=tz)
        local_dt = decision_dt.astimezone(tz)

        try:
            participant = _participant_info(db_reader)
            study_start = _study_start_date(participant)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            w = _optional_week_from_state(context_copy)
            if w is None:
                log_alert(
                    logger,
                    YELLOW,
                    "query state missing enrollment metadata and context week",
                    action_type=self.action_type,
                    user_id=str(user_id),
                    error=str(exc),
                )
                raise
            log_alert(
                logger,
                YELLOW,
                "using week from context; enrollment metadata missing",
                action_type=self.action_type,
                user_id=str(user_id),
                w=w,
                error=str(exc),
            )
        else:
            enrollment_sunday = _sunday_on_or_before(study_start)
            # if study_start != enrollment_sunday:
            #     log_alert(
            #         logger,
            #         GREEN,
            #         "study_start is not a Sunday; snapped to enrollment Sunday",
            #         action_type=self.action_type,
            #         user_id=str(user_id),
            #         study_start=study_start.isoformat(),
            #         enrollment_sunday=enrollment_sunday.isoformat(),
            #     )
            try:
                w = _study_week(study_start, local_dt.date())
            except ValueError as exc:
                log_alert(
                    logger,
                    RED,
                    "decision date is before enrollment Sunday",
                    action_type=self.action_type,
                    user_id=str(user_id),
                    local_decision_date=local_dt.date().isoformat(),
                    enrollment_sunday=enrollment_sunday.isoformat(),
                    context_w=_optional_week_from_state(context_copy),
                    error=str(exc),
                )
                raise

        context_copy.update(
            {
                "w": w,
                "decision_time_local": local_dt.isoformat(),
            }
        )

        return StateResult(state=context_copy)

    def get_action(
        self,
        user_id: str,
        state: dict,
        model_params: dict,
    ) -> ActionResult:
        try:
            w = _week_from_state(state or {})
        except (KeyError, TypeError, ValueError) as exc:
            log_alert(
                logger,
                YELLOW,
                "query action selection missing valid week",
                action_type=self.action_type,
                user_id=str(user_id),
                error=str(exc),
            )
            raise
        force_week = int(self.config.get("force_query_week", 2))
        query_prob = float(self.config.get("query_prob", 0.5))
        decision_seed = _decision_seed(self.config.get("seed"), user_id, w)

        if w == 1:
            log_alert(
                logger,
                YELLOW,
                "query handler invoked during enrollment week w=1",
                action_type=self.action_type,
                user_id=str(user_id),
                w=w,
            )

        if w == force_week:
            action = 1
            action_prob = 1.0
        else:
            action_prob = query_prob
            if not np.isfinite(action_prob) or not (0.0 <= action_prob <= 1.0):
                log_alert(
                    logger,
                    RED,
                    "invalid query action-selection probability",
                    action_type=self.action_type,
                    user_id=str(user_id),
                    w=w,
                    action_prob=action_prob,
                )
                raise ValueError(
                    f"query_prob must be finite and in [0, 1], got {action_prob}"
                )
            rng = np.random.default_rng(decision_seed)
            action = int(rng.binomial(1, action_prob))

        if action not in (0, 1):
            log_alert(
                logger,
                RED,
                "invalid binary query action",
                action_type=self.action_type,
                user_id=str(user_id),
                w=w,
                action=action,
            )

        return ActionResult(
            action=action,
            action_prob=float(action_prob),
            random_state={"seed": decision_seed},
        )

    def prepare_for_update(
        self,
        user_id: str,
        db_reader: DBReader,
    ) -> UpdatePrepResult:
        return UpdatePrepResult(rewards={})

    def update(
        self,
        user_id: str,
        model_params: dict,
        actions: list[ActionRecord],
        rewards: dict[int, float],
        db_reader: DBReader,
    ) -> UpdateResult:
        return UpdateResult(model_params=model_params)


QueryAgent = Query
