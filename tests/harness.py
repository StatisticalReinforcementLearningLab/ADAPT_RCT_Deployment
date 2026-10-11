"""In-memory deployment framework used by the RL_agents tests.

It reproduces the server contract the handler relies on: per-participant
DBReader, observations with ids and server timestamps, latest-wins stored
values, ActionRecords built from ``make_state``/``get_action``, and the
Monday-night ``prepare_for_update`` -> ``update`` sequence (both after
Monday's two decisions). Data are
synthetic but on the raw deployment scale.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np

from RL_agents.base import ActionRecord, ObservationRecord, StoredValue

START = date(2026, 1, 4)  # a Sunday


@dataclass
class FakeDB:
    user_id: str
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    stored: dict = field(default_factory=dict)
    retrieved: dict = field(default_factory=dict)
    _obs_id: int = 0
    _act_id: int = 0
    clock: datetime = datetime(2026, 1, 4, tzinfo=timezone.utc)

    def tick(self):
        self.clock += timedelta(minutes=1)
        return self.clock

    def add_obs(self, payload, obs_type="rl_observation", obs_id=None):
        if obs_id is None:
            self._obs_id += 1
            obs_id = self._obs_id
        ts = self.tick()
        self.observations.append(
            ObservationRecord(obs_id, self.user_id, obs_type, copy.deepcopy(payload), ts, ts)
        )
        return obs_id

    # DBReader interface -------------------------------------------------
    def get_actions(self, action_type=None, limit=None, before=None):
        return [a for a in self.actions if action_type is None or a.action_type == action_type]

    def get_observations(self, observation_type=None, since=None, before=None, limit=None):
        return list(self.observations)

    def get_stored_value(self, type):
        return self.stored.get(type)

    def get_retrieved_data(self):
        return self.retrieved


class Participant:
    """Synthetic participant with a mild true send effect on steps."""

    def __init__(self, handler, user_id="p1", seed=0, send_effect=0.3):
        self.h = handler
        self.db = FakeDB(user_id)
        self.rng = np.random.default_rng(seed)
        self.params = handler.get_initial_model_params(user_id)
        self.send_effect = send_effect
        self.results = []
        self.statuses = []
        self.db.retrieved["participantinfo"] = {"study_start_date": START.isoformat()}
        self.db.add_obs({"value": 5.2}, obs_type="rl_Y_1")
        self.M_Y = {}
        self.M_E = {}

    # data generation ----------------------------------------------------
    def _slot_context(self, k, d, t, missing_c=False):
        # server context: yesterday steps, prior-2h steps,
        # active-day fraction, suggestions sent (7d), interaction fraction
        c = [
            float(self.rng.integers(2000, 9000)),
            float(self.rng.integers(0, 1500)),
            float(self.rng.uniform(0, 1)),
            float(self.rng.integers(0, 10)),
            float(self.rng.uniform(0, 1)),
        ]
        if missing_c:
            return
        self.db.add_obs({"w": k + 1, "d": d, "t": t, "C": c})

    def _after_slot(self, k, d, t, action):
        my = self.M_Y.setdefault(k, np.full((6, 3), np.nan))
        me = self.M_E.setdefault(k, np.full((6, 4), np.nan))
        my[d, t] = max(0.0, self.rng.normal(900 + self.send_effect * 400 * action, 300))
        me[d, t] = float(self.rng.poisson(1.0))
        if t == 1:
            my[d, 2] = float(np.clip(self.rng.normal(4.5, 1.0), 1, 7))
            me[d, 2] = float(self.rng.random() < 0.8)
            me[d, 3] = float(self.rng.random() < 0.7)
        self.db.add_obs({"w": k + 1, "M_Y": my.tolist(), "M_E": me.tolist()})

    def _end_of_week(self, k, *, skip_outcome=False):
        if skip_outcome:
            return
        full = k % 2 == 0
        # Sunday survey at the end of week k (0-indexed) is labelled with the
        # week it opens: 1-indexed w = k + 2 (J_w, Y_w, U_w in the paper).
        payload = {"w": k + 2, "I_w": 1 if full else 0, "J_w": 1, "U1": 5, "U2": 4}
        y = float(np.clip(self.rng.normal(5.3, 0.6), 1, 7))
        # The short CAE is always computable (from the full items when I_w = 1).
        payload["Y_tilde"] = float(np.clip(y + self.rng.normal(0, 0.3), 1, 7))
        if full:
            payload["Y"] = y
        self.db.add_obs(payload)

    # framework loop -----------------------------------------------------
    def decide(self, k, d, t, *, missing_c=False, available=True):
        self._slot_context(k, d, t, missing_c=missing_c)
        ctx = {"w": k + 1, "d": d, "t": t, "availability": available}
        sr = self.h.make_state(self.db.user_id, ctx, self.db)
        ar = self.h.get_action(self.db.user_id, sr.state, self.params)
        self.db._act_id += 1
        ts = self.db.tick()
        rec = ActionRecord(
            action_id=self.db._act_id,
            user_id=self.db.user_id,
            action_type=self.h.action_type,
            action_idx=f"{k}:{d}:{t}",
            action=ar.action,
            action_prob=ar.action_prob,
            context=ctx,
            state=sr.state,
            retrieved_data=None,
            stored_values=None,
            transformed_observations=None,
            model_params_id=0,
            request_timestamp=ts,
            server_timestamp=ts,
        )
        self.db.actions.append(rec)
        self.results.append((k, d, t, ar))
        self._after_slot(k, d, t, ar.action)
        return ar

    def _retrieved(self, k):
        self.db.retrieved["model_params"] = self.params
        self.db.retrieved["update_week"] = k + 1

    def run_prepare(self, k):
        """Monday night, step 1: beliefs and e_hat for week k."""
        self._retrieved(k)
        prep = self.h.prepare_for_update(self.db.user_id, self.db)
        for sv in prep.stored_values or []:
            self.db.stored[sv.type] = StoredValue(sv.user_id, sv.type, copy.deepcopy(sv.value))
        self.last_prep = prep
        return prep

    def run_rlsvi(self, k):
        """Monday night, step 2: RLSVI refit."""
        self._retrieved(k)
        prep = getattr(self, "last_prep", None)
        rewards = prep.rewards if prep is not None else {}
        res = self.h.update(self.db.user_id, self.params, list(self.db.actions), rewards, self.db)
        self.params = res.model_params
        self.statuses.append(res.status)
        return res

    def run_update(self, k):
        self.run_prepare(k)
        return self.run_rlsvi(k)

    def run_week(self, k, *, update=True, rlsvi=True, missing_c_slots=(), skip_outcome=False,
                 skip_slots=()):
        """One study week. ``update=False`` skips Monday night entirely;
        ``rlsvi=False`` runs prepare but not the RLSVI step (a crash in between)."""
        for d in range(6):
            for t in range(2):
                if (d, t) not in skip_slots:
                    self.decide(k, d, t, missing_c=(d, t) in missing_c_slots)
                else:
                    self._after_slot(k, d, t, 0)  # no decision record; data still flow
                if d == 0 and t == 1 and k >= 1 and update:
                    self.run_prepare(k)
                    if rlsvi:
                        self.run_rlsvi(k)
        self._end_of_week(k, skip_outcome=skip_outcome)
