"""Single data layer for the RL handler: raw observations -> week arrays.

Every consumer (decision state, perceived utility, particle filter, RLSVI
training states) reads the same arrays built here, so ordering, duplicate
handling, labelling, and imputation are defined once.

Labelling contract (0-indexed ``k`` = study week ``w - 1``):

* per-slot payload ``{"w", "d", "t", "C": [6]}``: decision context of that slot
  (layout ``algorithm_helpers.SERVER_C_NAMES``; other lengths are rejected);
* week payload ``{"w", "M_Y": 6x3, "M_E": 6x4}`` (or ``stepCountNext4Hour`` /
  ``dailyAnticipatedAffect``): mediators observed during week ``w``;
* weekly outcome / check-in payload ``{"w", "Y"|"CAE", "Y_tilde"|"CAE_short",
  "I_w", "J_w", "U1", "U2"}``: the Sunday survey at the **end of week w-1**,
  i.e. ``J_w``, ``Y_w``, ``U_w`` in the paper's notation (it opens week ``w``);
* ``rl_Y_1``: baseline CAE.

Observations are processed in ``server_timestamp`` order; exact duplicates
(same ``observation_id``) are ignored; for conflicting values the latest wins.
Missing values stay NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from RL_agents.rl.algorithm_helpers import N_RL_DAYS, N_RL_SLOTS, N_SERVER_C
from RL_agents.rl.other_helpers import (
    deserialize_state,
    internal_week_index,
    ordered_observations,
    state_index,
)
from RL_agents.rl.standardization import (
    standardize_anticipated_affect,
    standardize_c_vector,
    standardize_cae,
    standardize_cae_short,
    standardize_foursc_count,
    standardize_pageview_count,
)

# Server context C (6 entries, see algorithm_helpers.SERVER_C_NAMES). Entries
# imputed by the mean of the previous 7 days (point measures) vs. carried
# forward (rolling 7-day summaries).
C_MEAN_7D_DAILY = (0,)      # yesterday's step count (one value per day)
C_MEAN_7D_SAME_SLOT = (2,)  # prior-2-hour step count (same slot, previous 7 days)
C_LOCF = (1, 3, 4, 5)       # 7-day EMA, active days, burden, interaction


def _set_context(data: "WeekData", slot: tuple, arr: np.ndarray) -> None:
    """Store one server context vector; a wrong length is rejected, not truncated."""
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size != N_SERVER_C:
        data.bad_context.append((*slot, int(arr.size)))
        return
    data.C[slot] = arr


def _week_of(payload: dict) -> int | None:
    for key in ("w", "rl_week", "week"):
        if payload.get(key) is not None:
            return internal_week_index(int(payload[key]))
    if payload.get("k") is not None:
        return int(payload["k"])
    return None


def _opt(value: Any) -> float:
    if value is None:
        return np.nan
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


@dataclass
class WeekData:
    W: int
    C: np.ndarray = field(init=False)           # (W, 6, 2, 6) raw server context
    foursc: np.ndarray = field(init=False)      # (W, 6, 2) raw steps
    antic: np.ndarray = field(init=False)       # (W, 6) raw 1-7
    pv: np.ndarray = field(init=False)          # (W, 6, 2) raw counts
    wear: np.ndarray = field(init=False)        # (W, 6)
    pj: np.ndarray = field(init=False)          # (W, 6)
    Y: np.ndarray = field(init=False)           # (W+1,) raw, label = opening week
    Y_tilde: np.ndarray = field(init=False)
    I_w: np.ndarray = field(init=False)
    J_w: np.ndarray = field(init=False)
    U1: np.ndarray = field(init=False)
    U2: np.ndarray = field(init=False)
    A: np.ndarray = field(init=False)           # (W, 6, 2) delivered action
    decided: np.ndarray = field(init=False)     # (W, 6, 2) a decision record exists
    conflicts: list = field(default_factory=list)

    def __post_init__(self):
        W = self.W
        self.C = np.full((W, N_RL_DAYS, N_RL_SLOTS, N_SERVER_C), np.nan)
        self.foursc = np.full((W, N_RL_DAYS, N_RL_SLOTS), np.nan)
        self.antic = np.full((W, N_RL_DAYS), np.nan)
        self.pv = np.full((W, N_RL_DAYS, N_RL_SLOTS), np.nan)
        self.wear = np.full((W, N_RL_DAYS), np.nan)
        self.pj = np.full((W, N_RL_DAYS), np.nan)
        for name in ("Y", "Y_tilde", "I_w", "J_w", "U1", "U2"):
            setattr(self, name, np.full(W + 1, np.nan))
        self.A = np.zeros((W, N_RL_DAYS, N_RL_SLOTS))
        self.decided = np.zeros((W, N_RL_DAYS, N_RL_SLOTS), dtype=bool)
        self.bad_context: list[tuple] = []

    # ------------------------------------------------------------------
    # imputation / model-scale views
    # ------------------------------------------------------------------
    def c_imputed(self, k: int, d: int, t: int) -> tuple[np.ndarray, list[int]]:
        """Raw server context C for a slot with missing entries imputed from the past 7 days.

        Point measures (yesterday's steps, prior-2-hour steps) use the mean of
        the previous 7 days' values (same slot for prior-2-hour); rolling
        summaries carry the last observed value forward. Returns the vector
        and the indices that were imputed. Entries with no usable history stay NaN.
        """
        c = self.C[k, d, t].copy()
        imputed: list[int] = []
        if np.all(np.isfinite(c)):
            return c, imputed
        order = [(kk, dd, tt) for kk in range(self.W) for dd in range(N_RL_DAYS) for tt in range(N_RL_SLOTS)]
        pos = order.index((k, d, t))
        past = order[max(0, pos - 7 * N_RL_SLOTS):pos]  # previous 7 decision days
        for i in range(N_SERVER_C):
            if np.isfinite(c[i]):
                continue
            if i in C_MEAN_7D_DAILY:
                vals, seen_days = [], set()
                for kk, dd, tt in past:
                    v = self.C[kk, dd, tt, i]
                    if np.isfinite(v) and (kk, dd) not in seen_days:
                        seen_days.add((kk, dd))
                        vals.append(v)
                if vals:
                    c[i] = float(np.mean(vals))
                    imputed.append(i)
            elif i in C_MEAN_7D_SAME_SLOT:
                vals = [self.C[kk, dd, tt, i] for kk, dd, tt in past if tt == t]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    c[i] = float(np.mean(vals))
                    imputed.append(i)
            else:
                for kk, dd, tt in reversed(order[:pos]):
                    v = self.C[kk, dd, tt, i]
                    if np.isfinite(v):
                        c[i] = float(v)
                        imputed.append(i)
                        break
        return c, imputed

    def antic_locf(self) -> np.ndarray:
        """Agent-visible anticipated affect: last observed value carried forward."""
        flat = self.antic.ravel().copy()
        last = np.nan
        for i in range(flat.size):
            if np.isfinite(flat[i]):
                last = flat[i]
            else:
                flat[i] = last
        return flat.reshape(self.antic.shape)

    def model_mediators(self, k: int, std: dict) -> tuple[np.ndarray, np.ndarray]:
        """Model-scale ``(M_Y (6,3), M_E (6,4))`` for week ``k`` (NaN = missing).

        Anticipated affect uses the agent-visible LOCF series, as in the testbed.
        """
        M_Y = np.full((N_RL_DAYS, 3), np.nan)
        M_E = np.full((N_RL_DAYS, 4), np.nan)
        antic = self.antic_locf()
        for d in range(N_RL_DAYS):
            for t in range(N_RL_SLOTS):
                v = self.foursc[k, d, t]
                M_Y[d, t] = standardize_foursc_count(v, std) if np.isfinite(v) else np.nan
                p = self.pv[k, d, t]
                M_E[d, t] = standardize_pageview_count(p, std) if np.isfinite(p) else np.nan
            a = antic[k, d]
            M_Y[d, 2] = standardize_anticipated_affect(a) if np.isfinite(a) else np.nan
            M_E[d, 2] = self.wear[k, d]
            M_E[d, 3] = self.pj[k, d]
        return M_Y, M_E

    def model_c(self, k: int, d: int, t: int, std: dict) -> tuple[np.ndarray, list[int]]:
        c, imputed = self.c_imputed(k, d, t)
        return standardize_c_vector(c, std), imputed

    def model_outcome(self, label: int, std: dict) -> dict:
        Y = self.Y[label]
        Yt = self.Y_tilde[label]
        return {
            "Y": float(standardize_cae(Y, std)) if np.isfinite(Y) else None,
            "Y_tilde": float(standardize_cae_short(Yt, std)) if np.isfinite(Yt) else None,
            "I_w": None if not np.isfinite(self.I_w[label]) else int(self.I_w[label]),
            "J_w": None if not np.isfinite(self.J_w[label]) else int(self.J_w[label]),
        }


def collect_week_data(db_reader, *, W: int, action_type: str) -> WeekData:
    data = WeekData(W)
    # Actions: earliest record per slot is authoritative.
    seen: dict[tuple, tuple] = {}
    for rec in sorted(
        db_reader.get_actions(action_type=action_type) or [],
        key=lambda r: (getattr(r, "server_timestamp", None) is None,
                       getattr(r, "server_timestamp", None) or 0, r.action_id),
    ):
        try:
            st = deserialize_state(rec.state or rec.context or {})
            w, d, t = state_index(st)
        except (KeyError, TypeError, ValueError):
            continue
        k = internal_week_index(w)
        if not (0 <= k < W and 0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS):
            continue
        val = (int(rec.action), float(rec.action_prob))
        if (k, d, t) in seen:
            if seen[(k, d, t)] != val:
                data.conflicts.append((k, d, t))
            continue
        seen[(k, d, t)] = val
        data.A[k, d, t] = float(rec.action)
        data.decided[k, d, t] = True

    for obs in ordered_observations(db_reader):
        payload = getattr(obs, "payload", {}) or {}
        if not isinstance(payload, dict):
            continue
        src = payload.get("weekly_series", payload)
        if not isinstance(src, dict):
            src = payload
        k = _week_of(src)
        if k is None:
            k = _week_of(payload)
        if k is None or not (0 <= k <= W):
            continue
        # decision context
        C = src.get("C")
        if C is not None and k < W:
            arr = np.asarray(C, dtype=float)
            d, t = src.get("d", src.get("day")), src.get("t", src.get("slot"))
            if arr.ndim == 1 and d is not None and t is not None:
                d, t = int(d), int(t)
                if 0 <= d < N_RL_DAYS and 0 <= t < N_RL_SLOTS:
                    _set_context(data, (k, d, t), arr)
            elif arr.ndim == 3 and arr.shape[:2] == (N_RL_DAYS, N_RL_SLOTS):
                for d in range(N_RL_DAYS):
                    for t in range(N_RL_SLOTS):
                        _set_context(data, (k, d, t), arr[d, t])
            elif arr.ndim == 2 and arr.shape[0] == N_RL_DAYS * N_RL_SLOTS:
                for i in range(N_RL_DAYS * N_RL_SLOTS):
                    _set_context(data, (k, i // N_RL_SLOTS, i % N_RL_SLOTS), arr[i])
        # mediators
        if k < W:
            my = src.get("M_Y", src.get("M^Y"))
            if my is not None:
                m = np.asarray(my, dtype=float)
                if m.ndim == 2 and m.shape[1] >= 3:
                    n = min(N_RL_DAYS, m.shape[0])
                    _merge(data.foursc[k, :n], m[:n, 0:2])
                    _merge(data.antic[k, :n], m[:n, 2])
            me = src.get("M_E", src.get("M^E"))
            if me is not None:
                m = np.asarray(me, dtype=float)
                if m.ndim == 2 and m.shape[1] >= 4:
                    n = min(N_RL_DAYS, m.shape[0])
                    _merge(data.pv[k, :n], m[:n, 0:2])
                    _merge(data.wear[k, :n], m[:n, 2])
                    _merge(data.pj[k, :n], m[:n, 3])
            if src.get("stepCountNext4Hour") is not None:
                arr = np.asarray(src["stepCountNext4Hour"], dtype=float).ravel()[: N_RL_DAYS * N_RL_SLOTS]
                tmp = np.full(N_RL_DAYS * N_RL_SLOTS, np.nan)
                tmp[: arr.size] = arr
                _merge(data.foursc[k], tmp.reshape(N_RL_DAYS, N_RL_SLOTS))
            if src.get("dailyAnticipatedAffect") is not None:
                arr = np.asarray(src["dailyAnticipatedAffect"], dtype=float).ravel()[:N_RL_DAYS]
                tmp = np.full(N_RL_DAYS, np.nan)
                tmp[: arr.size] = arr
                _merge(data.antic[k], tmp)
        # weekly outcome / check-in (label = opening week)
        for dst, keys in (
            ("Y", ("Y", "CAE", "cae", "Y_w")),
            ("Y_tilde", ("Y_tilde", "CAE_short", "cae_short", "tilde_Y")),
            ("I_w", ("I_w", "query_action")),
            ("J_w", ("J_w", "wp", "week_present")),
            ("U1", ("U1", "Exp-tool-1")),
            ("U2", ("U2", "Exp-tool-2")),
        ):
            for key in keys:
                if src.get(key) is not None:
                    v = _opt(src.get(key))
                    if np.isfinite(v):
                        getattr(data, dst)[k] = v
                    break
    return data


def _merge(dst: np.ndarray, src: np.ndarray) -> None:
    """Write finite entries of ``src`` into ``dst`` (later observations win)."""
    src = np.asarray(src, dtype=float)
    mask = np.isfinite(src)
    dst[mask] = src[mask]
