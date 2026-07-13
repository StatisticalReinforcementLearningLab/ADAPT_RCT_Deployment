# ADAPT RCT Deployment RL Agents

Python handlers for the ADAPT RCT deployment framework. This package contains
the decision policies used by the server to:

- choose whether to send a weekly query (`Query`)
- choose whether to send a walking suggestion at a real-time decision point
  (`RL`)
- prepare rewards and update per-participant model parameters after new
  observations arrive

The code is designed to be called by an external deployment framework. It does
not define a standalone web server or command-line application.

## Package Contents

```text
RL_agents/
|-- base.py                  # Shared dataclasses and ActionHandlerBase API
|-- logging_config.py        # JSON structured logging helpers
|-- query/
|   |-- Query.py             # Weekly query decision handler
|   `-- config.json          # Query policy settings
`-- rl/
    |-- RL.py                # Walking-suggestion RLSVI handler
    |-- algorithm_helpers.py # Feature construction and RLSVI math
    |-- pf_helpers.py        # Particle-filter belief updates
    |-- ew_helpers.py        # Perceived-utility (E_w) pooled-linear update
    |-- prior_helpers.py     # Prior loading helpers
    |-- state_helpers.py     # State coercion, validation, serialization
    |-- standardization.py   # Optional context standardization
    |-- config.json          # RL policy settings
    |-- rl_priors.json       # Particle-filter prior parameters
    |-- rl_q_priors.json     # RLSVI Q-function prior parameters
    |-- std_params.json      # Context standardization parameters
    `-- Ew_pooled_linear_coefs.json  # Perceived-utility (E_w) weights
```

## Runtime Requirements

- Python 3.10 or newer
- `numpy`
- `scipy`

There is currently no `pyproject.toml` or `requirements.txt` in this package.
Install the runtime dependencies in the Codespace or deployment environment:

```bash
python -m pip install numpy scipy
```

The modules use absolute imports such as `from RL_agents.base import ...`.
Run Python from the parent directory that contains `RL_agents`, or add that
parent directory to `PYTHONPATH`.

```bash
cd ..
python -c "from RL_agents.rl import RL; from RL_agents.query import Query; print(RL.action_type, Query.action_type)"
```

If this repository is cloned as a standalone folder whose name is not
`RL_agents`, rename the folder to `RL_agents` or update the Python path/import
layout before running the package.

## Handler API

Both handlers implement `ActionHandlerBase` from `base.py`.

```python
handler.get_initial_model_params(user_id: str) -> dict
handler.make_state(user_id: str, context: dict, db_reader: DBReader) -> StateResult
handler.get_action(user_id: str, state: dict, model_params: dict) -> ActionResult
handler.prepare_for_update(user_id: str, db_reader: DBReader) -> UpdatePrepResult
handler.update(
    user_id: str,
    model_params: dict,
    actions: list[ActionRecord],
    rewards: dict[int, float],
    db_reader: DBReader,
) -> UpdateResult
```

The deployment framework is responsible for persisting action records,
observation records, stored values, and model parameters. The handlers return
plain dataclasses so the framework can serialize the results into its own
database.

## Quick Usage

```python
from RL_agents.rl import RL
from RL_agents.query import Query

rl_handler = RL()
query_handler = Query()

rl_params = rl_handler.get_initial_model_params(user_id="participant-001")
query_params = query_handler.get_initial_model_params(user_id="participant-001")
```

Configure structured JSON logging once at process startup if the deployment
server wants log files from the handlers:

```python
from RL_agents.logging_config import configure_logging

configure_logging(log_dir="logs", log_file="rl_agents.log")
```

## Weekly Query Handler

Import path:

```python
from RL_agents.query import Query
```

Default policy from `query/config.json`:

- force a query in study week `w = 2`
- randomize all other invoked weeks with probability `query_prob = 0.5`
- use deterministic per-user, per-week random seeds

The framework should treat study weeks as 1-indexed. Week `w = 1` is the
enrollment/baseline week. If the query handler is invoked during `w = 1`, it
logs a warning and applies the configured Bernoulli rule.

### Query State Inputs

`Query.make_state(...)` can derive `w` from request context plus participant
metadata:

```python
context = {
    "decision_time": "2026-07-13T09:00:00-04:00",
    "timezone": "America/New_York",
}
```

`db_reader.get_retrieved_data()` should include participant metadata under one
of these keys:

- `participantinfo`
- `participant_info`
- `ParticipantInfo`

The participant object should include one of these enrollment/start date keys:

- `study_start_date`
- `enroll_date`
- `start_date`
- `StudyStartDate`
- `date_min`

If no decision timestamp is provided, the context must already include a week
field: `w`, `k`, `rl_week`, or `week`.

### Query Action Output

`Query.get_action(...)` returns:

```python
ActionResult(
    action=0 or 1,
    action_prob=<probability of action 1>,
    random_state={"seed": <deterministic_seed>},
)
```

## Walking-Suggestion RL Handler

Import path:

```python
from RL_agents.rl import RL
```

The RL handler uses an ensemble RLSVI policy for binary walking-suggestion
actions:

- `action = 1`: send a walking suggestion
- `action = 0`: do not send a walking suggestion

At internal week `k = 0`, the handler randomizes with probability `0.5`.
After that, it computes the ensemble probability that action `1` has greater
value than action `0`, then clips that probability to
`[epsilon_0, 1 - epsilon_0]`.

Default policy settings live in `rl/config.json`:

- `W = 36` study weeks
- `n_rl_days = 6` decision days per RL week
- `n_rl_slots = 2` decision slots per day
- `B = 50` RLSVI ensemble members
- `epsilon_0 = 0.1` minimum/maximum randomization clipping
- `gamma_bar = 0.5` discount/noise persistence parameter
- `target_update_C = 1`
- `p_rl = 77` runtime feature dimension
- `standardize_context = true`
- prior files: `rl_priors.json`, `rl_q_priors.json`

### RL State Inputs

`RL.make_state(...)` accepts a context dictionary containing the feature-builder
state for one decision point.

Week indexing:

- external framework/query weeks are 1-indexed: `w = 1` means baseline week
- RL internals are 0-indexed: `k = 0` means baseline week
- if context includes `w`, `rl_week`, or `week`, it is converted to `k = w - 1`
- if context includes only `k`, it is assumed to already be 0-indexed

Required decision index fields:

- `k` or external `w`/`rl_week`/`week`
- `d` or `day`: 0-indexed day within the RL week
- `t` or `slot`: 0-indexed decision slot within the day

Required feature fields:

- `E_w`, or `perceivedUtility`
- `M_Y`: mediator matrix with up to 6 rows and 3 columns
- `M_E`: mediator matrix with up to 6 rows and 4 columns
- `C`: 8-element context vector

Instead of `C`, callers may provide the raw context fields:

- `yesterdayStepCount`
- `stepCountLast7DaysEma`
- `prior2HourStepCountAgent`
- `activityCompletedLast7Days`
- `activeDaysLast7Days`
- `activitySuggestionsSentLast7Days`
- `salienceMessageSentYesterday`
- `activitySuggestionInteractLast7Days`

Belief fields:

- `b_hat`
- `b_tilde`

If `b_hat` or `b_tilde` are missing, `make_state(...)` tries to read them from
the framework stored values named in `rl/config.json`:

- `rl_b_hat`
- `rl_b_tilde`

At baseline week `k = 0`, missing beliefs default to `0.0`.

If `E_w`/`perceivedUtility` is missing from the context, `make_state(...)` reads
the perceived utility from the stored value `rl_E_w`, which the weekly E_w update
in `prepare_for_update(...)` maintains.

### Minimal RL Context Example

```python
context = {
    "w": 2,              # external 1-indexed week; converted to internal k = 1
    "d": 0,
    "t": 0,
    "E_w": 0.3,
    "M_Y": [[0, 0, 0]] * 6,
    "M_E": [[0, 0, 0, 0]] * 6,
    "C": [0, 0, 0, 0, 0, 0, 0, 0],
    "b_hat": 0.0,
    "b_tilde": 0.0,
}

state_result = rl_handler.make_state("participant-001", context, db_reader)
action_result = rl_handler.get_action(
    "participant-001",
    state_result.state,
    rl_params,
)
```

### RL Update Flow

The deployment framework typically calls updates in two phases.

1. `prepare_for_update(...)`

   - reads recent observations from `db_reader.get_observations()`
   - extracts explicit rewards from observation payloads containing
     `action_id` plus `reward`, `CAE`, or `cae`
   - consumes the latest particle-filter packet when available, runs the
     particle-filter belief update, refits the online noise variances, and
     assigns the belief mean `b_hat` as the weekly reward
   - runs the weekly perceived-utility (`E_w`) update from pooled linear
     coefficients and stores the result under `rl_E_w`
   - returns rewards, stored belief/PF/`E_w` values, and transformed observations

2. `update(...)`

   - records observed actions into `A_hist`
   - records action probabilities into `pi_A_hist`
   - stores serialized historical states in `state_hist`
   - applies prepared particle-filter belief state when available
   - updates `b_hat_hist` and `b_tilde_hist`
   - refits the RLSVI ensemble when enough historical states and rewards exist
   - returns updated `model_params` and any stored values that should be saved

Particle-filter packets are accepted in observation payloads with type
`rl_pf_packet` or `pf_packet`. A packet can use external `w` or internal `k`;
external `w` is converted to `k = w - 1`.

## Server-Built Data Contract (PF packet + E_w inputs)

The handler consumes weekly design matrices and aggregates; it does **not** build
them. The deployment/server side is responsible for assembling them from the
database for the just-finalized week and delivering them as observation payloads.
This section specifies exactly what to build.

### Perceived-utility (E_w) inputs

The weekly `E_w` update applies pooled, no-intercept linear coefficients
(`Ew_pooled_linear_coefs.json`) to five weekly aggregates. Provide them under an
`ew_inputs` dict, either inside the PF packet (`packet["ew_inputs"]`), inside any
observation payload (`payload["ew_inputs"]`), or as an observation of type
`rl_ew_inputs` whose payload is the inputs dict itself.

| Field | Meaning | How to build from the database |
|-------|---------|--------------------------------|
| `J_w` | Week-present / survey indicator (0 or 1) | Whether the week's survey was completed |
| `PV_sum` | Mean of the 14 standardized per-slot pageview counts | `mean(HourlyPageviewCount_norm)` over the 14 decision slots |
| `FW_sum` | Mean over 7 days of `nextday_wearing` | `nansum(nextday_wearing) / 7` (first row per day) |
| `PJ_sum` | Mean over 7 days of `daily_present` | `nansum(daily_present) / 7` (first row per day) |
| `half_J_tool8` *(optional)* | `0.5 * J_w * ((U1+1) + (U2+1)) / 8` | Provide directly, **or** provide `U1`/`U2` (aliases `Exp-tool-1`/`Exp-tool-2`) and the handler derives it (missing ratings default to 0) |

If `ew_inputs` is absent, the E_w update is skipped. If it is present but missing
`PV_sum`/`FW_sum`/`PJ_sum`, a yellow "perceived-utility (E_w) update failed" alert
is logged and E_w is left unchanged. The resulting `E_w` is stored under `rl_E_w`
and used by `make_state` for the following week's decisions when the request
context omits `E_w`/`perceivedUtility`.

### PF packet

Deliver one packet per completed RL week as an observation with type
`rl_pf_packet` (or a payload containing `pf_data`/`pf_packet`).

| Packet field | Type | Meaning |
|--------------|------|---------|
| `w` or `k` | int | Study week: 1-indexed `w` (converted to `k = w - 1`) or internal 0-indexed `k` |
| `I_w` | 0/1 | Query arm this week (1 = full CAE observed, 0 = short CAE observed) |
| `J_w` | 0/1 or null | Week-present gate; if `!= 1`, no CAE outcome is used this week |
| `Y_prev` / `Y_w` | float or null | Full weekly CAE from the finalized week (used when `I_w = 1`) |
| `tY_prev` / `tilde_Y_w` | float or null | Short CAE from the finalized week (used when `I_w = 0`) |
| `action_ids` *(optional)* | list[int] | Action ids of the completed week to receive the belief reward; if omitted, the handler matches actions whose internal week equals `k - 1` |
| `ew_inputs` *(optional)* | dict | E_w inputs (see above) |
| `pf_data` | dict | Particle-filter design matrices (see below) |

### `pf_data` fields

Let `K14 = 14` (slots/week), `N_MED_SLOT = 12` (Mon–Sat × 2 slots), and
`N_MED_ANTIC_DAY = 6` (Mon–Sat). `sim_w_prev = k - 1` is the just-finalized week.
The two mediator models are index 0 = "fourSC" (per-slot step-count next-4h) and
index 1 = "antic" (per-day anticipated affect). Cumulative arrays cover weeks
`0 .. sim_w_prev - 1` and are `null` until `k >= 2`.

Current-week design (always present):

| Field | Shape | Content |
|-------|-------|---------|
| `X_MY_base` | `[ (12, p_fourSC), (6, p_antic) ]` | Mediator design rows for the current week with the CAE column zeroed |
| `cae_delta_MY` | same shapes | Per-row deltas marking where the per-particle CAE is substituted |
| `M_Y_obs` | `[ (12,), (6,) ]` | Observed mediators (`stepCountNext4Hour`, `dailyAnticipatedAffect`); `NaN` where unobserved |
| `X_Y_base` | `(24,)` | CAE transition design for the current week (CAE AR slot zeroed) |
| `cae_delta_Y` | `(24,)` | Zeros except the AR-1 CAE column = 1 |
| `X_tY_base` | `(2,)` | Short-CAE design, `[1, 0]` |
| `cae_delta_tY` | `(2,)` | Short-CAE CAE slot, `[0, 1]` |
| `cae_all_0` | float | Baseline CAE (AR seed) at index 0 |
| `sim_w_prev` | int | `k - 1` |

Cumulative history (present when `k >= 2`, else `null`):

| Field | Shape | Content |
|-------|-------|---------|
| `X_cumul_MY_base` | `[ (n0, p_fourSC), (n1, p_antic) ]` | Mediator rows across past weeks (complete cases) |
| `cae_delta_cumul_MY` | same shapes | CAE deltas for the cumulative rows |
| `week_idx_cumul_MY` | `[ (n0,), (n1,) ]` | Week index of each cumulative mediator row (for per-particle CAE lookup) |
| `y_cumul_MY` | `[ (n0,), (n1,) ]` | Observed mediator responses (finite only) |
| `X_cumul_Y_base` | `(sim_w_prev, 24)` | CAE transition design for weeks `0 .. sim_w_prev - 1` |
| `cae_delta_cumul_Y` | `(sim_w_prev, 24)` | AR-1 CAE column = 1 per week |
| `y_cumul_Y` | `(sim_w_prev,)` | CAE trajectory targets (filled per-particle at update time) |
| `X_cumul_tY` | `(n_jw, 2)` | Short-CAE design for weeks where `J_w = 1` |
| `cae_delta_cumul_tY` | `(n_jw, 2)` | Short-CAE CAE slot = 1 |
| `week_idx_cumul_tY` | `(n_jw,)` | Week index of each short-CAE row |
| `y_cumul_tY` | `(n_jw,)` | Observed short-CAE outcomes |

The reference construction is `build_pf_data(...)` in the top-level
`algorithm_helpers.py` (simulation code); the deployment build should reproduce
the same shapes and column conventions from the participant's logged mediators,
actions, CAE outcomes, and week indicators. `prepare_for_update` also needs the
current `model_params` in `db_reader.get_retrieved_data()` under one of
`model_params`, `current_model_params`, or `rl_model_params` to run the PF update.

## DBReader Contract

The deployment framework should provide an object matching the `DBReader`
methods declared in `base.py`:

```python
class DBReader:
    def get_actions(self, action_type=None, limit=None, before=None):
        ...

    def get_observations(self, observation_type=None, since=None, before=None, limit=None):
        ...

    def get_stored_value(self, type):
        ...

    def get_retrieved_data(self):
        ...
```

The handlers only depend on this interface. They do not directly connect to a
database.

## Development Checks

Run a syntax/import smoke check from the parent directory:

```bash
cd ..
python -m compileall RL_agents
python -c "from RL_agents.rl import RL; from RL_agents.query import Query; print('ok')"
```

There is no test suite in this package yet. For deployment changes, add focused
tests around:

- state coercion and week-index conversion
- deterministic action seeding
- query-week calculation
- reward extraction from observations
- RLSVI update behavior when historical states are missing or complete
