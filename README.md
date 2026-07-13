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
    |-- prior_helpers.py     # Prior loading helpers
    |-- state_helpers.py     # State coercion, validation, serialization
    |-- standardization.py   # Optional context standardization
    |-- config.json          # RL policy settings
    |-- rl_priors.json       # Particle-filter prior parameters
    |-- rl_q_priors.json     # RLSVI Q-function prior parameters
    `-- std_params.json      # Context standardization parameters
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
   - consumes the latest particle-filter packet when available
   - returns rewards, stored belief values, and transformed observations

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
