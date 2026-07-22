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
    |-- other_helpers.py     # State coercion, validation, serialization
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

If no decision timestamp is provided, the context must already include a
1-indexed study week field: `w`, `rl_week`, or `week`.

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

At study week `w = 1` (enrollment/baseline), the handler randomizes with
probability `0.5`. After that, it computes the ensemble probability that
action `1` has greater value than action `0`, then clips that probability to
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

`RL.make_state(...)` uses a thin request context plus `db_reader` covariates /
beliefs (same observation-sourced contract as the PF packet and `update`).

Week indexing:

- Query and RL both use 1-indexed study week `w` at the API/state boundary
  (`w = 1` is enrollment/baseline)
- preferred: derive `(w, d, t)` from `decision_time` + `timezone` and
  `participantinfo` enrollment date (Mon–Sat RL days; slot from wakeup+1h /
  wakeup+6h when wakeup is present, else local hour `< 14` → morning)
- fallback for tests / legacy: explicit `w` (aliases `rl_week` / `week`),
  `d`/`day`, `t`/`slot`
- RL array math still uses local `k = w - 1` (e.g. `A_hist[k, d, t]`);
  PF packets retain that internal `k`

Required context fields:

- `decision_time` (or `decisionTime` / `timestamp`) and `timezone`
  **or** explicit `w`, `d`/`day`, `t`/`slot`
- `availability` (aliases `available` / `is_available`): when false,
  `get_action` returns `A=0` with probability 0

Feature / belief fields are **not** taken from context. `make_state` uses a
direct per-decision lookup (`lookup_decision_covariates`), not the full-week
PF series:

| Field | Source |
|-------|--------|
| `C` | observation / retrieved_data for the current `(w, d, t)` slot |
| `M_Y`, `M_E` | week-`w` observation matrices |
| `E_w` | stored value `rl_E_w` |
| `b_hat`, `b_tilde` | stored values `rl_b_hat` / `rl_b_tilde` |

At baseline week `w = 1`, missing `b_hat` falls back to observed `Y_1` and
missing `b_tilde` / `E_w` default to `0.0`. Missing slot `C` zeros out with a
yellow alert. When `standardize_context` is true, observation-sourced
`C` / `M_Y` / `M_E` are model-scaled in the same overlay path used by `update`.

### Minimal RL Context Example

```python
context = {
    "decision_time": "2026-07-13T09:00:00-04:00",
    "timezone": "America/New_York",
    "availability": True,
}

# Test / legacy fallback when no timestamp is provided:
# context = {"w": 2, "d": 0, "t": 0, "availability": True}

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
   - builds a PF packet from general-name series in `db_reader` (or uses a
     pre-assembled `rl_pf_packet` / `pf_packet` observation when it already
     includes `pf_data`), runs the particle-filter belief update, and assigns
     the belief mean `b_hat` as the weekly reward
   - builds weekly perceived-utility (`E_w`) inputs from `M_E` + weekly
     check-in (or uses pre-assembled `ew_inputs`), applies pooled linear
     coefficients, and stores the result under `rl_E_w`
   - returns rewards, stored belief/PF/`E_w` values, and transformed observations

2. `update(...)`

   - records observed actions into `A_hist`
   - records action probabilities into `pi_A_hist`
   - stores serialized historical states in `state_hist`
   - applies prepared particle-filter belief state when available
   - updates `b_hat_hist` and `b_tilde_hist`
   - refits the RLSVI ensemble when enough historical states and rewards exist
   - returns updated `model_params` and any stored values that should be saved

Pre-assembled PF packets (optional) may still use observation type
`rl_pf_packet` / `pf_packet` with external `w` or internal `k`
(`w` → `k = w - 1`).

## Data Contract (PF packet + E_w inputs)

`prepare_for_update` **builds** the PF design matrices (`pf_data`) from
general-name series in `db_reader` (actions + observations) via
`RL_agents.rl.pf_packet_builder.build_pf_packet_from_reader`. A pre-assembled
`rl_pf_packet` observation with `pf_data` is still accepted for backward
compatibility when present.

The server should provide the general-name fields below (per week / slot) so the
handler can construct the packet. This section specifies that contract.

### Perceived-utility (E_w) inputs

The weekly `E_w` update applies pooled, no-intercept linear coefficients
(`Ew_pooled_linear_coefs.json`) to the paper proxy

\[
\widehat e_w
=
\omega_1 J_w^{\mathrm{week}}
+
\omega_2 J_w^{\mathrm{week}}\frac{U_{w,1}+U_{w,2}}{14}
+
\omega_3 \overline M_{w-1}^{E,PJ}
+
\omega_4 \overline M_{w-1}^{E,PV}
+
\omega_5 \overline M_{w-1}^{E,FW}.
\]

`prepare_for_update` **builds** these aggregates via
`RL_agents.rl.ew_helpers.build_ew_inputs_from_reader` from:

1. **`M_E` summaries of week `w-1`** (action states and/or week observations)
2. **Weekly check-in** fields `J_w`, `U1`/`U2` (`Exp-tool-1`/`Exp-tool-2`)

A pre-assembled `ew_inputs` dict is still accepted when present
(`packet["ew_inputs"]`, `payload["ew_inputs"]`, or observation type
`rl_ew_inputs`).

| Field | Paper symbol | Source |
|-------|--------------|--------|
| `J_w` | \(J_w^{\mathrm{week}}\) | Weekly check-in completed (0/1) |
| `U1` / `U2` | \(U_{w,1}\), \(U_{w,2}\) | RCT 1–7 helpfulness / pleasantness (`Exp-tool-1` / `Exp-tool-2`); no shift |
| `half_J_tool8` | \(J_w(U_{w,1}+U_{w,2})/14\) | Derived from `J_w` and `U1`/`U2` when omitted (coef key name is historical) |
| `PJ_sum` | \(\overline M_{w-1}^{E,PJ}\) | Mean of `M_E[:, 3]` over 7 days (daily check-in surrogate) |
| `FW_sum` | \(\overline M_{w-1}^{E,FW}\) | Mean of `M_E[:, 2]` over 7 days (Fitbit wear) |
| `PV_sum` | \(\overline M_{w-1}^{E,PV}\) | `nansum` of slot pageviews `M_E[:, 0:2]` over 14 calendar slots, divided by 14 |

`M_E` is the RL `(≤7)×4` matrix: cols 0–1 = pageview slots, col 2 = wear, col 3 =
daily present. Mon–Sat-only matrices are padded with zeros for Sunday so the
denominators stay 7 / 14.

If inputs cannot be built, the E_w update is skipped. If aggregates are present
but invalid, a yellow "perceived-utility (E_w) update failed" alert is logged
and E_w is left unchanged. The resulting value is stored under `rl_E_w` and used
by `make_state` when the request context omits `E_w`/`perceivedUtility`.

### PF inputs from `db_reader` (general names)

The builder reads **actions** (`ActionRecord.state` / `action`) and optional weekly
observations (type `rl_pf_week` / `pf_week`, or any payload with a week index
`w`/`k` plus the fields below).

| General name | Source | Meaning |
|--------------|--------|---------|
| `C` | observation / retrieved_data only (not action state) | Per-slot baseline covariate vector (RLSVI length 9); server streams as available; action-time `state["C"]` is for replicability only |
| `A` | `ActionRecord.action` or week payload | Binary walking suggestion |
| `E_w` | action state / week payload / stored `rl_E_w` | Perceived utility for the week |
| `M_Y` / `M^Y` | observation only (not action state) | Cols 0–1 = fourSC slots; col 2 = antic day |
| `M_E` / `M^E` | week payload `(≤7,4)` | Cols 0–1 = pageview; col 2 = Fitbit wear; col 3 = daily present (also used for E_w) |
| `Exp-tool-1` / `Exp-tool-2` | weekly check-in payload | RCT 1–7 app helpfulness / pleasantness for E_w |
| `Y` / `CAE` | week payload | Full weekly CAE for the finalized week |
| `Y_tilde` / `tilde_Y` / `CAE_short` | week payload | Short CAE |

When `standardize_context` is true, `prepare_for_update` rebuilds the PF packet via
`build_pf_packet_from_reader(..., std_params=...)` and model-scales raw
observation fields **before** design rows: `C` counts/burden, fourSC
`log1p`+z, anticipated affect 1–7→[-1,1], CAE / short CAE / `Y_1` z-score,
`todayStepCount` `log1p`+z.
| `I_w` | week payload | Query arm (1 = full CAE, 0 = short CAE) |
| `J_w` / `wp` | week payload | Week-present gate |
| `wear7` *(optional)* | state or payload | `morningFitbitWearLast7Days` if not packed in `C[8]` |
| `antic_covariates` *(optional)* | week payload | Day-level antic covariates (`todayStepCount`, …) |

#### `C` → fourSC column map (current dims; may change later)

| `C[i]` | fourSC baseline role |
|--------|----------------------|
| `C[0]` | yesterdayStepCount |
| `C[1]` | stepCountLast7DaysEma |
| `C[2]` | prior2HourStepCount |
| `C[3]` | activityCompletedLast7Days |
| `C[4]` | activeDaysLast7Days |
| `C[5]` | activitySuggestionsSentLast7Days |
| `C[6]` | salienceMessageSentYesterday |
| `C[7]` | activitySuggestionInteractLast7Days |
| `C[8]` | morningFitbitWearLast7Days when `wear7` omitted |

`dayOfWeekNorm` and `decisionTimeSlot` are derived from `(d, t)`. Action
interactions and the CAE particle columns are filled by the row builders /
`cae_delta_*` (CAE base = 0; particles substitute at update time).

If a full server packet with `pf_data` is already in observations, it is used
as-is. Otherwise the handler builds one. On build/update failure, the existing
belief-fallback path runs (`b_hat_hist` left unset for that week).

### Built `pf_data` shapes (unchanged contract)

Let `N_MED_SLOT = 12` (Mon–Sat × 2), `N_MED_ANTIC_DAY = 6`, and CAE features pad
to a 7-day calendar (`14` fourSC + `7` antic). `sim_w_prev = k - 1`.
Mediator index 0 = fourSC (dim 25), index 1 = antic (dim 19). Cumulative arrays
cover weeks `0 .. sim_w_prev - 1` and are `null` until `k >= 2`.

| Field | Shape | Content |
|-------|-------|---------|
| `X_MY_base` | `[ (12, 25), (6, 19) ]` | Mediator design rows (CAE column zeroed) |
| `cae_delta_MY` | same shapes | Per-particle CAE substitution deltas |
| `M_Y_obs` | `[ (12,), (6,) ]` | Observed mediators; `NaN` where unobserved |
| `X_Y_base` | `(24,)` | CAE transition design (AR CAE slot zeroed) |
| `cae_delta_Y` | `(24,)` | AR-1 CAE column = 1 |
| `X_tY_base` / `cae_delta_tY` | `(2,)` | Short-CAE design `[1,0]` / `[0,1]` |
| `cae_all_0` | float | Baseline CAE (`Y_1`) |
| `sim_w_prev` | int | `k - 1` |
| cumulative `X_cumul_*` / `y_cumul_*` | (when `k >= 2`) | Complete-case history |

`prepare_for_update` also needs current `model_params` in
`db_reader.get_retrieved_data()` under one of `model_params`,
`current_model_params`, or `rl_model_params` to run the PF update.

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
