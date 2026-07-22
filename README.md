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
|-- base.py                     # Shared dataclasses and ActionHandlerBase API
|-- logging_config.py           # JSON structured logging helpers
|-- query/
|   |-- Query.py                # Weekly query decision handler
|   `-- config.json             # Query policy settings
`-- rl/
    |-- RL.py                   # Walking-suggestion RLSVI handler
    |-- algorithm_helpers.py    # Feature construction and RLSVI math
    |-- pf_packet_builder.py    # Build PF packets from db_reader series
    |-- pf_helpers.py           # Particle-filter belief updates
    |-- ew_helpers.py           # Perceived-utility (E_w) pooled-linear update
    |-- prior_helpers.py        # Prior loading helpers
    |-- other_helpers.py        # State, indexing, serialization, slot lookup
    |-- standardization.py      # Observation → model-scale transforms
    |-- config.json             # RL policy settings
    |-- rl_priors.json          # PF + base Q priors (`pf`, `q_no_td_modify`)
    |-- std_params.json         # Standardization parameters
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

## Handler API

Both handlers implement `ActionHandlerBase` from `base.py`.

```python
handler.get_initial_model_params(
    user_id: str,
    db_reader: DBReader | None = None,
) -> dict
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

rl_params = rl_handler.get_initial_model_params(
    user_id="participant-001",
    db_reader=db_reader,  # optional; used to read baseline Y_1
)
query_params = query_handler.get_initial_model_params(user_id="participant-001")
```

Configure structured JSON logging once at process startup if the deployment
server wants log files from the handlers:

```python
from RL_agents.logging_config import configure_logging

configure_logging(log_dir="logs", log_file="rl_agents.log")
```

## Week and Slot Indexing

Query and RL share the same study-week convention at the API boundary:

| Symbol | Meaning |
|--------|---------|
| `w` | 1-indexed study week (`w = 1` is enrollment / baseline) |
| `k` | 0-indexed internal week used in arrays / PF packets (`k = w - 1`) |
| `d` | 0-indexed RL day within the week (Mon–Sat → `0…5`) |
| `t` | 0-indexed decision slot within the day (`0` morning, `1` afternoon) |

Aliases accepted for week: `w`, `rl_week`, `week`. Legacy serialized states that
only have 0-indexed `k` are converted via `w = k + 1` on deserialize.

## Weekly Query Handler

Import path:

```python
from RL_agents.query import Query
```

Default policy from `query/config.json`:

- force a query in study week `w = 2`
- randomize all other invoked weeks with probability `query_prob = 0.5`
- use deterministic per-user, per-week random seeds

If the query handler is invoked during `w = 1`, it logs a warning and applies
the configured Bernoulli rule.

### Query State Inputs

```python
context = {
    "decision_time": "2026-07-13T09:00:00-04:00",
    "timezone": "America/New_York",
}
```

`db_reader.get_retrieved_data()` should include participant metadata under one
of `participantinfo` / `participant_info` / `ParticipantInfo`, with an enrollment
date under `study_start_date`, `enroll_date`, `start_date`, `StudyStartDate`, or
`date_min`.

If no decision timestamp is provided, the context must already include a
1-indexed study week field: `w`, `rl_week`, or `week`.

### Query Action Output

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

At study week `w = 1`, the handler randomizes with probability `0.5`. After that,
it computes the ensemble probability that action `1` has greater value than
action `0`, then clips that probability to `[epsilon_0, 1 - epsilon_0]`.

If `availability` is false, `get_action` returns `A = 0` with probability `0`
(no randomization).

### Config (`rl/config.json`)

| Key | Default | Role |
|-----|---------|------|
| `W` | `36` | Study weeks |
| `n_rl_days` / `n_rl_slots` | `6` / `2` | Decision grid (must match feature layout) |
| `B` | `50` | RLSVI ensemble size |
| `epsilon_0` | `0.1` | Randomization clip |
| `gamma_bar` | `0.5` | Discount / noise persistence |
| `target_update_C` | `1` | Target-network update period |
| `p_rl` | `77` | Feature dimension (validated at runtime) |
| `standardize_context` | `true` | Model-scale raw observation fields |
| `use_default_priors` | `false` | Q uses N(0,I); PF keeps structure with zero means |
| `pf_refit_sigma2` | `false` | Empirical-Bayes PF noise refit |
| `rl_prior_path` | `rl_priors.json` | Single prior file (`pf` + `q_no_td_modify`) |
| `y_1_observation_type` | `rl_Y_1` | Baseline CAE observation type |
| Stored-value types | `rl_b_hat`, `rl_b_tilde`, `rl_pf_state`, `rl_E_w` | Beliefs / PF / E_w |

### RL State Inputs (`make_state`)

Request `context` is intentionally thin: decision identity + MRT eligibility.
Features and beliefs are **not** read from context.

```python
context = {
    "decision_time": "2026-07-13T09:00:00-04:00",
    "timezone": "America/New_York",
    "availability": True,
}

# Test / legacy fallback when no timestamp is provided:
# context = {"w": 2, "d": 0, "t": 0, "availability": True}
```

`(w, d, t)` resolution:

1. Preferred: `decision_time` + `timezone` + `participantinfo` enrollment date
   (Mon–Sat RL days; slot from wakeup+1h / wakeup+6h when wakeup is present,
   else local hour `< 14` → morning)
2. Fallback: explicit `w` / `d`/`day` / `t`/`slot`

Covariates / beliefs from `db_reader` via `lookup_decision_covariates`
(per-decision lookup — **not** the full-week PF series):

| Field | Source |
|-------|--------|
| `C` | observation / `retrieved_data` for the current `(w, d, t)` slot |
| `M_Y`, `M_E` | week-`w` observation matrices |
| `E_w` | stored value `rl_E_w` |
| `b_hat`, `b_tilde` | stored values `rl_b_hat` / `rl_b_tilde` |

Baseline week `w = 1`: missing `b_hat` → observed `Y_1`; missing `b_tilde` /
`E_w` → `0.0`. Missing slot `C` zeros out with a yellow alert.

When `standardize_context` is true, looked-up `C` / `M_Y` / `M_E` are
model-scaled in `apply_decision_covariates` (same transforms as the update
path).

Serialized state includes `w`, `d`, `t`, `availability`, `E_w`, `C`, `M_Y`,
`M_E`, `b_hat`, `b_tilde`, `p_rl`. Action-time state is kept for **action
selection replicability**; weekly PF / RLSVI training re-reads covariates from
observations.

### Q Features (`p_rl = 77`)

`build_phi_action` builds a 77-dimensional feature map. Week matrices are stored
as `(6, 3)` and `(6, 4)`, then **raveled and slot-masked** before they enter φ:

| Block | Dim | Notes |
|-------|-----|-------|
| State base | 10 | intercept, time, `E_w`, `b_hat`, `b_tilde` + interactions |
| `M_Y` | **18** | `(6,3)` → flatten; per day: fourSC AM/PM, anticipated affect |
| `M_E` | **24** | `(6,4)` → flatten; per day: pageview AM/PM, wear, daily survey |
| `C` | 8 | current-slot context (RLSVI length; PF packs optional 9th wear) |
| Action interactions | 17 | `A` × {1, `E_w`, `b_hat`, time, `C`, …} — **not** × mediators |
| **Total** | **77** | |

At decision `(d, t)`, `_mask_mediators_for_slot` zeros mediators not yet
observed (slot-level cols need `(day, slot) < (d, t)`; day-level cols need
`day < d`). So early-week decisions see mostly zero mediator features even
though the matrices are full-week shaped.

### Minimal Action-Time Example

```python
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

   - reads rewards from observations (`action_id` + `reward` / `CAE` / `cae`)
   - builds a PF packet from observation-sourced series
     (`build_pf_packet_from_reader`; optional pre-assembled `rl_pf_packet` /
     `pf_packet` with `pf_data` still accepted when standardization is off)
   - runs the particle-filter belief update; assigns belief mean `b_hat` as the
     weekly reward
   - builds / updates perceived utility `E_w` and stores `rl_E_w`
   - returns rewards, stored belief / PF / `E_w` values

2. `update(...)`

   - ingest actions into `A_hist` / `pi_A_hist` / `state_hist`
   - **rebuild** training-state `C` / `M_Y` / `M_E` from observations
     (`collect_pf_week_series` + `collect_ew_week_sources`), keeping
     action-time `E_w` / `b_hat` / `b_tilde`
   - apply prepared PF beliefs to `b_hat_hist` / `b_tilde_hist`
   - gate RLSVI on finite `b_hat_hist[k_cur]` and stored `E_w`
   - refit `sigma2` (empirical Bayes; yellow alert + prior fallback if needed)
     and RLSVI ensemble → `betas_store[str(k_cur)]`

Week indexing example: finishing action week `k = 2` (`w = 3`) →
`max_k_seen = 2`, `k_cur = 3`, PF packet belief week `k = 3`, new ensemble in
`betas_store["3"]`.

## Data Contract (PF packet + E_w)

`prepare_for_update` builds PF design matrices (`pf_data`) from general-name
series in `db_reader` via `pf_packet_builder.build_pf_packet_from_reader`.

### Perceived-utility (`E_w`) inputs

Weekly `E_w` uses pooled, no-intercept linear coefficients
(`Ew_pooled_linear_coefs.json`):

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

Built by `ew_helpers.build_ew_inputs_from_reader` from:

1. `M_E` summaries of week `w-1`
2. Weekly check-in `J_w`, `U1`/`U2` (`Exp-tool-1`/`Exp-tool-2`)

Pre-assembled `ew_inputs` is still accepted (`packet["ew_inputs"]`,
`payload["ew_inputs"]`, or observation type `rl_ew_inputs`).

| Field | Paper symbol | Source |
|-------|--------------|--------|
| `J_w` | \(J_w^{\mathrm{week}}\) | Weekly check-in completed (0/1) |
| `U1` / `U2` | \(U_{w,1}\), \(U_{w,2}\) | RCT 1–7 helpfulness / pleasantness |
| `half_J_tool8` | \(J_w(U_{w,1}+U_{w,2})/14\) | Derived when omitted |
| `PJ_sum` | \(\overline M_{w-1}^{E,PJ}\) | Mean of `M_E[:, 3]` over 7 days |
| `FW_sum` | \(\overline M_{w-1}^{E,FW}\) | Mean of `M_E[:, 2]` over 7 days |
| `PV_sum` | \(\overline M_{w-1}^{E,PV}\) | `nansum` of `M_E[:, 0:2]` over 14 slots / 14 |

`M_E` is `(≤7)×4` (cols: pageview AM/PM, wear, daily present). Mon–Sat-only
matrices are padded with zeros for Sunday. Result stored under `rl_E_w` for
`make_state`.

### PF / action-time inputs from `db_reader`

| General name | Source | Meaning |
|--------------|--------|---------|
| `C` | observation / `retrieved_data` only (**not** action state) | Per-slot covariates; length 9 in PF (wear optional in `C[8]`); RLSVI uses first 8 |
| `A` | `ActionRecord.action` | Binary walking suggestion |
| `E_w` | stored `rl_E_w` (action state is soft fallback only) | Perceived utility |
| `M_Y` / `M^Y` | observation only | `(6,3)`: fourSC slots + anticipated affect |
| `M_E` / `M^E` | observation / week payload | `(≤7,4)` pageview / wear / daily present |
| `Exp-tool-1` / `Exp-tool-2` | weekly check-in | RCT tool ratings for `E_w` |
| `Y` / `CAE` | week payload | Full weekly CAE |
| `Y_tilde` / `CAE_short` | week payload | Short CAE |
| `I_w` | week payload | Query arm (1 = full CAE, 0 = short) |
| `J_w` / `wp` | week payload | Week-present gate |
| `wear7` *(optional)* | payload | `morningFitbitWearLast7Days` if not in `C[8]` |
| `antic_covariates` *(optional)* | week payload | Day-level antic covariates |
| `Y_1` | observation type `rl_Y_1` (or payload `Y_1`) | Baseline CAE |

How the server can send `C`:

1. Per-slot as available — `{w, d, t, C: [9]}`
2. Week grid — `{w, C: shape (6,2,9)}` or `(12,9)`
3. Prepare-time batch — `get_retrieved_data()["C"|"rl_C"|"pf_C"]`

Action-time `state["C"]` / mediators are for replicability only; PF and
`update` re-read observations.

When `standardize_context` is true, `prepare_for_update` rebuilds the PF packet
with `std_params` and model-scales raw observation fields **before** design
rows: `C` counts/burden, fourSC `log1p`+z, anticipated affect 1–7→`[-1,1]`,
CAE / short CAE / `Y_1` z-score, `todayStepCount` `log1p`+z.

#### `C` → fourSC column map

| `C[i]` | Role |
|--------|------|
| `C[0]` | yesterdayStepCount |
| `C[1]` | stepCountLast7DaysEma |
| `C[2]` | prior2HourStepCount |
| `C[3]` | activityCompletedLast7Days |
| `C[4]` | activeDaysLast7Days |
| `C[5]` | activitySuggestionsSentLast7Days |
| `C[6]` | salienceMessageSentYesterday |
| `C[7]` | activitySuggestionInteractLast7Days |
| `C[8]` | morningFitbitWearLast7Days when `wear7` omitted |

`dayOfWeekNorm` and `decisionTimeSlot` are derived from `(d, t)`.

### Built `pf_data` shapes

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

On build/update failure, the belief-fallback path runs (`b_hat_hist` left unset
for that week, with a yellow alert).

## DBReader Contract

```python
class DBReader:
    def get_actions(self, action_type=None, limit=None, before=None): ...
    def get_observations(self, observation_type=None, since=None, before=None, limit=None): ...
    def get_stored_value(self, type): ...
    def get_retrieved_data(self): ...
```

The handlers only depend on this interface. They do not directly connect to a
database.

## Development Checks

```bash
cd ..
python -m compileall RL_agents
python -c "from RL_agents.rl import RL; from RL_agents.query import Query; print('ok')"
```

There is no test suite in this package yet. Useful focused checks:

- thin context → `(w, d, t)` + `availability`; covariates from observations
- deterministic action seeding and unavailable → `A = 0`
- query-week calculation from enrollment + decision time
- PF / `update` ignore action-time `C` / mediators
- RLSVI when historical states or `E_w` / `b_hat` are missing
- empirical-Bayes `sigma2` fallback alert when no finite ensemble estimate
