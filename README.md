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
|-- seed_salt.py                # Secret randomization salt (RL and query)
|-- logging_config.py           # JSON structured logging helpers
|-- query/
|   |-- Query.py                # Weekly query decision handler
|   `-- config.json             # Query policy settings
`-- rl/
    |-- RL.py                   # Walking-suggestion RLSVI handler
    |-- algorithm_helpers.py    # Feature construction and RLSVI math
    |-- week_data.py            # Observations/actions -> per-week arrays (+ C imputation)
    |-- pf.py                   # Particle filter (port of the testbed PF)
    |-- ew_helpers.py           # Perceived-utility (E_w) pooled-linear update
    |-- prior_helpers.py        # Prior loading helpers
    |-- other_helpers.py        # State, indexing, serialization, slot lookup
    |-- standardization.py      # Observation → model-scale transforms
    |-- config.json             # RL policy settings
    |-- rl_priors.json          # PF priors (`pf`); its legacy 79-d Q block is not used
    |-- std_params.json         # Standardization parameters
    `-- Ew_pooled_linear_coefs.json  # e_hat weights (testbed format)
|-- tests/
    |-- harness.py              # in-memory framework used by the tests
    `-- test_rl_v6.py           # unit, parity, and end-to-end tests
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

Import path: `from RL_agents.rl import RL`.

The handler implements the validated simulation arm
**`rl_v6_invariant_redistributed`**: RLSVI with learned within-week reward
redistribution of the potential-shaped weekly target.

- `action = 1` send a walking suggestion, `0` do not.
- Study week `w = 1`: fixed randomization with probability 0.5.
- Later weeks: share of the B = 50 ensemble members preferring action 1,
  clipped to `[epsilon_0, 1 - epsilon_0]`.
- Unavailable decision (`availability` false): `A = 0`, `action_prob = 0.0`,
  `random_state.unavailable = true` (monitoring must not treat this as an
  out-of-range probability).

### Config (`rl/config.json`)

| Key | Value | Role |
|-----|-------|------|
| `W` | 36 | Study weeks |
| `n_rl_days` / `n_rl_slots` | 6 / 2 | Decision grid |
| `B` | 50 | Ensemble size |
| `epsilon_0` | 0.1 | Probability clip |
| `gamma_bar` | 0.9 | Week-boundary discount (1 within the week) and AR(1) exploration persistence |
| `target_update_C` | 1 | Target-network period (weeks) |
| `p_rl` | 27 | Feature dimension (checked at start-up) |
| `q_prior_mode` | `default` | Q prior N(0, I). `file` loads a prior whose `feature_names` must equal `phi_action_feature_names()` |
| `redistribution_prior_mode` | `default` | Stage-1/2 priors N(0, I), noise variance 1 |
| `sigma2_rl` | 1.0 | Initial TD noise variance (empirical-Bayes refit each week) |
| `std_params_path` | `std_params.json` | Standardization artifact (copied from the testbed) |
| `ew_coefs_path` | `Ew_pooled_linear_coefs.json` | e_hat coefficients (copied from the testbed) |
| `rl_prior_path` | `rl_priors.json` | PF working-model priors (`pf` block) |
| `J_particles` | 200 | Particle count (the testbed simulations used 50) |
| `pf_refit_sigma2` | `true` | Empirical-Bayes refit of the PF noise variances each week (as in the testbed) |
| `require_seed_salt` | `true` | Refuse to start without a randomization salt |
| `seed_salt_path` | `seed_salt.local` | Salt file (git-ignored), used when `ADAPT_SEED_SALT` is not set |

**Randomization salt** (`RL_agents/seed_salt.py`, shared with the query
handler). Each Bernoulli draw uses a seed hashed from
`[salt, user_id, w, d, t]`, so allocations cannot be predicted from participant
IDs. Lookup order: environment variable `ADAPT_SEED_SALT` (use this on the
server), then `seed_salt.local`, then config `seed` (legacy). At least 16
characters; a missing salt stops start-up. The salt is never logged or stored;
its fingerprint is stored in `model_params["seed_salt_fingerprint"]` and in
every `random_state`. If the running salt differs from the one a participant
was enrolled with, decisions continue and `RL_SEED_SALT_CHANGED` (red) is
logged. A salt was generated in `RL_agents/seed_salt.local`: copy it to the
server's `ADAPT_SEED_SALT`, keep a secure backup, and never change it during
the study.

### Feature map (27-d, identical to the testbed)

```
phi = [1, d_n, t_n, E_w, b_hat, b_tilde]                 # 6
    + [EWMA of AA, SC, PV, FW, PJ visible before (d,t)]  # 5
    + C (5-d)                                            # 5
    + A * [1, E_w, b_hat, b_tilde, C, d_n, t_n]           # 11
```

`d_n` = Saturday indicator, `t_n` = afternoon indicator. The EWMA weights sit on
calendar positions; missing measurements are skipped, never zero-filled.
`C` (5-d) is taken from the server's 6-entry context (all but the 7-day EMA,
which only the PF fourSC model uses).

### Weekly schedule

| When | Call | What it uses / does |
|------|------|---------------------|
| Monday AM / PM decisions of week k | `make_state` / `get_action` | belief, e_hat and policy of week k-1 (expected; not flagged as stale) |
| Monday night, after Monday's two decisions | `prepare_for_update`, then `update` | PF and e_hat for week k (from the Sunday survey), then RLSVI + V6 refit |
| Tuesday–Saturday decisions | `make_state` / `get_action` | belief, e_hat and policy of week k |

Both calls read `retrieved_data["model_params"]` and
`retrieved_data["update_week"]` (1-indexed current week), or `update_time` +
`timezone`. `make_state` records the expected input week
(`inputs.expected_input_week` = k-1 on Monday, k otherwise) and the tag of the
values it used; staleness is measured against that expectation.

Difference from the testbed: the simulation runs the PF at the start of week k,
so its Monday decisions already see `b_hat_k`, `e_k`. In deployment Monday uses
the week k-1 values. Training states for Monday slots still use the week-k
values (the testbed state definition), so only the two Monday action
probabilities are affected.

1. `prepare_for_update(user_id, db_reader)`
   - runs e_hat, then the particle filter, **for every week not yet filtered, in
     order** (a missed or crashed week is caught up, never skipped);
   - e_hat: `e_k = c0 + c_E e_{k-1} + c_PV PV_{k-1} + c_FW FW_{k-1} + c_PJ PJ_{k-1} + c_J J_k + c_U J_k(U1+U2)/14`
     (Mon–Sat means of week k-1; check-in of the survey that opens week k);
   - PF: the testbed working models (fourSC 20-d, antic 15-d, CAE transition 4-d,
     short CAE 2-d) with the testbed `pf` priors, per-particle posteriors,
     multinomial resampling at ESS < J/2, empirical-Bayes noise refit;
   - stores the tagged values and `rl_pf_state` (particles, weights, variances,
     RNG state, report).
2. `update(user_id, model_params, actions, rewards, db_reader)`
   - applies the prepared state only if its week, base PF state and version match;
   - fits Stage 1 (daily AA/FW/PJ), Stage 1b (slot step count) and Stage 2
     (weekly target `b_hat_{k+1} + gamma_bar e_{k+1} - e_k` on summed
     redistribution features incl. the realized page view);
   - builds TD rows from slot rewards `chi' eta`, refits sigma2 (empirical
     Bayes on the ensemble-mean target) and draws the ensemble;
   - **validates** the candidate (finite, not saturated, bounded change) and
     only then sets `active_policy_key = k`;
   - reruns for the same week return `already_updated_for_week` unchanged;
   - returns `UpdateResult.status` (also appended to `model_params["update_log"]`).

Training states are rebuilt from observations for every slot, so a slot with no
decision record (no suggestion delivered) enters as `A = 0` and the week keeps
its credit chain. Unavailable slots likewise enter as `A = 0`. Only weeks with no
data at all are excluded. Missing realized page views in the Stage-2 features
are set to 0 (model-scale centre) and counted in
`status["training"]["realized_pv_imputed_slots"]`.

### Data contract

- **Label = opening week.** The Sunday survey at the end of week w-1 carries
  `w`: `J_w`, `U1`, `U2`, `I_w`, `Y` / `Y_tilde`. It gates week-w mediators and
  is the CAE `Y_w` in the paper. `Y_1` (`rl_Y_1`, baseline) is required.
- Send `Y_tilde` in every completed survey, including full-CAE weeks
  (`I_w = 1`); the cumulative short-CAE model uses those rows.
- `I_w` / `J_w` not reported are treated as unknown, never as 1.
- Sunday is not collected, so the Monday-morning step-count lag in the fourSC
  model is 0 (as in the testbed).
- Observations are merged in `server_timestamp` order; exact duplicates (same
  `observation_id`) are ignored; for conflicting values the latest wins.
- **Decision context `C`: 6 entries, raw scale, in this order** (vectors of any
  other length are rejected and treated as missing):

  | Index | Field | Used by |
  |-------|-------|---------|
  | 0 | `yesterdayStepCount` | RLSVI, PF |
  | 1 | `stepCountLast7DaysEma` | PF only |
  | 2 | `prior2HourStepCount` | RLSVI, PF |
  | 3 | `activeDaysLast7Days` (fraction) | RLSVI, PF |
  | 4 | `activitySuggestionsSentLast7Days` (recent burden) | RLSVI, PF |
  | 5 | `activitySuggestionInteractLast7Days` (fraction) | RLSVI, PF |

**Missing context `C` at a decision** is imputed per entry, then flagged
(`RL_C_IMPUTED`, green):

| Entry | Rule |
|-------|------|
| `C[0]` yesterday steps | mean over the previous 7 days |
| `C[2]` prior-2h steps | mean of the same slot over the previous 7 days |
| `C[1]`, `C[3]`–`C[5]` (7-day summaries) | last observed value |

Only if an entry has no history does the decision fall back to the safe policy
(`RL_C_MISSING`).

### Failure containment

| Situation | Behaviour | Alert code |
|-----------|-----------|------------|
| Context `C` with a length other than 6 | treated as missing (then imputed) | `RL_C_BAD_LENGTH` (yellow) |
| Context `C` missing, imputable | RL decision on imputed C | `RL_C_IMPUTED` (green) |
| Context `C` missing, no history | safe policy pi = 0.5 | `RL_C_MISSING` (yellow) |
| Other non-finite feature | safe policy | `RL_FEATURES_NONFINITE` (red) |
| Belief / e_hat or policy one week older than expected | RL decision, flagged | `RL_STALE_ONE_WEEK` (yellow) |
| Belief / e_hat two or more weeks older than expected | safe policy | `RL_STALE_INPUTS` (red) |
| Validated policy two or more weeks older than expected | safe policy | `RL_STALE_POLICY` (red) |
| `model_params` from another package version | safe policy | `RL_VERSION_MISMATCH` (red) |
| Feature-dimension mismatch | safe policy | `RL_FEATURE_DIMENSION` (red) |
| Unexpected exception in `get_action` | safe policy | `RL_ACTION_UNEXPECTED_ERROR` (red) |
| Update cannot train / validation fails | last validated policy kept | `RL_UPDATE` status (yellow) |
| Update raises | params unchanged except status | `RL_UPDATE_FAILED` (red) |
| Conflicting action records for one slot | earliest kept | red log in `update` |
| `Y_1` missing at w = 1 | decision fails loudly | `RL_BASELINE_MISSING` (red) |
| prepare incomplete (PF/e_hat stopped) | stored report lists `pf_stop` | `RL_PREPARE` (yellow) |

Every `ActionResult.random_state` carries `seed`, `pi_hat` (unclipped),
`policy_key`, `policy_staleness`, input week tags, and `version_id`.

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

Test suite (unit, testbed parity, end-to-end through an in-memory framework):

```bash
cd ..
python -m unittest discover -s RL_agents/tests -v
```

The parity tests import the simulation testbed when `RL_agents` sits inside
`ADAPR-MRT-Testbed`, and check that the feature maps and the V6 TD design
reproduce the testbed exactly. Previously listed focused checks:

- thin context → `(w, d, t)` + `availability`; covariates from observations
- deterministic action seeding and unavailable → `A = 0`
- query-week calculation from enrollment + decision time
- PF / `update` ignore action-time `C` / mediators
- RLSVI when historical states or `E_w` / `b_hat` are missing
- empirical-Bayes `sigma2` fallback alert when no finite ensemble estimate
