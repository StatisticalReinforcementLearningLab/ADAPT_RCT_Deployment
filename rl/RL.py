"""Framework-facing walking-suggestion RL handler (validated arm V6).

Algorithm: RLSVI with the 27-d walking feature map, discount 1 within the week
and ``gamma_bar`` on the week boundary, and learned within-week reward
redistribution of the potential-shaped weekly target
``b_hat_{k} + gamma_bar * e_hat_{k} - e_hat_{k-1}`` (testbed arm
``rl_v6_invariant_redistributed``, no terminal residual).

Safety contract (see README "Failure containment"):

* Missing data are NaN; nothing is coerced to a valid-looking value.
* A decision that cannot be made from finite, fresh inputs and a validated
  policy uses the explicit safe policy (``pi = 0.5``) and logs a coded alert.
* A new policy is served only after it passes ``validate_policy``; the served
  ensemble is always ``betas_store[str(active_policy_key)]`` (or an earlier
  validated key). The week-0 prior is never re-served silently.
* Every update returns a status record (also appended to
  ``model_params["update_log"]``) and every artifact carries the package
  version hash.

Week indexing: ``w`` is the 1-indexed study week; ``k = w - 1``.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from RL_agents.base import (
    ActionHandlerBase,
    ActionRecord,
    ActionResult,
    DBReader,
    StateResult,
    StoredValue,
    UpdatePrepResult,
    UpdateResult,
)
from RL_agents.logging_config import GREEN, RED, YELLOW, get_logger, log_alert
from RL_agents.seed_salt import resolve_seed_salt
from RL_agents.rl.algorithm_helpers import (
    N_RL_DAYS,
    N_RL_SLOTS,
    P_RL,
    NonFiniteFeatures,
    betas_for_action,
    build_phi_action,
    build_v6_training_data,
    clip_prob,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
    ensemble_action_prob,
    gamma_dt_matrix,
    phi_action_feature_names,
    validate_policy,
)
from RL_agents.rl.ew_helpers import (
    compute_ew_for_week,
    ew_stored_value_type,
    load_ew_coefs,
)
from RL_agents.rl.other_helpers import (
    coerce_optional_float,
    decision_seed,
    deserialize_state,
    feature_state,
    history_arrays,
    internal_week_index,
    jsonable_float_array,
    jsonable_int_array,
    coerce_state,
    model_params_from_reader,
    none_to_nan_array,
    optional_state_index,
    participant_info,
    required_config,
    resolve_availability,
    resolve_decision_index,
    resolve_y1,
    serialize_state,
    stable_seed,
    state_index,
    stored_pf_state,
    stored_tagged,
    study_start_date,
    study_week_from_dates,
    tagged_value,
    validate_feature_dimension,
    context_decision_datetime,
    context_timezone,
)
from RL_agents.rl.pf import PFDesign, PFStateMismatch, load_pf_priors_v2, update_week
from RL_agents.rl.prior_helpers import load_q_prior, default_q_priors
from RL_agents.rl.algorithm_helpers import rl_context_from_server
from RL_agents.rl.week_data import WeekData, collect_week_data
from RL_agents.rl.standardization import (
    load_std_params,
    standardize_cae,
    uses_legacy_pageview_transform,
)

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ALGORITHM_ID = "rl_v6_invariant_redistributed"
UPDATE_LOG_MAX = 60
logger = get_logger(__name__)

# Alert codes (stable strings for the monitoring script).
CODE_SAFE_POLICY = "RL_SAFE_POLICY"
CODE_C_MISSING = "RL_C_MISSING"
CODE_NONFINITE_FEATURES = "RL_FEATURES_NONFINITE"
CODE_STALE_INPUTS = "RL_STALE_INPUTS"
CODE_STALE_POLICY = "RL_STALE_POLICY"
CODE_DIMENSION = "RL_FEATURE_DIMENSION"
CODE_VERSION = "RL_VERSION_MISMATCH"
CODE_UNEXPECTED = "RL_ACTION_UNEXPECTED_ERROR"
CODE_STALE_ONE_WEEK = "RL_STALE_ONE_WEEK"


def _resolve_local_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).parent / path
    return path


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RL config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(PACKAGE_ROOT)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def package_version(config: dict, artifact_paths: list[Path]) -> dict:
    code = sorted(PACKAGE_ROOT.rglob("*.py"))
    code = [p for p in code if "tests" not in p.parts and "__pycache__" not in p.parts]
    return {
        "algorithm": ALGORITHM_ID,
        "code_sha256": _sha256_files(code),
        # the salt never enters the version hash (it is secret and not in config)
        "config_sha256": hashlib.sha256(
            json.dumps({k: v for k, v in config.items() if k != "seed"}, sort_keys=True).encode()
        ).hexdigest(),
        "artifacts_sha256": _sha256_files([p for p in artifact_paths if p is not None and p.is_file()]),
    }


def _version_id(version: dict) -> str:
    return hashlib.sha256(json.dumps(version, sort_keys=True).encode()).hexdigest()[:16]


class RL(ActionHandlerBase):
    action_type = "walking_suggestion"

    def __init__(self, config: dict | None = None):
        self.config = load_config() if config is None else dict(config)
        self.n_rl_days = int(required_config(self.config, "n_rl_days"))
        self.n_rl_slots = int(required_config(self.config, "n_rl_slots"))
        if (self.n_rl_days, self.n_rl_slots) != (N_RL_DAYS, N_RL_SLOTS):
            raise ValueError("config n_rl_days/n_rl_slots differ from the package grid")
        if int(required_config(self.config, "p_rl")) != P_RL:
            raise ValueError(f"config p_rl must be {P_RL} for the V6 feature map")
        self._salt, self.salt_source, self.salt_fingerprint = resolve_seed_salt(self.config)
        if self._salt is None:
            log_alert(logger, RED, "RL randomization salt is unset; allocation is a public function of participant id",
                      action_type=self.action_type, code="RL_SEED_SALT_UNSET")
        elif self.salt_source == "config:seed":
            log_alert(logger, YELLOW, "RL randomization salt read from config.json; move it to ADAPT_SEED_SALT",
                      action_type=self.action_type, code="RL_SEED_SALT_IN_CONFIG")
        std_path = _resolve_local_path(required_config(self.config, "std_params_path"))
        self.std_params = load_std_params(std_path)
        if uses_legacy_pageview_transform(self.std_params):
            log_alert(
                logger,
                YELLOW,
                "std_params uses the legacy log(x+1) page-view transform; regenerate from the testbed",
                action_type=self.action_type,
            )
        self.rl_feature_names = phi_action_feature_names()

        q_mode = str(self.config.get("q_prior_mode", "default"))
        sigma2 = float(required_config(self.config, "sigma2_rl"))
        if q_mode == "default":
            mu_0, Sigma_0, q_sigma2, names, source = default_q_priors(self.rl_feature_names, sigma2=sigma2)
        elif q_mode == "file":
            loaded = load_q_prior(
                q_prior_path=_resolve_local_path(self.config.get("rl_q_prior_path")),
                combined_prior_path=_resolve_local_path(self.config.get("rl_prior_path")),
                expected_feature_names=self.rl_feature_names,
            )
            if loaded is None:
                raise FileNotFoundError("q_prior_mode='file' but no Q prior file was found")
            mu_0, Sigma_0, q_sigma2, names, source = loaded
        else:
            raise ValueError(f"unknown q_prior_mode {q_mode!r}")
        self.q_priors = {"mu_0": mu_0, "Sigma_0": Sigma_0, "sigma2": q_sigma2, "feature_names": names}
        self.q_prior_source = source

        if str(self.config.get("redistribution_prior_mode", "default")) != "default":
            raise ValueError("only redistribution_prior_mode='default' (zero/identity) is implemented")

        prior_path = _resolve_local_path(required_config(self.config, "rl_prior_path"))
        if prior_path is None or not prior_path.is_file():
            raise FileNotFoundError(f"PF priors not found in {prior_path}")
        with prior_path.open(encoding="utf-8") as f:
            self.pf_priors = load_pf_priors_v2(json.load(f))
        self.pf_prior_source = str(prior_path)
        ew_path = _resolve_local_path(required_config(self.config, "ew_coefs_path"))
        self.ew_coefs = load_ew_coefs(ew_path)
        self.version = package_version(self.config, [std_path, prior_path, ew_path])
        self.version_id = _version_id(self.version)
        log_alert(logger, GREEN, "RL handler initialized", action_type=self.action_type,
                  version_id=self.version_id, salt_source=self.salt_source,
                  salt_fingerprint=self.salt_fingerprint, **self.version)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _rng(self, model_params: dict) -> np.random.Generator:
        rng_state = model_params.get("rng_state")
        if rng_state is None:
            raise KeyError("model_params lacks rng_state; refusing to use an unseeded RNG")
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        return rng

    def _y1_model_scale(self, db_reader: DBReader | None, user_id: str) -> float:
        return float(standardize_cae(resolve_y1(db_reader, self.config, user_id), self.std_params))

    def _e_lag0(self) -> float:
        return float(self.ew_coefs.get("E_lag0", 2.0))

    def _safe_action(self, user_id, w, d, t, code: str, severity: str, **ctx) -> ActionResult:
        seed = decision_seed(self._salt, user_id, w, d, t, fallback=True)
        log_alert(
            logger,
            severity,
            "RL safe policy used (pi = 0.5)",
            action_type=self.action_type,
            user_id=str(user_id),
            code=code,
            w=w,
            d=d,
            t=t,
            version_id=self.version_id,
            **ctx,
        )
        rng = np.random.default_rng(seed)
        return ActionResult(
            action=int(rng.binomial(1, 0.5)),
            action_prob=0.5,
            random_state={
                "fallback": True,
                "code": code,
                "seed": seed,
                "salt_fingerprint": self.salt_fingerprint,
                "version_id": self.version_id,
            },
        )

    def resolve_update_week(self, db_reader: DBReader) -> int:
        """0-indexed week ``k`` of the Monday-night update (explicit contract).

        ``retrieved_data`` must carry ``update_week`` (1-indexed study week)
        or ``update_time`` (+ ``timezone``); the week is never inferred from
        whichever actions happen to exist.
        """
        data = db_reader.get_retrieved_data() or {}
        if data.get("update_week") is not None:
            return internal_week_index(int(data["update_week"]))
        dt = context_decision_datetime({"decision_time": data.get("update_time")}) if data.get("update_time") else None
        if dt is None:
            raise KeyError("retrieved_data must include update_week or update_time for RL updates")
        from zoneinfo import ZoneInfo

        tz = ZoneInfo(context_timezone(data))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tz)
        local = dt.astimezone(tz)
        start = study_start_date(participant_info(db_reader))
        return internal_week_index(study_week_from_dates(start, local.date()))

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def get_initial_model_params(self, user_id: str) -> dict:
        W = int(required_config(self.config, "W"))
        B = int(required_config(self.config, "B"))
        epsilon_0 = float(required_config(self.config, "epsilon_0"))
        gamma_bar = float(required_config(self.config, "gamma_bar"))
        target_update_C = int(required_config(self.config, "target_update_C"))
        J = int(required_config(self.config, "J_particles"))
        mu_0, Sigma_0 = self.q_priors["mu_0"], self.q_priors["Sigma_0"]
        initial_seed = stable_seed(json.dumps([self._salt, str(user_id), "init"]))
        rng = np.random.default_rng(initial_seed)
        L = np.linalg.cholesky(Sigma_0)
        z0 = [L @ rng.standard_normal(mu_0.size) for _ in range(B)]
        beta0 = [mu_0 + z for z in z0]
        e_hat_hist = np.full(W + 1, np.nan)
        e_hat_hist[0] = self._e_lag0()
        return {
            "version": 2,
            "algorithm": ALGORITHM_ID,
            "package_version_id": self.version_id,
            "package_version": self.version,
            "user_id": str(user_id),
            "initial_seed": initial_seed,
            "seed_salt_fingerprint": self.salt_fingerprint,
            "W": W,
            "B": B,
            "J": J,
            "n_rl_days": self.n_rl_days,
            "n_rl_slots": self.n_rl_slots,
            "p_rl": int(mu_0.size),
            "feature_names": list(self.q_priors["feature_names"]),
            "prior_source": self.q_prior_source,
            "epsilon_0": epsilon_0,
            "gamma_bar": gamma_bar,
            "gamma_dt": jsonable_float_array(gamma_dt_matrix(gamma_bar)),
            "target_update_C": target_update_C,
            "steps_since_target_update": 0,
            "mu_0_rl": jsonable_float_array(mu_0),
            "Sigma_0_rl": jsonable_float_array(Sigma_0),
            "sigma2_rl": float(self.q_priors["sigma2"]),
            "A_hist": jsonable_int_array(np.zeros((W, self.n_rl_days, self.n_rl_slots), dtype=int)),
            "pi_A_hist": jsonable_float_array(np.full((W, self.n_rl_days, self.n_rl_slots), np.nan)),
            "b_hat_hist": jsonable_float_array(np.full(W + 1, np.nan)),
            "b_tilde_hist": jsonable_float_array(np.full(W + 1, np.nan)),
            "e_hat_hist": jsonable_float_array(e_hat_hist),
            "pf_prior_source": self.pf_prior_source,
            "pf_y_hat": jsonable_float_array(np.full((J, 1), np.nan)),
            "pf_v_hat": jsonable_float_array(np.full(J, 1.0 / J)),
            "sigma2_MY": list(map(float, self.pf_priors["sigma2_MY"])),
            "sigma2_Y": float(self.pf_priors["sigma2_Y"]),
            "sigma2_tilde_Y": float(self.pf_priors["sigma2_tilde_Y"]),
            "betas_store": {"0": jsonable_float_array(np.asarray(beta0))},
            "z_store": {"0": jsonable_float_array(np.asarray(z0))},
            "betas_target": jsonable_float_array(np.asarray(beta0)),
            "active_policy_key": 0,
            "update_log": [],
            "rng_state": rng.bit_generator.state,
        }

    # ------------------------------------------------------------------
    # shared state construction (decision time and training use the same code)
    # ------------------------------------------------------------------
    def _week_data(self, db_reader, W: int) -> WeekData:
        return collect_week_data(db_reader, W=W, action_type=self.action_type)

    def _slot_state(self, data: WeekData, k: int, d: int, t: int, e_hat: float) -> tuple[dict, list[int]]:
        """Model-scale state for slot ``(k, d, t)`` and the imputed server-C indices."""
        c, imputed = data.model_c(k, d, t, self.std_params)
        M_Y, M_E = data.model_mediators(k, self.std_params)
        return {"E_w": float(e_hat), "M_Y": M_Y, "M_E": M_E, "C": rl_context_from_server(c)}, imputed

    def _week_has_data(self, data: WeekData, k: int) -> bool:
        if data.decided[k].any():
            return True
        arrays = (data.foursc[k], data.antic[k], data.pv[k], data.wear[k], data.pj[k], data.C[k])
        return any(np.isfinite(a).any() for a in arrays)

    # ------------------------------------------------------------------
    # decision time
    # ------------------------------------------------------------------
    @staticmethod
    def expected_input_week(k: int, d: int) -> int:
        """Week whose belief / e_hat a decision at ``(k, d)`` should use.

        ``prepare_for_update`` and ``update`` both run on Monday night, so
        Monday's two decisions use last week's values (``k - 1``) and
        Tuesday-Saturday use this week's (``k``), matching the policy pointer.
        """
        return int(k) - 1 if int(d) == 0 else int(k)

    def _tagged_input(self, db_reader, stored_type, k, fallback_week0):
        """Value for decisions expecting week-``k`` inputs, and its staleness."""
        if k <= 0:
            try:
                return float(fallback_week0()), 0, 0
            except KeyError as exc:
                log_alert(logger, RED, "baseline input missing at decision time",
                          action_type=self.action_type, code="RL_BASELINE_MISSING", error=str(exc))
                return None, 0, 0
        value, tag = stored_tagged(db_reader, stored_type)
        if value is None or tag is None:
            return None, None, None
        return value, max(0, k - tag), tag

    def make_state(self, user_id: str, context: dict, db_reader: DBReader) -> StateResult:
        try:
            w, d, t = resolve_decision_index(context or {}, db_reader)
            availability = resolve_availability(context or {})
            k = internal_week_index(w)
            W = int(required_config(self.config, "W"))
            k_in = self.expected_input_week(k, d)
            ew, ew_stale, ew_tag = self._tagged_input(db_reader, ew_stored_value_type(self.config), k_in, self._e_lag0)
            b_hat, b_stale, b_tag = self._tagged_input(
                db_reader, str(self.config["b_hat_stored_value_type"]), k_in,
                lambda: self._y1_model_scale(db_reader, user_id),
            )
            b_tilde, _, _ = self._tagged_input(db_reader, str(self.config["b_tilde_stored_value_type"]), k_in, lambda: 0.0)
            data = self._week_data(db_reader, W)
            fs, imputed = self._slot_state(data, k, d, t, np.nan if ew is None else ew)
            state = coerce_state({"w": w, "d": d, "t": t, **fs})
            state["availability"] = bool(availability)
            state["b_hat"], state["b_tilde"] = b_hat, b_tilde
        except (KeyError, TypeError, ValueError) as exc:
            log_alert(logger, YELLOW, "RL state preprocessing failed", action_type=self.action_type,
                      user_id=str(user_id), error=str(exc))
            raise
        out = serialize_state(state)
        c_missing = not bool(np.all(np.isfinite(state["C"])))
        out["inputs"] = {
            "expected_input_week": k_in,
            "e_hat_tag": ew_tag,
            "e_hat_staleness": ew_stale,
            "b_hat_tag": b_tag,
            "b_hat_staleness": b_stale,
            "C_imputed_indices": imputed,
            "C_missing": c_missing,
            "version_id": self.version_id,
        }
        bad = [b for b in data.bad_context if b[:3] == (k, d, t)]
        if bad:
            log_alert(logger, YELLOW, "decision context C has the wrong length; treated as missing",
                      action_type=self.action_type, user_id=str(user_id), code="RL_C_BAD_LENGTH",
                      w=w, d=d, t=t, length=bad[-1][3])
        if imputed:
            log_alert(logger, GREEN, "decision context partially imputed from the previous 7 days",
                      action_type=self.action_type, user_id=str(user_id), code="RL_C_IMPUTED",
                      w=w, d=d, t=t, indices=imputed)
        if c_missing:
            log_alert(logger, YELLOW, "decision context C unavailable even after imputation",
                      action_type=self.action_type, user_id=str(user_id), code=CODE_C_MISSING, w=w, d=d, t=t)
        return StateResult(state=out)

    def get_action(self, user_id: str, state: dict, model_params: dict) -> ActionResult:
        w = d = t = None
        try:
            inputs = dict((state or {}).get("inputs") or {})
            st = deserialize_state(state)
            w, d, t = state_index(st)
            k = internal_week_index(w)
            if not bool(st.get("availability", True)):
                return ActionResult(action=0, action_prob=0.0,
                                    random_state={"unavailable": True, "w": w, "d": d, "t": t,
                                                  "version_id": self.version_id})
            if model_params.get("seed_salt_fingerprint") != self.salt_fingerprint:
                # Randomization stays valid, but draws are no longer reproducible
                # from (salt, id, slot): alert, do not block delivery.
                log_alert(logger, RED, "randomization salt differs from the one used at enrollment",
                          action_type=self.action_type, user_id=str(user_id), code="RL_SEED_SALT_CHANGED",
                          w=w, d=d, t=t, enrolled=model_params.get("seed_salt_fingerprint"),
                          running=self.salt_fingerprint)
            if model_params.get("package_version_id") != self.version_id:
                return self._safe_action(user_id, w, d, t, CODE_VERSION, RED,
                                         params_version=model_params.get("package_version_id"))
            try:
                validate_feature_dimension(user_id=user_id, action_type=self.action_type,
                                           state=st, model_params=model_params)
            except ValueError as exc:
                return self._safe_action(user_id, w, d, t, CODE_DIMENSION, RED, error=str(exc))
            if w == 1:
                seed = decision_seed(self._salt, user_id, w, d, t)
                rng = np.random.default_rng(seed)
                return ActionResult(action=int(rng.binomial(1, 0.5)), action_prob=0.5,
                                    random_state={"initial_randomization": True, "seed": seed,
                                                  "salt_fingerprint": self.salt_fingerprint,
                                                  "inputs": inputs, "version_id": self.version_id})
            stale = [inputs.get("e_hat_staleness"), inputs.get("b_hat_staleness")]
            if any(s is None for s in stale) or max(stale) >= 2:
                return self._safe_action(user_id, w, d, t, CODE_STALE_INPUTS, RED, inputs=inputs)
            betas, key, policy_staleness = betas_for_action(model_params, k, d)
            if policy_staleness >= 2:
                return self._safe_action(user_id, w, d, t, CODE_STALE_POLICY, RED, policy_key=key)
            b_hat = coerce_optional_float(st.get("b_hat"))
            b_tilde = coerce_optional_float(st.get("b_tilde"))
            if b_hat is None or b_tilde is None:
                return self._safe_action(user_id, w, d, t, CODE_STALE_INPUTS, RED, inputs=inputs)
            fs = feature_state(st)
            phi_1 = build_phi_action(b_hat, b_tilde, fs, d, t, 1)
            phi_0 = build_phi_action(b_hat, b_tilde, fs, d, t, 0)
            try:
                pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
            except NonFiniteFeatures:
                code = CODE_C_MISSING if inputs.get("C_missing") else CODE_NONFINITE_FEATURES
                return self._safe_action(user_id, w, d, t, code, YELLOW if code == CODE_C_MISSING else RED)
            pi_A = clip_prob(pi_hat, float(model_params["epsilon_0"]))
            seed = decision_seed(self._salt, user_id, w, d, t)
            action = int(np.random.default_rng(seed).binomial(1, pi_A))
            if policy_staleness == 1 or max(stale) == 1:
                log_alert(logger, YELLOW, "RL decision used a policy or inputs one week old",
                          action_type=self.action_type, user_id=str(user_id), w=w, d=d, t=t,
                          code=CODE_STALE_ONE_WEEK, policy_key=key, inputs=inputs)
            return ActionResult(
                action=action,
                action_prob=float(pi_A),
                random_state={
                    "seed": seed,
                    "salt_fingerprint": self.salt_fingerprint,
                    "pi_hat": float(pi_hat),
                    "policy_key": int(key),
                    "policy_staleness": int(policy_staleness),
                    "inputs": inputs,
                    "version_id": self.version_id,
                },
            )
        except Exception as exc:  # noqa: BLE001 - decisions must never crash the server
            if w is None:
                w, d, t = optional_state_index(state)
            return self._safe_action(user_id, w, d, t, CODE_UNEXPECTED, RED,
                                     error_type=type(exc).__name__, error=str(exc))

    # ------------------------------------------------------------------
    # phase 1: beliefs and perceived utility (Monday night, just before update)
    # ------------------------------------------------------------------
    def prepare_for_update(self, user_id: str, db_reader: DBReader) -> UpdatePrepResult:
        if db_reader is None:
            raise ValueError("prepare_for_update requires a db_reader")
        params = model_params_from_reader(db_reader)
        if params is None:
            raise KeyError("retrieved_data must include the current model_params")
        k_target = self.resolve_update_week(db_reader)
        W = int(params["W"])
        if not (1 <= k_target <= W):
            raise ValueError(f"update week k={k_target} outside 1..{W}")
        report: dict[str, Any] = {"k_target": k_target, "version_id": self.version_id}
        work = deepcopy(params)
        y1 = self._seed_baseline(work, db_reader, user_id)
        data = self._week_data(db_reader, W)
        report["action_conflicts"] = data.conflicts
        report["bad_context_length"] = data.bad_context

        # perceived utility, in order, for every week not yet computed
        e_hat_hist = none_to_nan_array(work.get("e_hat_hist"), shape=(W + 1,))
        e_new: dict[int, float] = {}
        ew_info: dict[int, Any] = {}
        k0 = int(np.max(np.flatnonzero(np.isfinite(e_hat_hist[: k_target + 1])))) + 1
        for kk in range(k0, k_target + 1):  # bounded by W
            e, info = compute_ew_for_week(kk, e_lag=float(e_hat_hist[kk - 1]), data=data,
                                          coefs=self.ew_coefs, std=self.std_params)
            ew_info[kk] = info
            if e is None:
                break
            e_hat_hist[kk] = e
            e_new[kk] = e
        report["e_hat_info"] = ew_info

        # particle filter, in order, for every week not yet filtered
        rng = self._rng(work)
        J = int(work["J"])
        y_hat = none_to_nan_array(work["pf_y_hat"], shape=(J, len(work["pf_y_hat"][0])))
        v_hat = none_to_nan_array(work["pf_v_hat"], shape=(J,))
        sigma2 = {"sigma2_MY": work["sigma2_MY"], "sigma2_Y": work["sigma2_Y"],
                  "sigma2_tilde_Y": work["sigma2_tilde_Y"]}
        k_have = y_hat.shape[1] - 1
        design = PFDesign(data, self.std_params, e_hat_hist, y1)
        report["pf_start_week"] = k_have + 1
        report["pf_c_imputed_entries"] = design.c_imputed
        beliefs: dict[int, list[float]] = {}
        for kk in range(k_have + 1, k_target + 1):  # bounded by W
            if not np.isfinite(e_hat_hist[kk - 1]):
                report["pf_stop"] = {"k": kk, "reason": "e_hat_missing_for_previous_week"}
                break
            try:
                res = update_week(
                    k=kk, y_hat=y_hat, v_hat=v_hat, sigma2=sigma2, pf=design.build(kk),
                    outcome=design.outcome[kk], priors=self.pf_priors, rng=rng,
                    refit_sigma2=bool(self.config.get("pf_refit_sigma2", True)),
                )
            except (PFStateMismatch, FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
                report["pf_stop"] = {"k": kk, "reason": type(exc).__name__, "error": str(exc)}
                break
            y_hat, v_hat, sigma2 = res["y_hat"], res["v_hat"], res["sigma2"]
            beliefs[kk] = [res["b_hat"], res["b_tilde"]]
            report.setdefault("pf_weeks", {})[kk] = {k2: res[k2] for k2 in ("ess", "resampled", "observed")}

        stored: list[StoredValue] = []
        rewards: dict[int, float] = {}
        if beliefs:
            k_last = max(beliefs)
            b_hat, b_tilde = beliefs[k_last]
            stored += [
                StoredValue(user_id, str(self.config["b_hat_stored_value_type"]), tagged_value(k_last, b_hat)),
                StoredValue(user_id, str(self.config["b_tilde_stored_value_type"]), tagged_value(k_last, b_tilde)),
            ]
            if k_last == k_target:
                for rec in db_reader.get_actions(action_type=self.action_type) or []:
                    try:
                        wk = internal_week_index(state_index(deserialize_state(rec.state or rec.context or {}))[0])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if wk == k_target - 1:
                        rewards[int(rec.action_id)] = float(b_hat)
        if e_new:
            k_last_e = max(e_new)
            stored.append(StoredValue(user_id, ew_stored_value_type(self.config), tagged_value(k_last_e, e_new[k_last_e])))
        stored.append(
            StoredValue(
                user_id,
                str(self.config["pf_state_stored_value_type"]),
                {
                    "k": k_target,
                    "base_pf_weeks": k_have,
                    "beliefs": {str(kk): v for kk, v in beliefs.items()},
                    "e_hat": {str(kk): v for kk, v in e_new.items()},
                    "pf": {
                        "pf_y_hat": jsonable_float_array(y_hat),
                        "pf_v_hat": jsonable_float_array(v_hat),
                        "sigma2_MY": [float(x) for x in sigma2["sigma2_MY"]],
                        "sigma2_Y": float(sigma2["sigma2_Y"]),
                        "sigma2_tilde_Y": float(sigma2["sigma2_tilde_Y"]),
                    } if beliefs else None,
                    "rng_state": rng.bit_generator.state,
                    "version_id": self.version_id,
                    "report": report,
                },
            )
        )
        ok = bool(beliefs) and max(beliefs) == k_target and k_target in e_new
        log_alert(
            logger,
            GREEN if ok else YELLOW,
            "RL prepare_for_update complete" if ok else "RL prepare_for_update incomplete",
            action_type=self.action_type, user_id=str(user_id), code="RL_PREPARE", **report,
        )
        return UpdatePrepResult(rewards=rewards, stored_values=stored,
                                transformed_observations=[{"type": "rl_update_prep", **report}])

    def _seed_baseline(self, params: dict, db_reader: DBReader | None, user_id: str) -> float:
        """Seed ``b_hat_hist[0]`` / particles with ``Y_1`` (model scale); returns it."""
        W, J = int(params["W"]), int(params["J"])
        Y_1 = self._y1_model_scale(db_reader, user_id)
        b_hat_hist = none_to_nan_array(params.get("b_hat_hist"), shape=(W + 1,))
        b_tilde_hist = none_to_nan_array(params.get("b_tilde_hist"), shape=(W + 1,))
        if not np.isfinite(b_hat_hist[0]):
            b_hat_hist[0], b_tilde_hist[0] = Y_1, 0.0
            params["b_hat_hist"] = jsonable_float_array(b_hat_hist)
            params["b_tilde_hist"] = jsonable_float_array(b_tilde_hist)
        y_hat = none_to_nan_array(params["pf_y_hat"], shape=(J, len(params["pf_y_hat"][0])))
        if y_hat.shape[1] == 1 and not np.all(np.isfinite(y_hat)):
            params["pf_y_hat"] = jsonable_float_array(np.full((J, 1), Y_1))
            params["pf_v_hat"] = jsonable_float_array(np.full(J, 1.0 / J))
        return float(b_hat_hist[0])

    # ------------------------------------------------------------------
    # phase 2: RLSVI + V6 redistribution (Monday night)
    # ------------------------------------------------------------------
    def update(self, user_id: str, model_params: dict, actions: list[ActionRecord],
               rewards: dict[int, float], db_reader: DBReader) -> UpdateResult:
        params = deepcopy(model_params)
        status: dict[str, Any] = {"version_id": self.version_id, "algorithm": ALGORITHM_ID}
        try:
            return self._update(user_id, params, actions, db_reader, status)
        except Exception as exc:  # noqa: BLE001 - reported with full status; policy unchanged
            status.update(ok=False, stage=status.get("stage", "unknown"),
                          reason=type(exc).__name__, error=str(exc))
            log_alert(logger, RED, "RL update failed; last validated policy retained",
                      action_type=self.action_type, user_id=str(user_id), code="RL_UPDATE_FAILED", **status)
            safe = deepcopy(model_params)
            self._append_status(safe, status)
            return UpdateResult(model_params=safe, stored_values=None, status=status)

    def _append_status(self, params: dict, status: dict) -> None:
        log = list(params.get("update_log", []))
        log.append(status)
        params["update_log"] = log[-UPDATE_LOG_MAX:]
        params["last_update_status"] = status

    def _update(self, user_id, params, actions, db_reader, status) -> UpdateResult:
        status["stage"] = "setup"
        if params.get("package_version_id") != self.version_id:
            raise ValueError(
                f"model_params were created by package version {params.get('package_version_id')}, "
                f"running {self.version_id}; migrate explicitly"
            )
        W, B = int(params["W"]), int(params["B"])
        k_cur = self.resolve_update_week(db_reader)
        status["k_cur"] = k_cur
        self._seed_baseline(params, db_reader, user_id)
        A_hist, pi_A_hist, b_hat_hist, b_tilde_hist = history_arrays(
            params, n_rl_days=self.n_rl_days, n_rl_slots=self.n_rl_slots
        )
        e_hat_hist = none_to_nan_array(params.get("e_hat_hist"), shape=(W + 1,))

        status["stage"] = "data"
        data = self._week_data(db_reader, W)
        A_hist[:] = data.A.astype(int)
        for rec in actions:
            try:
                ww, dd, tt = state_index(deserialize_state(rec.state or rec.context or {}))
            except (KeyError, TypeError, ValueError):
                continue
            kk = internal_week_index(ww)
            if 0 <= kk < W and 0 <= dd < N_RL_DAYS and 0 <= tt < N_RL_SLOTS and np.isnan(pi_A_hist[kk, dd, tt]):
                pi_A_hist[kk, dd, tt] = float(rec.action_prob)
        status["duplicate_conflicts"] = data.conflicts
        status["bad_context_length"] = data.bad_context
        if data.conflicts:
            log_alert(logger, RED, "conflicting action records for the same decision slot",
                      action_type=self.action_type, user_id=str(user_id), slots=data.conflicts)

        status["stage"] = "apply_beliefs"
        prepared = stored_pf_state(self.config, db_reader)
        pf_cols = len(params["pf_y_hat"][0])
        if (
            prepared is None
            or int(prepared.get("k", -1)) != k_cur
            or int(prepared.get("base_pf_weeks", -2)) != pf_cols - 1
            or prepared.get("version_id") != self.version_id
        ):
            status["belief_applied"] = False
            status["belief_reason"] = "prepared_state_missing_or_mismatched"
        else:
            if prepared.get("pf"):
                params.update(prepared["pf"])
            params["rng_state"] = prepared["rng_state"]
            for kk, (bh, bt) in (prepared.get("beliefs") or {}).items():
                b_hat_hist[int(kk)], b_tilde_hist[int(kk)] = float(bh), float(bt)
            for kk, e in (prepared.get("e_hat") or {}).items():
                e_hat_hist[int(kk)] = float(e)
            status["belief_applied"] = True
            status["prepare_report"] = prepared.get("report")

        params["A_hist"] = jsonable_int_array(A_hist)
        params["pi_A_hist"] = jsonable_float_array(pi_A_hist)
        params["b_hat_hist"] = jsonable_float_array(b_hat_hist)
        params["b_tilde_hist"] = jsonable_float_array(b_tilde_hist)
        params["e_hat_hist"] = jsonable_float_array(e_hat_hist)

        def finish(ok: bool, reason: str, **extra) -> UpdateResult:
            status.update(ok=ok, reason=reason, active_policy_key=int(params.get("active_policy_key", 0)), **extra)
            self._append_status(params, status)
            log_alert(logger, GREEN if ok else YELLOW,
                      "RL update complete" if ok else "RL update skipped; last validated policy retained",
                      action_type=self.action_type, user_id=str(user_id), code="RL_UPDATE", **status)
            return UpdateResult(model_params=params, stored_values=None, status=status)

        if k_cur < 1:
            return finish(True, "baseline_week")
        if int(params.get("active_policy_key", 0)) >= k_cur:
            return finish(True, "already_updated_for_week")
        if not (np.isfinite(b_hat_hist[k_cur]) and np.isfinite(e_hat_hist[k_cur])):
            return finish(False, "current_week_belief_or_e_hat_missing")

        status["stage"] = "build_training_data"
        rng = self._rng(params)
        active = int(params.get("active_policy_key", 0))
        stores = {str(k): [np.asarray(b, dtype=float) for b in v] for k, v in params["betas_store"].items()}
        z_store = {str(k): [np.asarray(z, dtype=float) for z in v] for k, v in params["z_store"].items()}
        betas_eval = stores[str(active)]
        betas_target = [np.asarray(b, dtype=float) for b in params["betas_target"]]
        z_prev = z_store[str(active)]
        mu_0 = np.asarray(params["mu_0_rl"], dtype=float)
        Sigma_0 = np.asarray(params["Sigma_0_rl"], dtype=float)
        gamma_bar = float(params["gamma_bar"])
        gamma_dt = np.asarray(params["gamma_dt"], dtype=float)

        # Training states are rebuilt from observations for every slot, so a
        # slot without a decision record (no suggestion delivered, A = 0) still
        # carries the week's credit chain. Only weeks with no data are unusable.
        week_ok = {k: self._week_has_data(data, k) for k in range(min(k_cur + 1, W))}
        cache: dict[tuple, dict] = {}
        counts = {"c_imputed": 0, "undecided_slots": int((~data.decided[:k_cur]).sum())}

        def has_state(k, d, t):
            return bool(week_ok.get(k, False))

        def get_state(k, d, t):
            if (k, d, t) not in cache:
                st, imp = self._slot_state(data, k, d, t, e_hat_hist[k])
                counts["c_imputed"] += len(imp)
                cache[(k, d, t)] = st
            return cache[(k, d, t)]

        pv_imputed = {"n": 0}
        full_cache: dict[int, Any] = {}

        def get_full(k):
            if k not in full_cache:
                if not week_ok.get(k, False):
                    full_cache[k] = None
                else:
                    M_Y, M_E = data.model_mediators(k, self.std_params)
                    miss = ~np.isfinite(M_E[:, 0:2])
                    pv_imputed["n"] += int(miss.sum())
                    pv = M_E[:, 0:2]
                    pv[miss] = 0.0  # unrecorded realized page view -> model-scale centre
                    M_E[:, 0:2] = pv
                    full_cache[k] = (M_Y, M_E)
            return full_cache[k]

        Phi, targets, info = build_v6_training_data(
            k_cur, A_hist, b_hat_hist, b_tilde_hist, e_hat_hist, betas_eval, betas_target,
            gamma_dt, gamma_bar, get_state, has_state, get_full,
        )
        status["n_rows"] = int(Phi.shape[0])
        status["training"] = {k: v for k, v in info.items() if k != "stage2_eta"}
        status["training"].update(
            realized_pv_imputed_slots=pv_imputed["n"],
            c_imputed_entries=counts["c_imputed"],
            undecided_slots=counts["undecided_slots"],
        )
        if Phi.shape[0] == 0:
            return finish(False, "no_usable_training_rows")

        status["stage"] = "posterior"
        sigma2, eb_info = empirical_bayes_sigma2_ensemble(Phi, targets, mu_0, Sigma_0, float(params["sigma2_rl"]))
        status["sigma2"] = eb_info
        betas_new, z_new, post = compute_rlsvi_betas(Phi, targets, mu_0, Sigma_0, sigma2, gamma_bar, z_prev, rng)
        status["posterior"] = post
        if post.get("inverse", {}).get("path") == "pinv":
            return finish(False, "posterior_precision_singular")

        status["stage"] = "validate"
        probes = []
        for d in range(N_RL_DAYS):
            for t in range(N_RL_SLOTS):
                fs = get_state(k_cur - 1, d, t)
                p1 = build_phi_action(b_hat_hist[k_cur], b_tilde_hist[k_cur], fs, d, t, 1)
                p0 = build_phi_action(b_hat_hist[k_cur], b_tilde_hist[k_cur], fs, d, t, 0)
                if np.all(np.isfinite(p1)) and np.all(np.isfinite(p0)):
                    probes.append((p1, p0))
        ok, report = validate_policy(betas_new, probes, epsilon_0=float(params["epsilon_0"]), previous_betas=betas_eval)
        status["validation"] = report
        if not ok:
            return finish(False, f"validation_failed:{report.get('reason')}")

        status["stage"] = "commit"
        stores[str(k_cur)] = betas_new
        z_store[str(k_cur)] = z_new
        steps = int(params.get("steps_since_target_update", 0)) + 1
        if steps >= int(params["target_update_C"]):
            params["betas_target"] = jsonable_float_array(np.asarray(betas_new))
            steps = 0
        params["steps_since_target_update"] = steps
        params["sigma2_rl"] = float(sigma2)
        params["betas_store"] = {k: jsonable_float_array(np.asarray(v)) for k, v in stores.items()}
        params["z_store"] = {k: jsonable_float_array(np.asarray(v)) for k, v in z_store.items()}
        params["rng_state"] = rng.bit_generator.state
        params["active_policy_key"] = int(k_cur)
        params["B"] = B
        return finish(True, "policy_updated")


RLAgent = RL
