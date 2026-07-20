"""Framework-facing RL action handler.

#TODO:
1. Feature construction in make_state
2. pf_packet construction in prepare_for_update
3. how is delayed data handled in prepare_for_update or update

This is the base RL algorithm handler. It is used to train the RL model and to make actions.

Expected state/context fields
-----------------------------
``make_state`` accepts a context dict that already contains the feature-builder
state used by ``RL_agents.rl.algorithm_helpers.build_phi_action``:

    E_w, M_Y, M_E, C, w, d, t

Study week ``w`` is 1-indexed (``w=1`` is the first week of RL/active phase).
Array indexing inside this handler uses ``k = w - 1``.
If omitted,``make_state`` reads beliefs / ``E_w`` from ``DBReader.get_stored_value()``
using the stored value types in ``config.json``.
"""

from __future__ import annotations

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
from RL_agents.logging_config import RED, YELLOW, GREEN, get_logger, log_alert
from RL_agents.rl.algorithm_helpers import (
    build_phi_action,
    build_rl_training_data,
    clip_prob,
    compute_rlsvi_betas,
    empirical_bayes_sigma2_ensemble,
    ensemble_action_prob,
    gamma_dt_matrix,
    phi_action_feature_names,
)
from RL_agents.rl.ew_helpers import compute_ew_hat, load_ew_coefs
from RL_agents.rl.pf_helpers import (
    initial_pf_state,
    latest_pf_packet,
    resolve_pf_priors,
    update_pf_belief,
)
from RL_agents.rl.prior_helpers import (
    load_pf_priors,
    load_q_prior,
    resolve_q_priors,
)
from RL_agents.rl.standardization import load_std_params, standardize_context
from RL_agents.rl.other_helpers import (
    coerce_optional_float,
    coerce_state,
    decision_seed,
    deserialize_state,
    feature_state,
    internal_week_index,
    jsonable_float_array,
    jsonable_int_array,
    none_to_nan_array,
    normalize_study_week,
    optional_state_index,
    required_config,
    serialize_state,
    stable_seed,
    state_index,
    state_key,
    stored_float,
    to_float_array,
    to_int_array,
    validate_feature_dimension,
    y1_from_reader,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")
logger = get_logger(__name__)


# Stored value types
DEFAULT_B_HAT_STORED_VALUE_TYPE = "rl_b_hat"
DEFAULT_B_TILDE_STORED_VALUE_TYPE = "rl_b_tilde"
DEFAULT_PF_STATE_STORED_VALUE_TYPE = "rl_pf_state"
DEFAULT_PF_STATE_HISTORY_STORED_VALUE_TYPE_PREFIX = "rl_pf_state_week"
DEFAULT_EW_STORED_VALUE_TYPE = "rl_E_w"


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
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _log_invalid_binary_action(
    *,
    user_id: str,
    action_type: str,
    w: int,
    d: int,
    t: int,
    action: Any,
) -> None:
    if action in (0, 1):
        return
    log_alert(
        logger,
        RED,
        "invalid binary action",
        action_type=action_type,
        user_id=str(user_id),
        w=w,
        d=d,
        t=t,
        action=action,
    )


class RL(ActionHandlerBase):

    action_type = "walking_suggestion"

    def __init__(self, config: dict | None = None):
        self.config = load_config() if config is None else config
        self.n_rl_days = int(required_config(self.config, "n_rl_days"))
        self.n_rl_slots = int(required_config(self.config, "n_rl_slots"))
        std_path = _resolve_local_path(
            self.config.get("std_params_path", Path(__file__).with_name("std_params.json"))
        )
        self.std_params = (
            load_std_params(std_path)
            if bool(self.config.get("standardize_context", False))
            else None
        )
        self.rl_feature_names = phi_action_feature_names()
        self.use_default_priors = bool(self.config.get("use_default_priors", False))
        prior_path = _resolve_local_path(self.config.get("rl_prior_path"))
        pf_loaded = load_pf_priors(prior_path)
        self.pf_priors, self.pf_prior_source = resolve_pf_priors(
            pf_loaded,
            use_default_priors=self.use_default_priors,
        )
        q_loaded = load_q_prior(
            q_prior_path=_resolve_local_path(self.config.get("rl_q_prior_path")),
            combined_prior_path=prior_path,
            expected_feature_names=self.rl_feature_names,
        )
        (
            q_mu_0,
            q_Sigma_0,
            q_sigma2,
            q_feature_names,
            self.q_prior_source,
        ) = resolve_q_priors(
            q_loaded,
            config=self.config,
            feature_names=self.rl_feature_names,
            use_default_priors=self.use_default_priors,
        )
        self.q_priors = {
            "mu_0": q_mu_0,
            "Sigma_0": q_Sigma_0,
            "sigma2": q_sigma2,
            "feature_names": q_feature_names,
        }
        self.ew_coefs = load_ew_coefs(
            _resolve_local_path(
                self.config.get("ew_coefs_path", "Ew_pooled_linear_coefs.json")
            )
        )

    def _rng(self, model_params: dict) -> np.random.Generator:
        rng = np.random.default_rng()
        rng_state = model_params.get("rng_state")
        if rng_state is not None:
            rng.bit_generator.state = rng_state
        return rng

    def _initial_seed(self, user_id: str) -> int:
        seed = self.config.get("seed")
        return stable_seed(user_id) if seed is None else int(seed)

    def get_initial_model_params(
        self,
        user_id: str,
        db_reader: DBReader | None = None,
    ) -> dict:
        W = int(required_config(self.config, "W"))
        B = int(required_config(self.config, "B"))
        epsilon_0 = float(required_config(self.config, "epsilon_0"))
        gamma_bar = float(required_config(self.config, "gamma_bar"))
        target_update_C = int(required_config(self.config, "target_update_C"))
        J = int(self.config.get("J_particles", self.config.get("J", 50)))
        gamma_dt = to_float_array(
            self.config.get("gamma_dt"),
            default=gamma_dt_matrix(gamma_bar),
        )
        mu_0 = self.q_priors["mu_0"]
        Sigma_0 = self.q_priors["Sigma_0"]
        sigma2 = self.q_priors["sigma2"]
        feature_names = self.q_priors["feature_names"]
        prior_source = self.q_prior_source

        initial_seed = self._initial_seed(user_id)
        rng = np.random.default_rng(initial_seed)
        z0 = [
            rng.multivariate_normal(np.zeros(mu_0.size), Sigma_0)
            for _ in range(B)
        ]
        beta0 = [mu_0 + z for z in z0]

        A_hist = np.zeros((W, self.n_rl_days, self.n_rl_slots), dtype=int)
        pi_A_hist = np.full((W, self.n_rl_days, self.n_rl_slots), np.nan, dtype=float)
        b_hat_hist = np.full(W + 1, np.nan, dtype=float)
        b_tilde_hist = np.full(W + 1, np.nan, dtype=float)
        # Baseline (week 0) b_hat is hardcoded to 0 (no learned belief yet). The
        # per-participant baseline outcome Y_1 anchors the particle-filter CAE
        # trajectory, not b_hat. Y_1 is read from the ObservationRecord via
        # db_reader when available, falling back to config otherwise.
        y1_observed = y1_from_reader(db_reader, self.config, user_id)
        Y_1 = (
            float(y1_observed)
            if y1_observed is not None
            else float(self.config.get("Y_1", 0.0))
        )
        b_hat_hist[0] = 0.0
        b_tilde_hist[0] = 0.0
        pf_state = initial_pf_state(J=J, Y_1=Y_1)

        return {
            "version": 1,
            "user_id": str(user_id),
            "initial_seed": initial_seed,
            "W": W,
            "B": B,
            "J": J,
            "n_rl_days": self.n_rl_days,
            "n_rl_slots": self.n_rl_slots,
            "p_rl": int(mu_0.size),
            "feature_names": list(feature_names),
            "prior_source": prior_source,
            "epsilon_0": epsilon_0,
            "gamma_bar": gamma_bar,
            "gamma_dt": jsonable_float_array(gamma_dt),
            "target_update_C": target_update_C,
            "steps_since_target_update": 0,
            "mu_0_rl": jsonable_float_array(mu_0),
            "Sigma_0_rl": jsonable_float_array(Sigma_0),
            "sigma2_rl": sigma2,
            "A_hist": jsonable_int_array(A_hist),
            "pi_A_hist": jsonable_float_array(pi_A_hist),
            "b_hat_hist": jsonable_float_array(b_hat_hist),
            "b_tilde_hist": jsonable_float_array(b_tilde_hist),
            "pf_prior_source": self.pf_prior_source,
            "pf_y_hat": pf_state["y_hat"],
            "pf_v_hat": pf_state["v_hat"],
            "sigma2_MY": (
                list(map(float, self.pf_priors.get("sigma2_MY", [1.0, 1.0])))
                if self.pf_priors
                else [1.0, 1.0]
            ),
            "sigma2_Y": (
                float(self.pf_priors.get("sigma2_Y", 1.0)) if self.pf_priors else 1.0
            ),
            "sigma2_tilde_Y": (
                float(self.pf_priors.get("sigma2_tilde_Y", 1.0))
                if self.pf_priors
                else 1.0
            ),
            "betas_store": {"0": jsonable_float_array(np.asarray(beta0))},
            "z_store": {"0": jsonable_float_array(np.asarray(z0))},
            "betas_target": jsonable_float_array(np.asarray(beta0)),
            "state_hist": {},
            "rng_state": rng.bit_generator.state,
        }

    def make_state(
        self,
        user_id: str,
        context: dict,
        db_reader: DBReader,
    ) -> StateResult:
        try:
            # Normalize to 1-indexed study week w (same convention as Query).
            # Array indexing inside this handler uses k = w - 1.
            context_for_state = normalize_study_week(context)
            # Perceived utility E_w: if the request context omits it, fall back to
            # the value written weekly by prepare_for_update's E_w update.
            if (
                context_for_state.get("E_w") is None
                and context_for_state.get("perceivedUtility") is None
            ):
                ew_stored = stored_float(db_reader, self._ew_stored_value_type())
                if ew_stored is not None:
                    context_for_state = {**context_for_state, "E_w": ew_stored}
            if self.std_params is not None:
                context_for_state = standardize_context(
                    context_for_state, self.std_params
                )
            state = coerce_state(context_for_state)
            b_hat = coerce_optional_float(state.get("b_hat"))
            b_tilde = coerce_optional_float(state.get("b_tilde"))
            if b_hat is None:
                b_hat = stored_float(
                    db_reader,
                    str(
                        self.config.get(
                            "b_hat_stored_value_type",
                            DEFAULT_B_HAT_STORED_VALUE_TYPE,
                        )
                    ),
                )
            if b_tilde is None:
                b_tilde = stored_float(
                    db_reader,
                    str(
                        self.config.get(
                            "b_tilde_stored_value_type",
                            DEFAULT_B_TILDE_STORED_VALUE_TYPE,
                        )
                    ),
                )
            if b_hat is None or b_tilde is None:
                w_idx, _, _ = optional_state_index(state)
                if w_idx == 1:
                    # Baseline (enrollment) week: no learned belief exists yet, so
                    # b_hat is hardcoded to 0 (get_action randomizes at w=1 anyway).
                    # The per-participant baseline outcome Y_1 lives in the
                    # ParticipantInfo table and anchors the particle filter, not b_hat.
                    b_hat = 0.0 if b_hat is None else b_hat
                    b_tilde = 0.0 if b_tilde is None else b_tilde
                else:
                    raise KeyError(
                        "state must include b_hat/b_tilde or DBReader stored values "
                        "must provide them"
                    )
            state["b_hat"] = b_hat
            state["b_tilde"] = b_tilde
        except (KeyError, TypeError, ValueError) as exc:
            log_alert(
                logger,
                YELLOW,
                "RL state preprocessing failed",
                action_type=self.action_type,
                user_id=str(user_id),
                error=str(exc),
            )
            raise

        return StateResult(state=serialize_state(state))

    def _history_arrays(
        self, model_params: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        W = int(model_params["W"])
        n_days = int(model_params.get("n_rl_days", self.n_rl_days))
        n_slots = int(model_params.get("n_rl_slots", self.n_rl_slots))
        A_hist = to_int_array(
            model_params.get("A_hist"),
            default=np.zeros((W, n_days, n_slots), dtype=int),
        )
        pi_A_hist = none_to_nan_array(
            model_params.get("pi_A_hist"),
            shape=(W, n_days, n_slots),
        )
        b_hat_hist = none_to_nan_array(
            model_params.get("b_hat_hist"),
            shape=(W + 1,),
        )
        b_tilde_hist = none_to_nan_array(
            model_params.get("b_tilde_hist"),
            shape=(W + 1,),
        )
        return A_hist, pi_A_hist, b_hat_hist, b_tilde_hist

    def _belief_stored_values(
        self, user_id: str, b_hat: float, b_tilde: float
    ) -> list[StoredValue]:
        return [
            StoredValue(
                user_id=str(user_id),
                type=str(
                    self.config.get(
                        "b_hat_stored_value_type",
                        DEFAULT_B_HAT_STORED_VALUE_TYPE,
                    )
                ),
                value=float(b_hat),
            ),
            StoredValue(
                user_id=str(user_id),
                type=str(
                    self.config.get(
                        "b_tilde_stored_value_type",
                        DEFAULT_B_TILDE_STORED_VALUE_TYPE,
                    )
                ),
                value=float(b_tilde),
            ),
        ]

    def _pf_state_stored_values(
        self,
        user_id: str,
        *,
        packet: dict,
        pf_update: dict,
        b_hat: float,
        b_tilde: float,
        rng_state: dict,
    ) -> list[StoredValue]:
        k = int(packet["k"])
        value = {
            "k": k,
            "b_hat": float(b_hat),
            "b_tilde": float(b_tilde),
            "pf_update": pf_update,
            "rng_state": rng_state,
        }
        latest_type = str(
            self.config.get(
                "pf_state_stored_value_type",
                DEFAULT_PF_STATE_STORED_VALUE_TYPE,
            )
        )
        history_prefix = str(
            self.config.get(
                "pf_state_history_stored_value_type_prefix",
                DEFAULT_PF_STATE_HISTORY_STORED_VALUE_TYPE_PREFIX,
            )
        )
        return [
            StoredValue(
                user_id=str(user_id),
                type=latest_type,
                value=value,
            ),
            StoredValue(
                user_id=str(user_id),
                type=f"{history_prefix}_{k}",
                value=value,
            ),
        ]

    def _stored_pf_state(self, db_reader: DBReader) -> dict | None:
        if db_reader is None:
            return None
        stored = db_reader.get_stored_value(
            str(
                self.config.get(
                    "pf_state_stored_value_type",
                    DEFAULT_PF_STATE_STORED_VALUE_TYPE,
                )
            )
        )
        if stored is None or not isinstance(stored.value, dict):
            return None
        return stored.value

    def _stored_belief(self, db_reader: DBReader) -> tuple[float | None, float | None]:
        return (
            stored_float(
                db_reader,
                str(
                    self.config.get(
                        "b_hat_stored_value_type",
                        DEFAULT_B_HAT_STORED_VALUE_TYPE,
                    )
                ),
            ),
            stored_float(
                db_reader,
                str(
                    self.config.get(
                        "b_tilde_stored_value_type",
                        DEFAULT_B_TILDE_STORED_VALUE_TYPE,
                    )
                ),
            ),
        )

    def _ew_stored_value_type(self) -> str:
        return str(
            self.config.get("ew_stored_value_type", DEFAULT_EW_STORED_VALUE_TYPE)
        )

    def _ew_inputs(self, observations: list, packet: dict | None) -> dict | None:
        """Locate the weekly E_w predictor aggregates for the perceived-utility update.

        The server assembles these from the database for the just-finalized week.
        Accepted locations: ``packet["ew_inputs"]``, an observation payload with an
        ``ew_inputs`` dict, or an observation of type ``rl_ew_inputs``.
        """
        if packet is not None and isinstance(packet.get("ew_inputs"), dict):
            return packet["ew_inputs"]
        for obs in observations:
            payload = getattr(obs, "payload", {}) or {}
            if isinstance(payload.get("ew_inputs"), dict):
                return payload["ew_inputs"]
            if getattr(obs, "observation_type", None) == "rl_ew_inputs" and payload:
                return payload
        return None

    def _prepare_ew_update(
        self,
        user_id: str,
        observations: list,
        packet: dict | None,
    ) -> list[StoredValue] | None:
        """Weekly perceived-utility (E_w) update from pooled linear coefficients.

        Returns a stored value for the updated E_w (consumed by ``make_state`` for
        the next week's decisions), or ``None`` if coefficients or inputs are
        unavailable. Failures emit a yellow alert and return ``None``.
        """
        if self.ew_coefs is None:
            return None
        inputs = self._ew_inputs(observations, packet)
        if inputs is None:
            return None
        try:
            ew_hat = compute_ew_hat(self.ew_coefs, inputs)
        except Exception as exc:
            log_alert(
                logger,
                YELLOW,
                "perceived-utility (E_w) update failed",
                action_type=self.action_type,
                user_id=str(user_id),
                error=str(exc),
            )
            return None
        return [
            StoredValue(
                user_id=str(user_id),
                type=self._ew_stored_value_type(),
                value=float(ew_hat),
            )
        ]

    def _model_params_from_reader(self, db_reader: DBReader) -> dict | None:
        if db_reader is None:
            return None
        data = db_reader.get_retrieved_data() or {}
        for key in ("model_params", "current_model_params", "rl_model_params"):
            value = data.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _action_ids_for_pf_reward(
        self, db_reader: DBReader, packet: dict
    ) -> list[int]:
        if packet.get("action_ids") is not None:
            return [int(action_id) for action_id in packet["action_ids"]]
        if packet.get("action_id") is not None:
            return [int(packet["action_id"])]
        if db_reader is None or packet.get("k") is None:
            return []
        # PF packet k is the 0-indexed belief-week array index; rewards attach to
        # actions from the completed study week (internal index packet["k"] - 1).
        reward_k = int(packet["k"]) - 1
        out = []
        for action in db_reader.get_actions(action_type=self.action_type):
            try:
                state = deserialize_state(action.state or action.context or {})
                w, _, _ = state_index(state)
                k = internal_week_index(w)
            except (KeyError, TypeError, ValueError):
                continue
            if k == reward_k:
                out.append(int(action.action_id))
        return out

    def _infer_pf_packet_k(self, db_reader: DBReader) -> int | None:
        """Infer 0-indexed PF belief-week index from action states."""
        if db_reader is None:
            return None
        ks = []
        for action in db_reader.get_actions(action_type=self.action_type):
            try:
                state = deserialize_state(action.state or action.context or {})
                w, _, _ = state_index(state)
                ks.append(internal_week_index(w))
            except (KeyError, TypeError, ValueError):
                continue
        return max(ks) + 1 if ks else None

    def _prepare_belief_fallback(
        self,
        *,
        user_id: str,
        db_reader: DBReader,
        packet: dict | None,
        rewards: dict[int, float],
        reason: str,
        error: str | None = None,
    ) -> tuple[list[StoredValue] | None, list[dict] | None]:
        log_alert(
            logger,
            YELLOW,
            "PF prepare failed; using latest stored belief",
            action_type=self.action_type,
            user_id=str(user_id),
            reason=reason,
            error=error,
        )
        b_hat, b_tilde = self._stored_belief(db_reader)
        if b_hat is None or b_tilde is None:
            return None, [packet] if packet is not None else None

        action_ids = self._action_ids_for_pf_reward(db_reader, packet or {})
        for action_id in action_ids:
            rewards[int(action_id)] = float(b_hat)

        transformed = None
        if packet is not None:
            transformed = [
                {
                    **packet,
                    "pf_result": {
                        "fallback": True,
                        "reason": reason,
                        "b_hat": float(b_hat),
                        "b_tilde": float(b_tilde),
                        "reward_action_ids": action_ids,
                    },
                }
            ]
        return self._belief_stored_values(user_id, b_hat, b_tilde), transformed

    def _betas_for_action(
        self, model_params: dict, k: int, d: int
    ) -> list[np.ndarray]:
        stores = model_params.get("betas_store", {})
        if k <= 0:
            key = "0"
        elif d >= 1 and str(k) in stores:
            key = str(k)
        else:
            key = str(max(k - 1, 0))
        if key not in stores:
            key = "0"
        return [np.asarray(beta, dtype=float) for beta in stores[key]]

    def get_action(
        self,
        user_id: str,
        state: dict,
        model_params: dict,
    ) -> ActionResult:
        try:
            return self._get_action_impl(user_id, state, model_params)
        except Exception as exc:
            w, d, t = optional_state_index(state)
            fallback_seed = decision_seed(
                self.config.get("seed"),
                user_id,
                w,
                d,
                t,
                fallback=True,
            )
            log_alert(
                logger,
                YELLOW,
                "real-time action selection failed; using fallback randomization",
                action_type=self.action_type,
                user_id=str(user_id),
                w=w,
                d=d,
                t=t,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            rng = np.random.default_rng(fallback_seed)
            action = int(rng.binomial(1, 0.5))
            return ActionResult(
                action=action,
                action_prob=0.5,
                random_state={"fallback": True, "seed": fallback_seed},
            )

    def _get_action_impl(
        self,
        user_id: str,
        state: dict,
        model_params: dict,
    ) -> ActionResult:
        state = deserialize_state(state)
        w, d, t = state_index(state)
        k = internal_week_index(w)
        validate_feature_dimension(
            user_id=user_id,
            action_type=self.action_type,
            state=state,
            model_params=model_params,
        )
        if w == 1:
            action_seed = decision_seed(self.config.get("seed"), user_id, w, d, t)
            rng = np.random.default_rng(action_seed)
            action = int(rng.binomial(1, 0.5))
            action_prob = 0.5
            _log_invalid_binary_action(
                user_id=user_id,
                action_type=self.action_type,
                w=w,
                d=d,
                t=t,
                action=action,
            )
            return ActionResult(
                action=action,
                action_prob=action_prob,
                random_state={
                    "initial_randomization": True,
                    "seed": action_seed,
                },
            )

        b_hat = coerce_optional_float(state.get("b_hat"))
        if b_hat is None:
            raise KeyError("state must include finite b_hat")

        b_tilde = coerce_optional_float(state.get("b_tilde"))
        if b_tilde is None:
            raise KeyError("state must include finite b_tilde")

        betas = self._betas_for_action(model_params, k, d)
        phi_1 = build_phi_action(b_hat, b_tilde, feature_state(state), d, t, 1)
        phi_0 = build_phi_action(b_hat, b_tilde, feature_state(state), d, t, 0)
        pi_hat = ensemble_action_prob(phi_1, phi_0, betas)
        pi_A = clip_prob(pi_hat, float(model_params["epsilon_0"]))
        action_seed = decision_seed(self.config.get("seed"), user_id, w, d, t)
        rng = np.random.default_rng(action_seed)
        action = int(rng.binomial(1, pi_A))
        _log_invalid_binary_action(
            user_id=user_id,
            action_type=self.action_type,
            w=w,
            d=d,
            t=t,
            action=action,
        )
        return ActionResult(
            action=action,
            action_prob=float(pi_A),
            random_state={"seed": action_seed},
        )

    def prepare_for_update(
        self,
        user_id: str,
        db_reader: DBReader,
    ) -> UpdatePrepResult:
        rewards: dict[int, float] = {}
        if db_reader is None:
            return UpdatePrepResult(rewards=rewards)
        observations = db_reader.get_observations()
        for obs in observations:
            payload = obs.payload or {}
            action_id = payload.get("action_id")
            reward = payload.get("reward", payload.get("CAE", payload.get("cae")))
            if action_id is None or reward is None:
                continue
            rewards[int(action_id)] = float(reward)
        packet = latest_pf_packet(observations)
        transformed_observations = [packet] if packet is not None else None
        stored_values: list[StoredValue] | None = None
        if packet is not None and self.pf_priors:
            try:
                if packet.get("k") is None:
                    inferred_k = self._infer_pf_packet_k(db_reader)
                    if inferred_k is not None:
                        packet = {**packet, "k": inferred_k}
                        transformed_observations = [packet]
                model_params = self._model_params_from_reader(db_reader)
                if model_params is not None:
                    if packet.get("k") is None:
                        raise KeyError("PF packet must include k or infer k from actions")
                    params = deepcopy(model_params)
                    rng = self._rng(params)
                    pf_update, b_hat_pf, b_tilde_pf = update_pf_belief(
                        model_params=params,
                        packet=packet,
                        pf_priors=self.pf_priors,
                        rng=rng,
                        refit_sigma2=bool(
                            self.config.get("pf_refit_sigma2", True)
                        ),
                    )
                    if pf_update.get("sigma2_eb_error"):
                        log_alert(
                            logger,
                            YELLOW,
                            "online noise-variance update failed",
                            action_type=self.action_type,
                            user_id=str(user_id),
                            k=int(packet["k"]),
                            error=pf_update["sigma2_eb_error"],
                        )
                    pf_update["rng_state"] = rng.bit_generator.state
                    action_ids = self._action_ids_for_pf_reward(db_reader, packet)
                    for action_id in action_ids:
                        rewards[int(action_id)] = float(b_hat_pf)
                    stored_values = [
                        *self._belief_stored_values(user_id, b_hat_pf, b_tilde_pf),
                        *self._pf_state_stored_values(
                            user_id,
                            packet=packet,
                            pf_update=pf_update,
                            b_hat=b_hat_pf,
                            b_tilde=b_tilde_pf,
                            rng_state=rng.bit_generator.state,
                        ),
                    ]
                    transformed_observations = [
                        {
                            **packet,
                            "pf_result": {
                                "b_hat": float(b_hat_pf),
                                "b_tilde": float(b_tilde_pf),
                                "reward_action_ids": action_ids,
                            },
                        }
                    ]
                else:
                    stored_values, transformed_observations = self._prepare_belief_fallback(
                        user_id=user_id,
                        db_reader=db_reader,
                        packet=packet,
                        rewards=rewards,
                        reason="model_params_unavailable",
                    )
            except Exception as exc:
                stored_values, transformed_observations = self._prepare_belief_fallback(
                    user_id=user_id,
                    db_reader=db_reader,
                    packet=packet,
                    rewards=rewards,
                    reason="pf_update_failed",
                    error=str(exc),
                )

        # Weekly perceived-utility (E_w) update; independent of the PF/belief path.
        ew_stored = self._prepare_ew_update(user_id, observations, packet)
        if ew_stored:
            stored_values = [*(stored_values or []), *ew_stored]

        return UpdatePrepResult(
            rewards=rewards,
            stored_values=stored_values,
            transformed_observations=transformed_observations,
        )

    def _state_lookup(self, state_hist: dict[str, dict]):
        def get_state(k: int, d: int, t: int) -> dict:
            key = state_key(k, d, t)
            if key not in state_hist:
                raise KeyError(f"missing RL state for {key}")
            return feature_state(deserialize_state(state_hist[key]))

        return get_state

    def update(
        self,
        user_id: str,
        model_params: dict,
        actions: list[ActionRecord],
        rewards: dict[int, float],
        db_reader: DBReader,
    ) -> UpdateResult:
        params = deepcopy(model_params)
        W = int(params["W"])
        B = int(params["B"])
        A_hist, pi_A_hist, b_hat_hist, b_tilde_hist = self._history_arrays(params)
        state_hist = dict(params.get("state_hist", {}))
        # Last observed belief; used as the fallback uncertainty when PF is
        # unavailable so b_tilde is carried forward rather than forced to 0.
        latest_b_hat, latest_b_tilde = self._stored_belief(db_reader)

        max_k_seen = -1
        for record in actions:
            state = deserialize_state(record.state or record.context or {})
            w, d, t = state_index(state)
            k = internal_week_index(w)
            validate_feature_dimension(
                user_id=user_id,
                action_type=self.action_type,
                state=state,
                model_params=params,
            )
            if not (0 <= d < self.n_rl_days and 0 <= t < self.n_rl_slots):
                log_alert(
                    logger,
                    RED,
                    "action day/slot index out of range",
                    action_type=self.action_type,
                    user_id=str(user_id),
                    w=w,
                    d=d,
                    t=t,
                    n_rl_days=self.n_rl_days,
                    n_rl_slots=self.n_rl_slots,
                )
                continue
            if not (0 <= k < W):
                continue
            max_k_seen = max(max_k_seen, k)
            A_hist[k, d, t] = int(record.action)
            pi_A_hist[k, d, t] = float(record.action_prob)
            state_hist[state_key(k, d, t)] = serialize_state(state)

            reward = rewards.get(record.action_id)
            if reward is not None and k + 1 < b_hat_hist.size:
                b_hat_hist[k + 1] = float(reward)
                # Without PF, carry the last observed belief uncertainty forward.
                # PF (if prepared) overrides b_tilde_hist below.
                b_tilde_hist[k + 1] = (
                    float(latest_b_tilde) if latest_b_tilde is not None else 0.0
                )

        params["A_hist"] = jsonable_int_array(A_hist)
        params["pi_A_hist"] = jsonable_float_array(pi_A_hist)
        params["b_hat_hist"] = jsonable_float_array(b_hat_hist)
        params["b_tilde_hist"] = jsonable_float_array(b_tilde_hist)
        params["state_hist"] = state_hist

        if max_k_seen < 0:
            return UpdateResult(model_params=params)

        k_cur = min(max_k_seen + 1, W - 1)
        rng = self._rng(params)
        prepared_pf_state = self._stored_pf_state(db_reader)
        if prepared_pf_state is not None:
            try:
                pf_update = dict(prepared_pf_state.get("pf_update", {}))
                params.update(pf_update)
                if prepared_pf_state.get("rng_state") is not None:
                    params["rng_state"] = prepared_pf_state["rng_state"]
                k_pf = int(prepared_pf_state["k"])
                if 0 <= k_pf < b_hat_hist.size:
                    b_hat_hist[k_pf] = float(prepared_pf_state["b_hat"])
                    b_tilde_hist[k_pf] = float(prepared_pf_state["b_tilde"])
                params["b_hat_hist"] = jsonable_float_array(b_hat_hist)
                params["b_tilde_hist"] = jsonable_float_array(b_tilde_hist)
                rng = self._rng(params)
            except Exception as exc:
                log_alert(
                    logger,
                    YELLOW,
                    "prepared PF state could not be applied",
                    action_type=self.action_type,
                    user_id=str(user_id),
                    k=k_cur,
                    error=str(exc),
                )
        else:
            log_alert(
                logger,
                YELLOW,
                "prepared PF state missing during update; using prepared rewards or latest stored belief",
                action_type=self.action_type,
                user_id=str(user_id),
                k=k_cur,
            )
            if latest_b_hat is not None and not np.isfinite(b_hat_hist[k_cur]):
                b_hat_hist[k_cur] = float(latest_b_hat)
            if latest_b_tilde is not None:
                b_tilde_hist[k_cur] = float(latest_b_tilde)
            params["b_hat_hist"] = jsonable_float_array(b_hat_hist)
            params["b_tilde_hist"] = jsonable_float_array(b_tilde_hist)
        stored_values = None
        if np.isfinite(b_hat_hist[k_cur]) and np.isfinite(b_tilde_hist[k_cur]):
            stored_values = [
                *self._belief_stored_values(user_id, b_hat_hist[k_cur], b_tilde_hist[k_cur])
            ]
        if k_cur == 0:
            return UpdateResult(model_params=params, stored_values=stored_values)
        if not np.isfinite(b_hat_hist[k_cur]):
            return UpdateResult(model_params=params, stored_values=stored_values)

        mu_0 = np.asarray(params["mu_0_rl"], dtype=float)
        Sigma_0 = np.asarray(params["Sigma_0_rl"], dtype=float)
        sigma2 = float(params["sigma2_rl"])
        gamma_dt = np.asarray(params["gamma_dt"], dtype=float)
        gamma_bar = float(params["gamma_bar"])

        betas_store = {
            str(k): [np.asarray(beta, dtype=float) for beta in betas]
            for k, betas in params.get("betas_store", {}).items()
        }
        z_store = {
            str(k): [np.asarray(z, dtype=float) for z in zs]
            for k, zs in params.get("z_store", {}).items()
        }
        if "0" not in betas_store or "0" not in z_store:
            return UpdateResult(model_params=params, stored_values=stored_values)

        eval_key = str(max(k_cur - 1, 0))
        betas_eval = betas_store.get(eval_key, betas_store["0"])
        betas_target = [
            np.asarray(beta, dtype=float)
            for beta in params.get("betas_target", betas_store["0"])
        ]
        z_prev = z_store.get(eval_key, z_store["0"])

        try:
            Phi_rl, targets_rl = build_rl_training_data(
                k_cur,
                A_hist,
                b_hat_hist,
                b_tilde_hist,
                betas_eval,
                betas_target,
                gamma_dt,
                self._state_lookup(state_hist),
            )
        except KeyError as exc:
            log_alert(
                logger,
                YELLOW,
                "RLSVI update skipped: missing historical RL states",
                action_type=self.action_type,
                user_id=str(user_id),
                k=k_cur,
                error=str(exc),
            )
            return UpdateResult(model_params=params, stored_values=stored_values)

        if Phi_rl.shape[0] == 0:
            return UpdateResult(model_params=params, stored_values=stored_values)

        try:
            sigma2 = empirical_bayes_sigma2_ensemble(
                Phi_rl, targets_rl, mu_0, Sigma_0, sigma2
            )
            beta_new, z_new = compute_rlsvi_betas(
                Phi_rl, targets_rl, mu_0, Sigma_0, sigma2, gamma_bar, z_prev, rng
            )
        except Exception as exc:
            log_alert(
                logger,
                YELLOW,
                "RLSVI update failed",
                action_type=self.action_type,
                user_id=str(user_id),
                k=k_cur,
                error=str(exc),
            )
            return UpdateResult(model_params=params, stored_values=stored_values)

        betas_store[str(k_cur)] = beta_new
        z_store[str(k_cur)] = z_new
        steps = int(params.get("steps_since_target_update", 0)) + 1
        if steps >= int(params["target_update_C"]):
            params["betas_target"] = jsonable_float_array(np.asarray(beta_new))
            steps = 0
        params["steps_since_target_update"] = steps
        params["sigma2_rl"] = float(sigma2)
        params["betas_store"] = {
            k: jsonable_float_array(np.asarray(v)) for k, v in betas_store.items()
        }
        params["z_store"] = {
            k: jsonable_float_array(np.asarray(v)) for k, v in z_store.items()
        }
        params["rng_state"] = rng.bit_generator.state

        # Keep ensemble count honest if callers changed B in config/model params.
        params["B"] = int(B)
        return UpdateResult(model_params=params, stored_values=stored_values)


RLAgent = RL
