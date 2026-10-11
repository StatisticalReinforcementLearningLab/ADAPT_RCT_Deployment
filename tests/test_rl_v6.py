"""Deployment tests for the V6 RL handler.

Run from the directory containing ``RL_agents``::

    python -m unittest discover -s RL_agents/tests -v

The parity tests additionally import the simulation testbed
(``ADAPR-MRT-Testbed/algorithm_helpers.py``) when it is next to ``RL_agents``;
they are skipped otherwise.
"""

from __future__ import annotations

import copy
import logging
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logging.getLogger("RL_agents").setLevel(logging.CRITICAL)

from harness import Participant  # noqa: E402

from RL_agents.rl import RL  # noqa: E402
from RL_agents.rl import algorithm_helpers as ah  # noqa: E402

TESTBED = HERE.parent.parent
_HANDLER = None


def handler():
    global _HANDLER
    if _HANDLER is None:
        _HANDLER = RL()
    return _HANDLER


def rand_state(rng, *, nan_frac=0.0):
    M_Y = rng.normal(size=(6, 3))
    M_E = rng.normal(size=(6, 4))
    if nan_frac:
        M_Y[rng.random(M_Y.shape) < nan_frac] = np.nan
        M_E[rng.random(M_E.shape) < nan_frac] = np.nan
    return {"E_w": float(rng.normal()), "M_Y": M_Y, "M_E": M_E, "C": rng.normal(size=5)}


# ---------------------------------------------------------------------------
# Unit tests of the numerical core
# ---------------------------------------------------------------------------
class TestCore(unittest.TestCase):
    def test_feature_dimension_and_names(self):
        st = rand_state(np.random.default_rng(0))
        self.assertEqual(ah.build_phi_action(0.1, 0.2, st, 2, 1, 1).size, 27)
        self.assertEqual(len(ah.phi_action_feature_names()), 27)
        self.assertTrue(np.all(ah.build_phi_action(0.1, 0.2, st, 2, 1, 0)[16:] == 0))

    def test_discount_only_on_week_boundary(self):
        g = ah.gamma_dt_matrix(0.9)
        self.assertEqual(g[5, 1], 0.9)
        self.assertTrue(np.all(np.delete(g.ravel(), 11) == 1.0))

    def test_ewma_skips_nan_keeps_calendar(self):
        self.assertAlmostEqual(ah.ewma_gamma([np.nan, 2.0]), 2.0)
        self.assertEqual(ah.ewma_gamma([np.nan, np.nan]), 0.0)
        v = ah.ewma_gamma([1.0, np.nan, 3.0])
        g = 2 / 3
        self.assertAlmostEqual(v, (g**2 * 1 + 3) / (g**2 + 1))

    def test_ridge_matches_closed_form(self):
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(20, 3)), rng.normal(size=20)
        mu, S, s2 = rng.normal(size=3), np.diag([1.0, 2.0, 0.5]), 0.7
        expect = np.linalg.inv(np.linalg.inv(S) + X.T @ X / s2) @ (np.linalg.inv(S) @ mu + X.T @ y / s2)
        np.testing.assert_allclose(ah.ridge_posterior_mean(X, y, mu, S, s2), expect, rtol=1e-10)

    def test_rlsvi_zero_rows_returns_prior_and_rejects_nan(self):
        rng = np.random.default_rng(2)
        p, B = 27, 4
        z0 = [np.zeros(p)] * B
        betas, _, diag = ah.compute_rlsvi_betas(np.empty((0, p)), [np.empty(0)] * B, np.zeros(p), np.eye(p), 1.0, 0.9, z0, rng)
        self.assertEqual(diag["inverse"]["path"], "prior_only")
        Phi = rng.normal(size=(5, p))
        Phi[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ah.compute_rlsvi_betas(Phi, [np.zeros(5)] * B, np.zeros(p), np.eye(p), 1.0, 0.9, z0, rng)

    def test_nonfinite_features_raise_at_decision(self):
        st = rand_state(np.random.default_rng(3))
        st["C"][0] = np.nan
        p1 = ah.build_phi_action(0.1, 0.1, st, 1, 0, 1)
        p0 = ah.build_phi_action(0.1, 0.1, st, 1, 0, 0)
        with self.assertRaises(ah.NonFiniteFeatures):
            ah.ensemble_action_prob(p1, p0, [np.ones(27)])

    def test_policy_pointer_never_serves_week0_silently(self):
        params = {"betas_store": {"0": [[0.0] * 27], "3": [[0.0] * 27]}, "active_policy_key": 3}
        _, key, stale = ah.betas_for_action(params, 5, 2)
        self.assertEqual((key, stale), (3, 2))
        params["active_policy_key"] = 4
        with self.assertRaises(KeyError):
            ah.betas_for_action(params, 5, 2)

    def test_validate_policy(self):
        rng = np.random.default_rng(4)
        st = rand_state(rng)
        probes = [(ah.build_phi_action(0, 0, st, d, 0, 1), ah.build_phi_action(0, 0, st, d, 0, 0)) for d in range(6)]
        ok, _ = ah.validate_policy([np.full(27, np.nan)], probes, epsilon_0=0.1)
        self.assertFalse(ok)
        big = np.zeros(27)
        big[16] = 100.0  # A intercept: every member always sends
        ok, rep = ah.validate_policy([big] * 5, probes, epsilon_0=0.1)
        self.assertFalse(ok)
        self.assertEqual(rep["reason"], "policy_saturated")


# ---------------------------------------------------------------------------
# Parity with the validated simulation code
# ---------------------------------------------------------------------------
def _load_testbed():
    if not (TESTBED / "algorithm_helpers.py").is_file():
        return None
    sys.path.insert(0, str(TESTBED))
    try:
        import importlib

        return importlib.import_module("algorithm_helpers")
    except Exception:  # noqa: BLE001
        return None


class TestParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = _load_testbed()
        if cls.tb is None:
            raise unittest.SkipTest("testbed algorithm_helpers not importable")

    def test_feature_maps_match(self):
        rng = np.random.default_rng(5)
        for _ in range(50):
            st = rand_state(rng, nan_frac=0.2)
            st_tb = {**st, "M_Y": np.nan_to_num(st["M_Y"], nan=np.nan), "M_E": st["M_E"]}
            d, t, a = int(rng.integers(6)), int(rng.integers(2)), int(rng.integers(2))
            bh, bt = rng.normal(), abs(rng.normal())
            np.testing.assert_allclose(
                ah.build_phi_action(bh, bt, st, d, t, a), self.tb.build_phi_action(bh, bt, st_tb, d, t, a)
            )
            np.testing.assert_allclose(
                ah.build_daily_mediator_phi(bh, bt, st, d, t, a),
                self.tb.build_daily_mediator_phi(bh, bt, st_tb, d, t, a),
            )
            np.testing.assert_allclose(
                ah.build_foursc_stage1_phi(bh, bt, st, d, t, a),
                self.tb.build_foursc_stage1_phi(bh, bt, st_tb, d, t, a),
            )
        np.testing.assert_allclose(ah.gamma_dt_matrix(0.9), self._tb_gamma(0.9))

    def _tb_gamma(self, g):
        out = np.ones((6, 2))
        out[5, 1] = g
        return out

    def test_pf_design_rows_match_testbed(self):
        from RL_agents.rl.pf import antic_row, foursc_row
        rng = np.random.default_rng(5)
        for _ in range(20):
            c = rng.normal(size=5)  # model-scale server context
            d, t, a, pu, lag = int(rng.integers(6)), int(rng.integers(2)), int(rng.integers(2)), rng.normal(), rng.normal()
            tb4 = self.tb.build_fourSC_features(
                yesterdayStepCount=c[0], prior2HourStepCount=c[1],
                activeDaysLast7Days=c[2], activitySuggestionsSentLast7Days=c[3],
                activitySuggestionInteractLast7Days=c[4], isWeekend=float(d == 5),
                decisionTimeSlot=t, perceivedUtility=pu, caeAverageLastWeek=0.0, Ah=a,
                stepCountNext4HourLag1=lag,
            )
            np.testing.assert_allclose(foursc_row(c, lag1=lag, d=d, t=t, pu=pu, action=a), tb4)
            wm, wa, yest = int(rng.integers(2)), int(rng.integers(2)), rng.normal()
            tba = self.tb.build_antic_features(
                dailyAnticipatedAffectYesterday=yest, activeDaysLast7Days=c[2], isWeekend=float(d == 5),
                perceivedUtility=pu, caeAverageLastWeek=0.0, activitySuggestionsSentLast7Days=c[3],
                ws_morning=wm, ws_afternoon=wa,
            )
            np.testing.assert_allclose(
                antic_row(antic_yesterday=yest, active=c[2], d=d, pu=pu, rb=c[3], ws_m=wm, ws_a=wa), tba)

    def test_pf_cae_row_matches_testbed_with_missing(self):
        from RL_agents.rl.pf import cae_row
        import vani_env
        rng = np.random.default_rng(3)
        for _ in range(10):
            f = rng.normal(size=12)
            a = rng.normal(size=6)
            f[rng.random(12) < 0.3] = np.nan
            a[rng.random(6) < 0.5] = np.nan
            tb = vani_env.build_pf_CAE_features(0.0, f, a)
            np.testing.assert_allclose(cae_row(f, a), tb)
        np.testing.assert_allclose(cae_row(np.full(12, np.nan), np.full(6, np.nan))[2:], [0.0, 0.0])

    def test_v6_training_data_matches_testbed_construction(self):
        tb = self.tb
        rng = np.random.default_rng(6)
        K = 4
        A = rng.integers(0, 2, size=(K + 1, 6, 2))
        bh = rng.normal(size=K + 1)
        bt = np.abs(rng.normal(size=K + 1))
        eh = rng.normal(size=K + 1)
        states = {(k, d, t): rand_state(rng) for k in range(K + 1) for d in range(6) for t in range(2)}
        full = {k: (rng.normal(size=(6, 3)), rng.normal(size=(6, 4))) for k in range(K)}
        B = 3
        betas = [rng.normal(size=27) * 0.1 for _ in range(B)]
        target = [rng.normal(size=27) * 0.1 for _ in range(B)]
        get_state = lambda k, d, t: states[(k, d, t)]  # noqa: E731
        Phi, targets, info = ah.build_v6_training_data(
            K, A, bh, bt, eh, betas, target, ah.gamma_dt_matrix(0.9), 0.9,
            get_state, lambda k, d, t: (k, d, t) in states, lambda k: full.get(k),
        )
        # Testbed construction (mirrors agents/micro_query_reward_design V4/learned).
        etas = tb.fit_daily_mediator_decomposition(K, A, bh, bt, get_state, lambda k: full[k], priors={})
        rows, y = [], []

        def slot_phi(kp, d, t):
            shares = tb.daily_mediator_shares(etas, bh[kp], bt[kp], states[(kp, d, t)], d, t, A[kp, d, t])
            return tb.build_redistribution_phi(bh[kp], bt[kp], states[(kp, d, t)], d, t, A[kp, d, t], shares, full[kp])

        for kp in range(K):
            rows.append(sum(slot_phi(kp, d, t) for d in range(6) for t in range(2)))
            y.append(bh[kp + 1] + 0.9 * eh[kp + 1] - eh[kp])
        p2 = rows[0].size
        eta2, _ = tb.compute_reward_shaping_eta(np.asarray(rows), np.asarray(y), np.zeros(p2), np.eye(p2), 1.0)
        np.testing.assert_allclose(np.asarray(info["stage2_eta"]), eta2, rtol=1e-8, atol=1e-10)
        exp_rows, exp_t = [], [[] for _ in range(B)]
        g = ah.gamma_dt_matrix(0.9)
        for kp in range(K):
            for d in range(6):
                for t in range(2):
                    r = float(slot_phi(kp, d, t) @ eta2)
                    nxt = tb._next_slot(d, t)
                    kn, (dn, tn), bb = (kp + 1, (0, 0), (bh[kp + 1], bt[kp + 1])) if nxt is None else (kp, nxt, (bh[kp], bt[kp]))
                    exp_rows.append(tb.build_phi_action(bh[kp], bt[kp], states[(kp, d, t)], d, t, A[kp, d, t]))
                    p1 = tb.build_phi_action(*bb, states[(kn, dn, tn)], dn, tn, 1)
                    p0 = tb.build_phi_action(*bb, states[(kn, dn, tn)], dn, tn, 0)
                    for b in range(B):
                        q = (p1 if p1 @ target[b] > p0 @ target[b] else p0) @ betas[b]
                        exp_t[b].append(r + g[d, t] * q)
        np.testing.assert_allclose(Phi, np.asarray(exp_rows))
        for b in range(B):
            np.testing.assert_allclose(targets[b], np.asarray(exp_t[b]), rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end handler behaviour through the framework contract
# ---------------------------------------------------------------------------
def fresh(seed=0):
    return Participant(handler(), seed=seed)


class TestHandler(unittest.TestCase):
    def test_multi_week_run_updates_policy(self):
        p = fresh()
        for k in range(5):
            p.run_week(k)
        self.assertTrue(all(s["ok"] for s in p.statuses), [s.get("reason") for s in p.statuses])
        self.assertEqual(p.params["active_policy_key"], 4)
        probs = [a.action_prob for k, *_ , a in p.results if k >= 1]
        self.assertTrue(all(0.1 <= x <= 0.9 for x in probs))
        self.assertFalse([a for *_, a in p.results if a.random_state.get("fallback")])

    def test_deterministic_given_seed(self):
        a, b = fresh(3), fresh(3)
        for k in range(3):
            a.run_week(k)
            b.run_week(k)
        self.assertEqual([r[3].action for r in a.results], [r[3].action for r in b.results])
        np.testing.assert_allclose(np.asarray(a.params["betas_store"]["2"], dtype=float),
                                   np.asarray(b.params["betas_store"]["2"], dtype=float))

    def test_missing_context_is_imputed_from_previous_7_days(self):
        p = fresh()
        for k in range(2):
            p.run_week(k)
        ar = p.decide(2, 2, 0, missing_c=True)
        inp = ar.random_state["inputs"]
        self.assertFalse(inp["C_missing"])
        self.assertEqual(sorted(inp["C_imputed_indices"]), list(range(5)))
        self.assertIn("pi_hat", ar.random_state)  # RL decision, not the safe policy

    def test_unavailable_returns_zero_without_randomization(self):
        p = fresh()
        p.run_week(0)
        ar = p.decide(1, 1, 0, available=False)
        self.assertEqual((ar.action, ar.action_prob), (0, 0.0))
        self.assertTrue(ar.random_state["unavailable"])

    def test_monday_uses_last_week_inputs_without_staleness(self):
        p = fresh()
        p.run_week(0)
        p.run_week(1)
        ar = p.decide(2, 0, 0)  # before Monday-night prepare/update of week 2
        inp = ar.random_state["inputs"]
        self.assertEqual((inp["expected_input_week"], inp["b_hat_tag"], inp["e_hat_tag"]), (1, 1, 1))
        self.assertEqual((inp["b_hat_staleness"], ar.random_state["policy_staleness"]), (0, 0))
        self.assertEqual(ar.random_state["policy_key"], 1)
        p.run_update(2)
        ar = p.decide(2, 1, 0)  # Tuesday: this week's inputs and policy
        self.assertEqual((ar.random_state["inputs"]["b_hat_tag"], ar.random_state["policy_key"]), (2, 2))

    def test_monday_of_week_2_uses_baseline_inputs(self):
        p = fresh()
        p.run_week(0)
        ar = p.decide(1, 0, 0)
        self.assertIn("pi_hat", ar.random_state)
        self.assertEqual(ar.random_state["inputs"]["b_hat_staleness"], 0)

    def test_server_context_must_have_six_entries(self):
        from RL_agents.rl.week_data import collect_week_data
        p = fresh()
        p.db.add_obs({"w": 1, "d": 0, "t": 0, "C": [1.0] * 9})
        data = collect_week_data(p.db, W=int(p.params["W"]), action_type=p.h.action_type)
        self.assertEqual(data.bad_context, [(0, 0, 0, 9)])
        self.assertTrue(np.all(np.isnan(data.C[0, 0, 0])))

    def test_missed_rlsvi_updates_escalate_policy_staleness(self):
        p = fresh()
        p.run_week(0)
        p.run_week(1)
        p.run_week(2, rlsvi=False)
        ar = p.decide(3, 0, 0)  # Monday wants policy 2, has 1 -> RL, flagged
        self.assertEqual(ar.random_state.get("policy_staleness"), 1)
        p.run_week(3, rlsvi=False)
        ar = p.decide(4, 2, 0)
        self.assertEqual(ar.random_state.get("code"), "RL_STALE_POLICY")

    def test_missed_updates_escalate_inputs_and_pf_catches_up(self):
        p = fresh()
        p.run_week(0)
        p.run_week(1)
        p.run_week(2, update=False)
        p.run_week(3, update=False)
        ar = p.decide(4, 2, 0)  # inputs from week 1, expected 4 -> safe policy
        self.assertEqual(ar.random_state.get("code"), "RL_STALE_INPUTS")
        p.run_update(4)
        self.assertTrue(p.statuses[-1]["ok"], p.statuses[-1])
        self.assertEqual(len(p.params["pf_y_hat"][0]), 5)  # weeks 0..4 filtered in order
        ar = p.decide(4, 3, 0)
        self.assertIn("pi_hat", ar.random_state)

    def test_crash_between_prepare_and_update_recovers(self):
        p = fresh()
        p.run_week(0)
        p.run_week(1)
        p.run_week(2, rlsvi=False)  # prepare stored, RLSVI never ran
        p.run_week(3)
        self.assertTrue(p.statuses[-1]["ok"], p.statuses[-1])
        self.assertTrue(p.statuses[-1]["belief_applied"])
        self.assertEqual(len(p.params["pf_y_hat"][0]), 4)

    def test_missing_decision_record_keeps_week_in_training(self):
        a, b = fresh(3), fresh(3)
        for k in range(3):
            a.run_week(k)
            b.run_week(k, skip_slots={(3, 1)} if k == 1 else ())
        self.assertTrue(b.statuses[-1]["ok"], b.statuses[-1])
        self.assertEqual(a.statuses[-1].get("n_rows"), b.statuses[-1].get("n_rows"))

    def test_checkin_label_is_opening_week(self):
        from RL_agents.rl.week_data import collect_week_data
        p = fresh()
        p.run_week(0)
        data = collect_week_data(p.db, W=int(p.params["W"]), action_type=p.h.action_type)
        self.assertEqual(data.J_w[1], 1)  # Sunday survey after week 0 opens week 1 (0-idx)
        self.assertTrue(np.isfinite(data.Y[1]))

    def test_rerun_update_same_week_is_idempotent(self):
        p = fresh()
        for k in range(3):
            p.run_week(k)
        before = copy.deepcopy(p.params["betas_store"])
        p.run_update(2)
        self.assertEqual(p.statuses[-1]["reason"], "already_updated_for_week")
        self.assertEqual(before, p.params["betas_store"])

    def test_duplicate_observations_do_not_change_result(self):
        a, b = fresh(7), fresh(7)
        for k in range(3):
            a.run_week(k)
            b.run_week(k)
            # re-deliver every observation of b once more with the same ids
            b.db.observations.extend(list(b.db.observations))
        np.testing.assert_allclose(np.asarray(a.params["b_hat_hist"], dtype=float)[:3],
                                   np.asarray(b.params["b_hat_hist"], dtype=float)[:3])

    def test_missing_baseline_y1_is_a_hard_failure(self):
        p = fresh()
        p.db.observations = [o for o in p.db.observations if o.observation_type != "rl_Y_1"]
        p.run_week(0)
        with self.assertRaises(KeyError):
            p.run_update(1)

    def test_model_params_from_other_version_are_refused(self):
        p = fresh()
        p.run_week(0)
        p.params["package_version_id"] = "stale"
        ar = p.decide(1, 1, 0)
        self.assertEqual(ar.random_state.get("code"), "RL_VERSION_MISMATCH")

    def test_missing_outcome_week_excluded_not_zeroed(self):
        p = fresh()
        p.run_week(0)
        p.run_week(1, skip_outcome=True)
        p.run_week(2)
        p.run_week(3)
        st = p.statuses[-1]
        self.assertTrue(st["ok"], st)
        self.assertTrue(np.all(np.isfinite(np.asarray(p.params["betas_store"]["3"], dtype=float))))


if __name__ == "__main__":
    unittest.main()


class TestSeedSalt(unittest.TestCase):
    SALT_A = "a" * 32
    SALT_B = "b" * 32

    def _handler(self, salt):
        import os
        from RL_agents.seed_salt import SALT_ENV_VAR
        old = os.environ.get(SALT_ENV_VAR)
        os.environ[SALT_ENV_VAR] = salt
        try:
            return RL()
        finally:
            if old is None:
                os.environ.pop(SALT_ENV_VAR)
            else:
                os.environ[SALT_ENV_VAR] = old

    def test_salt_changes_draws_and_is_never_stored(self):
        pa, pb = Participant(self._handler(self.SALT_A)), Participant(self._handler(self.SALT_B))
        pa.run_week(0)
        pb.run_week(0)
        seeds_a = [ar.random_state["seed"] for *_, ar in pa.results]
        seeds_b = [ar.random_state["seed"] for *_, ar in pb.results]
        self.assertNotEqual(seeds_a, seeds_b)
        self.assertNotEqual(pa.params["seed_salt_fingerprint"], pb.params["seed_salt_fingerprint"])
        import json
        blob = json.dumps(pa.params, default=str) + json.dumps([ar.random_state for *_, ar in pa.results])
        self.assertNotIn(self.SALT_A, blob)

    def test_changed_salt_is_alerted_but_does_not_block(self):
        p = Participant(self._handler(self.SALT_A))
        p.run_week(0)
        p.run_week(1)
        p.h = self._handler(self.SALT_B)
        with self.assertLogs("RL_agents.rl.RL", level="WARNING") as cm:
            ar = p.decide(2, 2, 0)
        self.assertTrue(any("salt differs" in m for m in cm.output), cm.output)
        self.assertNotIn("fallback", ar.random_state)

    def test_missing_or_short_salt_refuses_to_start(self):
        import os
        from RL_agents.rl.RL import load_config
        from RL_agents.seed_salt import SALT_ENV_VAR, SeedSaltError
        cfg = load_config()
        cfg["seed_salt_path"] = "does_not_exist.local"
        old = os.environ.pop(SALT_ENV_VAR, None)
        try:
            with self.assertRaises(SeedSaltError):
                RL(cfg)
            os.environ[SALT_ENV_VAR] = "short"
            with self.assertRaises(SeedSaltError):
                RL(cfg)
        finally:
            os.environ.pop(SALT_ENV_VAR, None)
            if old is not None:
                os.environ[SALT_ENV_VAR] = old


class TestAnticipatedAffectInState(unittest.TestCase):
    def test_state_uses_only_answered_check_ins(self):
        from RL_agents.rl.week_data import WeekData
        from RL_agents.rl.standardization import standardize_anticipated_affect
        h = handler()
        data = WeekData(4)
        data.antic[1, :] = [6.0, np.nan, 2.0, np.nan, np.nan, np.nan]
        M_Y_obs, _ = data.model_mediators(1, h.std_params)
        M_Y_locf, _ = data.model_mediators(1, h.std_params, antic="locf")
        self.assertTrue(np.isnan(M_Y_obs[1, 2]))
        self.assertEqual(M_Y_locf[1, 2], M_Y_locf[0, 2])
        # EWMA at Thursday (d=3): calendar positions 0..2, position 1 skipped
        a0, a2 = standardize_anticipated_affect(6.0), standardize_anticipated_affect(2.0)
        g = 1 - 1 / 3
        expected = (g**2 * a0 + a2) / (g**2 + 1)
        got = ah.summarize_mediators_ewma(M_Y_obs, np.full((6, 4), np.nan), 3, 0)[0]
        self.assertAlmostEqual(got, expected)
