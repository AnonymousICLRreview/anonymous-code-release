"""Numerical regressions for the model, geometry and fixed-budget controller.

These tests exercise invariants and independent mathematical checks; they do not
assert any benchmark accuracy, GPU timing, or statistical audit outcome.
"""
import json
import math

import unittest
import torch

from ropc.model import (
    ActionGeometry, GraphData, RoleEncoder, Sensor, SpectralLinear,
    SplineSchedule, UnionSupport, build_sensed_set, equivariant_hash,
)
from ropc.control import (
    BasePolicyGRU, Controller, ConvexHead, Envelope, HealthRegion, Planner,
    PlannerConfig, ReliabilityScorer, SupportDistance, TransitionModel,
    eroded_bounds, margin_scaled_execution,
)


def graph(x, edges, batch=None, num_graphs=1):
    n = len(x)
    return GraphData("test", x, torch.zeros(n, dtype=torch.long), edges, n, 2,
                     torch.arange(n), torch.empty(0, dtype=torch.long),
                     torch.empty(0, dtype=torch.long), batch, num_graphs)


def small_controller_parts(dtype=torch.float64):
    geom = ActionGeometry(4, 2, .05, .4, 4)
    model = TransitionModel(6, 8, 10, 3, 2, .96, .08, 1., .35,
                            geom.tau, [0, 5]).to(dtype=dtype)
    value = ConvexHead(8, 8, 24).to(dtype=dtype)
    stage = ConvexHead(8, 8).to(dtype=dtype)
    health = HealthRegion(8, 2, 4.)
    support = SupportDistance(10, .5)
    ref = SplineSchedule(geom, 2).to(dtype=dtype)
    return geom, model, value, stage, health, support, ref


class ModelControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(127)
        self._old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self._old_threads)

    def test_projection_thin_trust_boxes_is_feasible_and_optimal(self):
        for floor in [0, 2, 4, 8, 16]:
            geom = ActionGeometry(4, 4, .05, .4, floor)
            center = geom.parameterize(torch.randn(20, 4, 6))
            lo = torch.maximum(geom.lo, center - .025)
            hi = torch.minimum(geom.hi, center + .025)
            query = 10 * torch.randn_like(center)
            result, _ = geom.project(query, lo, hi)
            assert geom.feasibility_violation(result).max() < 3e-7
            assert (lo - result).max() < 2e-7
            assert (result - hi).max() < 2e-7
            # The Euclidean projection satisfies its variational inequality against any
            # feasible point. Construct independent feasible points by convex mixing.
            for _ in range(12):
                other = geom.parameterize(torch.randn_like(center))
                delta = other - center
                scale = (.024 / delta.abs().amax(-1, keepdim=True).clamp_min(.024)).clamp_max(1)
                feasible = center + scale * delta
                normal_product = ((query - result) * (feasible - result)).sum(-1)
                assert normal_product.max() < 2e-5

    def test_projection_rejects_empty_eroded_polytope(self):
        geom = ActionGeometry(4, 1, .05, .4, 16)
        with self.assertRaisesRegex(ValueError, "empty|Empty"):
            geom.project(torch.zeros(1, 1, 6), beta=torch.tensor([.5]))


    def test_exact_erosion_keeps_all_antithetic_box_vertices_feasible(self):
        geom = ActionGeometry(4, 1, .05, .4, 4)
        radii = torch.full((5,), .002, dtype=torch.float64)
        lo, hi, beta = eroded_bounds(geom, radii)
        proposed = geom.parameterize(torch.zeros(1, 1, 6, dtype=torch.float64))
        center, _ = geom.project(proposed, lo, hi, beta)
        signs = torch.tensor([[1. if (row >> col) & 1 else -1. for col in range(5)]
                              for row in range(32)], dtype=torch.float64)
        actions = geom.from_tangent(geom.to_tangent(center) + signs * radii)
        assert geom.feasibility_violation(actions).max() < 1e-10
        expected_floor_erosion = (geom.n_vec.double() @ geom.T.double()).abs() @ radii
        assert torch.allclose(beta, 1 - expected_floor_erosion)


    def test_spectral_upper_bound_survives_parameter_edits_and_backward(self):
        layer = SpectralLinear(2, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.diag(torch.tensor([2., 1.5])))
            layer.u.copy_(torch.tensor([0., 1.]))
            layer.v.copy_(torch.tensor([0., 1.]))
        assert torch.linalg.matrix_norm(layer.effective_weight(), 2) <= 1 + 1e-6
        layer.update_sigma()
        assert torch.linalg.matrix_norm(layer.effective_weight(), 2) <= 1 + 1e-6
        x = torch.randn(7, 2)
        first = layer(x)
        assert torch.equal(first, layer(x))
        first.square().sum().backward()
        assert torch.isfinite(layer.weight.grad).all()


    def test_csr_row_sums_do_not_subtract_large_float32_prefixes(self):
        sup = UnionSupport(torch.empty(2, 0, dtype=torch.long), 5)
        values = torch.tensor([1e12, .01, .03, .25, 1.])
        assert torch.equal(sup.row_sums(values), values)
        edges = torch.tensor([[0, 0, 1, 2, 3], [1, 1, 2, 3, 0]])
        sup = UnionSupport(edges, 4)
        normalized = sup.normalized_values(sup.posA, torch.tensor([1e6, 1e-5, .2, .3, .4]), 1.)
        assert torch.allclose(sup.row_sums(normalized), torch.ones(4), atol=2e-7)


    def test_structural_likelihood_has_correct_population_and_true_nonedges(self):
        role = RoleEncoder(2, 2, 1, .001, hidden=4)
        x = torch.zeros(4, 2)
        # Two complete two-node graphs have no negative ordered distinct pairs.
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        data = graph(x, edges, torch.tensor([0, 0, 1, 1]), 2)
        theta = torch.full((4, 2), .5)
        B = torch.full((2, 2), .3)
        roles = {"Theta_f": theta, "B": B, "nu": torch.zeros(4)}
        logit = role.beta0 + role.beta1() * .3
        loss = role.struct_loss(data, roles, 8, 0, torch.Generator().manual_seed(1))
        assert torch.allclose(loss, torch.nn.functional.softplus(-logit))
        # On one graph, two unique arcs out of twelve possible pairs have exact
        # positive/negative population weights (all logits equal in this fixture).
        sparse = graph(x, torch.tensor([[0, 0, 1], [1, 1, 2]]))
        loss = role.struct_loss(sparse, roles, 8, 128, torch.Generator().manual_seed(2))
        expected = (2 * torch.nn.functional.softplus(-logit) + 10 * torch.nn.functional.softplus(logit)) / 12
        assert torch.allclose(loss, expected, atol=1e-6)


    def test_hash_is_invariant_to_edge_order_and_equivariant_to_nodes(self):
        x = torch.randn(25, 3)
        edges = torch.randint(25, (2, 120))
        scores = equivariant_hash(x, edges, len(x), 12)
        perm = torch.randperm(len(x))
        inv = perm.argsort()
        shuffled = inv[edges[:, torch.randperm(edges.shape[1])]]
        assert torch.equal(equivariant_hash(x[perm], shuffled, len(x), 12), scores[perm])
        n = 16
        ring = torch.stack([torch.arange(n), torch.arange(n).roll(1)])
        regular = graph(torch.zeros(n, 2), ring)
        sensed = build_sensed_set(regular, 4, 3)
        assert len(sensed.idx) == n  # ties are all kept, never broken by node IDs


    def test_sensor_uses_original_edge_weights_and_explicit_uncertainty(self):
        data = graph(torch.randn(6, 3), torch.tensor([[0, 1, 2, 3, 4], [1, 1, 1, 4, 5]]))
        sensed = build_sensed_set(data, 10, 4, gamma_sl=2.)
        assert torch.allclose(sensed.e_w[:3], torch.full((3,), 1 / 5))
        sensor = Sensor(4, 3, 2, 1, "cpu")
        H = torch.randn(6, 4)
        theta = torch.full((6, 2), .5)
        uncertainty = torch.linspace(.1, .9, 6)
        z = sensor.observe(H, H, theta, torch.zeros(6, 1), sensed, 1,
                           uncertainty=uncertainty)
        assert torch.allclose(z[0, [4, 8, 13]], uncertainty.mean().expand(3))
        assert torch.isfinite(z).all()


    def test_two_knot_reference_is_monotone_in_action_coordinates(self):
        geom = ActionGeometry(4, 2, .05, .4, 4)
        schedule = SplineSchedule(geom, 2)
        with torch.no_grad():
            schedule.raw.normal_(0, 3)
        values = schedule(torch.linspace(0, 1, 101))
        differences = values[1:] - values[:-1]
        sign = torch.sign(values[-1] - values[0])
        assert (differences * sign).min() >= -1e-7
        assert geom.feasibility_violation(values).max() < 3e-7


    def test_transition_shape_observer_and_depth_zero_derivatives(self):
        geom, model, *_ = small_controller_parts()
        assert model.tau.shape == (geom.du,)
        assert model.observer_contraction() < 1
        x = torch.randn(2, 6, dtype=torch.float64)
        context = torch.randn(2, 3, dtype=torch.float64)
        u = geom.to_tangent(geom.parameterize(torch.randn(2, 2, 6, dtype=torch.float64)))
        t = torch.zeros(2, dtype=torch.float64)
        depth0 = torch.tensor([True, False])
        response = model.response(x, t, depth0)
        assert torch.count_nonzero(response[0, :, [0, 5]]) == 0
        for i in range(2):
            jac = torch.autograd.functional.jacobian(
                lambda xx: model.predict(xx[None], t[i:i+1], u[i:i+1], context[i:i+1], depth0[i:i+1])[0], x[i])
            assert torch.allclose(jac, model.M_of(x, t, u, depth0)[i], atol=1e-12)


    def test_convex_head_jensen_and_actual_hessian_bound(self):
        head = ConvexHead(4, 7, 3).double()
        head.certify_(4.)
        z1, z2 = torch.randn(2, 4, dtype=torch.float64)
        context = torch.randn(1, 3, dtype=torch.float64)
        value = lambda z: head(z[None], context)[0]
        assert value(.3 * z1 + .7 * z2) <= .3 * value(z1) + .7 * value(z2) + 1e-12
        hessian = torch.autograd.functional.hessian(value, z1)
        ev = torch.linalg.eigvalsh(hessian)
        assert ev.min() >= -1e-10
        assert ev.max() <= head.hessian_bound() + 1e-10


    def test_planner_majorant_is_tangent_dominating_and_descending_with_health(self):
        geom, model, value, stage, health, support, _ = small_controller_parts()
        health.fit(torch.randn(45, 8), torch.linspace(0, 1, 45), torch.arange(8.))
        health.rho_c = .1  # keep the nonlinear hinge active in this regression
        planner = Planner(PlannerConfig(), geom, model, value, stage, health, support, x_max=.0001)
        x = 2 * torch.randn(2, 6, dtype=torch.float64)
        c = torch.randn(2, 3, dtype=torch.float64)
        base = geom.parameterize(torch.randn(2, 3, 2, 6, dtype=torch.float64))
        context = torch.randn(2, 24, dtype=torch.float64)
        mask = torch.tensor([[True, False], [True, True]])
        with torch.no_grad():
            plan, info = planner.plan(x, 0, c, base, None, base[:, 0], context,
                                      unit_test_points=6, role_mask=mask)
        assert geom.feasibility_violation(plan).max() < 1e-10
        assert torch.equal(plan[0, :, 1], base[0, :, 1])
        assert (info["J_plan"] <= info["J_warm"] + 1e-10).all()
        assert planner.last_info["majorant_tangent_error"] < 1e-10
        assert planner.last_info["majorant_min_slack"] > -1e-10
        assert info["betas"][0] == 0
        # Recursive bounds cover states away from the nominal action sequence.
        for _ in range(5):
            actions = geom.parameterize(torch.randn_like(base) * 4)
            with torch.no_grad():
                states, _, _ = planner.rollout(x, geom.to_tangent(actions), info["t_seq"], c,
                                               torch.ones(2, dtype=torch.bool), False)
            bound = torch.tensor(info["state_norm_bounds"], dtype=states.dtype)
            assert (states.norm(dim=-1) <= bound + 1e-10).all()


    def test_envelope_infinity_fallback_is_finite_and_strict_json_roundtrips(self):
        geom = ActionGeometry(4, 2, .05, .4, 4)
        base = geom.parameterize(torch.zeros(3, 2, 6))
        for slopes in [(1., 2., math.inf), (math.inf, math.inf, math.inf)]:
            env = Envelope(slopes)
            encoded = json.dumps(env.to_dict(), allow_nan=False)
            restored = Envelope.from_dict(json.loads(encoded))
            assert restored.slopes == env.slopes
            action, rho, certificate, *_ = margin_scaled_execution(
                base, torch.zeros_like(base), torch.zeros(3), torch.tensor([0., .94, 1.]), env, .001, geom)
            assert torch.equal(action, base)
            assert torch.isfinite(rho).all() and torch.isfinite(certificate).all()
            assert (rho == 0).all()


    def test_reliability_interpolation_is_continuous_at_first_knot(self):
        score = ReliabilityScorer(1, 1)
        score.knots = torch.tensor([1., 2.])
        score.cdf = torch.tensor([.2, 1.])
        raw = torch.tensor([[1. - 1e-7], [1.], [1. + 1e-7]], dtype=torch.float64).sqrt()
        result = score.score(raw, torch.zeros_like(raw))
        assert (result - .2).abs().max() < 1e-6
        score.fit_training(torch.ones(1, 1), torch.ones(1, 1))
        assert torch.isfinite(score.e_scale).all()


    def test_commit_uses_its_decision_record_after_other_branch_queries(self):
        geom, model, value, stage, health, support, ref = small_controller_parts()
        base_policy = BasePolicyGRU(geom, 6, 5).double()
        ctrl = Controller("base", geom, PlannerConfig(), model, value, stage, health,
                          support, Envelope(), .001, None, None, ref, base=base_policy, wA=5)
        context = torch.randn(2, 3, dtype=torch.float64)
        ctrl.reset(2, context)
        action, record = ctrl.step(0, torch.randn(2, 8, dtype=torch.float64), context)
        expected_eta = record["eta_next"].clone()
        expected_prior = model.predict(record["x"], torch.zeros(2, dtype=torch.float64),
                                       geom.to_tangent(action), context, torch.ones(2, dtype=torch.bool))
        # Simulate a later branch overwriting the legacy pending tuple.
        ctrl.step(9, 20 * torch.randn(2, 8, dtype=torch.float64), context)
        ctrl._pending = (torch.full_like(expected_eta, 999), torch.zeros_like(expected_eta), None)
        ctrl.commit(0, action, record, context)
        assert torch.equal(ctrl.eta, expected_eta)
        assert torch.allclose(ctrl.x_prior, expected_prior, atol=1e-12)


    def test_replay_retains_distinct_actions_for_each_graph(self):
        geom, _, _, _, _, _, ref = small_controller_parts()
        trace = geom.parameterize(torch.randn(3, 2, 2, 6, dtype=torch.float64))
        ctrl = Controller("replay", geom, PlannerConfig(), None, None, None, None,
                          None, Envelope(), .001, None, None, ref, trace=trace, wA=5)
        ctrl.reset(2, None)
        for layer in range(3):
            action, _ = ctrl.step(layer, None, None)
            assert torch.equal(action, trace[layer])


if __name__ == "__main__":
    unittest.main()
