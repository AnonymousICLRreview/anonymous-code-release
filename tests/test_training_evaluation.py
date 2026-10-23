"""Pipeline invariants: split isolation, interventions, frozen state and metrics."""
from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import numpy as np
import torch

from ropc.control import Envelope, RotatingDesign
from ropc.datasets import load_dataset
from ropc.evaluation import (PolicyBundle, audit_policy, build_policy_ctrl, calibrate_policy,
                             compile_and_replay, distill_mean_trace, evaluate_arm,
                             identification_diagnostics, library_regret, observation_noise,
                             policy_hash, quantile_index_slope, realized_gains,
                             run_with_snapshots)
from ropc.model import UnionSupport
from ropc.tasks import metric_from_predictions, predictions
from ropc.training import (ControllerSnapshot, Plant, RoundState, base_continuation,
                           build_backbone, collect_branches, evaluate_depths,
                           fit_heads, fit_model, fit_observation_normalizer,
                           fit_round_components, make_config, make_controller,
                           make_partition, pooled_blocks, run_pass, spearman,
                           train_task_route)


def setup_parts(name="ZINC", **overrides):
    torch.manual_seed(193)
    options = dict(dataset=name, d=8, dff=8, role_hidden=8, Kf=4, Kc=2,
                   sketch_b=4, wA=4, head_width=8, bank="P0,Pg", H=2,
                   L_max=2, L_star=2, L_a=2, T_cap=4, Kb=4, n_sens=32,
                   ess_min=1, No=1, Npd=2, epochs_heads=1, epochs_ident=1,
                   epochs_dagger=1, structural_samples=32, lam_struct=0.,
                   dy=2, batch_size=3, branch_traj=1, states_per_traj=4,
                   tune_samples=2, audit_samples=1, reserve_samples=1)
    options.update(overrides)
    cfg = make_config("preflight", options)
    graph = load_dataset(name, "/unused", seed=cfg.seed, synthetic=True)
    bb = build_backbone(cfg, graph)
    plant = Plant(cfg, bb, graph, UnionSupport(graph.edge_index,graph.num_nodes),
                  make_partition(graph,cfg), torch.device("cpu"))
    plant.prepare()
    rs = RoundState(cfg,plant.geom,torch.device("cpu"))
    def base():
        return make_controller(cfg,plant,"base",rs.model,rs.value,rs.stage,
                               rs.health,rs.support,rs.envelope,rs.scorer)
    return cfg,plant,rs,base


class TrainingEvaluationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_family_partitions_and_training_never_read_test_graphs(self):
        cfg,plant,_,_ = setup_parts()
        source = plant.source
        self.assertEqual(plant.part.tune_seeds,source.partitions["tune"][:2])
        original_sample = source.sample
        seen = []
        def guarded(sample_id, split=None):
            self.assertNotIn(sample_id,source.partitions["test"])
            seen.append((sample_id,split))
            return original_sample(sample_id,split)
        with tempfile.TemporaryDirectory() as directory, patch.object(source,"sample",side_effect=guarded):
            output = train_task_route(cfg,plant,lambda:make_controller(cfg,plant,"schedule"),
                                      1,Path(directory),"task",val_depths=[1,2])
        self.assertFalse(output["test_evaluated"])
        self.assertEqual(output["metric_mode"],"min")
        self.assertGreater(output["history"][0]["n_batches"],1)
        self.assertTrue(any(sample in source.partitions["valid"] for sample,_ in seen))
        self.assertFalse(any(sample in source.partitions["value"] for sample,_ in seen))

    def test_whole_family_metric_is_invariant_to_batch_size(self):
        cfg,plant,_,base = setup_parts("Peptides-func")
        cfg.batch_size = 1
        first = evaluate_depths(plant,base,[1,2],"valid")
        cfg.batch_size = 3
        second = evaluate_depths(plant,base,[1,2],"valid")
        for depth in first:
            self.assertAlmostEqual(first[depth],second[depth],places=7)

    def test_lower_is_better_depth_selection_and_pdd(self):
        cfg,plant,_,base = setup_parts("LRIM-16")
        def metrics(*args,**kwargs):
            split = args[3] if len(args)>3 else kwargs["split"]
            return {1: -1.,2: -3.,3: -2.,4: -1.5} if split=="valid" else {1:-1.,2:-4.,3:-3.,4:-2.}
        with patch("ropc.evaluation.evaluate_depths",side_effect=metrics):
            result = evaluate_arm(cfg,plant,base(),4)
        self.assertEqual(result["P1"]["L_selected"],2)
        self.assertEqual(result["P2"]["peak_depth"],2)
        self.assertEqual(result["P2"]["PDD"],2.)
        self.assertEqual(result["P2"]["DA"],-3.)
        self.assertIsNone(result["trace"])

    def test_stage_return_starts_at_entry_state_and_restores_memory(self):
        cfg,plant,_,base = setup_parts()
        ctrl = base()
        output = run_pass(plant,ctrl,3,prepared=True,stage_losses=True)
        self.assertEqual(len(output["g"]),3)
        self.assertAlmostEqual(output["g"][0],float(plant.stage_loss(plant.H0)),places=7)
        ctrl.reset(plant.B,plant.ctx)
        before = ControllerSnapshot(ctrl)
        value,_,_ = base_continuation(plant,ctrl,plant.H0,0,3,0,hazard=.25)
        expected = sum(.25*.75**i*g for i,g in enumerate(output["g"]))
        self.assertAlmostEqual(value,expected,places=7)
        torch.testing.assert_close(ctrl.A_prev,before.A_prev)
        torch.testing.assert_close(ctrl.eta,before.eta)
        with patch.object(plant,"layer",side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError,"injected"):
                base_continuation(plant,ctrl,plant.H0,0,3,0)
        torch.testing.assert_close(ctrl.A_prev,before.A_prev)

    def test_inference_snapshots_are_label_free_and_noise_replays_identically(self):
        cfg,plant,_,base = setup_parts()
        with patch.object(plant,"stage_loss",side_effect=AssertionError("label access")):
            output = run_with_snapshots(plant,base(),2,[1],0,obs_noise=.2,noise_seed=11)
        self.assertEqual(output["g"],[])
        raw = plant.observe(plant.H0,0)
        first = observation_noise(raw,plant,11,1,.2)
        torch.randn(117)
        second = observation_noise(raw,plant,11,1,.2)
        self.assertTrue(torch.equal(first,second))
        self.assertFalse(torch.equal(first,observation_noise(raw,plant,11,2,.2)))

    def test_completed_intervention_blocks_and_actual_branch_memory(self):
        cfg,plant,rs,base = setup_parts()
        fit_observation_normalizer(plant,sample_count=2)
        bd = collect_branches(cfg,plant,base,1,4,cfg.T_cap,seed=5,explore=False)
        self.assertEqual(bd.observation_space,"raw")
        self.assertTrue(set(bd.sample_ids.tolist()) <= set(plant.source.partitions["value"]))
        self.assertFalse(torch.equal(bd.z,plant.normalizer(bd.z)))
        blocks = pooled_blocks(bd,plant.normalizer,RotatingDesign(plant.geom.du,cfg.Kb))
        self.assertEqual(len(blocks),2)
        for bid in torch.unique(bd.block):
            radii = bd.delta_br[bd.block==bid]
            self.assertTrue(torch.equal(radii,radii[0].expand_as(radii)))
        actions = plant.geom.from_tangent(bd.u_br)
        self.assertLess(float(plant.geom.feasibility_violation(actions).max()),3e-6)
        memory_action = bd.ctx_br[...,1+cfg.dx+cfg.wA:]
        torch.testing.assert_close(memory_action,actions.flatten(-2),atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.isfinite(bd.G_br[bd.cont_br]).all())
        self.assertTrue(torch.isnan(bd.G_br[~bd.cont_br]).all())
        info = fit_round_components(cfg,plant,rs,bd,plant.device,RotatingDesign(plant.geom.du,cfg.Kb))
        self.assertTrue(math.isfinite(info["model"]["nmse1"]))
        frozen = {name:value.clone() for name,value in rs.value.state_dict().items()}
        diagnostic = fit_model(cfg,rs.model,rs.value,rs.health,bd,plant.normalizer,
                               plant.device,1,RotatingDesign(plant.geom.du,cfg.Kb))
        self.assertTrue(diagnostic["observer_fitted"])
        self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for name,p in rs.model.named_parameters() if "Ro" in name or name=="C"))
        for name,value in rs.value.state_dict().items():
            self.assertTrue(torch.equal(value,frozen[name]))

    def test_compiled_batch_trace_preserves_every_graph(self):
        cfg,plant,_,base = setup_parts()
        plant.bind_graph(next(plant.source.iter_batches("valid",3)))
        output = compile_and_replay(cfg,plant,base(),3)
        self.assertEqual(output["trace"].shape,(3,3,cfg.Kc,plant.geom.p))
        self.assertTrue(output["bitwise_equal"])
        self.assertEqual(output["max_abs_diff"],0.)

    def test_distilled_schedule_uses_only_value_training_graphs(self):
        _,plant,_,base = setup_parts()
        result = distill_mean_trace(plant,base,2,sample_count=2)
        self.assertEqual(result["sample_ids"],plant.source.partitions["value"][:2])
        self.assertEqual(result["n_graphs"],2)
        self.assertLess(float(plant.geom.feasibility_violation(result["trace"]).max()),2e-6)

    def test_quantile_unsupported_bins_and_tied_spearman(self):
        self.assertTrue(math.isinf(quantile_index_slope([],.95)))
        self.assertTrue(math.isinf(quantile_index_slope([1.,2.],.95)))
        self.assertEqual(quantile_index_slope(list(range(19)),.95),18.)
        with self.assertRaises(ValueError):
            quantile_index_slope([float("nan")],.5)
        self.assertAlmostEqual(spearman(torch.tensor([1.,1.,2.]),torch.tensor([1.,1.,2.])),1.)
        self.assertTrue(math.isnan(spearman(torch.ones(3),torch.arange(3))))

    def test_empty_audit_bins_fail_and_family_calibration_uses_real_ids(self):
        cfg,plant,rs,_ = setup_parts()
        pb = PolicyBundle("ropc",rs,envelope=Envelope((float("inf"),)))
        initial = copy.deepcopy(plant.bb.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            calibrated = calibrate_policy(cfg,plant,pb,plant.part.tune_seeds,Path(directory),n_passes=1)
            self.assertTrue(all(math.isinf(s) for s in calibrated["envelope"].slopes))
            self.assertGreater(calibrated["info"]["reliability_fit_rows"],0)
            self.assertEqual(calibrated["info"]["sample_unit"],"graph")
            audit = audit_policy(cfg,plant,pb,calibrated["envelope"],plant.part.audit_seeds,Path(directory),"audit")
        self.assertFalse(audit["policy_pass"])
        self.assertTrue(all(b["n"]==0 and not b["pass"] and b["lower_bound"]==0 for b in audit["bins"].values()))
        self.assertEqual(audit["decision_scope"],"all_acting")
        for key,value in plant.bb.state_dict().items():
            self.assertTrue(torch.equal(value,initial[key]))

    def test_policy_hash_covers_normalizer_reference_heads_and_student(self):
        cfg,plant,rs,_ = setup_parts()
        pb = PolicyBundle("ropc",rs)
        env = Envelope((1.,2.,3.))
        original = policy_hash(cfg,pb,env,plant.bb,plant)
        plant.loss_bar *= 2
        self.assertNotEqual(original,policy_hash(cfg,pb,env,plant.bb,plant))
        plant.loss_bar /= 2
        pb.amortized = rs.new_policy(plant.bb.ref_schedule.raw[0].detach())
        student_hash = policy_hash(cfg,pb,env,plant.bb,plant)
        self.assertNotEqual(original,student_hash)
        pb.heads_ref = (copy.deepcopy(rs.value),copy.deepcopy(rs.stage))
        self.assertNotEqual(student_hash,policy_hash(cfg,pb,env,plant.bb,plant))
        pb.heads_ref = None; pb.amortized = None
        self.assertEqual(original,policy_hash(cfg,pb,env,plant.bb,plant))

    def test_resume_matches_uninterrupted_optimizer_and_rng(self):
        cfg,plant,_,_ = setup_parts("ogbn-arxiv")
        initial = copy.deepcopy(plant.bb.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            train_task_route(cfg,plant,lambda:make_controller(cfg,plant,"schedule"),2,directory/"full","task",val_depths=[1,2])
            expected = copy.deepcopy(plant.bb.state_dict())
            plant.bb.load_state_dict(initial)
            train_task_route(cfg,plant,lambda:make_controller(cfg,plant,"schedule"),1,directory/"resume","task",val_depths=[1,2])
            plant.bb.load_state_dict(initial)
            result = train_task_route(cfg,plant,lambda:make_controller(cfg,plant,"schedule"),2,directory/"resume","task",val_depths=[1,2])
            self.assertEqual(result["resumed_from_epoch"],1)
            for key,value in plant.bb.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            cfg.lr *= 2
            with self.assertRaisesRegex(ValueError,"mismatch"):
                train_task_route(cfg,plant,lambda:make_controller(cfg,plant,"schedule"),2,directory/"resume","task",val_depths=[1,2])

    def test_diagnostic_budgets_cover_multiple_actual_graphs(self):
        cfg,plant,rs,base = setup_parts()
        pb=PolicyBundle("ropc",rs,envelope=Envelope((float("inf"),)))
        regret=library_regret(cfg,plant,{"base":base()},base(),base(),3,seed=3,T_cap=2)
        self.assertEqual(regret["base"]["n_states"],3)
        self.assertEqual(regret["base"]["n_distinct_graphs"],2)
        diagnostics=identification_diagnostics(cfg,plant,rs,base(),3,seed=2)
        self.assertEqual(sum(group["n"] for group in diagnostics.values()),3)
        gains=realized_gains(cfg,plant,pb,pb.envelope,3,seed=2)
        self.assertEqual(gains["n_pairs"],3)
        self.assertTrue(math.isfinite(gains["gain_max"]))
        self.assertTrue(set(gains["sample_ids"])<=set(plant.source.partitions["audit"]))


if __name__=="__main__":
    main()
