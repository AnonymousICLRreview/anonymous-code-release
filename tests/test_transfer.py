"""P5 transfer checks using small frozen source fixtures and real artifact hashes."""
from __future__ import annotations

import copy
import json
import tempfile
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

import torch

from ropc import runner
from ropc.control import Envelope
from ropc.evaluation import PolicyBundle, policy_hash
from ropc.transfer import run_transfer
from tests.test_training_evaluation import setup_parts


class TransferTests(TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def source_run(self, root, **overrides):
        options=dict(L_max=64,L_star=32,bank="P0,Pg,Px,Pc",r_min=8,variants="ropc")
        options.update(overrides)
        cfg,plant,rs,_=setup_parts("LRIM-16",**options)
        env=Envelope((float("inf"),float("inf"),float("inf")))
        rs.envelope=env
        pb=PolicyBundle("ropc",rs,envelope=env,H=cfg.H)
        scientific={"dataset":"LRIM-16","dataset_slug":"lrim-16","preset":"full",
                    "synthetic":True,"seeds":[0],"split_index":0,"config":asdict(cfg),
                    "python_sources":runner.source_identity(),
                    "datasets":{"0":runner.dataset_identity(plant.graph)}}
        spec_hash=runner.object_hash(scientific)
        runner.write_json(root/"run_spec.json",scientific|{"run_spec_sha256":spec_hash,"created_utc":"test"})
        runner.write_json(root/"status.json",{"status":"partial","run_spec_sha256":spec_hash})
        base=runner.artifacts_dir(root,cfg)
        for directory in (base/"refit",base/"openloop",base/"policies"/"ropc",base/"audit"/"ropc"):
            directory.mkdir(parents=True,exist_ok=True)
        torch.save(plant.bb.state_dict(),base/"refit"/"backbone_frozen.pt")
        torch.save(plant.normalizer.state_dict(),base/"openloop"/"normalizer.pt")
        torch.save(rs.state_dict(),base/"policies"/"ropc"/"state.pt")
        runner.write_json(base/"policies"/"ropc"/"envelope.json",env.to_dict())
        digest=policy_hash(cfg,pb,env,plant.bb,plant)
        (base/"policies"/"ropc"/"hash.txt").write_text(digest+"\n")
        runner.write_json(base/"audit"/"ropc"/"audit_summary.json",
                          {"hash":digest,"audit":{"policy_pass":False},
                           "qualification_scope":{"sample_unit":"graph","decision_scope":"all_acting"}})
        tasks=runner.make_tasks(cfg,[0])
        for stage,task in (("refit","refit_s0"),("calibrate","calibrate_ropc_s0"),("audit","audit_ropc_s0")):
            directory=runner.task_directory(root,stage,tasks[stage].index(task),task)
            runner.write_json(directory/"summary.json",{"stage":stage,"status":"complete"})
            runner.write_json(directory/"complete.json",{"status":"complete","run_spec_sha256":spec_hash,
                "artifacts":runner.task_artifacts(root,cfg,stage,task,directory)})
        args=Namespace(source_run=str(root),seed=0,data_root="/unused",sizes="32,64,128,256",
                       output=str(root.parent/"transfer"),device="cpu",threads=1,synthetic=True)
        return cfg,plant,pb,args

    def test_frozen_policy_transfer_uses_fixed_depths_and_complete_test_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            cfg,plant,pb,args=self.source_run(root)
            original=copy.deepcopy(plant.bb.state_dict())
            seen=[]
            def metric(target,make_ctrl,depths,split,sens_seed):
                side=int(target.graph.name.split("-")[-1])
                seen.append((side,depths,split,len(target.source.partitions[split])))
                self.assertIs(target.bb,plant.bb)
                self.assertIs(target.normalizer,plant.normalizer)
                self.assertIs(target.sensor,plant.sensor)
                self.assertEqual(target.loss_bar,plant.loss_bar)
                self.assertEqual(target.cfg.L_max,64)
                self.assertEqual(depths,[2*side])
                self.assertEqual(split,"test")
                self.assertFalse(any(parameter.requires_grad for parameter in target.bb.parameters()))
                controller=make_ctrl()
                self.assertIs(controller.model,pb.rs.model)
                return {depths[0]:-side/100.}
            with patch("ropc.runner.load_bundle",return_value=(plant,plant.graph,plant.sup,plant.part,pb)) as load, \
                 patch("ropc.training.evaluate_depths",side_effect=metric), \
                 patch.object(plant.normalizer,"fit",side_effect=AssertionError("target normalization fitting")):
                result=run_transfer(args)
            load.assert_called_once()
            self.assertEqual([row[1] for row in seen],[[64],[128],[256],[512]])
            self.assertTrue(all(row[3]==8 for row in seen))
            self.assertEqual(result["source_run_status"],"partial")
            self.assertFalse(result["source_audit_guarantee_transfers"])
            self.assertFalse(result["target_policy_audited"])
            self.assertFalse(any(result["adaptation"].values()))
            self.assertEqual(result["source_qualification"]["deployment_status"],"uncertified")
            for name,value in plant.bb.state_dict().items():
                self.assertTrue(torch.equal(value,original[name]))
            report=json.loads((Path(args.output)/"transfer.json").read_text())
            self.assertEqual(len(report["results"]),4)
            self.assertTrue(report["synthetic"])
            self.assertTrue(all(row["target_fingerprint"] for row in report["results"]))
            self.assertTrue(report["frozen_state_unchanged"])

    def test_primary_lrim16_depth_ceiling_cannot_masquerade_as_p5(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            _,_,_,args=self.source_run(root,L_max=32)
            with patch("ropc.runner.load_bundle") as load, self.assertRaisesRegex(ValueError,"L_max=64"):
                run_transfer(args)
            load.assert_not_called()

    def test_selected_fallback_keeps_its_audited_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            cfg,plant,pb,args=self.source_run(root)
            base=runner.artifacts_dir(root,cfg)
            policy_dir=base/"policies"/"ropc"
            fallback=Envelope((float("inf"),float("inf"),float("inf")))
            fallback_hash=policy_hash(cfg,pb,fallback,plant.bb,plant)
            runner.write_json(policy_dir/"envelope_fallback.json",fallback.to_dict())
            (policy_dir/"fallback_hash.txt").write_text(fallback_hash+"\n")
            audit_path=base/"audit"/"ropc"/"audit_summary.json"
            audit=json.loads(audit_path.read_text())
            audit["fallback"]={"hash":fallback_hash,"audit":{"policy_pass":False}}
            runner.write_json(audit_path,audit)
            audit_result=runner.task_directory(root,"audit",0,"audit_ropc_s0")
            marker=json.loads((audit_result/"complete.json").read_text())
            marker["artifacts"]=runner.task_artifacts(root,cfg,"audit","audit_ropc_s0",audit_result)
            runner.write_json(audit_result/"complete.json",marker)
            args.sizes="32"
            with patch("ropc.runner.load_bundle",return_value=(plant,plant.graph,plant.sup,plant.part,pb)), \
                 patch("ropc.training.evaluate_depths",return_value={64:-1.}):
                result=run_transfer(args)
            self.assertEqual(result["selected_source_envelope"],"envelope_fallback.json")
            self.assertEqual(result["source_policy_sha256"],fallback_hash)
            self.assertEqual(result["source_qualification"]["evaluated_policy"],"fallback")
            self.assertIn("ARTIFACTS/seed0/policies/ropc/fallback_hash.txt",result["source_artifact_sha256"])

    def test_synthetic_and_official_modes_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            _,_,_,args=self.source_run(root)
            args.synthetic=False
            with self.assertRaisesRegex(ValueError,"cannot be mixed"):
                run_transfer(args)

    def test_source_task_artifacts_are_verified_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            cfg,_,_,args=self.source_run(root)
            (runner.artifacts_dir(root,cfg)/"refit"/"backbone_frozen.pt").write_bytes(b"tampered")
            with patch("ropc.runner.load_bundle") as load, self.assertRaisesRegex(ValueError,"artifact changed"):
                run_transfer(args)
            load.assert_not_called()

    def test_source_audit_must_be_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            _,_,_,args=self.source_run(root)
            marker=runner.task_directory(root,"audit",0,"audit_ropc_s0")/"complete.json"
            marker.unlink()
            with self.assertRaisesRegex(ValueError,"completed source task: audit"):
                run_transfer(args)

    def test_mutating_a_weight_during_transfer_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            _,plant,pb,args=self.source_run(root)
            args.sizes="32"
            def mutated(target,*positional,**kwargs):
                target.bb.readout.weight.add_(.1)
                return {64:-1.}
            with patch("ropc.runner.load_bundle",return_value=(plant,plant.graph,plant.sup,plant.part,pb)), \
                 patch("ropc.training.evaluate_depths",side_effect=mutated), \
                 self.assertRaisesRegex(RuntimeError,"Frozen source state changed"):
                run_transfer(args)
            self.assertFalse((Path(args.output)/"transfer.json").exists())

    def test_duplicate_or_unknown_sizes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"source";root.mkdir()
            _,_,_,args=self.source_run(root)
            for sizes in ("32,32","16","512",""):
                with self.subTest(sizes=sizes),self.assertRaises(ValueError):
                    args.sizes=sizes
                    run_transfer(args)


if __name__=="__main__":
    main()
