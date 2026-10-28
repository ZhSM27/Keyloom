from tests.helpers import portable_generator
from unittest.mock import patch
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from keyloom_mania.density_coordination import coordinate_onsets,style_density_cost,density_execution_trace,validate_coordination
from keyloom_mania.generation_studio import StudioApp
from keyloom_mania.role_style_plan import RoleStylePolicy,active_style_plan
from keyloom_mania.segment_labels import LABELS


class DensityCoordinationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_bounded_calibration_preserves_music_order_and_impossible_positions(self):
        p=torch.linspace(.01,.95,24)[None]
        p[0,0]=0
        beats=torch.arange(24)[None]/24
        density=torch.tensor([5.,4.]).expand(1,24,2)
        result,stats=coordinate_onsets(p,density,beats,torch.empty(1,0,dtype=torch.long))
        self.assertEqual(result[0,0],0)
        self.assertTrue((result[:,2:]>=result[:,1:-1]).all())
        self.assertLess(abs(result.sum().item()-4),abs(p.sum().item()-4))
        delta=torch.logit(result[:,1:])-torch.logit(p[:,1:])
        self.assertLessEqual(delta.abs().max().item(),1.50001)
        self.assertFalse(stats['hard_row_budget'])

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_no_forced_onsets_from_near_silence_or_partial_beats(self):
        p=torch.full((1,24),.001)
        density=torch.tensor([8.,4.]).expand(1,24,2)
        result,stats=coordinate_onsets(p,density,torch.arange(24)[None]/24,torch.empty(1,0,dtype=torch.long))
        self.assertFalse((result>=.5).any())
        self.assertEqual(stats['bounded_beats'],1)
        partial,_=coordinate_onsets(p[:,:12],density[:,:12],torch.arange(12)[None]/24,torch.empty(1,0,dtype=torch.long))
        torch.testing.assert_close(partial,p[:,:12],rtol=0,atol=0)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_prefix_and_other_songs_cannot_change_new_beat_calibration(self):
        p=torch.full((2,48),.2)
        beats=torch.arange(48)[None].expand(2,-1)/24
        density=torch.tensor([4.,3.]).expand(2,48,2).clone()
        prefix=torch.zeros(2,24,dtype=torch.long)
        a,_=coordinate_onsets(p,density,beats,prefix)
        density[1,:,1]=10
        prefix[1]=3
        b,_=coordinate_onsets(p,density,beats,prefix)
        torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
        torch.testing.assert_close(a[:,:24],p[:,:24],rtol=0,atol=0)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_malformed_policy_density_and_bounds_are_rejected(self):
        for value in (None,[],False,'unknown'):
            with self.assertRaises(ValueError):validate_coordination(value)
        p=torch.ones(1,24)*.5;d=torch.ones(1,24,2);b=torch.arange(24)[None]/24;prefix=torch.empty(1,0,dtype=torch.long)
        for bad in (None,d[:,:12],d*float('nan'),-d):
            with self.assertRaises(ValueError):coordinate_onsets(p,bad,b,prefix)
        with self.assertRaises(ValueError):coordinate_onsets(p,d,b,prefix,max_logit_shift=-1)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_cost_is_energy_sensitive_and_does_not_invent_unsupported_styles(self):
        notes=np.full(12,4.5);rows=np.full(12,3.);quiet=np.zeros(12,dtype=bool)
        rhythm=np.zeros((12,40));rhythm[4:8,32:36]=5
        cost=style_density_cost(notes,rows,quiet,rhythm)
        self.assertGreater(cost[5,LABELS.index('Stream')],cost[0,LABELS.index('Stream')])
        self.assertEqual(cost[:,LABELS.index('Jumpstream')].sum(),0)
        p=np.full((12,15),.01);p[:,LABELS.index('Jumpstream')]=.4;p[:,LABELS.index('Stream')]=.6
        plan=active_style_plan(p,notes,rows,np.ones(15),{},'Jumpstream',['Stream'],rhythm=rhythm,
            policy=RoleStylePolicy(energy_density_weight=2.),owner_scores=p,style_evidence=p)
        self.assertTrue((plan['style_ids'][4:8]==LABELS.index('Jumpstream')+1).all())
        self.assertFalse(plan['patterns'][:,LABELS.index('Handstream')].any())
        self.assertTrue((plan['style_ids'][:4]==LABELS.index('Stream')+1).all())

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_single_note_target_keeps_lower_type_and_directional_contract(self):
        p=np.full((16,15),.01);p[:,LABELS.index('Handstream')]=.7;p[4:8,LABELS.index('Jumpstream')]=.99
        rhythm=np.zeros((16,40));rhythm[4:8,32:36]=5
        plan=active_style_plan(p,np.full(16,4.),np.full(16,3.),np.ones(15),{},'Handstream',
            rhythm=rhythm,policy=RoleStylePolicy(energy_density_weight=2.))
        self.assertTrue((plan['style_ids'][4:8]==LABELS.index('Jumpstream')+1).all())
        reverse=active_style_plan(p,np.full(16,4.),np.full(16,3.),np.ones(15),{},'Jumpstream',
            rhythm=rhythm,policy=RoleStylePolicy(energy_density_weight=2.))
        self.assertFalse(reverse['patterns'][:,LABELS.index('Handstream')].any())

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_final_density_reference_does_not_change_silence_or_budgets(self):
        p=np.ones((4,15))*.2;n=np.array([0,4,4,4.]);r=np.array([0,3,3,3.]);
        plan=active_style_plan(p,n,r,np.ones(15),{},'Jumpstream',policy=RoleStylePolicy(energy_density_weight=2.),density_reference=np.ones((4,2)))
        self.assertTrue(plan['quiet'][0]);self.assertFalse(plan['patterns'][0].any())
        np.testing.assert_array_equal(plan['target_note_budget'],n)
        with self.assertRaises(ValueError):active_style_plan(p,n,r,np.ones(15),{},'Jumpstream',policy=RoleStylePolicy(energy_density_weight=2.),density_reference=np.zeros((3,2)))

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_rounded_export_edges_do_not_lose_first_onsets(self):
        bars=[{'start_ms':216.3,'end_ms':1216.3,'quiet':False}]
        result=density_execution_trace(np.ones((1,4,2)),bars,[216,466,716,966],np.ones(1),250)
        self.assertEqual(result['bars'][0]['actual_beat_density'],[[1,1]]*4)
        self.assertEqual(result['summary']['notes_beat_mae'],0)

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_queue_persists_coordination_and_rejects_invalid_before_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source.wav';source.write_bytes(b'fixture');app=StudioApp(root,0);job='a'*32
            app.jobs[job]={'directory':str(root),'source_path':str(source),'filename':'source.wav','status':'ready'}
            with patch('keyloom_mania.generation_studio.threading.Thread') as worker:
                app.generate(job,{'main_style':'Jumpstream','coordination_id':'coordinated'})
                worker.return_value.start.assert_called_once()
            self.assertEqual(json.loads((root/'status.json').read_text(encoding='utf-8'))['settings']['coordination_id'],'coordinated')
            app.jobs[job]['status']='ready'
            with patch('keyloom_mania.generation_studio.threading.Thread') as worker:
                app.generation_lock.release()
                with self.assertRaises(ValueError):app.generate(job,{'coordination_id':'bad'})
                worker.assert_not_called()

    @patch('keyloom_mania.generation_studio.resolve_generator_model', new=portable_generator)
    def test_expression_reference_branch_keeps_its_count_contract_after_calibration(self):
        from keyloom_mania.expression_generator import ExpressionGenerator
        from keyloom_mania.tiny_model import TinyModelConfig
        from keyloom_mania.playstyle_control import studio_playstyle_controls
        from keyloom_mania.motion_state import candidate_grid
        torch.manual_seed(81)
        model=ExpressionGenerator(TinyModelConfig(width=12,heads=3,audio_layers=1,decoder_layers=1,feedforward=24),reference_count_fallback=True).eval()
        model.density_coordination=True
        inputs={'acoustic':torch.zeros(1,24,128),'music':torch.zeros(1,24,1024),'phase':torch.zeros(1,24,5),
            'local':torch.zeros(1,24,360),'condition':torch.zeros(1,5),'plan':torch.zeros(1,24,15),'density':torch.ones(1,24,2)}
        inputs['plan'][...,LABELS.index('Stream')]=1
        lane,phrase=studio_playstyle_controls('cpu',{'Stream'})
        before=torch.get_rng_state().clone()
        output=model.generate(**inputs,candidate_beats=candidate_grid(1,24,'cpu'),lane_state=lane,phrase_state=phrase,fast_empty=True)
        self.assertEqual(output['phrase_expression']['protected_count_changes'],0)
        self.assertEqual(output['density_alignment']['calibrated_beats'],1)
        torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)
        torch.testing.assert_close(output['count_probability'].sum(-1),torch.ones(1,24))


if __name__=='__main__':unittest.main()
