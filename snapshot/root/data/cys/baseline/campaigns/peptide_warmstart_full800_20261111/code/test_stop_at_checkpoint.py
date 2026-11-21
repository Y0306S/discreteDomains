import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import stop_at_checkpoint as stop

class StopChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.policy=dict(stop_step=1500,stop_epoch=3,original_scheduler_steps=2500)
        self.checkpoint=self.root/'model/checkpoint-1500';self.checkpoint.mkdir(parents=True)
        self.statuses=[dict(rank=i,step=1499,first_update_verified=True,saved_weight_samples_match=True,
            latest_checkpoint=str(self.root/'model/checkpoint-1400')) for i in range(2)]
    def tearDown(self):self.tmp.cleanup()
    def save(self):
        for i,s in enumerate(self.statuses):(self.root/f'rank{i}_status.json').write_text(json.dumps(s))
        (self.checkpoint/'trainer_state.json').write_text(json.dumps(dict(epoch=3,global_step=1500)))
    def test_both_ranks_and_checkpoint_required(self):
        self.statuses[0].update(step=1500,latest_checkpoint=str(self.checkpoint));self.save()
        with patch.object(stop,'checkpoint_files',return_value={'shard.safetensors'}) as check:
            self.assertIsNone(stop.readiness(self.root,self.policy)[0]);check.assert_not_called()
            self.statuses[1].update(step=1500,latest_checkpoint=str(self.checkpoint));self.save()
            self.assertEqual(stop.readiness(self.root,self.policy)[0],self.checkpoint);check.assert_called_once_with(self.checkpoint,1500)
    def test_bad_checkpoint_never_stops(self):
        for s in self.statuses:s.update(step=1500,latest_checkpoint=str(self.checkpoint))
        self.save()
        with patch.object(stop,'checkpoint_files',side_effect=ValueError('missing rank optimizer')):
            with self.assertRaises(ValueError):stop.readiness(self.root,self.policy)
        (self.checkpoint/'trainer_state.json').write_text(json.dumps(dict(epoch=2.8)))
        with patch.object(stop,'checkpoint_files'):
            with self.assertRaisesRegex(ValueError,'epoch'):stop.readiness(self.root,self.policy)
    def test_missed_target_fails_visibly(self):
        self.statuses[0]['step']=1502;self.save()
        with self.assertRaisesRegex(RuntimeError,'missed'):stop.readiness(self.root,self.policy)
    def test_process_identity_prevents_signaling_reused_pid(self):
        expected=stop.process_identity(os.getpid());self.assertTrue(stop.alive(expected))
        with patch.object(stop.os,'kill') as kill:
            stop.send(dict(expected,start_ticks='wrong'),9);kill.assert_not_called()
    def test_export_preserves_restart_and_sets_three_epoch_evidence(self):
        (self.root/'evidence').mkdir();policy_file=self.root/'stop_policy.json';policy_file.write_text(json.dumps(self.policy))
        (self.checkpoint/'config.json').write_text('{}')
        (self.checkpoint/'model.safetensors.index.json').write_text('{"weight_map":{}}')
        (self.checkpoint/'model-1.safetensors').write_bytes(b'example shard')
        (self.checkpoint/'rng_state_0.pth').write_bytes(b'restart rng')
        with patch.object(stop,'ROOT',self.root),patch.object(stop,'POLICY',policy_file),\
             patch.object(stop,'checkpoint_files',return_value={'model-1.safetensors'}):
            stop.publish_model(self.checkpoint,self.policy,self.statuses)
        model=self.root/'model/model-1.safetensors'
        self.assertEqual(model.stat().st_ino,(self.checkpoint/'model-1.safetensors').stat().st_ino)
        self.assertTrue((self.checkpoint/'rng_state_0.pth').exists())
        evidence=json.loads((self.root/'evidence/final_weights.json').read_text())
        self.assertEqual((evidence['optimizer_steps'],evidence['epochs']),(1500,3))
        self.assertEqual(json.loads((self.root/'training_status.json').read_text())['status'],'completed')

if __name__=='__main__':unittest.main()
