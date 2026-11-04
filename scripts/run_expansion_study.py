"""Matched excerpt study with frozen upstream predictions and explicit seeds.

The input manifest is private and contains audio paths. Public extracts exclude
recordings, feature matrices, titles and machine paths. No production defaults
or model files are changed.
"""
import argparse
from contextlib import ExitStack
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch
import wave

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.review_real_song import packaged_chart,sha


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();selection=read(args.selection)
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from keyloom_mania.studio import pipeline
    from keyloom_mania.features import AcousticConfig,decode_waveform,logmel_features
    from keyloom_mania.local_rhythm import rhythm_descriptors
    from keyloom_mania.music_features import build_encoder,extract_embeddings,MusicFeatureConfig
    from keyloom_mania.generator_density import ExecutionLedger
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    protocol={k:selection[k] for k in ['selection_rule','bars_per_excerpt','seeds','variants','training_overlap','rights','target_star','failure_policy']}
    protocol.update(scope='Two known-development excerpts, two seeds, three variants. No population or unseen-song claim.',
        upstream_policy='Cache first planner output, arrangement outputs and density-adapter output per track; deep-copy identical values into every later arm and seed.',
        seed_policy='Global Python/NumPy/Torch/CUDA seeds and generator count_seed set to the declared seed. Effective policy records the override.',
        runner_sha256=sha(Path(__file__)),catalog_sha256=sha(ROOT/'keyloom_mania/resources/catalog.json'),
        source_sha256={f:sha(ROOT/f) for f in ['keyloom_mania/studio/pipeline.py','keyloom_mania/hierarchical_planner.py','keyloom_mania/generator_density.py']},
        tracks=[{'track_id':f'track-{i+1}','source_sha256':x['source_sha256'],'bpm':x['bpm'],'offset_ms':x['offset_ms']} for i,x in enumerate(selection['selected'])])
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    all_results=[]
    for index,item in enumerate(selection['selected']):
        track_id=f'track-{index+1}';track_dir=args.output/track_id;track_dir.mkdir()
        source=Path(item['source'])
        if sha(source)!=item['source_sha256']:raise AssertionError('Selected source changed')
        waveform,_=decode_waveform(source,AcousticConfig())
        duration_ms=item['offset_ms']+selection['bars_per_excerpt']*4*60000/item['bpm']
        waveform=waveform[:int(round(duration_ms*24))]
        excerpt=track_dir/'excerpt.wav'
        pcm=(np.clip(waveform,-1,1)*32767).astype('<i2')
        with wave.open(str(excerpt),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(24000);f.writeframes(pcm.tobytes())
        waveform,_=decode_waveform(excerpt,AcousticConfig())
        print(json.dumps({'track':track_id,'stage':'features','seconds':len(waveform)/24000}),flush=True)
        acoustic=logmel_features(waveform,AcousticConfig(),device='cuda');rhythm=rhythm_descriptors(acoustic)
        encoder,_=build_encoder(pipeline.DEFAULT_MUSICFM_ASSETS,device='cuda')
        music,_=extract_embeddings(encoder,waveform,MusicFeatureConfig(),device='cuda')
        del encoder,waveform,pcm;torch.cuda.empty_cache()
        np.savez_compressed(track_dir/'features.npz',acoustic=acoustic,music=music)
        np.save(track_dir/'rhythm.npy',rhythm)
        protocol['tracks'][index].update(excerpt_sha256=sha(excerpt),feature_sha256=sha(track_dir/'features.npz'),rhythm_sha256=sha(track_dir/'rhythm.npy'))
        (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        cache={};ledgers=[]
        original_forward=pipeline.ArrangementPlanner.forward
        original_arrangement=pipeline.infer_arrangement;original_density=pipeline.infer_density
        original_loader=pipeline._generated_plan_model;original_shape=pipeline.shape_hierarchical_density
        def frozen_forward(model,*values,**kwargs):
            if 'planner' not in cache:
                output=original_forward(model,*values,**kwargs)
                cache['planner']={k:v.detach().cpu().clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in output.items()}
            return {k:v.to(values[0].device).clone() if isinstance(v,torch.Tensor) else copy.deepcopy(v) for k,v in cache['planner'].items()}
        def freeze_numpy(key,fn):
            def call(*a,**kw):
                if key not in cache:cache[key]=copy.deepcopy(fn(*a,**kw))
                return copy.deepcopy(cache[key])
            return call
        class CapturingLedger(ExecutionLedger):
            def __init__(self):super().__init__();ledgers.append(self)
        def identity_shape(density,structure):
            _,report=original_shape(density,structure)
            report.update(multipliers=[1.0]*len(density),changed_bars=0,shaped_note_budget_sum=report['original_note_budget_sum'],
                shaped_row_budget_sum=report['original_row_budget_sum'],shaped_beat_density=np.asarray(density).tolist(),research_ablation='identity_shape')
            return np.asarray(density).copy(),report
        for seed in selection['seeds']:
            seed_dir=track_dir/f'seed-{seed}';seed_dir.mkdir()
            settings={'bpm':item['bpm'],'first_beat_ms':item['offset_ms'],'target_star':selection['target_star'],
                'main_style':'Jumpstream','secondary_styles':['Stream','Rolls'],'model_id':'music-base',
                'arrangement_model_id':'standard','density_model_id':'contrast','coordination_id':'coordinated','budget_id':'legacy',
                'title':track_id,'artist':'Development excerpt, not redistributed'}
            seed_protocol={'settings':settings,'seed':seed,'variants':selection['variants'],'scope':protocol['scope'],
                'source_sha256':sha(excerpt),'upstream_policy':protocol['upstream_policy'],'decoder_seed_override':seed}
            (seed_dir/'protocol.json').write_text(json.dumps(seed_protocol,indent=2),encoding='utf-8')
            results={}
            def seeded_loader(*a,**kw):
                model=original_loader(*a,**kw)
                model.count_seed=seed
                model.checkpoint_inference_policy=copy.deepcopy(model.checkpoint_inference_policy)
                model.checkpoint_inference_policy.update(count_seed=seed,study_seed_override=seed)
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                return model
            for variant in selection['variants']:
                folder=seed_dir/variant;folder.mkdir();audio=folder/'source.wav';shutil_copy(excerpt,audio)
                requested={**settings,'structure_id':'legacy' if variant=='legacy' else 'hierarchical'}
                def progress(stage,fraction,message,**extra):
                    if stage!='generation' or int(fraction*100)%10==0:
                        print(json.dumps({'track':track_id,'seed':seed,'variant':variant,'stage':stage,'progress':round(fraction,3)}),flush=True)
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                started=time.perf_counter()
                try:
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(pipeline,'logmel_features',return_value=acoustic))
                        stack.enter_context(patch('keyloom_mania.local_rhythm.rhythm_descriptors',return_value=rhythm))
                        stack.enter_context(patch('keyloom_mania.music_features.build_encoder',return_value=(None,{})))
                        stack.enter_context(patch('keyloom_mania.music_features.extract_embeddings',return_value=(music,{'audio_seconds_per_inference_second':0.0})))
                        stack.enter_context(patch.object(pipeline.ArrangementPlanner,'forward',new=frozen_forward))
                        stack.enter_context(patch.object(pipeline,'infer_arrangement',side_effect=freeze_numpy('arrangement',original_arrangement)))
                        stack.enter_context(patch.object(pipeline,'infer_density',side_effect=freeze_numpy('density',original_density)))
                        stack.enter_context(patch.object(pipeline,'_generated_plan_model',side_effect=seeded_loader))
                        stack.enter_context(patch.object(pipeline,'ExecutionLedger',CapturingLedger))
                        if variant=='hierarchy-only':stack.enter_context(patch.object(pipeline,'shape_hierarchical_density',side_effect=identity_shape))
                        manifest=pipeline._generate_song(audio,requested,progress,device='cuda')
                    packaged_chart(folder,manifest['package_sha256'])
                    (folder/'committed-ledger.json').write_text(json.dumps(sorted([t,*v] for t,v in ledgers[-1].events.items())),encoding='utf-8')
                    result={'status':'complete','notes':manifest['notes'],'rows':manifest['rows'],'actual_star':manifest['actual_star'],
                        'elapsed_seconds':time.perf_counter()-started}
                except Exception as error:
                    result={'status':'failed','error_type':type(error).__name__,'error':str(error),'elapsed_seconds':time.perf_counter()-started}
                results[variant]=result;all_results.append({'track':track_id,'seed':seed,'variant':variant,**result})
                (seed_dir/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
                (args.output/'execution-results.json').write_text(json.dumps(all_results,indent=2),encoding='utf-8')
                print(json.dumps({'track':track_id,'seed':seed,'variant':variant,'result':result}),flush=True)
        # The equality audit covers inferred scores as well as raw pooled inputs.
        with np.load(track_dir/'seed-42/legacy/planner-inputs.npz') as reference:
            for seed in selection['seeds']:
                for variant in selection['variants']:
                    with np.load(track_dir/f'seed-{seed}'/variant/'planner-inputs.npz') as other:
                        for key in ['music','acoustic','rhythm','base_density','unshaped_beat_density','owner_scores','style_evidence']:
                            np.testing.assert_array_equal(reference[key],other[key])
        (track_dir/'frozen-upstream-check.json').write_text(json.dumps({'all_cached_and_inferred_arrays_bitwise_equal':True,'across_seeds':selection['seeds']}),encoding='utf-8')
    if any(r['status']!='complete' for r in all_results):raise RuntimeError('Retained one or more failed arms')
    print('Completed all expanded frozen-upstream arms.',flush=True)


def shutil_copy(source,target):
    import shutil
    shutil.copyfile(source,target)


if __name__=='__main__':main()
