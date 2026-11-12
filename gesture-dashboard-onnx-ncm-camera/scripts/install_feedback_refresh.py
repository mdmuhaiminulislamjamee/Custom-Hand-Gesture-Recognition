"""Install the preservation-qualified feedback candidate with rollback inputs."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from backend.artifact_integrity import ArtifactRegistry
from backend.config import CLASS_NAMES
from research.v18_20 import load_npz
import scripts.train_stop_inverted as trainer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out=ROOT/'artifacts/feedback_refresh_20261111';models=ROOT/'models';baseline=out/'baseline'
    report=json.loads((out/'evaluation.json').read_text());candidate=out/'candidate.onnx'
    if not report['qualified'] or not all(report['release_checks'].values()):raise ValueError('Offline qualification failed')
    if sha(candidate)!=report['model_sha256']:raise ValueError('Candidate hash mismatch')
    if sha(models/'gesture_mlp_production.onnx')!=report['audit']['baseline_model_sha256']:raise ValueError('Production changed since baseline capture')
    before=json.loads((out/'baseline_images.json').read_text());after=json.loads((out/'candidate_images.json').read_text())
    latency=json.loads((out/'onnx_latency.json').read_text())
    if after['model_sha256']!=sha(candidate) or before['selection_sha256']!=after['selection_sha256']:raise ValueError('Image replay inputs differ')
    image_ok=all(after['per_class'][k]['correct']>=v['correct'] and after['per_class'][k]['actions']>=v['actions'] for k,v in before['per_class'].items())
    if not image_ok or not latency['passed'] or latency['model_sha256']['candidate']!=sha(candidate):raise ValueError('Image/speed gate failed')
    checks={**report['release_checks'],'saved_image_holds_preserved':image_ok,'host_onnx_latency_preserved':True}
    test=report['splits']['test']['scalar']['candidate']
    runtime=report['splits']['test']['runtime']['candidate']
    sources=load_npz(ROOT/'data/ten_gesture/hagrid_train.npz')
    calls=sources['source_id'][sources['variant']=='source_call']
    heldout=load_npz(out/'test.npz')
    call_test=trainer.score(candidate,trainer.subset(heldout,np.isin(heldout['source_id'],calls)))
    report['retired_call_test']=call_test
    checks.update(offline_macro_f1_at_least_0_97=test['macro_f1']>=.97,
                  offline_accepted_precision_at_least_0_99=test['accepted_precision']>=.99,
                  offline_unknown_false_accept_at_most_0_02=test['unknown_false_accept_rate']<=.02,
                  old_command_correct_accept_at_least_0_98=test['old_command_correct_accept_rate']>=.98,
                  new_command_recall_at_least_0_97=test['stop_recall']>=.97,
                  retired_call_false_accept_at_most_0_02=call_test['unknown_false_accept_rate']<=.02,
                  runtime_peace_recall_at_least_0_95=runtime['per_class']['peace']['recall']>=.95,
                  runtime_rock_recall_at_least_0_97=runtime['per_class']['rock']['recall']>=.97,
                  runtime_unknown_false_accept_at_most_0_02=runtime['unknown_false_accept_rate']<=.02,
                  stop_precision=test['stop_precision']>=.97,
                  runtime_stop_precision=runtime['stop_precision']>=.97,
                  runtime_stop_recall=runtime['stop_recall']>=.95)
    if not all(checks.values()):raise ValueError(checks)
    metadata=json.loads((baseline/'gesture_mlp_onnx_metadata.json').read_text());config=json.loads((baseline/'gesture_mobile_runtime_config.json').read_text())
    lineage=metadata.pop('right_head_training',None)
    metadata.update(created_utc=datetime.now(timezone.utc).isoformat(),model_name='StaticFeedbackRefreshMLP',
        onnx_sha256=sha(candidate),onnx_bytes=candidate.stat().st_size,
        parity=dict(passed=True,prediction_agreement=report['parity']['label_agreement'],
                    maximum_absolute_probability_error=report['parity']['max_probability_error'],
                    maximum_absolute_mass_error=report['parity']['max_mass_error']),
        quality=dict(macro_f1=test['macro_f1'],accepted_precision=test['accepted_precision'],
                     unknown_false_acceptance_rate=test['unknown_false_accept_rate'],
                     old_command_correct_accept_rate=test['old_command_correct_accept_rate'],new_command_recall={'stop_inverted':test['stop_recall']}),
        feedback_refresh_training=dict(unused_records=report['audit']['unused_records'],class_counts=report['audit']['unused_class_counts'],
            feedback_sha256=report['audit']['feedback_sha256'],fitted_parameters=report['fitted_parameters'],
            encoder_and_unaffected_output_columns_frozen=True,graph_unchanged=True,selected_step=report['selection'],
            earlier_right_training=lineage,training_only_feedback=True,validation_and_test_never_used_for_gradient=True),
        qualification=dict(offline_passed=True,live_camera_validation_pending=True,checks=checks,data_audit=report['audit'],
            selection=dict(model_sha256=sha(candidate),**{k:config['temporal_smoothing'][k] for k in ('known_mass_floor','confidence_floor','probability_margin_floor')})),
        runtime_note='Preservation-constrained fine tuning using 990 unused reviewed landmark records; fixed regression benchmarks retained. Larger fits failed preservation gates and were rejected. Physical camera/device acceptance remains pending.')
    config['feedback_refresh_release']=dict(model_sha256=sha(candidate),evaluation='artifacts/feedback_refresh_20261111/evaluation.json',
        unused_records=report['audit']['unused_records'],offline_checks=checks,live_camera_validation_pending=True)
    if 'stop_inverted_release' in config:
        config['stop_inverted_release'].update(model_sha256=sha(candidate),evaluation='artifacts/feedback_refresh_20261111/evaluation.json',offline_checks=checks)
    stage=out/'deployment_stage';stage.mkdir(exist_ok=True)
    shutil.copy2(candidate,stage/'gesture_mlp_production.onnx')
    train=load_npz(out/'train.npz');indices=trainer.replay_indices(train)
    new_ids={'feedback:'+x for x in report['audit']['unused_ids']}
    fresh=np.flatnonzero(np.isin(train['source_id'],list(new_ids)))
    indices=np.unique(np.r_[indices,fresh]);replay=trainer.subset(train,indices)
    metadata['reviewed_feedback_replay_counts']={name:int(np.sum((replay['y']==i)&np.char.startswith(replay['source_id'].astype(str),'feedback:'))) for i,name in enumerate(trainer.LABELS)}
    replay['y']=np.where(np.isin(replay['y'],[6,13]),-1,replay['y'])
    np.savez_compressed(stage/'gesture_online_replay_cache.npz',**replay,class_names=np.asarray(CLASS_NAMES))
    trainer.write_json(stage/'gesture_mlp_onnx_metadata.json',metadata);trainer.write_json(stage/'gesture_mobile_runtime_config.json',config)
    modified=('gesture_mlp_production.onnx','gesture_mlp_onnx_metadata.json','gesture_mobile_runtime_config.json','gesture_online_replay_cache.npz')
    try:
        for name in modified:
            temporary=models/('.'+name+'.feedback_refresh.tmp');temporary.write_bytes((stage/name).read_bytes());temporary.replace(models/name)
        ArtifactRegistry(models).build()
    except Exception:
        for name in (*modified,'gesture_artifact_manifest.json'):shutil.copy2(baseline/name,models/name)
        raise
    report.update(deployed=True,release_checks=checks,image_replay=dict(baseline_correct=before['correct'],candidate_correct=after['correct'],
        baseline_actions=before['actions'],candidate_actions=after['actions'],rows=after['total']),
        acceptance_scope='Fixed regression benchmarks used for release acceptance. Reviewed data/image fits are training regressions, not independent generalization or physical camera accuracy.',
        rejected_alternatives=['unconstrained joint head fit','larger coordinate fit (scalar precision regression)','warm hidden-layer fit (validation/old-feedback regressions)'])
    trainer.write_json(out/'evaluation.json',report)
    print('Installed',sha(candidate),'replay rows',len(replay['y']),'all checks',all(checks.values()),flush=True)


if __name__=='__main__':main()
