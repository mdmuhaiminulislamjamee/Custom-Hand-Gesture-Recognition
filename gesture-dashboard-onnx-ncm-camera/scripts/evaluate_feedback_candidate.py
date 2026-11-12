"""Run fixed regression acceptance after validation-only candidate selection."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import onnx
from onnx import numpy_helper
from threadpoolctl import threadpool_limits
from research.v18_20 import load_npz,predict_onnx
from scripts.train_feedback_refresh import feedback,hidden,from_logits,gated
import scripts.train_stop_inverted as trainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/feedback_refresh_20261111')
    args=parser.parse_args();out=args.directory
    report=copy.deepcopy(json.loads((out/'evaluation.json').read_text()))
    candidate=out/'coordinate_candidate.onnx'
    original={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(out/'baseline/gesture_mlp_production.onnx').graph.initializer}
    updated={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(candidate).graph.initializer}
    selected=json.loads((out/'coordinate_validation.json').read_text())
    _,data,_=feedback(out/'feedback_snapshot.csv')
    is_new=np.isin(data['source_id'],['feedback:'+s for s in report['audit']['unused_ids']])
    splits={'val':load_npz(out/'val.npz'),'test':load_npz(out/'test.npz'),
        'old_feedback':trainer.subset(data,~is_new),'new_feedback':trainer.subset(data,is_new)}
    for split,rows in splits.items():
        for mode in ('scalar','runtime'):
            score=trainer.score(candidate,rows,runtime=mode=='runtime')
            report['splits'][split][mode]['candidate']=score
            print(split,mode,score['macro_f1'],flush=True)
    changed=[i for i in range(14) if not np.array_equal(original['w2'][:,i],updated['w2'][:,i]) or original['b2'][i]!=updated['b2'][i]]
    frozen=all(np.array_equal(original[k],updated[k]) for k in original if k not in ('w2','b2'))
    report.update(selection=selected['selected'],fitted_parameters=65*len(changed),changed_columns=[trainer.LABELS[i] for i in changed],
        model_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),model_bytes=candidate.stat().st_size,
        frozen_weights_identical=bool(frozen),deployed=False)
    p,m=predict_onnx(candidate,splits['val']['X'])
    expected,em=from_logits(hidden(splits['val']['X'],updated).astype(np.float64)@updated['w2']+updated['b2'])
    report['parity']=dict(label_agreement=float(np.mean(p.argmax(1)==expected.argmax(1))),
        max_probability_error=float(np.max(np.abs(p-expected))),max_mass_error=float(np.max(np.abs(m-em))))
    checks={'frozen_encoder_and_other_heads':bool(frozen),'same_graph':report['same_graph_operations'],
        'same_model_bytes':report['model_bytes']==report['baseline_bytes'],'onnx_parity':report['parity']['label_agreement']==1. and report['parity']['max_probability_error']<1e-4,
        'heldout_files_unchanged':report['release_checks']['heldout_files_unchanged']}
    for mode in ('scalar','runtime'):
        a,b=(report['splits']['test'][mode][k] for k in ('baseline','candidate'))
        checks[mode+'_test_per_class_preserved']=all(b['per_class'][name][key]+1e-12>=a['per_class'][name][key]
            for name in trainer.LABELS if name!='dorsal' for key in ('precision','recall','f1-score'))
        checks[mode+'_test_accepted_precision_preserved']=b['accepted_precision']+1e-12>=a['accepted_precision']
        checks[mode+'_test_unknown_false_accept_preserved']=b['unknown_false_accept_rate']<=a['unknown_false_accept_rate']+1e-12
    report['release_checks']=checks;report['qualified']=all(checks.values())
    trainer.write_json(out/'coordinate_evaluation.json',report)
    print('Qualification',checks,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
