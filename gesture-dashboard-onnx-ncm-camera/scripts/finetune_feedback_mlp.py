"""Warm-fit the existing graph with supervised feedback and teacher replay.

The first layer and normalization remain fixed. Validation and old reviewed
records gate checkpoints. No test rows enter optimization or checkpoint choice.
"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,onnx
from onnx import numpy_helper
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from research.v18_20 import load_npz,predict_onnx
from scripts.train_feedback_refresh import feedback,from_logits,gated,preserves_metrics
import scripts.train_stop_inverted as trainer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'artifacts/feedback_refresh_20261111')
    parser.add_argument('--steps',type=int,default=1800)
    args=parser.parse_args();out=args.directory
    baseline=out/'baseline/gesture_mlp_production.onnx';original={v.name:numpy_helper.to_array(v).copy() for v in onnx.load(baseline).graph.initializer}
    _,reviewed,_=feedback(out/'feedback_snapshot.csv');audit=json.loads((out/'data_audit.json').read_text())
    isnew=np.isin(reviewed['source_id'],['feedback:'+s for s in audit['unused_ids']])
    data={'train':load_npz(ROOT/'artifacts/right_stability_20261111/train.npz'),'new':trainer.subset(reviewed,isnew),
          'old':trainer.subset(reviewed,~isnew),'val':load_npz(out/'val.npz')}
    h0={k:np.maximum(((d['X']-original['mean'])/original['scale'])@original['w0']+original['b0'],0) for k,d in data.items()}
    def logits(k,w):return np.maximum(h0[k]@w['w1']+w['b1'],0)@w['w2']+w['b2']
    teacher={k:softmax(logits(k,original),axis=1) for k in ('train','old')}
    before={k:gated(*predict_onnx(baseline,data[k]['X'])) for k in ('new','old','val')}
    truth={k:np.where(d['y']==6,13,d['y']) for k,d in data.items()}
    new_groups=[np.flatnonzero(data['new']['y']==label) for label in np.unique(data['new']['y'])]
    rng=np.random.default_rng(20261111);w={k:v.copy() for k,v in original.items()}
    names=('w1','b1','w2','b2');m={k:np.zeros_like(w[k]) for k in names};v={k:np.zeros_like(w[k]) for k in names}
    best=int(np.sum(before['new']==truth['new']));history=[];accepted=[]
    for step in range(1,args.steps+1):
        ti=rng.integers(len(h0['train']),size=256);oi=rng.integers(len(h0['old']),size=128)
        ni=np.asarray([rng.choice(new_groups[int(i)]) for i in rng.integers(len(new_groups),size=128)])
        x=np.concatenate([h0['train'][ti],h0['old'][oi],h0['new'][ni]])
        target=np.concatenate([teacher['train'][ti],teacher['old'][oi],np.eye(14,dtype=np.float32)[data['new']['y'][ni]]])
        weights=np.r_[np.full(256,2./256),np.full(128,6./128),np.full(128,1./128)].astype(np.float32)
        z=x@w['w1']+w['b1'];h=np.maximum(z,0);prob=softmax(h@w['w2']+w['b2'],axis=1)
        error=(prob-target)*weights[:,None]
        grad={'w2':h.T@error,'b2':error.sum(0)}
        dh=(error@w['w2'].T)*(z>0);grad.update(w1=x.T@dh,b1=dh.sum(0))
        rate=2e-5 if step<=1200 else 5e-6
        for k in names:
            grad[k]+=.0001*(w[k]-original[k])
            m[k]=.9*m[k]+.1*grad[k];v[k]=.999*v[k]+.001*grad[k]**2
            w[k]-=rate*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        if step%50:continue
        pred={k:gated(*from_logits(logits(k,w))) for k in ('new','old','val')}
        checks={k:preserves_metrics(truth[k],before[k],pred[k]) for k in ('old','val')}
        checks['new_correct_retained']=not np.any((before['new']==truth['new'])&(pred['new']!=truth['new']))
        correct=int(np.sum(pred['new']==truth['new']));row=dict(step=step,new_correct=correct,checks={k:bool(q) for k,q in checks.items()})
        history.append(row)
        if all(checks.values()) and correct>best:
            graph=onnx.load(baseline)
            for tensor in graph.graph.initializer:
                if tensor.name in names:tensor.CopyFrom(numpy_helper.from_array(w[tensor.name],tensor.name))
            candidate=out/'warm_candidate.onnx';onnx.checker.check_model(graph);onnx.save(graph,candidate)
            actual={k:gated(*predict_onnx(candidate,data[k]['X'])) for k in ('new','old','val')}
            if all(preserves_metrics(truth[k],before[k],actual[k]) for k in ('old','val')) and not np.any((before['new']==truth['new'])&(actual['new']!=truth['new'])):
                best=correct;np.savez_compressed(out/'warm_best_weights.npz',**w);accepted.append(row)
        if step%100==0:print(row,flush=True)
    if accepted:
        with np.load(out/'warm_best_weights.npz') as saved:best_weights={k:saved[k] for k in saved.files}
        graph=onnx.load(baseline)
        for tensor in graph.graph.initializer:
            if tensor.name in names:tensor.CopyFrom(numpy_helper.from_array(best_weights[tensor.name],tensor.name))
        onnx.save(graph,out/'warm_candidate.onnx')
    elif (out/'warm_candidate.onnx').exists():(out/'warm_candidate.onnx').unlink()
    report=dict(steps=args.steps,history=history,accepted_checkpoints=accepted,baseline_new_correct=int(np.sum(before['new']==truth['new'])),
                selected_new_correct=best,test_used_for_selection=False)
    trainer.write_json(out/'warm_validation.json',report);print({k:v for k,v in report.items() if k!='history'},flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
