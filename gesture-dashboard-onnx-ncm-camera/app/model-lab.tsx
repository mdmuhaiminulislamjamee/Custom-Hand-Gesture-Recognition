'use client';

import { useCallback, useEffect, useState } from 'react';
import { LabReport, labLabel, labRequest } from './lab-report';

type JobKind = 'audit' | 'train' | 'evaluate';
type Job = {
  id: string;
  kind: JobKind;
  status: string;
  created_at?: string;
  created_at_utc?: string;
  logs?: string[];
  result?: Record<string, unknown> | null;
  error?: string | null;
  candidate_available?: boolean;
};
type Status = { available: boolean; dataset_root?: string; jobs: Job[]; class_names?: string[]; capabilities?: Record<string, unknown>; missing_dependencies?: string[] };
const inProgress = (job: Job) => job.status === 'queued' || job.status === 'running';

export default function ModelLab({ api, active }: { api: string; active: boolean }) {
  const [status, setStatus] = useState<Status | null>(null);
  const [selected, setSelected] = useState('');
  const [candidate, setCandidate] = useState('');
  const [epochs, setEpochs] = useState(50);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const refresh = useCallback(async () => {
    const result = await labRequest<Status>(`${api}/api/model-lab/status`);
    setStatus(result);
    setSelected(current => current || result.jobs[0]?.id || '');
    setCandidate(current => current || result.jobs.find(job => job.candidate_available && job.status === 'succeeded')?.id || '');
  }, [api]);

  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    const load = async () => { try { await refresh(); if (!cancelled) setError(''); } catch (cause) { if (!cancelled) setError(cause instanceof Error ? cause.message : 'Could not load training jobs.'); } };
    void load();
    const timer = setInterval(() => { void load(); }, 3000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [active, refresh]);

  const jobs = status?.jobs ?? [];
  const current = jobs.find(job => job.id === selected);
  const candidates = jobs.filter(job => job.candidate_available && job.status === 'succeeded');
  const running = jobs.some(inProgress);
  const dependencyStatus = status?.capabilities?.dependencies;
  const missingDependencies = status?.missing_dependencies ?? (dependencyStatus && typeof dependencyStatus === 'object' ? Object.entries(dependencyStatus).filter(([, installed]) => !installed).map(([name]) => name) : []);

  async function start(kind: JobKind) {
    setBusy(true); setError(''); setNotice('');
    try {
      const result = await labRequest<Job & { job_id?: string }>(`${api}/api/model-lab/jobs`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ kind, ...(kind === 'train' ? { epochs } : {}), ...(kind === 'evaluate' ? { candidate_job_id: candidate } : {}) }),
      });
      const id = result.job_id ?? result.id;
      setSelected(id);
      setNotice(`${labLabel(kind)} queued. You can follow its progress below.`);
      await refresh();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not start this job.'); }
    finally { setBusy(false); }
  }

  async function cancel(job: Job) {
    setBusy(true); setError('');
    try { await labRequest(`${api}/api/model-lab/jobs/${encodeURIComponent(job.id)}/cancel`, { method: 'POST' }); await refresh(); setNotice('Cancellation requested.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not cancel the job.'); }
    finally { setBusy(false); }
  }

  return <div className="lab-page" hidden={!active}>
    <div className="section-heading"><div><p className="eyebrow">07 · REVIEWED DATA TO CANDIDATE MODEL</p><h2>Model Training</h2><p className="lab-intro">Audit your collection, train a candidate, and inspect held-out results before choosing a release.</p></div><button type="button" className="secondary-button" disabled={busy} onClick={() => { void refresh().catch(cause => setError(cause.message)); }}>Refresh</button></div>
    <div className="lab-policy"><strong>Train reviewed hand shapes and check them in Video Lab.</strong><p>Palm and Stop inverted accept the upper XY semicircle from 0° through −90° to −180°, with either hand and fingers together or spread. Stop inverted returns to the main position. A candidate download does not replace the active model.</p></div>
    {error && <div role="alert" className="lab-error">{error}</div>}
    {status && !status.available && <div className="lab-error" role="status">Training dependencies are unavailable{missingDependencies.length ? `: ${missingDependencies.join(', ')}` : ''}. Install the packages in requirements-notebook.txt in the backend environment, then refresh.</div>}
    {notice && <p role="status" className="lab-notice">{notice}</p>}
    <div className="lab-workflow-cards">
      <section className="panel lab-workflow-card"><span className="lab-step">01</span><h3>Audit the dataset</h3><p>Check labels, hand features, participant splits, and excluded context examples. Resolve dataset errors before training.</p><button className="secondary-button" type="button" disabled={!status || !status.available || busy || running} onClick={() => { void start('audit'); }}>Run dataset audit</button></section>
      <section className="panel lab-workflow-card"><span className="lab-step">02</span><h3>Train a candidate</h3><p>Train an isolated twelve-command candidate from the reviewed collection. Participant-separated data measures generalization.</p><label className="lab-field">Epochs<input type="number" min={1} max={500} step={1} value={epochs} onChange={event => setEpochs(Number(event.target.value))} /></label><button className="primary-button" type="button" disabled={!status || !status.available || busy || running || !Number.isInteger(epochs) || epochs < 1 || epochs > 500} onClick={() => { void start('train'); }}>Train candidate</button></section>
      <section className="panel lab-workflow-card"><span className="lab-step">03</span><h3>Evaluate a candidate</h3><p>Compare the candidate and production model on frozen public and camera test snapshots. Complete continuous video testing before deployment.</p><label className="lab-field">Candidate<select value={candidate} onChange={event => setCandidate(event.target.value)}><option value="">Choose a trained candidate</option>{candidates.map(job => <option key={job.id} value={job.id}>{job.id}</option>)}</select></label><button className="secondary-button" type="button" disabled={!candidate || busy || running} onClick={() => { void start('evaluate'); }}>Evaluate candidate</button></section>
    </div>
    <div className="lab-storage"><span>Collection folder</span><code>{status?.dataset_root ?? 'Waiting for the local server'}</code><small>{status?.class_names?.filter(label => label !== 'no_gesture').length ?? 12} commands + no gesture rejection · context and detector data retained separately</small></div>
    <div className="lab-grid lab-job-layout">
      <section className="panel"><div className="panel-header"><div>JOBS</div><span>{jobs.length} RECORDED</span></div><div className="lab-list">{jobs.length ? jobs.map(job => <button type="button" key={job.id} className={selected === job.id ? 'selected' : ''} onClick={() => setSelected(job.id)}><strong>{labLabel(job.kind)}</strong><span className={`lab-badge lab-status-${job.status}`}>{labLabel(job.status)}</span><small>{job.id}</small><small>{job.created_at || job.created_at_utc ? new Date(job.created_at ?? job.created_at_utc!).toLocaleString() : ''}</small></button>) : <p className="empty-note">No jobs yet. Start with a dataset audit.</p>}</div></section>
      <section className="panel lab-job-detail"><div className="panel-header"><div>{current ? `${labLabel(current.kind).toUpperCase()} · ${labLabel(current.status).toUpperCase()}` : 'JOB DETAILS'}</div>{current && inProgress(current) && <button type="button" className="lab-text-button" disabled={busy} onClick={() => { void cancel(current); }}>Cancel job</button>}</div>{current ? <div className="lab-body">
        <code className="lab-job-id">{current.id}</code>
        {current.error && <div role="alert" className="lab-error">{current.error}</div>}
        {inProgress(current) && <p role="status" className="lab-hint">This job is {current.status}. Progress refreshes automatically.</p>}
        {!!current.logs?.length && <details open={inProgress(current)}><summary>Job log</summary><pre className="lab-log">{current.logs.join('\n')}</pre></details>}
        {current.result && <LabReport result={current.result} title={current.kind === 'audit' ? 'Dataset audit' : 'Evaluation results'} />}
        {current.candidate_available && <a className="primary-button lab-download" href={`${api}/api/model-lab/jobs/${encodeURIComponent(current.id)}/candidate`}>Download candidate ONNX</a>}
      </div> : <p className="empty-note">Choose a job to inspect its results and log.</p>}</section>
    </div>
  </div>;
}
