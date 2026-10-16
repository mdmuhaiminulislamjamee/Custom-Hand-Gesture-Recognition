'use client';
/* eslint-disable @next/next/no-img-element */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { CameraSource } from './data-collection';
import { LabReport, labLabel, labRequest } from './lab-report';

type TrainingUse = 'hand_classifier' | 'context_only' | 'detector_only' | 'excluded';
type Segment = { start: number; end: number; label: string; negative_reason: string; training_use: TrainingUse; handedness?: string; finger_orientation?: string };
type Scenario = { id: string; title: string; instruction?: string; training_use: TrainingUse };
type Clip = { id: string; participant_id: string; session_id: string; source: string; scenario: string; duration_seconds: number; width?: number; height?: number; fps?: number; reviewed: boolean; segments: Segment[]; extraction?: Record<string, unknown> | null; reports?: Record<string, unknown>[]; note?: string };
type Root = { clips: Clip[]; classes: string[]; scenarios: Scenario[]; max_upload_bytes?: number; max_duration_seconds?: number };
type Job = { id: string; job_id?: string; kind: string; status: string; progress?: number | Record<string, unknown>; result?: Record<string, unknown>; error?: string };
type Metadata = { participant_id: string; session_id: string; source: string; scenario: string; lighting: string; distance: string; note: string };
type PendingMedia = { blob: Blob; filename: string; url: string; metadata?: Metadata; marks: number[] };
type Props = { api: string; active: boolean; ready: boolean; connected: boolean; cameraSource: CameraSource; webcamStream: MediaStream | null; streamRevision: number; setCameraSource: (source: CameraSource) => void; toggleCamera: () => void };
const initialDraft: Segment = { start: 0, end: 1, label: 'no_gesture', negative_reason: 'standing_arm_hanging', training_use: 'context_only', handedness: 'right', finger_orientation: 'toward_camera' };
const formatTime = (seconds: number) => `${Math.floor(seconds / 60)}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;

export default function VideoLab({ api, active, ready, connected, cameraSource, webcamStream, streamRevision, setCameraSource, toggleCamera }: Props) {
  const [root, setRoot] = useState<Root | null>(null);
  const [metadata, setMetadata] = useState<Metadata>({ participant_id: '', session_id: `session-${new Date().toISOString().slice(0, 10)}`, source: 'webcam', scenario: 'standing_arm_hanging', lighting: 'normal', distance: 'medium', note: '' });
  const [pending, setPending] = useState<PendingMedia | null>(null);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [marks, setMarks] = useState<number[]>([]);
  const [selected, setSelected] = useState('');
  const [clip, setClip] = useState<Clip | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [draft, setDraft] = useState<Segment>(initialDraft);
  const [editing, setEditing] = useState<number | null>(null);
  const [reviewed, setReviewed] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [intervalSeconds, setIntervalSeconds] = useState(1);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [report, setReport] = useState<Record<string, unknown> | null>(null);
  const [modelChoices, setModelChoices] = useState<{ id: string }[]>([]);
  const [evaluationModel, setEvaluationModel] = useState('');
  const player = useRef<HTMLVideoElement | null>(null);
  const webcam = useRef<HTMLVideoElement | null>(null);
  const ncmImage = useRef<HTMLImageElement | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const recordingStream = useRef<MediaStream | null>(null);
  const recordingTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAt = useRef(0);
  const mistakeMarks = useRef<number[]>([]);
  const pendingRef = useRef<PendingMedia | null>(null);
  const scenarioLookup = useRef(new Map<string, Scenario>());

  const refresh = useCallback(async () => { const result = await labRequest<Root>(`${api}/api/video-lab`); scenarioLookup.current = new Map(result.scenarios.map(item => [item.id, item])); setRoot(result); return result; }, [api]);
  useEffect(() => {
    if (!active) return;
    let cancelled = false;
    void labRequest<Root>(`${api}/api/video-lab`).then(result => { if (!cancelled) { scenarioLookup.current = new Map(result.scenarios.map(item => [item.id, item])); setRoot(result); } }).catch(cause => { if (!cancelled) setError(cause.message); });
    return () => { cancelled = true; };
  }, [active, api]);
  useEffect(() => { if (active) void labRequest<{ jobs: { id: string; candidate_available?: boolean }[] }>(`${api}/api/model-lab/status`).then(result => setModelChoices(result.jobs.filter(item => item.candidate_available))).catch(() => setModelChoices([])); }, [active, api]);
  useEffect(() => {
    const video = webcam.current;
    if (!video) return;
    video.srcObject = active ? webcamStream : null;
    if (active && webcamStream) void video.play().catch(() => undefined);
    return () => { video.srcObject = null; };
  }, [webcamStream, active]);
  useEffect(() => {
    if (!active && recorder.current && recorder.current.state !== 'inactive') recorder.current.stop();
  }, [active]);

  const loadClip = useCallback(async (id: string) => {
    const result = await labRequest<Clip>(`${api}/api/video-lab/clips/${encodeURIComponent(id)}`);
    setClip(result); setSegments(result.segments ?? []); setReviewed(result.reviewed); setDirty(false); setEditing(null);
    const reason = result.scenario === 'valid_command' || result.scenario === 'ordinary_activity' ? 'standing_arm_hanging' : result.scenario;
    setDraft({ ...initialDraft, end: Math.min(result.duration_seconds, 10), negative_reason: reason, training_use: scenarioLookup.current.get(reason)?.training_use ?? 'context_only' });
    setReport(result.reports?.[result.reports.length - 1] ?? result.extraction ?? null);
  }, [api]);

  useEffect(() => {
    if (!selected) return;
    let cancelled = false;
    void labRequest<Clip>(`${api}/api/video-lab/clips/${encodeURIComponent(selected)}`).then(result => {
      if (cancelled) return;
      setClip(result); setSegments(result.segments ?? []); setReviewed(result.reviewed); setDirty(false); setEditing(null);
      const reason = result.scenario === 'valid_command' || result.scenario === 'ordinary_activity' ? 'standing_arm_hanging' : result.scenario;
      setDraft({ ...initialDraft, end: Math.min(result.duration_seconds, 10), negative_reason: reason, training_use: scenarioLookup.current.get(reason)?.training_use ?? 'context_only' });
      setReport(result.reports?.[result.reports.length - 1] ?? result.extraction ?? null);
    }).catch(cause => { if (!cancelled) setError(cause.message); });
    return () => { cancelled = true; };
  }, [api, selected]);

  const jobId = job?.job_id ?? job?.id;
  const jobStatus = job?.status;
  useEffect(() => {
    if (!jobId || !jobStatus || !['queued', 'running'].includes(jobStatus)) return;
    let cancelled = false;
    const update = async () => {
      try {
        const result = await labRequest<Job>(`${api}/api/video-lab/jobs/${encodeURIComponent(jobId)}`);
        if (cancelled) return;
        setJob(result);
        if (!['queued', 'running'].includes(result.status)) {
          if (result.error) setError(result.error);
          if (result.result) setReport(result.result);
          await refresh();
          if (selected) await loadClip(selected);
          if (result.result) setReport(result.result);
        }
      } catch (cause) { if (!cancelled) setError(cause instanceof Error ? cause.message : 'Could not update the job.'); }
    };
    void update();
    const timer = setInterval(() => { void update(); }, 2000);
    return () => { cancelled = true; clearInterval(timer); };
  }, [api, jobId, jobStatus, loadClip, refresh, selected]);

  const replacePending = (value: PendingMedia | null) => {
    if (pendingRef.current) URL.revokeObjectURL(pendingRef.current.url);
    pendingRef.current = value; setPending(value);
  };

  useEffect(() => () => {
    if (recordingTimer.current) clearInterval(recordingTimer.current);
    if (recorder.current && recorder.current.state !== 'inactive') { recorder.current.onstop = null; recorder.current.stop(); }
    recordingStream.current?.getTracks().forEach(track => track.stop());
    if (pendingRef.current) URL.revokeObjectURL(pendingRef.current.url);
  }, []);

  function chooseFile(file?: File) {
    if (!file) return;
    if (root?.max_upload_bytes && file.size > root.max_upload_bytes) { setError(`This video exceeds the ${(root.max_upload_bytes / 1024 / 1024).toFixed(0)} MB upload limit.`); return; }
    replacePending({ blob: file, filename: file.name, url: URL.createObjectURL(file), marks: [] });
    mistakeMarks.current = []; setMarks([]);
    setError(''); setNotice('Video selected. Check its metadata, then save it for interval review.');
  }

  function stopRecording() { if (recorder.current && recorder.current.state !== 'inactive') recorder.current.stop(); }

  function startRecording() {
    setError(''); setNotice('');
    if (!metadata.participant_id.trim() || !metadata.session_id.trim()) { setError('Enter participant and session IDs before recording.'); return; }
    if (typeof MediaRecorder === 'undefined') { setError('This browser cannot record video. Upload an existing recording instead.'); return; }
    try {
      let stream: MediaStream;
      let draw: (() => void) | null = null;
      if (cameraSource === 'webcam') {
        const track = webcamStream?.getVideoTracks().find(item => item.readyState === 'live');
        if (!track) throw new Error('Connect the webcam before recording.');
        stream = new MediaStream([track.clone()]);
      } else {
        const image = ncmImage.current;
        if (!image?.naturalWidth) throw new Error('Wait for the board camera preview before recording.');
        const canvas = document.createElement('canvas');
        canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
        const context = canvas.getContext('2d');
        if (!context) throw new Error('Video recording is unavailable in this browser.');
        draw = () => { context.drawImage(image, 0, 0, canvas.width, canvas.height); };
        draw();
        stream = canvas.captureStream(10);
      }
      const mimeType = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(type => MediaRecorder.isTypeSupported(type));
      recordingStream.current = stream;
      const media = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorder.current = media;
      const snapshot = { ...metadata, source: cameraSource };
      const chunks: Blob[] = [];
      mistakeMarks.current = []; setMarks([]); setElapsed(0); startedAt.current = performance.now();
      media.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      media.onerror = () => { setError('Recording failed. Try uploading a saved video.'); stopRecording(); };
      media.onstop = () => {
        if (recordingTimer.current) clearInterval(recordingTimer.current);
        recordingTimer.current = null; setRecording(false);
        stream.getTracks().forEach(track => track.stop()); recordingStream.current = null;
        const type = media.mimeType || mimeType || 'video/webm';
        const blob = new Blob(chunks, { type });
        if (!blob.size) { setError('The recording was empty. Connect the camera and try again.'); return; }
        const filename = `${snapshot.participant_id}-${Date.now()}.${type.includes('mp4') ? 'mp4' : 'webm'}`;
        replacePending({ blob, filename, url: URL.createObjectURL(blob), metadata: snapshot, marks: [...mistakeMarks.current] });
        setNotice('Recording ready. Save the video, then label its time intervals. Mistake markers are saved in its notes.');
      };
      media.start(1000); setRecording(true);
      recordingTimer.current = setInterval(() => {
        try { draw?.(); } catch { setError('The board camera preview could not be recorded. Upload an exported recording instead.'); stopRecording(); }
        const seconds = (performance.now() - startedAt.current) / 1000;
        setElapsed(seconds);
        if (seconds >= Math.min(root?.max_duration_seconds ?? 300, 300) - 1) stopRecording();
      }, 100);
    } catch (cause) {
      recordingStream.current?.getTracks().forEach(track => track.stop()); recordingStream.current = null;
      setError(cause instanceof Error ? cause.message : 'Could not start recording.');
    }
  }

  function markMistake() {
    if (mistakeMarks.current.length >= 50) return;
    const time = (performance.now() - startedAt.current) / 1000;
    mistakeMarks.current.push(Number(time.toFixed(2))); setMarks([...mistakeMarks.current]);
  }

  async function upload() {
    if (!pending) return;
    setBusy(true); setError('');
    try {
      const fields = pending.metadata ?? metadata;
      if (!fields.participant_id.trim() || !fields.session_id.trim()) throw new Error('Participant and session IDs are required.');
      const form = new FormData();
      form.append('video', pending.blob, pending.filename);
      Object.entries(fields).forEach(([key, value]) => form.append(key, key === 'note' && pending.marks.length ? `${value}\nReview mistake timestamps (seconds): ${pending.marks.join(', ')}`.trim() : value));
      const result = await labRequest<Clip & { clip?: Clip }>(`${api}/api/video-lab/upload`, { method: 'POST', body: form });
      const saved = result.clip ?? result;
      setSelected(saved.id); await refresh(); replacePending(null);
      setNotice('Video saved. Add and review labeled intervals before extraction or evaluation.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not upload video.'); }
    finally { setBusy(false); }
  }

  function updateLabel(label: string) {
    setDraft(current => ({ ...current, label, training_use: label === 'no_gesture' ? root?.scenarios.find(item => item.id === current.negative_reason)?.training_use ?? 'context_only' : label === 'uncertain' ? 'context_only' : 'hand_classifier' }));
  }

  function addSegment() {
    setError('');
    if (!clip || !Number.isFinite(draft.start) || !Number.isFinite(draft.end) || draft.start < 0 || draft.end <= draft.start || draft.end > clip.duration_seconds + .01) { setError('Choose an interval inside the video with its end after its start.'); return; }
    const candidate = { ...draft, negative_reason: draft.label === 'no_gesture' ? draft.negative_reason : '', training_use: draft.label === 'no_gesture' ? scenarioLookup.current.get(draft.negative_reason)?.training_use ?? 'context_only' : draft.label === 'uncertain' ? 'context_only' : 'hand_classifier' } as Segment;
    if (segments.some((item, index) => index !== editing && candidate.start < item.end && candidate.end > item.start)) { setError('Labeled intervals cannot overlap. Edit or remove the existing interval first.'); return; }
    if (candidate.label === 'fist' && candidate.finger_orientation !== 'toward_camera') { setError('A valid Fist must face the camera. Label the reverse-facing example no_gesture.'); return; }
    setSegments(current => [...current.filter((_, index) => index !== editing), candidate].sort((a, b) => a.start - b.start));
    setDirty(true); setReviewed(false); setEditing(null);
  }

  async function saveAnnotations() {
    if (!clip) return;
    setBusy(true); setError('');
    try {
      await labRequest(`${api}/api/video-lab/clips/${encodeURIComponent(clip.id)}/annotations`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ segments, reviewed }) });
      await loadClip(clip.id); await refresh(); setNotice(reviewed ? 'Reviewed intervals saved. The clip is ready for extraction and testing.' : 'Draft intervals saved. Confirm review before extraction and testing.');
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not save annotations.'); }
    finally { setBusy(false); }
  }

  async function startJob(kind: 'extract' | 'evaluate') {
    if (!clip) return;
    setBusy(true); setError(''); setReport(null);
    try {
      const result = await labRequest<Job>(`${api}/api/video-lab/clips/${encodeURIComponent(clip.id)}/${kind}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(kind === 'extract' ? { interval_seconds: intervalSeconds } : evaluationModel ? { candidate_job_id: evaluationModel } : {}) });
      setJob({ ...result, id: result.id ?? result.job_id! });
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Could not start processing.'); }
    finally { setBusy(false); }
  }

  const jobRunning = !!job && ['queued', 'running'].includes(job.status);
  const canProcess = !!clip?.reviewed && reviewed && !dirty && !!segments.length && !busy && !jobRunning;
  const reasons = root?.scenarios.filter(item => !['valid_command', 'ordinary_activity'].includes(item.id)) ?? [];
  const scenario = root?.scenarios.find(item => item.id === metadata.scenario);

  return <div className="lab-page" hidden={!active}>
    <div className="section-heading"><div><p className="eyebrow">06 · CONTINUOUS DATA AND FALSE ACTION TESTING</p><h2>Video Lab</h2><p className="lab-intro">Record or upload video, review time intervals, and test the complete recognition pipeline.</p></div><button type="button" className="secondary-button" disabled={busy} onClick={() => { void refresh().catch(cause => setError(cause.message)); }}>Refresh clips</button></div>
    <div className="lab-policy"><strong>Dorsal requires visible arm and torso evidence.</strong><p>Record the shoulder, elbow, wrist, and waist. Relaxed hanging or folded arms belong to context negatives. They are retained for runtime testing and excluded from static hand training. Use Data Collection for individual still frames.</p></div>
    {error && <div className="lab-error" role="alert">{error}</div>}{notice && <p className="lab-notice" role="status">{notice}</p>}
    <div className="lab-duration-guide"><span><b>5–10 sec</b>Valid command: relax → present → hold → release</span><span><b>10–20 sec</b>Hard negative: rest, grasp, turn, or near-command shape</span><span><b>2–5 min</b>Ordinary activity: measure false actions over time</span></div>
    <div className="lab-grid">
      <section className="panel"><div className="panel-header"><div>RECORD OR UPLOAD</div><span>RAW VIDEO · NO OVERLAYS</span></div><div className="lab-body">
        <fieldset className="lab-fieldset" disabled={recording || busy || !!pending?.metadata}><div className="lab-form-grid">
          <label className="lab-field">Participant ID<input value={metadata.participant_id} maxLength={48} placeholder="person-001" onChange={event => setMetadata(current => ({ ...current, participant_id: event.target.value }))} /></label>
          <label className="lab-field">Session ID<input value={metadata.session_id} maxLength={48} onChange={event => setMetadata(current => ({ ...current, session_id: event.target.value }))} /></label>
          <label className="lab-field lab-span">Recording scenario<select value={metadata.scenario} onChange={event => setMetadata(current => ({ ...current, scenario: event.target.value }))}>{root?.scenarios.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>
          <label className="lab-field">Lighting<select value={metadata.lighting} onChange={event => setMetadata(current => ({ ...current, lighting: event.target.value }))}><option value="normal">Normal</option><option value="dim">Dim</option><option value="backlit">Window behind</option></select></label>
          <label className="lab-field">Distance<select value={metadata.distance} onChange={event => setMetadata(current => ({ ...current, distance: event.target.value }))}><option value="near">Near</option><option value="medium">Medium</option><option value="far">Far</option></select></label>
          <label className="lab-field">Uploaded video source<select value={metadata.source} onChange={event => setMetadata(current => ({ ...current, source: event.target.value }))}><option value="webcam">PC webcam</option><option value="ncm">NCM board camera</option></select></label>
          <label className="lab-field lab-span">Notes<textarea value={metadata.note} maxLength={1000} placeholder="Pose differences, camera angle, or details to review" onChange={event => setMetadata(current => ({ ...current, note: event.target.value }))} /></label>
        </div></fieldset>
        {scenario?.instruction && <p className="lab-hint">{scenario.instruction}</p>}
        <div className="lab-record-controls"><label className="lab-field">Live camera<select value={cameraSource} disabled={connected || recording} onChange={event => setCameraSource(event.target.value as CameraSource)}><option value="webcam">PC webcam</option><option value="ncm">NCM board camera</option></select></label><button type="button" className="secondary-button" disabled={!ready || recording || busy} onClick={toggleCamera}>{connected ? 'Disconnect' : 'Connect camera'}</button></div>
        <div className="lab-camera-preview">{cameraSource === 'webcam' ? <video ref={webcam} autoPlay muted playsInline aria-label="Raw webcam recording preview" /> : active && connected ? <img ref={ncmImage} crossOrigin="anonymous" src={`${api}/api/ncm/stream.mjpg?revision=${streamRevision}`} alt="Raw board camera recording preview" /> : null}{!connected && <span>Connect a camera to record, or upload a saved video below.</span>}{recording && <strong className="lab-recording-indicator">● REC {formatTime(elapsed)}</strong>}</div>
        <div className="lab-actions">{recording ? <><button type="button" className="primary-button" onClick={stopRecording}>Stop recording</button><button type="button" className="secondary-button" disabled={marks.length >= 50} onClick={markMistake}>Mark mistake at {formatTime(elapsed)}</button></> : <><button type="button" className="primary-button" disabled={!connected || busy || !metadata.participant_id.trim() || !metadata.session_id.trim()} onClick={startRecording}>Start recording</button><label className="secondary-button file-button">Choose video<input type="file" accept="video/*" disabled={busy} onChange={event => { chooseFile(event.target.files?.[0]); event.target.value = ''; }} /></label></>}</div>
        {marks.length > 0 && <p className="lab-hint">Review markers: {marks.map(value => formatTime(value)).join(', ')}. Markers identify moments to inspect; they do not assign labels.</p>}
        {pending && <div className="lab-pending"><strong>{pending.filename}</strong><small>{(pending.blob.size / 1024 / 1024).toFixed(1)} MB {pending.metadata ? '· metadata captured at recording start' : ''}</small><video src={pending.url} controls preload="metadata" aria-label="Review video before saving" /><div className="lab-actions"><button className="primary-button" type="button" disabled={busy || recording} onClick={() => { void upload(); }}>{busy ? 'Saving…' : 'Save video for review'}</button><a className="secondary-button" href={pending.url} download={pending.filename}>Download recording</a><button className="lab-text-button" type="button" disabled={busy || recording} onClick={() => replacePending(null)}>Discard selection</button></div></div>}
        <p className="lab-hint">Recording stops at {Math.min(root?.max_duration_seconds ?? 300, 300) - 1} seconds to stay within the upload limit. Keep each person in one dataset split and record valid commands and negatives in the same conditions.</p>
      </div></section>
      <section className="panel"><div className="panel-header"><div>SAVED CLIPS</div><span>{root?.clips.length ?? 0} VIDEOS</span></div><div className="lab-list lab-clips">{root?.clips.length ? root.clips.map(item => <button key={item.id} type="button" className={selected === item.id ? 'selected' : ''} disabled={busy || jobRunning} onClick={() => { setSelected(item.id); setError(''); }}><strong>{item.participant_id}</strong><span className={`lab-badge ${item.reviewed ? 'lab-status-succeeded' : ''}`}>{item.reviewed ? 'Reviewed' : 'Needs review'}</span><small>{labLabel(item.scenario)} · {formatTime(item.duration_seconds)}</small><small>{item.session_id} · {labLabel(item.source)}</small></button>) : <p className="empty-note">Save a short video to begin. Uploading keeps it unlabelled until you review its intervals.</p>}</div></section>
    </div>
    {clip && <section className="panel lab-review-panel"><div className="panel-header"><div>REVIEW · {clip.participant_id.toUpperCase()}</div><span>{clip.width} × {clip.height} · {formatTime(clip.duration_seconds)}</span></div><div className="lab-body">
      <div className="lab-grid lab-review-grid"><div><video key={clip.id} ref={player} className="lab-review-video" src={`${api}/api/video-lab/clips/${encodeURIComponent(clip.id)}/media`} controls preload="metadata" aria-label="Video interval review player" />{clip.note && <p className="lab-hint lab-note">{clip.note}</p>}<p className="lab-hint">Seek to each boundary, mark the interval, then choose its true label. Gaps and uncertain spans are excluded from scoring. Use only distinguishable single-hand shapes for static training.</p></div><div>
        <div className="lab-form-grid"><label className="lab-field">Start (seconds)<input type="number" min={0} max={clip.duration_seconds} step={0.1} value={draft.start} onChange={event => setDraft(current => ({ ...current, start: Number(event.target.value) }))} /><button type="button" className="lab-text-button" onClick={() => setDraft(current => ({ ...current, start: Number((player.current?.currentTime ?? 0).toFixed(2)) }))}>Mark current time as start</button></label><label className="lab-field">End (seconds)<input type="number" min={0} max={clip.duration_seconds} step={0.1} value={draft.end} onChange={event => setDraft(current => ({ ...current, end: Number(event.target.value) }))} /><button type="button" className="lab-text-button" onClick={() => setDraft(current => ({ ...current, end: Number((player.current?.currentTime ?? 0).toFixed(2)) }))}>Mark current time as end</button></label>
          <label className="lab-field lab-span">True label<select value={draft.label} onChange={event => updateLabel(event.target.value)}>{(root?.classes ?? ['no_gesture', 'dorsal', 'fist', 'uncertain']).map(label => <option key={label} value={label}>{labLabel(label)}</option>)}</select></label>
          {draft.label === 'no_gesture' && <label className="lab-field lab-span">Why this is no gesture<select value={draft.negative_reason} onChange={event => { const choice = reasons.find(item => item.id === event.target.value); setDraft(current => ({ ...current, negative_reason: event.target.value, training_use: choice?.training_use ?? 'context_only' })); }}>{reasons.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select></label>}
          <label className="lab-field">Hand shown<select value={draft.handedness} onChange={event => setDraft(current => ({ ...current, handedness: event.target.value }))}><option value="right">Right hand</option><option value="left">Left hand</option><option value="both">Both hands</option><option value="none">No hand</option></select></label><label className="lab-field">Finger direction<select value={draft.finger_orientation} onChange={event => setDraft(current => ({ ...current, finger_orientation: event.target.value }))}><option value="toward_camera">Toward camera</option><option value="away_from_camera">Away from camera</option><option value="none">Not applicable</option></select></label>
        </div><p className="lab-hint">Use: <strong>{draft.label === 'uncertain' ? 'Excluded / uncertain' : labLabel(draft.training_use)}</strong>. {draft.training_use === 'context_only' ? 'Kept for context checks and video evaluation; excluded from the static hand model.' : draft.training_use === 'detector_only' ? 'Kept for detector testing; excluded from static hand training.' : 'Reviewed frames with valid single-hand features can enter static training.'}</p><button type="button" className="secondary-button" disabled={busy || jobRunning} onClick={addSegment}>{editing === null ? 'Add labeled interval' : 'Update interval'}</button>{editing !== null && <button type="button" className="lab-text-button" onClick={() => setEditing(null)}>Cancel edit</button>}
      </div></div>
      <div className="table-wrap"><table className="lab-results-table lab-segments"><thead><tr><th>Interval</th><th>True label</th><th>Reason / use</th><th>Review</th></tr></thead><tbody>{segments.map((item, index) => <tr key={`${index}-${item.start}`}><td><button className="lab-text-button" type="button" onClick={() => { if (player.current) player.current.currentTime = item.start; }}>{formatTime(item.start)} – {formatTime(item.end)}</button></td><td>{labLabel(item.label)}</td><td>{item.negative_reason ? `${labLabel(item.negative_reason)} · ` : ''}{labLabel(item.training_use)}</td><td><button className="lab-text-button" type="button" disabled={busy || jobRunning} onClick={() => { setDraft({ ...item }); setEditing(index); }}>Edit</button><button className="lab-text-button lab-danger" type="button" disabled={busy || jobRunning} onClick={() => { setSegments(current => current.filter((_, row) => row !== index)); setDirty(true); setReviewed(false); setEditing(null); }}>Remove</button></td></tr>)}</tbody></table>{!segments.length && <p className="empty-note">No labeled intervals yet. Add at least one interval and review it.</p>}</div>
      <label className="lab-check"><input type="checkbox" checked={reviewed} disabled={busy || jobRunning || !segments.length} onChange={event => { setReviewed(event.target.checked); setDirty(true); }} />I reviewed the labels, interval boundaries, hand direction, and intended training use.</label><button type="button" className="primary-button" disabled={busy || jobRunning || !segments.length} onClick={() => { void saveAnnotations(); }}>Save {reviewed ? 'reviewed' : 'draft'} intervals</button>
      <div className="lab-process-controls"><label className="lab-field">Frame interval (seconds)<input type="number" min={0.2} max={10} step={0.1} value={intervalSeconds} onChange={event => setIntervalSeconds(Number(event.target.value))} /></label><button className="secondary-button" type="button" disabled={!canProcess || !Number.isFinite(intervalSeconds) || intervalSeconds < .2 || intervalSeconds > 10} onClick={() => { void startJob('extract'); }}>Extract diverse frames</button><label className="lab-field">Runtime model<select value={evaluationModel} onChange={event => setEvaluationModel(event.target.value)}><option value="">Current production model</option>{modelChoices.map(item => <option key={item.id} value={item.id}>Candidate {item.id.slice(0, 10)}</option>)}</select></label><button className="primary-button" type="button" disabled={!canProcess} onClick={() => { void startJob('evaluate'); }}>Check runtime on video</button></div>
      <p className="lab-hint">Extraction skips uncertain spans and duplicate frames. Context negatives remain excluded from static training. Video testing reports false actions, missed commands, substitutions, duplicates, and delay. Replay fixes the selected ONNX classifier for the whole clip and excludes live feedback updates.</p>
      {job && <div className="lab-policy" role="status"><strong>{labLabel(job.kind)} · {labLabel(job.status)}</strong>{job.progress !== undefined && <p>{typeof job.progress === 'number' ? `Progress: ${(job.progress * 100).toFixed(0)}%` : JSON.stringify(job.progress)}</p>}</div>}
      {report && <LabReport result={report} title="Clip results" />}
    </div></section>}
  </div>;
}
