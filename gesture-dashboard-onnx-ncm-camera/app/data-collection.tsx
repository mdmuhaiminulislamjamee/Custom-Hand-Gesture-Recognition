'use client';
/* eslint-disable @next/next/no-img-element */
import { useCallback, useEffect, useRef, useState } from 'react';

type Handedness = 'left' | 'right';
type FingerOrientation = 'toward_camera' | 'away_from_camera';
type Step = { id: string; label: string; title: string; view: string; instruction: string; variation: string; target: number; per_hand_target: number | null; per_orientation_target: number | null };
type HandCounts = { left?: number; right?: number; none?: number; unspecified?: number };
type OrientationCounts = { toward_camera?: number; away_from_camera?: number; none?: number; unspecified?: number };
type RecentSample = { sample_id: string; label: string; view: string; handedness: string; finger_orientation?: string; image_path: string };
type CollectedSample = RecentSample & { sample_ref: string; step_id: string; created_at_utc: string };
type Summary = { root: string; available: boolean; participants: string[]; participant_id: string | null; counts: Record<string, number>; counts_by_hand?: Record<string, HandCounts>; counts_by_hand_orientation?: Record<string, Partial<Record<Handedness | 'none' | 'unspecified', OrientationCounts>>>; total: number; plan: Step[]; recent: RecentSample[] };
type Snapshot = { token: string; image_url: string; frame_id: number; prediction: string; reason: string; landmarks: number[][]; has_features: boolean; issues: string[] };
type Props = { api: string; active: boolean; connected: boolean; ready: boolean; streamRevision: number; liveLandmarks?: number[][]; toggleCamera: () => void };
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const pretty = (value: string) => value.replaceAll('_', ' ');
const viewLabels: Record<string, string> = {
  front: 'Face the camera',
  yaw_left: 'Turn hand slightly left',
  yaw_right: 'Turn hand slightly right',
  pitch_toward: 'Tilt fingers toward camera',
  pitch_away: 'Tilt fingers away from camera',
  roll_left: 'Tilt wrist left',
  roll_right: 'Tilt wrist right',
  casual: 'Natural / casual',
  background: 'Empty background',
  face_ear: 'Face and ear only',
  middle_finger: 'Middle finger only',
  ring_finger: 'Ring finger only',
  little_finger: 'Little finger only',
  relaxed_hand: 'Relaxed hand',
};
const viewLabel = (value: string) => viewLabels[value] ?? pretty(value);
const orientationLabels: Record<string, string> = {
  toward_camera: 'Fingers toward camera',
  away_from_camera: 'Fingers away from camera',
  none: 'No hand',
  unspecified: 'Angle not recorded',
};
const orientationLabel = (value?: string) => orientationLabels[value ?? 'unspecified'] ?? pretty(value ?? 'unspecified');
const handCountsFor = (summary: Summary | null, step: Step | undefined) => ({
  left: step ? summary?.counts_by_hand?.[step.id]?.left ?? 0 : 0,
  right: step ? summary?.counts_by_hand?.[step.id]?.right ?? 0 : 0,
});
const orientationCountsFor = (summary: Summary | null, step: Step | undefined, hand: Handedness) => ({
  toward_camera: step ? summary?.counts_by_hand_orientation?.[step.id]?.[hand]?.toward_camera ?? 0 : 0,
  away_from_camera: step ? summary?.counts_by_hand_orientation?.[step.id]?.[hand]?.away_from_camera ?? 0 : 0,
});
const stepComplete = (summary: Summary, step: Step) => {
  if (step.per_orientation_target) {
    return (['right', 'left'] as Handedness[]).every(hand => {
      const orientations = orientationCountsFor(summary, step, hand);
      return orientations.toward_camera >= step.per_orientation_target! && orientations.away_from_camera >= step.per_orientation_target!;
    });
  }
  return (summary.counts[step.id] ?? 0) >= step.target;
};
const preferredCombination = (summary: Summary, step: Step | undefined): { hand: Handedness; orientation: FingerOrientation } => {
  if (!step?.per_orientation_target) return { hand: 'right', orientation: 'toward_camera' };
  for (const hand of ['right', 'left'] as Handedness[]) {
    const counts = orientationCountsFor(summary, step, hand);
    for (const orientation of ['toward_camera', 'away_from_camera'] as FingerOrientation[]) {
      if (counts[orientation] < step.per_orientation_target) return { hand, orientation };
    }
  }
  return { hand: 'right', orientation: 'toward_camera' };
};
const arrows: Record<string, string> = { left: '←', right: '→', up: '↑', down: '↓', open_palm: '✋', like: '👍', ok: '👌', dorsal: '↩', no_gesture: '○' };
const gestureGuides: Record<string, { src: string; alt: string }> = {
  left: { src: '/gesture-guides/index-only.png', alt: 'Index finger pointing toward your left' },
  right: { src: '/gesture-guides/index-only.png', alt: 'Index finger pointing toward your right' },
  up: { src: '/gesture-guides/index-only.png', alt: 'Index finger pointing upward with all other fingers folded' },
  down: { src: '/gesture-guides/index-only.png', alt: 'Index finger pointing downward with all other fingers folded' },
  open_palm: { src: '/gesture-guides/open-palm.png', alt: 'Open palm with all five fingers visible' },
  like: { src: '/gesture-guides/thumbs-up.png', alt: 'Thumbs-up with the other four fingers folded' },
  dorsal: { src: '/gesture-guides/dorsal.png', alt: 'Back of the hand with four fingers together and pointing down' },
  ok: { src: '/gesture-guides/ok.png', alt: 'OK sign with thumb and index finger forming a circle' },
};
const negativeGuides: Record<string, { src: string; alt: string }> = {
  background: { src: '/gesture-guides/empty-background-negative.png', alt: 'Empty camera view with no person or hand' },
  face_ear: { src: '/gesture-guides/face-ear-negative.png', alt: 'Face and ear visible while both hands stay out of view' },
  middle_finger: { src: '/gesture-guides/middle-finger-negative.png', alt: 'Only the middle finger raised; this is not a command' },
  ring_finger: { src: '/gesture-guides/ring-finger-negative.png', alt: 'Only the ring finger raised; this is not a command' },
  little_finger: { src: '/gesture-guides/little-finger-negative.png', alt: 'Only the little finger raised; this is not a command' },
  relaxed_hand: { src: '/gesture-guides/relaxed-hand-negative.png', alt: 'Relaxed loosely curled hand; this is not a command' },
};
const viewSymbols: Record<string, string> = {
  front: '◎', yaw_left: '↶', yaw_right: '↷', pitch_toward: '⇣', pitch_away: '⇡',
  roll_left: '↙', roll_right: '↘', casual: '〰',
};

const validLandmarks = (landmarks?: number[][]): landmarks is number[][] =>
  landmarks?.length === 21 && landmarks.every(point => point.length >= 2 && Number.isFinite(point[0]) && Number.isFinite(point[1]));

const isFormControl = (target: EventTarget | null) =>
  target instanceof Element && Boolean(target.closest('input, textarea, select, button, a, [contenteditable="true"], [role="button"]'));

function HandLandmarkPreview({ landmarks, expectHand }: { landmarks?: number[][]; expectHand: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hasLandmarks = validLandmarks(landmarks);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const width = Math.max(1, Math.round(canvas.clientWidth || 480));
    const height = Math.max(1, Math.round(canvas.clientHeight || width * .75));
    const pixelRatio = Math.max(1, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    const context = canvas.getContext('2d');
    if (!context) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (!validLandmarks(landmarks)) return;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.lineWidth = Math.max(2, width / 190);
    context.strokeStyle = 'rgba(73, 217, 209, .95)';
    context.beginPath();
    HAND_CONNECTIONS.forEach(([from, to]) => {
      context.moveTo(landmarks[from][0] * width, landmarks[from][1] * height);
      context.lineTo(landmarks[to][0] * width, landmarks[to][1] * height);
    });
    context.stroke();
    landmarks.forEach(([x, y], index) => {
      context.beginPath();
      context.fillStyle = index === 0 ? '#ffb454' : '#f4f7ff';
      context.arc(x * width, y * height, Math.max(2.5, width / 125), 0, Math.PI * 2);
      context.fill();
    });
  }, [landmarks]);

  useEffect(() => {
    draw();
    if (typeof ResizeObserver === 'undefined' || !canvasRef.current) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvasRef.current);
    return () => observer.disconnect();
  }, [draw]);

  return <div className="collection-preview-pane collection-landmark-preview" aria-label="Detected hand landmarks">
    <span className="collection-preview-label">HAND LANDMARKS</span>
    <canvas ref={canvasRef} className="collection-landmark-canvas" />
    {!hasLandmarks && <div className="collection-landmark-empty"><span aria-hidden="true">◇</span><strong>No hand landmarks detected</strong><small>{expectHand ? 'Keep the complete hand and wrist inside the camera view.' : 'This is expected for the current no-hand prompt.'}</small></div>}
  </div>;
}

export default function DataCollection({ api, active, connected, ready, streamRevision, liveLandmarks, toggleCamera }: Props) {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [participant, setParticipant] = useState('');
  const [newParticipant, setNewParticipant] = useState('');
  const [stepIndex, setStepIndex] = useState(0);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [countdown, setCountdown] = useState<number | null>(null);
  const [handedness, setHandedness] = useState<Handedness>('right');
  const [fingerOrientation, setFingerOrientation] = useState<FingerOrientation>('toward_camera');
  const [deleteArmed, setDeleteArmed] = useState(false);
  const [deleteSelectionArmed, setDeleteSelectionArmed] = useState(false);
  const [samples, setSamples] = useState<CollectedSample[]>([]);
  const [selectedSamples, setSelectedSamples] = useState<string[]>([]);
  const [lighting, setLighting] = useState('normal');
  const [distance, setDistance] = useState('medium');
  const [note, setNote] = useState('');
  const session = useRef('');
  const captureGeneration = useRef(0);
  const step = summary?.plan[stepIndex];
  const count = step ? summary?.counts[step.id] ?? 0 : 0;
  const handCounts = handCountsFor(summary, step);
  const requiresHands = !!step?.per_hand_target;
  const complete = summary?.plan.filter(s => stepComplete(summary, s)).length ?? 0;
  const lastSave = summary?.recent[0];
  const guide = step ? (step.label === 'no_gesture' ? negativeGuides[step.view] : gestureGuides[step.label]) : undefined;
  const showGuide = !!guide && countdown === null && snapshot === null;
  const displayedLandmarks = snapshot ? snapshot.landmarks : connected && active ? liveLandmarks : undefined;

  const request = useCallback(async (path: string, body?: object) => {
    const response = await fetch(`${api}/api/collection${path}`, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'The collection request failed.');
    return result;
  }, [api]);

  const cancelCapture = useCallback(() => { captureGeneration.current++; }, []);

  useEffect(() => {
    if (!active) { cancelCapture(); return; }
    let cancelled = false;
    request(participant ? `?participant_id=${encodeURIComponent(participant)}` : '')
      .then(result => { if (!cancelled) { setSummary(result); setError(''); } })
      .catch(e => { if (!cancelled) setError(e.message); });
    return () => { cancelled = true; cancelCapture(); };
  }, [active, participant, request, cancelCapture]);

  useEffect(() => {
    if (!active || !participant || !step) return;
    let cancelled = false;
    request(`/samples?participant_id=${encodeURIComponent(participant)}&step_id=${encodeURIComponent(step.id)}`)
      .then((result: { samples: CollectedSample[] }) => {
        if (!cancelled) { setSamples(result.samples); setSelectedSamples([]); setDeleteSelectionArmed(false); }
      })
      .catch(e => { if (!cancelled) setError(e instanceof Error ? e.message : 'Could not load saved images.'); });
    return () => { cancelled = true; };
  }, [active, participant, step, request]);

  async function selectParticipant(id: string) {
    setBusy(true); setError(''); setSnapshot(null); setSamples([]); setSelectedSamples([]); setDeleteSelectionArmed(false);
    try {
      const result: Summary = await request('/participants', { participant_id: id });
      setParticipant(result.participant_id ?? id); setSummary(result); setNewParticipant('');
      session.current = `session-${new Date().toISOString().replace(/[^0-9]/g, '')}`;
      const next = result.plan.findIndex(s => !stepComplete(result, s));
      const nextIndex = Math.max(0, next);
      const choice = preferredCombination(result, result.plan[nextIndex]);
      setStepIndex(nextIndex); setHandedness(choice.hand); setFingerOrientation(choice.orientation); setDeleteArmed(false);
      setMessage(`Ready for ${result.participant_id}. Collect a few varied photos at each prompt.`);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not open participant.'); }
    finally { setBusy(false); }
  }

  const capture = useCallback(async () => {
    setBusy(true); setError(''); setMessage('');
    const generation = ++captureGeneration.current;
    try {
      for (const seconds of [3, 2, 1]) {
        setCountdown(seconds);
        await new Promise(resolve => setTimeout(resolve, 1000));
        if (generation !== captureGeneration.current) return;
      }
      setCountdown(null);
      const result: Snapshot = await request('/preview', {});
      if (generation === captureGeneration.current) setSnapshot(result);
    } catch (e) { setError(e instanceof Error ? e.message : 'Capture failed.'); }
    finally { setBusy(false); setCountdown(null); }
  }, [request]);

  const save = useCallback(async () => {
    if (!snapshot || !step) return;
    setBusy(true); setError('');
    try {
      const result: { summary: Summary; image_path: string } = await request('/samples', { token: snapshot.token, participant_id: participant,
        session_id: session.current, step_id: step.id, handedness: step.view === 'background' || step.view === 'face_ear' ? 'none' : handedness,
        finger_orientation: step.view === 'background' || step.view === 'face_ear' ? 'none' : fingerOrientation,
        lighting, distance, note });
      setSummary(result.summary); setSnapshot(null);
      let savedMessage = `Saved ${step.title} / ${viewLabel(step.view)} / ${requiresHands ? `${handedness} hand / ${fingerOrientation === 'toward_camera' ? 'fingers toward camera' : 'fingers away from camera'}` : 'no hand'}. ${result.image_path}`;
      if (step.per_orientation_target && !stepComplete(result.summary, step)) {
        const nextChoice = preferredCombination(result.summary, step);
        if (nextChoice.hand !== handedness || nextChoice.orientation !== fingerOrientation) {
          setHandedness(nextChoice.hand); setFingerOrientation(nextChoice.orientation);
          savedMessage += ` Next: ${nextChoice.hand} hand, ${nextChoice.orientation === 'toward_camera' ? 'fingers toward camera' : 'fingers away from camera'}.`;
        }
      }
      setMessage(savedMessage); setDeleteArmed(false);
      if (stepComplete(result.summary, step)) {
        const next = result.summary.plan.findIndex((s: Step, index: number) => index > stepIndex && !stepComplete(result.summary, s));
        if (next >= 0) {
          const nextChoice = preferredCombination(result.summary, result.summary.plan[next]);
          setStepIndex(next); setHandedness(nextChoice.hand); setFingerOrientation(nextChoice.orientation);
        }
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Save failed.'); }
    finally { setBusy(false); }
  }, [distance, fingerOrientation, handedness, lighting, note, participant, request, requiresHands, snapshot, step, stepIndex]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if (!active || busy || event.repeat || event.isComposing || isFormControl(event.target)) return;
      if (event.code !== 'Space' && event.key !== ' ') return;
      if (deleteArmed || deleteSelectionArmed) return;
      if (snapshot && step) {
        event.preventDefault();
        void save();
      } else if (countdown === null && participant && connected && step) {
        event.preventDefault();
        void capture();
      }
    };
    window.addEventListener('keydown', handleShortcut);
    return () => window.removeEventListener('keydown', handleShortcut);
  }, [active, busy, capture, connected, countdown, deleteArmed, deleteSelectionArmed, participant, save, snapshot, step]);

  async function deleteLast() {
    if (!participant || !lastSave) return;
    setBusy(true); setError('');
    try {
      const result: { summary: Summary; deleted: RecentSample } = await request('/samples/delete-last', { participant_id: participant });
      setSummary(result.summary); setDeleteArmed(false);
      if (step) {
        const choice = preferredCombination(result.summary, step);
        setHandedness(choice.hand); setFingerOrientation(choice.orientation);
      }
      setMessage(`Deleted last image: ${pretty(result.deleted.label)} / ${viewLabel(result.deleted.view)}${result.deleted.handedness === 'left' || result.deleted.handedness === 'right' ? ` / ${result.deleted.handedness} hand / ${orientationLabel(result.deleted.finger_orientation)}` : ''}.`);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not delete the last image.'); }
    finally { setBusy(false); }
  }

  async function deleteSelected() {
    if (!participant || !selectedSamples.length) return;
    setBusy(true); setError(''); setMessage('');
    try {
      const result: { summary: Summary; deleted: CollectedSample[] } = await request('/samples/delete-selected', {
        participant_id: participant, sample_refs: selectedSamples,
      });
      setSummary(result.summary); setSelectedSamples([]); setDeleteSelectionArmed(false); setDeleteArmed(false);
      if (step) {
        const choice = preferredCombination(result.summary, step);
        setHandedness(choice.hand); setFingerOrientation(choice.orientation);
      }
      setMessage(`Deleted ${result.deleted.length} selected image${result.deleted.length === 1 ? '' : 's'} and matching label file${result.deleted.length === 1 ? '' : 's'}.`);
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not delete the selected images.'); }
    finally { setBusy(false); }
  }

  function toggleSample(reference: string) {
    setDeleteSelectionArmed(false);
    setSelectedSamples(current => current.includes(reference) ? current.filter(item => item !== reference) : [...current, reference]);
  }

  function changeStep(index: number) {
    setStepIndex(index); setSnapshot(null); setMessage(''); setError(''); setDeleteArmed(false);
    setSamples([]); setSelectedSamples([]); setDeleteSelectionArmed(false);
    if (summary) {
      const choice = preferredCombination(summary, summary.plan[index]);
      setHandedness(choice.hand); setFingerOrientation(choice.orientation);
    }
  }

  return <div hidden={!active} className="collection-page">
    <div className="section-heading"><div><p className="eyebrow">05 · TRAINING IMAGES</p><h2>Data Collection</h2><p className="collection-intro">A few natural poses from each person. Guided prompts keep the dataset organized.</p></div>
      <button type="button" className="primary-button" disabled={!ready || busy} onClick={toggleCamera}>{connected ? 'Disconnect camera' : 'Connect board camera'}</button></div>
    <div className="collection-storage"><span>LOCAL STORAGE</span><strong>{summary?.root ?? 'D:\\Data-Collection'}</strong><small>{summary?.participants.length ?? 0} participants · {summary?.total ?? 0} images for this person</small></div>
    <div className="collection-layout">
      <aside className="panel collection-controls">
        <div className="panel-header"><div>PARTICIPANT & CONDITIONS</div></div>
        <div className="collection-fields">
          <label>Continue a participant<select aria-label="Continue a participant" value={participant} disabled={busy || !!snapshot} onChange={e => { if (e.target.value) void selectParticipant(e.target.value); }}><option value="">Select a person</option>{summary?.participants.map(id => <option key={id} value={id}>{id}</option>)}</select></label>
          <form onSubmit={e => { e.preventDefault(); void selectParticipant(newParticipant); }}><label>New participant ID<input aria-label="New participant ID" placeholder="person-001" maxLength={48} value={newParticipant} disabled={busy || !!snapshot} onChange={e => setNewParticipant(e.target.value)} /></label><button className="secondary-button wide" disabled={!newParticipant.trim() || busy || !!snapshot}>Create / open participant</button></form>
          <p className="collection-help">Use the same ID for the same person on every visit. Use IDs instead of names, and collect with their permission. Photos may include faces and stay on this PC.</p>
          <div className="collection-condition-grid"><label>Hand shown<select value={requiresHands ? handedness : 'none'} disabled={!requiresHands || busy || !!snapshot} onChange={e => setHandedness(e.target.value as Handedness)}>{requiresHands ? <><option value="right">Right hand</option><option value="left">Left hand</option></> : <option value="none">No hand required</option>}</select></label><label>Finger angle<select value={requiresHands ? fingerOrientation : 'none'} disabled={!requiresHands || busy || !!snapshot} onChange={e => setFingerOrientation(e.target.value as FingerOrientation)}>{requiresHands ? <><option value="toward_camera">Toward camera</option><option value="away_from_camera">Away from camera</option></> : <option value="none">No hand required</option>}</select></label></div>
          <div className="collection-condition-grid"><label>Lighting<select value={lighting} disabled={busy || !!snapshot} onChange={e => setLighting(e.target.value)}><option value="normal">Normal</option><option value="dim">Dim</option><option value="backlit">Window behind</option></select></label><label>Distance<select value={distance} disabled={busy || !!snapshot} onChange={e => setDistance(e.target.value)}><option value="near">Near</option><option value="medium">Medium</option><option value="far">Far</option></select></label></div>
          <label>Notes (optional)<textarea value={note} disabled={busy || !!snapshot} maxLength={500} onChange={e => setNote(e.target.value)} placeholder="e.g. index finger missed at this angle" /></label>
        </div>
      </aside>
      <div className="collection-main">
        {step && <article className="collection-prompt">
          <div className="collection-symbol" aria-hidden="true">{arrows[step.label]}</div>
          <div className="collection-prompt-copy"><p className="eyebrow">COLLECT NOW · {stepIndex + 1} / {summary?.plan.length}</p><h3>{step.title} <span>/ {viewLabel(step.view)}</span></h3><p>{step.instruction}</p><small>{step.variation}</small>
            {step.per_orientation_target ? <div className="collection-hand-options" role="group" aria-label="Choose the hand and finger angle to collect">
              {(['right', 'left'] as Handedness[]).map(hand => <section key={hand} className={handedness === hand ? 'active' : ''}><header><span>{hand === 'right' ? 'R' : 'L'}</span><strong>{hand === 'right' ? 'Right hand' : 'Left hand'}</strong><small>{handCounts[hand]} / {step.per_hand_target}</small></header><div>
                {(['toward_camera', 'away_from_camera'] as FingerOrientation[]).map(orientation => {
                  const orientationCounts = orientationCountsFor(summary, step, hand);
                  const activeChoice = handedness === hand && fingerOrientation === orientation;
                  return <button key={orientation} type="button" className={activeChoice ? 'active' : ''} aria-pressed={activeChoice} disabled={busy || !!snapshot} onClick={() => { setHandedness(hand); setFingerOrientation(orientation); }}><span aria-hidden="true">{orientation === 'toward_camera' ? '⇣' : '⇡'}</span><strong>{orientationLabel(orientation)}</strong><small>{orientationCounts[orientation]} / {step.per_orientation_target}</small></button>;
                })}
              </div></section>)}
            </div> : <div className="collection-hand-free"><span aria-hidden="true">✓</span> No hand required for this prompt</div>}
          </div>
          {showGuide && <figure key={step.id} className={`collection-gesture-guide ${step.label === 'no_gesture' ? 'is-negative' : ''}`} data-gesture-guide={step.id}>
            <figcaption><span>VISUAL EXAMPLE</span><strong>{step.label === 'no_gesture' ? 'Not a command' : requiresHands ? `${handedness} hand · ${fingerOrientation === 'toward_camera' ? 'toward camera' : 'away from camera'}` : 'Copy this pose'}</strong></figcaption>
            <div className={`collection-guide-pose collection-guide-view-${step.view}`}>
              <img className={`collection-guide-image collection-guide-direction-${step.label} ${requiresHands ? `collection-guide-hand-${handedness} collection-guide-angle-${fingerOrientation}` : ''}`} src={guide.src} alt={`${guide.alt}${requiresHands ? ` using the ${handedness} hand with ${orientationLabel(fingerOrientation).toLowerCase()}` : ''}`} />
              {['left', 'right', 'up', 'down'].includes(step.label) && <span className="collection-guide-direction-arrow" aria-hidden="true">{arrows[step.label]}</span>}
              {step.label === 'no_gesture' && <span className="collection-guide-no-command" aria-hidden="true">×</span>}
              <span className="collection-guide-view-symbol" aria-hidden="true">{viewSymbols[step.view] ?? '◎'}</span>
              {requiresHands && <span className="collection-guide-depth" aria-hidden="true">{fingerOrientation === 'toward_camera' ? <><b>FINGERS</b><i>→</i><b>CAMERA</b></> : <><b>CAMERA</b><i>→</i><b>FINGERS</b></>}</span>}
            </div>
          </figure>}
          <div className="collection-count"><strong>{count}<span> / {step.target}</span></strong><small>images saved</small></div>
        </article>}
        <article className="panel">
          <div className="panel-header"><div>{snapshot ? 'REVIEW THIS IMAGE' : 'BOARD CAMERA'}</div><span>{snapshot ? `FRAME ${snapshot.frame_id} · RAW IMAGE KEPT CLEAN` : 'RAW IMAGE + LIVE HAND LANDMARKS'}</span></div>
          <div className="collection-preview">
            <div className="collection-preview-pane collection-image-preview">
              <span className="collection-preview-label">{snapshot ? 'CAPTURED IMAGE' : connected && active ? 'LIVE CAMERA' : 'CAMERA'}</span>
              {snapshot ? <img src={snapshot.image_url} alt={`Captured ${step?.title} image for review`} /> : connected && active ? <img src={`${api}/api/ncm/stream.mjpg?collection=${streamRevision}`} alt="Live camera for data collection" /> : <div className="collection-empty"><span aria-hidden="true">◎</span><strong>Connect the camera to begin</strong><p>Select a participant, follow the prompt, and capture an image.</p></div>}
            </div>
            <HandLandmarkPreview landmarks={displayedLandmarks} expectHand={requiresHands} />
            {countdown !== null && <div className="collection-countdown" role="status">{countdown}</div>}
          </div>
          <div className="collection-review">
            {snapshot ? <><p>Does this image match <strong>{step?.title} / {viewLabel(step?.view ?? '')}</strong>? Press <kbd>Space</kbd> to confirm the label and save it, even if the current model is wrong.</p><small>Model: {pretty(snapshot.prediction)} · {snapshot.has_features ? 'matching landmarks shown beside the image' : 'raw image saved for later extraction'}{snapshot.reason ? ` · ${snapshot.reason}` : ''}</small><div className="heading-actions"><button className="secondary-button" disabled={busy} onClick={() => setSnapshot(null)}>Retake</button><button className="primary-button" aria-keyshortcuts="Space" disabled={busy} onClick={() => void save()}>{busy ? 'Saving…' : 'Confirm label & save image · Space'}</button></div></> : <><p>Move slightly between photos. Keep the intended gesture recognizable and the whole hand in view. Press <kbd>Space</kbd> to capture.</p><button className="primary-button" aria-keyshortcuts="Space" disabled={busy || !participant || !connected || !step} onClick={() => void capture()}>{countdown ? 'Get ready…' : 'Capture image · Space · 3 second timer'}</button></>}
          </div>
        </article>
        {error && <p className="collection-error" role="alert">{error}</p>}
        {message && <p className="collection-message" role="status">{message}</p>}
        <article className="panel collection-gallery">
          <div className="panel-header"><div>SAVED IMAGES FOR THIS SECTION</div><span>{samples.length} / {step?.target ?? 12} IMAGES</span></div>
          <div className="collection-gallery-body">
            <div className="collection-gallery-actions"><p>Select one or more images to remove. Deleting also removes their matching training labels.</p><div className="heading-actions"><button type="button" className="secondary-button" disabled={!samples.length || busy || !!snapshot} onClick={() => setSelectedSamples(selectedSamples.length === samples.length ? [] : samples.map(sample => sample.sample_ref))}>{selectedSamples.length === samples.length && samples.length ? 'Clear selection' : 'Select all'}</button><button type="button" className="danger-button" disabled={!selectedSamples.length || busy || !!snapshot} onClick={() => setDeleteSelectionArmed(true)}>Delete selected ({selectedSamples.length})</button></div></div>
            {deleteSelectionArmed && !!selectedSamples.length && <div className="collection-delete-confirm" role="alert"><p>Permanently delete {selectedSamples.length} selected image{selectedSamples.length === 1 ? '' : 's'} and matching label file{selectedSamples.length === 1 ? '' : 's'}?</p><div className="heading-actions"><button type="button" className="secondary-button" disabled={busy} onClick={() => setDeleteSelectionArmed(false)}>Cancel</button><button type="button" className="danger-button confirm" disabled={busy} onClick={() => void deleteSelected()}>{busy ? 'Deleting…' : `Yes, delete ${selectedSamples.length}`}</button></div></div>}
            {participant && step ? samples.length ? <div className="collection-image-grid">{samples.map((sample, index) => {
              const selected = selectedSamples.includes(sample.sample_ref);
              return <label key={sample.sample_ref} className={selected ? 'selected' : ''}><input type="checkbox" aria-label={`Select saved image ${index + 1}`} checked={selected} disabled={busy || !!snapshot} onChange={() => toggleSample(sample.sample_ref)} /><img src={`${api}/api/collection/image?participant_id=${encodeURIComponent(participant)}&sample_ref=${encodeURIComponent(sample.sample_ref)}`} alt={`Saved ${pretty(sample.label)} training image ${index + 1}`} /><span><strong>{sample.handedness === 'left' || sample.handedness === 'right' ? `${sample.handedness} hand` : 'No hand'}</strong><small>{orientationLabel(sample.finger_orientation)}</small></span></label>;
            })}</div> : <p className="empty-note">No images saved for this section yet.</p> : <p className="empty-note">Select or create a participant to view saved images.</p>}
          </div>
        </article>
        <article className="panel collection-progress"><div className="panel-header"><div>COLLECTION PROGRESS</div><span>{complete} / {summary?.plan.length ?? 0} PROMPTS COMPLETE</span></div><div className="collection-fields">
          <label>Jump to a gesture / angle<select aria-label="Collection prompt" value={stepIndex} disabled={busy || !!snapshot} onChange={e => changeStep(Number(e.target.value))}>{summary?.plan.map((s, i) => <option value={i} key={s.id}>{stepComplete(summary, s) ? '✓ ' : ''}{i + 1}. {s.title} / {viewLabel(s.view)} ({summary.counts[s.id] ?? 0}/{s.target})</option>)}</select></label>
          <progress value={complete} max={summary?.plan.length || 1} aria-label="Completed collection prompts" />
          <div className="heading-actions"><button className="secondary-button" disabled={stepIndex === 0 || busy || !!snapshot} onClick={() => changeStep(stepIndex - 1)}>Previous prompt</button><button className="secondary-button" disabled={!summary || stepIndex >= summary.plan.length - 1 || busy || !!snapshot} onClick={() => changeStep(stepIndex + 1)}>Skip / next prompt</button></div>
          <p className="collection-help">Directions come first, followed by palm and other commands, then no-gesture examples: other fingers, face/ear, and background. Repeat with both hands and different conditions.</p>
          <div className="collection-recent">
            <div className="collection-recent-heading"><strong>Recent saves</strong><button type="button" className="danger-button" disabled={!lastSave || busy || !!snapshot} onClick={() => setDeleteArmed(true)}>Delete last image</button></div>
            {deleteArmed && lastSave && <div className="collection-delete-confirm" role="alert"><p>Permanently delete the latest image and its label?<small>{pretty(lastSave.label)} · {viewLabel(lastSave.view)}{lastSave.handedness === 'left' || lastSave.handedness === 'right' ? ` · ${lastSave.handedness} hand · ${orientationLabel(lastSave.finger_orientation)}` : ''}</small></p><div className="heading-actions"><button type="button" className="secondary-button" disabled={busy} onClick={() => setDeleteArmed(false)}>Cancel</button><button type="button" className="danger-button confirm" disabled={busy} onClick={() => void deleteLast()}>{busy ? 'Deleting…' : 'Yes, delete last image'}</button></div></div>}
            {summary?.recent.length ? summary.recent.map((r, index) => <div className="collection-recent-entry" key={r.sample_id + r.image_path}><span>{index === 0 ? 'LATEST' : String(index + 1).padStart(2, '0')}</span><div>{pretty(r.label)} · {viewLabel(r.view)}{r.handedness === 'left' || r.handedness === 'right' ? ` · ${r.handedness} hand · ${orientationLabel(r.finger_orientation)}` : ''}<small>{r.image_path}</small></div></div>) : <small>No images saved for this participant yet.</small>}
          </div>
        </div></article>
      </div>
    </div>
  </div>;
}
