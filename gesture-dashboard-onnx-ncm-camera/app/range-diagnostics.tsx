'use client';

import { useEffect, useRef, useState } from 'react';
import { calibrateRange, estimateRange, type PalmObservation, type RangeCalibration } from './range-math';

export type RangePrediction = {
  status?: string; runtime_prediction?: string; action_reason?: string; message?: string;
  confidence?: number; known_gesture_mass?: number; ncm_frame_id?: number;
  diagnostics?: Record<string, { execute?: boolean }>;
  quality?: { detector?: PalmObservation; image?: { source_width?: number; source_height?: number }; landmarks?: { accepted?: boolean } };
  timing?: { total_ms?: number };
};
type Sample = { at: number; measured_m: number; expected: string; hand_found: boolean; command_correct: boolean;
  predicted: string; estimated_m: number | null; palm_px: number | null; processing_ms: number | null; reason: string };
type Trial = { distance: number; expected: string; samples: Sample[] };
const COMMANDS = ['left', 'right', 'up', 'down', 'open_palm', 'like', 'dorsal', 'ok', 'fist', 'thumb_down', 'no_gesture'];

export default function RangeDiagnostics({ prediction, connected }: { prediction: RangePrediction; connected: boolean }) {
  const [calibration, setCalibration] = useState<RangeCalibration | null>(null);
  const [reference, setReference] = useState('0.5');
  const [station, setStation] = useState('1.0');
  const [expected, setExpected] = useState('left');
  const [message, setMessage] = useState('');
  const [trials, setTrials] = useState<Trial[]>([]);
  const [running, setRunning] = useState(false);
  const [clock, setClock] = useState(0);
  const [view, setView] = useState<{ frameAt: number; last: { at: number; distance: number } | null; trialStart: number }>({ frameAt: 0, last: null, trialStart: 0 });
  const samples = useRef<{ at: number; observation: PalmObservation }[]>([]);
  const active = useRef<{ start: number; distance: number; expected: string; samples: Sample[] } | null>(null);
  const lastFrame = useRef<{ prediction: RangePrediction; at: number } | null>(null);
  const lastEstimate = useRef<{ at: number; distance: number } | null>(null);
  const latestCalibration = useRef(calibration);
  const connectedRef = useRef(connected);
  useEffect(() => { latestCalibration.current = calibration; }, [calibration]);
  useEffect(() => { connectedRef.current = connected; }, [connected]);
  useEffect(() => {
    if (!['predicted', 'no_hand'].includes(prediction.status ?? '')) return;
    const now = Date.now();
    lastFrame.current = { prediction, at: now };
    const observation = prediction.quality?.detector;
    if (prediction.quality?.landmarks?.accepted && observation?.palm_segments_px) {
      samples.current = [...samples.current.filter(row => now - row.at < 2500), { at: now, observation }].slice(-20);
      const estimate = estimateRange(latestCalibration.current, observation);
      if (estimate) lastEstimate.current = { at: now, distance: estimate.distance };
    } else samples.current = [];
  }, [prediction]);
  useEffect(() => {
    const timer = setInterval(() => {
      const now = Date.now(); setClock(now);
      setView({ frameAt: lastFrame.current?.at ?? 0, last: lastEstimate.current, trialStart: active.current?.start ?? now });
      const trial = active.current;
      if (!trial) return;
      const frame = lastFrame.current;
      const fresh = connectedRef.current && frame && now - frame.at <= 350;
      const value = fresh ? frame.prediction : undefined;
      const handFound = value?.quality?.landmarks?.accepted === true;
      const estimate = handFound ? estimateRange(latestCalibration.current, value?.quality?.detector) : null;
      const correct = trial.expected === 'no_gesture'
        ? !!value && value.runtime_prediction === 'no_gesture'
        : !!value && value.runtime_prediction === trial.expected && value.diagnostics?.ONNX?.execute === true;
      trial.samples.push({ at: now, measured_m: trial.distance, expected: trial.expected, hand_found: handFound,
        command_correct: correct, predicted: value?.runtime_prediction ?? 'no_frame', estimated_m: estimate?.distance ?? null,
        palm_px: value?.quality?.detector?.palm_scale_px ?? null, processing_ms: value?.timing?.total_ms ?? null,
        reason: (handFound ? value?.action_reason ?? value?.message : value?.message) ?? 'No fresh hand tracking' });
      if (now - trial.start >= 10000) {
        setTrials(previous => [...previous, { distance: trial.distance, expected: trial.expected, samples: trial.samples }]);
        active.current = null; setRunning(false); setMessage('Test saved. Move to another measured distance, then repeat.');
      }
    }, 100);
    return () => clearInterval(timer);
  }, []);
  const fresh = connected && view.frameAt > 0 && clock - view.frameAt < 1200;
  const observation = fresh ? prediction.quality?.detector : undefined;
  const handFound = fresh && prediction.quality?.landmarks?.accepted === true;
  const estimate = handFound ? estimateRange(calibration, observation) : null;
  const calibrate = () => {
    try {
      if (!connected || !lastFrame.current || Date.now() - lastFrame.current.at > 400) throw Error('Connect the camera and show a steady palm first.');
      const rows = samples.current.filter(row => Date.now() - row.at < 2500).map(row => row.observation);
      setCalibration(calibrateRange(rows, Number(reference))); lastEstimate.current = null;
      setMessage('Calibration set for this hand and camera view. Recalibrate after zoom, camera, or person changes.');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Calibration failed.'); }
  };
  const start = () => {
    const distance = Number(station);
    if (!Number.isFinite(distance) || distance < .1 || distance > 20) { setMessage('Enter a measured test distance from 0.1 to 20 m.'); return; }
    active.current = { start: Date.now(), distance, expected, samples: [] }; setRunning(true);
    setMessage('Hold the selected gesture at the measured distance for 10 seconds.');
  };
  const download = () => {
    const rows = trials.flatMap(trial => trial.samples);
    if (!rows.length) return;
    const keys = Object.keys(rows[0]) as (keyof Sample)[];
    const csv = [keys.join(','), ...rows.map(row => keys.map(key => `"${String(row[key] ?? '').replaceAll('"', '""')}"`).join(','))].join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a'); link.href = url; link.download = 'gesture-distance-tests.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <details className="range-panel">
    <summary>Hand distance &amp; detection range <span>{estimate ? `~${estimate.distance.toFixed(2)} m · ` : calibration ? 'Distance unavailable · ' : 'Calibrate distance · '}{handFound ? 'Hand tracked' : connected ? 'Tracking unavailable' : 'Camera offline'}</span></summary>
    <div className="range-readings">
      <div><span>Estimated hand distance</span><strong>{estimate ? `~${estimate.distance.toFixed(2)} m` : calibration ? 'Unavailable' : 'Calibrate first'}</strong>
        <small>{estimate ? `Palm-segment estimates: ${estimate.lower.toFixed(2)}–${estimate.upper.toFixed(2)} m${estimate.inconsistent ? ' · palm angle or tracking is inconsistent' : ''}` :
          view.last ? `Last seen: ~${view.last.distance.toFixed(2)} m, ${Math.max(0, (clock-view.last.at)/1000).toFixed(1)} s ago. Current distance unknown.` : 'No metric distance until a known-distance calibration is set.'}</small></div>
      <div><span>Palm / hand size in analysis pixels</span><strong>{observation?.palm_scale_px != null ? `${observation.palm_scale_px.toFixed(0)} / ${observation.hand_span_px?.toFixed(0)} px` : '—'}</strong>
        <small>{observation?.small_hand ? 'Small palm: tracking is more vulnerable to blur and occlusion.' : 'Fewer pixels means less detail for separating fingers.'}</small></div>
      <div><span>Detection status</span><strong>{handFound ? prediction.runtime_prediction === 'no_gesture' ? 'Gesture rejected' : 'Hand tracked' : 'Hand not tracked'}</strong>
        <small>{fresh ? (handFound ? prediction.action_reason ?? prediction.message ?? 'Waiting for confirmation' : prediction.message ?? 'Hand tracking unavailable') : 'Connect the selected camera to inspect live tracking.'}</small></div>
    </div>
    <div className="range-controls">
      <label>Known calibration distance (m)<input type="number" min="0.1" max="10" step="0.05" value={reference} onChange={event => setReference(event.target.value)} disabled={running}/></label>
      <button type="button" className="secondary-button" onClick={calibrate} disabled={!connected || running}>Calibrate steady palm</button>
      <button type="button" className="secondary-button" onClick={() => { setCalibration(null); lastEstimate.current = null; }} disabled={running || !calibration}>Clear calibration</button>
    </div>
    <p>Measure lens-to-palm distance with a ruler or tape, face your palm toward the camera, and hold still for one second before calibrating. This is an apparent-size estimate; palm rotation, different people, blur, and zoom can bias it. Segment spread is not a statistical confidence interval.</p>
    <div className="range-controls">
      <label>Measured test distance (m)<input type="number" min="0.1" max="20" step="0.1" value={station} onChange={event => setStation(event.target.value)} disabled={running}/></label>
      <label>Gesture to hold<select value={expected} onChange={event => setExpected(event.target.value)} disabled={running}>{COMMANDS.map(name => <option key={name} value={name}>{name.replaceAll('_', ' ')}</option>)}</select></label>
      <button type="button" className="primary-button" onClick={start} disabled={!connected || running}>{running ? `Testing · ${Math.max(0, 10 - (clock - view.trialStart) / 1000).toFixed(0)} s` : 'Run 10-second range test'}</button>
      <button type="button" className="secondary-button" onClick={download} disabled={!trials.length || running}>Export CSV</button>
    </div>
    <p role="status">{message || 'Range tests use your measured distance, so missed landmarks and camera outages still count. Trials are sampled at 10 Hz; keep this tab visible. Results remain here until the page reloads.'}</p>
    {!!trials.length && <div className="table-wrap"><table className="diagnostics-table"><thead><tr><th>Measured distance</th><th>Expected</th><th>Hand tracked</th><th>Correct confirmed gesture</th><th>Observations</th></tr></thead><tbody>{trials.map((trial, index) => <tr key={index}>
      <td>{trial.distance.toFixed(2)} m</td><td>{trial.expected}</td><td>{(100*trial.samples.filter(row => row.hand_found).length/trial.samples.length).toFixed(1)}%</td>
      <td>{(100*trial.samples.filter(row => row.command_correct).length/trial.samples.length).toFixed(1)}%</td><td>{trial.samples.length}</td>
    </tr>)}</tbody></table></div>}
  </details>;
}
