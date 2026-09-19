'use client';

/* Local blob image previews must use native img elements. */
/* eslint-disable @next/next/no-img-element */

import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import RangeDiagnostics, { type RangePrediction } from './range-diagnostics';
import DataCollection, { type CameraSource } from './data-collection';
import { planeAnglesFromLandmarks, validPlaneAngles, type PlaneAngles } from './landmark-angles';

const API_URL = process.env.NEXT_PUBLIC_GESTURE_API_URL ?? 'http://127.0.0.1:8200';
const WS_URL = API_URL.replace(/^http/, 'ws');
const DEFAULT_CLASSES = [
  'left', 'right', 'up', 'down', 'open_palm', 'like', 'dorsal', 'ok', 'fist', 'thumb_down',
];
const DEFAULT_ACTIONS: Record<string, string> = {
  left: 'Move Left',
  right: 'Move Right',
  up: 'Move Up',
  down: 'Move Down',
  open_palm: 'Enable / Disable Object Tracking · Upward Only',
  like: 'Play / Pause',
  dorsal: 'Return to Default Position',
  ok: 'Start / Stop Recording',
  fist: 'Mute',
  thumb_down: 'Volume Down',
};
const DEMO_COMMANDS: Record<string, string> = {
  left: 'MOVE_LEFT',
  right: 'MOVE_RIGHT',
  up: 'MOVE_UP',
  down: 'MOVE_DOWN',
  open_palm: 'TOGGLE_TRACKING',
  like: 'TOGGLE_PLAYBACK',
  dorsal: 'RETURN_HOME',
  ok: 'TOGGLE_RECORDING',
  fist: 'MUTE_AUDIO',
  thumb_down: 'VOLUME_DOWN',
};
const CANONICAL_CLASS_SET = new Set(DEFAULT_CLASSES);
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

type Artifact = {
  label: string;
  pattern: string;
  present: boolean;
  required: boolean;
  files: string[];
  bytes: number;
};
type Timing = {
  mediapipe_ms?: number;
  feature_ms?: number;
  classifier_ms?: number;
  diagnostics_ms?: number;
  total_ms?: number;
  uncapped_fps?: number;
  configured_fps?: number;
  effective_application_fps?: number;
  frame_budget_ms?: number;
  budget_used_percent?: number;
  headroom_ms?: number;
  target_fps_capacity_pass?: boolean;
  ten_fps_capacity_pass?: boolean;
  verdict?: 'PASS' | 'FAIL';
};
type Diagnostic = {
  used_for_runtime: boolean;
  raw_prediction: string;
  prediction: string;
  confidence: number;
  stable_frames: number;
  execute: boolean;
  reason: string;
  classifier_ms: number;
};
type Health = {
  status: string;
  operational_ready?: boolean;
  engine: {
    ready: boolean;
    model: {
      ready: boolean;
      available_models: string[];
      selectable_models?: string[];
      selected_model_name: string;
      follow_object_ensemble_ready?: boolean;
      errors: string[];
    };
    mediapipe: { ready: boolean; error?: string };
  };
  config: {
    class_names: string[];
    feedback_labels?: string[];
    gesture_to_action: Record<string, string>;
    target_fps: number;
    frame_interval_ms: number;
    frame_budget_ms: number;
    ema_alpha: number;
    confidence_floor: number;
    stable_frames_required: number;
    roi_size_ratio: number;
    follow_hold_seconds: number;
    follow_success_display_seconds: number;
  };
  artifacts: Artifact[];
  online_learning?: {
    ready: boolean;
    accepted_updates: number;
    forced_updates: number;
    rejected_updates: number;
    validation_macro_f1?: number;
    last_updated_utc?: string;
    force_learning_enabled?: boolean;
    backup_count?: number;
    tflite_policy: string;
  };
  artifact_integrity?: {
    verified: boolean;
    status: string;
    release_fingerprint?: string;
    checked_files: number;
    errors: string[];
  };
  production?: {
    environment: string;
    production_mode: boolean;
    allow_force_learning: boolean;
    allow_artifact_reload: boolean;
    max_active_sessions: number;
    release_qualification?: {
      status: string;
      current?: boolean;
      pc_release_ready: boolean;
      selected_model?: string;
      gated_macro_f1?: number;
      gated_minimum_per_class_f1?: number;
      deferred_classes?: string[];
      failing_gated_classes?: string[];
      mobile_parity?: { current?: boolean };
      onnx_parity?: {
        passed?: boolean;
        prediction_agreement?: number;
        maximum_absolute_probability_error?: number;
        validation_samples?: number;
      };
      message?: string;
    };
  };
  runtime_metrics?: {
    frames_total: number;
    errors_total: number;
    error_rate: number;
    target_fps_budget_pass_rate?: number;
    ten_fps_budget_pass_rate?: number;
    active_sessions: number;
    peak_sessions: number;
    timing: Record<string, { count: number; mean?: number; p50?: number; p95?: number; p99?: number }>;
  };
  ncm_camera?: NcmStatus;
};
type NcmStatus = {
  state: string;
  running: boolean;
  connected: boolean;
  camera_fps: number;
  frames_received: number;
  packets_received: number;
  bytes_received: number;
  crc_errors: number;
  header_errors: number;
  invalid_jpeg_frames: number;
  sequence_gaps: number;
  reconnects: number;
  last_error?: string;
  last_frame_utc?: string;
  device_metadata?: Record<string, unknown> | string;
  config: {
    host_ip: string;
    device_ip: string;
    tcp_port: number;
    discovery_port: number;
    discovery_payload: string;
  };
};
type Prediction = RangePrediction & {
  status: string;
  message?: string;
  model?: string;
  raw_prediction?: string;
  runtime_prediction?: string;
  mapped_action?: string;
  runtime_action?: string;
  action_reason?: string;
  confidence?: number;
  stable_frames?: number;
  held_seconds?: number;
  probabilities?: Record<string, number>;
  raw_probabilities?: Record<string, number>;
  base_probabilities?: Record<string, number>;
  feature_vector?: number[];
  landmarks?: number[][];
  display_landmarks?: number[][];
  world_landmarks?: number[][];
  landmarks_3d?: number[][];
  plane_angles?: Partial<PlaneAngles>;
  actual_fps?: number;
  camera_fps?: number;
  ncm_camera?: NcmStatus;
  ncm_frame_id?: number;
  timing?: Timing;
  diagnostics?: Record<string, Diagnostic>;
};
type MetricResponse = { files: string[]; rows: Record<string, Record<string, string>[]> };
type ActionEvent = { action: string; prediction: string; confidence: number; session_id?: string };
type DemoEvent = {
  command: string;
  gesture: string;
  detail: string;
  at: string;
  status: 'performed' | 'ignored' | 'failed';
};
type ActionToast = { label: string; gesture: string; status: 'performed' | 'failed' };
type DemoMediaAsset = {
  url: string;
  name: string;
  kind: 'video' | 'image';
  duration?: number;
  width?: number;
  height?: number;
};
type VideoTransform = { x: number; y: number; scale: number };
type LearningMode = 'audit' | 'safe';

function canonicalGesture(value?: string) {
  if (!value) return undefined;
  return CANONICAL_CLASS_SET.has(value) ? value : undefined;
}

function humanize(value?: string) {
  if (!value) return 'Waiting';
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}
function percent(value?: number) { return `${((value ?? 0) * 100).toFixed(1)}%`; }
function milliseconds(value?: number) { return value == null ? '—' : `${value.toFixed(1)} ms`; }
function fps(value?: number | null) { return value == null || !Number.isFinite(value) ? '—' : value.toFixed(2); }

function WebcamVideo({ stream, className }: { stream: MediaStream | null; className: string }) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.srcObject = stream;
    if (stream) void video.play().catch(() => undefined);
    return () => { video.srcObject = null; };
  }, [stream]);
  return <video ref={videoRef} className={className} autoPlay muted playsInline aria-label="Live unmirrored PC webcam" />;
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'live' | 'analytics' | 'feedback' | 'setup' | 'data_collection'>('live');
  const [health, setHealth] = useState<Health | null>(null);
  const [metrics, setMetrics] = useState<MetricResponse>({ files: [], rows: {} });
  const [actions, setActions] = useState<ActionEvent[]>([]);
  const [prediction, setPrediction] = useState<Prediction>({ status: 'idle' });
  const [connection, setConnection] = useState<'offline' | 'connecting' | 'online'>('offline');
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraSource, setCameraSource] = useState<CameraSource>('ncm');
  const [activeCameraSource, setActiveCameraSource] = useState<CameraSource | null>(null);
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [webcamDetails, setWebcamDetails] = useState('Browser webcam has not been started yet.');
  const [cameraFps, setCameraFps] = useState<number | null>(null);
  const [ncmStatus, setNcmStatus] = useState<NcmStatus | null>(null);
  const [ncmMessage, setNcmMessage] = useState('Development-board link has not been tested yet.');
  const [streamRevision, setStreamRevision] = useState(0);
  const [selectedModel, setSelectedModel] = useState('');
  const [feedbackCount, setFeedbackCount] = useState(0);
  const [actualLabel, setActualLabel] = useState('no_gesture');
  const [learningMode, setLearningMode] = useState<LearningMode>('safe');
  const [feedbackNote, setFeedbackNote] = useState('');
  const [feedbackMessage, setFeedbackMessage] = useState('');
  const [uploading, setUploading] = useState(false);
  const [snapshotReady, setSnapshotReady] = useState(false);
  const [demoVideo, setDemoVideo] = useState<DemoMediaAsset | null>(null);
  const [demoVideoMessage, setDemoVideoMessage] = useState('Upload a video (up to 60 seconds) or an image to enable the action demo.');
  const [demoTransform, setDemoTransform] = useState<VideoTransform>({ x: 0, y: 0, scale: 1 });
  const [demoEvents, setDemoEvents] = useState<DemoEvent[]>([]);
  const [demoRecording, setDemoRecording] = useState(false);
  const [recordingDownloadUrl, setRecordingDownloadUrl] = useState<string | null>(null);
  const [objectTrackingEnabled, setObjectTrackingEnabled] = useState(false);
  const [actionToast, setActionToast] = useState<ActionToast | null>(null);
  const [imagePreviewActive, setImagePreviewActive] = useState(true);
  const [demoMuted, setDemoMuted] = useState(false);
  const [demoVolume, setDemoVolume] = useState(1);

  const demoVideoRef = useRef<HTMLVideoElement | null>(null);
  const demoImageRef = useRef<HTMLImageElement | null>(null);
  const demoRecordingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const landmarkCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const feedbackLandmarkCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const webcamVideoRef = useRef<HTMLVideoElement | null>(null);
  const webcamCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const webcamStreamRef = useRef<MediaStream | null>(null);
  const cameraSourceRef = useRef<CameraSource>('ncm');
  const activeCameraSourceRef = useRef<CameraSource | null>(null);
  const webcamFrameTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const webcamFramePendingRef = useRef(false);
  const lastSnapshotRef = useRef<string | null>(null);
  const stopCameraRef = useRef<(preservePrediction?: boolean) => void>(() => undefined);
  const executeDemoPredictionRef = useRef<(message: Prediction) => void>(() => undefined);
  const demoVideoAssetRef = useRef<DemoMediaAsset | null>(null);
  const demoTransformRef = useRef<VideoTransform>({ x: 0, y: 0, scale: 1 });
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordingChunksRef = useRef<Blob[]>([]);
  const lastExecutedGestureRef = useRef<string | null>(null);
  const gestureReleaseFramesRef = useRef(0);
  const actionToastTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const recordingFrameTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const recordingStreamRef = useRef<MediaStream | null>(null);
  const cameraStartPendingRef = useRef(false);

  const targetFps = health?.config.target_fps ?? 10;
  const frameBudgetMs = health?.config.frame_budget_ms ?? prediction.timing?.frame_budget_ms ?? (1000 / targetFps);
  const stableRequired = health?.config.stable_frames_required ?? 3;
  const classNames = DEFAULT_CLASSES;
  const actionMap = DEFAULT_ACTIONS;
  const feedbackLabels = useMemo(() => {
    const configured = health?.config.feedback_labels;
    if (!configured?.length) return [...DEFAULT_CLASSES, 'no_gesture'];
    const normalized = configured
      .map((label) => label === 'no_gesture' ? label : canonicalGesture(label))
      .filter((label): label is string => Boolean(label));
    const labels = [...new Set(normalized)];
    if (!labels.includes('no_gesture')) labels.push('no_gesture');
    return labels;
  }, [health?.config.feedback_labels]);
  const selectedFeedbackLabel = feedbackLabels.includes(actualLabel)
    ? actualLabel
    : feedbackLabels[0] ?? 'no_gesture';
  const selectableModels = health?.engine.model.selectable_models ?? health?.engine.model.available_models ?? [];
  const serverReady = Boolean(health?.operational_ready ?? health?.engine.ready);
  const qualification = health?.production?.release_qualification;
  const runtimeSummary = health?.runtime_metrics;
  const onlineLearning = health?.online_learning;
  const runtimeGesture = canonicalGesture(prediction.runtime_prediction);
  const feedbackPrediction = runtimeGesture ?? (prediction.runtime_prediction === 'no_gesture' ? 'no_gesture' : undefined);
  const mappedLabel = (runtimeGesture ? actionMap[runtimeGesture] : undefined)
    ?? 'Waiting for prediction';
  const timing = prediction.timing;
  const pipelineVerdict = timing?.verdict ?? 'WAITING';
  const targetCapacityPass = timing?.target_fps_capacity_pass ?? timing?.ten_fps_capacity_pass;
  const targetBudgetPassRate = runtimeSummary?.target_fps_budget_pass_rate
    ?? runtimeSummary?.ten_fps_budget_pass_rate;

  const refreshServerData = async () => {
    try {
      const [healthResponse, metricResponse, actionResponse, feedbackResponse] = await Promise.all([
        fetch(`${API_URL}/api/health`),
        fetch(`${API_URL}/api/metrics`),
        fetch(`${API_URL}/api/actions`),
        fetch(`${API_URL}/api/feedback`),
      ]);
      if (!healthResponse.ok) throw new Error('Health request failed.');
      const healthData = await healthResponse.json() as Health;
      setHealth(healthData);
      setNcmStatus(healthData.ncm_camera ?? null);
      if (cameraSourceRef.current === 'ncm' && activeCameraSourceRef.current !== 'webcam') setCameraFps(healthData.ncm_camera?.camera_fps ?? null);
      const choices = healthData.engine.model.selectable_models ?? healthData.engine.model.available_models;
      setSelectedModel((current) => current && choices.includes(current)
        ? current
        : healthData.engine.model.selected_model_name || choices[0] || '');
      if (metricResponse.ok) setMetrics(await metricResponse.json() as MetricResponse);
      if (actionResponse.ok) {
        const actionData = await actionResponse.json() as { recent?: ActionEvent[] };
        const canonicalActions = (actionData.recent ?? []).flatMap((item) => {
          const gesture = canonicalGesture(item.prediction);
          return gesture ? [{ ...item, prediction: gesture, action: DEFAULT_ACTIONS[gesture] }] : [];
        });
        setActions(canonicalActions);
      }
      if (feedbackResponse.ok) {
        const feedbackData = await feedbackResponse.json() as { count?: number };
        setFeedbackCount(feedbackData.count ?? 0);
      }
    } catch {
      setHealth(null);
    }
  };

  useEffect(() => {
    const initial = setTimeout(refreshServerData, 0);
    const refresh = setInterval(refreshServerData, 10000);
    return () => { clearTimeout(initial); clearInterval(refresh); };
  }, []);

  useEffect(() => () => {
    if (actionToastTimerRef.current) clearTimeout(actionToastTimerRef.current);
    if (recordingFrameTimerRef.current) clearInterval(recordingFrameTimerRef.current);
    if (mediaRecorderRef.current?.state !== 'inactive') mediaRecorderRef.current?.stop();
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (webcamFrameTimerRef.current) clearInterval(webcamFrameTimerRef.current);
    webcamStreamRef.current?.getTracks().forEach((track) => track.stop());
    socketRef.current?.close();
    if (demoVideoAssetRef.current) URL.revokeObjectURL(demoVideoAssetRef.current.url);
  }, []);

  useEffect(() => { demoVideoAssetRef.current = demoVideo; }, [demoVideo]);
  useEffect(() => { demoTransformRef.current = demoTransform; }, [demoTransform]);
  useEffect(() => () => {
    if (recordingDownloadUrl) URL.revokeObjectURL(recordingDownloadUrl);
  }, [recordingDownloadUrl]);

  const probabilityRows = useMemo(() => {
    const values = prediction.probabilities ?? Object.fromEntries(classNames.map((name) => [name, 0]));
    return classNames
      .map((name) => ({ name, value: values[name] ?? 0 }))
      .sort((first, second) => second.value - first.value);
  }, [prediction.probabilities, classNames]);

  const diagnostics = useMemo(
    () => Object.entries(prediction.diagnostics ?? {}),
    [prediction.diagnostics],
  );

  const livePlaneAngles = useMemo(() => {
    if (validPlaneAngles(prediction.plane_angles)) return prediction.plane_angles;
    return planeAnglesFromLandmarks(prediction.world_landmarks ?? prediction.landmarks_3d);
  }, [prediction.landmarks_3d, prediction.plane_angles, prediction.world_landmarks]);

  const drawLandmarksOnCanvas = useCallback((canvas: HTMLCanvasElement | null, landmarks?: number[][]) => {
    if (!canvas) return;
    const size = Math.max(1, Math.round(canvas.clientWidth || 400));
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, size, size);
    if (!landmarks || landmarks.length !== 21) return;
    context.lineWidth = Math.max(2, size / 180);
    context.strokeStyle = 'rgba(73, 217, 209, .92)';
    context.beginPath();
    HAND_CONNECTIONS.forEach(([from, to]) => {
      context.moveTo(landmarks[from][0] * size, landmarks[from][1] * size);
      context.lineTo(landmarks[to][0] * size, landmarks[to][1] * size);
    });
    context.stroke();
    landmarks.forEach(([x, y], index) => {
      context.beginPath();
      context.fillStyle = index === 0 ? '#ffb454' : '#f4f7ff';
      context.arc(x * size, y * size, Math.max(2.4, size / 120), 0, Math.PI * 2);
      context.fill();
    });
  }, []);

  const drawLandmarks = useCallback((landmarks?: number[][]) => {
    drawLandmarksOnCanvas(landmarkCanvasRef.current, landmarks);
    drawLandmarksOnCanvas(feedbackLandmarkCanvasRef.current, landmarks);
  }, [drawLandmarksOnCanvas]);

  useEffect(() => { drawLandmarks(prediction.display_landmarks ?? prediction.landmarks); }, [prediction.display_landmarks, prediction.landmarks, activeTab, drawLandmarks]);

  const captureWebcamFrame = useCallback(async () => {
    const video = webcamVideoRef.current;
    const canvas = webcamCanvasRef.current;
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA || !video.videoWidth || !video.videoHeight) {
      throw new Error('The webcam is still warming up. Wait for the live preview and try again.');
    }
    const maximumWidth = 960;
    const scale = Math.min(1, maximumWidth / video.videoWidth);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext('2d');
    if (!context) throw new Error('The browser could not prepare a webcam frame.');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    return await new Promise<Blob>((resolve, reject) => canvas.toBlob(
      blob => blob ? resolve(blob) : reject(new Error('The browser could not encode a webcam frame.')),
      'image/jpeg',
      .88,
    ));
  }, []);

  const stopCamera = (preservePrediction = false) => {
    cameraStartPendingRef.current = false;
    if (webcamFrameTimerRef.current) clearInterval(webcamFrameTimerRef.current);
    webcamFrameTimerRef.current = null;
    webcamFramePendingRef.current = false;
    socketRef.current?.close();
    socketRef.current = null;
    webcamStreamRef.current?.getTracks().forEach(track => track.stop());
    webcamStreamRef.current = null;
    setWebcamStream(null);
    if (activeCameraSourceRef.current === 'ncm') {
      void fetch(`${API_URL}/api/ncm/disconnect`, { method: 'POST' }).finally(refreshServerData);
    }
    activeCameraSourceRef.current = null;
    setActiveCameraSource(null);
    setCameraActive(false);
    setCameraFps(null);
    setConnection('offline');
    if (!preservePrediction) {
      setPrediction({ status: 'idle', message: 'Camera disconnected.' });
      setSnapshotReady(false);
      drawLandmarks();
    }
  };

  useEffect(() => {
    stopCameraRef.current = stopCamera;
  });

  const handlePrediction = (message: Prediction) => {
    setPrediction(message);
    setSnapshotReady(true);
    executeDemoPredictionRef.current(message);
    if (message.runtime_action && message.runtime_action !== 'Wait / No Action') {
      const canonicalPrediction = canonicalGesture(message.runtime_prediction);
      if (canonicalPrediction) {
        setActions((current) => [{
          action: DEFAULT_ACTIONS[canonicalPrediction],
          prediction: canonicalPrediction,
          confidence: message.confidence ?? 0,
        }, ...current].slice(0, 30));
      }
    }
  };

  const startNcmCamera = async () => {
    if (cameraActive || cameraStartPendingRef.current || socketRef.current) return;
    cameraStartPendingRef.current = true;
    activeCameraSourceRef.current = 'ncm';
    setActiveCameraSource('ncm');
    setConnection('connecting');
    setPrediction({
      status: 'starting',
      message: 'Connecting to the development-board camera over USB-NCM...',
    });
    try {
      const connectResponse = await fetch(`${API_URL}/api/ncm/connect`, { method: 'POST' });
      if (!connectResponse.ok) throw new Error('The backend could not start the NCM camera client.');
      const connectData = await connectResponse.json() as { message?: string; camera?: NcmStatus };
      if (connectData.camera) setNcmStatus(connectData.camera);
      setNcmMessage(connectData.message ?? 'NCM camera connection started.');
      setStreamRevision((current) => current + 1);
      const socket = new WebSocket(`${WS_URL}/ws/ncm-live`);
      socketRef.current = socket;
      socket.onopen = () => {
        cameraStartPendingRef.current = false;
        setConnection('online');
        setCameraActive(true);
        activeCameraSourceRef.current = 'ncm';
        setActiveCameraSource('ncm');
        if (selectedModel) socket.send(JSON.stringify({ type: 'select_model', model: selectedModel }));
      };
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'prediction') {
          handlePrediction(message);
          setCameraFps(message.camera_fps ?? null);
          if (message.ncm_camera) setNcmStatus(message.ncm_camera);
        } else if (message.type === 'ncm_status' && message.camera) {
          setNcmStatus(message.camera);
          setCameraFps(message.camera.camera_fps ?? null);
          if (message.camera.last_error) setNcmMessage(message.camera.last_error);
        } else if (message.type === 'error') {
          setPrediction({ status: 'error', message: message.message });
        }
      };
      socket.onerror = () => {
        cameraStartPendingRef.current = false;
        setConnection('offline');
        setCameraActive(false);
        setPrediction({ status: 'error', message: 'The NCM inference WebSocket could not be reached.' });
        socket.close();
      };
      socket.onclose = () => {
        if (socketRef.current !== socket) return;
        socketRef.current = null;
        activeCameraSourceRef.current = null;
        setActiveCameraSource(null);
        cameraStartPendingRef.current = false;
        setConnection('offline');
        setCameraActive(false);
      };
    } catch (error) {
      cameraStartPendingRef.current = false;
      stopCameraRef.current();
      setPrediction({
        status: 'camera_error',
        message: error instanceof Error ? error.message : 'NCM camera connection failed.',
      });
    }
  };

  const startWebcam = async () => {
    if (cameraActive || cameraStartPendingRef.current || socketRef.current) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setPrediction({ status: 'camera_error', message: 'This browser does not provide webcam access.' });
      return;
    }
    cameraStartPendingRef.current = true;
    activeCameraSourceRef.current = 'webcam';
    setActiveCameraSource('webcam');
    setConnection('connecting');
    setPrediction({ status: 'starting', message: 'Requesting access to the PC webcam...' });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: targetFps, max: 30 } },
      });
      webcamStreamRef.current = stream;
      setWebcamStream(stream);
      const video = webcamVideoRef.current;
      if (!video) throw new Error('The webcam preview is not available.');
      video.srcObject = stream;
      await video.play();
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings();
      const label = track?.label || 'PC webcam';
      setWebcamDetails(`${label} · ${settings?.width ?? video.videoWidth} × ${settings?.height ?? video.videoHeight} · unmirrored`);
      setCameraFps(settings?.frameRate ?? targetFps);

      const socket = new WebSocket(`${WS_URL}/ws/live`);
      socket.binaryType = 'arraybuffer';
      socketRef.current = socket;
      const sendFrame = async () => {
        if (webcamFramePendingRef.current || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount > 1_000_000) return;
        webcamFramePendingRef.current = true;
        try {
          const frame = await captureWebcamFrame();
          if (socket.readyState === WebSocket.OPEN) socket.send(await frame.arrayBuffer());
        } catch (error) {
          setPrediction({ status: 'camera_error', message: error instanceof Error ? error.message : 'Webcam capture failed.' });
        } finally {
          webcamFramePendingRef.current = false;
        }
      };
      socket.onopen = () => {
        cameraStartPendingRef.current = false;
        setConnection('online');
        setCameraActive(true);
        activeCameraSourceRef.current = 'webcam';
        setActiveCameraSource('webcam');
        if (selectedModel) socket.send(JSON.stringify({ type: 'select_model', model: selectedModel }));
        void sendFrame();
        webcamFrameTimerRef.current = setInterval(() => void sendFrame(), Math.max(50, Math.round(1000 / targetFps)));
      };
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'prediction') {
          handlePrediction(message);
          setCameraFps(message.actual_fps ?? settings?.frameRate ?? targetFps);
        } else if (message.type === 'error') {
          setPrediction({ status: 'error', message: message.message });
        }
      };
      socket.onerror = () => {
        cameraStartPendingRef.current = false;
        setPrediction({ status: 'error', message: 'The webcam inference WebSocket could not be reached.' });
        socket.close();
      };
      socket.onclose = () => {
        if (socketRef.current !== socket) return;
        socketRef.current = null;
        if (webcamFrameTimerRef.current) clearInterval(webcamFrameTimerRef.current);
        webcamFrameTimerRef.current = null;
        webcamStreamRef.current?.getTracks().forEach(item => item.stop());
        webcamStreamRef.current = null;
        setWebcamStream(null);
        activeCameraSourceRef.current = null;
        setActiveCameraSource(null);
        cameraStartPendingRef.current = false;
        setConnection('offline');
        setCameraActive(false);
      };
    } catch (error) {
      cameraStartPendingRef.current = false;
      stopCameraRef.current();
      setPrediction({
        status: 'camera_error',
        message: error instanceof Error ? error.message : 'Webcam connection failed.',
      });
    }
  };

  const startCamera = async () => {
    if (cameraSource === 'webcam') await startWebcam();
    else await startNcmCamera();
  };

  const toggleCamera = () => {
    if (cameraStartPendingRef.current || connection === 'connecting') return;
    if (cameraActive) stopCameraRef.current();
    else void startCamera();
  };

  const changeCameraSource = (source: CameraSource) => {
    if (cameraActive || cameraStartPendingRef.current) return;
    cameraSourceRef.current = source;
    setCameraSource(source);
    setCameraFps(null);
    setSnapshotReady(false);
    setPrediction({ status: 'idle', message: source === 'webcam' ? 'PC webcam selected.' : 'NCM board camera selected.' });
    drawLandmarks();
  };

  const discoverNcmCamera = async () => {
    setNcmMessage('Sending JL-CAMERA-DISCOVER over UDP...');
    try {
      const response = await fetch(`${API_URL}/api/ncm/discover`, { method: 'POST' });
      const result = await response.json() as {
        ok: boolean;
        message?: string;
        peer_ip?: string;
        peer_port?: number;
        round_trip_ms?: number;
      };
      setNcmMessage(result.ok
        ? `Board replied from ${result.peer_ip}:${result.peer_port} in ${result.round_trip_ms} ms.`
        : result.message ?? 'The development board did not answer discovery.');
      await refreshServerData();
    } catch {
      setNcmMessage('The local backend could not run the NCM discovery test.');
    }
  };

  const resetSession = () => {
    socketRef.current?.send(JSON.stringify({ type: 'reset' }));
    lastSnapshotRef.current = null;
    setSnapshotReady(false);
    setPrediction({ status: 'reset', message: 'Temporal smoothing and gesture state reset.' });
    drawLandmarks();
  };

  const changeModel = (event: ChangeEvent<HTMLSelectElement>) => {
    const model = event.target.value;
    setSelectedModel(model);
    socketRef.current?.send(JSON.stringify({ type: 'select_model', model }));
  };

  const addDemoEvent = (
    command: string,
    gesture: string,
    detail: string,
    status: DemoEvent['status'] = 'performed',
    toastLabel = command.replaceAll('_', ' '),
  ) => {
    const event: DemoEvent = {
      command,
      gesture,
      detail,
      status,
      at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    };
    setDemoEvents((current) => [event, ...current].slice(0, 20));
    setDemoVideoMessage(detail);
    if (status !== 'ignored') {
      setActionToast({ label: status === 'failed' ? 'ACTION FAILED' : toastLabel, gesture, status });
      if (actionToastTimerRef.current) clearTimeout(actionToastTimerRef.current);
      actionToastTimerRef.current = setTimeout(() => setActionToast(null), 1800);
    }
  };

  const updateDemoTransform = (update: (current: VideoTransform) => VideoTransform) => {
    const next = update(demoTransformRef.current);
    demoTransformRef.current = next;
    setDemoTransform(next);
  };

  const drawDemoRecordingFrame = () => {
    const canvas = demoRecordingCanvasRef.current;
    const asset = demoVideoAssetRef.current;
    const source = asset?.kind === 'video' ? demoVideoRef.current : demoImageRef.current;
    if (!canvas || !asset || !source) return;
    const sourceWidth = asset.kind === 'video'
      ? (source as HTMLVideoElement).videoWidth
      : (source as HTMLImageElement).naturalWidth;
    const sourceHeight = asset.kind === 'video'
      ? (source as HTMLVideoElement).videoHeight
      : (source as HTMLImageElement).naturalHeight;
    if (!sourceWidth || !sourceHeight) return;
    const context = canvas.getContext('2d');
    if (!context) return;
    const { x, y, scale } = demoTransformRef.current;
    context.fillStyle = '#060b13';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const fit = Math.min((canvas.width * 0.82) / sourceWidth, (canvas.height * 0.82) / sourceHeight);
    const drawWidth = sourceWidth * fit;
    const drawHeight = sourceHeight * fit;
    context.save();
    context.translate(
      canvas.width / 2 + (x / 100) * drawWidth,
      canvas.height / 2 + (y / 100) * drawHeight,
    );
    context.scale(scale, scale);
    context.drawImage(source, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight);
    context.restore();
    context.strokeStyle = 'rgba(73, 217, 209, .72)';
    context.lineWidth = 3;
    context.strokeRect(canvas.width * 0.035, canvas.height * 0.055, canvas.width * 0.93, canvas.height * 0.89);
  };

  const finishRecordingResources = () => {
    if (recordingFrameTimerRef.current) clearInterval(recordingFrameTimerRef.current);
    recordingFrameTimerRef.current = null;
    recordingStreamRef.current?.getTracks().forEach((track) => track.stop());
    recordingStreamRef.current = null;
  };

  const startDemoRecording = async (): Promise<string> => {
    const asset = demoVideoAssetRef.current;
    const canvas = demoRecordingCanvasRef.current as (HTMLCanvasElement & {
      captureStream?: (frameRate?: number) => MediaStream;
    }) | null;
    if (!asset || !canvas) throw new Error('Upload a demonstration video or image first.');
    if (mediaRecorderRef.current?.state === 'recording') return 'Recording is already active.';
    if (!canvas.captureStream || typeof MediaRecorder === 'undefined') {
      throw new Error('Demo recording requires current Chrome or Edge on Windows.');
    }
    setRecordingDownloadUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    if (asset.kind === 'video') await demoVideoRef.current?.play();
    canvas.width = 1280;
    canvas.height = 720;
    drawDemoRecordingFrame();
    if (recordingFrameTimerRef.current) clearInterval(recordingFrameTimerRef.current);
    recordingFrameTimerRef.current = setInterval(drawDemoRecordingFrame, 100);
    const stream = canvas.captureStream(10);
    recordingStreamRef.current = stream;
    const mimeCandidates = ['video/webm;codecs=vp9,opus', 'video/webm;codecs=vp8,opus', 'video/webm'];
    const mimeType = mimeCandidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recordingChunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size) recordingChunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(recordingChunksRef.current, { type: recorder.mimeType || 'video/webm' });
      if (blob.size) {
        const url = URL.createObjectURL(blob);
        setRecordingDownloadUrl((current) => {
          if (current) URL.revokeObjectURL(current);
          return url;
        });
      }
      finishRecordingResources();
      setDemoRecording(false);
      mediaRecorderRef.current = null;
    };
    recorder.onerror = () => {
      finishRecordingResources();
      setDemoRecording(false);
      setDemoVideoMessage('The browser could not finish the demonstration recording.');
    };
    mediaRecorderRef.current = recorder;
    recorder.start(500);
    setDemoRecording(true);
    return `Recording started for the uploaded ${asset.kind}. Show OK again to stop and create the downloadable WebM clip.`;
  };

  const stopDemoRecording = (): string => {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === 'inactive') return 'No demonstration recording is active.';
    recorder.stop();
    return 'Recording ended. The download will appear when the browser finishes the file.';
  };

  const performDemoCommand = async (command: string, gesture: string) => {
    const asset = demoVideoAssetRef.current;
    const targetVideo = demoVideoRef.current;
    const targetImage = demoImageRef.current;
    if (!asset || (asset.kind === 'video' ? !targetVideo : !targetImage)) return;
    try {
      let detail = '';
      let toastLabel = command.replaceAll('_', ' ');
      switch (command) {
        case 'TOGGLE_PLAYBACK':
          if (asset.kind === 'image') {
            const next = !imagePreviewActive;
            setImagePreviewActive(next);
            detail = next ? 'Uploaded image preview continued.' : 'Uploaded image preview stopped temporarily.';
            toastLabel = next ? 'IMAGE CONTINUED' : 'IMAGE STOPPED';
          } else if (targetVideo?.paused) {
            await targetVideo.play();
            detail = 'Uploaded video continued.';
            toastLabel = 'VIDEO CONTINUED';
          } else {
            targetVideo?.pause();
            detail = 'Uploaded video stopped temporarily.';
            toastLabel = 'VIDEO PAUSED';
          }
          break;
        case 'TOGGLE_RECORDING':
          if (mediaRecorderRef.current?.state === 'recording') {
            detail = stopDemoRecording();
            toastLabel = 'RECORDING STOPPED';
          } else {
            detail = await startDemoRecording();
            toastLabel = detail.startsWith('Recording is') ? 'RECORDING ALREADY ACTIVE' : 'RECORDING STARTED';
          }
          break;
        case 'MUTE_AUDIO':
          if (asset.kind === 'image') {
            detail = 'Mute recognized. The uploaded image has no audio track.';
            toastLabel = 'MUTE · NO AUDIO TRACK';
          } else {
            if (targetVideo) targetVideo.muted = true;
            setDemoMuted(true);
            detail = 'Uploaded video muted.';
            toastLabel = 'AUDIO MUTED';
          }
          break;
        case 'VOLUME_DOWN':
          if (asset.kind === 'image') {
            detail = 'Volume Down recognized. The uploaded image has no audio track.';
            toastLabel = 'VOLUME DOWN · NO AUDIO TRACK';
          } else if (targetVideo) {
            const nextVolume = Math.max(0, Math.round((targetVideo.volume - .15) * 100) / 100);
            targetVideo.volume = nextVolume;
            setDemoVolume(nextVolume);
            detail = `Uploaded video volume reduced to ${Math.round(nextVolume * 100)}%.`;
            toastLabel = `VOLUME ${Math.round(nextVolume * 100)}%`;
          }
          break;
        case 'MOVE_UP':
          updateDemoTransform((current) => ({ ...current, y: Math.max(-24, current.y - 8) }));
          detail = 'Video target moved up.';
          toastLabel = 'MOVED UP';
          break;
        case 'MOVE_DOWN':
          updateDemoTransform((current) => ({ ...current, y: Math.min(24, current.y + 8) }));
          detail = 'Video target moved down.';
          toastLabel = 'MOVED DOWN';
          break;
        case 'MOVE_LEFT':
          updateDemoTransform((current) => ({ ...current, x: Math.max(-24, current.x - 8) }));
          detail = 'Video target moved left.';
          toastLabel = 'MOVED LEFT';
          break;
        case 'MOVE_RIGHT':
          updateDemoTransform((current) => ({ ...current, x: Math.min(24, current.x + 8) }));
          detail = 'Video target moved right.';
          toastLabel = 'MOVED RIGHT';
          break;
        case 'TOGGLE_TRACKING': {
          const next = !objectTrackingEnabled;
          setObjectTrackingEnabled(next);
          detail = next
            ? 'Object tracking enabled. Directional gestures reposition the tracked target.'
            : 'Object tracking disabled.';
          toastLabel = next ? 'OBJECT TRACKING ENABLED' : 'OBJECT TRACKING DISABLED';
          break;
        }
        case 'RETURN_HOME':
          updateDemoTransform(() => ({ x: 0, y: 0, scale: 1 }));
          setObjectTrackingEnabled(false);
          setImagePreviewActive(true);
          detail = 'Target returned to its default position and object tracking was cleared.';
          toastLabel = 'RETURNED TO DEFAULT POSITION';
          break;
        default:
          return;
      }
      addDemoEvent(command, gesture, detail, 'performed', toastLabel);
    } catch (error) {
      addDemoEvent(
        command,
        gesture,
        error instanceof Error ? error.message : 'The action could not be performed.',
        'failed',
        'ACTION FAILED',
      );
    }
  };

  const executeDemoPrediction = (message: Prediction) => {
    const reportedGesture = message.runtime_prediction;
    if (!reportedGesture || !demoVideoAssetRef.current) return;
    const gesture = canonicalGesture(reportedGesture);
    const lastGesture = lastExecutedGestureRef.current;
    if (reportedGesture === 'no_gesture') {
      lastExecutedGestureRef.current = null;
      gestureReleaseFramesRef.current = 0;
      return;
    }
    if (!gesture) return;
    if (lastGesture && message.raw_prediction && canonicalGesture(message.raw_prediction) !== lastGesture) {
      gestureReleaseFramesRef.current += 1;
      if (gestureReleaseFramesRef.current >= 2) {
        lastExecutedGestureRef.current = null;
        gestureReleaseFramesRef.current = 0;
      }
    } else if (gesture === lastGesture) {
      gestureReleaseFramesRef.current = 0;
    }
    const action = message.runtime_action;
    if (!action || action === 'Wait / No Action' || action === 'No Gesture') return;
    const command = DEMO_COMMANDS[gesture];
    if (!command || command === 'NONE' || lastExecutedGestureRef.current === gesture) return;
    lastExecutedGestureRef.current = gesture;
    gestureReleaseFramesRef.current = 0;
    void performDemoCommand(command, gesture);
  };

  useEffect(() => { executeDemoPredictionRef.current = executeDemoPrediction; });

  const uploadDemoMedia = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    const kind = file.type.startsWith('video/') ? 'video' : file.type.startsWith('image/') ? 'image' : null;
    if (!kind) {
      setDemoVideoMessage('Choose a valid video or image file.');
      return;
    }
    const url = URL.createObjectURL(file);
    try {
      let asset: DemoMediaAsset;
      if (kind === 'video') {
        const duration = await new Promise<number>((resolve, reject) => {
          const probe = document.createElement('video');
          probe.preload = 'metadata';
          probe.onloadedmetadata = () => resolve(probe.duration);
          probe.onerror = () => reject(new Error('The browser could not read this video.'));
          probe.src = url;
        });
        if (!Number.isFinite(duration) || duration <= 0 || duration > 60) {
          throw new Error('The demonstration video must be 60 seconds or shorter.');
        }
        asset = { url, name: file.name, kind, duration };
      } else {
        const dimensions = await new Promise<{ width: number; height: number }>((resolve, reject) => {
          const probe = new Image();
          probe.onload = () => resolve({ width: probe.naturalWidth, height: probe.naturalHeight });
          probe.onerror = () => reject(new Error('The browser could not read this image.'));
          probe.src = url;
        });
        asset = { url, name: file.name, kind, ...dimensions };
      }
      if (demoVideoAssetRef.current) URL.revokeObjectURL(demoVideoAssetRef.current.url);
      demoVideoAssetRef.current = asset;
      setDemoVideo(asset);
      updateDemoTransform(() => ({ x: 0, y: 0, scale: 1 }));
      setDemoEvents([]);
      setActionToast(null);
      setObjectTrackingEnabled(false);
      setImagePreviewActive(true);
      setDemoMuted(false);
      setDemoVolume(1);
      setRecordingDownloadUrl((current) => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      lastExecutedGestureRef.current = null;
      setDemoVideoMessage(`${kind === 'video' ? 'Video' : 'Image'} ready. Connect the selected camera and hold a gesture until the stable action fires.`);
      if (kind === 'video') setTimeout(() => { void demoVideoRef.current?.play().catch(() => undefined); }, 0);
    } catch (error) {
      URL.revokeObjectURL(url);
      setDemoVideoMessage(error instanceof Error ? error.message : 'The media file could not be loaded.');
    }
  };

  const removeDemoVideo = () => {
    if (demoRecording) return;
    if (demoVideoAssetRef.current) URL.revokeObjectURL(demoVideoAssetRef.current.url);
    demoVideoAssetRef.current = null;
    setDemoVideo(null);
    setDemoEvents([]);
    setActionToast(null);
    setObjectTrackingEnabled(false);
    setImagePreviewActive(true);
    setDemoMuted(false);
    setDemoVolume(1);
    setRecordingDownloadUrl((current) => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    updateDemoTransform(() => ({ x: 0, y: 0, scale: 1 }));
    setDemoVideoMessage('Upload a video (up to 60 seconds) or an image to enable the action demo.');
    lastExecutedGestureRef.current = null;
  };

  const predictUploadedImage = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('image', file);
      const query = selectedModel ? `?model_name=${encodeURIComponent(selectedModel)}` : '';
      const response = await fetch(`${API_URL}/api/predict-image${query}`, { method: 'POST', body: form });
      const result = await response.json() as Prediction;
      setPrediction(result);
      setActiveTab('live');
      lastSnapshotRef.current = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.readAsDataURL(file);
      });
      setSnapshotReady(true);
    } catch {
      setPrediction({ status: 'error', message: 'Uploaded-image prediction failed.' });
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const submitFeedback = async () => {
    setFeedbackMessage(learningMode === 'audit' ? 'Saving reviewed sample…' : 'Validating a guarded online update…');
    try {
      if (prediction.ncm_frame_id != null) {
        const snapshotResponse = await fetch(`${API_URL}/api/ncm/frame.jpg`, { cache: 'no-store' });
        if (snapshotResponse.ok) {
          const snapshot = await snapshotResponse.blob();
          lastSnapshotRef.current = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(String(reader.result));
            reader.readAsDataURL(snapshot);
          });
          setSnapshotReady(true);
        }
      } else if (activeCameraSource === 'webcam' && webcamStream) {
        const snapshot = await captureWebcamFrame();
        lastSnapshotRef.current = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.readAsDataURL(snapshot);
        });
        setSnapshotReady(true);
      }
      const response = await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          actual_label: selectedFeedbackLabel,
          predicted_label: feedbackPrediction ?? 'no_gesture',
          runtime_action: prediction.runtime_action ?? 'Wait / No Action',
          confidence: prediction.confidence ?? 0,
          model_name: prediction.model ?? selectedModel ?? 'unknown',
          probabilities: prediction.probabilities ?? {},
          base_probabilities: prediction.base_probabilities ?? prediction.raw_probabilities ?? null,
          feature_vector: prediction.feature_vector ?? null,
          landmarks: prediction.landmarks ?? null,
          note: feedbackNote,
          snapshot_data_url: lastSnapshotRef.current,
          learning_mode: learningMode,
        }),
      });
      const payload = await response.json() as { detail?: string; online_update?: { message?: string } };
      if (!response.ok) throw new Error(payload.detail ?? 'Feedback failed.');
      setFeedbackMessage(payload.online_update?.message ?? 'Feedback saved.');
      setFeedbackNote('');
      setFeedbackCount((count) => count + 1);
      if (learningMode !== 'audit') await refreshServerData();
    } catch (error) {
      setFeedbackMessage(error instanceof Error ? error.message : 'Feedback could not be saved.');
    }
  };

  const reloadArtifacts = async () => {
    try {
      await fetch(`${API_URL}/api/reload`, { method: 'POST' });
      await refreshServerData();
    } catch {
      setFeedbackMessage('The local server is not available.');
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">G</span>
          <div><p className="eyebrow">WINDOWS · ONNX · DUAL CAMERA</p><h1>Gesture Control Lab</h1></div>
        </div>
        <div className="topbar-status">
          <span className={`status-dot ${connection === 'online' ? 'is-online' : ''}`} />
          <span>{connection === 'online' ? 'Live session connected' : health ? 'Local server ready' : 'Server offline'}</span>
          <span className="model-pill">Strict {targetFps} FPS</span>
        </div>
      </header>

      <nav className="nav-rail" aria-label="Dashboard sections">
        {(['live', 'analytics', 'feedback', 'setup', 'data_collection'] as const).map((tab, index) => (
          <button className={`nav-item ${activeTab === tab ? 'active' : ''}`} type="button" key={tab} onClick={() => setActiveTab(tab)}>
            <span>0{index + 1}</span>{humanize(tab)}
          </button>
        ))}
        <div className="rail-footer">
          <p>Runtime contract</p>
          <strong>EMA α {health?.config.ema_alpha ?? 0.65}</strong>
          <span>{stableRequired} stable frames · 76-D</span>
        </div>
      </nav>

      <section className="workspace">
        <video ref={webcamVideoRef} className="webcam-capture-source" autoPlay muted playsInline aria-hidden="true" />
        <canvas ref={webcamCanvasRef} className="webcam-capture-source" aria-hidden="true" />
        <DataCollection
          api={API_URL}
          active={activeTab === 'data_collection'}
          connected={cameraActive && activeCameraSource === cameraSource}
          ready={serverReady}
          streamRevision={streamRevision}
          liveLandmarks={prediction.landmarks}
          livePlaneAngles={livePlaneAngles}
          cameraSource={cameraSource}
          webcamStream={webcamStream}
          captureWebcamFrame={captureWebcamFrame}
          setCameraSource={changeCameraSource}
          toggleCamera={toggleCamera}
        />
        <div hidden={activeTab !== 'live' && activeTab !== 'feedback'}><RangeDiagnostics prediction={prediction} connected={cameraActive} /></div>
        {activeTab === 'live' && <>
          <div className="section-heading">
            <div><p className="eyebrow">LIVE GESTURE RECOGNITION</p><h2>{targetFps.toFixed(0)} FPS {cameraSource === 'webcam' ? 'webcam' : 'JLIP camera'} qualification</h2></div>
            <div className="heading-actions">
              <label className="primary-button file-button demo-upload-button">
                {demoVideo ? 'Replace demo media' : 'Upload video / image'}
                <input type="file" accept="video/*,image/*" onChange={uploadDemoMedia} disabled={demoRecording} />
              </label>
              {demoVideo && <button className="secondary-button" type="button" onClick={removeDemoVideo} disabled={demoRecording}>Remove media</button>}
              <label className="secondary-button file-button">
                {uploading ? 'Processing…' : 'Classify image'}
                <input type="file" accept="image/*" onChange={predictUploadedImage} disabled={uploading || !serverReady} />
              </label>
              <button className="secondary-button" type="button" onClick={resetSession}>Reset</button>
              <label className="camera-source-control">Camera source<select aria-label="Live testing camera source" value={cameraSource} disabled={cameraActive || connection === 'connecting'} onChange={event => changeCameraSource(event.target.value as CameraSource)}><option value="ncm">NCM board camera</option><option value="webcam">PC webcam</option></select></label>
              <button className="primary-button" type="button" onClick={toggleCamera} disabled={!serverReady || connection === 'connecting'}>
                {connection === 'connecting' ? 'Connecting…' : cameraActive ? `Disconnect ${cameraSource === 'webcam' ? 'webcam' : 'board camera'}` : `Connect ${cameraSource === 'webcam' ? 'webcam' : 'board camera'}`}
              </button>
            </div>
          </div>

          <div className={`demo-status-banner ${demoVideo ? 'ready' : ''}`}>
            <div><strong>{demoVideo ? 'ACTION DEMO READY' : 'UPLOAD MEDIA DEMO'}</strong>
              <span>{demoVideo
                ? `${demoVideo.name} · ${demoVideo.kind === 'video' ? `${demoVideo.duration?.toFixed(1)} seconds` : `${demoVideo.width} × ${demoVideo.height} image`} · processed locally`
                : 'Use a local video of 60 seconds or less, or an image.'}</span>
            </div>
            <p>{demoVideoMessage}</p>
          </div>

          {cameraSource === 'ncm' ? <div className={`ncm-link-panel ${ncmStatus?.connected ? 'connected' : ''}`}>
              <div className="ncm-link-title">
                <span className={`status-dot ${ncmStatus?.connected ? 'is-online' : ''}`} />
                <div><strong>USB-NCM / JLIP CAMERA LINK</strong><p>{ncmMessage}</p></div>
              </div>
              <dl>
                <div><dt>Windows host</dt><dd>{ncmStatus?.config.host_ip ?? '192.168.50.1'}/30</dd></div>
                <div><dt>Board</dt><dd>{ncmStatus?.config.device_ip ?? '192.168.50.2'}:{ncmStatus?.config.tcp_port ?? 5000}</dd></div>
                <div><dt>State</dt><dd>{humanize(ncmStatus?.state)}</dd></div>
                <div><dt>Frames / FPS</dt><dd>{ncmStatus?.frames_received ?? 0} / {fps(ncmStatus?.camera_fps)}</dd></div>
                <div><dt>CRC / sequence gaps</dt><dd>{ncmStatus?.crc_errors ?? 0} / {ncmStatus?.sequence_gaps ?? 0}</dd></div>
              </dl>
              <button className="secondary-button" type="button" onClick={discoverNcmCamera}>Run UDP discovery</button>
            </div> : <div className={`ncm-link-panel webcam-link-panel ${cameraActive ? 'connected' : ''}`}>
              <div className="ncm-link-title">
                <span className={`status-dot ${cameraActive ? 'is-online' : ''}`} />
                <div><strong>PC WEBCAM / BROWSER LINK</strong><p>{webcamDetails}</p></div>
              </div>
              <dl>
                <div><dt>Source</dt><dd>Browser MediaStream</dd></div>
                <div><dt>Transport</dt><dd>JPEG / WebSocket</dd></div>
                <div><dt>State</dt><dd>{cameraActive ? 'Connected' : 'Ready'}</dd></div>
                <div><dt>Inference FPS</dt><dd>{cameraActive ? fps(cameraFps) : '—'}</dd></div>
                <div><dt>Preview</dt><dd>Unmirrored</dd></div>
              </dl>
              <span className="camera-source-note">Permission stays local to this browser session.</span>
            </div>}

          {!serverReady && <div className="setup-banner">
            <strong>Model setup required.</strong>
            <span>The UI is ready, but the signed ONNX model, metadata, runtime configuration, or MediaPipe detector is missing or invalid.</span>
            <button type="button" onClick={() => setActiveTab('setup')}>Open setup</button>
          </div>}

          <article className={`qualification-strip ${pipelineVerdict.toLowerCase()}`}>
            <div><span>{targetFps.toFixed(0)} FPS CAPACITY TEST</span><strong>{pipelineVerdict}</strong></div>
            <p>{timing
              ? targetCapacityPass
                ? `Pipeline uses ${timing.budget_used_percent?.toFixed(1)}% of the ${frameBudgetMs.toFixed(1)} ms frame budget. This PC can sustain the configured ${targetFps.toFixed(0)} FPS workload.`
                : `Pipeline needs ${timing.total_ms?.toFixed(1)} ms per frame, above the ${frameBudgetMs.toFixed(1)} ms budget. Reduce workload or use faster hardware.`
              : 'Start the camera and hold a gesture to measure end-to-end processing capacity.'}</p>
          </article>

          <div className="dashboard-grid">
            <article className="camera-card panel">
              <div className="panel-header">
                <div><span className={`live-dot ${cameraActive ? 'is-online' : ''}`} /> {demoVideo ? 'UPLOADED MEDIA ACTION STAGE' : 'LIVE CAMERA PREVIEW'}</div>
                <span>{demoVideo ? `${cameraSource === 'webcam' ? 'Webcam' : 'NCM camera'} + landmarks in upper-right` : cameraSource === 'webcam' ? 'Browser webcam · unmirrored JPEG frames' : 'JLIP JPEG stream · central 92% ROI'}</span>
              </div>
              <div className={`camera-stage live-stage ${demoVideo ? 'has-demo-video' : ''}`}>
                {demoVideo && <div className="demo-video-viewport">
                  {demoVideo.kind === 'video' ? <video
                      ref={demoVideoRef}
                      src={demoVideo.url}
                      className="demo-target-media demo-target-video"
                      controls
                      muted={demoMuted}
                      playsInline
                      onVolumeChange={event => { setDemoMuted(event.currentTarget.muted); setDemoVolume(event.currentTarget.volume); }}
                      onEnded={() => {
                        if (mediaRecorderRef.current?.state === 'recording') stopDemoRecording();
                        setDemoVideoMessage('The uploaded video reached the end. Replay it or upload another clip.');
                      }}
                      style={{ transform: `translate(${demoTransform.x}%, ${demoTransform.y}%) scale(${demoTransform.scale})` }}
                    /> : <img
                      ref={demoImageRef}
                      src={demoVideo.url}
                      alt="Uploaded gesture action target"
                      className={`demo-target-media demo-target-image ${imagePreviewActive ? '' : 'preview-paused'}`}
                      style={{ transform: `translate(${demoTransform.x}%, ${demoTransform.y}%) scale(${demoTransform.scale})` }}
                    />}
                </div>}
                {cameraActive && cameraSource === 'ncm' && <img
                  key={streamRevision}
                  src={`${API_URL}/api/ncm/stream.mjpg?revision=${streamRevision}`}
                  alt="Live JPEG stream from the USB-NCM development-board camera"
                  className="camera-video active ncm-camera-video"
                />}
                {cameraActive && cameraSource === 'webcam' && <WebcamVideo stream={webcamStream} className="camera-video active webcam-camera-video" />}
                {demoVideo && <div className="demo-live-action" aria-live="polite">
                  <span>CONFIRMED LIVE ACTION</span>
                  <strong>{prediction.runtime_action && prediction.runtime_action !== 'Wait / No Action' ? mappedLabel : 'Wait / No Action'}</strong>
                  <small>{humanize(runtimeGesture)} · {percent(prediction.confidence)}</small>
                </div>}
                {demoVideo && <div className="demo-state-badges">
                  {demoRecording && <span className="recording-badge"><i /> REC</span>}
                  {objectTrackingEnabled && <span className="tracking-badge">OBJECT TRACKING ON</span>}
                  {demoVideo.kind === 'video' && <span className="audio-badge">{demoMuted ? 'AUDIO MUTED' : `VOLUME ${Math.round(demoVolume * 100)}%`}</span>}
                </div>}
                {demoVideo && actionToast && <div className={`action-toast ${actionToast.status}`} role="status" aria-live="assertive">
                  <span>{humanize(actionToast.gesture)} gesture</span>
                  <strong>{actionToast.label}</strong>
                </div>}
                <div className="guide-box live-guide">
                  <span className="corner top-left" /><span className="corner top-right" />
                  <span className="corner bottom-left" /><span className="corner bottom-right" />
                  <canvas ref={landmarkCanvasRef} className="landmark-canvas" />
                  {demoVideo && <div className="webcam-pip-label"><span className={`live-dot ${cameraActive ? 'is-online' : ''}`} /> {cameraSource === 'webcam' ? 'PC WEBCAM' : 'NCM BOARD CAMERA'} + LANDMARKS</div>}
                  {!cameraActive && <div className="camera-empty">
                    <span className="hand-orbit" /><strong>Camera is ready</strong>
                    <p>{demoVideo ? `Connect the ${cameraSource === 'webcam' ? 'webcam' : 'board camera'} to control the uploaded media.` : `Connect the ${cameraSource === 'webcam' ? 'PC webcam' : 'USB-NCM camera'} and keep one complete hand and wrist inside the large guide.`}</p>
                  </div>}
                </div>
                <canvas ref={demoRecordingCanvasRef} className="capture-canvas" />
              </div>
              <div className="camera-footer">
                <span>{demoVideo ? `Target ${demoTransform.scale.toFixed(1)}x · X ${demoTransform.x} · Y ${demoTransform.y}` : prediction.status === 'predicted' ? 'ONNX Runtime · CPU' : humanize(prediction.status)}</span>
                <span>Unmirrored {cameraSource === 'webcam' ? 'PC webcam' : 'NCM camera'} · calibrated Left/Right · {fps(cameraFps)} FPS · inference capped at {targetFps.toFixed(0)} FPS</span>
              </div>
              <div className="live-angle-panel" aria-label="Live three-dimensional palm-axis angles">
                <div><span>LIVE 3-D PALM AXIS</span><small>Signed wrist → palm angle in each coordinate plane</small></div>
                {(['xy', 'yz', 'xz'] as const).map(plane => <div key={plane}><span>{plane.toUpperCase()}</span><strong>{livePlaneAngles ? `${livePlaneAngles[plane].toFixed(1)}°` : '—'}</strong></div>)}
                {!livePlaneAngles && <p>Waiting for a complete hand and depth landmarks.</p>}
              </div>
              {demoVideo && <div className="demo-action-console">
                <div className="demo-console-summary">
                  <div><span>ACTION DISPATCH</span><strong>{demoEvents[0]?.command.replaceAll('_', ' ') ?? 'WAITING'}</strong></div>
                  <div className="demo-console-buttons">
                    {recordingDownloadUrl && <a href={recordingDownloadUrl} download="gesture-action-demo.webm">Download recording</a>}
                    <button type="button" onClick={() => { updateDemoTransform(() => ({ x: 0, y: 0, scale: 1 })); setObjectTrackingEnabled(false); setImagePreviewActive(true); if (demoVideoRef.current) { demoVideoRef.current.muted = false; demoVideoRef.current.volume = 1; } setDemoMuted(false); setDemoVolume(1); }}>Reset media</button>
                  </div>
                </div>
                <div className="demo-event-list">
                  {demoEvents.length ? demoEvents.slice(0, 4).map((event, index) => <div key={`${event.at}-${event.command}-${index}`} className={event.status}>
                    <span>{event.at}</span><strong>{humanize(event.gesture)} → {event.command.replaceAll('_', ' ')}</strong><p>{event.detail}</p>
                  </div>) : <p className="empty-note">Stable gesture actions will be proven here as they control the uploaded media.</p>}
                </div>
              </div>}
            </article>

            <aside className="result-stack">
              <article className="result-card accent-mint">
                <p>RUNTIME PREDICTION</p><strong>{humanize(runtimeGesture)}</strong>
                <span>{prediction.message ?? `${percent(prediction.confidence)} confidence · ${prediction.stable_frames ?? 0}/${stableRequired} stable`}</span>
              </article>
              <article className="result-card label-card" aria-live="polite">
                <p>LABEL</p><strong>{mappedLabel}</strong>
                <span>{prediction.runtime_prediction
                  ? `Mapped from ${humanize(runtimeGesture)}`
                  : 'The mapped action will appear after prediction.'}</span>
              </article>
              <article className="result-card accent-orange">
                <p>RUNTIME ACTION</p><strong>{prediction.runtime_action && prediction.runtime_action !== 'Wait / No Action' ? mappedLabel : 'Wait / No Action'}</strong>
                <span>{prediction.action_reason ?? 'Confidence and stability gates are active.'}</span>
              </article>
              <div className="model-control">
                <label htmlFor="model-select">Runtime model</label>
                <select id="model-select" value={selectedModel} onChange={changeModel} disabled={!selectableModels.length}>
                  {selectableModels.map((model) => <option key={model}>{model}</option>)}
                  {!selectableModels.length && <option>No model loaded</option>}
                </select>
              </div>
            </aside>

            <article className="probability-card panel">
              <div className="panel-header"><div>CLASS SCORES</div><span>{prediction.runtime_prediction === 'no_gesture' ? 'CANDIDATE ONLY · NO COMMAND ACCEPTED' : `EMA · ALL ${classNames.length} CLASSES`}</span></div>
              <div className="probability-list full-list">
                {probabilityRows.map(({ name, value }) => <div className="probability-row" key={name}>
                  <span>{humanize(name)}</span><div><i style={{ width: `${Math.min(100, value * 100)}%` }} /></div><strong>{percent(value)}</strong>
                </div>)}
              </div>
            </article>

            <article className="performance-card panel">
              <div className="panel-header"><div>PIPELINE HEALTH</div><span>{pipelineVerdict}</span></div>
              <div className="metric-grid metric-grid-wide">
                <div><span>Camera FPS</span><strong>{fps(cameraFps)}</strong></div>
                <div><span>Configured limit</span><strong>{targetFps.toFixed(2)}</strong></div>
                <div><span>Effective app FPS</span><strong>{fps(timing?.effective_application_fps ?? prediction.actual_fps)}</strong></div>
                <div><span>Uncapped pipeline</span><strong>{fps(timing?.uncapped_fps)}</strong></div>
                <div><span>Total pipeline</span><strong>{milliseconds(timing?.total_ms)}</strong></div>
                <div><span>Frame budget</span><strong>{milliseconds(frameBudgetMs)}</strong></div>
                <div><span>Budget used</span><strong>{timing?.budget_used_percent == null ? '—' : `${timing.budget_used_percent.toFixed(1)}%`}</strong></div>
                <div><span>Headroom</span><strong>{milliseconds(timing?.headroom_ms)}</strong></div>
              </div>
            </article>
          </div>
        </>}

        {activeTab === 'analytics' && <>
          <div className="section-heading">
            <div><p className="eyebrow">MODEL OBSERVABILITY</p><h2>Diagnostics & evidence</h2></div>
            <button className="secondary-button" type="button" onClick={refreshServerData}>Refresh data</button>
          </div>
          <div className="analytics-grid">
            <article className="panel summary-panel">
              <div className="panel-header"><div>{targetFps.toFixed(0)} FPS SUMMARY</div><span>END-TO-END</span></div>
              <div className="summary-metrics">
                <div><span>Verdict</span><strong className={pipelineVerdict === 'PASS' ? 'text-pass' : 'text-warn'}>{pipelineVerdict}</strong></div>
                <div><span>Runtime model</span><strong>{(prediction.model ?? selectedModel) || '—'}</strong></div>
                <div><span>MediaPipe</span><strong>{milliseconds(timing?.mediapipe_ms)}</strong></div>
                <div><span>Classifier</span><strong>{milliseconds(timing?.classifier_ms)}</strong></div>
              </div>
            </article>

            <article className="panel diagnostics-table-panel">
              <div className="panel-header"><div>MODEL DIAGNOSTICS</div><span>ONNX RUNTIME</span></div>
              {diagnostics.length ? <div className="table-wrap"><table className="diagnostics-table">
                <thead><tr><th>Model</th><th>Runtime</th><th>Raw</th><th>EMA result</th><th>Confidence</th><th>Stable</th><th>Decision</th><th>Time</th></tr></thead>
                <tbody>{diagnostics.map(([name, row]) => <tr key={name}>
                  <td>{name}</td><td>{row.used_for_runtime ? 'YES' : '—'}</td><td>{humanize(row.raw_prediction)}</td><td>{humanize(row.prediction)}</td>
                  <td>{percent(row.confidence)}</td><td>{row.stable_frames}</td><td>{row.execute ? 'EXECUTE' : row.reason}</td><td>{milliseconds(row.classifier_ms)}</td>
                </tr>)}</tbody>
              </table></div> : <p className="empty-note">Start a live session to populate per-model diagnostics.</p>}
            </article>

            <article className="panel action-log">
              <div className="panel-header"><div>ACTION HISTORY</div><span>{actions.length} RECENT</span></div>
              <div className="log-list">{actions.length ? actions.slice(0, 12).map((item, index) => <div key={`${item.action}-${index}`}>
                <span>{String(index + 1).padStart(2, '0')}</span><strong>{item.action}</strong><em>{humanize(item.prediction)} · {percent(item.confidence)}</em>
              </div>) : <p className="empty-note">Stable confirmed actions will appear here.</p>}</div>
            </article>

            <article className="panel metric-files">
              <div className="panel-header"><div>NOTEBOOK EVALUATION EXPORTS</div><span>CSV</span></div>
              {metrics.files.length ? <ul>{metrics.files.map((file) => <li key={file}><span>{file}</span><strong>{metrics.rows[file]?.length ?? 0} rows</strong></li>)}</ul>
                : <p className="empty-note">Copy the notebook evaluation CSV files into the models folder to inspect them here.</p>}
            </article>

            <article className="panel class-map">
              <div className="panel-header"><div>COMMAND TAXONOMY</div><span>{classNames.length} CLASSES</span></div>
              <div className="class-map-grid">{classNames.map((name) => <div key={name}><span>{humanize(name)}</span><strong>{actionMap[name]}</strong></div>)}</div>
            </article>
          </div>
        </>}

        {activeTab === 'feedback' && <>
          <div className="section-heading">
            <div><p className="eyebrow">GUARDED ONLINE LEARNING</p><h2>Correct, capture, and safely learn</h2></div>
            <div className="feedback-heading-actions">
              <span className="count-pill">{feedbackCount} reviewed samples</span>
              <label className="camera-source-control">Camera source<select aria-label="Feedback camera source" value={cameraSource} disabled={cameraActive || connection === 'connecting'} onChange={event => changeCameraSource(event.target.value as CameraSource)}><option value="ncm">NCM board camera</option><option value="webcam">PC webcam</option></select></label>
              <button className="primary-button" type="button" onClick={toggleCamera} disabled={!serverReady || connection === 'connecting'}>
                {connection === 'connecting' ? 'Connecting…' : cameraActive ? `Disconnect ${cameraSource === 'webcam' ? 'webcam' : 'board camera'}` : `Connect ${cameraSource === 'webcam' ? 'webcam' : 'board camera'}`}
              </button>
            </div>
          </div>
          <div className="feedback-layout">
            <article className="panel feedback-preview">
              <div className="panel-header"><div><span className={`live-dot ${cameraActive ? 'is-online' : ''}`} /> LIVE FEEDBACK CAMERA</div><span>{cameraActive ? `${fps(cameraFps)} FPS` : 'OFFLINE'}</span></div>
              <div className="camera-stage live-stage feedback-camera-stage">
                {cameraActive && cameraSource === 'ncm' && <img
                  key={`feedback-${streamRevision}`}
                  src={`${API_URL}/api/ncm/stream.mjpg?revision=${streamRevision}`}
                  alt="Unmirrored live JPEG stream from the USB-NCM development-board camera"
                  className="camera-video active ncm-camera-video feedback-camera-video"
                />}
                {cameraActive && cameraSource === 'webcam' && <WebcamVideo stream={webcamStream} className="camera-video active webcam-camera-video feedback-camera-video" />}
                <div className="guide-box live-guide feedback-live-guide">
                  <span className="corner top-left" /><span className="corner top-right" />
                  <span className="corner bottom-left" /><span className="corner bottom-right" />
                  <canvas ref={feedbackLandmarkCanvasRef} className="landmark-canvas" />
                  {!cameraActive && <div className="camera-empty">
                    <span className="hand-orbit" /><strong>Camera stays available here</strong>
                    <p>Connect the selected camera, show a gesture, then confirm or correct the result beside the preview.</p>
                  </div>}
                </div>
              </div>
              <div className="camera-footer"><span>Unmirrored {cameraSource === 'webcam' ? 'webcam' : 'NCM'} preview + calibrated Left/Right + live landmarks</span><span>{connection === 'online' ? 'Feedback capture ready' : 'Connect to capture'}</span></div>
              <div className="feedback-prediction"><p>The system predicted</p><strong>{humanize(feedbackPrediction)}</strong><span>{percent(prediction.confidence)} confidence · {prediction.stable_frames ?? 0}/{stableRequired} stable</span></div>
              <div className="feedback-note"><strong>Safe Learn is guarded</strong><p>Each confirmed sample includes the current snapshot, landmarks, and feature vector. The server validates an online candidate before accepting it and keeps the deployed model when validation fails.</p></div>
              <div className="online-status-grid">
                <div><span>Safe learner</span><strong>{onlineLearning?.ready ? 'Ready' : 'Needs setup'}</strong></div>
                <div><span>Accepted updates</span><strong>{onlineLearning?.accepted_updates ?? 0}</strong></div>
                <div><span>Rejected updates</span><strong>{onlineLearning?.rejected_updates ?? 0}</strong></div>
                <div><span>Validation macro F1</span><strong>{onlineLearning?.validation_macro_f1 == null ? '—' : percent(onlineLearning.validation_macro_f1)}</strong></div>
              </div>
            </article>

            <article className="panel feedback-form">
              <div className="panel-header"><div>CORRECT THE LABEL</div><span>USER CONFIRMATION REQUIRED</span></div>
              <div className="form-body">
                <label>Actual gesture<select value={selectedFeedbackLabel} onChange={(event) => setActualLabel(event.target.value)}>
                  {feedbackLabels.map((name) => <option key={name} value={name}>{name === 'no_gesture' ? 'No Gesture — reject false detection' : `${humanize(name)} — ${actionMap[name]}`}</option>)}
                </select></label>
                <label>Feedback action<select value={learningMode} onChange={(event) => setLearningMode(event.target.value as LearningMode)}>
                  <option value="safe">Safe Learn — validate and update</option>
                  <option value="audit">Save only — no model update</option>
                </select></label>
                <label>Reviewer note<textarea value={feedbackNote} onChange={(event) => setFeedbackNote(event.target.value)} placeholder="Describe lighting, angle, occlusion, or the confusion you observed." rows={4} /></label>
                <div className="form-details"><span>Prediction: {humanize(feedbackPrediction)}</span><span>Snapshot: {snapshotReady ? 'ready' : 'not captured'}</span><span>76-D features: {prediction.feature_vector?.length === 76 ? 'ready' : 'not available'}</span><span>Camera: {cameraActive ? 'running' : 'offline'}</span></div>
                <button className="primary-button wide" type="button" onClick={submitFeedback} disabled={!prediction.runtime_prediction}>{learningMode === 'safe' ? 'Confirm feedback & safely learn' : 'Save reviewed feedback only'}</button>
                {feedbackMessage && <p className="form-message">{feedbackMessage}</p>}
              </div>
            </article>
          </div>
        </>}

        {activeTab === 'setup' && <>
          <div className="section-heading">
            <div><p className="eyebrow">WINDOWS LOCAL DEPLOYMENT</p><h2>Setup & model artifacts</h2></div>
            <button className="primary-button" type="button" onClick={reloadArtifacts} disabled={!health?.production?.allow_artifact_reload}>Reload artifacts</button>
          </div>
          {!!health?.engine.model.errors.length && <div className="setup-banner error-stack">
            <strong>Artifact validation:</strong>
            <span>{health.engine.model.errors.join(' · ')}</span>
          </div>}
          <div className="setup-layout">
            <article className="panel runtime-contract">
              <div className="panel-header"><div>PRODUCTION RELEASE GATE</div><span>{qualification?.pc_release_ready ? 'PASS' : 'ACTION NEEDED'}</span></div>
              <dl>
                <div><dt>Artifact integrity</dt><dd>{health?.artifact_integrity?.verified ? `Verified (${health.artifact_integrity.checked_files})` : 'Failed / unsigned'}</dd></div>
                <div><dt>Qualification report</dt><dd>{qualification?.current ? qualification.status.toUpperCase() : qualification?.status === 'not_run' ? 'NOT RUN' : 'STALE'}</dd></div>
                <div><dt>ONNX prediction agreement</dt><dd>{qualification?.onnx_parity?.prediction_agreement == null ? '—' : percent(qualification.onnx_parity.prediction_agreement)}</dd></div>
                <div><dt>Maximum probability error</dt><dd>{qualification?.onnx_parity?.maximum_absolute_probability_error?.toExponential(3) ?? '—'}</dd></div>
                <div><dt>Deferred quality work</dt><dd>{qualification?.deferred_classes?.map(humanize).join(', ') || 'None'}</dd></div>
                <div><dt>Other failing classes</dt><dd>{qualification?.failing_gated_classes?.length ? qualification.failing_gated_classes.map(humanize).join(', ') : 'None'}</dd></div>
              </dl>
            </article>

            <article className="panel runtime-contract">
              <div className="panel-header"><div>OPERATIONAL SLO WINDOW</div><span>{runtimeSummary?.frames_total ?? 0} FRAMES</span></div>
              <dl>
                <div><dt>Pipeline p50</dt><dd>{milliseconds(runtimeSummary?.timing?.total_ms?.p50)}</dd></div>
                <div><dt>Pipeline p95</dt><dd>{milliseconds(runtimeSummary?.timing?.total_ms?.p95)}</dd></div>
                <div><dt>Pipeline p99</dt><dd>{milliseconds(runtimeSummary?.timing?.total_ms?.p99)}</dd></div>
                <div><dt>{targetFps.toFixed(0)} FPS budget pass</dt><dd>{percent(targetBudgetPassRate)}</dd></div>
                <div><dt>Error rate</dt><dd>{percent(runtimeSummary?.error_rate)}</dd></div>
                <div><dt>Sessions active / peak</dt><dd>{runtimeSummary?.active_sessions ?? 0} / {runtimeSummary?.peak_sessions ?? 0}</dd></div>
              </dl>
            </article>

            <article className="panel artifact-panel">
              <div className="panel-header"><div>ARTIFACT CHECKLIST</div><span>{health?.status === 'ready' ? 'READY' : 'ACTION NEEDED'}</span></div>
              <div className="artifact-list">{(health?.artifacts ?? []).map((artifact) => <div key={`${artifact.label}-${artifact.pattern}`}>
                <span className={`artifact-state ${artifact.present ? 'present' : ''}`}>{artifact.present ? '✓' : '!'}</span>
                <div><strong>{artifact.label}</strong><p>{artifact.present ? artifact.files.join(', ') : `Expected ${artifact.pattern}`}</p></div>
                <em>{artifact.present ? `${(artifact.bytes / 1024).toFixed(1)} KB` : artifact.required ? 'Required' : 'Optional path'}</em>
              </div>)}</div>
            </article>

            <article className="panel setup-steps">
              <div className="panel-header"><div>FIRST-TIME ONNX + CAMERA SETUP</div><span>4 STEPS</span></div>
              <ol>
                <li><span>01</span><div><strong>Install the Microsoft runtime</strong><p>Install the Microsoft Visual C++ 2015–2022 x64 Redistributable required by ONNX Runtime on Windows.</p></div></li>
                <li><span>02</span><div><strong>Choose a camera source</strong><p>Allow browser camera permission for the PC webcam, or configure the NCM adapter at 192.168.50.1/30 for the board camera.</p></div></li>
                <li><span>03</span><div><strong>Run the separate application</strong><p>Double-click start_ncm_onnx_dashboard.bat. Services use ports 3200 and 8200.</p></div></li>
                <li><span>04</span><div><strong>Connect and test</strong><p>Select PC Webcam or NCM Board Camera in Live Testing, connect it, then verify the {targetFps.toFixed(0)} FPS frame budget.</p></div></li>
              </ol>
            </article>

            <article className="panel runtime-contract">
              <div className="panel-header"><div>NCM TRANSPORT DIAGNOSTICS</div><span>{ncmStatus?.connected ? 'CONNECTED' : 'CHECK LINK'}</span></div>
              <dl>
                <div><dt>Protocol</dt><dd>JLIP v1 over TCP</dd></div>
                <div><dt>Discovery</dt><dd>UDP {ncmStatus?.config.discovery_port ?? 5001}</dd></div>
                <div><dt>Packets / bytes</dt><dd>{ncmStatus?.packets_received ?? 0} / {ncmStatus?.bytes_received ?? 0}</dd></div>
                <div><dt>Header / JPEG errors</dt><dd>{ncmStatus?.header_errors ?? 0} / {ncmStatus?.invalid_jpeg_frames ?? 0}</dd></div>
                <div><dt>Reconnects</dt><dd>{ncmStatus?.reconnects ?? 0}</dd></div>
                <div><dt>Last transport error</dt><dd>{ncmStatus?.last_error ?? 'None reported'}</dd></div>
              </dl>
            </article>

            <article className="panel runtime-contract">
              <div className="panel-header"><div>ONNX RUNTIME CONTRACT</div><span>EXACT</span></div>
              <dl>
                <div><dt>Inference limit</dt><dd>{targetFps} FPS / {frameBudgetMs.toFixed(1)} ms</dd></div>
                <div><dt>Taxonomy</dt><dd>{classNames.length} classes</dd></div>
                <div><dt>Feature vector</dt><dd>76 dimensions</dd></div>
                <div><dt>Execution provider</dt><dd>CPUExecutionProvider</dd></div>
                <div><dt>Temporal filter</dt><dd>EMA α {health?.config.ema_alpha ?? 0.65}</dd></div>
                <div><dt>Confidence floor</dt><dd>{percent(health?.config.confidence_floor ?? 0.70)}</dd></div>
                <div><dt>Object tracking</dt><dd>Toggled only by an upward Open Palm</dd></div>
              </dl>
            </article>

            <article className="panel server-card">
              <div className="panel-header"><div>LOCAL ENDPOINTS</div><span>PRIVATE TO THIS PC</span></div>
              <div className="endpoint-list"><code>Dashboard   http://127.0.0.1:3200</code><code>API         http://127.0.0.1:8200</code><code>API docs    http://127.0.0.1:8200/docs</code><code>NCM socket  ws://127.0.0.1:8200/ws/ncm-live</code><code>Webcam      ws://127.0.0.1:8200/ws/live</code></div>
            </article>
          </div>
        </>}
      </section>
    </main>
  );
}
