/** Coordinates are normalized in the upright, unmirrored source image. */
export type OverlayHand = {
  landmarks: number[][];
  display_landmarks?: number[][];
  selected: boolean;
  track_id?: number | null;
  handedness?: string;
};
export type OverlayViewport = { left: number; top: number; width: number; height: number };
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

export function containViewport(sourceWidth: number, sourceHeight: number, width: number, height: number): OverlayViewport {
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const w = sourceWidth * scale, h = sourceHeight * scale;
  return { left: (width - w) / 2, top: (height - h) / 2, width: w, height: h };
}

export function validHandPoints(points?: number[][]): points is number[][] {
  return points?.length === 21 && points.every(p => p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]));
}

/** Visual enclosure of current landmarks; separate from the detector crop ROI. */
export function landmarkBounds(points: number[][], viewport: OverlayViewport): OverlayViewport | null {
  if (!validHandPoints(points)) return null;
  const xs = points.map(p => viewport.left + p[0] * viewport.width);
  const ys = points.map(p => viewport.top + p[1] * viewport.height);
  const pad = Math.max(5, Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) * .10);
  const left = Math.max(viewport.left, Math.min(...xs) - pad);
  const top = Math.max(viewport.top, Math.min(...ys) - pad);
  const right = Math.min(viewport.left + viewport.width, Math.max(...xs) + pad);
  const bottom = Math.min(viewport.top + viewport.height, Math.max(...ys) + pad);
  return right > left && bottom > top ? { left, top, width: right - left, height: bottom - top } : null;
}

export function drawHandOverlay(context: CanvasRenderingContext2D, viewport: OverlayViewport, landmarks?: number[][], classified = false, hands?: OverlayHand[]) {
  const rows: OverlayHand[] = hands?.length ? hands : validHandPoints(landmarks) ? [{ landmarks, selected: true }] : [];
  const size = Math.min(viewport.width, viewport.height);
  const project = ([x, y]: number[]) => [viewport.left + x * viewport.width, viewport.top + y * viewport.height];
  context.save();
  context.beginPath();
  context.rect(viewport.left, viewport.top, viewport.width, viewport.height);
  context.clip();
  for (const hand of rows) {
    // Current observations keep boxes aligned with the paired source JPEG.
    const points = validHandPoints(hand.landmarks) ? hand.landmarks
      : validHandPoints(hand.display_landmarks) ? hand.display_landmarks
        : hand.selected && validHandPoints(landmarks) ? landmarks : undefined;
    if (!points) continue;
    const box = landmarkBounds(points, viewport);
    if (!box) continue;
    const boxColor = hand.selected ? '#ff7484' : '#9b8cff';
    context.strokeStyle = boxColor;
    context.lineWidth = Math.max(1.5, size / 230);
    context.strokeRect(box.left, box.top, box.width, box.height);
    context.strokeStyle = '#45d6a1';
    context.lineWidth = Math.max(1.5, size / 250);
    context.beginPath();
    for (const [from, to] of CONNECTIONS) {
      const a = project(points[from]), b = project(points[to]);
      context.moveTo(a[0], a[1]);
      context.lineTo(b[0], b[1]);
    }
    context.stroke();
    for (const [x, y] of points.map(project)) {
      context.beginPath();
      context.fillStyle = '#b7ff65';
      context.arc(x, y, Math.max(2, size / 170), 0, Math.PI * 2);
      context.fill();
    }
    const state = hand.selected ? (classified ? 'selected' : 'checking') : 'observed';
    const label = `${hand.handedness ?? 'Hand'}${hand.track_id != null ? ` #${hand.track_id}` : ''} · ${state}`;
    const fontSize = Math.max(10, Math.min(15, size / 30));
    context.font = `${fontSize}px sans-serif`;
    const textWidth = context.measureText(label).width;
    const x = Math.max(viewport.left + 3, Math.min(viewport.left + viewport.width - textWidth - 6, box.left));
    const y = Math.max(viewport.top + fontSize + 5, box.top - 6);
    context.fillStyle = 'rgba(10, 15, 24, .88)';
    context.fillRect(x - 3, y - fontSize - 3, textWidth + 6, fontSize + 7);
    context.fillStyle = boxColor;
    context.fillText(label, x, y);
  }
  context.restore();
}
