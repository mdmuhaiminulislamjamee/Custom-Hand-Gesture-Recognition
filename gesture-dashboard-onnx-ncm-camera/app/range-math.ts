export type PalmObservation = {
  palm_segments_px?: number[];
  palm_scale_px?: number;
  hand_span_px?: number;
  analysis_width?: number;
  analysis_height?: number;
  selected_handedness?: string;
  rejection_reason?: string | null;
  small_hand?: boolean;
};
export type RangeCalibration = {
  distance: number; segments: number[]; width: number; height: number; hand?: string;
};
export function median(values: number[]) {
  const rows = [...values].sort((a, b) => a - b);
  return rows.length ? (rows[Math.floor((rows.length - 1) / 2)] + rows[Math.floor(rows.length / 2)]) / 2 : NaN;
}
export function calibrateRange(rows: PalmObservation[], distance: number): RangeCalibration {
  if (!Number.isFinite(distance) || distance < .1 || distance > 10) throw Error('Enter a measured distance from 0.1 to 10 m.');
  if (rows.length < 10) throw Error('Keep your palm visible and still for at least one second.');
  const last = rows.at(-1)!;
  if (!last.analysis_width || !last.analysis_height || rows.some(row =>
    row.rejection_reason || row.palm_segments_px?.length !== 4 ||
    row.palm_segments_px.some(v => !Number.isFinite(v) || v < 12) ||
    row.analysis_width !== last.analysis_width || row.analysis_height !== last.analysis_height ||
    row.selected_handedness !== last.selected_handedness)) throw Error('Use the same hand and camera view, with a larger, clearly visible palm.');
  const segments = [0, 1, 2, 3].map(i => median(rows.map(row => row.palm_segments_px![i])));
  if (rows.some(row => row.palm_segments_px!.some((v, i) => Math.abs(v / segments[i] - 1) > .10))) {
    throw Error('Palm size is changing. Hold still and try again.');
  }
  return { distance, segments, width: last.analysis_width, height: last.analysis_height, hand: last.selected_handedness };
}
export function estimateRange(calibration: RangeCalibration | null, observation?: PalmObservation) {
  if (!calibration || !observation || observation.rejection_reason || observation.palm_segments_px?.length !== 4 ||
    observation.analysis_width !== calibration.width || observation.analysis_height !== calibration.height ||
    (calibration.hand && observation.selected_handedness !== calibration.hand)) return null;
  const distances = observation.palm_segments_px.map((v, i) => calibration.distance * calibration.segments[i] / v);
  if (observation.palm_segments_px.some(v => !Number.isFinite(v) || v < 8) || distances.some(v => !Number.isFinite(v))) return null;
  const distance = median(distances), lower = Math.min(...distances), upper = Math.max(...distances);
  return { distance, lower, upper, inconsistent: (upper - lower) / distance > .3 };
}

/**
 * Estimates distance in meters from calibration if available, or apparent palm scale via pinhole model.
 */
export function estimateGestureDistance(
  observation?: PalmObservation,
  calibration?: RangeCalibration | null,
  backendDistance?: number | null
): { distance: number; text: string; isCalibrated: boolean } | null {
  if (calibration && observation) {
    const calibrated = estimateRange(calibration, observation);
    if (calibrated && Number.isFinite(calibrated.distance)) {
      return {
        distance: calibrated.distance,
        text: `~${calibrated.distance.toFixed(2)} m`,
        isCalibrated: true,
      };
    }
  }
  if (typeof backendDistance === 'number' && Number.isFinite(backendDistance)) {
    return {
      distance: backendDistance,
      text: `~${backendDistance.toFixed(2)} m`,
      isCalibrated: false,
    };
  }
  if (observation?.palm_scale_px && observation.palm_scale_px >= 8) {
    const width = observation.analysis_width || 640;
    const f = width * 0.85;
    const d = (f * 0.075) / observation.palm_scale_px;
    if (Number.isFinite(d)) {
      const clamped = Math.max(0.15, Math.min(5.0, d));
      return {
        distance: clamped,
        text: `~${clamped.toFixed(2)} m`,
        isCalibrated: false,
      };
    }
  }
  return null;
}

