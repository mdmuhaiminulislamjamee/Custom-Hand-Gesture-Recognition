import test from 'node:test';
import assert from 'node:assert/strict';
import { calibrateRange, estimateRange, estimateGestureDistance } from './range-math.ts';

const palm = { palm_segments_px: [60, 70, 60, 50], analysis_width: 442, analysis_height: 442, selected_handedness: 'Right' };
test('known-distance calibration obeys inverse apparent size', () => {
  const calibration = calibrateRange(Array(12).fill(palm), .5);
  const estimate = estimateRange(calibration, { ...palm, palm_segments_px: palm.palm_segments_px.map(v => v / 2) });
  assert.equal(estimate?.distance, 1);
  assert.equal(estimate?.lower, 1);
  assert.equal(estimate?.inconsistent, false);
});
test('missing, rejected, different hand and resized views have no distance', () => {
  const calibration = calibrateRange(Array(12).fill(palm), .5);
  assert.equal(estimateRange(calibration), null);
  assert.equal(estimateRange(null, palm), null);
  assert.equal(estimateRange(calibration, { ...palm, rejection_reason: 'insufficient_hand_pixels' }), null);
  assert.equal(estimateRange(calibration, { ...palm, selected_handedness: 'Left' }), null);
  assert.equal(estimateRange(calibration, { ...palm, analysis_width: 640 }), null);
});
test('bad or moving calibration data cannot manufacture metric distance', () => {
  assert.throws(() => calibrateRange(Array(12).fill(palm), NaN));
  assert.throws(() => calibrateRange([palm], .5));
  assert.throws(() => calibrateRange([...Array(11).fill(palm), { ...palm, palm_segments_px: [30, 35, 30, 25] }], .5));
  assert.throws(() => calibrateRange(Array(12).fill({ ...palm, palm_segments_px: [NaN, 0, 4, 3] }), .5));
});
test('pose-dependent segment disagreement is disclosed', () => {
  const calibration = calibrateRange(Array(12).fill(palm), .5);
  const estimate = estimateRange(calibration, { ...palm, palm_segments_px: [30, 70, 60, 50] });
  assert.equal(estimate?.inconsistent, true);
  assert.equal(estimate?.lower, .5);
  assert.equal(estimate?.upper, 1);
});

test('estimateGestureDistance provides calibrated and pinhole fallback distances', () => {
  const calibration = calibrateRange(Array(12).fill(palm), .5);
  const calibrated = estimateGestureDistance({ ...palm, palm_segments_px: palm.palm_segments_px.map(v => v / 2) }, calibration);
  assert.equal(calibrated?.isCalibrated, true);
  assert.equal(calibrated?.text, '~1.00 m');

  const uncalibrated = estimateGestureDistance({ palm_scale_px: 60, analysis_width: 640 });
  assert.equal(uncalibrated?.isCalibrated, false);
  assert.match(uncalibrated?.text ?? '', /^~\d+\.\d{2} m$/);
});
