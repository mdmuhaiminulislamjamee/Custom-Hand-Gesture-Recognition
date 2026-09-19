import test from 'node:test';
import assert from 'node:assert/strict';
import { planeAnglesFromLandmarks, validPlaneAngles } from './landmark-angles.ts';

const landmarksAlong = (x, y, z) => Array.from({ length: 21 }, (_, index) =>
  index === 0 ? [0, 0, 0] : [x, y, z]);

test('computes signed wrist-to-palm orientation in all three planes', () => {
  const angles = planeAnglesFromLandmarks(landmarksAlong(1, 1, 1));
  assert.equal(angles?.xy, 45);
  assert.equal(angles?.yz, 45);
  assert.equal(angles?.xz, 45);
  assert.equal(validPlaneAngles(angles), true);
});

test('preserves axis signs and rejects data without real depth', () => {
  const angles = planeAnglesFromLandmarks(landmarksAlong(1, -1, -1));
  assert.equal(angles?.xy, -45);
  assert.equal(angles?.yz, -135);
  assert.equal(angles?.xz, -45);
  assert.equal(planeAnglesFromLandmarks(Array.from({ length: 21 }, () => [.2, .3])), null);
  assert.equal(planeAnglesFromLandmarks(Array.from({ length: 21 }, () => [0, 0, 0])), null);
});
