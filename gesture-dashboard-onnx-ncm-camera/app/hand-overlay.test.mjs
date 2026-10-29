import test from 'node:test';
import assert from 'node:assert/strict';
import { containViewport, drawHandOverlay, landmarkBounds } from './hand-overlay.ts';

const points = (x = .2) => Array.from({ length: 21 }, (_, i) => [x + (i % 5) * .02, .2 + i * .015]);

test('4:3 frames fit square and wide previews without stretching or mirroring', () => {
  assert.deepEqual(containViewport(640, 480, 400, 400), { left: 0, top: 50, width: 400, height: 300 });
  assert.deepEqual(containViewport(640, 480, 800, 450), { left: 100, top: 0, width: 600, height: 450 });
  const viewport = containViewport(640, 480, 400, 400);
  const left = landmarkBounds(points(.1), viewport);
  const right = landmarkBounds(points(.6), viewport);
  assert.ok(left.left < right.left);
  assert.ok(left.top >= viewport.top);
});

test('boxes stay inside image boundaries and invalid hands are skipped individually', () => {
  const viewport = containViewport(640, 480, 400, 400);
  const box = landmarkBounds(points(-.02), viewport);
  assert.equal(box.left, 0);
  const invalid = points();
  invalid[5][0] = NaN;
  assert.equal(landmarkBounds(invalid, viewport), null);
  const boxes = [], dots = [], labels = [];
  const context = new Proxy({
    strokeRect: (...args) => boxes.push(args),
    arc: (...args) => dots.push(args),
    fillText: label => labels.push(label),
    measureText: label => ({ width: label.length * 6 }),
  }, { get: (target, name) => target[name] ?? (() => {}) });
  drawHandOverlay(context, viewport, undefined, false, [
    { landmarks: invalid, selected: false },
    { landmarks: points(.1), selected: false, handedness: 'Left', track_id: 1 },
    { landmarks: points(.6), selected: true, handedness: 'Right', track_id: 2 },
  ]);
  assert.equal(boxes.length, 2);
  assert.equal(dots.length, 42);
  assert.match(labels[0], /Left #1/);
  assert.match(labels[1], /Right #2/);
});

test('empty detector results draw no boxes or dots', () => {
  let shapes = 0;
  const context = new Proxy({}, { get: (_, name) => name === 'strokeRect' || name === 'arc' ? () => shapes++ : () => {} });
  drawHandOverlay(context, containViewport(640, 480, 400, 300), undefined, false, []);
  assert.equal(shapes, 0);
});
