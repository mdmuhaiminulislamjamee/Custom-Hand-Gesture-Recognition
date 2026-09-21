export type PlaneAngles = {
  xy: number;
  yz: number;
  xz: number;
  zx: number;
};

const PALM_MCP_INDICES = [5, 9, 13, 17];

function signedDegrees(secondAxis: number, firstAxis: number) {
  const degrees = Math.atan2(secondAxis, firstAxis) * 180 / Math.PI;
  return Object.is(degrees, -0) ? 0 : degrees;
}

/**
 * Returns the signed orientation of the wrist-to-palm axis in each 3-D plane (XY, YZ, XZ/ZX).
 * MediaPipe's coordinate signs are preserved, so the values remain useful for
 * comparing poses without pretending that monocular depth is metric distance.
 */
export function planeAnglesFromLandmarks(landmarks?: number[][] | null): PlaneAngles | null {
  if (landmarks?.length !== 21 || landmarks.some(point =>
    point.length < 3 || !point.slice(0, 3).every(Number.isFinite))) return null;

  const wrist = landmarks[0];
  const palm = [0, 1, 2].map(axis =>
    PALM_MCP_INDICES.reduce((sum, index) => sum + landmarks[index][axis], 0) / PALM_MCP_INDICES.length);
  const [x, y, z] = palm.map((value, axis) => value - wrist[axis]);
  if (Math.hypot(x, y, z) < 1e-7) return null;

  return {
    xy: signedDegrees(y, x),
    yz: signedDegrees(z, y),
    xz: signedDegrees(z, x),
    zx: signedDegrees(x, z),
  };
}

export function validPlaneAngles(value?: Partial<PlaneAngles> | null): value is PlaneAngles {
  return Boolean(
    value &&
    Number.isFinite(value.xy) &&
    Number.isFinite(value.yz) &&
    (Number.isFinite(value.xz) || Number.isFinite(value.zx))
  );
}
