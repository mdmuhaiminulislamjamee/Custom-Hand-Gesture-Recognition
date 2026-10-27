"""Face boxes for collection diagnostics and a bounded live mesh fallback."""
from pathlib import Path
import time

import numpy as np


def overlaps_face_as_small_hand(points, image_shape, boxes):
    """Diagnostic size comparison; callers must not treat it as a veto."""
    pixels = np.asarray(points) * [image_shape[1], image_shape[0]]
    span = float(np.ptp(pixels, axis=0).max())
    palm = pixels[[0, 5, 9, 13, 17]].mean(axis=0)
    for x, y, width, height in boxes:
        inside = (x - .30 * width <= palm[0] <= x + 1.30 * width
                  and y - .10 * height <= palm[1] <= y + 1.05 * height)
        if inside and span < .95 * max(width, height):
            return True
    return False


def face_context_observation(points, image_shape, boxes):
    """Report broad overlap for diagnostics, not evidence of skin contact."""
    pixels = np.asarray(points) * [image_shape[1], image_shape[0]]
    fingers = pixels[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    fingertips = pixels[[4, 8, 12, 16, 20]]
    span = float(np.ptp(pixels, axis=0).max())
    palm = pixels[[0, 5, 9, 13, 17]].mean(axis=0)
    for x, y, width, height in boxes:
        if width <= 0 or height <= 0:
            continue
        # A much smaller face is likely behind a foreground hand; image
        # overlap there cannot mean that the hand touches that face.
        if span > 2.0 * max(width, height):
            continue
        palm_inside = (x - .30 * width <= palm[0] <= x + 1.30 * width
                       and y - .10 * height <= palm[1] <= y + 1.05 * height)
        # Include fingers beside the ear and fingertips touching the face,
        # even when the palm centre lies outside the face box.
        inside = ((fingers[:, 0] >= x - .55 * width)
                  & (fingers[:, 0] <= x + 1.55 * width)
                  & (fingers[:, 1] >= y - .20 * height)
                  & (fingers[:, 1] <= y + 1.15 * height))
        fingertip_inside = ((fingertips[:, 0] >= x - .10 * width)
                            & (fingertips[:, 0] <= x + 1.10 * width)
                            & (fingertips[:, 1] >= y - .10 * height)
                            & (fingertips[:, 1] <= y + 1.10 * height))
        if palm_inside or np.any(fingertip_inside) or np.count_nonzero(inside) >= 6:
            return dict(near_face=True, hand_to_face_span_ratio=round(span / max(width, height), 3))
    return dict(near_face=False)


class FaceGuard:
    def __init__(self, model_path: Path):
        self.detector = None
        self.error = None
        self.boxes = []
        self._recent_boxes = []
        self._image_shape = None
        try:
            import mediapipe as mp
            self.mp = mp
            self.detector = mp.tasks.vision.FaceDetector.create_from_options(
                mp.tasks.vision.FaceDetectorOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    min_detection_confidence=.65,
                ))
        except Exception as error:
            self.error = str(error)

    def detect(self, rgb, *, state=None):
        if self.detector is None:
            return []
        import cv2
        height, width = rgb.shape[:2]
        now = time.monotonic()
        if state is None and self._image_shape != (height, width):
            self.boxes = []
            self._recent_boxes = []
            self._image_shape = (height, width)
        if state is not None:
            cache = state.setdefault("face_boxes", {})
            if cache.get("shape") != (height, width):
                cache.clear()
                cache["shape"] = (height, width)
            recent = cache.get("recent", [])
        else:
            recent = self._recent_boxes
        scale = min(1., 320 / max(height, width))
        small = cv2.resize(rgb, (round(width * scale), round(height * scale)))
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                    data=np.ascontiguousarray(small)))
        boxes = [(d.bounding_box.origin_x / scale, d.bounding_box.origin_y / scale,
                  d.bounding_box.width / scale, d.bounding_box.height / scale)
                 for d in result.detections]
        def refreshed(old):
            x, y, w, h = old
            for bx, by, bw, bh in boxes:
                overlap = (max(0., min(x + w, bx + bw) - max(x, bx))
                           * max(0., min(y + h, by + bh) - max(y, by)))
                if overlap > .3 * min(w * h, bw * bh):
                    return True
            return False

        # Retain each missed face separately: a visible background face must
        # not erase the foreground user's briefly occluded head region.
        retained = [(box, seen) for box, seen in recent
                    if now - seen <= .5 and not refreshed(box)]
        recent = retained + [(box, now) for box in boxes]
        if state is not None:
            cache["recent"] = recent
        else:
            self._recent_boxes = recent
            self.boxes = [box for box, _ in recent]
        # Brief profile/hand occlusions must not turn a resting hand into a
        # command. Expire the cache by elapsed time, never by frame count.
        return [box for box, _ in recent]

    def close(self):
        if self.detector is not None:
            self.detector.close()
