"""Reject hallucinated hands and resting/touching hands beside a face or ear.

BlazeFace supplies face boxes only; no identity or demographic inference.
The face/ear zone is not a command area. Much larger foreground hands remain
eligible; image coordinates alone cannot prove physical contact or intent.
"""
from pathlib import Path
import time

import numpy as np


def overlaps_face_as_small_hand(points, image_shape, boxes):
    pixels = np.asarray(points) * [image_shape[1], image_shape[0]]
    span = float(np.ptp(pixels, axis=0).max())
    palm = pixels[[0, 5, 9, 13, 17]].mean(axis=0)
    for x, y, width, height in boxes:
        inside = (x - .30 * width <= palm[0] <= x + 1.30 * width
                  and y - .10 * height <= palm[1] <= y + 1.05 * height)
        if inside and span < .95 * max(width, height):
            return True
    return False


def face_context_rejection(points, image_shape, boxes):
    """Return a veto before classifier scoring, independent of gesture label.

    Face boxes omit ears/hair and a resting hand's wrist/palm center can be
    below or outside them. Use the finger joints around an expanded head zone
    too, without letting a long wrist-to-finger span bypass the old check.
    """
    if overlaps_face_as_small_hand(points, image_shape, boxes):
        return "hand_candidate_overlaps_face"
    pixels = np.asarray(points) * [image_shape[1], image_shape[0]]
    fingers = pixels[[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    span = float(np.ptp(pixels, axis=0).max())
    for x, y, width, height in boxes:
        if width <= 0 or height <= 0:
            continue
        # A nearby foreground command hand can project over a smaller face.
        # Same-depth hands at the ear are generally no larger than the head.
        if span > 1.5 * max(width, height):
            continue
        inside = ((fingers[:, 0] >= x - .55 * width)
                  & (fingers[:, 0] <= x + 1.55 * width)
                  & (fingers[:, 1] >= y - .20 * height)
                  & (fingers[:, 1] <= y + 1.15 * height))
        if np.count_nonzero(inside) >= 6:
            return "hand_near_face_or_ear"
    return None


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

    def detect(self, rgb):
        if self.detector is None:
            return []
        import cv2
        height, width = rgb.shape[:2]
        now = time.monotonic()
        if self._image_shape != (height, width):
            self.boxes = []
            self._recent_boxes = []
            self._image_shape = (height, width)
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
        retained = [(box, seen) for box, seen in self._recent_boxes
                    if now - seen <= .5 and not refreshed(box)]
        self._recent_boxes = retained + [(box, now) for box in boxes]
        self.boxes = [box for box, _ in self._recent_boxes]
        # Brief profile/hand occlusions must not turn a resting hand into a
        # command. Expire the cache by elapsed time, never by frame count.
        return list(self.boxes)

    def close(self):
        if self.detector is not None:
            self.detector.close()
