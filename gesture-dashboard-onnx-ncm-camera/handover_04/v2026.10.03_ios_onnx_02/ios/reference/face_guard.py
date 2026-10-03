"""Reject small hallucinated hands attached to a detected face or ear.

BlazeFace supplies face boxes only; no identity or demographic inference.
Large hands in front of a face are not excluded by this spatial check.
"""
from pathlib import Path

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


class FaceGuard:
    def __init__(self, model_path: Path):
        self.detector = None
        self.error = None
        self.boxes = []
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
        scale = min(1., 320 / max(height, width))
        small = cv2.resize(rgb, (round(width * scale), round(height * scale)))
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                                    data=np.ascontiguousarray(small)))
        self.boxes = [(d.bounding_box.origin_x / scale, d.bounding_box.origin_y / scale,
                       d.bounding_box.width / scale, d.bounding_box.height / scale)
                      for d in result.detections]
        return self.boxes

    def close(self):
        if self.detector is not None:
            self.detector.close()
