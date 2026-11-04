"""Export synthetic candidate-policy expectations without shipping private photos."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.face_guard import overlaps_face_as_small_hand
from backend.hand_detection import hand_pixel_metrics
from backend.hand_validation import invalid_hand_geometry


def main():
    hand = np.zeros((21, 2), dtype=np.float64)
    hand[0] = [.5, .9]
    for finger, base in enumerate([.32,.43,.54,.65,.76]):
        start = 1 if finger == 0 else 5 + (finger - 1) * 4
        for joint in range(4):
            hand[start + joint] = [base + .015 * joint, .78 - .13 * joint]
    ear = (hand - .5) * .20 + [.56,.31]
    collapsed = (hand - .5) * .5 + [.3,.5]
    for start in (5,9,13,17):
        collapsed[start:start+4] = collapsed[start]
    fingertip = (hand - .5) * .4 + [.38,.3]
    fingertip[8] = [.6,.3]
    world = np.column_stack((hand, np.zeros(21)))
    cases = []
    scenes = [('ear',ear,None),('palm',hand,None),('finger_only',fingertip,None),
              ('collapsed',collapsed,None),('world_rescue',collapsed,world),
              ('tiny',(hand-.5)*.04+[.1,.5],None),('out_of_frame',hand+[2,0],None)]
    for name, source, world_points in scenes:
        for mirrored in (False,True):
            points = source.copy()
            face = [350,80,140,210]
            if mirrored:
                points[:,0] = 1-points[:,0]
                face[0] = 640-face[0]-face[2]
            for scale in (.75,1,1.5):
                shape = (round(480*scale), round(640*scale), 3)
                boxes = [[value*scale for value in face]]
                metrics = hand_pixel_metrics(points,shape)
                reason = None
                if metrics['palm_scale_px'] < 18 or metrics['hand_span_px'] < 40 or metrics['hand_short_side_px'] < 12:
                    reason = 'insufficient_hand_pixels'
                if np.mean(np.all((points >= -.05) & (points <= 1.05),axis=1)) < .9:
                    reason = reason or 'incomplete_hand'
                if not reason and invalid_hand_geometry(points,shape,world_points):
                    reason = 'invalid_hand_geometry'
                if overlaps_face_as_small_hand(points,shape,boxes):
                    reason = 'face_detected_as_hand'
                cases.append(dict(id=f'{name}_{mirrored}_{scale}',width=shape[1],height=shape[0],
                    normalized=points.tolist(),pixels=(points*[shape[1],shape[0]]).tolist(),
                    faces=boxes,world=None if world_points is None else world_points.tolist(),reason=reason))
    destination=ROOT/'static_gesture_runtime/validation/false_hand_policy.json'
    destination.write_text(json.dumps(dict(schema_version=1,source='Python candidate policy; synthetic landmarks',cases=cases),indent=2)+'\n',encoding='utf-8')
    print(f'Exported {len(cases)} policy cases')

if __name__ == '__main__': main()
