"""Verify a native false-hand source-update bundle using Python 3.10+."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys

def verify(root):
    root=root.resolve()
    hashes=json.loads((root/'SHA256SUMS.json').read_text())
    for relative,expected in hashes.items():
        path=PurePosixPath(relative)
        if path.is_absolute() or '..' in path.parts or '\\' in relative or ':' in relative:raise ValueError('Unsafe path')
        file=root.joinpath(*path.parts).resolve()
        if not file.is_relative_to(root) or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError('Changed or missing: '+relative)
    face=root/'models/blaze_face_short_range.tflite'
    if hashlib.sha256(face.read_bytes()).hexdigest()!='b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f':raise ValueError('Wrong face model')
    cases=json.loads((root/'false_hand_policy.json').read_text())['cases']
    if len(cases)!=42:raise ValueError('Wrong policy fixture set')
    return dict(passed=True,verified_files=len(hashes),policy_cases=len(cases),physical_device_tested=False,ios_compile_tested=False)

if __name__=='__main__':print(json.dumps(verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent),indent=2))
