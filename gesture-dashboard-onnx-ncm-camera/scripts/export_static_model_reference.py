"""Extract a small model-only fixture for native Flutter inference tests."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def export():
    source = ROOT / "static_gesture_runtime/validation/python_reference.json"
    reference = json.loads(source.read_text(encoding="utf-8"))
    small = {key: reference[key] for key in ("schema_version", "model_sha256", "class_names", "feature_count", "probability_tolerance", "model_cases")}
    target = source.with_name("model_reference.json")
    target.write_text(json.dumps(small, indent=2) + "\n", encoding="utf-8")
    print(f"{len(small['model_cases'])} model cases -> {target}")


if __name__ == "__main__":
    export()
