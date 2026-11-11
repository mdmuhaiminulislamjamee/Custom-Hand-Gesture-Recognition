import base64
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from backend.config import RuntimeConfig
from backend.storage import FeedbackRecord, FeedbackStore


def _record(data_url=None):
    return FeedbackRecord(
        actual_label='left', predicted_label='right', runtime_action='Right',
        confidence=.9, model_name='ONNX', probabilities={'left': .9},
        snapshot_data_url=data_url)


def _jpeg_data_url():
    stream = BytesIO()
    Image.new('RGB', (4, 4), 'red').save(stream, format='JPEG')
    return 'data:image/jpeg;base64,' + base64.b64encode(stream.getvalue()).decode()


def test_feedback_image_and_csv_commit_together(tmp_path):
    store = FeedbackStore(RuntimeConfig(), tmp_path)
    saved = store.save(_record(_jpeg_data_url()))
    assert (tmp_path / 'images' / (saved['feedback_id'] + '.jpg')).is_file()
    assert len(store.summary()['recent']) == 1


def test_feedback_csv_rolls_back_when_image_commit_fails(tmp_path, monkeypatch):
    store = FeedbackStore(RuntimeConfig(), tmp_path)
    original = Path.replace

    def fail_replace(self, target):
        if self.name.endswith('.jpg.tmp'):
            raise OSError('simulated image commit failure')
        return original(self, target)

    monkeypatch.setattr(Path, 'replace', fail_replace)
    with pytest.raises(OSError, match='simulated image commit failure'):
        store.save(_record(_jpeg_data_url()))
    assert not (tmp_path / 'misclassification_log.csv').exists() or (tmp_path / 'misclassification_log.csv').read_text() == ''
    assert not list((tmp_path / 'images').glob('*.jpg'))
    assert not list((tmp_path / 'images').glob('*.tmp'))
