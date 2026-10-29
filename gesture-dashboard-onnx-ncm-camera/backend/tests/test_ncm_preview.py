import base64
import time

from fastapi.testclient import TestClient


def test_ncm_preview_carries_the_exact_inferred_jpeg(tmp_path, monkeypatch):
    import backend.app as module
    from backend.data_collection import CollectionStore

    source_jpeg = b'\xff\xd8source-frame-7\xff\xd9'

    class Camera:
        def start(self):
            pass

        def stop(self):
            pass

        def wait_for_frame(self, after_frame_id, timeout):
            if after_frame_id >= 7:
                time.sleep(.02)
            return 7, source_jpeg

        def status(self):
            # The receiver has already advanced while inference ran.
            return {'frame_id': 8, 'camera_fps': 16.0}

    async def process(frame, session, model):
        assert frame is source_jpeg
        return {'status': 'no_hand', 'runtime_prediction': 'no_gesture',
                'runtime_action': 'Wait / No Action', 'hands': []}

    monkeypatch.setattr(module, 'ncm_camera', Camera())
    monkeypatch.setattr(module, '_process_frame_with_limit', process)
    monkeypatch.setattr(module, 'current_health', lambda: {})
    monkeypatch.setattr(module, 'collection_store', CollectionStore(tmp_path))
    with TestClient(module.app).websocket_connect('/ws/ncm-live') as socket:
        assert socket.receive_json()['type'] == 'hello'
        output = socket.receive_json()
    assert output['type'] == 'prediction'
    assert output['ncm_frame_id'] == 7
    assert output['ncm_camera']['frame_id'] == 8
    assert base64.b64decode(output['ncm_preview_jpeg_base64']) == source_jpeg
    assert output['hands'] == []
