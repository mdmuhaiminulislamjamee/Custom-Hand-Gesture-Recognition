from __future__ import annotations

import struct
import socket
import threading
import zlib

from backend.ncm_camera import (
    JLIP_HEADER_SIZE,
    JLIP_MAGIC,
    JLIP_TYPE_HEARTBEAT,
    JLIP_TYPE_JPEG,
    JlipStreamParser,
    NcmCameraClient,
    NcmCameraConfig,
    build_jlip_packet,
)


def test_parser_accepts_fragmented_and_coalesced_packets() -> None:
    first = build_jlip_packet(JLIP_TYPE_HEARTBEAT, 3, b"ping", timestamp_ms=7)
    second = build_jlip_packet(JLIP_TYPE_JPEG, 4, b"\xff\xd8image\xff\xd9", timestamp_ms=8)
    parser = JlipStreamParser()

    assert parser.feed(first[:9]) == []
    packets = parser.feed(first[9:] + second)

    assert [(packet.packet_type, packet.sequence) for packet in packets] == [
        (JLIP_TYPE_HEARTBEAT, 3),
        (JLIP_TYPE_JPEG, 4),
    ]
    assert packets[1].payload == b"\xff\xd8image\xff\xd9"


def test_parser_resynchronizes_after_noise() -> None:
    parser = JlipStreamParser()
    packet = build_jlip_packet(JLIP_TYPE_HEARTBEAT, 9)

    parsed = parser.feed(b"bad-prefix" + packet)

    assert len(parsed) == 1
    assert parsed[0].sequence == 9
    assert parser.discarded_bytes == len(b"bad-prefix")


def test_parser_rejects_crc_mismatch_and_continues() -> None:
    corrupt = bytearray(build_jlip_packet(JLIP_TYPE_HEARTBEAT, 1, b"bad"))
    corrupt[-1] ^= 0xFF
    valid = build_jlip_packet(JLIP_TYPE_HEARTBEAT, 2, b"good")
    parser = JlipStreamParser()

    parsed = parser.feed(bytes(corrupt) + valid)

    assert [packet.sequence for packet in parsed] == [2]
    assert parser.crc_errors == 1


def test_parser_rejects_payload_above_limit_without_allocating_it() -> None:
    oversized_header = struct.pack(
        ">4sBBHIIII",
        JLIP_MAGIC,
        1,
        JLIP_TYPE_JPEG,
        JLIP_HEADER_SIZE,
        1,
        4097,
        zlib.crc32(b"") & 0xFFFFFFFF,
        0,
    )
    parser = JlipStreamParser(max_payload_bytes=4096)

    assert parser.feed(oversized_header) == []
    assert parser.header_errors >= 1


def test_packet_builder_uses_network_byte_order_and_crc32() -> None:
    payload = b"hello"
    packet = build_jlip_packet(JLIP_TYPE_HEARTBEAT, 0x01020304, payload, 99)
    fields = struct.unpack(">4sBBHIIII", packet[:JLIP_HEADER_SIZE])

    assert fields[0] == JLIP_MAGIC
    assert fields[2] == JLIP_TYPE_HEARTBEAT
    assert fields[3] == JLIP_HEADER_SIZE
    assert fields[4] == 0x01020304
    assert fields[5] == len(payload)
    assert fields[6] == zlib.crc32(payload) & 0xFFFFFFFF
    assert fields[7] == 99


def test_horizontal_mirroring_is_not_a_supported_camera_option(monkeypatch) -> None:
    monkeypatch.setenv("NCM_MIRROR_HORIZONTAL", "true")

    config = NcmCameraConfig.from_environment()

    assert not hasattr(config, "mirror_horizontal")
    assert "mirror_horizontal" not in config.public_dict()


def test_udp_discovery_uses_expected_payload_and_returns_reply() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    received: list[bytes] = []

    def answer() -> None:
        payload, peer = server.recvfrom(1024)
        received.append(payload)
        server.sendto(b"JL-CAMERA-HERE", peer)
        server.close()

    worker = threading.Thread(target=answer, daemon=True)
    worker.start()
    client = NcmCameraClient(NcmCameraConfig(
        host_ip="127.0.0.1",
        device_ip="127.0.0.1",
        discovery_port=port,
        discovery_timeout_seconds=1.0,
    ))

    result = client.discover()
    worker.join(timeout=1.0)

    assert result["ok"] is True
    assert result["payload_text"] == "JL-CAMERA-HERE"
    assert received == [b"JL-CAMERA-DISCOVER"]


def test_tcp_client_receives_jpeg_and_replies_to_heartbeat() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    heartbeat_reply: list[bytes] = []

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            stream = (
                build_jlip_packet(0x02, 10, b'{"camera":"mock"}')
                + build_jlip_packet(JLIP_TYPE_HEARTBEAT, 11)
                + build_jlip_packet(JLIP_TYPE_JPEG, 12, b"\xff\xd8mock-jpeg\xff\xd9")
            )
            connection.sendall(stream[:17])
            connection.sendall(stream[17:])
            connection.settimeout(1.0)
            heartbeat_reply.append(connection.recv(1024))
        server.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = NcmCameraClient(NcmCameraConfig(
        host_ip="127.0.0.1",
        device_ip="127.0.0.1",
        tcp_port=port,
        connect_timeout_seconds=1.0,
        receive_timeout_seconds=0.1,
        reconnect_delay_seconds=0.1,
    ))
    client.start()
    frame_id, jpeg = client.wait_for_frame(0, timeout_seconds=2.0)
    client.stop()
    worker.join(timeout=1.0)

    assert frame_id == 1
    assert jpeg == b"\xff\xd8mock-jpeg\xff\xd9"
    replies = JlipStreamParser().feed(heartbeat_reply[0])
    assert len(replies) == 1
    assert replies[0].packet_type == JLIP_TYPE_HEARTBEAT
    status = client.status()
    assert status["frames_received"] == 1
    assert status["device_metadata"] == {"camera": "mock"}


def test_tcp_client_accepts_jpeg_with_trailing_padding() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve() -> None:
        connection, _ = server.accept()
        with connection:
            padded_payload = b"\xff\xd8real-jpeg\xff\xd9" + (b"\x00" * 256)
            stream = build_jlip_packet(JLIP_TYPE_JPEG, 1, padded_payload)
            connection.sendall(stream)
            connection.settimeout(1.0)
            try:
                connection.recv(1024)
            except (socket.timeout, OSError):
                pass
        server.close()

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = NcmCameraClient(NcmCameraConfig(
        host_ip="127.0.0.1",
        device_ip="127.0.0.1",
        tcp_port=port,
        connect_timeout_seconds=1.0,
        receive_timeout_seconds=0.1,
        reconnect_delay_seconds=0.1,
    ))
    client.start()
    frame_id, jpeg = client.wait_for_frame(0, timeout_seconds=2.0)
    client.stop()
    worker.join(timeout=1.0)

    assert frame_id == 1
    assert jpeg == b"\xff\xd8real-jpeg\xff\xd9"
    status = client.status()
    assert status["frames_received"] == 1
    assert status["invalid_jpeg_frames"] == 0
