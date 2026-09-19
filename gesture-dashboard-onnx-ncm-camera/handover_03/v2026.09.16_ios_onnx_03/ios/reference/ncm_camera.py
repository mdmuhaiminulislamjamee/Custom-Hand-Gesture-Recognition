from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
import socket
import struct
import threading
import time
from typing import Any
import zlib


JLIP_MAGIC = b"JLIP"
JLIP_VERSION = 1
JLIP_HEADER_SIZE = 24
JLIP_TYPE_HELLO_ACK = 0x02
JLIP_TYPE_HEARTBEAT = 0x03
JLIP_TYPE_HEARTBEAT_ACK = 0x04
JLIP_TYPE_JPEG = 0x10
DEFAULT_MAX_PAYLOAD = 200 * 1024


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class NcmCameraConfig:
    host_ip: str = "192.168.50.1"
    device_ip: str = "192.168.50.2"
    tcp_port: int = 5000
    discovery_port: int = 5001
    discovery_payload: bytes = b"JL-CAMERA-DISCOVER"
    connect_timeout_seconds: float = 3.0
    receive_timeout_seconds: float = 0.5
    discovery_timeout_seconds: float = 1.2
    reconnect_delay_seconds: float = 1.0
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD
    rotate_180: bool = False
    auto_connect: bool = False

    @classmethod
    def from_environment(cls) -> "NcmCameraConfig":
        return cls(
            host_ip=os.getenv("NCM_HOST_IP", cls.host_ip),
            device_ip=os.getenv("NCM_DEVICE_IP", cls.device_ip),
            tcp_port=int(os.getenv("NCM_TCP_PORT", str(cls.tcp_port))),
            discovery_port=int(
                os.getenv("NCM_DISCOVERY_PORT", str(cls.discovery_port))
            ),
            discovery_payload=os.getenv(
                "NCM_DISCOVERY_PAYLOAD", "JL-CAMERA-DISCOVER"
            ).encode("ascii"),
            connect_timeout_seconds=float(
                os.getenv("NCM_CONNECT_TIMEOUT_SECONDS", "3.0")
            ),
            receive_timeout_seconds=float(
                os.getenv("NCM_RECEIVE_TIMEOUT_SECONDS", "0.5")
            ),
            discovery_timeout_seconds=float(
                os.getenv("NCM_DISCOVERY_TIMEOUT_SECONDS", "1.2")
            ),
            reconnect_delay_seconds=float(
                os.getenv("NCM_RECONNECT_DELAY_SECONDS", "1.0")
            ),
            max_payload_bytes=int(
                os.getenv("NCM_MAX_JPEG_BYTES", str(DEFAULT_MAX_PAYLOAD))
            ),
            rotate_180=_env_bool("NCM_ROTATE_180", False),
            auto_connect=_env_bool("NCM_AUTO_CONNECT", False),
        )

    def public_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["discovery_payload"] = self.discovery_payload.decode(
            "ascii", errors="replace"
        )
        return values


@dataclass(frozen=True)
class JlipPacket:
    packet_type: int
    sequence: int
    timestamp_ms: int
    payload: bytes


def build_jlip_packet(
    packet_type: int,
    sequence: int,
    payload: bytes = b"",
    timestamp_ms: int | None = None,
) -> bytes:
    timestamp = (
        int(time.time() * 1000) & 0xFFFFFFFF
        if timestamp_ms is None
        else timestamp_ms & 0xFFFFFFFF
    )
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    header = struct.pack(
        ">4sBBHIIII",
        JLIP_MAGIC,
        JLIP_VERSION,
        packet_type & 0xFF,
        JLIP_HEADER_SIZE,
        sequence & 0xFFFFFFFF,
        len(payload),
        checksum,
        timestamp,
    )
    return header + payload


class JlipStreamParser:
    """Incrementally parses fragmented/coalesced JLIP packets from TCP."""

    def __init__(self, max_payload_bytes: int = DEFAULT_MAX_PAYLOAD) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.buffer = bytearray()
        self.crc_errors = 0
        self.header_errors = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> list[JlipPacket]:
        if data:
            self.buffer.extend(data)
        packets: list[JlipPacket] = []
        while True:
            magic_index = self.buffer.find(JLIP_MAGIC)
            if magic_index < 0:
                keep = min(len(self.buffer), len(JLIP_MAGIC) - 1)
                discard = len(self.buffer) - keep
                if discard:
                    del self.buffer[:discard]
                    self.discarded_bytes += discard
                break
            if magic_index:
                del self.buffer[:magic_index]
                self.discarded_bytes += magic_index
            if len(self.buffer) < JLIP_HEADER_SIZE:
                break

            (
                magic,
                version,
                packet_type,
                header_size,
                sequence,
                payload_size,
                expected_crc,
                timestamp_ms,
            ) = struct.unpack(">4sBBHIIII", self.buffer[:JLIP_HEADER_SIZE])
            if (
                magic != JLIP_MAGIC
                or version != JLIP_VERSION
                or header_size != JLIP_HEADER_SIZE
                or payload_size > self.max_payload_bytes
            ):
                del self.buffer[0]
                self.header_errors += 1
                self.discarded_bytes += 1
                continue

            packet_size = header_size + payload_size
            if len(self.buffer) < packet_size:
                break
            payload = bytes(self.buffer[header_size:packet_size])
            del self.buffer[:packet_size]
            actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
            if actual_crc != expected_crc:
                self.crc_errors += 1
                continue
            packets.append(
                JlipPacket(
                    packet_type=packet_type,
                    sequence=sequence,
                    timestamp_ms=timestamp_ms,
                    payload=payload,
                )
            )
        return packets


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _looks_like_jpeg(payload: bytes) -> bool:
    return len(payload) >= 4 and payload[:2] == b"\xff\xd8" and payload[-2:] == b"\xff\xd9"


class NcmCameraClient:
    """Reconnectable JLIP camera client for the USB-NCM development board."""

    def __init__(self, config: NcmCameraConfig | None = None) -> None:
        self.config = config or NcmCameraConfig.from_environment()
        self._lock = threading.RLock()
        self._frame_ready = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_id = 0
        self._frame_times: list[float] = []
        self._state = "stopped"
        self._last_error: str | None = None
        self._last_frame_utc: str | None = None
        self._connected_since_utc: str | None = None
        self._device_metadata: dict[str, Any] | str | None = None
        self._packets_received = 0
        self._frames_received = 0
        self._bytes_received = 0
        self._crc_errors = 0
        self._header_errors = 0
        self._invalid_jpeg_frames = 0
        self._sequence_gaps = 0
        self._last_sequence: int | None = None
        self._reconnects = 0
        self._client_sequence = 0

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._state = "connecting"
            self._last_error = None
            self._latest_jpeg = None
            self._frame_times = []
            self._thread = threading.Thread(
                target=self._run,
                name="ncm-jlip-camera",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._close_socket()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        with self._frame_ready:
            self._thread = None
            self._state = "stopped"
            self._connected_since_utc = None
            self._frame_ready.notify_all()

    def _close_socket(self) -> None:
        with self._lock:
            connection = self._socket
            self._socket = None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def discover(self) -> dict[str, Any]:
        started = time.perf_counter()
        discovery = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        discovery.settimeout(self.config.discovery_timeout_seconds)
        try:
            discovery.bind((self.config.host_ip, 0))
            discovery.sendto(
                self.config.discovery_payload,
                (self.config.device_ip, self.config.discovery_port),
            )
            payload, peer = discovery.recvfrom(4096)
            return {
                "ok": True,
                "peer_ip": peer[0],
                "peer_port": peer[1],
                "payload_text": payload.decode("utf-8", errors="replace"),
                "payload_hex": payload.hex(),
                "round_trip_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except socket.timeout:
            return {
                "ok": False,
                "stage": "udp_discovery",
                "message": (
                    f"No UDP reply from {self.config.device_ip}:"
                    f"{self.config.discovery_port} within "
                    f"{self.config.discovery_timeout_seconds:.1f} seconds."
                ),
            }
        except OSError as error:
            return {
                "ok": False,
                "stage": "host_bind_or_udp_send",
                "message": str(error),
            }
        finally:
            discovery.close()

    def latest_frame(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._frame_id, self._latest_jpeg

    def wait_for_frame(
        self, after_frame_id: int, timeout_seconds: float = 1.0
    ) -> tuple[int, bytes | None]:
        deadline = time.monotonic() + timeout_seconds
        with self._frame_ready:
            while (
                self._frame_id <= after_frame_id
                and not self._stop_event.is_set()
                and time.monotonic() < deadline
            ):
                self._frame_ready.wait(timeout=max(0.0, deadline - time.monotonic()))
            return self._frame_id, self._latest_jpeg

    def status(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            recent = [value for value in self._frame_times if now - value <= 3.0]
            camera_fps = 0.0
            if len(recent) > 1:
                camera_fps = (len(recent) - 1) / (recent[-1] - recent[0])
            return {
                "state": self._state,
                "running": bool(self._thread and self._thread.is_alive()),
                "connected": self._state == "connected",
                "config": self.config.public_dict(),
                "frame_id": self._frame_id,
                "frames_received": self._frames_received,
                "camera_fps": round(camera_fps, 3),
                "packets_received": self._packets_received,
                "bytes_received": self._bytes_received,
                "crc_errors": self._crc_errors,
                "header_errors": self._header_errors,
                "invalid_jpeg_frames": self._invalid_jpeg_frames,
                "sequence_gaps": self._sequence_gaps,
                "reconnects": self._reconnects,
                "last_error": self._last_error,
                "last_frame_utc": self._last_frame_utc,
                "connected_since_utc": self._connected_since_utc,
                "device_metadata": self._device_metadata,
            }

    def _set_state(self, state: str, error: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._last_error = error

    def _connect(self) -> socket.socket:
        connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(self.config.connect_timeout_seconds)
        try:
            connection.bind((self.config.host_ip, 0))
            connection.connect((self.config.device_ip, self.config.tcp_port))
            connection.settimeout(self.config.receive_timeout_seconds)
            return connection
        except OSError:
            connection.close()
            raise

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._set_state("connecting")
            try:
                connection = self._connect()
                with self._lock:
                    self._socket = connection
                    self._state = "connected"
                    self._last_error = None
                    self._connected_since_utc = _utc_now()
                    self._last_sequence = None
                self._receive_stream(connection)
                if not self._stop_event.is_set():
                    raise ConnectionError("The board closed the JLIP TCP stream.")
            except (OSError, ConnectionError) as error:
                if self._stop_event.is_set():
                    break
                with self._lock:
                    self._state = "reconnecting"
                    self._last_error = str(error)
                    self._connected_since_utc = None
                    self._reconnects += 1
                self._stop_event.wait(self.config.reconnect_delay_seconds)
            finally:
                self._close_socket()
        self._set_state("stopped")

    def _receive_stream(self, connection: socket.socket) -> None:
        parser = JlipStreamParser(self.config.max_payload_bytes)
        last_crc_errors = 0
        last_header_errors = 0
        while not self._stop_event.is_set():
            try:
                data = connection.recv(65536)
            except socket.timeout:
                continue
            if not data:
                return
            with self._lock:
                self._bytes_received += len(data)
            packets = parser.feed(data)
            if parser.crc_errors != last_crc_errors or parser.header_errors != last_header_errors:
                with self._lock:
                    self._crc_errors += parser.crc_errors - last_crc_errors
                    self._header_errors += parser.header_errors - last_header_errors
                last_crc_errors = parser.crc_errors
                last_header_errors = parser.header_errors
            for packet in packets:
                self._handle_packet(connection, packet)

    def _handle_packet(self, connection: socket.socket, packet: JlipPacket) -> None:
        with self._lock:
            self._packets_received += 1
            if self._last_sequence is not None:
                expected = (self._last_sequence + 1) & 0xFFFFFFFF
                if packet.sequence != expected:
                    self._sequence_gaps += (packet.sequence - expected) & 0xFFFFFFFF
            self._last_sequence = packet.sequence

        if packet.packet_type == JLIP_TYPE_HELLO_ACK:
            try:
                metadata: dict[str, Any] | str = json.loads(
                    packet.payload.decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                metadata = packet.payload.decode("utf-8", errors="replace")
            with self._lock:
                self._device_metadata = metadata
            return

        if packet.packet_type == JLIP_TYPE_HEARTBEAT:
            with self._lock:
                sequence = self._client_sequence
                self._client_sequence = (self._client_sequence + 1) & 0xFFFFFFFF
            # The supplied host tester replies with type 0x03, not 0x04.
            connection.sendall(build_jlip_packet(JLIP_TYPE_HEARTBEAT, sequence))
            return

        if packet.packet_type != JLIP_TYPE_JPEG:
            return
        if not _looks_like_jpeg(packet.payload):
            with self._lock:
                self._invalid_jpeg_frames += 1
            return
        jpeg = self._apply_orientation(packet.payload)
        if jpeg is None:
            with self._lock:
                self._invalid_jpeg_frames += 1
            return
        now = time.monotonic()
        with self._frame_ready:
            self._latest_jpeg = jpeg
            self._frame_id += 1
            self._frames_received += 1
            self._last_frame_utc = _utc_now()
            self._frame_times = [
                value for value in self._frame_times if now - value <= 3.0
            ]
            self._frame_times.append(now)
            self._frame_ready.notify_all()

    def _apply_orientation(self, payload: bytes) -> bytes | None:
        if not self.config.rotate_180:
            return payload
        try:
            import cv2
            import numpy as np
        except ImportError:
            return None
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        if self.config.rotate_180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        encoded, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        )
        return bytes(buffer) if encoded else None
