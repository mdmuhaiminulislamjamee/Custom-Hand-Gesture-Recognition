# USB-NCM / JLIP camera protocol

Both mobile integrations use the board camera, not the built-in phone camera. The app runs inference locally; it does not need a Windows Python server, browser or BAT file.

Reference network: host `192.168.50.1/30`, board `192.168.50.2`, TCP `5000`. Optional UDP discovery uses `5001` with ASCII `JL-CAMERA-DISCOVER`. The phone OS must expose a usable accessory network route. The native TCP receivers do not implement USB-NCM drivers or address assignment. Validate the intended phone/OS/accessory/firmware/cable combination.

On iOS use the supplied Network.framework receiver after local-network permission and routing succeed. On Android choose the OS-provided network using `NcmNetworkSelector` and its socket factory; do not redirect unrelated application traffic.

## Header

The fixed header is 24 bytes, Python format `>4sBBHIIII`; multibyte integers are big endian.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | ASCII `JLIP` |
| 4 | 1 | Version 1 |
| 5 | 1 | Packet type |
| 6 | 2 | Header size 24 |
| 8 | 4 | Sequence number |
| 12 | 4 | Payload length, bounded to 200 KiB |
| 16 | 4 | IEEE/zlib CRC32 of payload |
| 20 | 4 | Board timestamp in milliseconds, wrapping uint32 |
| 24 | variable | Payload |

Types include hello ACK `0x02`, heartbeat `0x03`, heartbeat ACK constant `0x04`, and JPEG `0x10`. Match the current receiver/firmware: reply to heartbeat using **type 0x03**, an incrementing client sequence, current wrapping millisecond timestamp and empty payload.

## Receiver and frame handling

TCP callbacks may contain partial headers, partial images or several packets. Maintain a bounded persistent byte buffer, search/resynchronize on magic, validate header/version/size/length, wait for the complete payload, verify CRC and consume exactly one packet at a time. Never treat heartbeat/metadata payloads as images.

JPEG starts `FF D8` and must contain a final `FF D9`. Trim trailing padding after the last end marker and require successful image decoding. Keep only the newest complete valid JPEG for inference; do not grow an unbounded queue. Retain board sequence/timestamp and CRC/JPEG error counts for diagnostics.

After disconnect, clear partial bytes and pending frames, reset inference/ownership/temporal/action state and reconnect with a bounded delay (reference one second). Never dispatch a cached action on reconnect. Use a local monotonic clock for detector timestamps and holds; board timestamps serve transport diagnostics.

Follow the selected reference crop/orientation consistently. The dashboard uses the 0.92 center crop and serial 10-FPS admission; the standalone viewer uses the full board frame. Preserve the resulting image dimensions alongside landmarks for pixel-angle rules.

Authoritative implementations are the delivered `reference/backend/ncm_camera.py`, the native NCM source, and `reference/protocol/NcmCameraTester.cpp`. First prove a rising valid-JPEG counter, heartbeat survival and unplug/replug recovery, then run gesture acceptance. A successful Windows USB connection is not evidence of phone USB-NCM compatibility.
