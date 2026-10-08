# USB-NCM / JLIP camera integration

Windows reference: `desktop_reference/backend/ncm_camera.py` and `desktop_reference/reference/NcmCameraTester.cpp`. The same Python receiver source is in `ios/reference/backend/ncm_camera.py`.

## Network

Host `192.168.50.1/30`, board `192.168.50.2`, TCP image service `5000`. Optional UDP discovery uses port `5001` and ASCII payload `JL-CAMERA-DISCOVER`. Begin with direct TCP to the known IP so discovery permissions are not confused with inference problems.

On iOS use a Network.framework NWConnection TCP client once the OS exposes a usable route. An app cannot assume it can configure an arbitrary USB interface or install the Windows NCM driver. Confirm the exact phone/iPad, USB adapter, firmware, power and iOS combination with the accessory team. If the accessory is not exposed as a network interface, an app-side classifier change cannot fix that; use a supported accessory/network transport.

Provide NSLocalNetworkUsageDescription and exercise local-network permission. For UDP broadcast/multicast discovery, assess Apple's [multicast entitlement](https://developer.apple.com/documentation/bundleresources/entitlements/com.apple.developer.networking.multicast). Bonjour service declarations are relevant only if the firmware actually advertises Bonjour; this protocol uses its own discovery string.

## Packet layout

All multibyte header fields are unsigned, big endian. The fixed header is 24 bytes, Python format `>4sBBHIIII`.

| Offset | Bytes | Meaning |
|---:|---:|---|
| 0 | 4 | ASCII JLIP |
| 4 | 1 | Version 1 |
| 5 | 1 | Packet type |
| 6 | 2 | Header size, 24 |
| 8 | 4 | Sequence |
| 12 | 4 | Payload length |
| 16 | 4 | CRC32 of payload (zlib/IEEE CRC32) |
| 20 | 4 | Sender timestamp in milliseconds, wrapping uint32 |
| 24 | payload length | Payload |

Types: hello ACK `0x02`; heartbeat `0x03`; heartbeat ACK constant `0x04`; JPEG `0x10`. The current receiver responds to heartbeat using **type 0x03**, using its own incrementing client sequence and current uint32 millisecond timestamp with empty payload. Follow the supplied receiver/firmware behavior rather than guessing from the ACK constant's name.

## Receiver requirements

Maintain a persistent byte buffer: a TCP receive callback can contain a partial header, partial JPEG, several packets, or any combination. Search for magic, validate version/header size and payload bound (default 200 KiB), wait for the complete packet, verify CRC, then consume exactly that packet. Reject invalid headers/CRC and resynchronize with bounded memory.

JPEG payload starts FF D8. Require FF D9; trim trailing padding after the final marker as the Python receiver does. Decode successfully before inference. Keep the newest complete valid image; do not queue every transport frame. Heartbeats/metadata are not image data.

On disconnect cancel the connection, clear partial bytes and stale frames, reset detection/decision state, and retry with a bounded delay (reference: one second). Do not dispatch a cached action on reconnect. Count CRC/header/JPEG errors and sequence gaps for diagnostics.

Use local monotonic capture/admission times for MediaPipe and the hold gate; retain the board timestamp only for transport diagnostics. Board frame rate and inference frame rate are different. Use the exact same crop/orientation policy in native code and in the reference you compare against.

## Acceptance

Before recognition testing, prove a rising valid-frame count, correct upright unmirrored image, heartbeat survival and recovery from unplug/replug. Then test all twelve gestures and rejected poses using VALIDATION.md. This handover has not established physical iOS USB-NCM compatibility.
