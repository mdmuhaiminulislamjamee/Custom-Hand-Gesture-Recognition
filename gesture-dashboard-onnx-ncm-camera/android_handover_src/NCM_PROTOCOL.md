# Android USB-NCM / JLIP transport

The board camera exposes a TCP service at `192.168.50.2:5000` on a USB-NCM network. The Android device must act as USB host, recognize the board as a network interface, and have a route to `192.168.50.2`. The Windows reference uses host address `192.168.50.1/30`; Android must be checked on the actual target device. USB-NCM enumeration and address assignment are properties of the device/OS integration and are **not** implemented by the Kotlin receiver.

When Android exposes the board as an Ethernet `Network` with `192.168.50.1/30`, `NcmNetworkSelector.find()` can choose it. Supply that `Network` to `GestureNcmClient` so its socket uses the NCM route even while Wi-Fi/cellular is the default. If it returns null, inspect the device's USB and network configuration; do not assume that a regular socket will reach the board. The supplied code does not contain a userspace USB-NCM driver.

## JLIP packet

All multibyte header integers are big-endian. TCP receive boundaries do not match packet boundaries.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 4 | ASCII `JLIP` |
| 4 | 1 | Version `1` |
| 5 | 1 | Type |
| 6 | 2 | Header length `24` |
| 8 | 4 | Sequence |
| 12 | 4 | Payload length, maximum 200 KiB |
| 16 | 4 | IEEE CRC32 of payload |
| 20 | 4 | Board timestamp in milliseconds, uint32 |
| 24 | variable | Payload |

Type `0x10` carries a JPEG. Require FF D8 at the beginning, find the final FF D9, trim any padding after it, and decode successfully. Type `0x03` is a heartbeat; reply with a type `0x03` packet containing an empty payload, a client sequence number and a current uint32 millisecond timestamp. The header has a separate ACK constant `0x04`, but the working receiver/firmware replies to heartbeat with `0x03`. Other packet types are ignored by this image path.

`GestureNcmClient` keeps a persistent parser buffer, validates length and CRC, emits only valid JPEGs, and retries after a disconnect. On reconnect clear partial transport bytes and the app's pending frame/gesture state. Use `SystemClock.elapsedRealtime()` for MediaPipe and hold timing; the board timestamp is for diagnostics and wraps at uint32.

Before any gesture test, prove the device has a route, JPEG count rises, the frame is upright and unmirrored, heartbeats keep the stream alive, and unplug/replug recovers. Hardware compatibility is still unverified for this Android handover.

See the official [Android `Network` API](https://developer.android.com/reference/android/net/Network) for socket binding and [USB host guide](https://developer.android.com/develop/connectivity/usb/host) if the product needs a custom USB driver path.
