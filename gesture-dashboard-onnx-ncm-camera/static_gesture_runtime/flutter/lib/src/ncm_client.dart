import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

/// The board clock is retained for diagnostics. [receivedTimestampMs] uses the
/// client's monotonic clock and is the gesture-runtime timestamp.
class NcmJpegFrame {
  const NcmJpegFrame(
      {required this.jpeg,
      required this.sequence,
      required this.boardTimestampMs,
      required this.receivedTimestampMs});
  final Uint8List jpeg;
  final int sequence;
  final int boardTimestampMs;
  final double receivedTimestampMs;
}

class JlipPacket {
  const JlipPacket(this.type, this.sequence, this.timestampMs, this.payload);
  final int type, sequence, timestampMs;
  final Uint8List payload;
}

/// Big-endian JLIP v1 parser, including fragmented/coalesced TCP reads and CRC.
class JlipParser {
  JlipParser({this.maximumPayloadBytes = 200 * 1024});
  final int maximumPayloadBytes;
  Uint8List _pending = Uint8List(0);
  int crcErrors = 0, headerErrors = 0, discardedBytes = 0;
  int get bufferedBytes => _pending.length;

  void reset() => _pending = Uint8List(0);

  List<JlipPacket> feed(List<int> bytes) {
    final input = Uint8List(_pending.length + bytes.length)
      ..setAll(0, _pending)
      ..setAll(_pending.length, bytes);
    final packets = <JlipPacket>[];
    var offset = 0;
    while (true) {
      var magic = -1;
      for (var i = offset; i + 3 < input.length; i++) {
        if (input[i] == 74 &&
            input[i + 1] == 76 &&
            input[i + 2] == 73 &&
            input[i + 3] == 80) {
          magic = i;
          break;
        }
      }
      if (magic < 0) {
        final keep = (input.length - offset).clamp(0, 3);
        discardedBytes += input.length - offset - keep;
        offset = input.length - keep;
        break;
      }
      discardedBytes += magic - offset;
      offset = magic;
      if (input.length - offset < 24) break;
      final header = ByteData.sublistView(input, offset, offset + 24);
      final length = header.getUint32(12);
      if (header.getUint8(4) != 1 ||
          header.getUint16(6) != 24 ||
          length > maximumPayloadBytes) {
        headerErrors++;
        discardedBytes++;
        offset++;
        continue;
      }
      if (input.length - offset < 24 + length) break;
      final payload =
          Uint8List.fromList(input.sublist(offset + 24, offset + 24 + length));
      offset += 24 + length;
      if (crc32(payload) != header.getUint32(16)) {
        crcErrors++;
        continue;
      }
      packets.add(JlipPacket(header.getUint8(5), header.getUint32(8),
          header.getUint32(20), payload));
    }
    _pending = Uint8List.fromList(input.sublist(offset));
    return packets;
  }

  static final List<int> _crcTable = List<int>.generate(256, (value) {
    var c = value;
    for (var bit = 0; bit < 8; bit++) {
      c = c & 1 != 0 ? (c >> 1) ^ 0xedb88320 : c >> 1;
    }
    return c;
  });
  static int crc32(List<int> bytes) {
    var crc = 0xffffffff;
    for (final value in bytes) {
      crc = (crc >> 8) ^ _crcTable[(crc ^ value) & 255];
    }
    return (crc ^ 0xffffffff) & 0xffffffff;
  }

  static Uint8List packet(
      {required int type,
      required int sequence,
      required int timestampMs,
      List<int> payload = const []}) {
    final result = Uint8List(24 + payload.length);
    result.setAll(0, [74, 76, 73, 80, 1, type]);
    final data = ByteData.sublistView(result);
    data.setUint16(6, 24);
    data.setUint32(8, sequence & 0xffffffff);
    data.setUint32(12, payload.length);
    data.setUint32(16, crc32(payload));
    data.setUint32(20, timestampMs & 0xffffffff);
    result.setAll(24, payload);
    return result;
  }

  static Uint8List? jpegPayload(Uint8List payload) {
    if (payload.length < 4 || payload[0] != 255 || payload[1] != 216)
      return null;
    for (var i = payload.length - 2; i >= 2; i--) {
      if (payload[i] == 255 && payload[i + 1] == 217) {
        return Uint8List.sublistView(payload, 0, i + 2);
      }
    }
    return null;
  }
}

/// Reads JPEG dimensions from SOF metadata, without decoding pixels.
({int width, int height}) jpegDimensions(Uint8List jpeg) {
  if (jpeg.length < 4 || jpeg[0] != 255 || jpeg[1] != 216) {
    throw const FormatException('Expected JPEG SOI');
  }
  var index = 2;
  while (index < jpeg.length) {
    if (jpeg[index++] != 255)
      throw const FormatException('Invalid JPEG marker');
    while (index < jpeg.length && jpeg[index] == 255) {
      index++;
    }
    if (index >= jpeg.length) break;
    final marker = jpeg[index++];
    if (marker == 217 || marker == 218) break;
    if (marker == 216 || marker == 1 || (marker >= 208 && marker <= 215))
      continue;
    if (index + 2 > jpeg.length) break;
    final length = (jpeg[index] << 8) | jpeg[index + 1];
    if (length < 2 || index + length > jpeg.length) break;
    const sof = <int>{
      192,
      193,
      194,
      195,
      197,
      198,
      199,
      201,
      202,
      203,
      205,
      206,
      207
    };
    if (sof.contains(marker) && length >= 8) {
      final height = (jpeg[index + 3] << 8) | jpeg[index + 4];
      final width = (jpeg[index + 5] << 8) | jpeg[index + 6];
      if (width > 0 && height > 0 && width <= 4096 && height <= 4096) {
        return (width: width, height: height);
      }
      throw const FormatException('Unsupported JPEG dimensions');
    }
    index += length;
  }
  throw const FormatException('JPEG SOF dimensions missing');
}

/// Reconnecting TCP client. The receiving application retains only the newest
/// frame. Stop/restart invalidates callbacks from a previous connection epoch.
class NcmCameraClient {
  NcmCameraClient(
      {required this.onFrame,
      this.onState,
      this.onError,
      this.onDisconnect,
      this.host = '192.168.50.2',
      this.port = 5000,
      this.sourceAddress,
      this.connectTimeout = const Duration(seconds: 3),
      this.reconnectDelay = const Duration(seconds: 1),
      this.idleTimeout = const Duration(seconds: 5)});
  final void Function(NcmJpegFrame) onFrame;
  final void Function(String)? onState;
  final void Function(Object)? onError;
  final void Function()? onDisconnect;
  final String host;
  final int port;
  final String? sourceAddress;
  final Duration connectTimeout, reconnectDelay, idleTimeout;
  final Stopwatch _clock = Stopwatch()..start();
  Socket? _socket;
  Timer? _reconnect, _idle;
  StreamSubscription<Uint8List>? _subscription;
  int _generation = 0, _sequence = 0;
  bool _running = false;
  int framesReceived = 0, crcErrors = 0, headerErrors = 0;
  double get monotonicMs => _clock.elapsedMicroseconds / 1000;

  Future<void> start() async {
    if (_running) return;
    if (port < 1 || port > 65535) throw ArgumentError.value(port, 'port');
    _running = true;
    await _connect(++_generation);
  }

  Future<void> _connect(int epoch) async {
    if (!_running || epoch != _generation) return;
    onState?.call('connecting');
    try {
      final socket = await Socket.connect(host, port,
          timeout: connectTimeout, sourceAddress: sourceAddress);
      if (!_running || epoch != _generation) {
        socket.destroy();
        return;
      }
      socket.setOption(SocketOption.tcpNoDelay, true);
      _socket = socket;
      final parser = JlipParser();
      onState?.call('connected');
      void armIdle() {
        _idle?.cancel();
        _idle = Timer(
            idleTimeout, () => _lost(epoch, StateError('NCM stream idle')));
      }

      armIdle();
      _subscription = socket.listen((bytes) {
        if (!_running || epoch != _generation) return;
        armIdle();
        final packets = parser.feed(bytes);
        crcErrors += parser.crcErrors;
        parser.crcErrors = 0;
        headerErrors += parser.headerErrors;
        parser.headerErrors = 0;
        for (final packet in packets) {
          if (packet.type == 3) {
            socket.add(JlipParser.packet(
                type: 3,
                sequence: _sequence++,
                timestampMs: monotonicMs.toInt()));
          } else if (packet.type == 0x10) {
            final jpeg = JlipParser.jpegPayload(packet.payload);
            if (jpeg != null) {
              framesReceived++;
              onFrame(NcmJpegFrame(
                  jpeg: jpeg,
                  sequence: packet.sequence,
                  boardTimestampMs: packet.timestampMs,
                  receivedTimestampMs: monotonicMs));
            }
          }
        }
      },
          onError: (Object error) => _lost(epoch, error),
          onDone: () => _lost(epoch, StateError('NCM stream closed')),
          cancelOnError: true);
    } catch (error) {
      _lost(epoch, error);
    }
  }

  void _lost(int epoch, Object error) {
    if (!_running || epoch != _generation) return;
    _generation++;
    _idle?.cancel();
    _socket?.destroy();
    _socket = null;
    unawaited(_subscription?.cancel());
    _subscription = null;
    onError?.call(error);
    onDisconnect?.call();
    onState?.call('reconnecting');
    final next = _generation;
    _reconnect?.cancel();
    _reconnect = Timer(reconnectDelay, () => unawaited(_connect(next)));
  }

  Future<void> stop() async {
    _running = false;
    _generation++;
    _reconnect?.cancel();
    _idle?.cancel();
    _socket?.destroy();
    _socket = null;
    await _subscription?.cancel();
    _subscription = null;
    onState?.call('stopped');
  }
}
