import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:static_gesture_runtime/src/ncm_client.dart';

void main() {
  test('CRC agrees with standard vector and zero-length heartbeat', () {
    expect(JlipParser.crc32(ascii.encode('123456789')), 0xcbf43926);
    expect(JlipParser.crc32([]), 0);
  });

  test('fragmented magic/header/payload and coalesced packets retain metadata',
      () {
    final packet = JlipParser.packet(
        type: 16,
        sequence: 0xffffffff,
        timestampMs: 0xfedcba98,
        payload: [255, 216, 1, 2, 255, 217]);
    final heartbeat = JlipParser.packet(type: 3, sequence: 0, timestampMs: 0);
    for (var split = 0; split <= packet.length; split++) {
      final parser = JlipParser();
      final first = parser.feed(packet.sublist(0, split));
      final rest = parser.feed([...packet.sublist(split), ...heartbeat]);
      final packets = [...first, ...rest];
      expect(packets.length, 2, reason: 'split=$split');
      expect(packets[0].sequence, 0xffffffff);
      expect(packets[0].timestampMs, 0xfedcba98);
      expect(packets[0].payload, [255, 216, 1, 2, 255, 217]);
      expect(packets[1].type, 3);
      expect(parser.bufferedBytes, 0);
    }
  });

  test('bad CRC, garbage, oversized headers resynchronize', () {
    final parser = JlipParser();
    final bad = JlipParser.packet(
        type: 16, sequence: 1, timestampMs: 2, payload: [3, 4]);
    bad[24] ^= 1;
    final large = JlipParser.packet(type: 16, sequence: 4, timestampMs: 5);
    ByteData.sublistView(large).setUint32(12, 300 * 1024);
    final good = JlipParser.packet(type: 3, sequence: 6, timestampMs: 7);
    final packets = parser.feed([0, 1, 2, ...bad, ...large, ...good]);
    expect(packets.single.sequence, 6);
    expect(parser.crcErrors, 1);
    expect(parser.headerErrors, 1);
    expect(parser.discardedBytes, greaterThanOrEqualTo(3));
    expect(parser.bufferedBytes, 0);
  });

  test('bounds garbage storage and trims JPEG payload padding', () {
    final parser = JlipParser();
    parser.feed(List<int>.filled(100000, 42));
    expect(parser.bufferedBytes, 3);
    expect(
        JlipParser.jpegPayload(
            Uint8List.fromList([255, 216, 0, 255, 217, 0, 0])),
        [255, 216, 0, 255, 217]);
    expect(JlipParser.jpegPayload(Uint8List.fromList([255, 216, 0, 0])), null);
  });

  test(
      'SOF image dimensions read without decoding and reject malformed lengths',
      () {
    final jpeg = Uint8List.fromList([
      255,
      216,
      255,
      224,
      0,
      4,
      0,
      0,
      255,
      192,
      0,
      11,
      8,
      1,
      224,
      2,
      128,
      1,
      1,
      17,
      0,
      255,
      217
    ]);
    expect(jpegDimensions(jpeg), (width: 640, height: 480));
    expect(() => jpegDimensions(Uint8List.fromList([255, 216, 255, 224, 0, 0])),
        throwsFormatException);
    expect(() => jpegDimensions(Uint8List.fromList([255, 216, 255, 218, 0, 2])),
        throwsFormatException);
  });
}
