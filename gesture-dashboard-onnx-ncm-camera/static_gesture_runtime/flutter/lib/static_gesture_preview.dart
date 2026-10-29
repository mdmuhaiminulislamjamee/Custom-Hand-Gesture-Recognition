import 'package:flutter/material.dart';
import 'package:hand_detection/hand_detection.dart';

import 'ncm_static_gesture_runtime.dart';

/// Fits an unmirrored NCM frame and the dynamic runtime's original overlay to
/// the same rectangle. Shows boxes, 21 dots and skeletons for both detections.
class StaticNcmPreview extends StatelessWidget {
  const StaticNcmPreview({
    super.key,
    required this.frame,
    this.showOverlay = true,
  });

  final NcmStaticPreviewFrame frame;
  final bool showOverlay;

  @override
  Widget build(BuildContext context) => Semantics(
        label: 'NCM camera: ${frame.detections.length} detected hands',
        image: true,
        child: AspectRatio(
          aspectRatio: frame.width / frame.height,
          child: ClipRect(
            child: ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(
                child: Image.memory(
                  frame.source.jpeg,
                  key: ValueKey(
                      '${frame.source.sequence}:${frame.source.receivedTimestampMs}'),
                  fit: BoxFit.fill,
                  gaplessPlayback: false,
                  excludeFromSemantics: true,
                  frameBuilder: (context, child, imageFrame, synchronous) =>
                      CustomPaint(
                    // Wait for this JPEG to decode before painting its overlay.
                    // A previous image must never sit beneath new landmarks.
                    foregroundPainter: showOverlay && imageFrame != null
                        ? MultiOverlayPainter(results: frame.detections)
                        : null,
                    child: child,
                  ),
                ),
              ),
            ),
          ),
        ),
      );
}
