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
                  fit: BoxFit.fill,
                  gaplessPlayback: true,
                  excludeFromSemantics: true,
                  frameBuilder: (context, child, imageFrame, synchronous) =>
                      CustomPaint(
                    // Retain the last decoded picture while this JPEG loads,
                    // but show landmarks only once their matching image arrives.
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
