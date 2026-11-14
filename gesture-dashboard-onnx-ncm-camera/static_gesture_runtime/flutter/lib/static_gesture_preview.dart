import 'package:flutter/material.dart';
import 'package:hand_detection/hand_detection.dart';

import 'ncm_static_gesture_runtime.dart';

/// Fits an unmirrored NCM frame and the dynamic runtime's original overlay to
/// the same rectangle. Shows boxes, 21 dots and skeletons for both detections.
class StaticNcmPreview extends StatefulWidget {
  const StaticNcmPreview({
    super.key,
    required this.frame,
    this.showOverlay = true,
  });

  final NcmStaticPreviewFrame frame;
  final bool showOverlay;

  @override
  State<StaticNcmPreview> createState() => _StaticNcmPreviewState();
}

class _StaticNcmPreviewState extends State<StaticNcmPreview> {
  NcmStaticPreviewFrame? _displayedFrame;

  @override
  void didUpdateWidget(covariant StaticNcmPreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.frame.width != widget.frame.width ||
        oldWidget.frame.height != widget.frame.height) {
      _displayedFrame = null;
    }
  }

  @override
  Widget build(BuildContext context) => Semantics(
        label: 'NCM camera: ${widget.frame.detections.length} detected hands',
        image: true,
        child: AspectRatio(
          aspectRatio: widget.frame.width / widget.frame.height,
          child: ClipRect(
            child: ColoredBox(
              color: Colors.black,
              child: SizedBox.expand(
                child: Image.memory(
                  widget.frame.source.jpeg,
                  key: ValueKey((widget.frame.width, widget.frame.height)),
                  fit: BoxFit.fill,
                  gaplessPlayback: true,
                  excludeFromSemantics: true,
                  frameBuilder: (context, child, imageFrame, synchronous) {
                    // gaplessPlayback retains the previous decoded image while
                    // the next JPEG loads. Keep its own overlay too, then swap
                    // both together, including a genuine empty-hand result.
                    if (imageFrame != null) _displayedFrame = widget.frame;
                    final displayed = _displayedFrame;
                    return CustomPaint(
                      foregroundPainter: widget.showOverlay && displayed != null
                          ? MultiOverlayPainter(results: displayed.detections)
                          : null,
                      child: child,
                    );
                  },
                ),
              ),
            ),
          ),
        ),
      );
}
