import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:hand_detection/hand_detection.dart';

/// Camera formats supported by the existing mobile hand-landmark pipeline.
List<ImageFormatGroup> preferredImageFormats({
  required bool isAndroid,
  required bool isIOS,
}) {
  if (isAndroid) {
    return const <ImageFormatGroup>[
      ImageFormatGroup.nv21,
      ImageFormatGroup.yuv420,
    ];
  }
  if (isIOS) return const <ImageFormatGroup>[ImageFormatGroup.bgra8888];
  return const <ImageFormatGroup>[ImageFormatGroup.bgra8888];
}

/// Migrates the previous two-position percentage setting to a real preset.
ResolutionPreset cameraResolutionPresetFromLegacyScale(double? request) =>
    request != null && request < 0.5
    ? ResolutionPreset.low
    : ResolutionPreset.medium;

String cameraResolutionPresetLabel(ResolutionPreset preset) => switch (preset) {
  ResolutionPreset.low => 'Low',
  ResolutionPreset.medium => 'Medium',
  ResolutionPreset.high => 'High',
  ResolutionPreset.veryHigh => 'Very high',
  ResolutionPreset.ultraHigh => 'Ultra high',
  ResolutionPreset.max => 'Max',
};

/// The PC MJPEG preview follows the collector's established Android mirror
/// convention. iOS camera frames must not receive this additional Dart-side
/// flip; landmark drawing and transmitted coordinates use a separate policy.
bool shouldMirrorPcPreviewFrame({
  required bool isIOS,
  required CameraLensDirection lensDirection,
}) => !isIOS && lensDirection == CameraLensDirection.front;

/// iOS front-camera detector coordinates already match its native preview.
/// Android still needs the established front-camera display-space mirror.
bool shouldMirrorLocalLandmarkOverlay({
  required bool isIOS,
  required CameraLensDirection lensDirection,
}) => !isIOS && lensDirection == CameraLensDirection.front;

/// Applies the display-space X reflection only when the detector input did not
/// already receive the platform's front-camera mirror.
bool shouldMirrorTelemetryLandmarks({
  required bool isIOS,
  required CameraLensDirection lensDirection,
}) => !isIOS && lensDirection == CameraLensDirection.front;

/// Whether the horizontal orientation of the frame sent to the landmark model
/// matches MediaPipe's mirrored/selfie handedness convention.
///
/// The current AVFoundation plugin mirrors its front-camera video-data output.
/// Its back-camera output and CameraX ImageAnalysis buffers are not mirrored.
bool isDetectorInputMirrored({
  required bool isIOS,
  required CameraLensDirection lensDirection,
}) => isIOS && lensDirection == CameraLensDirection.front;

/// Final mirror parity after the camera plugin and telemetry adapter stages.
bool areTelemetryCoordinatesMirrored({
  required bool detectorInputMirrored,
  required bool telemetryMirrorApplied,
}) => detectorInputMirrored != telemetryMirrorApplied;

/// Converts the decoded model label to physical handedness.
///
/// MediaPipe handedness assumes a mirrored model input. An unmirrored detector
/// frame therefore requires swapping the decoded model label.
bool? physicalIsRightHand({
  required bool modelInputMirrored,
  required Handedness? modelHandedness,
}) {
  final modelIsRight = switch (modelHandedness) {
    Handedness.right => true,
    Handedness.left => false,
    null => null,
  };
  if (modelIsRight == null) return null;
  return modelInputMirrored ? modelIsRight : !modelIsRight;
}

/// Passes packed Android NV21 or iOS BGRA directly to the detector isolate.
///
/// Multi-plane YUV420 is returned as `null`; the detector's compatibility
/// conversion path handles that format.
CameraFrame? directCameraFrame({
  required CameraImage image,
  required CameraFrameRotation? rotation,
  required bool isAndroid,
  required bool isIOS,
}) {
  if (image.planes.length != 1) return null;
  final plane = image.planes.first;

  if (isAndroid && image.format.group == ImageFormatGroup.nv21) {
    final minimumLength = image.width * image.height * 3 ~/ 2;
    if (image.width <= 0 ||
        image.height <= 0 ||
        plane.bytes.length < minimumLength) {
      return null;
    }
    // CameraX's NV21 converter can allocate from the padded Y-plane capacity
    // and leave unused bytes after the tightly packed frame. OpenCV expects
    // exactly width * height * 1.5 bytes, so expose only the valid prefix.
    final bytes = plane.bytes.length == minimumLength
        ? plane.bytes
        : Uint8List.sublistView(plane.bytes, 0, minimumLength);
    return CameraFrame(
      bytes: bytes,
      width: image.width,
      height: image.height,
      strideCols: image.width,
      conversion: CameraFrameConversion.yuv2bgrNv21,
      rotation: rotation,
    );
  }

  if (isIOS && image.format.group == ImageFormatGroup.bgra8888) {
    final strideCols = plane.bytesPerRow ~/ 4;
    if (image.width <= 0 ||
        image.height <= 0 ||
        strideCols < image.width ||
        plane.bytes.length < plane.bytesPerRow * image.height) {
      return null;
    }
    return CameraFrame(
      bytes: plane.bytes,
      width: image.width,
      height: image.height,
      strideCols: strideCols,
      conversion: CameraFrameConversion.bgra2bgr,
      rotation: rotation,
    );
  }

  return null;
}
