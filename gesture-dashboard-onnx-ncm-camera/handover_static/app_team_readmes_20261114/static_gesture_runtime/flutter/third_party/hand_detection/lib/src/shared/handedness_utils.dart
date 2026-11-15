import 'hand_types.dart';

/// Decoded MediaPipe binary handedness output.
typedef DecodedModelHandedness = ({
  Handedness handedness,
  double score,
});

/// Decodes the hand landmark model's scalar as the probability of `Left`.
///
/// The model's associated `handedness.txt` orders the binary classes as
/// `Left`, then `Right`. MediaPipe assigns the scalar to class 0 and
/// `1 - scalar` to class 1.
DecodedModelHandedness decodeMediaPipeHandedness(double leftProbability) {
  final pLeft = leftProbability.clamp(0.0, 1.0);
  if (pLeft >= 0.5) {
    return (handedness: Handedness.left, score: pLeft);
  }
  return (handedness: Handedness.right, score: 1 - pLeft);
}
