import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import '../lib/src/feature_extractor.dart';
import '../lib/src/geometry_policy.dart';

Map<String, Object?> checkGeometryParity(String fixturePath) {
  final fixture =
      jsonDecode(File(fixturePath).readAsStringSync()) as Map<String, dynamic>;
  final cases = fixture['geometry_cases'] as List<dynamic>;
  var maxFeatureError = 0.0, maxProbabilityError = 0.0;
  var featurePasses = 0, rowPasses = 0, decisionPasses = 0, supportPasses = 0;
  var massCases = 0, massPasses = 0;
  final failures = <Map<String, Object?>>[];
  for (final value in cases) {
    final row = value as Map<String, dynamic>;
    List<List<double>> points(Object? data) => (data as List<dynamic>)
        .map((p) =>
            (p as List<dynamic>).map((v) => (v as num).toDouble()).toList())
        .toList();
    List<double> values(Object? data) =>
        (data as List<dynamic>).map((v) => (v as num).toDouble()).toList();
    final landmarks = points(row['landmarks']);
    final features = StaticFeatureExtractor.extract(landmarks);
    final expectedFeatures = values(row['features']);
    var featureError = 0.0;
    for (var i = 0; i < features.length; i++) {
      featureError =
          math.max(featureError, (features[i] - expectedFeatures[i]).abs());
    }
    maxFeatureError = math.max(maxFeatureError, featureError);
    if (featureError <= .0001) featurePasses++;
    // Python geometry fixtures already name the physical hand. Detector raw
    // selfie categories are converted by StaticDetectorAdapter, outside these
    // geometry-only fixtures.
    final physicalHand = (row['handedness'] as String?)?.toLowerCase() ?? '';
    final shape = row['image_shape'] as List<dynamic>?;
    final result = StaticGeometryPolicy.evaluate(
      probabilities: values(row['probabilities']),
      knownMass: 1,
      normalizedLandmarks: landmarks,
      physicalHand: physicalHand,
      handednessConfidence:
          (row['handedness_confidence'] as num?)?.toDouble() ?? 0,
      width: shape == null ? 1 : (shape[1] as num).toDouble(),
      height: shape == null ? 1 : (shape[0] as num).toDouble(),
      worldLandmarks: row['world_landmarks'] == null
          ? null
          : points(row['world_landmarks']),
      requireFistDepth: row['require_fist_depth'] as bool? ?? false,
    );
    final expected = row['expected'] as Map<String, dynamic>;
    final expectedProbs = values(expected['probabilities']);
    var probabilityError = 0.0;
    if (result.probabilities.length != expectedProbs.length) {
      probabilityError = 1;
    } else {
      for (var i = 0; i < expectedProbs.length; i++) {
        probabilityError = math.max(probabilityError,
            (result.probabilities[i] - expectedProbs[i]).abs());
      }
    }
    maxProbabilityError = math.max(maxProbabilityError, probabilityError);
    if (probabilityError <= .0001) rowPasses++;
    final expectedLabel =
        expected['valid'] == true ? expected['gesture'] : 'no_gesture';
    final decisionPass =
        result.label == expectedLabel && result.accepted == expected['valid'];
    if (decisionPass) decisionPasses++;
    final supportPass = result.diagnostics['geometry_supported'] ==
        expected['geometry_supported'];
    if (supportPass) supportPasses++;
    for (final massValue in expected['temporal'] as List<dynamic>) {
      final temporal = massValue as Map<String, dynamic>;
      final mass = (temporal['mass'] as num).toDouble();
      final qualified = StaticGeometryPolicy.evaluate(
        probabilities: values(row['probabilities']),
        knownMass: mass,
        normalizedLandmarks: landmarks,
        physicalHand: physicalHand,
        handednessConfidence:
            (row['handedness_confidence'] as num?)?.toDouble() ?? 0,
        width: shape == null ? 1 : (shape[1] as num).toDouble(),
        height: shape == null ? 1 : (shape[0] as num).toDouble(),
        worldLandmarks: row['world_landmarks'] == null
            ? null
            : points(row['world_landmarks']),
        requireFistDepth: row['require_fist_depth'] as bool? ?? false,
      );
      final shouldQualify =
          (temporal['execute'] as List<dynamic>).contains(true);
      massCases++;
      if (qualified.accepted == shouldQualify) {
        massPasses++;
      } else {
        failures.add(<String, Object?>{
          'id': row['id'],
          'source': row['source'],
          'known_mass': mass,
          'expected_qualification': shouldQualify,
          'actual_qualification': qualified.accepted,
          'reason': qualified.reason,
          'actual_diagnostics': qualified.diagnostics,
          'expected_pose': expected['pose_validation']
        });
      }
    }
    if (featureError > .0001 ||
        probabilityError > .0001 ||
        !decisionPass ||
        !supportPass) {
      failures.add(<String, Object?>{
        'id': row['id'],
        'source': row['source'],
        'feature_error': featureError,
        'probability_error': probabilityError,
        'expected_label': expectedLabel,
        'actual_label': result.label,
        'expected_valid': expected['valid'],
        'actual_valid': result.accepted,
        'expected_support': expected['geometry_supported'],
        'actual_support': result.diagnostics['geometry_supported'],
        'reason': result.reason,
        'actual_diagnostics': result.diagnostics,
        'expected_pose': expected['pose_validation']
      });
    }
  }
  return <String, Object?>{
    'cases': cases.length,
    'feature_passes': featurePasses,
    'probability_row_passes': rowPasses,
    'decision_passes': decisionPasses,
    'geometry_support_passes': supportPasses,
    'known_mass_cases': massCases,
    'known_mass_qualification_passes': massPasses,
    'maximum_feature_absolute_error': maxFeatureError,
    'maximum_probability_absolute_error': maxProbabilityError,
    'passed': featurePasses == cases.length &&
        rowPasses == cases.length &&
        decisionPasses == cases.length &&
        supportPasses == cases.length &&
        massPasses == massCases,
    'failures': failures
  };
}

void main(List<String> args) {
  final report = checkGeometryParity(
      args.isEmpty ? '../validation/python_reference.json' : args[0]);
  if (args.length > 1)
    File(args[1]).writeAsStringSync(
        '${const JsonEncoder.withIndent('  ').convert(report)}\n');
  final summary = Map<String, Object?>.of(report)..remove('failures');
  stdout.writeln(jsonEncode(summary));
  for (final item in (report['failures'] as List).take(20)) {
    final failure = Map<String, Object?>.of(item as Map<String, Object?>)
      ..remove('actual_diagnostics')
      ..remove('expected_pose');
    stdout.writeln(jsonEncode(failure));
  }
  if (report['passed'] != true) exitCode = 1;
}
