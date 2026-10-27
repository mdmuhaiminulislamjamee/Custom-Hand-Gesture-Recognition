package com.example.gesturehandover

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs

data class GestureParityReport(val featureCases: Int, val onnxCases: Int, val maximumError: Float, val angleCases: Int = 0)

/** Run on a debug build/device before enabling any gesture action. */
object GestureParity {
    fun run(context: Context): GestureParityReport {
        val json = JSONObject(context.assets.open("parity_vectors.json").bufferedReader().use { it.readText() })
        val metadata = JSONObject(context.assets.open("gesture_mlp_onnx_metadata.json").bufferedReader().use { it.readText() })
        require(json.getString("model_sha256") == metadata.getString("onnx_sha256")) { "Wrong model fixtures" }
        val featureTolerance = json.getDouble("feature_tolerance").toFloat()
        val probabilityTolerance = json.getDouble("probability_tolerance").toFloat()
        var maximum = 0f
        var angleCases = 0
        fun numbers(values: JSONArray): FloatArray = FloatArray(values.length()) { values.getDouble(it).toFloat() }
        fun compare(actual: FloatArray, expected: FloatArray, tolerance: Float, name: String) {
            require(actual.size == expected.size) { "$name output length mismatch" }
            for (i in actual.indices) {
                val error = abs(actual[i] - expected[i])
                maximum = maxOf(maximum, error)
                check(error <= tolerance) { "$name mismatch at index $i: $error" }
            }
        }

        val geometry = json.getJSONArray("geometry_cases")
        for (i in 0 until geometry.length()) {
            val item = geometry.getJSONObject(i)
            val landmarks = item.getJSONArray("landmarks")
            val points = (0 until landmarks.length()).map { index ->
                val pair = landmarks.getJSONArray(index)
                GesturePoint(pair.getDouble(0).toFloat(), pair.getDouble(1).toFloat())
            }
            compare(GestureFeatureExtractor.extract(points), numbers(item.getJSONArray("features")),
                featureTolerance, "feature case ${item.getInt("id")}")
            if (!item.isNull("image_shape")) {
                val shape = item.getJSONArray("image_shape")
                val policy = item.getJSONObject("expected").getJSONObject("angle_policy")
                val xy = GestureRuntimePolicy.openHandXY(points, shape.getInt(1), shape.getInt(0))
                compare(floatArrayOf(xy), floatArrayOf(policy.getDouble("xy_degrees").toFloat()),
                    .0002f, "image XY angle ${item.getInt("id")}")
                check(GestureRuntimePolicy.allowsAngle(xy, "open_palm") == policy.getBoolean("open_palm"))
                check(GestureRuntimePolicy.allowsAngle(xy, "stop_inverted") == policy.getBoolean("stop_inverted"))
                angleCases++
            }
        }

        val modelCases = json.getJSONArray("model_cases")
        GestureOnnxClassifier(context).use { classifier ->
            for (i in 0 until modelCases.length()) {
                val item = modelCases.getJSONObject(i)
                val output = classifier.predict(numbers(item.getJSONArray("features")))
                compare(output.probabilities, numbers(item.getJSONArray("probabilities")),
                    probabilityTolerance, "ONNX case $i")
                compare(floatArrayOf(output.knownGestureMass),
                    floatArrayOf(item.getDouble("known_gesture_mass").toFloat()),
                    probabilityTolerance, "known-mass case $i")
            }
        }
        return GestureParityReport(geometry.length(), modelCases.length(), maximum, angleCases)
    }
}
