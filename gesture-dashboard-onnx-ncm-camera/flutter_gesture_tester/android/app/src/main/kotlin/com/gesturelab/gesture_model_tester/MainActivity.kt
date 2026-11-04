package com.gesturelab.gesture_model_tester

import io.flutter.embedding.engine.FlutterEngine
import io.flutter.embedding.android.FlutterActivity

/**
 * Inference and JLIP/NCM transport run in the shared Flutter package. This
 * activity owns the socket-specific USB-Ethernet bridge for the Dart client.
 */
class MainActivity : FlutterActivity() {
    private var ncmBridge: NcmNetworkBridge? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        ncmBridge = NcmNetworkBridge(this, flutterEngine.dartExecutor.binaryMessenger)
    }

    override fun cleanUpFlutterEngine(flutterEngine: FlutterEngine) {
        ncmBridge?.close()
        ncmBridge = null
        super.cleanUpFlutterEngine(flutterEngine)
    }
}
