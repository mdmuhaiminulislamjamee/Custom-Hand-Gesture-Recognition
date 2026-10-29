package com.gesturelab.gesture_model_tester

import io.flutter.embedding.android.FlutterActivity

/**
 * Inference and JLIP/NCM transport now run in the shared Flutter package.
 * Keeping this activity intentionally empty avoids the old exact-/30 route
 * pre-check and lets Android's route table select the local USB Ethernet link.
 */
class MainActivity : FlutterActivity()
