# iPhone and NCM acceptance

Run these checks in a release build on each supported iPhone. Record aggregate
counts/timings locally; no Boss app images or recordings need to be shared.

1. Record iPhone model, iOS version, Flutter/native dependency versions, board
   firmware, source resolution, selected detector backend and model SHA-256.
   Confirm the qualified FP32 model is loaded once and interpreter creation is
   absent from the per-frame loop.
2. Grant local-network permission. Connect the NCM board and confirm TCP
   `192.168.50.2:5000`, valid JLIP frames, and heartbeat replies. Test a cold
   launch, reconnect, background/foreground and USB disconnect during inference.
   A disconnect must clear the preview and prevent stale command dispatch.
3. With each physical hand, check all twelve gestures in the runtime README.
   Check both hands simultaneously, one hand leaving/reentering, crossing and
   near-overlap. Boxes/points must match the same upright source frame. A rejected
   pose can still have a visible raw hand overlay.
   Confirm a palm facing the camera is Open Palm and the back of the upright
   open hand is Stop Inverted. Check Like with an upward thumb and Thumb Down
   with a downward thumb, including the reported sideways-arm view. Repeat on
   phone/front/rear and NCM sources that the consuming app supports.
4. Verify event behavior: a deliberate pose confirms, holding does not repeatedly
   trigger the same action, and a confirmed neutral/no-gesture releases it for
   another trigger. Exercise transitions and independently held left/right
   poses. Expect at most one command from one source frame.
5. Check rejection with empty scene, partial hands and poses outside the source
   geometry restrictions. In particular, validate Open Palm/Stop inverted angle
   bands and Fist world-depth/surface rules. Do not loosen gates merely to hide a
   coordinate or handedness mismatch.
6. Measure a sustained session with one hand, two hands and an empty scene.
   Record p50/p95 timings after warm-up, replacement counts, invalid frames and
   false/duplicate commands. Repeat with preview disabled to isolate rendering
   overhead. Observe heat and performance over at least ten minutes.
7. Compare the existing dynamic pipeline and static integration on the same
   phone, source resolution, route and detector backend. Separate hand-detection
   time from static classification/geometry/decision time. Fix any duplicated
   detector or queued-frame issue before changing the model.

Suggested result record:

| Measurement | Empty scene | One hand | Two hands |
|---|---|---|---|
| Detector backend | | | |
| Processed frames / duration | | | |
| Achieved inference FPS (admission cap 10) | | | |
| Decode + detector p50/p95 ms | | | |
| Classification + decision p50/p95 ms | | | |
| Receiver to result p50/p95 ms | | | |
| Replaced / invalid frames | | | |
| Attempted / correct / missed commands | | | |
| False / duplicate commands | | | |
| Sustained heat/performance notes | | | |

Receiver timing begins after a complete JPEG reaches the app. The board and
phone clocks are not synchronized; this telemetry does not measure exposure,
USB transit or true capture-to-action latency. Define product acceptance limits
with the app owner using the supported devices.

If the network is unreachable, check the OS-created USB Ethernet/NCM route,
board power/firmware, data cable and local-network permission. The app cannot
create an unsupported kernel NCM interface. Connection refused points to the
board service/port; valid TCP with no preview points to protocol/framing or
detector errors. Use the observable counters/backend to distinguish those cases.
