# 03 - Show the preview and selected hand

## What this part does

`StaticNcmPreview` shows the NCM JPEG with the landmarks and detector overlay
that belong to that same image. It keeps the previous matching image/overlay
visible while the next JPEG decodes, avoiding the preview flicker seen when
the overlay is cleared early.

The frame result also identifies `selectedHand`: the hand that currently owns
the public gesture. A second detected hand is not automatically the owner.

## What the app team integrates

1. Save the `NcmStaticPreviewFrame` received in `onPreview` and rebuild the
   relevant screen area. Display `StaticNcmPreview(frame: savedFrame)` in a
   bounded-width layout.
2. For gesture text, use `savedFrame.result.selectedHand`. Find the matching
   entry in `savedFrame.result.hands` and display that hand's gesture. If
   `selectedHand` is null, show no selected gesture for that frame.
3. When `switchPending` is true, optionally show "Confirming hand switch".
   Commands are paused during the handoff.
4. Clear the saved frame/result when the NCM session disconnects, stops or
   reports an error. Keep the JPEG and overlay together; do not blank the
   overlay independently during each decode.

Do not choose the first detected hand or sort hands by confidence for control.
The runtime's ownership decision is in `selectedHand`. For UI code, see the
[example screen](../../flutter/example/lib/main.dart) and the
[full guide](../../INTEGRATION_GUIDE.md).
