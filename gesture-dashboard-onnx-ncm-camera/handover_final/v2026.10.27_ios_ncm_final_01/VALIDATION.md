# Validation and app-team acceptance

Machine-readable host evidence is in `validation/`. The release manifest records exact source/model hashes and fixture counts. The desktop snapshot was verified separately from the sender's working tree. Host checks do not establish native compilation or physical camera accuracy.

## Checks required before enabling app actions

1. Run `python verify_package.py --onnx`; retain output and the outer ZIP's companion SHA-256. Verify the original folder before adding build outputs or editing delivered files.
2. Build/link native dependencies on the target platform, record their exact locked versions, and run the supplied feature/angle/ONNX helpers.
3. Complete the native decision/frame port in `RUNTIME_PORTING.md`. Compare adjusted probabilities, all pose/recovery flags, temporal decisions and the limit-crossing sequences against the supplied vectors. Record every mismatch and its resolution.
4. Establish the physical USB-NCM route, valid rising JPEG count, CRC/sequence diagnostics, heartbeat survival and unplug/replug recovery. No stale frame or action may replay on reconnect.
5. Confirm analysis pixels remain unmirrored: positive index dx → Left and negative dx → Right. Test both hands, palm/back/edge views, thumb variants, image aspect changes and near head-on views.
6. Check Open Palm at −160°, −90°, −20° and just outside −160.01°/−19.99°. Check Stop inverted at −145°, −90°, −35° and just outside −145.01°/−34.99°. Use the wrist-to-knuckle image-pixel angle, not HUD world-plane angles. Outside means no gesture even after an accepted hold; reentry requires a fresh hold.
7. Test Peace with index/middle extended, ring/pinky folded, touching or separated tips, rotated hands and the strict low-mass reported view. A raised extra finger, lone middle finger, contradictory world joints or mass below 0.005 must not use the strict low-mass exception. Do not lower global thresholds.
8. Check Like versus Fist/Thumbs Down, index-only OK pinch, Rock, Up/Down, Fist world-depth and upper/lower orientation rules, all two-hand ownership cases and retired Dorsal rejection. The face/body vetoes remain disabled; face overlap or clothing alone does not reject a valid hand gesture.
9. Verify one app action per deliberate hold, release/cooldown rearming, short tracking loss without duplicate toggles, ownership changes, permissions, foreground/background, stream restart and invalid timestamps.
10. Measure median/p95 end-to-end latency, delivered FPS, queue age, memory/thermal behavior, live per-class precision/recall and false actions per minute. Record device, OS, app build, release/model hash, dependency locks, accessory/firmware/cable, crop/orientation, lighting and date.

Native builds, full native decision parity, physical USB-NCM compatibility and live-device accuracy remain unqualified until the app team records them. Supplied overlay/synthetic vectors are regression cases, not independent new camera viewpoints.
