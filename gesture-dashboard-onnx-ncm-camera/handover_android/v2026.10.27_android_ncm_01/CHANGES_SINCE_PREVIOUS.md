# Changes since the 8 October handovers

- Production classifier is now `StopInvertedDirectionalFeedbackMLP`, with 13 fixed tensor outputs and 12 enabled commands. Index 6 Dorsal is disabled rejection evidence. Index 12 Stop inverted maps to `Return to main position`.
- The true feature input is 76 floats. Earlier iOS README references to 63 features were documentation errors. This delivery's metadata, code and checks use `[1,76]`.
- The directional feedback update includes 30 Left and 48 Right corrections in training. Left/Right sectors and open-hand angles use source-image pixels to avoid normalized-coordinate aspect distortion. Keep the calibrated horizontal semantic swap and unmirrored analysis pixels.
- A strongly verified Peace pair can recover the reported near-zero classifier mass with the strict pair flag, minimum mass 0.005 and a 0.50 s hold. Ordinary/ambiguous shapes do not receive that exception.
- Open Palm now accepts image XY −160° to −20°, inclusive. Stop inverted accepts −145° to −35°, inclusive. Crossing a limit immediately produces no gesture and resets the hold. Returning requires a fresh hold.
- The current detector/selection path uses two hands and current recovery/ownership rules. Face and body vetoes are disabled. Do not copy their former rejection policy from an older handover.
- Both mobile deliveries contain the same freshly built desktop ZIP, model/configuration, source hashes and current parity vectors. Android no longer copies the desktop ZIP from the October 8 iOS release.
- Native model wrappers now accept the 13-output model. Current diagnostic adapters preserve the dimensions and candidate metadata required by the port. Older Swift decision implementations are explicitly separated into `reference/legacy_swift`; full current native decision parity and physical-device acceptance remain app-team work.

The model was not retrained while preparing this delivery. The packages contain the current working source snapshot, including the approved angle/Peace changes; the source base commit and exact file hashes are recorded separately in each release manifest. Previous handover directories and ZIPs are retained unchanged.
