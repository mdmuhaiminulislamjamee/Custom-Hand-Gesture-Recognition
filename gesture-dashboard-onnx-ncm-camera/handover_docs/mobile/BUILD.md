# Build a fresh iOS and Android delivery pair

The current pair is `v2026.10.27_ios_ncm_final_01` and `v2026.10.27_android_ncm_01`. Existing directories, ZIPs and checksum companions are immutable. Choose an unused date/sequence for another release.

From the repository root, using the prepared Python environment with `requirements-handover-checks.txt` installed:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe scripts\export_mobile_parity.py .runtime\mobile_export\parity_vectors.json
$env:NODE_OPTIONS='--max-old-space-size=384 --max-semi-space-size=2'
$env:RAYON_NUM_THREADS='1'
npm.cmd run build *> .runtime\mobile_export\frontend_build.log
# Confirm the build succeeded before continuing.
.\.venv\Scripts\python.exe scripts\create_mobile_handovers.py --date YYYY.MM.DD --sequence NN --fixtures .runtime\mobile_export\parity_vectors.json --frontend-log .runtime\mobile_export\frontend_build.log
```

Replace `YYYY.MM.DD` and `NN` with the selected date and two-digit sequence. The exporter binds vectors to the current model and runtime source hashes. Regenerate after geometry, timing, recovery, configuration or model changes.

The builder freezes an explicit source/model/qualification allowlist, runs preflight, replays the vectors against that isolated desktop copy and runs the complete backend suite in separate processes. Generated runtime/cache/private files are excluded from the desktop ZIP by the pre-test inventory. It then creates both platform packages from the same desktop ZIP, validates hashes/contracts/ONNX vectors and writes outer ZIP SHA-256 companions. It does not alter a previous release or retrain the model.

Each package includes platform diagnostics, current resources, the full Python behavior reference, legacy native decisions marked as reference only, guides, manifests, parity vectors and host evidence. Native compilation and physical-device acceptance cannot be inferred from the Windows checks; retain the pending flags until the app teams qualify those stages.
