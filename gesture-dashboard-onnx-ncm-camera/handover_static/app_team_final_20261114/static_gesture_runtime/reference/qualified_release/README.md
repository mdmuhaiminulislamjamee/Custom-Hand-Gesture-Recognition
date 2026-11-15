# Evidence from the qualified executable release

HOST_CHECKS.json and RELEASE_NOTES.md are immutable evidence from the 14 November 2026 ownership/preview delivery. They mention the historical tester app/APK and original repository paths because that build tested the shared runtime. The tester app/APK is deliberately not included here. No tester installation is required for integration.

All flutter/ files and the two optional phone helpers in this package are byte-identical to that checked release. The main integration guide describes the simplified folder layout. The dependency inventory and phone-camera host lockfile are references for merging into an existing app; the consuming app must review its own resolved dependency lock.
