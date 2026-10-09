# October 9 pre-publication code check

Scope: a focused review and fresh regression run for the 0.6.10 optimized
preview, followed by packaging the matching source and portable EXE. This is
not an exhaustive audit, original-game parity certification or Unity playtest.
No existing game was closed, no older EXE was overwritten, and the 5060 Ti was
not used. Native tests ran on CUDA device 0, NVIDIA GeForce RTX 4070 Ti.

## Reviewed paths

- Food cap projection, zero/low target handling, segmented overflow sensing,
  and the uniform dense-food cache flag/barrier gating.
- Fused/graph dispatch, cadence/cache invalidation, disabled neural-link
  semantics and matching managed/native step metrics.
- Owned render snapshots, graphics callback completion, worker retirement,
  asynchronous checkpoint capture/write and two-file save transactions.
- Portable payload integrity, safe extraction paths, companion DLL consistency,
  third-party notices and exclusion of runtime/personal files.

## Fix found during this review

`CompleteCapturedRenderRequest(..., true)` and `ExecuteRenderThreadRequest`
could poll indefinitely if a graphics callback never completed. Both blocking
polls now have a 15-second deadline. Completed/cancelled callbacks retain normal
retirement; unknown or running callbacks are **not forcibly freed**. The failed
handoff trips the process interop circuit breaker, retains the native world and
render resources for the process lifetime, and reports the failure. Already
abandoned handles skip further graphics teardown. The captured request's managed
state is completed on failure so Unity does not keep trying the same handoff.

Pure regression tests check pending/running/unknown states, the deadline
boundary, terminal states and invalid elapsed time. There is no test claim that
a genuinely wedged graphics driver was recovered. A CUDA/native API call that
itself never returns can still block before the managed deadline is checked.
Quarantine can retain VRAM until process exit; restart after an interop fault.

## Fresh verification results

| Check | Result |
| --- | --- |
| Core regression executable | Passed, including graphics deadline, kernel policy, warp, food caps, diagram/visibility, save transaction and menu/presentation safety tests |
| Managed Release build | Passed; 0 warnings, 0 errors |
| CUDA native build | Passed; source up-to-date, DLL unchanged from the final optimized engine |
| CUDA brain/vision/pheromone parity | Passed on RTX 4070 Ti |
| Full native world suite | Passed: dense-food reference parity, crowded contacts, graph cache, imported pool, checkpoint validation/reuse, action routing and biology |
| Captured D3D11 render overlap | 20 concurrent updates and 672 ticks passed without a window |
| Crowded save/reload/resume | Eight cycles, 1536 ticks, about 1.25 wall seconds in this run |
| Portable integrity/privacy verification | Passed: 243 entries, 189,653,080 uncompressed bytes, matching three canonical plugin DLLs, included notices/licenses, no runtime/personal files |

Evidence is retained locally under `work/verification`:
`release-core-tests-20261009-final.log`,
`release-managed-build-20261009-final.log`,
`release-build-native-20261009.log`, `release-brain-tests-20261009.log`,
`0.6.10-optimized-native/release-native-tests-20261009.log`, and
`release-portable-verification-20261009-final.log`.
The logs are not bundled into the public EXE or committed as personal runtime data.

Reproduce from the repository root with the required stock/BepInEx assemblies:

```powershell
dotnet run --project 0.6.10/tests/BibitesGpuFork.CoreTests.csproj -c Release
dotnet build 0.6.10/managed/BibitesGpuFork.csproj -c Release -p:GameManagedDir='<The Bibites_Data/Managed>' -p:BepInExCoreDir='<BepInEx/core>'
cmake -S 0.6.10 -B build/0.6.10 -G 'Visual Studio 17 2022' -A x64
cmake --build build/0.6.10 --config Release --target BibitesGpuNativeTests BibitesGpuWorldTests
ctest --test-dir build/0.6.10 -C Release --output-on-failure
& 0.6.10/portable/Verify-Portable.ps1 -Path '0.6.10/Bibites GPU Fork 0.6.10 - PREVIEW.exe'
```

## Remaining limits

The five-simulated-minute large-map test retained **152 of 512 founders**, with
8 births, 368 starvation deaths and 805 eaten pellets. The invariant suite passes,
but these results do not establish balanced ecology or healthy long-run evolution.
Hidden-layer widths are fixed at 12+12; connection masks/weights mutate, not
hidden-neuron counts. Biology, egg production and grabbing remain approximations.

Final Unity GUI responsiveness, save-wrapper operation, menu exit and textured
FPS remain unverified for this exact EXE. Original sprite parts remain CPU
GameObjects; full GPU-instanced original textures are not implemented. AMD
acceleration remains deferred. Format-12 checkpoints are unreadable by older
binaries. Uniform dense-food performance can regress; earlier headless benchmark
numbers are not guaranteed achieved graphical warp.

Publish as **0.6.10 PREVIEW**, with these limits visible, not as a stable release
or a claim that every issue ticket is resolved. Keep older builds/saves available
for rollback and never overwrite an important save merely to test migration.
