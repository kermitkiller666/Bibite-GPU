# 0.6.8 GUI and stability verification — 1 October 2026

This is a repair pass within 0.6.8, not a new numbered release. The supplied
reference application is The Bibites 0.6.3.1 Windows x64, not 0.6.4.

## Release checklist

- [x] Preserve the existing game configuration and player saves.
- [x] Back up the previous managed/native DLLs before integration.
- [x] Split work between menu integration, selected-Bibite panels, and engine stability.
- [x] Compare the original application and the modified application's actual windows.
- [x] Verify normal world save/load and custom tag round-trip in the graphical application.
- [x] Verify live Dynamic Settings food-density changes reach the GPU world.
- [x] Complete final managed/native builds and regression tests after all edits.
- [x] Pass shared-renderer startup, sustained rendering, UI interaction, and reload checks.
- [x] Pass a higher-warp graphical smoke test on the RTX 4070 Ti.
- [x] Remove the temporary GUI audit probe from the deliverable.
- [x] Verify packaged DLL hashes and both executables in the final ZIP.
- [x] Leave the game closed and retain recoverable test/rollback artifacts.

There is no CI service, server deployment, database migration, or remote upload
in this local repair. The RTX 5060 Ti is reserved for the user's other project.

## Observed tests

Evidence is under `work/gui-audit-probe/` in the containing workspace. Test
worlds use unique `Codex_GUI_*_20261001` names; the player's `quick` save is not
overwritten.

### Menu and save checks

- Fallback-rendered 1x run: 60 FPS, maximum frame 17.5 ms, no measured frame over
  50 ms in the initial 20-second sample (integrated 5).
- The original Stats / Genes / Biology / Brain / Expanded Brain panel roots,
  toolbar icons, and readable original-style borders/buttons are used.
- The GPU settings tab appears in the original settings navigation. Its opaque,
  scrollable dialog blocks world clicks, has a persistent close button, and
  unwinds correctly with Escape.
- The full-screen chart no longer has the inspector, selection reticle, or
  diagnostics drawn over it.
- Normal Save Game and Load Game preserve the native population and show GPU
  counts in the original preview. Integrated 2 loaded a 613-Bibite checkpoint.
- Integrated 5 tagged Bibite #128 `Audit Aurora 128` while paused, then saved
  a 536-Bibite, 44-second world. A fresh integrated-7 process loaded it through
  the normal UI, and the selected Bibite displayed the same friendly tag.
- Camera Follow centres the selected Bibite and changes to Stop following.
- Changing original biomass density 7.5 to 1.4 changed the GPU pellet target
  8,192 to 1,527, with 1,436 active pellets at observation; restoring 7.5 restored
  target 8,192, with 8,002 active at observation (integrated 2).
- A paused Lay egg failure exposed a missing-error-feedback bug: the selected
  status filter did not accept a colon after the slot ID. After correction,
  integrated 8 displayed the persistent, accurate insufficient-energy error
  for Bibite #128 without modifying the world or freezing the UI.
- Rectangle selection remains awaiting conclusive UI verification; the
  automation drag may have been shorter than the stock 0.15-second threshold.

### Shared-renderer failures reproduced during development

These are rejected candidates, not evidence that the release is stable.

- The original synchronous shared-buffer path hung and reported CUDA error 999.
- An initial async double-buffer candidate hung at startup.
- A pointer-precreation candidate crashed in NVIDIA `nvwgf2umx` after a few
  buffer reuses, without any desktop-automation interaction.
- Enabling D3D11 immediate-context multithread protection allowed the initial
  20-second sample to pass (60 FPS, max 17.1 ms, 489 render updates), but the
  application subsequently froze at simulation time 23.575 seconds. This
  change alone is insufficient.
- The replacement moves interoperability operations onto Unity's render thread
  with explicit exclusive ownership of the native world. Integrated 8 passed
  a 30-second 1x sample at 60 FPS (frame p95/max 16.7/17.2 ms, no measured frame
  over 50 ms, 634 direct-render updates), then minutes of interaction including
  the Brain panel, a native checkpoint reload (190 ms native handoff), pause,
  tag recovery and saving while shared rendering was active.
- Integrated 8 then exposed a shutdown-order bug: Unity's render loop could
  stop before the worker's unregister callback was delivered. Its exact test
  process was terminated after saving. The replacement defers Application quit
  while normal Unity frames pump unregister and finish pending saves. Both
  integrated 9 and final integrated 10 then exited without forced termination.

### Final live checks

Configuration was preserved: RTX 4070 Ti (device 0), 512 starters, 131,072 cap,
8,192 target pellets, 2,048 detailed textures, 60 FPS target, Extreme profile.
These are smoke tests, not matched headless throughput benchmarks.

| Run | Observation |
| --- | --- |
| Integrated 9, 1x | 30-second sample: 60.0 FPS, frame p95/max 16.7/17.6 ms, zero frames over 50 ms, 523 living at report. Stats, Brain and fullscreen chart interaction passed. |
| Integrated 10, requested 1000x | 30-second sample: 59.8 FPS, frame p95/max 16.7/31.7 ms, zero frames over 50 ms. 79,796 living at report; achieved speed then was 142.2x, not the requested 1000x. |
| Growing world | The same run reached the 131,072 cap. Selection remained responsive; the paused species panel showed a clearly labeled sampled lineage estimate. |
| Reload | Normal Load Game preview showed the shared-renderer save's 569 Bibites, 8,133 pellets and 82.1 simulated seconds. Native handoff took 193 ms; Bibite #128 recovered its `Audit Aurora 128` tag. |
| Exit | In both final runs the unregister event was issued/reclaimed, then the GPU-shutdown-complete message preceded Unity physics/input shutdown. Both exact game processes exited. |

All final managed builds have zero warnings/errors. The core regression suite
passes, including warp, half conversion, save rollback, session mode, metadata,
action routing, snapshot cadence, process-wide interop circuit breaker and
deferred quit. Both native CTest tests pass (brain parity and world invariants).
The GUI audit probe is removed from the package. The three temporary test-save
pairs were moved, with hash checks, to `work/gui-audit-probe/preserved-test-saves/`.
The existing quick-save pair and game configuration retained their hashes.

The release archive is checked byte-for-byte against 301 package files,
including the normally hidden `The Bibites.exe`, its visible launcher, the
three tested DLLs and preserved configuration. Test probes and diagnostic logs
are excluded. The previous ZIP remains in the dated rollback directory.

Final tested binary SHA256 values:

- Managed: `98DF2B4B5D05F19B184EE9E44BE85386CD095291F819348F617DE85E8E8F9EF9`
- Core: `BD9628C97089E4EAB943E7689193145E66167F1EF687857C87B88F06C59F154A`
- Native: `BF8E0A816CB407D2824C58C1379565BB1380211848D215D2200887B37ED14F47`

Still unverified: rectangle-drag activation with a long physical drag, every
original scenario/editor path, destructive save-menu actions on player data,
same-scene native-to-stock QuickLoad in the graphical app, every resolution,
and overnight stability. These limits must not be presented as passed tests.

### Test-runner issue

An early console test run used Unity's repacked Newtonsoft.Json assembly,
which desktop .NET Framework rejected. The test project now uses the signed
cached dependency and reports failures to stderr with a nonzero exit code.
All managed regression tests and both native CTest tests pass after correction.
The old Windows hard-error dialog is not a game error; the user was asked to
dismiss it because its system-owned window cannot safely be automated here.

## Rollback and limitations

Previous binaries/configuration are in
`build-backup-gui-20261001-013208/`. Do not replace player saves with test saves.
Retain the old release archive before replacing it. Any repeatable hang,
driver crash, save corruption, or menu input lock rejects the release candidate.

The native model does not contain all original organ, combat, family-history,
egg-body, and editable `.bb8` state. Restored panels must distinguish available
native data from unsupported original data. See `gui-parity-0.6.8.md`.

Native checkpoint contents are validated during load. Only header preflight
occurs before retiring an existing world; a late corrupt-payload failure does
not guarantee preservation of that unsaved world. Save-wrapper/native-file
replacement rolls back ordinary I/O failures, but the two-file pair is not a
single power-loss-atomic transaction.
