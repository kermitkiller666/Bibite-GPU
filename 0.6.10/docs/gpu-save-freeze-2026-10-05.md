# GPU save freeze investigation — 2026-10-05

The reported running preview logged an autosave request, then repeated
`savePending=True` during application shutdown. Its GPU worker had not
completed the queued checkpoint. The 4070 Ti was busy, but the old build did
not record the precise worker phase. This does not establish that checkpoint
file writing itself was the bottleneck.

## Changes

- Fixed a race in the fused cooperative kernel's sparse contact/sensing grid
  clear. All blocks must read previous occupied-list lengths before thread
  zero resets their counters. Without that barrier, late blocks could skip
  clearing stale overflow heads; reinsertion could create cyclic lists.
- Bounded overflow-chain traversal by valid cell length and Bibite capacity,
  rejecting invalid indexes and self-links. The large-body count is bounded.
- Replaced repeated coordinate subtraction with bounded modulo wrapping.
  Extreme finite FP32 values could previously loop forever because subtracting
  one world width did not change the value. Non-finite values are left for the
  existing invalid-state rejection.
- Added worker phase/elapsed-time diagnostics to pending-save messages and
  shutdown logs. Saving remains queued on the simulation worker, not a
  blocking wait on Unity's main thread.
- Added `--save-progress-only`: a crowded overflow cluster coexists with
  hundreds of sparse cells, advances 1,536 ticks, and performs eight native
  checkpoint save/reload/resume cycles. It also tests extreme finite positions.
- Added an opt-in full-wrapper save probe to the separate automated test
  session. Normal sessions do not save or quit through this probe.

## Verification and limits

- Managed/core regression suite passed; the managed plugin built with zero
  warnings/errors. Matching core, managed, and native DLLs were rebuilt.
- The new focused save regression passed eight cycles in 1.59826 wall seconds
  on CUDA device 0, the RTX 4070 Ti. This time includes simulation and reloads,
  not just the checkpoint writes. The complete native suite also passed.
- The full Unity save probe did not reach the world: startup encountered
  `PlayerPrefsException: Could not store preference value` in `AppInitializer`.
  An initial staging error also left duplicate plugin DLLs. That test is not
  evidence of full menu/ZIP-wrapper success. The portable payload uses the
  clean base and replaces only the canonical plugin entries, avoiding duplicates.
- Native checkpoint format remains 11; no ABI or save-format change was made.
- The user's running files were not replaced and their process was not stopped.
  Closing the separate test process was blocked by an unavailable automatic
  approval reviewer; no attempt was made to bypass it.
- The race and endless-loop risks are real code defects, but the exact cause
  of the user's frozen save is not confirmed. A stuck old session cannot be
  hot-patched safely. Restarting is needed to load the rebuilt DLLs.
- GUI save/overwrite/load, autosave, and shutdown still need interactive
  validation. Existing ecology limitations remain open. Nothing was committed
  or pushed to GitHub by this pass.

The separately named local portable artifact is
`work/portable-0.6.10/Bibites GPU Fork 0.6.10 - SAVE FIX PREVIEW.exe`.

## Portable artifact verification

- EXE size: 68,655,104 bytes.
- EXE SHA-256:
  `5B90AE1B9AC003BDED14C63CEA5821DC95B2823FB28F053BBECD13C447EB44A8`.
- Embedded ZIP SHA-256:
  `F975ADFF9EBEB3A7BCB492ABD42B166A4ED0D0D2506CA04333EE7243EFDA9864`.
- Read back the embedded ZIP resource and verified each core/managed/native
  DLL against the compiled output hash. Exactly three plugin DLLs are present.
  Required game/launcher files and notices are included; no configuration,
  personal checkpoint, log, or debug-symbol files are bundled.
- Compared its entry names with the preceding preview: only two redundant
  directory entries differ; no game files are missing.
- The preceding PARITY PREVIEW EXE hash remains
  `7AEFB16AC298D7E00E6B0FAA40F3895AE95A966F24A726E24AC8A269B2DAE77F`.
- The new launcher uses a separate `save-fix-data-<hash>` extraction directory
  and was not launched into the user's session during this pass.
