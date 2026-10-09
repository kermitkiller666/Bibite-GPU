# Focused 0.6.10 code check — 2026-10-05

Scope: the preceding pellet-cap, detailed-sprite, menu, time-warp and stability
changes. Existing unrelated work was preserved. No GitHub push, permission
requests, game restart, running-game file replacements, or use of the 5060 Ti.

## Corrected in source

- **Saved zero-food worlds:** the loaded-world settings path previously used
  the configured GPU cap when the saved target was zero. A growth-only zone
  remains in the zone list even with zero biomass, so the fallback could
  restore thousands of plants. `FoodPelletCap.ResolveLoaded` keeps unchanged
  saved zero targets at zero, uses the actual stock estimate when biomass is
  raised from zero, and preserves proportional changes to nonzero saved
  targets. The GPU cap and native allocation limit still apply.
- **Stale viewport coverage:** a short visible list from the preceding camera
  position could hide the entire silhouette mesh after a pan or zoom. Each
  snapshot now carries its sampled bounds. Hiding the mesh requires coverage
  of the current camera; moving outside those bounds immediately restores the
  mesh between snapshots. Native sampling now includes the same 12-unit edge
  padding as the sprite pool. The coverage decision uses the actual living
  population rather than the truncated statistics sample.

Regression tests cover zero-food reload, regrowth from zero, saved-target
scaling and caps, invalid bounds, padded camera coverage, and stale pan/zoom
coverage.

## Verification

- Managed/core regression suite passed.
- Managed game plugin built with zero warnings and errors against the supplied
  Unity/BepInEx assemblies.
- Native focused food-zone, brain-I/O, biology/meat, brain-link and checkpoint
  checks passed on CUDA device 0, the RTX 4070 Ti.
- Full existing native regression suite passed on that device. Its large-map
  test simulates 12,000 ticks (five simulated minutes), not five wall-clock
  minutes: 152 living Bibites, 8 births, 368 starvation deaths, 820 pellets
  eaten, and 1,697 active plants at the end.
- The user's running PARITY PREVIEW window was observed, without input. It
  showed the RTX 4070 Ti, 30 graphics FPS and roughly 250x at a requested 250x,
  with zero living Bibites. That empty-world reading is not a populated-world
  performance benchmark or proof of healthy ecology.

## Remaining limits

- Ecology is not proven stable: most founders still starve in the regression.
- The preceding drift fix suppresses silhouettes only when every sampled
  visible Bibite fits the texture budget. Crowded/truncated views can still
  combine fast GPU silhouettes with older CPU sprite positions.
- Selected-Bibite priority currently operates after native truncation: a
  selected Bibite omitted from a full visible list is not guaranteed a texture.
- Menu/save transaction tests are automated checks, not a new complete visual
  menu or shutdown test. The source fixes above have not been visually tested
  in a newly launched Unity session.
- These edits and rebuilt DLLs do not update the already-running portable EXE.
  The native DLL was unchanged by this pass. No release or commit was made.
