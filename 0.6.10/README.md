# Bibites GPU Fork 0.6.10 — preview

This folder contains the **0.6.10 preview source and portable build**, branched
from the 0.6.9 code. It is not a stable release or an upstream version of The
Bibites. Do not mix a 0.6.10 managed DLL with a 0.6.9 native DLL or vice versa.

## Download the October 9 preview

[Bibites GPU Fork 0.6.10 - PREVIEW.exe](Bibites%20GPU%20Fork%200.6.10%20-%20PREVIEW.exe?raw=1)
is a roughly 73 MB, single-file Windows x64 download. Double-click it in a
writable folder; it extracts about 181 MiB beside itself. No separate game
installation is required. The original public 0.6.3.1 assets, BepInEx and
UnityDoorstop are included; this is **not a rebase onto 0.6.4**.

The October 9 code check bounded two graphics-handoff polling loops to 15
seconds. A timed-out callback retains/quarantines its native world and graphics
resources until process exit, rather than freeing resources the driver may
still use. Deadline boundary tests and fresh core/native regressions pass.
This does not prove all shutdown hangs fixed: a native driver call that itself
does not return is outside this managed polling guard.

See the [review and remaining limits](docs/code-check-2026-10-09.md),
[bundle hashes/provenance](BUNDLE_MANIFEST.md), and
[portable build instructions](portable/README.md). **Back up saves:** new saves
use checkpoint format 12, which older binaries cannot read. Keep older builds.

## October 6 optimization preview

The separate **OPTIMIZED PREVIEW** adds local food-overflow searches,
shared-memory crowded-contact tiles, compact live-work lists, specialized
neural kernels and cached CUDA Graphs, owned asynchronous render snapshots,
compact imported-brain storage, and smaller live-row checkpoints. The standard
fused kernel remains the default after comparative tests. Full GPU-instanced
original sprite rendering and final Unity GUI/save validation are still
outstanding. See [implementation, measurements and limits](docs/optimization-preview-2026-10-06.md).

## October 5 save-freeze preview

The local **SAVE FIX PREVIEW** corrects a sparse GPU grid-reset race, bounds
overflow-list and coordinate processing, and logs pending-save worker phases.
Eight crowded-world native save/reload/resume cycles and the full native suite
passed. Full Unity save-menu and ZIP-wrapper validation is still outstanding;
the separate automated run was blocked by a Windows settings write. See
[the investigation and test boundary](docs/gpu-save-freeze-2026-10-05.md).

## Issue #2 work in progress

- Imported/stock brains can read the Constant input plus all 33 named
  0.6.3.1 sensors and address all 15 named actions. Newly generated GPU
  Bibites now use 34 inputs, two 12-neuron hidden layers, and 15 actions.
  Plant/meat vision and diet-aware targeting, eating,
  digestion, healing, biting, egg investment, growth, clock reset, herding,
  grabbing, and directional pheromone sensing have GPU-native effects.
  A focused native regression checks each sensor slot, action routing,
  grabbing, growth, egg investment, clock reset, pheromones, and save/load.
  These effects are **approximations**, not an exact port of the original
  organs, physics joints, eggs, or gene formulas. In particular, offspring
  appear directly rather than incubating as world eggs, and grabbing currently
  holds one target rather than the stock game's multi-object joints.
- New GPU brains inherit and mutate FP16 weights and sparse connection masks,
  including the additional stock inputs and outputs. Disabling a connection
  makes its effective weight zero; re-enabling restores its saved weight.
  Hidden-layer width remains fixed at 12+12; neuron-count mutations are not
  implemented. Older GPU brains are imported best-effort and may retain the
  previous 16-input/6-output layout.
- Neural weights and biases are FP16, with FP32 accumulation. The default
  evaluator uses CUDA scalar/half2 operations. Separate sparse/dense graph
  evaluators and an opt-in padded Tensor Core prototype are available for
  comparison. The Tensor prototype is not consistently faster and is not the
  default; unsupported GPUs retain the ordinary CUDA path.
- Live settings project plant zones, pellet-size multipliers, food target,
  growth, drag, collision/bite factors, diet-affinity powers and material
  conversion-efficiency ranges into the native worker. Remaining
  stock settings need an explicit parity audit.
  For new worlds, stock plant-zone biomass and pellet size determine a
  **maximum plant-pellet count**, bounded by the GPU `PelletCount` ceiling
  (default 8192) and the 32768-slot reserve. The previous fixed 8192 starting
  count and 256-pellet floor no longer override low or zero food settings.
  This cap change is included in the local previews; an already running
  preview EXE does not update itself.
- Death can produce meat pellets. Plant and meat have separate stomach
  compartments and affinity/efficiency-weighted digestion; health and contact/bite damage
  are simulated. The current formulas are GPU approximations, not exact
  ports of the original mouth, stomach, body and material system.
- Selected-Bibite Biology and Genes tabs expose health, diet, stomach state,
  egg/growth/clock/grab state, and action outputs. Native saves use checkpoint
  format 12 with compact live rows; this source reads older GPU checkpoints,
  but older binaries cannot read format 12. Back up worlds and retain old builds.
- Original close-up Bibite sprite parts retain their stock sorting order.
  Detailed textures are budgeted only from the visible viewport, remain active
  while their projected size is useful, and prioritize the selected Bibite.
  New installs default to 512 textured on-screen Bibites; the 0-2048 budget is
  adjustable in GPU Settings for the next simulation. When every visible
  Bibite fits in that budget, the batched silhouette layer is hidden so a
  high-warp snapshot cannot leave a second, drifting shape under each texture.

The native CUDA regression suite includes five-minute large-map ecology,
old-save migration, food-zone sizing, and a focused bite → meat → digestion
test. These tests and the managed build have passed on the RTX 4070 Ti.
An isolated 0.6.10 Unity staging copy launched on that GPU. The default
scenario and 3 Islands opened; the latter displayed Bibites and pellets on a
fertile island after the starter-spawn and camera-focus fixes. Returning to
the main menu worked; a short default-world run exited cleanly. However,
"Exit to Desktop" from a longer 3 Islands run remained in the GPU-cleanup
state for over a minute, so shutdown is **not** verified reliable. The worker
now logs its pending shutdown state after five seconds to diagnose the next
reproduction. The selected-Bibite 34 → 12 → 12 → 15 brain diagram and
link-strength colors were visually checked in an isolated Unity run. A stale
companion DLL initially caused a blank diagram; matching the built libraries
fixed it. In the latest five-minute large-map native run, only about 150 of
512 founders survived and fewer than 10 offspring were born. Ecology and
shutdown remain open problems; this is not broad stability or performance
validation. The published portable preview is not a stable release or broad
parity certification; older local previews are retained separately.

## Building source

You need Visual Studio 2022 with C++, CUDA 13, .NET SDK, the public Bibites
0.6.3.1 managed assemblies and BepInEx 5.4. The project still uses the
provided public 0.6.3.1 game assets; it has not been rebased onto a 0.6.4
game assembly.

```powershell
cmake -S 0.6.10 -B build/0.6.10 -G 'Visual Studio 17 2022' -A x64
cmake --build build/0.6.10 --config Release --target BibitesGpuNative BibitesGpuWorldTests
dotnet build 0.6.10/managed/BibitesGpuFork.csproj -c Release -p:GameManagedDir='<Bibites_Data/Managed>' -p:BepInExCoreDir='<BepInEx/core>'
```

Use the matching managed and native outputs together, and back up worlds
before opening them with development binaries. The native tests accept
`--brain-io-only`, `--biology-only`, `--food-zones-only`, `--brain-links-only` and
`--checkpoint-only`, `--save-progress-only`, `--optimizations-only` and
`--kernel-parity-only` for focused checks.
